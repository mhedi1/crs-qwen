import ast
import pickle
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import my_crs

from my_crs.ckg_retriever import (
    CKGRetriever,
    KBRD_DATA_DIR,
    ReDialKBRDMapping,
    assert_complete_mapping,
    build_count_data,
    build_weighted_graph,
    causal_weight,
    conversation_weight,
    validate_train_source_path,
)
from my_crs.evaluate_ckg_complementarity import (
    DEFAULT_VALID_PATH,
    MetricAccumulator,
    count_frozen_evaluable_turns,
    evaluate_valid,
    parity_verification,
    run_kbrd_parity,
    strict_id_is_hit,
)


def _load_frozen_evaluator_helpers():
    """Load exact pure helper definitions without importing model dependencies."""
    evaluate_path = Path(__file__).with_name("evaluate.py")
    tree = ast.parse(evaluate_path.read_text(encoding="utf-8"), filename=str(evaluate_path))
    wanted = {
        "normalize_title",
        "strict_title_match",
        "is_hit",
        "get_rank",
        "get_recommended_movies_at_turn",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    namespace = {"re": re}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(evaluate_path), "exec"), namespace)
    return namespace


FROZEN = _load_frozen_evaluator_helpers()


def _mapping(*redial_ids: int) -> ReDialKBRDMapping:
    mapping = {redial_id: 100 + redial_id for redial_id in redial_ids}
    return ReDialKBRDMapping(mapping, frozenset(mapping.values()))


def _conversation(messages, suggested_ids=(), respondent=2):
    questions = {
        str(movie_id): {"suggested": int(movie_id in suggested_ids)}
        for movie_id in suggested_ids
    }
    return {
        "respondentWorkerId": respondent,
        "initiatorWorkerId": 1,
        "respondentQuestions": questions,
        "messages": messages,
    }


class TestCKGRetriever(unittest.TestCase):
    def test_deterministic_redial_to_kbrd_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            with (data_dir / "id2entity.pkl").open("wb") as handle:
                pickle.dump({10: "<movie-uri>", 20: None}, handle)
            with (data_dir / "entity2entityId.pkl").open("wb") as handle:
                pickle.dump({"<movie-uri>": 501, 20: 502}, handle)
            with (data_dir / "movie_ids.pkl").open("wb") as handle:
                pickle.dump([501, 502], handle)

            first = ReDialKBRDMapping.load(data_dir)
            second = ReDialKBRDMapping.load(data_dir)
            self.assertEqual(first.redial_to_entity, {10: 501, 20: 502})
            self.assertEqual(first, second)
            self.assertEqual(first.map_id("10"), 501)

    def test_real_uri_less_movie_uses_integer_mapping_branch(self):
        with (KBRD_DATA_DIR / "id2entity.pkl").open("rb") as handle:
            id2entity = pickle.load(handle)
        with (KBRD_DATA_DIR / "entity2entityId.pkl").open("rb") as handle:
            entity2id = pickle.load(handle)

        redial_id = next(int(movie_id) for movie_id, uri in id2entity.items() if uri is None)
        self.assertIn(redial_id, entity2id)
        mapping = ReDialKBRDMapping.load(KBRD_DATA_DIR)
        self.assertEqual(mapping.map_id(redial_id), int(entity2id[redial_id]))

    def test_conversation_graph_does_not_double_count_repeated_mentions(self):
        mapping = _mapping(1, 2)
        conversation = _conversation(
            [
                {"senderWorkerId": 1, "text": "@1 @1 @1"},
                {"senderWorkerId": 2, "text": "@2 and @2"},
            ]
        )
        counts = build_count_data([conversation], mapping)["conversation"]
        self.assertEqual(counts["node_counts"][101], 1)
        self.assertEqual(counts["node_counts"][102], 1)
        self.assertEqual(counts["pair_counts"][(101, 102)], 1)

    def test_causal_graph_never_uses_future_messages(self):
        mapping = _mapping(1, 2, 3)
        conversation = _conversation(
            [
                {"senderWorkerId": 1, "text": "I liked @1"},
                {"senderWorkerId": 2, "text": "Try @2"},
                {"senderWorkerId": 1, "text": "Later mention @3"},
            ],
            suggested_ids=(2,),
        )
        causal = build_count_data([conversation], mapping)["causal"]
        self.assertEqual(causal["pair_counts"][(101, 102)], 1)
        self.assertNotIn((103, 102), causal["pair_counts"])

    def test_multiple_targets_create_separate_target_events(self):
        mapping = _mapping(1, 2, 3)
        conversation = _conversation(
            [
                {"senderWorkerId": 1, "text": "I liked @1"},
                {"senderWorkerId": 2, "text": "Try @2 or @3"},
            ],
            suggested_ids=(2, 3),
        )
        causal = build_count_data([conversation], mapping)["causal"]
        self.assertEqual(causal["N"], 2)
        self.assertEqual(causal["source_counts"][101], 2)
        self.assertEqual(causal["target_counts"][102], 1)
        self.assertEqual(causal["target_counts"][103], 1)
        self.assertEqual(causal["pair_counts"][(101, 102)], 1)
        self.assertEqual(causal["pair_counts"][(101, 103)], 1)

    def test_empty_list_recommendation_annotations_are_explicitly_non_events(self):
        mapping = _mapping(1)
        conversation = {
            "respondentWorkerId": 2,
            "respondentQuestions": [],
            "messages": [{"senderWorkerId": 2, "text": "Mention @1 without labels"}],
        }
        causal = build_count_data([conversation], mapping)["causal"]
        self.assertEqual(causal["N"], 0)
        self.assertEqual(dict(causal["target_counts"]), {})

    def test_causal_self_pair_is_retained(self):
        mapping = _mapping(1)
        conversation = _conversation(
            [
                {"senderWorkerId": 1, "text": "I saw @1"},
                {"senderWorkerId": 2, "text": "Then I suggest @1"},
            ],
            suggested_ids=(1,),
        )
        causal = build_count_data([conversation], mapping)["causal"]
        self.assertEqual(causal["pair_counts"][(101, 101)], 1)

    def test_non_movie_entities_cannot_enter_retrieval(self):
        graph = {
            "metadata": {},
            "movie_ids": frozenset({1, 2}),
            "adjacency": {1: [(2, 1.0)], 999: [(2, 100.0)]},
            "suggested_target_popularity": {1: 0, 2: 1},
            "fallback_ranking": [2, 1],
        }
        candidates, diagnostics = CKGRetriever(graph).retrieve([999, 1], top_k=50)
        self.assertEqual(diagnostics["movie_seed_ids"], [1])
        self.assertEqual([candidate["id"] for candidate in candidates], [2])

    def test_conditional_score_calculation(self):
        self.assertAlmostEqual(conversation_weight(2, 4, 3, 8, "conditional"), 0.5)
        self.assertAlmostEqual(causal_weight(2, 4, 2, 8, "conditional"), 0.5)

    def test_ppmi_hand_computable_example(self):
        # log2((2 * 8) / (4 * 2)) = log2(2) = 1
        self.assertAlmostEqual(conversation_weight(2, 4, 2, 8, "ppmi"), 1.0)
        self.assertAlmostEqual(causal_weight(2, 4, 2, 8, "ppmi"), 1.0)
        # A negative PMI is clamped to zero, with no smoothing.
        self.assertEqual(causal_weight(1, 8, 8, 8, "ppmi"), 0.0)

    def test_deterministic_score_then_entity_id_tie_break(self):
        graph = {
            "metadata": {},
            "movie_ids": frozenset({1, 2, 3}),
            "adjacency": {1: [(3, 0.5), (2, 0.5)]},
            "suggested_target_popularity": {},
            "fallback_ranking": [1, 2, 3],
        }
        candidates, _ = CKGRetriever(graph).retrieve([1], top_k=2)
        self.assertEqual([candidate["id"] for candidate in candidates], [2, 3])

    def test_zero_seed_fallback_uses_train_suggested_target_popularity_only(self):
        mapping = _mapping(1, 2, 3)
        conversation = _conversation(
            [
                {"senderWorkerId": 1, "text": "Many mentions @1 @1 @1 @1"},
                {"senderWorkerId": 2, "text": "Suggested target @2"},
            ],
            suggested_ids=(2,),
        )
        count_data = build_count_data([conversation], mapping)
        graph = build_weighted_graph(
            count_data,
            mapping,
            graph_type="causal",
            weighting_method="conditional",
            min_support=1,
        )
        candidates, diagnostics = CKGRetriever(graph).retrieve([], top_k=3)
        self.assertTrue(diagnostics["fallback_used"])
        self.assertEqual(diagnostics["num_movie_seeds"], 0)
        self.assertEqual([candidate["id"] for candidate in candidates], [102, 101, 103])
        self.assertEqual(candidates[0]["score"], 1.0)
        self.assertEqual(candidates[1]["score"], 0.0)

    def test_zero_seed_budget_view_has_distinct_fallback_provenance(self):
        retriever = self._dense_retriever()
        views = retriever.retrieve_views([], top_k=50)
        diagnostics = views["diagnostics"]
        self.assertEqual(views["graph_only"], [])
        self.assertEqual(len(views["budget_controlled"]), 50)
        self.assertTrue(diagnostics["fallback_used"])
        self.assertFalse(diagnostics["fill_used"])
        self.assertEqual(diagnostics["pre_fill_candidate_count"], 0)
        self.assertEqual(diagnostics["post_fill_candidate_count"], 50)
        self.assertEqual(
            {candidate["source"] for candidate in views["budget_controlled"]},
            {"CKG_ZERO_SEED_FALLBACK"},
        )

    def test_nonzero_seed_without_neighbours_uses_popularity_fill(self):
        retriever = self._dense_retriever()
        views = retriever.retrieve_views([1], top_k=50)
        diagnostics = views["diagnostics"]
        self.assertEqual(views["graph_only"], [])
        self.assertFalse(diagnostics["fallback_used"])
        self.assertTrue(diagnostics["fill_used"])
        self.assertEqual(len(views["budget_controlled"]), 50)
        self.assertEqual(
            {candidate["source"] for candidate in views["budget_controlled"]},
            {"CKG_POPULARITY_FILL"},
        )

    def test_graph_only_view_can_contain_fewer_than_50_candidates(self):
        retriever = self._dense_retriever(adjacency={1: [(2, 0.8), (3, 0.4)]})
        views = retriever.retrieve_views([1], top_k=50)
        self.assertEqual([candidate["id"] for candidate in views["graph_only"]], [2, 3])
        self.assertEqual(views["diagnostics"]["pre_fill_candidate_count"], 2)

    def test_budget_view_is_filled_to_exactly_50_without_duplicates(self):
        retriever = self._dense_retriever(adjacency={1: [(2, 0.8), (3, 0.4)]})
        views = retriever.retrieve_views([1], top_k=50)
        candidates = views["budget_controlled"]
        self.assertEqual(len(candidates), 50)
        self.assertEqual(len({candidate["id"] for candidate in candidates}), 50)
        self.assertEqual([candidate["source"] for candidate in candidates[:2]], [
            "CKG_GRAPH",
            "CKG_GRAPH",
        ])
        self.assertEqual(
            {candidate["source"] for candidate in candidates[2:]},
            {"CKG_POPULARITY_FILL"},
        )

    def test_multi_seed_scores_are_summed_before_ranking(self):
        retriever = self._dense_retriever(
            adjacency={1: [(4, 0.4), (5, 0.7)], 2: [(4, 0.5)]}
        )
        candidates = retriever.retrieve_views([2, 1], top_k=2)["graph_only"]
        self.assertEqual([candidate["id"] for candidate in candidates], [4, 5])
        self.assertAlmostEqual(candidates[0]["score"], 0.9)

    def test_already_mentioned_movie_remains_eligible(self):
        retriever = self._dense_retriever(adjacency={1: [(1, 0.9), (2, 0.5)]})
        candidates = retriever.retrieve_views([1], top_k=50)["graph_only"]
        self.assertEqual(candidates[0]["id"], 1)

    def test_graph_construction_rejects_valid_and_test_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("valid_data.jsonl", "test_data.jsonl"):
                path = root / filename
                path.write_text("{}\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_train_source_path(path)
            train_path = root / "train_data.jsonl"
            train_path.write_text("{}\n", encoding="utf-8")
            self.assertEqual(validate_train_source_path(train_path), train_path.resolve())

    def test_incomplete_mapping_is_not_silently_ignored(self):
        with self.assertRaises(AssertionError):
            assert_complete_mapping(
                {
                    "unmapped_unique_train_annotation_movie_ids": 1,
                    "unmapped_unique_train_suggested_target_movie_ids": 0,
                }
            )

    def test_valid_fixture_matches_frozen_797_conversations_and_2588_instances(self):
        counts = count_frozen_evaluable_turns(
            DEFAULT_VALID_PATH, FROZEN["get_recommended_movies_at_turn"]
        )
        self.assertEqual(counts["input_conversations_seen"], 811)
        self.assertEqual(counts["evaluated_conversations"], 797)
        self.assertEqual(counts["evaluation_instances"], 2588)

    def test_parity_verification_checks_metrics_counts_and_configuration(self):
        result = {
            "evaluated_conversations": 797,
            "evaluation_instances": 2588,
            "recommendation": {
                "Recall@1": 0.02319,
                "Recall@10": 0.17001,
                "Recall@50": 0.38409,
                "MRR": 0.06869,
            },
            "configuration": {
                "resolver_version": "v3",
                "use_legacy_non_movie_entities": True,
                "seed_selection": "all",
                "retrieval_mode": "kbrd",
                "top_k": 50,
                "skip_reranker": True,
                "recommendation_only": True,
                "split": "valid",
                "fusion": False,
            },
        }
        self.assertTrue(parity_verification(result)["passed"])
        result["configuration"]["seed_selection"] = "recent"
        verification = parity_verification(result)
        self.assertFalse(verification["passed"])
        self.assertFalse(verification["checks"]["configuration.seed_selection"]["passed"])

    def test_ckg_evaluation_is_gated_before_cache_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid_data.jsonl"
            valid_path.write_text("", encoding="utf-8")
            with patch.object(CKGRetriever, "load") as load:
                with self.assertRaisesRegex(RuntimeError, "parity report"):
                    evaluate_valid(
                        valid_path=valid_path,
                        cache_dir=root / "missing-cache",
                        output_path=root / "output.json",
                        instance_output_path=root / "instances.jsonl",
                        parity_report_path=root / "missing-parity.json",
                    )
            load.assert_not_called()

    def test_kbrd_parity_only_mode_never_loads_or_retrieves_ckg(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid_data.jsonl"
            conversation = {
                "conversationId": "fixture",
                "respondentWorkerId": 2,
                "initiatorWorkerId": 1,
                "movieMentions": {"7": "Gold Movie (2000)"},
                "respondentQuestions": {"7": {"suggested": 1}},
                "messages": [
                    {"senderWorkerId": 1, "text": "Please recommend something"},
                    {"senderWorkerId": 2, "text": "Try @7"},
                ],
            }
            valid_path.write_text(__import__("json").dumps(conversation) + "\n", encoding="utf-8")
            calls = []
            fake_evaluator = types.ModuleType("my_crs.evaluate")
            fake_evaluator._cfg = {
                "extraction": {
                    "resolver_version": "v3",
                    "use_legacy_non_movie_entities": True,
                    "seed_selection": "all",
                }
            }
            fake_evaluator.get_recommended_movies_at_turn = FROZEN[
                "get_recommended_movies_at_turn"
            ]
            fake_evaluator.build_dialogue_up_to = lambda sample, turn: "User: context"
            fake_evaluator.is_hit = FROZEN["is_hit"]
            fake_evaluator.get_rank = FROZEN["get_rank"]
            original_tmdb_cache = root / "existing-tmdb-cache.json"
            original_tmdb_cache.write_text("preserve-me", encoding="utf-8")
            fake_adapter = types.ModuleType("fixture_kbrd_adapter")
            fake_adapter._TMDB_CACHE_PATH = str(original_tmdb_cache)

            def fake_candidates(dialogue, **kwargs):
                calls.append((dialogue, kwargs))
                Path(fake_adapter._TMDB_CACHE_PATH).write_text(
                    "isolated-write", encoding="utf-8"
                )
                return [{"id": 7, "title": "gold movie"}], []

            fake_candidates.__module__ = fake_adapter.__name__
            fake_evaluator.get_kbrd_candidates = fake_candidates
            output_path = root / "parity.json"
            with patch.dict(
                sys.modules,
                {
                    "my_crs.evaluate": fake_evaluator,
                    fake_adapter.__name__: fake_adapter,
                },
            ), patch.object(
                my_crs, "evaluate", fake_evaluator, create=True
            ), patch.object(CKGRetriever, "load") as load, patch.object(
                CKGRetriever, "retrieve_views"
            ) as retrieve:
                result = run_kbrd_parity(valid_path, output_path, max_conversations=1)
            load.assert_not_called()
            retrieve.assert_not_called()
            self.assertEqual(result["evaluation_instances"], 1)
            self.assertFalse(result["ckg_loaded"])
            self.assertFalse(result["ckg_retrieved"])
            self.assertTrue(result["tmdb_cache_isolated"])
            self.assertEqual(original_tmdb_cache.read_text(encoding="utf-8"), "preserve-me")
            self.assertEqual(fake_adapter._TMDB_CACHE_PATH, str(original_tmdb_cache))
            self.assertEqual(
                calls[0][1],
                {
                    "top_k": 50,
                    "diagnostics": None,
                    "use_fusion": False,
                    "retrieval_mode": "kbrd",
                },
            )

    def test_normalized_title_scoring_matches_frozen_evaluator(self):
        candidates = [{"id": 1, "title": "The Matrix (1999)"}]
        ground_truth = ["the matrix!"]
        self.assertTrue(FROZEN["is_hit"](candidates, ground_truth, 1))
        self.assertEqual(FROZEN["get_rank"](candidates, ground_truth), 1)

    def test_strict_id_can_miss_when_primary_normalized_title_hits(self):
        candidates = [{"id": 1, "title": "Same Film (1999)"}]
        ground_truth = ["same film"]
        self.assertTrue(FROZEN["is_hit"](candidates, ground_truth, 1))
        self.assertFalse(strict_id_is_hit([1], {2}, 1))

    def test_multiple_gold_titles_use_any_hit_and_first_matching_rank(self):
        candidates = [
            {"id": 1, "title": "Unrelated"},
            {"id": 2, "title": "Second Gold (2001)"},
        ]
        ground_truth = ["first gold", "second gold"]
        self.assertFalse(FROZEN["is_hit"](candidates, ground_truth, 1))
        self.assertTrue(FROZEN["is_hit"](candidates, ground_truth, 10))
        self.assertEqual(FROZEN["get_rank"](candidates, ground_truth), 2)

    def test_complementarity_recovery_and_partition_invariant(self):
        metrics = MetricAccumulator()
        is_hit = FROZEN["is_hit"]
        get_rank = FROZEN["get_rank"]
        metrics.add(
            [{"id": 2, "title": "Gold"}],
            [{"id": 3, "title": "Other"}],
            ["gold"],
            is_hit,
            get_rank,
        )
        metrics.add(
            [{"id": 4, "title": "Miss"}],
            [{"id": 3, "title": "Other"}],
            ["gold"],
            is_hit,
            get_rank,
        )
        full_result = metrics.result()
        result = full_result["against_frozen_kbrd_at_50"]
        self.assertEqual(result["CKG_only_hit"], 1)
        self.assertEqual(result["total_KBRD_misses"], 2)
        self.assertEqual(result["CKG_recovery_among_KBRD_failures"], 0.5)
        self.assertEqual(result["partition_total"], full_result["instances"])
        self.assertTrue(result["partition_invariant_passed"])

    @staticmethod
    def _dense_retriever(adjacency=None):
        movie_ids = frozenset(range(1, 61))
        graph = {
            "metadata": {},
            "movie_ids": movie_ids,
            "adjacency": adjacency or {},
            "suggested_target_popularity": {
                movie_id: 61 - movie_id for movie_id in movie_ids
            },
            "fallback_ranking": list(range(1, 61)),
        }
        return CKGRetriever(graph)


if __name__ == "__main__":
    unittest.main()
