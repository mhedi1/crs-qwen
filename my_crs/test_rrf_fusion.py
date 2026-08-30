import ast
import copy
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import my_crs

from my_crs.ckg_retriever import CKGRetriever
from my_crs.evaluate_rrf_fusion import (
    CKG_WEIGHT,
    FROZEN_CKG_CACHE_NAME,
    FROZEN_CKG_GRAPH_TYPE,
    FROZEN_CKG_MIN_SUPPORT,
    FROZEN_CKG_WEIGHTING,
    KBRD_WEIGHT,
    RRF_K,
    TOP_K,
    RankingMetricAccumulator,
    evaluate_rrf_valid,
    load_frozen_ckg,
    reciprocal_rank_fusion,
)


def _load_frozen_scoring_helpers():
    evaluate_path = Path(__file__).with_name("evaluate.py")
    tree = ast.parse(evaluate_path.read_text(encoding="utf-8"), filename=str(evaluate_path))
    wanted = {"normalize_title", "strict_title_match", "is_hit", "get_rank"}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted
    ]
    namespace = {"re": re}
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(evaluate_path),
            "exec",
        ),
        namespace,
    )
    return namespace


FROZEN = _load_frozen_scoring_helpers()


def _candidate(entity_id: int, *, score: float = 0.0) -> dict:
    return {
        "id": entity_id,
        "title": f"Movie {entity_id}",
        "score": score,
        "source": "fixture",
    }


class TestReciprocalRankFusion(unittest.TestCase):
    def test_hand_computable_rrf(self):
        fused = reciprocal_rank_fusion(
            [_candidate(1), _candidate(2)],
            [_candidate(2), _candidate(3)],
        )

        self.assertEqual([candidate["id"] for candidate in fused], [2, 1, 3])
        by_id = {candidate["id"]: candidate for candidate in fused}
        self.assertAlmostEqual(by_id[2]["rrf_score"], 1 / 62 + 1 / 61)
        self.assertAlmostEqual(by_id[1]["rrf_score"], 1 / 61)
        self.assertAlmostEqual(by_id[3]["rrf_score"], 1 / 62)

    def test_movie_present_in_both_rankings_has_both_contributions(self):
        fused = reciprocal_rank_fusion([_candidate(7)], [_candidate(7)])

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["kbrd_rank"], 1)
        self.assertEqual(fused[0]["ckg_rank"], 1)
        self.assertAlmostEqual(fused[0]["kbrd_contribution"], 1 / 61)
        self.assertAlmostEqual(fused[0]["ckg_contribution"], 1 / 61)

    def test_movie_present_in_only_one_ranking_gets_zero_absent_contribution(self):
        fused = reciprocal_rank_fusion([_candidate(7)], [])

        self.assertEqual(fused[0]["kbrd_rank"], 1)
        self.assertIsNone(fused[0]["ckg_rank"])
        self.assertAlmostEqual(fused[0]["kbrd_contribution"], 1 / 61)
        self.assertEqual(fused[0]["ckg_contribution"], 0.0)

    def test_duplicate_removal_keeps_first_occurrence_and_compresses_ranks(self):
        fused = reciprocal_rank_fusion(
            [_candidate(1), _candidate(1), _candidate(2)], []
        )

        self.assertEqual([candidate["id"] for candidate in fused], [1, 2])
        self.assertEqual(fused[0]["kbrd_rank"], 1)
        self.assertEqual(fused[1]["kbrd_rank"], 2)

    def test_deterministic_ties_use_entity_id(self):
        expected = [1, 2]
        for _ in range(5):
            fused = reciprocal_rank_fusion([_candidate(2)], [_candidate(1)])
            self.assertEqual([candidate["id"] for candidate in fused], expected)

    def test_top_50_budget_contains_no_duplicates(self):
        kbrd = [_candidate(entity_id) for entity_id in range(1, 76)] + [_candidate(1)]
        ckg = [_candidate(entity_id) for entity_id in range(51, 126)] + [_candidate(51)]

        fused = reciprocal_rank_fusion(kbrd, ckg)
        fused_ids = [candidate["id"] for candidate in fused]

        self.assertEqual(len(fused_ids), TOP_K)
        self.assertEqual(len(fused_ids), len(set(fused_ids)))

    def test_input_lists_are_not_mutated(self):
        kbrd = [_candidate(1, score=999.0), _candidate(2, score=-1.0)]
        ckg = [_candidate(2, score=-999.0), _candidate(3, score=1.0)]
        original_kbrd = copy.deepcopy(kbrd)
        original_ckg = copy.deepcopy(ckg)

        reciprocal_rank_fusion(kbrd, ckg)

        self.assertEqual(kbrd, original_kbrd)
        self.assertEqual(ckg, original_ckg)

    def test_equal_weight_fusion_is_symmetric(self):
        first = [_candidate(1), _candidate(2), _candidate(3)]
        second = [_candidate(3), _candidate(4), _candidate(1)]

        forward = reciprocal_rank_fusion(first, second)
        reverse = reciprocal_rank_fusion(second, first)

        self.assertEqual(KBRD_WEIGHT, CKG_WEIGHT)
        self.assertEqual(
            [(candidate["id"], candidate["rrf_score"]) for candidate in forward],
            [(candidate["id"], candidate["rrf_score"]) for candidate in reverse],
        )

    def test_raw_scores_do_not_influence_rrf(self):
        low_high = reciprocal_rank_fusion(
            [_candidate(1, score=-1_000_000), _candidate(2, score=1_000_000)],
            [_candidate(2, score=-999), _candidate(3, score=999)],
        )
        high_low = reciprocal_rank_fusion(
            [_candidate(1, score=1_000_000), _candidate(2, score=-1_000_000)],
            [_candidate(2, score=999), _candidate(3, score=-999)],
        )

        self.assertEqual(low_high, high_low)
        self.assertTrue(all("score" not in candidate for candidate in low_high))

    def test_metric_accumulator_uses_frozen_normalized_title_scoring(self):
        accumulator = RankingMetricAccumulator()
        accumulator.add(
            [{"id": 1, "title": "The Matrix (1999)"}],
            ["the matrix!"],
            FROZEN["is_hit"],
            FROZEN["get_rank"],
        )

        self.assertEqual(
            accumulator.result(),
            {
                "instances": 1,
                "Recall@1": 1.0,
                "Recall@10": 1.0,
                "Recall@50": 1.0,
                "MRR": 1.0,
            },
        )

    def test_loads_only_frozen_conversation_conditional_support2_cache(self):
        retriever = Mock()
        retriever.metadata = {
            "graph_type": FROZEN_CKG_GRAPH_TYPE,
            "weighting_method": FROZEN_CKG_WEIGHTING,
            "min_support": FROZEN_CKG_MIN_SUPPORT,
        }

        with tempfile.TemporaryDirectory() as directory, patch.object(
            CKGRetriever, "load", return_value=retriever
        ) as load_mock:
            loaded, cache_path = load_frozen_ckg(directory)

        self.assertIs(loaded, retriever)
        self.assertEqual(cache_path.name, FROZEN_CKG_CACHE_NAME)
        load_mock.assert_called_once_with(Path(directory).resolve() / FROZEN_CKG_CACHE_NAME)

    def test_test_split_path_is_rejected_before_cache_or_parity_access(self):
        with tempfile.TemporaryDirectory() as directory:
            test_path = Path(directory) / "test_data.jsonl"
            test_path.write_text("", encoding="utf-8")
            with patch(
                "my_crs.evaluate_rrf_fusion.require_passing_parity_report"
            ) as parity_gate, patch(
                "my_crs.evaluate_rrf_fusion.load_frozen_ckg"
            ) as load_mock:
                with self.assertRaisesRegex(ValueError, "VALID-only"):
                    evaluate_rrf_valid(valid_path=test_path)

        parity_gate.assert_not_called()
        load_mock.assert_not_called()

    def test_valid_evaluation_uses_frozen_sources_and_writes_provenance(self):
        fake_evaluator = types.ModuleType("my_crs.evaluate")
        fake_evaluator._cfg = {
            "extraction": {
                "resolver_version": "v3",
                "use_legacy_non_movie_entities": True,
                "use_aux_dbpedia_uri_matching": True,
                "use_aux_genre_mapping": True,
                "use_aux_person_matching": False,
                "seed_selection": "all",
            }
        }
        fake_evaluator.get_recommended_movies_at_turn = Mock(return_value=["Movie 2!"])
        fake_evaluator.build_dialogue_up_to = Mock(return_value="User: fixture")
        fake_evaluator.get_kbrd_candidates = Mock(
            return_value=([_candidate(1), _candidate(2)], [])
        )
        fake_evaluator.is_hit = FROZEN["is_hit"]
        fake_evaluator.get_rank = FROZEN["get_rank"]

        fake_adapter = types.ModuleType("kbrd_adapter")
        fake_adapter.prepare_input = Mock(return_value=([99], [], [], [], {}))
        fake_catalogue = types.ModuleType("my_crs.movie_catalogue")
        fake_catalogue.load_catalogue = Mock()
        fake_catalogue.get_title = Mock(
            side_effect=lambda entity_id: {2: "Movie 2", 3: "Movie 3"}.get(entity_id)
        )

        retriever = Mock()
        retriever.metadata = {
            "graph_type": "conversation",
            "weighting_method": "conditional",
            "min_support": 2,
        }
        retriever.retrieve_views.return_value = {
            "graph_only": [{"id": 2, "score": 0.1, "source": "CKG_GRAPH"}],
            "budget_controlled": [
                {"id": 2, "score": 0.1, "source": "CKG_GRAPH"},
                {"id": 3, "score": 999.0, "source": "CKG_POPULARITY_FILL"},
            ],
            "diagnostics": {
                "fallback_used": False,
                "fill_used": True,
                "num_movie_seeds": 1,
                "movie_seed_ids": [99],
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid_data.jsonl"
            valid_path.write_text(
                json.dumps(
                    {
                        "conversationId": "fixture",
                        "respondentWorkerId": 2,
                        "messages": [{"senderWorkerId": 2, "text": "Try @2"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = root / "result.json"
            instance_path = root / "result.instances.jsonl"
            cache_path = root / FROZEN_CKG_CACHE_NAME

            with patch.dict(
                sys.modules,
                {
                    "my_crs.evaluate": fake_evaluator,
                    "kbrd_adapter": fake_adapter,
                    "my_crs.movie_catalogue": fake_catalogue,
                },
            ), patch.object(
                my_crs, "evaluate", fake_evaluator, create=True
            ), patch.object(
                my_crs, "movie_catalogue", fake_catalogue, create=True
            ), patch(
                "my_crs.evaluate_rrf_fusion.require_passing_parity_report",
                return_value={},
            ), patch(
                "my_crs.evaluate_rrf_fusion.load_frozen_ckg",
                return_value=(retriever, cache_path),
            ):
                result = evaluate_rrf_valid(
                    valid_path=valid_path,
                    cache_dir=root,
                    output_path=output_path,
                    instance_output_path=instance_path,
                    parity_report_path=root / "parity.json",
                )

            provenance = [
                json.loads(line)
                for line in instance_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(result["evaluation_instances"], 1)
        self.assertEqual(result["failures"], [])
        self.assertEqual(result["metrics"]["KBRD"]["MRR"], 0.5)
        self.assertEqual(result["metrics"]["CKG"]["MRR"], 1.0)
        self.assertEqual(result["metrics"]["RRF"]["MRR"], 1.0)
        self.assertEqual(result["oracle_union_coverage_upper_bound"]["coverage"], 1.0)
        self.assertIn("UPPER BOUND ONLY", result["oracle_union_coverage_upper_bound"]["label"])
        self.assertFalse(
            result["extraction_configuration"]["use_aux_person_matching"]
        )
        self.assertEqual(result["rrf_parameters"]["k"], RRF_K)
        self.assertEqual(result["rrf_parameters"]["weights"], {"KBRD": 1.0, "CKG": 1.0})
        self.assertFalse(result["rrf_parameters"]["raw_scores_used"])
        self.assertEqual(result["ckg_configuration"]["view"], "budget_controlled")
        self.assertEqual(len(provenance), 1)
        self.assertEqual(provenance[0]["rrf_candidates"][0]["id"], 2)
        fake_evaluator.get_kbrd_candidates.assert_called_once_with(
            "User: fixture",
            top_k=50,
            diagnostics=None,
            use_fusion=False,
            retrieval_mode="kbrd",
        )
        retriever.retrieve_views.assert_called_once_with([99], top_k=50)


if __name__ == "__main__":
    unittest.main()
