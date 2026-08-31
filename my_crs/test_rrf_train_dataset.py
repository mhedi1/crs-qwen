import copy
import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from my_crs.build_rrf_train_dataset import (
    AUDIT_FILENAME,
    CONTRIBUTIONS_FILENAME,
    EXPECTED_TRAIN_SHA256,
    EXPECTED_EXTRACTION_CONFIGURATION,
    OFFICIAL_TRAIN_PATH,
    SFT_FILENAME,
    ReconstructionExpectations,
    _sha256,
    _load_project_extraction_configuration,
    _training_extraction_configuration,
    _validate_kbrd_checkpoint_selection,
    _validate_neural_kbrd_candidates,
    analyze_sft_token_lengths,
    assign_conversation_split,
    build_rrf_train_dataset,
    construct_assistant_target,
    positive_candidate_positions,
    reconstruct_train_instances,
    validate_official_train_path,
)
from my_crs.ckg_retriever import ReDialKBRDMapping, build_count_data
from my_crs.evaluate_rrf_fusion import reciprocal_rank_fusion as frozen_rrf
from my_crs.rrf_list_reranker import parse_ranked_positions


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestTrainSourceAndReconstruction(unittest.TestCase):
    def test_official_train_source_and_sha_are_frozen(self):
        self.assertEqual(
            validate_official_train_path(OFFICIAL_TRAIN_PATH),
            OFFICIAL_TRAIN_PATH.resolve(),
        )
        self.assertEqual(_sha256(OFFICIAL_TRAIN_PATH), EXPECTED_TRAIN_SHA256)

    def test_valid_test_wrong_sha_and_alternate_train_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official = root / "official" / "train_data.jsonl"
            official.parent.mkdir()
            official.write_text("{}\n", encoding="utf-8")
            source_sha = _sha256(official)
            self.assertEqual(
                validate_official_train_path(
                    official,
                    official_path=official,
                    expected_sha256=source_sha,
                ),
                official.resolve(),
            )
            for filename in ("valid_data.jsonl", "test_data.jsonl"):
                path = root / filename
                path.write_text("{}\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    validate_official_train_path(
                        path,
                        official_path=path,
                        expected_sha256=_sha256(path),
                    )
            alternate = root / "alternate" / "train_data.jsonl"
            alternate.parent.mkdir()
            alternate.write_bytes(official.read_bytes())
            with self.assertRaisesRegex(ValueError, "official TRAIN path"):
                validate_official_train_path(
                    alternate,
                    official_path=official,
                    expected_sha256=source_sha,
                )
            with self.assertRaisesRegex(ValueError, "TRAIN SHA mismatch"):
                validate_official_train_path(
                    official,
                    official_path=official,
                    expected_sha256="0" * 64,
                )

    def test_authoritative_train_reconstruction_counts(self):
        reconstruction = reconstruct_train_instances(OFFICIAL_TRAIN_PATH)
        self.assertEqual(len(reconstruction.conversations), 7293)
        self.assertEqual(reconstruction.evaluable_conversations, 7161)
        self.assertEqual(len(reconstruction.events), 23686)
        self.assertEqual(reconstruction.unique_target_occurrences, 26708)
        self.assertEqual(reconstruction.max_targets_per_instance, 7)

    def test_history_excludes_target_and_future_turns(self):
        conversation = {
            "conversationId": "c",
            "initiatorWorkerId": 1,
            "respondentWorkerId": 2,
            "movieMentions": {"1": "Prior", "2": "Target", "3": "Future"},
            "respondentQuestions": {"2": {"suggested": 1}},
            "messages": [
                {"senderWorkerId": 1, "text": "I liked @1"},
                {"senderWorkerId": 2, "text": "Try @2 TARGET_SENTINEL"},
                {"senderWorkerId": 1, "text": "FUTURE_SENTINEL @3"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_data.jsonl"
            path.write_text(json.dumps(conversation) + "\n", encoding="utf-8")
            expected = ReconstructionExpectations(1, 1, 1, 1, 1)
            result = reconstruct_train_instances(path, expectations=expected)
            from my_crs.evaluate_rrf_zeroshot import build_dialogue_up_to

            history = build_dialogue_up_to(conversation, result.events[0].turn_index - 1)
            self.assertIn("Prior", history)
            self.assertNotIn("TARGET_SENTINEL", history)
            self.assertNotIn("FUTURE_SENTINEL", history)


class TestEligibilityAndTarget(unittest.TestCase):
    def test_zero_one_multiple_and_more_than_ten_positives(self):
        candidates = [
            {"id": position, "title": f"Movie {position}"}
            for position in range(1, 51)
        ]
        self.assertEqual(positive_candidate_positions(candidates, ["Absent"]), [])
        self.assertEqual(positive_candidate_positions(candidates, ["Movie 17"]), [17])
        self.assertEqual(
            positive_candidate_positions(candidates, ["Movie 17", "Movie 4"]),
            [4, 17],
        )
        positions, truncated, target = construct_assistant_target([17])
        self.assertEqual(positions, [17, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertFalse(truncated)
        self.assertEqual(target, '{"ranked_ids":[17,1,2,3,4,5,6,7,8,9]}')

        positions, truncated, target = construct_assistant_target([4, 17])
        self.assertEqual(positions, [4, 17, 1, 2, 3, 5, 6, 7, 8, 9])
        self.assertFalse(truncated)
        self.assertEqual(
            parse_ranked_positions(target, candidate_count=50), positions
        )

        positions, truncated, target = construct_assistant_target(range(1, 12))
        self.assertEqual(positions, list(range(1, 11)))
        self.assertTrue(truncated)
        self.assertEqual(parse_ranked_positions(target, candidate_count=50), positions)

    def test_normalized_title_matching_does_not_insert_gold(self):
        candidates = [
            {"id": 1, "title": "The Film (1999)"},
            {"id": 2, "title": "Another"},
        ]
        original = [dict(candidate) for candidate in candidates]
        self.assertEqual(positive_candidate_positions(candidates, ["the film!"]), [1])
        self.assertEqual(positive_candidate_positions(candidates, ["Missing"]), [])
        self.assertEqual(candidates, original)

    def test_neural_kbrd_fallback_is_rejected(self):
        movie_ids = frozenset(range(1, 60))
        static = [{"id": value, "title": "debug"} for value in range(1, 6)]
        with self.assertRaisesRegex(RuntimeError, "exactly 50"):
            _validate_neural_kbrd_candidates(static, movie_ids)
        wrong_source = [
            {"id": value, "title": str(value), "source": "STATIC"}
            for value in range(1, 51)
        ]
        with self.assertRaisesRegex(RuntimeError, "fallback"):
            _validate_neural_kbrd_candidates(wrong_source, movie_ids)

    def test_only_retrained_checkpoint_is_accepted(self):
        _validate_kbrd_checkpoint_selection(
            Path("somewhere/kbrd_model_retrained"),
            using_real_kbrd=False,
        )
        with self.assertRaisesRegex(ValueError, "retrained"):
            _validate_kbrd_checkpoint_selection(
                Path("somewhere/kbrd_model"),
                using_real_kbrd=False,
            )


class TestSplitAndTokenAccounting(unittest.TestCase):
    def test_split_is_stable_and_conversation_level(self):
        keys = [f"{line}:conversation-{line}" for line in range(1, 100)]
        first = {key: assign_conversation_split(key) for key in keys}
        second = {key: assign_conversation_split(key) for key in reversed(keys)}
        self.assertEqual(first, second)
        self.assertEqual(assign_conversation_split(keys[0]), first[keys[0]])
        self.assertEqual(set(first.values()), {"train", "dev"})

    def test_relevant_extraction_change_changes_scientific_configuration(self):
        with Path("my_crs/config.yaml").open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        original = _training_extraction_configuration(config)
        self.assertEqual(original, EXPECTED_EXTRACTION_CONFIGURATION)
        changed = copy.deepcopy(config)
        changed["extraction"]["spacy_model"] = "different_model"
        self.assertNotEqual(_training_extraction_configuration(changed), original)

    def test_optional_token_analysis_uses_full_chat_without_truncation(self):
        class Tokenizer:
            model_max_length = 3072

            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                return list(range(2050 if len(self.calls) == 1 else 10))

        records = [
            {
                "messages": [{"role": "user", "content": "prompt"}],
                "assistant_target": '{"ranked_ids":[1,2,3,4,5,6,7,8,9,10]}',
            },
            {
                "messages": [{"role": "user", "content": "prompt two"}],
                "assistant_target": '{"ranked_ids":[1,2,3,4,5,6,7,8,9,10]}',
            },
        ]
        tokenizer = Tokenizer()
        result = analyze_sft_token_lengths(records, tokenizer)
        self.assertEqual(result["count_above_2048"], 1)
        self.assertEqual(result["count_above_3072"], 0)
        self.assertFalse(result["truncation"])
        for messages, kwargs in tokenizer.calls:
            self.assertEqual(messages[-1]["role"], "assistant")
            self.assertEqual(
                kwargs,
                {"tokenize": True, "add_generation_prompt": False},
            )


class _Provider:
    def __init__(self, candidates, fail_on_call=None):
        self.candidates = candidates
        self.fail_on_call = fail_on_call
        self.calls = []

    def __call__(self, dialogue, **kwargs):
        self.calls.append((dialogue, kwargs))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("synthetic interruption")
        return ([dict(candidate) for candidate in self.candidates], [])


class TestBuilderIntegration(unittest.TestCase):
    def _fixture(self, root: Path):
        movie_ids = list(range(101, 156))
        mapping_dir = root / "mapping"
        mapping_dir.mkdir()
        id2entity = {value - 100: f"<movie-{value}>" for value in movie_ids}
        entity2id = {uri: 100 + redial for redial, uri in id2entity.items()}
        for name, value in (
            ("id2entity.pkl", id2entity),
            ("entity2entityId.pkl", entity2id),
            ("movie_ids.pkl", movie_ids),
        ):
            with (mapping_dir / name).open("wb") as handle:
                pickle.dump(value, handle)
        catalogue_path = mapping_dir / "movies_with_mentions.csv"
        catalogue_path.write_text(
            "movieId,title\n1,Synthetic title 01\n",
            encoding="utf-8",
        )
        mapping = ReDialKBRDMapping.load(mapping_dir)

        mentions = {
            str(redial): f"Synthetic title {redial:02d}"
            for redial in range(1, 56)
        }
        conversations = [
            {
                "conversationId": "alpha",
                "initiatorWorkerId": 1,
                "respondentWorkerId": 2,
                "movieMentions": mentions,
                "respondentQuestions": {
                    "2": {"suggested": 1},
                    "3": {"suggested": 1},
                },
                "messages": [
                    {"senderWorkerId": 1, "text": "prior @1"},
                    {"senderWorkerId": 2, "text": "target-one @2"},
                    {"senderWorkerId": 1, "text": "middle context"},
                    {"senderWorkerId": 2, "text": "target-two @3"},
                    {"senderWorkerId": 1, "text": "future sentinel"},
                ],
            },
            {
                "conversationId": "beta",
                "initiatorWorkerId": 1,
                "respondentWorkerId": 2,
                "movieMentions": mentions,
                "respondentQuestions": {"5": {"suggested": 1}},
                "messages": [
                    {"senderWorkerId": 1, "text": "prior @4"},
                    {"senderWorkerId": 2, "text": "target-three @5"},
                ],
            },
            {
                "conversationId": "gamma",
                "initiatorWorkerId": 1,
                "respondentWorkerId": 2,
                "movieMentions": mentions,
                "respondentQuestions": {"55": {"suggested": 1}},
                "messages": [
                    {"senderWorkerId": 1, "text": "prior @54"},
                    {"senderWorkerId": 2, "text": "unreachable target @55"},
                ],
            },
        ]
        source = root / "train_data.jsonl"
        source.write_text(
            "".join(json.dumps(item) + "\n" for item in conversations),
            encoding="utf-8",
        )
        source_sha = _sha256(source)
        counts = build_count_data(conversations, mapping)
        counts["metadata"] = {"source_split": "TRAIN", "source_sha256": source_sha}
        count_path = root / "train_counts.pkl"
        with count_path.open("wb") as handle:
            pickle.dump(counts, handle)
        checkpoint = root / "kbrd_model_retrained"
        checkpoint.write_bytes(b"synthetic checkpoint")
        title_by_id = {
            entity_id: f"Synthetic title {entity_id - 100:02d}"
            for entity_id in movie_ids
        }
        candidates = [
            {
                "id": entity_id,
                "title": title_by_id[entity_id],
                "source": "KBRD_NEURAL",
            }
            for entity_id in movie_ids[:50]
        ]
        expectations = ReconstructionExpectations(3, 3, 4, 4, 1)
        return {
            "source": source,
            "source_sha": source_sha,
            "count_path": count_path,
            "mapping_dir": mapping_dir,
            "catalogue_path": catalogue_path,
            "checkpoint": checkpoint,
            "expectations": expectations,
            "candidates": candidates,
            "title_lookup": title_by_id.get,
        }

    def _build(self, fixture, output, provider, **overrides):
        options = {
            "train_path": fixture["source"],
            "official_path": fixture["source"],
            "expected_train_sha256": fixture["source_sha"],
            "count_path": fixture["count_path"],
            "mapping_dir": fixture["mapping_dir"],
            "movie_catalogue_path": fixture["catalogue_path"],
            "kbrd_checkpoint": fixture["checkpoint"],
            "output_dir": output,
            "reconstruction_expectations": fixture["expectations"],
            "kbrd_candidate_fn": provider,
            "prepare_input_fn": lambda history: ([101], [], [], [], []),
            "title_lookup": fixture["title_lookup"],
        }
        options.update(overrides)
        return build_rrf_train_dataset(**options)

    def test_full_synthetic_build_reuses_loo_view_and_frozen_rrf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            provider = _Provider(fixture["candidates"])
            from my_crs.loo_ckg_retriever import LazyLOOCKGRetriever

            original_view = LazyLOOCKGRetriever.for_conversation
            with patch.object(
                LazyLOOCKGRetriever,
                "for_conversation",
                autospec=True,
                side_effect=original_view,
            ) as view_mock, patch(
                "my_crs.build_rrf_train_dataset.reciprocal_rank_fusion",
                wraps=frozen_rrf,
            ) as rrf_mock:
                summary = self._build(fixture, root / "output", provider)
            self.assertEqual(view_mock.call_count, 3)
            self.assertEqual(rrf_mock.call_count, 4)
            for call in rrf_mock.call_args_list:
                self.assertEqual(call.kwargs, {"rrf_k": 60, "top_k": 50})
            self.assertEqual(len(provider.calls), 4)
            for dialogue, kwargs in provider.calls:
                self.assertEqual(kwargs["top_k"], 50)
                self.assertFalse(kwargs["use_fusion"])
                self.assertEqual(kwargs["retrieval_mode"], "kbrd")
                self.assertNotIn("future sentinel", dialogue)
            self.assertEqual(summary["processed"]["instances"], 4)
            self.assertEqual(summary["processed"]["eligible"], 3)
            self.assertEqual(summary["processed"]["excluded"], 1)

            audit = [
                json.loads(line)
                for line in (root / "output" / AUDIT_FILENAME).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            sft = [
                json.loads(line)
                for line in (root / "output" / SFT_FILENAME).read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(audit), 4)
            self.assertEqual(len(sft), 3)
            self.assertNotIn("target-one", audit[0]["history"])
            self.assertNotIn("target-two", audit[0]["history"])
            user_prompt = sft[0]["messages"][1]["content"]
            for forbidden in (
                "ground_truth_titles",
                "positive_positions",
                "rrf_score",
                "kbrd_rank",
                "ckg_rank",
                '"id":',
                "target-one",
                "future sentinel",
            ):
                self.assertNotIn(forbidden, user_prompt)
            self.assertEqual(
                parse_ranked_positions(sft[0]["assistant_target"], candidate_count=50),
                audit[0]["target_positions"],
            )
            ineligible = next(record for record in audit if record["conversation_id"] == "gamma")
            self.assertFalse(ineligible["eligible"])
            self.assertEqual(
                ineligible["exclusion_reason"],
                "ground_truth_absent_from_rrf_top50",
            )
            self.assertIsNone(ineligible["assistant_target"])
            self.assertNotIn(155, [item["id"] for item in ineligible["rrf_top50"]])
            self.assertNotIn(ineligible["instance_key"], {record["instance_key"] for record in sft})
            catalogue_provenance = summary["provenance"]["mapping_artifacts"][
                "movies_with_mentions.csv"
            ]
            self.assertEqual(
                catalogue_provenance,
                {
                    "path": str(fixture["catalogue_path"].resolve()),
                    "sha256": _sha256(fixture["catalogue_path"]),
                },
            )

    def test_partial_resume_equals_fresh_and_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            resumed_dir = root / "resumed"
            with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                self._build(
                    fixture,
                    resumed_dir,
                    _Provider(fixture["candidates"], fail_on_call=2),
                )
            with self.assertRaisesRegex(ValueError, "fingerprint/configuration"):
                self._build(
                    fixture,
                    resumed_dir,
                    _Provider(fixture["candidates"]),
                    resume=True,
                    max_instances=2,
                )
            self._build(
                fixture,
                resumed_dir,
                _Provider(fixture["candidates"]),
                resume=True,
            )
            fresh_dir = root / "fresh"
            self._build(
                fixture,
                fresh_dir,
                _Provider(fixture["candidates"]),
            )
            for filename in (CONTRIBUTIONS_FILENAME, AUDIT_FILENAME, SFT_FILENAME):
                self.assertEqual(
                    _digest(resumed_dir / filename),
                    _digest(fresh_dir / filename),
                )

    def test_changed_movie_catalogue_hash_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            output = root / "output"
            self._build(fixture, output, _Provider(fixture["candidates"]))
            fixture["catalogue_path"].write_text(
                "movieId,title\n1,Changed title\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fingerprint/configuration"):
                self._build(
                    fixture,
                    output,
                    _Provider(fixture["candidates"]),
                    resume=True,
                )

    def test_changed_extraction_provenance_rejects_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            output = root / "output"
            self._build(fixture, output, _Provider(fixture["candidates"]))
            changed = dict(_load_project_extraction_configuration())
            changed["weak_seed_threshold"] += 1
            with patch(
                "my_crs.build_rrf_train_dataset._load_project_extraction_configuration",
                return_value=changed,
            ), self.assertRaisesRegex(ValueError, "fingerprint/configuration"):
                self._build(
                    fixture,
                    output,
                    _Provider(fixture["candidates"]),
                    resume=True,
                )

    def test_ineligible_record_stays_in_audit_and_out_of_sft(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            output = root / "output"
            self._build(fixture, output, _Provider(fixture["candidates"]))
            audit = [
                json.loads(line)
                for line in (output / AUDIT_FILENAME).read_text(encoding="utf-8").splitlines()
            ]
            sft_keys = {
                json.loads(line)["instance_key"]
                for line in (output / SFT_FILENAME).read_text(encoding="utf-8").splitlines()
            }
            record = next(item for item in audit if item["conversation_id"] == "gamma")
            self.assertFalse(record["eligible"])
            self.assertEqual(
                record["exclusion_reason"],
                "ground_truth_absent_from_rrf_top50",
            )
            self.assertNotIn(record["instance_key"], sft_keys)
            self.assertNotIn(155, [candidate["id"] for candidate in record["rrf_top50"]])

    def test_subset_limits_do_not_change_global_counts_contributions_or_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            full_dir = root / "full"
            subset_dir = root / "subset"
            full = self._build(
                fixture,
                full_dir,
                _Provider(fixture["candidates"]),
            )
            subset = self._build(
                fixture,
                subset_dir,
                _Provider(fixture["candidates"]),
                max_conversations=1,
                max_instances=1,
            )
            full_record = json.loads(
                (full_dir / AUDIT_FILENAME).read_text(encoding="utf-8").splitlines()[0]
            )
            subset_record = json.loads(
                (subset_dir / AUDIT_FILENAME).read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(full_record["loo_ckg_top50"], subset_record["loo_ckg_top50"])
            self.assertEqual(full_record["rrf_top50"], subset_record["rrf_top50"])
            self.assertEqual(
                full_record["conversation_contribution_digest"],
                subset_record["conversation_contribution_digest"],
            )
            self.assertEqual(full_record["split"], subset_record["split"])
            self.assertEqual(
                _digest(full_dir / CONTRIBUTIONS_FILENAME),
                _digest(subset_dir / CONTRIBUTIONS_FILENAME),
            )
            self.assertEqual(
                len((subset_dir / CONTRIBUTIONS_FILENAME).read_text(encoding="utf-8").splitlines()),
                3,
            )
            self.assertEqual(
                full["provenance"]["ckg"]["global_count_sha256"],
                subset["provenance"]["ckg"]["global_count_sha256"],
            )
            self.assertEqual(subset["authoritative_reconstruction"]["conversations"], 3)
            self.assertNotEqual(full["run_fingerprint"], subset["run_fingerprint"])
            self.assertEqual(
                subset["provenance"]["limits"],
                {"max_conversations": 1, "max_instances": 1},
            )

    def test_truncated_resume_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self._fixture(root)
            output = root / "partial"
            with self.assertRaises(RuntimeError):
                self._build(
                    fixture,
                    output,
                    _Provider(fixture["candidates"], fail_on_call=2),
                )
            with (output / AUDIT_FILENAME).open("a", encoding="utf-8") as handle:
                handle.write("{")
            with self.assertRaisesRegex(ValueError, "Malformed/truncated"):
                self._build(
                    fixture,
                    output,
                    _Provider(fixture["candidates"]),
                    resume=True,
                )


if __name__ == "__main__":
    unittest.main()
