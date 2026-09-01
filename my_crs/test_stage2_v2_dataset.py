from __future__ import annotations

import copy
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from my_crs.build_stage2_v2_dataset import (
    DATASET_FILENAME,
    DEFAULT_SOURCE_AUDIT,
    EXPECTED_ACCOUNTING,
    EXPECTED_SOURCE_AUDIT_SHA256,
    GOLD_ABSENT_EXCLUSION,
    LABEL_ANCHOR_ONLY,
    LABEL_OBSERVED,
    IMPLEMENTATION_BASE_COMMIT,
    MANIFEST_FILENAME,
    NO_SEED_EXCLUSION,
    SUMMARY_FILENAME,
    TOP_K,
    UNORDERED_SET_SEMANTICS,
    build_stage2_v2_dataset,
    canonical_json_bytes,
    canonical_json_digest,
    normalize_title,
    scan_source_audit,
    serialize_candidates,
    sha256_file,
    transform_audit_record,
    validate_source_audit,
)


def _source_candidate_digest(candidates: list[dict]) -> str:
    by_rank = sorted(candidates, key=lambda candidate: candidate["rank"])
    return canonical_json_digest(
        [
            {
                "position": position,
                "id": candidate["id"],
                "title": candidate["title"],
            }
            for position, candidate in enumerate(by_rank, 1)
        ]
    )


def _candidates() -> list[dict]:
    candidates = []
    for rank in range(1, TOP_K + 1):
        title = f"Film {rank}"
        if rank == 4:
            title = "Shared Film (1999)"
        elif rank == 28:
            title = "shared film!"
        candidates.append(
            {
                "ckg_contribution": 1.0 / (60 + rank),
                "ckg_rank": TOP_K + 1 - rank,
                "id": 1000 + rank,
                "kbrd_contribution": 1.0 / (60 + rank),
                "kbrd_rank": rank,
                "rank": rank,
                "rrf_score": 2.0 / (60 + rank),
                "source": "RRF",
                "title": title,
            }
        )
    return candidates


def _record(
    kind: str,
    *,
    instance_number: int,
    split: str,
    conversation_key: str,
) -> dict:
    history = f"SEEKER: prior dialogue for {instance_number}\nRECOMMENDER: prior reply"
    base = {
        "assistant_target": None,
        "candidate_digest": None,
        "conversation_contribution_digest": "c" * 64,
        "conversation_id": instance_number,
        "conversation_key": conversation_key,
        "eligible": False,
        "exclusion_reason": None,
        "failures": [],
        "ground_truth_titles": ["Unlisted Target"],
        "history": history,
        "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
        "instance_key": f"fixture:{instance_number}",
        "kbrd_top50": [],
        "line_number": instance_number,
        "loo_ckg_top50": [],
        "normalized_ground_truth_titles": ["unlisted target"],
        "positive_positions": [],
        "positive_positions_truncated": False,
        "rrf_top50": [],
        "run_fingerprint": "f" * 64,
        "schema_version": "rrf_train_audit_v2",
        "source_split": "TRAIN",
        "split": split,
        "target_positions": [],
        "turn_index": instance_number,
        "unique_annotated_target_count": 1,
    }
    if kind == "no_seed":
        base["exclusion_reason"] = NO_SEED_EXCLUSION
        base["diagnostics"] = {
            "all_extracted_entity_ids": [],
            "kbrd": {
                "fallback_reason": "no_inference_seeds",
                "seed_entity_ids": [],
            },
            "loo_ckg": None,
        }
        return base

    candidates = _candidates()
    base.update(
        {
            "candidate_digest": _source_candidate_digest(candidates),
            "diagnostics": {"kbrd": {"fallback_reason": None}, "loo_ckg": {}},
            "kbrd_top50": [{"id": candidate["id"]} for candidate in candidates],
            "loo_ckg_top50": [{"id": candidate["id"]} for candidate in candidates],
            "rrf_top50": candidates,
        }
    )
    if kind == "reachable":
        base.update(
            {
                "assistant_target": '{"ranked_ids":[4,28,1,2,3,5,6,7,8,9]}',
                "eligible": True,
                "ground_truth_titles": ["Shared Film"],
                "normalized_ground_truth_titles": ["shared film"],
                "positive_positions": [4, 28],
                "target_positions": [4, 28, 1, 2, 3, 5, 6, 7, 8, 9],
            }
        )
    elif kind == "gold_absent":
        base["exclusion_reason"] = GOLD_ABSENT_EXCLUSION
    else:
        raise ValueError(kind)
    return base


def _fixture_records() -> list[dict]:
    return [
        _record(
            "reachable",
            instance_number=1,
            split="train",
            conversation_key="train-conversation",
        ),
        _record(
            "gold_absent",
            instance_number=2,
            split="dev",
            conversation_key="dev-conversation",
        ),
        _record(
            "no_seed",
            instance_number=3,
            split="train",
            conversation_key="train-no-seed-conversation",
        ),
    ]


FIXTURE_ACCOUNTING = {
    "global": {
        "total_events": 3,
        "retrieval_completed": 2,
        "reachable": 1,
        "gold_absent": 1,
        "no_seed": 1,
    },
    "train": {
        "total_events": 2,
        "retrieval_completed": 1,
        "reachable": 1,
        "gold_absent": 0,
        "no_seed": 1,
    },
    "dev": {
        "total_events": 1,
        "retrieval_completed": 1,
        "reachable": 0,
        "gold_absent": 1,
        "no_seed": 0,
    },
    "conversation_overlap_count": 0,
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(record) + b"\n" for record in records))


def _build_fixture(source: Path, output: Path) -> dict:
    return build_stage2_v2_dataset(
        source_audit=source,
        output_dir=output,
        expected_source_sha256=sha256_file(source),
        expected_accounting=FIXTURE_ACCOUNTING,
    )


class FrozenAuditIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.observed_sha256 = sha256_file(DEFAULT_SOURCE_AUDIT)
        cls.statistics = scan_source_audit(
            DEFAULT_SOURCE_AUDIT,
            expected_source_sha256=cls.observed_sha256,
        )

    def test_source_audit_sha256(self):
        self.assertEqual(self.observed_sha256, EXPECTED_SOURCE_AUDIT_SHA256)

    def test_exact_global_accounting(self):
        self.assertEqual(
            self.statistics["accounting"]["global"], EXPECTED_ACCOUNTING["global"]
        )

    def test_exact_train_accounting(self):
        self.assertEqual(
            self.statistics["accounting"]["train"], EXPECTED_ACCOUNTING["train"]
        )

    def test_exact_dev_accounting(self):
        self.assertEqual(
            self.statistics["accounting"]["dev"], EXPECTED_ACCOUNTING["dev"]
        )

    def test_train_dev_conversations_do_not_overlap(self):
        self.assertEqual(self.statistics["conversation_overlap_count"], 0)

    def test_full_scan_validates_every_retrieval_completed_top50(self):
        self.assertEqual(
            self.statistics["accounting"]["global"]["retrieval_completed"],
            22199,
        )
        self.assertEqual(
            sum(
                int(count)
                for count in self.statistics["observed_positive_set_cardinality"][
                    "global"
                ].values()
            ),
            22199,
        )


class CandidateSerializationTests(unittest.TestCase):
    def test_hash_order_is_deterministic_and_has_exact_positions(self):
        first = serialize_candidates(_candidates())
        second = serialize_candidates(copy.deepcopy(_candidates()))
        self.assertEqual(first, second)
        self.assertEqual(
            [candidate["serialization_position"] for candidate in first],
            list(range(1, TOP_K + 1)),
        )
        self.assertEqual(
            {candidate["rrf_rank"] for candidate in first},
            set(range(1, TOP_K + 1)),
        )
        ids = [candidate["canonical_entity_id"] for candidate in first]
        self.assertEqual(len(ids), len(set(ids)))

    def test_incoming_array_shuffle_does_not_change_serialization(self):
        original = _record(
            "reachable", instance_number=1, split="train", conversation_key="c1"
        )
        shuffled = copy.deepcopy(original)
        random.Random(9173).shuffle(shuffled["rrf_top50"])
        first = transform_audit_record(original, 1)
        second = transform_audit_record(shuffled, 1)
        self.assertEqual(first["candidates"], second["candidates"])
        self.assertEqual(first["serialization_digest"], second["serialization_digest"])

    def test_order_does_not_depend_on_rrf_rank_score_or_labels(self):
        candidates = _candidates()
        baseline_ids = [
            candidate["canonical_entity_id"] for candidate in serialize_candidates(candidates)
        ]
        changed = copy.deepcopy(candidates)
        for candidate in changed:
            candidate["rank"] = TOP_K + 1 - candidate["rank"]
            candidate["rrf_score"] = float(candidate["id"] * 1000)
        changed_ids = [
            candidate["canonical_entity_id"] for candidate in serialize_candidates(changed)
        ]
        self.assertEqual(baseline_ids, changed_ids)

        positive_one = _record(
            "reachable", instance_number=1, split="train", conversation_key="c1"
        )
        positive_two = copy.deepcopy(positive_one)
        positive_two.update(
            {
                "ground_truth_titles": ["Film 2"],
                "normalized_ground_truth_titles": ["film 2"],
                "positive_positions": [2],
            }
        )
        self.assertEqual(
            transform_audit_record(positive_one, 1)["candidates"],
            transform_audit_record(positive_two, 1)["candidates"],
        )

    def test_positive_mapping_is_exact_and_set_semantics_are_explicit(self):
        source = _record(
            "reachable", instance_number=1, split="train", conversation_key="c1"
        )
        output = transform_audit_record(source, 1)
        by_id = {
            candidate["canonical_entity_id"]: candidate["serialization_position"]
            for candidate in output["candidates"]
        }
        expected_mapping = {
            1004: (4, by_id[1004]),
            1028: (28, by_id[1028]),
        }
        observed_mapping = {
            mapping["canonical_entity_id"]: (
                mapping["rrf_position"],
                mapping["serialization_position"],
            )
            for mapping in output["observed_positive_position_mappings"]
        }
        self.assertEqual(observed_mapping, expected_mapping)
        self.assertEqual(set(output["observed_positive_rrf_positions"]), {4, 28})
        self.assertEqual(
            set(output["observed_positive_serialization_positions"]),
            {by_id[1004], by_id[1028]},
        )
        self.assertEqual(output["label_set_semantics"], UNORDERED_SET_SEMANTICS)
        self.assertEqual(output["label_status"], LABEL_OBSERVED)

    def test_duplicate_normalized_titles_are_not_collapsed(self):
        output = transform_audit_record(
            _record(
                "reachable", instance_number=1, split="train", conversation_key="c1"
            ),
            1,
        )
        shared = [
            candidate
            for candidate in output["candidates"]
            if normalize_title(candidate["title_original"]) == "shared film"
        ]
        self.assertEqual(len(output["candidates"]), TOP_K)
        self.assertEqual(len(shared), 2)
        self.assertEqual(len(output["observed_positive_serialization_positions"]), 2)

    def test_serialization_digest_is_deterministic(self):
        output = transform_audit_record(
            _record(
                "reachable", instance_number=1, split="train", conversation_key="c1"
            ),
            1,
        )
        self.assertEqual(
            output["serialization_digest"],
            canonical_json_digest(output["candidates"]),
        )


class RecordAndArtifactTests(unittest.TestCase):
    def test_gold_absent_is_retained_as_anchor_only(self):
        output = transform_audit_record(
            _record(
                "gold_absent", instance_number=2, split="dev", conversation_key="c2"
            ),
            2,
        )
        self.assertIsNotNone(output)
        self.assertEqual(len(output["candidates"]), TOP_K)
        self.assertEqual(output["observed_positive_rrf_positions"], [])
        self.assertEqual(output["observed_positive_serialization_positions"], [])
        self.assertEqual(output["observed_positive_position_mappings"], [])
        self.assertEqual(output["label_status"], LABEL_ANCHOR_ONLY)

    def test_no_seed_does_not_create_a_ranker_record(self):
        self.assertIsNone(
            transform_audit_record(
                _record(
                    "no_seed", instance_number=3, split="train", conversation_key="c3"
                ),
                3,
            )
        )

    def test_history_is_copied_exactly_without_reconstruction(self):
        source = _record(
            "reachable", instance_number=1, split="train", conversation_key="c1"
        )
        output = transform_audit_record(source, 1)
        self.assertEqual(output["history"], source["history"])
        self.assertEqual(output["history_sha256"], source["history_sha256"])

    def test_fixture_build_has_exact_records_and_no_seed_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.audit.jsonl"
            output = root / "output"
            _write_jsonl(source, _fixture_records())
            result = _build_fixture(source, output)
            lines = (output / DATASET_FILENAME).read_text(encoding="utf-8").splitlines()
            self.assertEqual(result["output_records"], 2)
            self.assertEqual(len(lines), 2)
            for line in lines:
                record = json.loads(line)
                self.assertEqual(record["candidate_count"], TOP_K)
                self.assertEqual(len(record["candidates"]), TOP_K)
                self.assertEqual(
                    len(
                        {
                            candidate["canonical_entity_id"]
                            for candidate in record["candidates"]
                        }
                    ),
                    TOP_K,
                )
            summary = json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8"))
            manifest = json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(summary["accounting"]["global"]["no_seed"], 1)
            self.assertEqual(manifest["accounting"]["global"]["no_seed"], 1)
            self.assertEqual(summary["dataset"]["records"], 2)

    def test_core_outputs_are_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.audit.jsonl"
            _write_jsonl(source, _fixture_records())
            first = root / "first"
            second = root / "second"
            _build_fixture(source, first)
            _build_fixture(source, second)
            for filename in (DATASET_FILENAME, SUMMARY_FILENAME, MANIFEST_FILENAME):
                self.assertEqual((first / filename).read_bytes(), (second / filename).read_bytes())

    def test_builder_does_not_depend_on_git_head_or_git_availability(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.audit.jsonl"
            output = root / "output"
            _write_jsonl(source, _fixture_records())
            with mock.patch("subprocess.check_output") as git_command:
                _build_fixture(source, output)
            git_command.assert_not_called()
            manifest = json.loads((output / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["implementation_base_commit"], IMPLEMENTATION_BASE_COMMIT
            )
            self.assertEqual(
                manifest["builder_configuration"]["implementation_base_commit"],
                IMPLEMENTATION_BASE_COMMIT,
            )
            self.assertNotIn("expected_repository_head", manifest)
            self.assertNotIn(
                "expected_repository_head", manifest["builder_configuration"]
            )

    def test_wrong_source_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.audit.jsonl"
            _write_jsonl(source, _fixture_records())
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                validate_source_audit(source, expected_sha256="0" * 64)

    def test_malformed_candidate_records_fail_closed(self):
        valid = _record(
            "reachable", instance_number=1, split="train", conversation_key="c1"
        )
        malformed_count = copy.deepcopy(valid)
        malformed_count["rrf_top50"].pop()
        duplicate_id = copy.deepcopy(valid)
        duplicate_id["rrf_top50"][1]["id"] = duplicate_id["rrf_top50"][0]["id"]
        duplicate_id["candidate_digest"] = _source_candidate_digest(
            duplicate_id["rrf_top50"]
        )
        bad_history = copy.deepcopy(valid)
        bad_history["history"] += "\nTARGET: leaked"
        for malformed in (malformed_count, duplicate_id, bad_history):
            with self.subTest(instance=malformed):
                with self.assertRaises(ValueError):
                    transform_audit_record(malformed, 1)

    def test_malformed_no_seed_provenance_fails_closed(self):
        valid = _record(
            "no_seed", instance_number=3, split="train", conversation_key="c3"
        )
        for field, value in (
            ("candidate_digest", "not-null"),
            ("target_positions", [1]),
        ):
            malformed = copy.deepcopy(valid)
            malformed[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    transform_audit_record(malformed, 3)
        contradictory = copy.deepcopy(valid)
        contradictory["diagnostics"]["kbrd"]["seed_entity_ids"] = [123]
        with self.assertRaises(ValueError):
            transform_audit_record(contradictory, 3)


if __name__ == "__main__":
    unittest.main()
