from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from my_crs.analyze_stage2_v2_tokens import (
    EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
    MANIFEST_FILENAME,
    MODEL_ID,
    PERCENTILE_DEFINITION,
    RECORDS_FILENAME,
    REQUESTED_MODEL_REVISION,
    STANDARD_DEVIATION_DEFINITION,
    SUMMARY_FILENAME,
    _distribution,
    _threshold_statistics,
    analysis_configuration,
    analyze_stage2_v2_tokens,
    load_production_tokenizer,
    nearest_rank_percentile,
    validate_tokenizer,
)
from my_crs.build_stage2_v2_dataset import (
    CANDIDATE_ORDER_VERSION,
    DATASET_SCHEMA_VERSION,
    canonical_json_bytes,
    canonical_json_digest,
    serialize_candidates,
    sha256_file,
)
from my_crs.joint_rrf_ranker import tokenize_scoring_event as frozen_tokenize


TOP_K = 50
TOKENIZER_VOCAB_SIZE = 521


class RecordingFastOffsetTokenizer:
    """Deterministic lightweight tokenizer with the frozen call contract."""

    is_fast = True
    model_max_length = 8192
    pad_token_id = 0
    _commit_hash = REQUESTED_MODEL_REVISION
    init_kwargs = {"_commit_hash": REQUESTED_MODEL_REVISION}

    def __init__(self) -> None:
        self.calls: list[dict[str, bool]] = []

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_attention_mask,
        return_offsets_mapping,
        truncation,
    ):
        policy = {
            "add_special_tokens": add_special_tokens,
            "return_attention_mask": return_attention_mask,
            "return_offsets_mapping": return_offsets_mapping,
            "truncation": truncation,
        }
        self.calls.append(policy)
        if policy != {
            "add_special_tokens": False,
            "return_attention_mask": False,
            "return_offsets_mapping": True,
            "truncation": False,
        }:
            raise AssertionError(f"Unexpected tokenizer policy: {policy}")
        ids = []
        offsets = []
        for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
            token = match.group(0).encode("utf-8")
            token_id = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
            ids.append(1 + token_id % (TOKENIZER_VOCAB_SIZE - 1))
            offsets.append((match.start(), match.end()))
        return {"input_ids": ids, "offset_mapping": offsets}


class SlowOffsetTokenizer(RecordingFastOffsetTokenizer):
    is_fast = False


class MissingOffsetTokenizer(RecordingFastOffsetTokenizer):
    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        return {"input_ids": encoded["input_ids"]}


class WrongRevisionTokenizer(RecordingFastOffsetTokenizer):
    _commit_hash = "f" * 40
    init_kwargs = {"_commit_hash": "f" * 40}


def _record(*, instance_key: str, split: str, history_suffix: str = "") -> dict:
    raw_candidates = []
    for rank in range(1, TOP_K + 1):
        title = f"Candidate Movie {rank}"
        if rank == 7:
            title = "Sensitive Candidate\nTitle Seven"
        raw_candidates.append(
            {
                "ckg_contribution": 1.0 / (200 + rank),
                "ckg_rank": TOP_K + 1 - rank,
                "id": 900000 + rank,
                "kbrd_contribution": 1.0 / (100 + rank),
                "kbrd_rank": rank,
                "rank": rank,
                "rrf_score": 1.0 / (60 + rank) + 1.0 / (110 + rank),
                "source": "RRF",
                "title": title,
            }
        )
    candidates = serialize_candidates(raw_candidates)
    history = "SEEKER: PRIVATE_DIALOGUE_SENTINEL" + history_suffix
    return {
        "candidate_count": TOP_K,
        "candidates": candidates,
        "ground_truth_titles": ["PRIVATE_LABEL_SENTINEL"],
        "history": history,
        "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
        "instance_key": instance_key,
        "observed_positive_rrf_positions": [7],
        "observed_positive_serialization_positions": [3],
        "schema_version": DATASET_SCHEMA_VERSION,
        "serialization_digest": canonical_json_digest(candidates),
        "serialization_order_version": CANDIDATE_ORDER_VERSION,
        "split": split,
        "target_response": "PRIVATE_TARGET_SENTINEL",
    }


def _fixture_records() -> list[dict]:
    # The first two intentionally tie for maximum total/prefix length so key
    # ordering is exercised independently from source order.
    return [
        _record(instance_key="z-maximum", split="train", history_suffix=" long" * 30),
        _record(instance_key="a-maximum", split="train", history_suffix=" long" * 30),
        _record(instance_key="dev-short", split="dev"),
    ]


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("wb") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record) + b"\n")


def _fixture_counts(records: list[dict]) -> dict[str, int]:
    return {
        "all": len(records),
        "train": sum(record["split"] == "train" for record in records),
        "dev": sum(record["split"] == "dev" for record in records),
    }


class Stage2V2TokenAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.records = _fixture_records()
        self.source = self.root / "stage2_v2_candidates.jsonl"
        _write_jsonl(self.source, self.records)
        self.source_sha = sha256_file(self.source)
        self.counts = _fixture_counts(self.records)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, name: str, **overrides):
        arguments = {
            "tokenizer": RecordingFastOffsetTokenizer(),
            "dataset_path": self.source,
            "output_dir": self.root / name,
            "expected_dataset_sha256": self.source_sha,
            "expected_counts": self.counts,
        }
        arguments.update(overrides)
        return analyze_stage2_v2_tokens(**arguments)

    def test_scientific_configuration_pins_frozen_identity(self):
        configuration = analysis_configuration(
            dataset_sha256="a" * 64,
            expected_counts=self.counts,
        )
        self.assertEqual(configuration["model_id"], MODEL_ID)
        self.assertEqual(
            configuration["requested_model_revision"], REQUESTED_MODEL_REVISION
        )
        self.assertEqual(
            configuration["phase2_architecture_fingerprint"],
            EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
        )
        self.assertFalse(configuration["truncation"])
        self.assertFalse(configuration["padding"])
        self.assertFalse(configuration["additional_special_tokens"])

    def test_tokenizer_must_be_fast_and_provide_offsets(self):
        with self.assertRaisesRegex(ValueError, "fast tokenizer"):
            validate_tokenizer(SlowOffsetTokenizer())
        with self.assertRaisesRegex(ValueError, "offset mappings"):
            validate_tokenizer(MissingOffsetTokenizer())

    def test_verifiable_tokenizer_revision_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "resolved commit mismatch"):
            validate_tokenizer(WrongRevisionTokenizer())

    def test_production_loader_pins_model_revision_and_fast_tokenizer(self):
        tokenizer = RecordingFastOffsetTokenizer()
        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as loader:
            self.assertIs(load_production_tokenizer(), tokenizer)
        loader.assert_called_once_with(
            MODEL_ID,
            revision=REQUESTED_MODEL_REVISION,
            use_fast=True,
        )

    def test_dataset_sha_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "dataset SHA256 mismatch"):
            self._build("bad-sha", expected_dataset_sha256="0" * 64)

    def test_wrong_record_count_fails_closed(self):
        wrong_counts = dict(self.counts)
        wrong_counts["all"] += 1
        wrong_counts["train"] += 1
        with self.assertRaisesRegex(ValueError, "accounting mismatch"):
            self._build("bad-count", expected_counts=wrong_counts)

    def test_wrong_candidate_count_fails_closed(self):
        records = _fixture_records()
        records[0]["candidates"].pop()
        records[0]["candidate_count"] = 49
        _write_jsonl(self.source, records)
        with self.assertRaisesRegex(ValueError, "exactly 50 candidates"):
            self._build(
                "bad-candidates",
                expected_dataset_sha256=sha256_file(self.source),
                expected_counts=_fixture_counts(records),
            )

    def test_streaming_reuses_exact_frozen_tokenizer_path_without_truncation(self):
        tokenizer = RecordingFastOffsetTokenizer()
        with mock.patch.object(
            Path,
            "read_text",
            side_effect=AssertionError("source must be streamed"),
        ), mock.patch(
            "my_crs.analyze_stage2_v2_tokens.tokenize_scoring_event",
            wraps=frozen_tokenize,
        ) as reused:
            self._build("streaming", tokenizer=tokenizer)
        self.assertEqual(reused.call_count, len(self.records))
        # One probe plus 51 exact frozen segment-tokenization calls per record.
        self.assertEqual(len(tokenizer.calls), 1 + 51 * len(self.records))
        self.assertTrue(all(not call["truncation"] for call in tokenizer.calls))

    def test_percentiles_and_threshold_counts_are_exact(self):
        values = [1, 2, 3, 4]
        self.assertEqual(nearest_rank_percentile(values, 50.0), 2)
        self.assertEqual(nearest_rank_percentile(values, 90.0), 4)
        self.assertEqual(nearest_rank_percentile(values, 99.5), 4)
        self.assertEqual(
            PERCENTILE_DEFINITION,
            "nearest_rank_ceiling_empirical_v1",
        )
        # This distinguishes nearest-rank empirical p50 from linear
        # interpolation, which would return 50 for the same sample.
        self.assertEqual(nearest_rank_percentile([0, 100], 50.0), 0)

        self.assertEqual(
            STANDARD_DEVIATION_DEFINITION,
            "population_standard_deviation_v1",
        )
        population_std = _distribution([1, 2, 3, 4, 5])["standard_deviation"]
        self.assertAlmostEqual(population_std, math.sqrt(2.0), delta=1e-15)
        self.assertNotAlmostEqual(population_std, math.sqrt(2.5), delta=1e-12)

        thresholds = _threshold_statistics([1024, 1025, 1280, 1281, 4097])
        self.assertEqual(thresholds["gt_1024"]["count"], 4)
        self.assertEqual(thresholds["gt_1024"]["percentage"], 80.0)
        self.assertEqual(thresholds["gt_1280"]["count"], 2)
        self.assertEqual(thresholds["gt_4096"]["count"], 1)
        self.assertEqual(
            thresholds["gt_1280"]["records_requiring_truncation"], 2
        )

    def test_accounting_split_statistics_and_maximum_keys(self):
        result = self._build("statistics")
        summary = json.loads(Path(result["summary_path"]).read_text("utf-8"))
        self.assertEqual(summary["counts"], self.counts)
        self.assertEqual(summary["splits"]["all"]["total_packed_tokens"]["count"], 3)
        self.assertEqual(summary["splits"]["train"]["total_packed_tokens"]["count"], 2)
        self.assertEqual(summary["splits"]["dev"]["total_packed_tokens"]["count"], 1)
        self.assertEqual(
            summary["splits"]["all"]["candidate_block_tokens"]["count"],
            TOP_K * len(self.records),
        )
        self.assertEqual(
            summary["splits"]["all"]["maximum_total_length_instance_keys"],
            ["a-maximum", "z-maximum"],
        )
        self.assertEqual(
            summary["splits"]["all"]["maximum_prefix_length_instance_keys"],
            ["a-maximum", "z-maximum"],
        )
        self.assertEqual(summary["tokenization_failures"], 0)

    def test_outputs_are_byte_reproducible_and_hashes_are_recorded(self):
        first = self._build("first")
        second = self._build("second")
        for key in ("records", "summary", "manifest"):
            first_path = Path(first[f"{key}_path"])
            second_path = Path(second[f"{key}_path"])
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
        manifest = json.loads(Path(first["manifest_path"]).read_text("utf-8"))
        self.assertEqual(
            manifest["artifacts"]["records"]["sha256"], first["records_sha256"]
        )
        self.assertEqual(
            manifest["artifacts"]["summary"]["sha256"], first["summary_sha256"]
        )
        self.assertEqual(
            manifest["analysis_configuration"]["phase2_architecture_fingerprint"],
            EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
        )
        self.assertEqual(
            manifest["runtime_provenance"]["resolved_tokenizer_commit"],
            REQUESTED_MODEL_REVISION,
        )
        self.assertEqual(manifest["source"]["record_count"], len(self.records))

    def test_per_record_output_contains_only_compact_analytical_metadata(self):
        result = self._build("privacy")
        contents = Path(result["records_path"]).read_text("utf-8")
        for forbidden in (
            "PRIVATE_DIALOGUE_SENTINEL",
            "Sensitive Candidate",
            "PRIVATE_LABEL_SENTINEL",
            "PRIVATE_TARGET_SENTINEL",
            '"history"',
            '"candidates"',
            '"ground_truth_titles"',
            '"observed_positive',
            '"target_response"',
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, contents)
        records = [json.loads(line) for line in contents.splitlines()]
        self.assertEqual(len(records), len(self.records))
        self.assertTrue(all(record["split"] in {"train", "dev"} for record in records))
        self.assertTrue(
            all(len(record["score_marker_token_indices"]) == TOP_K for record in records)
        )
        self.assertTrue(
            all(len(record["score_marker_position_ids"]) == TOP_K for record in records)
        )

    def test_expected_artifact_names_are_used(self):
        result = self._build("names")
        self.assertEqual(Path(result["records_path"]).name, RECORDS_FILENAME)
        self.assertEqual(Path(result["summary_path"]).name, SUMMARY_FILENAME)
        self.assertEqual(Path(result["manifest_path"]).name, MANIFEST_FILENAME)


if __name__ == "__main__":
    unittest.main()
