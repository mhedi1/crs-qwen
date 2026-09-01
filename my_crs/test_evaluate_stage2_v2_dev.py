from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from my_crs.build_stage2_v2_dataset import (
    CANDIDATE_ORDER_VERSION,
    DATASET_SCHEMA_VERSION,
    NO_SEED_EXCLUSION,
    SOURCE_AUDIT_SCHEMA_VERSION,
    TOP_K,
    canonical_json_bytes,
    canonical_json_digest,
    serialize_candidates,
)
from my_crs.evaluate_stage2_v2_dev import (
    EVALUATION_INSTANCE_SCHEMA,
    EvaluationCheckpoint,
    REFERENCE_BETA_0_10_LOSS_FINGERPRINT,
    _atomic_jsonl,
    _fingerprint,
    apply_inference_checkpoint_weights,
    build_dev_offset_index,
    compact_instance_record,
    evaluation_configuration,
    load_authoritative_dev_provenance,
    load_evaluation_checkpoint,
    rank_record_from_residuals,
    summarize_rank_records,
    validate_artifact_privacy,
    validate_historical_rrf_baseline,
)
from my_crs.stage2_v2_loss import loss_scientific_fingerprint
from my_crs.train_stage2_v2 import (
    CHECKPOINT_SCHEMA,
    training_scientific_configuration,
    training_scientific_fingerprint,
)


def _record(
    instance_key: str,
    *,
    split: str = "dev",
    positives: list[int] | None = None,
    tied_rrf_scores: bool = False,
) -> dict:
    raw_candidates = []
    for rank in range(1, TOP_K + 1):
        raw_candidates.append(
            {
                "ckg_contribution": 1.0 / (200 + rank),
                "ckg_rank": TOP_K + 1 - rank,
                "id": 800000 + rank,
                "kbrd_contribution": 1.0 / (100 + rank),
                "kbrd_rank": rank,
                "rank": rank,
                "rrf_score": 1.0 if tied_rrf_scores else 1.0 / (60 + rank),
                "source": "RRF",
                "title": f"DEV Fixture Movie {rank}",
            }
        )
    candidates = serialize_candidates(raw_candidates)
    history = f"SEEKER: private fixture for {instance_key}"
    return {
        "candidate_count": TOP_K,
        "candidates": candidates,
        "history": history,
        "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
        "instance_key": instance_key,
        "observed_positive_serialization_positions": positives or [],
        "schema_version": DATASET_SCHEMA_VERSION,
        "serialization_digest": canonical_json_digest(candidates),
        "serialization_order_version": CANDIDATE_ORDER_VERSION,
        "split": split,
    }


def _write_jsonl(path: Path, records: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for record in records:
            encoded = canonical_json_bytes(record) + b"\n"
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def _no_seed_audit_record(instance_key: str = "dev-no-seed") -> dict:
    history = "SEEKER: frozen pre-target history"
    return {
        "assistant_target": None,
        "candidate_digest": None,
        "conversation_key": "dev-conversation",
        "diagnostics": {
            "kbrd": {
                "fallback_reason": "no_inference_seeds",
                "seed_entity_ids": [],
            },
            "loo_ckg": None,
        },
        "eligible": False,
        "exclusion_reason": NO_SEED_EXCLUSION,
        "failures": [],
        "history": history,
        "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
        "instance_key": instance_key,
        "kbrd_top50": [],
        "loo_ckg_top50": [],
        "positive_positions": [],
        "positive_positions_truncated": False,
        "rrf_top50": [],
        "run_fingerprint": "fixture-run-fingerprint",
        "schema_version": SOURCE_AUDIT_SCHEMA_VERSION,
        "source_split": "TRAIN",
        "split": "dev",
        "target_positions": [],
    }


def _tiny_expected_audit_accounting() -> dict:
    empty = {
        "total_events": 0,
        "retrieval_completed": 0,
        "reachable": 0,
        "gold_absent": 0,
        "no_seed": 0,
    }
    one_no_seed = dict(empty, total_events=1, no_seed=1)
    return {
        "global": one_no_seed,
        "train": empty,
        "dev": one_no_seed,
        "conversation_overlap_count": 0,
    }


def _training_configuration() -> dict:
    return training_scientific_configuration(
        beta=0.10,
        seed=42,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        max_optimizer_steps=3,
    )


def _checkpoint_payload() -> dict:
    configuration = _training_configuration()
    fingerprint = training_scientific_fingerprint(
        beta=0.10,
        seed=42,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        max_optimizer_steps=3,
    )
    return {
        "adapter_state": {"lora_A.weight": torch.tensor([1.0])},
        "checkpoint_contents": [
            "lora_adapter",
            "shared_scorer_head",
            "optimizer",
            "rng",
            "training_state",
            "scientific_configuration",
        ],
        "optimizer_state": {"must_not_restore": True},
        "rng_state": {"must_not_restore": True},
        "schema_version": CHECKPOINT_SCHEMA,
        "scientific_configuration": configuration,
        "scientific_fingerprint": fingerprint,
        "scorer_head_state": {"projection.weight": torch.tensor([[2.0]])},
        "training_state": {
            "epoch": 1,
            "events_processed": 3,
            "next_epoch_position": 3,
            "nonzero_lora_gradient_observed": True,
            "optimizer_step": 3,
        },
    }


def _rank_record(
    key: str,
    category: str,
    rrf_rank: int | None,
    model_rank: int | None,
) -> dict:
    delta = model_rank - rrf_rank if model_rank is not None and rrf_rank is not None else None
    return compact_instance_record(
        instance_key=key,
        category=category,
        rrf_rank=rrf_rank,
        model_rank=model_rank,
        rank_delta=delta,
    )


class MetricAndRankingTests(unittest.TestCase):
    def test_metric_formulas_full_denominator_and_movement_accounting(self):
        records = [
            _rank_record("improved", "reachable", 2, 1),
            _rank_record("worsened", "reachable", 1, 3),
            _rank_record("unchanged", "reachable", 4, 4),
            _rank_record("absent", "gold_absent", None, None),
            _rank_record("no-seed", "no_seed", None, None),
        ]
        expected = {
            "total_events": 5,
            "retrieval_completed": 4,
            "reachable": 3,
            "gold_absent": 1,
            "no_seed": 1,
        }
        summary = summarize_rank_records(
            records,
            expected_accounting=expected,
            full_evaluation=True,
        )
        model = summary["metrics"]["model"]["evaluation_scope"]
        self.assertEqual(model["denominator"], 5)
        self.assertEqual(model["hits_at_1"], 1)
        self.assertEqual(model["hits_at_10"], 3)
        self.assertEqual(model["hits_at_50"], 3)
        self.assertAlmostEqual(model["mrr"], (1.0 + 1.0 / 3.0 + 1.0 / 4.0) / 5.0)
        self.assertEqual(summary["metrics"]["model"]["retrieval_completed"]["denominator"], 4)
        self.assertEqual(summary["metrics"]["model"]["reachable_positive"]["denominator"], 3)
        self.assertEqual(
            summary["movement"],
            {
                "improved": 1,
                "mean_model_rank_minus_rrf_rank": 1.0 / 3.0,
                "mean_rrf_rank_minus_model_rank": -1.0 / 3.0,
                "moved_out_of_rank_1": 1,
                "moved_to_rank_1": 1,
                "unchanged": 1,
                "worsened": 1,
            },
        )

    def test_exact_frozen_full_denominator_and_r50_invariant(self):
        records = [
            _rank_record(f"reachable-{index}", "reachable", 50, 50)
            for index in range(1364)
        ]
        records.extend(
            _rank_record(f"absent-{index}", "gold_absent", None, None)
            for index in range(780)
        )
        records.extend(
            _rank_record(f"no-seed-{index}", "no_seed", None, None)
            for index in range(159)
        )
        summary = summarize_rank_records(
            records,
            expected_accounting={
                "total_events": 2303,
                "retrieval_completed": 2144,
                "reachable": 1364,
                "gold_absent": 780,
                "no_seed": 159,
            },
            full_evaluation=True,
        )
        self.assertEqual(summary["accounting"]["total_events"], 2303)
        for system in ("rrf", "model"):
            metric = summary["metrics"][system]["evaluation_scope"]
            self.assertEqual(metric["hits_at_50"], 1364)
            self.assertEqual(metric["recall_at_50"], 1364 / 2303)

    def test_multiple_positive_rank_uses_best_candidate(self):
        positive_positions = (3, 40, 47)
        record = _record("multi-positive", positives=list(positive_positions))
        result = rank_record_from_residuals(record, torch.zeros(TOP_K))
        candidates = sorted(record["candidates"], key=lambda item: item["serialization_position"])
        first_positive_rank = candidates[positive_positions[0] - 1]["rrf_rank"]
        expected = min(
            candidates[position - 1]["rrf_rank"] for position in positive_positions
        )
        self.assertNotEqual(expected, first_positive_rank)
        self.assertEqual(result["rrf_rank"], expected)
        self.assertEqual(result["model_rank"], expected)

    def test_candidate_alignment_survives_input_array_permutation(self):
        record = _record("candidate-order", positives=[7])
        original = rank_record_from_residuals(record, torch.linspace(-1.0, 1.0, TOP_K))
        permuted = dict(record, candidates=list(reversed(record["candidates"])))
        observed = rank_record_from_residuals(permuted, torch.linspace(-1.0, 1.0, TOP_K))
        self.assertEqual(observed, original)

    def test_deterministic_score_ties_use_frozen_rrf_rank(self):
        record = _record("tied", positives=[1], tied_rrf_scores=True)
        result = rank_record_from_residuals(record, torch.zeros(TOP_K))
        candidates = sorted(record["candidates"], key=lambda item: item["serialization_position"])
        self.assertEqual(result["model_rank"], candidates[0]["rrf_rank"])

    def test_synthetic_rrf_baseline_is_recomputed_from_same_records(self):
        records = [
            _rank_record("r1", "reachable", 1, 3),
            _rank_record("r10", "reachable", 10, 20),
            _rank_record("r50", "reachable", 50, 1),
            _rank_record("miss", "gold_absent", None, None),
            _rank_record("seed-miss", "no_seed", None, None),
        ]
        summary = summarize_rank_records(
            records,
            expected_accounting={
                "total_events": 5,
                "retrieval_completed": 4,
                "reachable": 3,
                "gold_absent": 1,
                "no_seed": 1,
            },
            full_evaluation=True,
        )
        rrf = summary["metrics"]["rrf"]["evaluation_scope"]
        self.assertEqual(rrf["recall_at_1"], 1 / 5)
        self.assertEqual(rrf["recall_at_10"], 2 / 5)
        self.assertEqual(rrf["recall_at_50"], 3 / 5)
        self.assertAlmostEqual(rrf["mrr"], (1.0 + 0.1 + 0.02) / 5.0)

    def test_historical_baseline_sanity_is_not_hardcoded_as_output(self):
        validate_historical_rrf_baseline(
            {
                "recall_at_1": 0.05297438,
                "recall_at_10": 0.26834564,
                "recall_at_50": 0.59227095,
                "mrr": 0.12208811,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "sanity check failed"):
            validate_historical_rrf_baseline(
                {
                    "recall_at_1": 0.0,
                    "recall_at_10": 0.26834564,
                    "recall_at_50": 0.59227095,
                    "mrr": 0.12208811,
                }
            )

    def test_partial_smoke_with_no_reachable_event_is_marked_and_safe(self):
        records = [_rank_record("absent-only", "gold_absent", None, None)]
        summary = summarize_rank_records(
            records,
            expected_accounting=None,
            full_evaluation=False,
        )
        self.assertEqual(summary["accounting"]["retrieval_completed"], 1)
        reachable = summary["metrics"]["model"]["reachable_positive"]
        self.assertEqual(reachable["denominator"], 0)
        self.assertIsNone(reachable["mrr"])
        self.assertIsNone(reachable["recall_at_50"])
        checkpoint = EvaluationCheckpoint(
            path=Path("fixture-checkpoint.pt"),
            sha256="a" * 64,
            scientific_configuration=_training_configuration(),
            scientific_fingerprint="b" * 64,
            optimizer_step=3,
            adapter_state={"adapter": torch.tensor([1.0])},
            scorer_head_state={"head": torch.tensor([2.0])},
        )
        configuration = evaluation_configuration(
            checkpoint=checkpoint,
            source_audit_sha256="c" * 64,
            full_evaluation=False,
            max_retrieval_completed_events=1,
        )
        self.assertFalse(configuration["evaluation_scope"]["comparable_full_dev"])
        self.assertEqual(
            configuration["evaluation_scope"]["status"],
            "smoke_partial_not_comparable",
        )


class StreamingAndAccountingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_dataset_index_streams_and_retains_dev_only(self):
        path = self.root / "dataset.jsonl"
        records = [
            _record("train-only", split="train"),
            _record("dev-one"),
            _record("dev-two"),
        ]
        sha = _write_jsonl(path, records)
        with mock.patch.object(
            Path,
            "read_text",
            side_effect=AssertionError("full-file read_text is forbidden"),
        ), mock.patch.object(
            Path,
            "readlines",
            side_effect=AssertionError("readlines is forbidden"),
            create=True,
        ):
            index = build_dev_offset_index(
                path,
                expected_sha256=sha,
                expected_counts={"all": 3, "train": 1, "dev": 2},
            )
        self.assertEqual([entry.instance_key for entry in index.entries], ["dev-one", "dev-two"])
        self.assertEqual(index.read_record(0)["split"], "dev")

    def test_valid_and_test_splits_fail_closed(self):
        for forbidden in ("valid", "test"):
            with self.subTest(split=forbidden):
                path = self.root / f"{forbidden}.jsonl"
                sha = _write_jsonl(path, [_record("forbidden", split=forbidden)])
                with self.assertRaisesRegex(ValueError, "train/dev only"):
                    build_dev_offset_index(
                        path,
                        expected_sha256=sha,
                        expected_counts={"all": 1, "train": 0, "dev": 1},
                    )

    def test_authoritative_audit_supplies_no_seed_count_and_identity(self):
        path = self.root / "audit.jsonl"
        sha = _write_jsonl(path, [_no_seed_audit_record()])
        provenance = load_authoritative_dev_provenance(
            path,
            expected_source_sha256=sha,
            expected_accounting=_tiny_expected_audit_accounting(),
        )
        self.assertEqual(provenance.accounting["total_events"], 1)
        self.assertEqual(provenance.accounting["no_seed"], 1)
        self.assertEqual(provenance.no_seed_instance_keys, ("dev-no-seed",))
        self.assertEqual(provenance.source_audit_sha256, sha)


class CheckpointAndArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_checkpoint_validation_accepts_frozen_science_and_rejects_incompatibility(self):
        path = self.root / "checkpoint.pt"
        torch.save(_checkpoint_payload(), path)
        checkpoint = load_evaluation_checkpoint(path)
        self.assertEqual(checkpoint.optimizer_step, 3)
        self.assertEqual(
            checkpoint.scientific_configuration["loss_fingerprint"],
            REFERENCE_BETA_0_10_LOSS_FINGERPRINT,
        )
        self.assertEqual(loss_scientific_fingerprint(0.10), REFERENCE_BETA_0_10_LOSS_FINGERPRINT)

        bad = _checkpoint_payload()
        bad["scientific_configuration"] = dict(
            bad["scientific_configuration"],
            loss_fingerprint="0" * 64,
        )
        bad["scientific_fingerprint"] = _fingerprint(bad["scientific_configuration"])
        bad_path = self.root / "bad.pt"
        torch.save(bad, bad_path)
        with self.assertRaisesRegex(ValueError, "loss fingerprint mismatch"):
            load_evaluation_checkpoint(bad_path)

    def test_inference_weight_application_never_restores_optimizer_rng_or_progress(self):
        path = self.root / "checkpoint.pt"
        torch.save(_checkpoint_payload(), path)
        checkpoint = load_evaluation_checkpoint(path)
        self.assertFalse(hasattr(checkpoint, "optimizer_state"))
        self.assertFalse(hasattr(checkpoint, "rng_state"))
        self.assertFalse(hasattr(checkpoint, "training_state"))
        ranker = mock.Mock()
        ranker.base_model = object()
        peft_module = SimpleNamespace(set_peft_model_state_dict=mock.Mock())
        with mock.patch(
            "my_crs.evaluate_stage2_v2_dev.load_scorer_head_state_dict"
        ) as load_head:
            apply_inference_checkpoint_weights(
                ranker,
                checkpoint,
                peft_module=peft_module,
            )
        peft_module.set_peft_model_state_dict.assert_called_once_with(
            ranker.base_model,
            checkpoint.adapter_state,
        )
        load_head.assert_called_once_with(ranker, checkpoint.scorer_head_state)
        ranker.eval.assert_called_once_with()

    def test_evaluation_fingerprint_is_deterministic_and_runtime_free(self):
        configuration = _training_configuration()
        checkpoint = EvaluationCheckpoint(
            path=self.root / "checkpoint.pt",
            sha256="a" * 64,
            scientific_configuration=configuration,
            scientific_fingerprint="b" * 64,
            optimizer_step=9,
            adapter_state={"adapter": torch.tensor([1.0])},
            scorer_head_state={"head": torch.tensor([2.0])},
        )
        first = evaluation_configuration(
            checkpoint=checkpoint,
            source_audit_sha256="c" * 64,
            full_evaluation=True,
            max_retrieval_completed_events=None,
        )
        second = evaluation_configuration(
            checkpoint=checkpoint,
            source_audit_sha256="c" * 64,
            full_evaluation=True,
            max_retrieval_completed_events=None,
        )
        self.assertEqual(_fingerprint(first), _fingerprint(second))
        serialized = json.dumps(first, sort_keys=True)
        for forbidden in (
            "runtime_torch_version",
            "runtime_transformers_version",
            "runtime_peft_version",
            "cuda_version",
            "gpu_name",
            "timestamp",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_compact_artifact_has_no_dialogue_title_or_target_leakage(self):
        record = _rank_record("safe-instance", "reachable", 4, 2)
        self.assertEqual(record["schema_version"], EVALUATION_INSTANCE_SCHEMA)
        validate_artifact_privacy(record)
        path = self.root / "instances.jsonl"
        _atomic_jsonl(path, [record])
        contents = path.read_text("utf-8").lower()
        for forbidden in ("dialogue", "history", "title", "ground_truth", "prompt"):
            self.assertNotIn(forbidden, contents)
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            validate_artifact_privacy({"candidate_titles": ["PRIVATE"]})


if __name__ == "__main__":
    unittest.main()
