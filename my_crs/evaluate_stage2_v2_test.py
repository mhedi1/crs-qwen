"""Final one-shot TEST evaluation for the selected Stage-2-v2 beta=1 model.

This evaluator is intentionally TEST-only.

Scientific policy:
- model selection was completed on TRAIN-derived DEV;
- beta=1.0 is already frozen as the selected configuration;
- only the selected checkpoint is accepted;
- only the frozen TEST candidate artifacts are accepted;
- TEST cannot be evaluated partially;
- candidate membership remains the frozen RRF Top-50;
- Stage-2 changes only candidate ordering;
- gold-absent and no-seed events are full-denominator misses;
- --validate-only performs no Qwen model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from my_crs.build_stage2_v2_dataset import (
    CANDIDATE_ORDER_VERSION,
    DATASET_SCHEMA_VERSION,
    GOLD_ABSENT_EXCLUSION,
    LABEL_ANCHOR_ONLY,
    LABEL_OBSERVED,
    NO_SEED_EXCLUSION,
    PROJECT_ROOT,
    TOP_K,
    canonical_json_bytes,
)
from my_crs.evaluate_stage2_v2_dev import (
    load_evaluation_checkpoint,
    load_inference_stack,
)
from my_crs.joint_rrf_ranker import (
    RANKING_POLICY_VERSION,
    RRF_PRIOR_POLICY_VERSION,
    canonicalize_phase1_candidates,
    combine_rrf_prior,
    rank_candidate_ids,
)
from my_crs.stage2_v2_loss import (
    EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT,
)
from my_crs.stage2_v2_peft import (
    EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT,
    MODEL_ID,
    REQUESTED_MODEL_REVISION,
    load_production_tokenizer,
    require_single_cuda_device,
    runtime_provenance,
    tokenize_single_smoke_event,
    validate_tokenizer,
)
from my_crs.analyze_stage2_v2_tokens import (
    EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
)


DEFAULT_TEST_DIR = (
    PROJECT_ROOT / "experiments" / "stage2_v2_test_full"
)

DEFAULT_DATASET_PATH = (
    DEFAULT_TEST_DIR / "stage2_v2_test_candidates.jsonl"
)
DEFAULT_SOURCE_AUDIT = (
    DEFAULT_TEST_DIR / "stage2_v2_test_audit.jsonl"
)
DEFAULT_SOURCE_SUMMARY = (
    DEFAULT_TEST_DIR / "stage2_v2_test_summary.json"
)
DEFAULT_SOURCE_MANIFEST = (
    DEFAULT_TEST_DIR / "stage2_v2_test_manifest.json"
)

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "stage2_v2_3b_beta100_seed42"
    / "checkpoints"
    / "checkpoint_step_00001254.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "stage2_v2_test_eval_beta100_seed42"
)


EXPECTED_TEST_DATASET_SHA256 = (
    "ddb81b08accd4df6860bf18a8178727dc74604624a95782c2c23930ef68fa6f5"
)
EXPECTED_TEST_AUDIT_SHA256 = (
    "c420bef4e2bd6ba03b1dadcfea1279c60708258ce1e0416857fe793cb366cdd0"
)
EXPECTED_TEST_SUMMARY_SHA256 = (
    "1bca3053b8d1320f0e3a56b57d56ae252481f798f47309cb0b94232ebe43bcc6"
)
EXPECTED_TEST_MANIFEST_SHA256 = (
    "8c654e03728038733dcbd04b8972a58ab1f261b3364ac56c2a89cb6d27bcd7c7"
)

EXPECTED_TEST_RUN_FINGERPRINT = (
    "c45aff8646dff3ff5bcfb7e12a93c5cbdd8b9a35679b902a6d152dbc68c38514"
)

EXPECTED_CHECKPOINT_SHA256 = (
    "ed4a90bd4b9d4c7fcca6f41ef6ee54038a2f548b3a2217ac97697b438ca994a4"
)
EXPECTED_CHECKPOINT_SCIENTIFIC_FINGERPRINT = (
    "2d987c1a5b83db04b32c1f06e5d05e9253917b0536f4cb9be396cb3376cd5109"
)
EXPECTED_CHECKPOINT_STEP = 1254
EXPECTED_SELECTED_BETA = 1.0

AUDIT_SCHEMA_VERSION = "rrf_test_audit_v1"

EXPECTED_TEST_ACCOUNTING = {
    "total_events": 3898,
    "retrieval_completed": 3551,
    "reachable": 1364,
    "gold_absent": 2187,
    "no_seed": 347,
}

EXPECTED_TEST_RRF = {
    "hits_at_1": 134,
    "hits_at_10": 664,
    "hits_at_50": 1364,
    "recall_at_1": 0.03437660338635198,
    "recall_at_10": 0.17034376603386353,
    "recall_at_50": 0.34992303745510517,
    "mrr": 0.07711970685790721,
    "reciprocal_rank_sum": 300.61261733212234,
}

EVALUATION_VERSION = "stage2_v2_final_test_evaluator_v1"
EVALUATION_MANIFEST_SCHEMA = "stage2_v2_test_eval_manifest_v1"
EVALUATION_SUMMARY_SCHEMA = "stage2_v2_test_eval_summary_v1"
EVALUATION_INSTANCE_SCHEMA = "stage2_v2_test_eval_instance_v1"
METRIC_POLICY_VERSION = "full_denominator_best_observed_positive_rank_v1"
ARTIFACT_ORDER_POLICY = (
    "retrieval_dataset_order_then_no_seed_audit_order_v1"
)

MANIFEST_FILENAME = "stage2_v2_test_eval_manifest.json"
SUMMARY_FILENAME = "stage2_v2_test_eval_summary.json"
INSTANCES_FILENAME = "stage2_v2_test_eval_instances.jsonl"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def require_file_sha(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> Path:
    source = Path(path).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    observed = sha256_file(source)

    if observed.lower() != expected_sha256.lower():
        raise ValueError(
            f"{label} SHA256 mismatch: "
            f"expected={expected_sha256.lower()} "
            f"observed={observed.lower()}"
        )

    return source


def validated_positions(
    value: Any,
    field: str,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")

    positions: list[int] = []

    for item in value:
        if type(item) is not int or not 1 <= item <= TOP_K:
            raise ValueError(
                f"{field} must contain integers in 1..{TOP_K}"
            )
        positions.append(item)

    if len(positions) != len(set(positions)):
        raise ValueError(f"{field} contains duplicates")

    return tuple(positions)


@dataclass(frozen=True)
class FrozenTestInputs:
    dataset_path: Path
    audit_path: Path
    summary_path: Path
    manifest_path: Path
    records: tuple[dict[str, Any], ...]
    no_seed_instance_keys: tuple[str, ...]
    accounting: dict[str, int]


def validate_frozen_test_inputs(
    *,
    dataset_path: str | Path,
    audit_path: str | Path,
    summary_path: str | Path,
    manifest_path: str | Path,
) -> FrozenTestInputs:

    dataset = require_file_sha(
        dataset_path,
        EXPECTED_TEST_DATASET_SHA256,
        "Frozen TEST candidate dataset",
    )

    audit = require_file_sha(
        audit_path,
        EXPECTED_TEST_AUDIT_SHA256,
        "Frozen TEST audit",
    )

    summary = require_file_sha(
        summary_path,
        EXPECTED_TEST_SUMMARY_SHA256,
        "Frozen TEST source summary",
    )

    manifest = require_file_sha(
        manifest_path,
        EXPECTED_TEST_MANIFEST_SHA256,
        "Frozen TEST source manifest",
    )

    audit_categories: Counter[str] = Counter()
    audit_keys: set[str] = set()
    retrieval_keys: set[str] = set()
    no_seed_keys: list[str] = []

    audit_record_numbers: dict[str, int] = {}
    audit_candidate_digests: dict[str, str | None] = {}
    audit_positive_positions: dict[str, tuple[int, ...]] = {}

    with audit.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"Blank TEST audit line {line_number}"
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid TEST audit JSON at line {line_number}"
                ) from error

            if not isinstance(record, Mapping):
                raise ValueError(
                    f"TEST audit line {line_number} is not an object"
                )

            if record.get("schema_version") != AUDIT_SCHEMA_VERSION:
                raise ValueError(
                    f"TEST audit schema mismatch at line {line_number}"
                )

            if (
                record.get("source_split") != "TEST"
                or record.get("split") != "test"
            ):
                raise ValueError(
                    f"TEST audit split mismatch at line {line_number}"
                )

            if record.get("run_fingerprint") != EXPECTED_TEST_RUN_FINGERPRINT:
                raise ValueError(
                    f"TEST audit run fingerprint mismatch at line {line_number}"
                )

            if record.get("failures") != []:
                raise ValueError(
                    f"TEST audit line {line_number} contains failures"
                )

            instance_key = record.get("instance_key")

            if not isinstance(instance_key, str) or not instance_key:
                raise ValueError(
                    f"TEST audit line {line_number} lacks instance_key"
                )

            if instance_key in audit_keys:
                raise ValueError(
                    f"Duplicate TEST audit instance_key: {instance_key}"
                )

            audit_keys.add(instance_key)
            audit_record_numbers[instance_key] = line_number

            category = record.get("category")

            if category not in {
                "reachable",
                "gold_absent",
                "no_seed",
            }:
                raise ValueError(
                    f"Unknown TEST category at line {line_number}: "
                    f"{category!r}"
                )

            audit_categories[str(category)] += 1

            positives = validated_positions(
                record.get("positive_positions"),
                f"audit[{line_number}].positive_positions",
            )

            audit_positive_positions[instance_key] = positives

            candidate_digest = record.get("candidate_digest")

            if category == "no_seed":
                if record.get("eligible") is not False:
                    raise ValueError(
                        "No-seed TEST event cannot be eligible"
                    )

                if record.get("exclusion_reason") != NO_SEED_EXCLUSION:
                    raise ValueError(
                        "No-seed TEST exclusion reason mismatch"
                    )

                if positives:
                    raise ValueError(
                        "No-seed TEST event cannot contain positives"
                    )

                for field in (
                    "kbrd_top50",
                    "ckg_top50",
                    "rrf_top50",
                ):
                    if record.get(field) != []:
                        raise ValueError(
                            f"No-seed TEST {field} must be empty"
                        )

                if candidate_digest is not None:
                    raise ValueError(
                        "No-seed TEST candidate digest must be null"
                    )

                no_seed_keys.append(instance_key)

            else:
                retrieval_keys.add(instance_key)

                rrf = record.get("rrf_top50")

                if not isinstance(rrf, list) or len(rrf) != TOP_K:
                    raise ValueError(
                        f"Retrieval TEST event {instance_key} "
                        "must contain exactly 50 RRF candidates"
                    )

                ids = [
                    int(candidate["id"])
                    for candidate in rrf
                ]

                if len(ids) != len(set(ids)):
                    raise ValueError(
                        f"Retrieval TEST event {instance_key} "
                        "contains duplicate RRF IDs"
                    )

                if not isinstance(candidate_digest, str) or not candidate_digest:
                    raise ValueError(
                        f"Retrieval TEST event {instance_key} "
                        "lacks candidate digest"
                    )

                if category == "reachable":
                    if not positives:
                        raise ValueError(
                            "Reachable TEST event must contain positives"
                        )
                    if record.get("eligible") is not True:
                        raise ValueError(
                            "Reachable TEST event must be eligible"
                        )
                    if record.get("exclusion_reason") is not None:
                        raise ValueError(
                            "Reachable TEST event cannot have exclusion"
                        )

                if category == "gold_absent":
                    if positives:
                        raise ValueError(
                            "Gold-absent TEST event cannot contain positives"
                        )
                    if record.get("eligible") is not False:
                        raise ValueError(
                            "Gold-absent TEST event cannot be eligible"
                        )
                    if (
                        record.get("exclusion_reason")
                        != GOLD_ABSENT_EXCLUSION
                    ):
                        raise ValueError(
                            "Gold-absent TEST exclusion mismatch"
                        )

            audit_candidate_digests[instance_key] = candidate_digest

    accounting = {
        "total_events": len(audit_keys),
        "retrieval_completed": (
            audit_categories["reachable"]
            + audit_categories["gold_absent"]
        ),
        "reachable": audit_categories["reachable"],
        "gold_absent": audit_categories["gold_absent"],
        "no_seed": audit_categories["no_seed"],
    }

    if accounting != EXPECTED_TEST_ACCOUNTING:
        raise ValueError(
            f"Frozen TEST accounting mismatch: "
            f"{accounting} != {EXPECTED_TEST_ACCOUNTING}"
        )

    dataset_records: list[dict[str, Any]] = []
    dataset_keys: set[str] = set()

    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"Blank TEST dataset line {line_number}"
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid TEST dataset JSON at line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"TEST dataset line {line_number} is not an object"
                )

            if record.get("schema_version") != DATASET_SCHEMA_VERSION:
                raise ValueError(
                    f"TEST dataset schema mismatch at line {line_number}"
                )

            if (
                record.get("source_split") != "TEST"
                or record.get("split") != "test"
            ):
                raise ValueError(
                    f"TEST dataset split mismatch at line {line_number}"
                )

            if (
                record.get("source_run_fingerprint")
                != EXPECTED_TEST_RUN_FINGERPRINT
            ):
                raise ValueError(
                    f"TEST dataset fingerprint mismatch at line {line_number}"
                )

            if (
                record.get("serialization_order_version")
                != CANDIDATE_ORDER_VERSION
            ):
                raise ValueError(
                    f"TEST serialization policy mismatch at line {line_number}"
                )

            instance_key = record.get("instance_key")

            if not isinstance(instance_key, str) or not instance_key:
                raise ValueError(
                    f"TEST dataset line {line_number} lacks instance_key"
                )

            if instance_key in dataset_keys:
                raise ValueError(
                    f"Duplicate TEST dataset instance_key: {instance_key}"
                )

            if instance_key not in retrieval_keys:
                raise ValueError(
                    f"TEST dataset instance {instance_key} "
                    "has no retrieval-completed audit event"
                )

            dataset_keys.add(instance_key)

            if (
                record.get("candidate_count") != TOP_K
                or not isinstance(record.get("candidates"), list)
                or len(record["candidates"]) != TOP_K
            ):
                raise ValueError(
                    f"TEST dataset instance {instance_key} "
                    "must contain exactly 50 candidates"
                )

            canonical_ids = [
                int(candidate["canonical_entity_id"])
                for candidate in record["candidates"]
            ]

            if len(canonical_ids) != len(set(canonical_ids)):
                raise ValueError(
                    f"TEST dataset instance {instance_key} "
                    "contains duplicate canonical IDs"
                )

            expected_history_sha = hashlib.sha256(
                record["history"].encode("utf-8")
            ).hexdigest()

            if record.get("history_sha256") != expected_history_sha:
                raise ValueError(
                    f"TEST history SHA mismatch for {instance_key}"
                )

            source_record_number = record.get(
                "source_audit_record_number"
            )

            if (
                source_record_number
                != audit_record_numbers[instance_key]
            ):
                raise ValueError(
                    f"TEST audit-record provenance mismatch "
                    f"for {instance_key}"
                )

            if (
                record.get("source_candidate_digest")
                != audit_candidate_digests[instance_key]
            ):
                raise ValueError(
                    f"TEST candidate digest provenance mismatch "
                    f"for {instance_key}"
                )

            observed_rrf = validated_positions(
                record.get("observed_positive_rrf_positions"),
                f"dataset[{line_number}].observed_positive_rrf_positions",
            )

            if observed_rrf != audit_positive_positions[instance_key]:
                raise ValueError(
                    f"TEST positive-position provenance mismatch "
                    f"for {instance_key}"
                )

            observed_serialized = validated_positions(
                record.get(
                    "observed_positive_serialization_positions"
                ),
                (
                    f"dataset[{line_number}]"
                    ".observed_positive_serialization_positions"
                ),
            )

            expected_label = (
                LABEL_OBSERVED
                if observed_serialized
                else LABEL_ANCHOR_ONLY
            )

            if record.get("label_status") != expected_label:
                raise ValueError(
                    f"TEST label status mismatch for {instance_key}"
                )

            # Frozen JointRRFRanker validation.
            canonicalize_phase1_candidates(record)

            dataset_records.append(record)

    if len(dataset_records) != EXPECTED_TEST_ACCOUNTING[
        "retrieval_completed"
    ]:
        raise ValueError(
            "Frozen TEST candidate count mismatch"
        )

    if dataset_keys != retrieval_keys:
        missing = retrieval_keys - dataset_keys
        extra = dataset_keys - retrieval_keys
        raise ValueError(
            "TEST audit/dataset identity mismatch: "
            f"missing={len(missing)} extra={len(extra)}"
        )

    if len(no_seed_keys) != EXPECTED_TEST_ACCOUNTING["no_seed"]:
        raise ValueError(
            "Frozen TEST no-seed identity count mismatch"
        )

    if set(no_seed_keys) & dataset_keys:
        raise ValueError(
            "No-seed TEST events leaked into candidate dataset"
        )

    return FrozenTestInputs(
        dataset_path=dataset,
        audit_path=audit,
        summary_path=summary,
        manifest_path=manifest,
        records=tuple(dataset_records),
        no_seed_instance_keys=tuple(no_seed_keys),
        accounting=accounting,
    )


def validate_selected_checkpoint(
    checkpoint_path: str | Path,
):
    checkpoint = load_evaluation_checkpoint(checkpoint_path)

    if checkpoint.sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "Final TEST evaluator rejects non-selected checkpoint SHA"
        )

    if (
        checkpoint.scientific_fingerprint
        != EXPECTED_CHECKPOINT_SCIENTIFIC_FINGERPRINT
    ):
        raise ValueError(
            "Final TEST evaluator rejects non-selected "
            "scientific fingerprint"
        )

    if checkpoint.optimizer_step != EXPECTED_CHECKPOINT_STEP:
        raise ValueError(
            "Final TEST evaluator requires optimizer step 1254"
        )

    loss = checkpoint.scientific_configuration.get("loss")

    if not isinstance(loss, Mapping):
        raise ValueError(
            "Selected checkpoint lacks loss configuration"
        )

    beta = loss.get("beta")

    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or float(beta) != EXPECTED_SELECTED_BETA
    ):
        raise ValueError(
            f"Final TEST evaluator requires beta="
            f"{EXPECTED_SELECTED_BETA}"
        )

    return checkpoint


def compact_instance_record(
    *,
    instance_key: str,
    category: str,
    rrf_rank: int | None,
    model_rank: int | None,
    rank_delta: int | None,
) -> dict[str, Any]:

    if not isinstance(instance_key, str) or not instance_key:
        raise ValueError("Evaluation instance key must be nonempty")

    if category not in {
        "reachable",
        "gold_absent",
        "no_seed",
    }:
        raise ValueError("Unknown TEST evaluation category")

    if category == "reachable":
        for name, rank in (
            ("rrf_rank", rrf_rank),
            ("model_rank", model_rank),
        ):
            if type(rank) is not int or not 1 <= rank <= TOP_K:
                raise ValueError(
                    f"Reachable {name} must be in 1..50"
                )

        if rank_delta != model_rank - rrf_rank:
            raise ValueError(
                "Reachable rank delta is inconsistent"
            )

    elif any(
        value is not None
        for value in (
            rrf_rank,
            model_rank,
            rank_delta,
        )
    ):
        raise ValueError(
            "TEST miss categories cannot contain a rank"
        )

    return {
        "category": category,
        "hit_at_1": bool(
            model_rank is not None and model_rank <= 1
        ),
        "hit_at_10": bool(
            model_rank is not None and model_rank <= 10
        ),
        "hit_at_50": bool(
            model_rank is not None and model_rank <= 50
        ),
        "instance_key": instance_key,
        "model_rank": model_rank,
        "rank_delta_model_minus_rrf": rank_delta,
        "rrf_rank": rrf_rank,
        "schema_version": EVALUATION_INSTANCE_SCHEMA,
    }


def rank_record_from_residuals(
    record: Mapping[str, Any],
    residuals: torch.Tensor | Sequence[float],
) -> dict[str, Any]:

    if (
        record.get("split") != "test"
        or record.get("source_split") != "TEST"
    ):
        raise ValueError(
            "Final Stage-2-v2 evaluator accepts TEST records only"
        )

    instance_key = record.get("instance_key")

    if not isinstance(instance_key, str) or not instance_key:
        raise ValueError(
            "Stage-2-v2 TEST record lacks instance_key"
        )

    candidates = canonicalize_phase1_candidates(record)

    positive_positions = validated_positions(
        record.get(
            "observed_positive_serialization_positions"
        ),
        "observed_positive_serialization_positions",
    )

    canonical_ids = [
        int(candidate["canonical_entity_id"])
        for candidate in candidates
    ]
    rrf_ranks = [
        int(candidate["rrf_rank"])
        for candidate in candidates
    ]
    rrf_scores = [
        float(candidate["rrf_score"])
        for candidate in candidates
    ]

    raw = (
        residuals
        if isinstance(residuals, torch.Tensor)
        else torch.tensor(residuals)
    )

    if raw.shape != (TOP_K,):
        raise ValueError(
            "One TEST event requires exactly 50 residuals"
        )

    prior = torch.tensor(
        rrf_scores,
        dtype=torch.float64,
        device=raw.device,
    )

    combination = combine_rrf_prior(
        prior,
        raw,
    )

    ranked_ids = rank_candidate_ids(
        combination.final_scores,
        canonical_ids,
        rrf_ranks,
    )

    if set(ranked_ids) != set(canonical_ids):
        raise RuntimeError(
            "Stage-2 TEST evaluation changed candidate membership"
        )

    if positive_positions:
        positive_ids = {
            canonical_ids[position - 1]
            for position in positive_positions
        }

        model_by_id = {
            entity_id: rank
            for rank, entity_id in enumerate(
                ranked_ids,
                1,
            )
        }

        model_rank = min(
            model_by_id[entity_id]
            for entity_id in positive_ids
        )

        rrf_rank = min(
            rrf_ranks[position - 1]
            for position in positive_positions
        )

        category = "reachable"
        rank_delta = model_rank - rrf_rank

    else:
        model_rank = None
        rrf_rank = None
        category = "gold_absent"
        rank_delta = None

    return compact_instance_record(
        instance_key=instance_key,
        category=category,
        rrf_rank=rrf_rank,
        model_rank=model_rank,
        rank_delta=rank_delta,
    )


def evaluate_model_record(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    ranker: Any,
    device: torch.device,
) -> dict[str, Any]:

    candidates = canonicalize_phase1_candidates(record)

    event, batch, _actual_tokens = (
        tokenize_single_smoke_event(
            record,
            tokenizer,
        )
    )

    expected_ids = tuple(
        int(candidate["canonical_entity_id"])
        for candidate in candidates
    )
    expected_ranks = tuple(
        int(candidate["rrf_rank"])
        for candidate in candidates
    )
    expected_scores = tuple(
        float(candidate["rrf_score"])
        for candidate in candidates
    )

    if (
        event.canonical_entity_ids != expected_ids
        or event.rrf_ranks != expected_ranks
        or event.rrf_scores != expected_scores
    ):
        raise RuntimeError(
            "TEST tokenized candidate order disagrees "
            "with frozen logical order"
        )

    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        residuals = ranker(
            batch.to(device)
        )

    if (
        not isinstance(residuals, torch.Tensor)
        or residuals.shape != (1, TOP_K)
    ):
        raise RuntimeError(
            "JointRRFRanker must return 50 residuals"
        )

    return rank_record_from_residuals(
        record,
        residuals[0],
    )


def rank_metrics(
    records: Sequence[Mapping[str, Any]],
    rank_field: str,
) -> dict[str, Any]:

    denominator = len(records)

    ranks: list[int] = []

    for record in records:
        rank = record.get(rank_field)

        if rank is None:
            continue

        if type(rank) is not int or not 1 <= rank <= TOP_K:
            raise ValueError(
                f"{rank_field} must be None or integer 1..50"
            )

        ranks.append(rank)

    hits = {
        cutoff: sum(
            rank <= cutoff
            for rank in ranks
        )
        for cutoff in (1, 10, 50)
    }

    reciprocal_rank_sum = math.fsum(
        1.0 / rank
        for rank in ranks
    )

    result: dict[str, Any] = {
        "denominator": denominator,
        "evaluated_rank_count": len(ranks),
        "reciprocal_rank_sum": reciprocal_rank_sum,
        "mrr": (
            reciprocal_rank_sum / denominator
            if denominator
            else None
        ),
        "mrr_percent": (
            100.0 * reciprocal_rank_sum / denominator
            if denominator
            else None
        ),
    }

    for cutoff in (1, 10, 50):
        result[f"hits_at_{cutoff}"] = hits[cutoff]

        result[f"recall_at_{cutoff}"] = (
            hits[cutoff] / denominator
            if denominator
            else None
        )

        result[f"recall_at_{cutoff}_percent"] = (
            100.0 * hits[cutoff] / denominator
            if denominator
            else None
        )

    return result


def summarize_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:

    if not records:
        raise ValueError(
            "No Stage-2-v2 TEST evaluation records produced"
        )

    categories = Counter(
        str(record.get("category"))
        for record in records
    )

    observed = {
        "total_events": len(records),
        "retrieval_completed": (
            categories["reachable"]
            + categories["gold_absent"]
        ),
        "reachable": categories["reachable"],
        "gold_absent": categories["gold_absent"],
        "no_seed": categories["no_seed"],
    }

    if observed != EXPECTED_TEST_ACCOUNTING:
        raise ValueError(
            f"Full TEST accounting mismatch: "
            f"{observed} != {EXPECTED_TEST_ACCOUNTING}"
        )

    retrieval_records = [
        record
        for record in records
        if record["category"] != "no_seed"
    ]

    reachable_records = [
        record
        for record in records
        if record["category"] == "reachable"
    ]

    metrics = {
        system: {
            "evaluation_scope": rank_metrics(
                records,
                rank_field,
            ),
            "retrieval_completed": rank_metrics(
                retrieval_records,
                rank_field,
            ),
            "reachable_positive": rank_metrics(
                reachable_records,
                rank_field,
            ),
        }
        for system, rank_field in (
            ("rrf", "rrf_rank"),
            ("model", "model_rank"),
        )
    }

    movements = [
        int(
            record[
                "rank_delta_model_minus_rrf"
            ]
        )
        for record in reachable_records
    ]

    movement = {
        "improved": sum(
            delta < 0
            for delta in movements
        ),
        "worsened": sum(
            delta > 0
            for delta in movements
        ),
        "unchanged": sum(
            delta == 0
            for delta in movements
        ),
        "mean_model_rank_minus_rrf_rank": (
            math.fsum(movements) / len(movements)
            if movements
            else 0.0
        ),
        "mean_rrf_rank_minus_model_rank": (
            -math.fsum(movements) / len(movements)
            if movements
            else 0.0
        ),
        "moved_to_rank_1": sum(
            record["model_rank"] == 1
            and record["rrf_rank"] > 1
            for record in reachable_records
        ),
        "moved_out_of_rank_1": sum(
            record["rrf_rank"] == 1
            and record["model_rank"] > 1
            for record in reachable_records
        ),
    }

    if (
        movement["improved"]
        + movement["worsened"]
        + movement["unchanged"]
        != len(reachable_records)
    ):
        raise RuntimeError(
            "TEST movement accounting is incomplete"
        )

    expected_reachability = (
        EXPECTED_TEST_ACCOUNTING["reachable"]
        / EXPECTED_TEST_ACCOUNTING["total_events"]
    )

    for system in ("rrf", "model"):
        metric = metrics[system]["evaluation_scope"]

        if metric["hits_at_50"] != EXPECTED_TEST_ACCOUNTING[
            "reachable"
        ]:
            raise RuntimeError(
                f"{system} TEST R@50 no longer equals "
                "candidate reachability"
            )

        if (
            abs(
                metric["recall_at_50"]
                - expected_reachability
            )
            > 1e-15
        ):
            raise RuntimeError(
                f"{system} full-denominator TEST R@50 "
                "invariant failed"
            )

    return {
        "accounting": observed,
        "metrics": metrics,
        "movement": movement,
    }


def validate_frozen_rrf_baseline(
    metric: Mapping[str, Any],
) -> None:

    for field in (
        "hits_at_1",
        "hits_at_10",
        "hits_at_50",
    ):
        if int(metric[field]) != int(EXPECTED_TEST_RRF[field]):
            raise RuntimeError(
                f"Frozen TEST RRF mismatch for {field}: "
                f"{metric[field]} != {EXPECTED_TEST_RRF[field]}"
            )

    for field in (
        "recall_at_1",
        "recall_at_10",
        "recall_at_50",
        "mrr",
        "reciprocal_rank_sum",
    ):
        observed = float(metric[field])
        expected = float(EXPECTED_TEST_RRF[field])

        if abs(observed - expected) > 1e-12:
            raise RuntimeError(
                f"Frozen TEST RRF mismatch for {field}: "
                f"{observed} != {expected}"
            )


def evaluation_configuration(
    *,
    checkpoint: Any,
) -> dict[str, Any]:

    return {
        "artifact_order_policy": ARTIFACT_ORDER_POLICY,
        "checkpoint": {
            "optimizer_step": checkpoint.optimizer_step,
            "scientific_fingerprint":
                checkpoint.scientific_fingerprint,
            "sha256": checkpoint.sha256,
        },
        "dataset_sha256": EXPECTED_TEST_DATASET_SHA256,
        "source_audit_sha256": EXPECTED_TEST_AUDIT_SHA256,
        "source_summary_sha256": EXPECTED_TEST_SUMMARY_SHA256,
        "source_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "denominator_policy": {
            "full_test":
                EXPECTED_TEST_ACCOUNTING["total_events"],
            "retrieval_completed_test":
                EXPECTED_TEST_ACCOUNTING[
                    "retrieval_completed"
                ],
            "reachable_positive_test":
                EXPECTED_TEST_ACCOUNTING["reachable"],
            "gold_absent_test":
                EXPECTED_TEST_ACCOUNTING["gold_absent"],
            "no_seed_test":
                EXPECTED_TEST_ACCOUNTING["no_seed"],
            "policy":
                "all_test_events_misses_for_absent_or_no_seed",
        },
        "evaluation_scope": {
            "status": "full_test",
            "partial_test_evaluation_allowed": False,
        },
        "evaluation_version": EVALUATION_VERSION,
        "metric_policy": {
            "mrr":
                "mean_reciprocal_best_observed_positive_rank_misses_zero",
            "recall":
                "count_best_observed_positive_rank_le_k_divided_by_full_test_denominator",
            "version": METRIC_POLICY_VERSION,
        },
        "model_selection": {
            "selection_split": "TRAIN-derived DEV",
            "selection_metric": "full_dev_mrr",
            "selected_beta": EXPECTED_SELECTED_BETA,
            "test_used_for_selection": False,
        },
        "model": {
            "model_id": MODEL_ID,
            "requested_revision": REQUESTED_MODEL_REVISION,
        },
        "ranking_policy": {
            "candidate_membership":
                "frozen_rrf_top50_unchanged",
            "combination": RRF_PRIOR_POLICY_VERSION,
            "positive_rank":
                "minimum_rank_over_observed_positive_serialization_positions",
            "tie_breaking": RANKING_POLICY_VERSION,
        },
        "upstream_fingerprints": {
            "phase2_architecture":
                EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
            "phase3a":
                EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT,
            "phase3b":
                EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT,
        },
    }


def atomic_json(
    path: Path,
    value: Any,
) -> None:

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
    )

    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
            )
            handle.write(b"\n")

        os.replace(
            temporary_name,
            path,
        )

    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def atomic_jsonl(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
    )

    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                handle.write(
                    canonical_json_bytes(record)
                    + b"\n"
                )

        os.replace(
            temporary_name,
            path,
        )

    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def validate_without_inference(
    *,
    inputs: FrozenTestInputs,
    checkpoint: Any,
) -> dict[str, Any]:

    # Tokenization validation is permitted:
    # no Qwen model is loaded and no inference occurs.
    tokenizer = load_production_tokenizer()
    resolved_tokenizer_commit = validate_tokenizer(
        tokenizer
    )

    maximum_tokens = 0
    minimum_tokens: int | None = None

    for index, record in enumerate(
        inputs.records,
        1,
    ):
        _event, _batch, actual_tokens = (
            tokenize_single_smoke_event(
                record,
                tokenizer,
            )
        )

        maximum_tokens = max(
            maximum_tokens,
            actual_tokens,
        )

        minimum_tokens = (
            actual_tokens
            if minimum_tokens is None
            else min(
                minimum_tokens,
                actual_tokens,
            )
        )

        if index % 500 == 0:
            print(
                f"[validate-only] tokenized "
                f"{index}/{len(inputs.records)}",
                flush=True,
            )

    return {
        "status": "validation_only_passed_no_model_inference",
        "test_accounting": inputs.accounting,
        "dataset_records": len(inputs.records),
        "no_seed_records": len(
            inputs.no_seed_instance_keys
        ),
        "tokenization": {
            "records": len(inputs.records),
            "minimum_tokens": minimum_tokens,
            "maximum_tokens": maximum_tokens,
            "resolved_tokenizer_commit":
                resolved_tokenizer_commit,
        },
        "checkpoint": {
            "optimizer_step": checkpoint.optimizer_step,
            "scientific_fingerprint":
                checkpoint.scientific_fingerprint,
            "sha256": checkpoint.sha256,
            "selected_beta": EXPECTED_SELECTED_BETA,
        },
    }


def evaluate_stage2_v2_test(
    *,
    dataset_path: str | Path,
    source_audit: str | Path,
    source_summary: str | Path,
    source_manifest: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device_name: str,
) -> dict[str, Any]:

    inputs = validate_frozen_test_inputs(
        dataset_path=dataset_path,
        audit_path=source_audit,
        summary_path=source_summary,
        manifest_path=source_manifest,
    )

    checkpoint = validate_selected_checkpoint(
        checkpoint_path
    )

    destination = Path(output_dir).resolve()

    paths = {
        "manifest":
            destination / MANIFEST_FILENAME,
        "summary":
            destination / SUMMARY_FILENAME,
        "instances":
            destination / INSTANCES_FILENAME,
    }

    collisions = [
        path
        for path in paths.values()
        if path.exists()
    ]

    if collisions:
        raise FileExistsError(
            "Final TEST evaluation output already exists: "
            + ", ".join(map(str, collisions))
        )

    configuration = evaluation_configuration(
        checkpoint=checkpoint,
    )

    evaluation_fingerprint = fingerprint(
        configuration
    )

    device = require_single_cuda_device(
        device_name
    )

    (
        tokenizer,
        ranker,
        peft_module,
        model_identity,
        resolved_tokenizer_commit,
    ) = load_inference_stack(
        checkpoint,
        device,
    )

    records: list[dict[str, Any]] = []

    total = len(inputs.records)

    for index, record in enumerate(
        inputs.records,
        1,
    ):
        records.append(
            evaluate_model_record(
                record,
                tokenizer=tokenizer,
                ranker=ranker,
                device=device,
            )
        )

        if index % 100 == 0 or index == total:
            print(
                f"[TEST] evaluated "
                f"{index}/{total} retrieval-completed events",
                flush=True,
            )

    records.extend(
        compact_instance_record(
            instance_key=instance_key,
            category="no_seed",
            rrf_rank=None,
            model_rank=None,
            rank_delta=None,
        )
        for instance_key
        in inputs.no_seed_instance_keys
    )

    summary_payload = summarize_records(
        records
    )

    validate_frozen_rrf_baseline(
        summary_payload[
            "metrics"
        ][
            "rrf"
        ][
            "evaluation_scope"
        ]
    )

    runtime = runtime_provenance(
        device=device,
        peft_module=peft_module,
        tokenizer=tokenizer,
        resolved_tokenizer_commit=
            resolved_tokenizer_commit,
    )

    summary = {
        **summary_payload,
        "comparable_full_test": True,
        "evaluation_configuration":
            configuration,
        "evaluation_fingerprint":
            evaluation_fingerprint,
        "schema_version":
            EVALUATION_SUMMARY_SCHEMA,
        "status": "full_test",
    }

    manifest = {
        "artifacts": {
            name: path.name
            for name, path in paths.items()
        },
        "checkpoint": {
            "optimizer_step":
                checkpoint.optimizer_step,
            "path":
                str(checkpoint.path),
            "scientific_fingerprint":
                checkpoint.scientific_fingerprint,
            "sha256":
                checkpoint.sha256,
        },
        "dataset": {
            "path":
                str(inputs.dataset_path),
            "records":
                len(inputs.records),
            "sha256":
                EXPECTED_TEST_DATASET_SHA256,
        },
        "evaluation_configuration":
            configuration,
        "evaluation_fingerprint":
            evaluation_fingerprint,
        "model_identity":
            model_identity,
        "runtime_provenance":
            runtime,
        "schema_version":
            EVALUATION_MANIFEST_SCHEMA,
        "source_audit": {
            "path":
                str(inputs.audit_path),
            "sha256":
                EXPECTED_TEST_AUDIT_SHA256,
            "no_seed_test":
                len(
                    inputs.no_seed_instance_keys
                ),
        },
        "source_summary": {
            "path":
                str(inputs.summary_path),
            "sha256":
                EXPECTED_TEST_SUMMARY_SHA256,
        },
        "source_manifest": {
            "path":
                str(inputs.manifest_path),
            "sha256":
                EXPECTED_TEST_MANIFEST_SHA256,
        },
    }

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    atomic_jsonl(
        paths["instances"],
        records,
    )

    atomic_json(
        paths["summary"],
        summary,
    )

    atomic_json(
        paths["manifest"],
        manifest,
    )

    return {
        "evaluation_fingerprint":
            evaluation_fingerprint,
        "instances":
            len(records),
        "output_dir":
            str(destination),
        "status":
            "full_test",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )

    parser.add_argument(
        "--source-audit",
        type=Path,
        default=DEFAULT_SOURCE_AUDIT,
    )

    parser.add_argument(
        "--source-summary",
        type=Path,
        default=DEFAULT_SOURCE_SUMMARY,
    )

    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate frozen TEST artifacts, selected checkpoint, "
            "and all TEST tokenization without loading Qwen "
            "or performing model inference."
        ),
    )

    return parser


def main(
    argv: Sequence[str] | None = None,
) -> int:

    args = _parser().parse_args(argv)

    inputs = validate_frozen_test_inputs(
        dataset_path=args.dataset_path,
        audit_path=args.source_audit,
        summary_path=args.source_summary,
        manifest_path=args.source_manifest,
    )

    checkpoint = validate_selected_checkpoint(
        args.checkpoint
    )

    if args.validate_only:
        result = validate_without_inference(
            inputs=inputs,
            checkpoint=checkpoint,
        )

    else:
        result = evaluate_stage2_v2_test(
            dataset_path=args.dataset_path,
            source_audit=args.source_audit,
            source_summary=args.source_summary,
            source_manifest=args.source_manifest,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            device_name=args.device,
        )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
