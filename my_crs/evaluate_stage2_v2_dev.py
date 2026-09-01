"""Evaluate one frozen Stage-2-v2 checkpoint on TRAIN-derived DEV only."""

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
from typing import Any, BinaryIO

import torch

from my_crs.analyze_stage2_v2_tokens import (
    DEFAULT_DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
)
from my_crs.build_stage2_v2_dataset import (
    DEFAULT_SOURCE_AUDIT,
    EXPECTED_ACCOUNTING,
    EXPECTED_SOURCE_AUDIT_SHA256,
    NO_SEED_EXCLUSION,
    PROJECT_ROOT,
    TOP_K,
    canonical_json_bytes,
    scan_source_audit,
    transform_audit_record,
)
from my_crs.joint_rrf_ranker import (
    RANKING_POLICY_VERSION,
    RRF_PRIOR_POLICY_VERSION,
    canonicalize_phase1_candidates,
    combine_rrf_prior,
    load_scorer_head_state_dict,
    rank_candidate_ids,
)
from my_crs.stage2_v2_loss import (
    EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT,
    loss_scientific_configuration,
    loss_scientific_fingerprint,
)
from my_crs.stage2_v2_peft import (
    EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT,
    MAX_PACKED_TOKENS,
    MODEL_ID,
    REQUESTED_MODEL_REVISION,
    TRUNCATION_ALLOWED,
    apply_peft_and_build_ranker,
    load_base_qwen,
    load_production_tokenizer,
    require_single_cuda_device,
    runtime_provenance,
    tokenize_single_smoke_event,
    validate_tokenizer,
)
from my_crs.train_stage2_v2 import (
    CHECKPOINT_SCHEMA,
    EXPECTED_DATASET_COUNTS,
    TrainingState,
)


EVALUATION_VERSION = "stage2_v2_train_derived_dev_evaluator_v1"
EVALUATION_MANIFEST_SCHEMA = "stage2_v2_dev_eval_manifest_v1"
EVALUATION_SUMMARY_SCHEMA = "stage2_v2_dev_eval_summary_v1"
EVALUATION_INSTANCE_SCHEMA = "stage2_v2_dev_eval_instance_v1"
METRIC_POLICY_VERSION = "full_denominator_best_observed_positive_rank_v1"
ARTIFACT_ORDER_POLICY = "retrieval_dataset_order_then_no_seed_audit_order_v1"

REFERENCE_BETA_0_10_LOSS_FINGERPRINT = (
    "2de3b93ec1ee122d86d75d29c06dee12ceeba6ad971c5bad35461a020fce3ee8"
)
EXPECTED_DEV_ACCOUNTING = dict(EXPECTED_ACCOUNTING["dev"])
EXPECTED_HISTORICAL_RRF = {
    "recall_at_1": 0.05297438,
    "recall_at_10": 0.26834564,
    "recall_at_50": 0.59227095,
    "mrr": 0.12208811,
}
HISTORICAL_SANITY_TOLERANCE = 1e-6

MANIFEST_FILENAME = "stage2_v2_dev_eval_manifest.json"
SUMMARY_FILENAME = "stage2_v2_dev_eval_summary.json"
INSTANCES_FILENAME = "stage2_v2_dev_eval_instances.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "stage2_v2_dev_eval"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _validated_positions(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    positions: list[int] = []
    for item in value:
        if type(item) is not int or not 1 <= item <= TOP_K:
            raise ValueError(f"{field} must contain integers in 1..50")
        positions.append(item)
    if len(positions) != len(set(positions)):
        raise ValueError(f"{field} contains duplicate positions")
    return tuple(positions)


@dataclass(frozen=True)
class DevOffsetEntry:
    byte_offset: int
    byte_length: int
    instance_key: str


@dataclass(frozen=True)
class DevJsonlOffsetIndex:
    path: Path
    dataset_sha256: str
    counts: dict[str, int]
    entries: tuple[DevOffsetEntry, ...]

    def read_record(
        self,
        index: int,
        *,
        handle: BinaryIO | None = None,
    ) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise IndexError("DEV offset index is outside the dataset")
        entry = self.entries[index]
        owns_handle = handle is None
        stream = handle or self.path.open("rb")
        try:
            stream.seek(entry.byte_offset)
            line = stream.read(entry.byte_length)
        finally:
            if owns_handle:
                stream.close()
        if len(line) != entry.byte_length:
            raise RuntimeError("Indexed DEV record is truncated")
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Indexed DEV record is not valid UTF-8 JSON") from error
        if not isinstance(record, dict):
            raise RuntimeError("Indexed DEV record is not an object")
        if record.get("instance_key") != entry.instance_key:
            raise RuntimeError("Indexed DEV instance identity changed")
        if record.get("split") != "dev":
            raise RuntimeError("Indexed evaluation record no longer has split=dev")
        return record


def build_dev_offset_index(
    dataset_path: str | Path,
    *,
    expected_sha256: str = EXPECTED_DATASET_SHA256,
    expected_counts: Mapping[str, int] = EXPECTED_DATASET_COUNTS,
) -> DevJsonlOffsetIndex:
    """Hash, validate, and retain only DEV offsets in one binary pass."""

    source = Path(dataset_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    counts = {"all": 0, "train": 0, "dev": 0}
    entries: list[DevOffsetEntry] = []
    instance_keys: set[str] = set()
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                raise ValueError(f"Blank dataset line at byte offset {offset}")
            digest.update(line)
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid dataset JSON at byte offset {offset}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Dataset record at byte offset {offset} is not an object")
            split = record.get("split")
            if split not in {"train", "dev"}:
                raise ValueError("Stage-2 v2 evaluation dataset may contain train/dev only")
            instance_key = record.get("instance_key")
            if not isinstance(instance_key, str) or not instance_key:
                raise ValueError("Stage-2 v2 dataset record lacks an instance key")
            if instance_key in instance_keys:
                raise ValueError(f"Duplicate Stage-2 v2 instance key: {instance_key}")
            instance_keys.add(instance_key)
            candidates = record.get("candidates")
            if (
                record.get("candidate_count") != TOP_K
                or not isinstance(candidates, list)
                or len(candidates) != TOP_K
            ):
                raise ValueError(f"Dataset instance {instance_key} must contain 50 candidates")
            counts["all"] += 1
            counts[str(split)] += 1
            if split == "dev":
                entries.append(DevOffsetEntry(offset, len(line), instance_key))

    observed_sha = digest.hexdigest()
    if observed_sha.lower() != str(expected_sha256).lower():
        raise ValueError(
            "Frozen Stage-2 v2 dataset SHA256 mismatch: "
            f"expected={str(expected_sha256).lower()} observed={observed_sha.lower()}"
        )
    required_counts = {
        scope: _strict_nonnegative_int(expected_counts[scope], f"expected_counts.{scope}")
        for scope in ("all", "train", "dev")
    }
    if counts != required_counts:
        raise ValueError(f"Stage-2 v2 dataset accounting mismatch: {counts} != {required_counts}")
    if len(entries) != counts["dev"]:
        raise RuntimeError("DEV offset count disagrees with dataset accounting")
    return DevJsonlOffsetIndex(source, observed_sha, counts, tuple(entries))


@dataclass(frozen=True)
class AuthoritativeDevProvenance:
    source_audit: Path
    source_audit_sha256: str
    accounting: dict[str, int]
    no_seed_instance_keys: tuple[str, ...]


def load_authoritative_dev_provenance(
    source_audit: str | Path,
    *,
    expected_source_sha256: str = EXPECTED_SOURCE_AUDIT_SHA256,
    expected_accounting: Mapping[str, Any] = EXPECTED_ACCOUNTING,
) -> AuthoritativeDevProvenance:
    """Validate the authoritative audit and recover exact DEV no-seed identities."""

    source = Path(source_audit).resolve()
    statistics = scan_source_audit(
        source,
        expected_source_sha256=expected_source_sha256,
        expected_accounting=expected_accounting,
    )
    dev_accounting = {
        field: int(statistics["accounting"]["dev"][field])
        for field in (
            "total_events",
            "retrieval_completed",
            "reachable",
            "gold_absent",
            "no_seed",
        )
    }
    no_seed_keys: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank source-audit line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid source-audit JSON at line {line_number}") from error
            transformed = transform_audit_record(record, line_number)
            if record.get("split") != "dev" or transformed is not None:
                continue
            if record.get("exclusion_reason") != NO_SEED_EXCLUSION:
                raise ValueError("DEV audit exclusion without candidates is not no-seed")
            no_seed_keys.append(str(record["instance_key"]))
    if len(no_seed_keys) != dev_accounting["no_seed"]:
        raise ValueError("Authoritative DEV no-seed identity count mismatch")
    if len(no_seed_keys) != len(set(no_seed_keys)):
        raise ValueError("Authoritative DEV no-seed instance keys are not unique")
    return AuthoritativeDevProvenance(
        source_audit=source,
        source_audit_sha256=str(expected_source_sha256).lower(),
        accounting=dev_accounting,
        no_seed_instance_keys=tuple(no_seed_keys),
    )


@dataclass(frozen=True)
class EvaluationCheckpoint:
    path: Path
    sha256: str
    scientific_configuration: dict[str, Any]
    scientific_fingerprint: str
    optimizer_step: int
    adapter_state: Mapping[str, torch.Tensor]
    scorer_head_state: Mapping[str, torch.Tensor]


def _validate_training_scientific_configuration(configuration: Any) -> dict[str, Any]:
    if not isinstance(configuration, Mapping):
        raise ValueError("Checkpoint scientific configuration must be an object")
    config = dict(configuration)
    if config.get("dataset") != {
        "counts": dict(EXPECTED_DATASET_COUNTS),
        "sha256": EXPECTED_DATASET_SHA256,
        "training_split": "train",
    }:
        raise ValueError("Checkpoint frozen dataset configuration mismatch")
    if config.get("model_id") != MODEL_ID:
        raise ValueError("Checkpoint Qwen model ID mismatch")
    if config.get("requested_model_revision") != REQUESTED_MODEL_REVISION:
        raise ValueError("Checkpoint Qwen revision mismatch")
    if config.get("max_packed_tokens") != MAX_PACKED_TOKENS:
        raise ValueError("Checkpoint packed-token ceiling mismatch")
    if config.get("truncation") is not TRUNCATION_ALLOWED:
        raise ValueError("Checkpoint truncation policy mismatch")
    if config.get("phase3b_integration_fingerprint") != EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT:
        raise ValueError("Checkpoint Phase-3B fingerprint mismatch")
    loss = config.get("loss")
    if not isinstance(loss, Mapping):
        raise ValueError("Checkpoint loss configuration is missing")
    beta = loss.get("beta")
    expected_loss = loss_scientific_configuration(beta)
    if dict(loss) != expected_loss:
        raise ValueError("Checkpoint loss scientific configuration mismatch")
    expected_loss_fingerprint = loss_scientific_fingerprint(beta)
    if config.get("loss_fingerprint") != expected_loss_fingerprint:
        raise ValueError("Checkpoint loss fingerprint mismatch")
    if loss.get("phase2_architecture_fingerprint") != EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT:
        raise ValueError("Checkpoint Phase-2 fingerprint mismatch")
    if loss.get("phase3a_analysis_fingerprint") != EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT:
        raise ValueError("Checkpoint Phase-3A fingerprint mismatch")
    if loss.get("phase3b_integration_fingerprint") != EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT:
        raise ValueError("Checkpoint loss Phase-3B fingerprint mismatch")
    return config


def load_evaluation_checkpoint(
    checkpoint_path: str | Path,
) -> EvaluationCheckpoint:
    """Load and validate checkpoint science without restoring training state."""

    source = Path(checkpoint_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Stage-2 v2 checkpoint schema mismatch")
    expected_contents = [
        "lora_adapter",
        "shared_scorer_head",
        "optimizer",
        "rng",
        "training_state",
        "scientific_configuration",
    ]
    if payload.get("checkpoint_contents") != expected_contents:
        raise ValueError("Stage-2 v2 checkpoint content policy mismatch")
    required = {
        "adapter_state",
        "optimizer_state",
        "rng_state",
        "scientific_configuration",
        "scientific_fingerprint",
        "scorer_head_state",
        "training_state",
    }
    if not required.issubset(payload):
        raise ValueError("Stage-2 v2 checkpoint is incomplete")
    configuration = _validate_training_scientific_configuration(
        payload.get("scientific_configuration")
    )
    fingerprint = payload.get("scientific_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != _fingerprint(configuration):
        raise ValueError("Stage-2 v2 checkpoint scientific fingerprint mismatch")
    state = TrainingState.from_mapping(payload.get("training_state"))
    adapter_state = payload.get("adapter_state")
    scorer_head_state = payload.get("scorer_head_state")
    if not isinstance(adapter_state, Mapping) or not isinstance(scorer_head_state, Mapping):
        raise ValueError("Stage-2 v2 inference checkpoint weights are incomplete")
    for name, state_dict in (("adapter", adapter_state), ("scorer head", scorer_head_state)):
        if not state_dict or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in state_dict.items()
        ):
            raise ValueError(f"Stage-2 v2 {name} state is malformed")
    return EvaluationCheckpoint(
        path=source,
        sha256=_sha256_file(source),
        scientific_configuration=configuration,
        scientific_fingerprint=fingerprint,
        optimizer_step=state.optimizer_step,
        adapter_state=adapter_state,
        scorer_head_state=scorer_head_state,
    )


def apply_inference_checkpoint_weights(
    ranker: Any,
    checkpoint: EvaluationCheckpoint,
    *,
    peft_module: Any,
) -> None:
    """Apply only LoRA and scorer-head weights; never restore trainer state."""

    peft_module.set_peft_model_state_dict(ranker.base_model, checkpoint.adapter_state)
    load_scorer_head_state_dict(ranker, checkpoint.scorer_head_state)
    ranker.eval()


def load_inference_stack(
    checkpoint: EvaluationCheckpoint,
    device: torch.device,
) -> tuple[Any, Any, Any, dict[str, Any], str | None]:
    try:
        import peft
    except ImportError as error:
        raise RuntimeError("Stage-2 v2 DEV evaluation requires PEFT") from error
    tokenizer = load_production_tokenizer()
    resolved_tokenizer_commit = validate_tokenizer(tokenizer)
    base_model, model_identity = load_base_qwen(device)
    ranker, _trainability = apply_peft_and_build_ranker(
        base_model,
        device,
        peft_module=peft,
    )
    apply_inference_checkpoint_weights(ranker, checkpoint, peft_module=peft)
    return tokenizer, ranker, peft, model_identity, resolved_tokenizer_commit


def rank_record_from_residuals(
    record: Mapping[str, Any],
    residuals: torch.Tensor | Sequence[float],
) -> dict[str, Any]:
    """Rank one canonical DEV record from raw residuals without text relabeling."""

    if record.get("split") != "dev":
        raise ValueError("Stage-2 v2 evaluation accepts DEV records only")
    instance_key = record.get("instance_key")
    if not isinstance(instance_key, str) or not instance_key:
        raise ValueError("Stage-2 v2 DEV record lacks an instance key")
    candidates = canonicalize_phase1_candidates(record)
    positive_positions = _validated_positions(
        record.get("observed_positive_serialization_positions"),
        "observed_positive_serialization_positions",
    )
    canonical_ids = [int(candidate["canonical_entity_id"]) for candidate in candidates]
    rrf_ranks = [int(candidate["rrf_rank"]) for candidate in candidates]
    rrf_scores = [float(candidate["rrf_score"]) for candidate in candidates]
    raw = residuals if isinstance(residuals, torch.Tensor) else torch.tensor(residuals)
    if raw.shape != (TOP_K,):
        raise ValueError("One evaluation event requires exactly 50 residuals")
    prior = torch.tensor(rrf_scores, dtype=torch.float64, device=raw.device)
    combination = combine_rrf_prior(prior, raw)
    ranked_ids = rank_candidate_ids(combination.final_scores, canonical_ids, rrf_ranks)
    if set(ranked_ids) != set(canonical_ids):
        raise RuntimeError("Stage-2 v2 evaluation changed candidate membership")

    if positive_positions:
        positive_ids = {canonical_ids[position - 1] for position in positive_positions}
        model_by_id = {entity_id: rank for rank, entity_id in enumerate(ranked_ids, 1)}
        model_rank = min(model_by_id[entity_id] for entity_id in positive_ids)
        rrf_rank = min(rrf_ranks[position - 1] for position in positive_positions)
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
    if category not in {"reachable", "gold_absent", "no_seed"}:
        raise ValueError("Unknown Stage-2 v2 evaluation category")
    if category == "reachable":
        for name, rank in (("rrf_rank", rrf_rank), ("model_rank", model_rank)):
            if type(rank) is not int or not 1 <= rank <= TOP_K:
                raise ValueError(f"Reachable {name} must be in 1..50")
        if rank_delta != model_rank - rrf_rank:
            raise ValueError("Reachable rank delta must equal model_rank - rrf_rank")
    elif any(value is not None for value in (rrf_rank, model_rank, rank_delta)):
        raise ValueError("Miss categories cannot contain a rank")
    return {
        "category": category,
        "hit_at_1": bool(model_rank is not None and model_rank <= 1),
        "hit_at_10": bool(model_rank is not None and model_rank <= 10),
        "hit_at_50": bool(model_rank is not None and model_rank <= 50),
        "instance_key": instance_key,
        "model_rank": model_rank,
        "rank_delta_model_minus_rrf": rank_delta,
        "rrf_rank": rrf_rank,
        "schema_version": EVALUATION_INSTANCE_SCHEMA,
    }


def evaluate_model_record(
    record: Mapping[str, Any],
    *,
    tokenizer: Any,
    ranker: Any,
    device: torch.device,
) -> dict[str, Any]:
    candidates = canonicalize_phase1_candidates(record)
    event, batch, _actual_tokens = tokenize_single_smoke_event(record, tokenizer)
    expected_ids = tuple(int(candidate["canonical_entity_id"]) for candidate in candidates)
    expected_ranks = tuple(int(candidate["rrf_rank"]) for candidate in candidates)
    expected_scores = tuple(float(candidate["rrf_score"]) for candidate in candidates)
    if (
        event.canonical_entity_ids != expected_ids
        or event.rrf_ranks != expected_ranks
        or event.rrf_scores != expected_scores
    ):
        raise RuntimeError("Tokenized candidate order disagrees with frozen logical order")
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        residuals = ranker(batch.to(device))
    if not isinstance(residuals, torch.Tensor) or residuals.shape != (1, TOP_K):
        raise RuntimeError("JointRRFRanker must return one residual for each of 50 candidates")
    return rank_record_from_residuals(record, residuals[0])


def _rank_metrics(records: Sequence[Mapping[str, Any]], rank_field: str) -> dict[str, Any]:
    denominator = len(records)
    ranks: list[int] = []
    for record in records:
        rank = record.get(rank_field)
        if rank is None:
            continue
        if type(rank) is not int or not 1 <= rank <= TOP_K:
            raise ValueError(f"{rank_field} must be None or an integer in 1..50")
        ranks.append(rank)
    hits = {cutoff: sum(rank <= cutoff for rank in ranks) for cutoff in (1, 10, 50)}
    reciprocal_rank_sum = math.fsum(1.0 / rank for rank in ranks)
    result: dict[str, Any] = {
        "denominator": denominator,
        "evaluated_rank_count": len(ranks),
        "reciprocal_rank_sum": reciprocal_rank_sum,
        "mrr": reciprocal_rank_sum / denominator if denominator else None,
        "mrr_percent": 100.0 * reciprocal_rank_sum / denominator if denominator else None,
    }
    for cutoff in (1, 10, 50):
        result[f"hits_at_{cutoff}"] = hits[cutoff]
        result[f"recall_at_{cutoff}"] = hits[cutoff] / denominator if denominator else None
        result[f"recall_at_{cutoff}_percent"] = (
            100.0 * hits[cutoff] / denominator if denominator else None
        )
    return result


def summarize_rank_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_accounting: Mapping[str, int] | None,
    full_evaluation: bool,
) -> dict[str, Any]:
    if not records:
        raise ValueError("No Stage-2 v2 DEV evaluation records were produced")
    categories = Counter(str(record.get("category")) for record in records)
    observed = {
        "total_events": len(records),
        "retrieval_completed": categories["reachable"] + categories["gold_absent"],
        "reachable": categories["reachable"],
        "gold_absent": categories["gold_absent"],
        "no_seed": categories["no_seed"],
    }
    if full_evaluation:
        required = {field: int(expected_accounting[field]) for field in observed}
        if observed != required:
            raise ValueError(f"Full DEV accounting mismatch: {observed} != {required}")
    elif categories["no_seed"]:
        raise ValueError("Partial retrieval smoke must not masquerade as full no-seed coverage")

    retrieval_records = [record for record in records if record.get("category") != "no_seed"]
    reachable_records = [record for record in records if record.get("category") == "reachable"]
    metrics = {
        system: {
            "evaluation_scope": _rank_metrics(records, rank_field),
            "retrieval_completed": _rank_metrics(retrieval_records, rank_field),
            "reachable_positive": _rank_metrics(reachable_records, rank_field),
        }
        for system, rank_field in (("rrf", "rrf_rank"), ("model", "model_rank"))
    }

    movements = [
        int(record["rank_delta_model_minus_rrf"])
        for record in reachable_records
    ]
    movement = {
        "improved": sum(delta < 0 for delta in movements),
        "worsened": sum(delta > 0 for delta in movements),
        "unchanged": sum(delta == 0 for delta in movements),
        "mean_model_rank_minus_rrf_rank": (
            math.fsum(movements) / len(movements) if movements else 0.0
        ),
        "mean_rrf_rank_minus_model_rank": (
            -math.fsum(movements) / len(movements) if movements else 0.0
        ),
        "moved_to_rank_1": sum(
            record["model_rank"] == 1 and record["rrf_rank"] > 1
            for record in reachable_records
        ),
        "moved_out_of_rank_1": sum(
            record["rrf_rank"] == 1 and record["model_rank"] > 1
            for record in reachable_records
        ),
    }
    if sum(movement[field] for field in ("improved", "worsened", "unchanged")) != len(
        reachable_records
    ):
        raise RuntimeError("Reachable movement accounting is incomplete")

    if full_evaluation:
        expected_reachability = observed["reachable"] / observed["total_events"]
        for system in ("rrf", "model"):
            metric = metrics[system]["evaluation_scope"]
            if metric["hits_at_50"] != observed["reachable"]:
                raise RuntimeError(f"{system} R@50 no longer equals candidate reachability")
            if metric["recall_at_50"] != expected_reachability:
                raise RuntimeError(f"{system} full-denominator R@50 invariant failed")
    return {
        "accounting": observed,
        "metrics": metrics,
        "movement": movement,
    }


def validate_historical_rrf_baseline(metrics: Mapping[str, Any]) -> None:
    for field, expected in EXPECTED_HISTORICAL_RRF.items():
        observed = float(metrics[field])
        if abs(observed - expected) > HISTORICAL_SANITY_TOLERANCE:
            raise RuntimeError(
                f"Frozen full-DEV RRF baseline sanity check failed for {field}: "
                f"expected~={expected} observed={observed}"
            )


def evaluation_configuration(
    *,
    checkpoint: EvaluationCheckpoint,
    source_audit_sha256: str,
    full_evaluation: bool,
    max_retrieval_completed_events: int | None,
) -> dict[str, Any]:
    loss_fingerprint = checkpoint.scientific_configuration["loss_fingerprint"]
    return {
        "artifact_order_policy": ARTIFACT_ORDER_POLICY,
        "checkpoint": {
            "optimizer_step": checkpoint.optimizer_step,
            "scientific_fingerprint": checkpoint.scientific_fingerprint,
            "sha256": checkpoint.sha256,
        },
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "denominator_policy": {
            "full_dev": EXPECTED_DEV_ACCOUNTING["total_events"],
            "gold_absent_dev": EXPECTED_DEV_ACCOUNTING["gold_absent"],
            "no_seed_dev": EXPECTED_DEV_ACCOUNTING["no_seed"],
            "no_seed_source_audit_sha256": source_audit_sha256,
            "policy": "all_train_derived_dev_events_misses_for_absent_or_no_seed",
            "reachable_positive_dev": EXPECTED_DEV_ACCOUNTING["reachable"],
            "retrieval_completed_dev": EXPECTED_DEV_ACCOUNTING["retrieval_completed"],
        },
        "evaluation_scope": {
            "comparable_full_dev": full_evaluation,
            "max_retrieval_completed_events": max_retrieval_completed_events,
            "status": "full_dev" if full_evaluation else "smoke_partial_not_comparable",
        },
        "evaluation_version": EVALUATION_VERSION,
        "metric_policy": {
            "mrr": "mean_reciprocal_best_observed_positive_rank_misses_zero",
            "primary_model_selection_metric_later": "full_dev_mrr",
            "recall": "count_best_observed_positive_rank_le_k_divided_by_scope_denominator",
            "version": METRIC_POLICY_VERSION,
        },
        "model": {
            "model_id": MODEL_ID,
            "requested_revision": REQUESTED_MODEL_REVISION,
        },
        "ranking_policy": {
            "candidate_membership": "frozen_rrf_top50_unchanged",
            "combination": RRF_PRIOR_POLICY_VERSION,
            "positive_rank": "minimum_rank_over_observed_positive_serialization_positions",
            "tie_breaking": RANKING_POLICY_VERSION,
        },
        "upstream_fingerprints": {
            "loss": loss_fingerprint,
            "phase2_architecture": EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
            "phase3a": EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT,
            "phase3b": EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT,
            "reference_beta_0_10_loss": REFERENCE_BETA_0_10_LOSS_FINGERPRINT,
        },
    }


_FORBIDDEN_ARTIFACT_KEYS = {
    "candidate_titles",
    "candidates",
    "conversation",
    "dialogue",
    "ground_truth",
    "ground_truth_titles",
    "hidden_states",
    "history",
    "prompt",
    "prompts",
    "target_response",
    "title",
    "titles",
}
_FORBIDDEN_ARTIFACT_KEY_FRAGMENTS = (
    "candidate_text",
    "conversation_text",
    "dialogue_text",
    "ground_truth",
    "hidden_state",
    "history_text",
    "prompt_text",
    "target_response",
    "title_text",
)


def validate_artifact_privacy(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_ARTIFACT_KEYS or any(
                fragment in normalized for fragment in _FORBIDDEN_ARTIFACT_KEY_FRAGMENTS
            ):
                raise ValueError(f"Sensitive field is forbidden in DEV artifact: {key}")
            validate_artifact_privacy(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_artifact_privacy(child)


def _atomic_json(path: Path, value: Any) -> None:
    validate_artifact_privacy(value)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                validate_artifact_privacy(record)
                handle.write(canonical_json_bytes(record) + b"\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def evaluate_stage2_v2_dev(
    *,
    dataset_path: str | Path,
    source_audit: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device_name: str,
    max_retrieval_completed_events: int | None = None,
) -> dict[str, Any]:
    if max_retrieval_completed_events is not None:
        if type(max_retrieval_completed_events) is not int or not (
            1 <= max_retrieval_completed_events < EXPECTED_DEV_ACCOUNTING["retrieval_completed"]
        ):
            raise ValueError("Smoke maximum must be in 1..2143; omit it for full DEV")
    full_evaluation = max_retrieval_completed_events is None
    destination = Path(output_dir).resolve()
    paths = {
        "manifest": destination / MANIFEST_FILENAME,
        "summary": destination / SUMMARY_FILENAME,
        "instances": destination / INSTANCES_FILENAME,
    }
    collisions = [path for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"DEV evaluation output already exists: {collisions}")

    checkpoint = load_evaluation_checkpoint(checkpoint_path)
    dev_index = build_dev_offset_index(dataset_path)
    audit = load_authoritative_dev_provenance(source_audit)
    if audit.accounting != EXPECTED_DEV_ACCOUNTING:
        raise ValueError("Frozen authoritative DEV accounting mismatch")
    if dev_index.counts["dev"] != audit.accounting["retrieval_completed"]:
        raise ValueError("Candidate dataset and authoritative DEV audit disagree")

    configuration = evaluation_configuration(
        checkpoint=checkpoint,
        source_audit_sha256=audit.source_audit_sha256,
        full_evaluation=full_evaluation,
        max_retrieval_completed_events=max_retrieval_completed_events,
    )
    evaluation_fingerprint = _fingerprint(configuration)
    device = require_single_cuda_device(device_name)
    tokenizer, ranker, peft_module, model_identity, resolved_tokenizer_commit = (
        load_inference_stack(checkpoint, device)
    )

    limit = (
        len(dev_index.entries)
        if max_retrieval_completed_events is None
        else max_retrieval_completed_events
    )
    records: list[dict[str, Any]] = []
    with dev_index.path.open("rb") as handle:
        for index in range(limit):
            record = dev_index.read_record(index, handle=handle)
            records.append(
                evaluate_model_record(
                    record,
                    tokenizer=tokenizer,
                    ranker=ranker,
                    device=device,
                )
            )
    if full_evaluation:
        records.extend(
            compact_instance_record(
                instance_key=instance_key,
                category="no_seed",
                rrf_rank=None,
                model_rank=None,
                rank_delta=None,
            )
            for instance_key in audit.no_seed_instance_keys
        )
    summary_payload = summarize_rank_records(
        records,
        expected_accounting=audit.accounting if full_evaluation else None,
        full_evaluation=full_evaluation,
    )
    if full_evaluation:
        validate_historical_rrf_baseline(
            summary_payload["metrics"]["rrf"]["evaluation_scope"]
        )

    runtime = runtime_provenance(
        device=device,
        peft_module=peft_module,
        tokenizer=tokenizer,
        resolved_tokenizer_commit=resolved_tokenizer_commit,
    )
    summary = {
        **summary_payload,
        "comparable_full_dev": full_evaluation,
        "evaluation_configuration": configuration,
        "evaluation_fingerprint": evaluation_fingerprint,
        "schema_version": EVALUATION_SUMMARY_SCHEMA,
        "status": "full_dev" if full_evaluation else "smoke_partial_not_comparable",
    }
    manifest = {
        "artifacts": {name: path.name for name, path in paths.items()},
        "checkpoint": {
            "optimizer_step": checkpoint.optimizer_step,
            "path": str(checkpoint.path),
            "scientific_fingerprint": checkpoint.scientific_fingerprint,
            "sha256": checkpoint.sha256,
        },
        "dataset": {
            "path": str(dev_index.path),
            "sha256": dev_index.dataset_sha256,
            "retrieval_completed_dev": dev_index.counts["dev"],
        },
        "evaluation_configuration": configuration,
        "evaluation_fingerprint": evaluation_fingerprint,
        "model_identity": model_identity,
        "runtime_provenance": runtime,
        "schema_version": EVALUATION_MANIFEST_SCHEMA,
        "source_audit": {
            "no_seed_dev": audit.accounting["no_seed"],
            "path": str(audit.source_audit),
            "sha256": audit.source_audit_sha256,
        },
    }
    validate_artifact_privacy(summary)
    validate_artifact_privacy(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_jsonl(paths["instances"], records)
    _atomic_json(paths["summary"], summary)
    _atomic_json(paths["manifest"], manifest)
    return {
        "evaluation_fingerprint": evaluation_fingerprint,
        "instances": len(records),
        "output_dir": str(destination),
        "status": summary["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--source-audit", type=Path, default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-retrieval-completed-events", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_stage2_v2_dev(
        dataset_path=args.dataset_path,
        source_audit=args.source_audit,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device_name=args.device,
        max_retrieval_completed_events=args.max_retrieval_completed_events,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
