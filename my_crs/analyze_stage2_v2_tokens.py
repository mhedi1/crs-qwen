"""Measure exact frozen Stage-2 v2 packed-input lengths without training.

The analyzer streams the frozen Phase-1 JSONL and delegates every prompt,
candidate, boundary, and token construction decision to
``joint_rrf_ranker.tokenize_scoring_event``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import transformers

from my_crs.build_stage2_v2_dataset import (
    PROJECT_ROOT,
    TOP_K,
    canonical_json_bytes,
    sha256_file,
)
from my_crs.joint_rrf_ranker import (
    INPUT_SERIALIZATION_VERSION,
    TOKEN_BOUNDARY_POLICY_VERSION,
    phase2_architecture_fingerprint,
    tokenize_scoring_event,
)


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
REQUESTED_MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT = (
    "2d833955d53e2903a5cc48f1c6a71cb550283d99b9c0edb6f94520254120b3db"
)
EXPECTED_DATASET_SHA256 = (
    "0859cb796082cb31772e85761efe1f54d707ab80f2debff6f15ad20734d46bcb"
)
EXPECTED_COUNTS = {"all": 22199, "train": 20055, "dev": 2144}

DEFAULT_DATASET_PATH = (
    PROJECT_ROOT / "experiments" / "stage2_v2_dataset" / "stage2_v2_candidates.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "stage2_v2_token_analysis_qwen25_3b"
)

ANALYSIS_VERSION = "stage2_v2_qwen25_3b_token_analysis_v1"
RECORD_SCHEMA_VERSION = "stage2_v2_token_length_record_v1"
SUMMARY_SCHEMA_VERSION = "stage2_v2_token_length_summary_v1"
MANIFEST_SCHEMA_VERSION = "stage2_v2_token_length_manifest_v1"
PERCENTILE_DEFINITION = "nearest_rank_ceiling_empirical_v1"
STANDARD_DEVIATION_DEFINITION = "population_standard_deviation_v1"
THRESHOLDS = (1024, 1280, 1536, 1792, 2048, 2304, 2560, 3072, 4096)

RECORDS_FILENAME = "stage2_v2_qwen25_3b_token_lengths.jsonl"
SUMMARY_FILENAME = "stage2_v2_qwen25_3b_token_summary.json"
MANIFEST_FILENAME = "stage2_v2_qwen25_3b_token_manifest.json"


def _fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def nearest_rank_percentile(values: Sequence[int], percentile: float) -> int:
    """Return sorted[ceil(p / 100 * n) - 1], clamped to the sample."""

    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sample")
    if not math.isfinite(percentile) or not 0.0 < percentile <= 100.0:
        raise ValueError("Percentile must be finite and in (0, 100]")
    ordered = sorted(int(value) for value in values)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("Cannot summarize an empty distribution")
    count = len(values)
    total = sum(int(value) for value in values)
    mean = total / count
    variance = sum((int(value) - mean) ** 2 for value in values) / count
    return {
        "count": count,
        "minimum": min(values),
        "mean": mean,
        "standard_deviation": math.sqrt(variance),
        "median": nearest_rank_percentile(values, 50.0),
        "p90": nearest_rank_percentile(values, 90.0),
        "p95": nearest_rank_percentile(values, 95.0),
        "p99": nearest_rank_percentile(values, 99.0),
        "p99_5": nearest_rank_percentile(values, 99.5),
        "p99_9": nearest_rank_percentile(values, 99.9),
        "maximum": max(values),
    }


def _threshold_statistics(values: Sequence[int]) -> dict[str, dict[str, int | float]]:
    count = len(values)
    if count == 0:
        raise ValueError("Cannot calculate threshold counts for an empty sample")
    result: dict[str, dict[str, int | float]] = {}
    for threshold in THRESHOLDS:
        exceeded = sum(int(value) > threshold for value in values)
        result[f"gt_{threshold}"] = {
            "count": exceeded,
            "percentage": 100.0 * exceeded / count,
            "records_requiring_truncation": exceeded,
        }
    return result


class _ScopeStatistics:
    def __init__(self) -> None:
        self.total_lengths = array("I")
        self.prefix_lengths = array("I")
        self.block_lengths = array("I")
        self.max_total = -1
        self.max_total_keys: list[str] = []
        self.max_prefix = -1
        self.max_prefix_keys: list[str] = []
        self.maximum_position_id = -1

    def observe(
        self,
        *,
        instance_key: str,
        total_tokens: int,
        prefix_tokens: int,
        block_lengths: Sequence[int],
        max_position_id: int,
    ) -> None:
        self.total_lengths.append(total_tokens)
        self.prefix_lengths.append(prefix_tokens)
        self.block_lengths.extend(block_lengths)
        if total_tokens > self.max_total:
            self.max_total = total_tokens
            self.max_total_keys = [instance_key]
        elif total_tokens == self.max_total:
            self.max_total_keys.append(instance_key)
        if prefix_tokens > self.max_prefix:
            self.max_prefix = prefix_tokens
            self.max_prefix_keys = [instance_key]
        elif prefix_tokens == self.max_prefix:
            self.max_prefix_keys.append(instance_key)
        self.maximum_position_id = max(self.maximum_position_id, max_position_id)

    def result(self) -> dict[str, Any]:
        return {
            "total_packed_tokens": _distribution(self.total_lengths),
            "prefix_tokens": _distribution(self.prefix_lengths),
            "candidate_block_tokens": _distribution(self.block_lengths),
            "thresholds": _threshold_statistics(self.total_lengths),
            "maximum_total_length_instance_keys": sorted(self.max_total_keys),
            "maximum_prefix_length_instance_keys": sorted(self.max_prefix_keys),
            "maximum_observed_individual_block_length": max(self.block_lengths),
            "maximum_observed_position_id": self.maximum_position_id,
        }


def _source_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolved_tokenizer_commit(tokenizer: Any) -> str | None:
    candidates = [getattr(tokenizer, "_commit_hash", None)]
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        candidates.append(init_kwargs.get("_commit_hash"))
    commits = {str(value) for value in candidates if value}
    if len(commits) > 1:
        raise ValueError("Tokenizer exposes conflicting resolved commit hashes")
    return next(iter(commits), None)


def validate_tokenizer(tokenizer: Any) -> str | None:
    if getattr(tokenizer, "is_fast", False) is not True:
        raise ValueError("Stage-2 v2 token analysis requires a fast tokenizer")
    try:
        probe = tokenizer(
            "Contextual fit:",
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            truncation=False,
        )
    except Exception as error:
        raise ValueError("Tokenizer must provide offset mappings") from error
    if not isinstance(probe, Mapping) or "offset_mapping" not in probe:
        raise ValueError("Tokenizer must provide offset mappings")
    if not probe.get("offset_mapping"):
        raise ValueError("Tokenizer returned empty offset mappings")
    resolved_commit = _resolved_tokenizer_commit(tokenizer)
    if resolved_commit is not None and resolved_commit != REQUESTED_MODEL_REVISION:
        raise ValueError(
            "Tokenizer resolved commit mismatch: "
            f"requested={REQUESTED_MODEL_REVISION} resolved={resolved_commit}"
        )
    return resolved_commit


def load_production_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=REQUESTED_MODEL_REVISION,
        use_fast=True,
    )
    validate_tokenizer(tokenizer)
    return tokenizer


def analysis_configuration(
    *,
    dataset_sha256: str = EXPECTED_DATASET_SHA256,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> dict[str, Any]:
    architecture_fingerprint = phase2_architecture_fingerprint()
    if architecture_fingerprint != EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT:
        raise ValueError(
            "Frozen Phase-2 architecture fingerprint mismatch: "
            f"expected={EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT} "
            f"observed={architecture_fingerprint}"
        )
    return {
        "analysis_version": ANALYSIS_VERSION,
        "candidate_count": TOP_K,
        "expected_counts": {
            scope: int(expected_counts[scope]) for scope in ("all", "train", "dev")
        },
        "input_dataset_sha256": dataset_sha256.lower(),
        "input_serialization_version": INPUT_SERIALIZATION_VERSION,
        "model_id": MODEL_ID,
        "percentile_definition": PERCENTILE_DEFINITION,
        "phase2_architecture_fingerprint": architecture_fingerprint,
        "requested_model_revision": REQUESTED_MODEL_REVISION,
        "standard_deviation_definition": STANDARD_DEVIATION_DEFINITION,
        "thresholds": list(THRESHOLDS),
        "token_boundary_policy_version": TOKEN_BOUNDARY_POLICY_VERSION,
        "truncation": False,
        "padding": False,
        "additional_special_tokens": False,
    }


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
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
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _analyze_stream(
    source: Path,
    destination: Path,
    tokenizer: Any,
) -> tuple[str, dict[str, Any]]:
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    output_digest = hashlib.sha256()
    scopes = {scope: _ScopeStatistics() for scope in ("all", "train", "dev")}
    split_counts = {"all": 0, "train": 0, "dev": 0}
    instance_keys: set[str] = set()
    try:
        with source.open("r", encoding="utf-8") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            for source_line_number, line in enumerate(source_handle, 1):
                if not line.strip():
                    raise ValueError(f"Blank dataset line at {source_line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at dataset line {source_line_number}"
                    ) from error
                if not isinstance(record, Mapping):
                    raise ValueError(f"Dataset line {source_line_number} is not an object")
                instance_key = record.get("instance_key")
                split = record.get("split")
                if not isinstance(instance_key, str) or not instance_key:
                    raise ValueError(f"Dataset line {source_line_number} has no instance key")
                if instance_key in instance_keys:
                    raise ValueError(f"Duplicate dataset instance key: {instance_key}")
                instance_keys.add(instance_key)
                if split not in {"train", "dev"}:
                    raise ValueError(f"Dataset line {source_line_number} has invalid split")
                candidates = record.get("candidates")
                if record.get("candidate_count") != TOP_K or not isinstance(
                    candidates, list
                ) or len(candidates) != TOP_K:
                    raise ValueError(
                        f"Dataset line {source_line_number} must contain exactly 50 candidates"
                    )

                event = tokenize_scoring_event(record, tokenizer)
                total_tokens = int(event.input_ids.shape[0])
                prefix_tokens = int(event.common_prefix_length)
                block_lengths = [end - start for start, end in event.block_spans]
                if len(block_lengths) != TOP_K:
                    raise ValueError("Frozen tokenizer did not return exactly 50 blocks")
                score_indices = event.score_token_indices.tolist()
                score_positions = event.position_ids[event.score_token_indices].tolist()
                if len(score_indices) != TOP_K or len(score_positions) != TOP_K:
                    raise ValueError("Frozen tokenizer did not return exactly 50 score markers")
                max_position_id = int(event.position_ids.max().item())
                analytical_record = {
                    "instance_key": instance_key,
                    "max_block_tokens": max(block_lengths),
                    "max_position_id": max_position_id,
                    "mean_block_tokens": sum(block_lengths) / TOP_K,
                    "min_block_tokens": min(block_lengths),
                    "prefix_tokens": prefix_tokens,
                    "schema_version": RECORD_SCHEMA_VERSION,
                    "score_marker_position_ids": score_positions,
                    "score_marker_position_id_max": max(score_positions),
                    "score_marker_position_id_min": min(score_positions),
                    "score_marker_token_indices": score_indices,
                    "score_marker_token_index_max": max(score_indices),
                    "score_marker_token_index_min": min(score_indices),
                    "split": split,
                    "total_tokens": total_tokens,
                }
                encoded = canonical_json_bytes(analytical_record) + b"\n"
                output_handle.write(encoded)
                output_digest.update(encoded)
                for scope in ("all", str(split)):
                    scopes[scope].observe(
                        instance_key=instance_key,
                        total_tokens=total_tokens,
                        prefix_tokens=prefix_tokens,
                        block_lengths=block_lengths,
                        max_position_id=max_position_id,
                    )
                    split_counts[scope] += 1
                del event
            statistics = {
                "counts": split_counts,
                "scopes": {
                    scope: scopes[scope].result()
                    for scope in ("all", "train", "dev")
                },
                "tokenization_failures": 0,
            }
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return output_digest.hexdigest(), statistics


def analyze_stage2_v2_tokens(
    *,
    tokenizer: Any,
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    expected_dataset_sha256: str = EXPECTED_DATASET_SHA256,
    expected_counts: Mapping[str, int] = EXPECTED_COUNTS,
) -> dict[str, Any]:
    """Stream and measure the exact frozen scorer inputs."""

    resolved_commit = validate_tokenizer(tokenizer)
    configuration = analysis_configuration(
        dataset_sha256=expected_dataset_sha256,
        expected_counts=expected_counts,
    )
    analysis_fingerprint = _fingerprint(configuration)
    source = Path(dataset_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    observed_sha = sha256_file(source)
    if observed_sha.lower() != expected_dataset_sha256.lower():
        raise ValueError(
            "Frozen Stage-2 v2 dataset SHA256 mismatch: "
            f"expected={expected_dataset_sha256.lower()} observed={observed_sha.lower()}"
        )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "records": output / RECORDS_FILENAME,
        "summary": output / SUMMARY_FILENAME,
        "manifest": output / MANIFEST_FILENAME,
    }
    collisions = [path for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "Token-analysis outputs already exist; use a new output directory: "
            + ", ".join(str(path) for path in collisions)
        )

    records_sha, statistics = _analyze_stream(source, paths["records"], tokenizer)
    try:
        observed_counts = statistics["counts"]
        required_counts = {
            scope: int(expected_counts[scope]) for scope in ("all", "train", "dev")
        }
        if observed_counts != required_counts:
            raise ValueError(
                f"Stage-2 v2 dataset accounting mismatch: {observed_counts} != {required_counts}"
            )
        summary = {
            "analysis_configuration": configuration,
            "analysis_fingerprint": analysis_fingerprint,
            "counts": observed_counts,
            "failures": [],
            "per_record_artifact": {
                "filename": RECORDS_FILENAME,
                "records": observed_counts["all"],
                "sha256": records_sha,
            },
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "source": {
                "dataset_path": _source_path(source),
                "dataset_sha256": observed_sha,
                "record_count": observed_counts["all"],
            },
            "splits": statistics["scopes"],
            "tokenization_failures": statistics["tokenization_failures"],
        }
        _atomic_json(paths["summary"], summary)
        summary_sha = sha256_file(paths["summary"])
        runtime_provenance = {
            "resolved_tokenizer_commit": resolved_commit,
            "runtime_torch_version": torch.__version__,
            "runtime_transformers_version": transformers.__version__,
            "tokenizer_class": (
                f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}"
            ),
            "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
            "tokenizer_model_max_length": getattr(tokenizer, "model_max_length", None),
        }
        manifest = {
            "analysis_configuration": configuration,
            "analysis_fingerprint": analysis_fingerprint,
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "artifacts": {
                "records": {
                    "filename": RECORDS_FILENAME,
                    "records": observed_counts["all"],
                    "sha256": records_sha,
                },
                "summary": {
                    "filename": SUMMARY_FILENAME,
                    "sha256": summary_sha,
                },
            },
            "runtime_provenance": runtime_provenance,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "source": summary["source"],
        }
        _atomic_json(paths["manifest"], manifest)
    except Exception:
        for path in paths.values():
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    return {
        "analysis_fingerprint": analysis_fingerprint,
        "counts": statistics["counts"],
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "records_path": str(paths["records"]),
        "records_sha256": records_sha,
        "summary_path": str(paths["summary"]),
        "summary_sha256": summary_sha,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    tokenizer = load_production_tokenizer()
    result = analyze_stage2_v2_tokens(
        tokenizer=tokenizer,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
