"""Build deterministic Stage-2 v2 ranker data from the frozen TRAIN audit.

This module is intentionally downstream of the frozen Stage-1/TRAIN audit.  It
does not import or execute extraction, KBRD, CKG, RRF, ReDial reconstruction,
or any model code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_AUDIT = (
    PROJECT_ROOT
    / ".codex_stage2_v2_review"
    / "codex_stage2_v2_review"
    / "train_rrf_candidates.audit.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "stage2_v2_dataset"

IMPLEMENTATION_BASE_COMMIT = "ed034c1be1cd1d64af1d9bd0dc343755f22d1563"
EXPECTED_SOURCE_AUDIT_SHA256 = (
    "d25fb6d05268b1e72570dd0eae680a5b93c4298588ba6410e41a8d32ac4be0a9"
)
SOURCE_AUDIT_SCHEMA_VERSION = "rrf_train_audit_v2"

DATASET_SCHEMA_VERSION = "stage2_v2_candidates_v1"
SUMMARY_SCHEMA_VERSION = "stage2_v2_summary_v1"
MANIFEST_SCHEMA_VERSION = "stage2_v2_manifest_v1"
BUILDER_VERSION = "stage2_v2_phase1_builder_v1"
CANDIDATE_ORDER_VERSION = "stage2_v2_candidate_order_v1"
CANDIDATE_ORDER_SALT = "stage2_v2_candidate_order_v1|"
TITLE_SANITIZER_VERSION = "single_line_control_whitespace_v1"
LABEL_SEMANTICS_VERSION = "frozen_normalized_title_unordered_set_v1"
HISTORY_POLICY_VERSION = "copy_frozen_pre_target_history_v1"

DATASET_FILENAME = "stage2_v2_candidates.jsonl"
SUMMARY_FILENAME = "stage2_v2_summary.json"
MANIFEST_FILENAME = "stage2_v2_manifest.json"

TOP_K = 50
NO_SEED_EXCLUSION = "no_neural_kbrd_inference_seeds"
GOLD_ABSENT_EXCLUSION = "ground_truth_absent_from_rrf_top50"
LABEL_OBSERVED = "observed_positive_in_top50"
LABEL_ANCHOR_ONLY = "gold_absent_from_rrf_top50_anchor_only"
UNORDERED_SET_SEMANTICS = "unordered_set_canonicalized_for_serialization_only"

EXPECTED_ACCOUNTING = {
    "global": {
        "total_events": 23686,
        "retrieval_completed": 22199,
        "reachable": 14334,
        "gold_absent": 7865,
        "no_seed": 1487,
    },
    "train": {
        "total_events": 21383,
        "retrieval_completed": 20055,
        "reachable": 12970,
        "gold_absent": 7085,
        "no_seed": 1328,
    },
    "dev": {
        "total_events": 2303,
        "retrieval_completed": 2144,
        "reachable": 1364,
        "gold_absent": 780,
        "no_seed": 159,
    },
    "conversation_overlap_count": 0,
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_audit(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_SOURCE_AUDIT_SHA256,
) -> Path:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    observed = sha256_file(source)
    if observed.lower() != expected_sha256.lower():
        raise ValueError(
            "Frozen TRAIN audit SHA256 mismatch: "
            f"expected={expected_sha256.lower()} observed={observed.lower()}"
        )
    return source


def sanitize_title(value: Any) -> str:
    sanitized = "".join(
        " " if ord(character) < 32 or ord(character) == 127 else character
        for character in str(value)
    )
    return " ".join(sanitized.split())


def normalize_title(value: Any) -> str:
    title = re.sub(r"\(\d{4}\)", "", str(value))
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.lower().strip()


def candidate_order_hash(canonical_entity_id: int) -> str:
    material = f"{CANDIDATE_ORDER_SALT}{int(canonical_entity_id)}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def candidate_order_key(candidate: Mapping[str, Any]) -> tuple[str, int]:
    entity_id = _strict_int(candidate.get("id"), "candidate.id")
    return candidate_order_hash(entity_id), entity_id


def _strict_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _optional_rank(value: Any, field: str) -> int | None:
    if value is None:
        return None
    rank = _strict_int(value, field)
    if not 1 <= rank <= TOP_K:
        raise ValueError(f"{field} must be between 1 and {TOP_K}")
    return rank


def _finite_number(value: Any, field: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if nonnegative and number < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return number


def _source_candidate_digest(candidates_by_rrf_rank: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_digest(
        [
            {
                "position": position,
                "id": _strict_int(candidate.get("id"), "rrf_top50[].id"),
                "title": str(candidate.get("title", "")),
            }
            for position, candidate in enumerate(candidates_by_rrf_rank, 1)
        ]
    )


def _validate_rrf_candidates(
    value: Any,
    *,
    source_candidate_digest: Any,
    context: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != TOP_K:
        raise ValueError(f"{context} must contain exactly {TOP_K} RRF candidates")
    candidates: list[dict[str, Any]] = []
    ranks: list[int] = []
    ids: list[int] = []
    for index, candidate in enumerate(value, 1):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{context} candidate {index} must be an object")
        item = dict(candidate)
        entity_id = _strict_int(item.get("id"), f"{context}[{index}].id")
        rank = _strict_int(item.get("rank"), f"{context}[{index}].rank")
        if not isinstance(item.get("title"), str) or not item["title"].strip():
            raise ValueError(f"{context}[{index}].title must be nonempty")
        if item.get("source") != "RRF":
            raise ValueError(f"{context}[{index}].source must be RRF")
        _finite_number(item.get("rrf_score"), f"{context}[{index}].rrf_score")
        _finite_number(
            item.get("kbrd_contribution"),
            f"{context}[{index}].kbrd_contribution",
            nonnegative=True,
        )
        _finite_number(
            item.get("ckg_contribution"),
            f"{context}[{index}].ckg_contribution",
            nonnegative=True,
        )
        _optional_rank(item.get("kbrd_rank"), f"{context}[{index}].kbrd_rank")
        _optional_rank(item.get("ckg_rank"), f"{context}[{index}].ckg_rank")
        ranks.append(rank)
        ids.append(entity_id)
        candidates.append(item)
    if sorted(ranks) != list(range(1, TOP_K + 1)):
        raise ValueError(f"{context} ranks must be exactly 1..{TOP_K}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context} contains duplicate canonical IDs")
    by_rank = sorted(candidates, key=lambda candidate: int(candidate["rank"]))
    expected_digest = _source_candidate_digest(by_rank)
    if source_candidate_digest != expected_digest:
        raise ValueError(f"{context} source candidate digest mismatch")
    return by_rank


def serialize_candidates(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the deterministic RRF-independent candidate serialization."""

    if len(candidates) != TOP_K:
        raise ValueError(f"Expected exactly {TOP_K} candidates")
    serialized: list[dict[str, Any]] = []
    for serialization_position, candidate in enumerate(
        sorted((dict(candidate) for candidate in candidates), key=candidate_order_key),
        1,
    ):
        serialized.append(
            {
                "canonical_entity_id": _strict_int(candidate.get("id"), "candidate.id"),
                "ckg_contribution": _finite_number(
                    candidate.get("ckg_contribution"), "candidate.ckg_contribution"
                ),
                "ckg_rank": _optional_rank(candidate.get("ckg_rank"), "candidate.ckg_rank"),
                "kbrd_contribution": _finite_number(
                    candidate.get("kbrd_contribution"), "candidate.kbrd_contribution"
                ),
                "kbrd_rank": _optional_rank(
                    candidate.get("kbrd_rank"), "candidate.kbrd_rank"
                ),
                "local_id": f"C{serialization_position:02d}",
                "rrf_rank": _strict_int(candidate.get("rank"), "candidate.rank"),
                "rrf_score": _finite_number(candidate.get("rrf_score"), "candidate.rrf_score"),
                "serialization_position": serialization_position,
                "source": str(candidate.get("source")),
                "title_original": str(candidate.get("title", "")),
                "title_sanitized": sanitize_title(candidate.get("title", "")),
            }
        )
    ids = [candidate["canonical_entity_id"] for candidate in serialized]
    if len(ids) != len(set(ids)):
        raise ValueError("Serialized candidates contain duplicate canonical IDs")
    return serialized


def _validate_source_record(record: Any, line_number: int) -> tuple[str, str]:
    context = f"audit line {line_number}"
    if not isinstance(record, Mapping):
        raise ValueError(f"{context} must be an object")
    if record.get("schema_version") != SOURCE_AUDIT_SCHEMA_VERSION:
        raise ValueError(f"{context} source schema mismatch")
    if record.get("source_split") != "TRAIN":
        raise ValueError(f"{context} must have source_split=TRAIN")
    split = record.get("split")
    if split not in {"train", "dev"}:
        raise ValueError(f"{context} split must be train or dev")
    instance_key = record.get("instance_key")
    conversation_key = record.get("conversation_key")
    if not isinstance(instance_key, str) or not instance_key:
        raise ValueError(f"{context} has invalid instance_key")
    if not isinstance(conversation_key, str) or not conversation_key:
        raise ValueError(f"{context} has invalid conversation_key")
    if not isinstance(record.get("history"), str):
        raise ValueError(f"{context} history must be a string")
    expected_history_sha = hashlib.sha256(record["history"].encode("utf-8")).hexdigest()
    if record.get("history_sha256") != expected_history_sha:
        raise ValueError(f"{context} history SHA256 mismatch")
    if record.get("failures") != []:
        raise ValueError(f"{context} contains source failures")
    return split, conversation_key


def _validate_no_seed_record(record: Mapping[str, Any], context: str) -> None:
    for field in ("kbrd_top50", "loo_ckg_top50", "rrf_top50"):
        if record.get(field) != []:
            raise ValueError(f"{context} no-seed {field} must be empty")
    if (
        record.get("positive_positions") != []
        or record.get("target_positions") != []
        or record.get("positive_positions_truncated") is not False
        or record.get("assistant_target") is not None
        or record.get("candidate_digest") is not None
    ):
        raise ValueError(f"{context} no-seed label/candidate provenance is malformed")
    diagnostics = record.get("diagnostics")
    kbrd_diagnostics = (
        diagnostics.get("kbrd") if isinstance(diagnostics, Mapping) else None
    )
    if (
        not isinstance(kbrd_diagnostics, Mapping)
        or kbrd_diagnostics.get("fallback_reason") != "no_inference_seeds"
        or kbrd_diagnostics.get("seed_entity_ids") != []
        or diagnostics.get("loo_ckg") is not None
    ):
        raise ValueError(f"{context} no-seed diagnostics provenance is malformed")


def transform_audit_record(record: Mapping[str, Any], line_number: int) -> dict[str, Any] | None:
    """Validate one source record and transform retrieval-completed records."""

    split, conversation_key = _validate_source_record(record, line_number)
    context = f"audit line {line_number}"
    exclusion_reason = record.get("exclusion_reason")
    eligible = record.get("eligible")
    if type(eligible) is not bool:
        raise ValueError(f"{context} eligible must be boolean")

    if exclusion_reason == NO_SEED_EXCLUSION:
        if eligible:
            raise ValueError(f"{context} no-seed record cannot be eligible")
        _validate_no_seed_record(record, context)
        return None

    if eligible:
        if exclusion_reason is not None:
            raise ValueError(f"{context} eligible record has an exclusion reason")
    elif exclusion_reason != GOLD_ABSENT_EXCLUSION:
        raise ValueError(f"{context} has an unknown exclusion reason")

    for field in ("kbrd_top50", "loo_ckg_top50"):
        if not isinstance(record.get(field), list) or len(record[field]) != TOP_K:
            raise ValueError(f"{context} {field} must contain exactly {TOP_K} candidates")

    by_rrf_rank = _validate_rrf_candidates(
        record.get("rrf_top50"),
        source_candidate_digest=record.get("candidate_digest"),
        context=f"{context} rrf_top50",
    )
    source_positive_positions = record.get("positive_positions")
    if not isinstance(source_positive_positions, list) or any(
        type(position) is not int for position in source_positive_positions
    ):
        raise ValueError(f"{context} positive_positions must be a list of integers")
    observed_positive_rrf_positions = sorted(set(source_positive_positions))
    if observed_positive_rrf_positions != source_positive_positions:
        raise ValueError(f"{context} positive_positions must be unique and ascending")
    if any(not 1 <= position <= TOP_K for position in observed_positive_rrf_positions):
        raise ValueError(f"{context} has a positive position outside 1..{TOP_K}")

    ground_truth = record.get("ground_truth_titles")
    if not isinstance(ground_truth, list) or not ground_truth or not all(
        isinstance(title, str) for title in ground_truth
    ):
        raise ValueError(f"{context} ground_truth_titles must be nonempty strings")
    normalized_ground_truth = {normalize_title(title) for title in ground_truth}
    if record.get("normalized_ground_truth_titles") != sorted(normalized_ground_truth):
        raise ValueError(f"{context} normalized ground-truth provenance mismatch")
    recomputed_positions = [
        rank
        for rank, candidate in enumerate(by_rrf_rank, 1)
        if normalize_title(candidate["title"]) in normalized_ground_truth
    ]
    if recomputed_positions != observed_positive_rrf_positions:
        raise ValueError(f"{context} frozen positive-position semantics mismatch")
    if eligible != bool(observed_positive_rrf_positions):
        raise ValueError(f"{context} eligibility and observed positives disagree")

    candidates = serialize_candidates(by_rrf_rank)
    serialization_by_id = {
        candidate["canonical_entity_id"]: candidate["serialization_position"]
        for candidate in candidates
    }
    positive_ids = {
        _strict_int(by_rrf_rank[position - 1].get("id"), "positive candidate ID")
        for position in observed_positive_rrf_positions
    }
    observed_positive_serialization_positions = sorted(
        serialization_by_id[entity_id] for entity_id in positive_ids
    )
    positive_position_mappings = sorted(
        (
            {
                "canonical_entity_id": _strict_int(
                    by_rrf_rank[rrf_position - 1].get("id"),
                    "positive candidate ID",
                ),
                "rrf_position": rrf_position,
                "serialization_position": serialization_by_id[
                    _strict_int(
                        by_rrf_rank[rrf_position - 1].get("id"),
                        "positive candidate ID",
                    )
                ],
            }
            for rrf_position in observed_positive_rrf_positions
        ),
        key=lambda item: item["canonical_entity_id"],
    )
    label_status = LABEL_OBSERVED if eligible else LABEL_ANCHOR_ONLY

    return {
        "candidate_count": TOP_K,
        "candidates": candidates,
        "conversation_contribution_digest": record.get(
            "conversation_contribution_digest"
        ),
        "conversation_id": record.get("conversation_id"),
        "conversation_key": conversation_key,
        "history": record["history"],
        "history_sha256": record["history_sha256"],
        "instance_key": record["instance_key"],
        "label_set_semantics": UNORDERED_SET_SEMANTICS,
        "label_status": label_status,
        "line_number": _strict_int(record.get("line_number"), f"{context}.line_number"),
        "observed_positive_rrf_positions": observed_positive_rrf_positions,
        "observed_positive_serialization_positions": (
            observed_positive_serialization_positions
        ),
        "observed_positive_position_mappings": positive_position_mappings,
        "schema_version": DATASET_SCHEMA_VERSION,
        "serialization_digest": canonical_json_digest(candidates),
        "serialization_order_version": CANDIDATE_ORDER_VERSION,
        "source_audit_schema_version": SOURCE_AUDIT_SCHEMA_VERSION,
        "source_audit_record_number": line_number,
        "source_candidate_digest": record["candidate_digest"],
        "source_run_fingerprint": record.get("run_fingerprint"),
        "source_split": "TRAIN",
        "split": split,
        "turn_index": _strict_int(record.get("turn_index"), f"{context}.turn_index"),
    }


class AuditStatistics:
    def __init__(self) -> None:
        self.accounting: dict[str, Counter[str]] = {
            "global": Counter(),
            "train": Counter(),
            "dev": Counter(),
        }
        self.conversations: dict[str, set[str]] = defaultdict(set)
        self.positive_cardinality: dict[str, Counter[int]] = {
            "global": Counter(),
            "train": Counter(),
            "dev": Counter(),
        }
        self.collision_events = 0
        self.reachable_collision_events = 0
        self.collision_extra_positions = 0
        self.max_title_multiplicity = 0
        self.positive_collision_events = 0
        self.source_run_fingerprints: set[str] = set()
        self.instance_keys: set[str] = set()

    def observe_source(
        self,
        record: Mapping[str, Any],
        *,
        transformed: Mapping[str, Any] | None,
    ) -> None:
        split = str(record["split"])
        for scope in ("global", split):
            self.accounting[scope]["total_events"] += 1
        self.conversations[split].add(str(record["conversation_key"]))
        instance_key = str(record["instance_key"])
        if instance_key in self.instance_keys:
            raise ValueError(f"Duplicate source instance_key: {instance_key}")
        self.instance_keys.add(instance_key)
        run_fingerprint = record.get("run_fingerprint")
        if not isinstance(run_fingerprint, str) or not run_fingerprint:
            raise ValueError(f"Source record {instance_key} has no run fingerprint")
        self.source_run_fingerprints.add(run_fingerprint)

        if transformed is None:
            for scope in ("global", split):
                self.accounting[scope]["no_seed"] += 1
            return

        positive_count = len(transformed["observed_positive_rrf_positions"])
        category = "reachable" if positive_count else "gold_absent"
        for scope in ("global", split):
            self.accounting[scope]["retrieval_completed"] += 1
            self.accounting[scope][category] += 1
            self.positive_cardinality[scope][positive_count] += 1

        title_counts = Counter(
            normalize_title(candidate["title_original"])
            for candidate in transformed["candidates"]
        )
        maximum = max(title_counts.values())
        extras = sum(count - 1 for count in title_counts.values() if count > 1)
        if extras:
            self.collision_events += 1
            self.collision_extra_positions += extras
            if positive_count:
                self.reachable_collision_events += 1
        self.max_title_multiplicity = max(self.max_title_multiplicity, maximum)
        if positive_count and any(
            title_counts[
                normalize_title(
                    next(
                        candidate["title_original"]
                        for candidate in transformed["candidates"]
                        if candidate["rrf_rank"] == rrf_position
                    )
                )
            ]
            > 1
            for rrf_position in transformed["observed_positive_rrf_positions"]
        ):
            self.positive_collision_events += 1

    def result(self) -> dict[str, Any]:
        overlap = self.conversations["train"] & self.conversations["dev"]
        accounting = {
            scope: {
                field: int(self.accounting[scope][field])
                for field in (
                    "total_events",
                    "retrieval_completed",
                    "reachable",
                    "gold_absent",
                    "no_seed",
                )
            }
            for scope in ("global", "train", "dev")
        }
        return {
            "accounting": accounting,
            "conversation_counts": {
                "train": len(self.conversations["train"]),
                "dev": len(self.conversations["dev"]),
            },
            "conversation_overlap_count": len(overlap),
            "normalized_title_collisions": {
                "retrieval_completed_events_with_collision": self.collision_events,
                "reachable_events_with_collision": self.reachable_collision_events,
                "extra_candidate_positions": self.collision_extra_positions,
                "maximum_title_multiplicity": self.max_title_multiplicity,
                "observed_positive_events_with_collision": self.positive_collision_events,
            },
            "observed_positive_set_cardinality": {
                scope: {
                    str(cardinality): count
                    for cardinality, count in sorted(
                        self.positive_cardinality[scope].items()
                    )
                }
                for scope in ("global", "train", "dev")
            },
            "source_run_fingerprints": sorted(self.source_run_fingerprints),
        }


def _validate_accounting(
    observed: Mapping[str, Any], expected: Mapping[str, Any] | None
) -> None:
    if expected is None:
        return
    comparable = {
        "global": observed["accounting"]["global"],
        "train": observed["accounting"]["train"],
        "dev": observed["accounting"]["dev"],
        "conversation_overlap_count": observed["conversation_overlap_count"],
    }
    if comparable != expected:
        raise ValueError(f"Authoritative accounting mismatch: {comparable} != {expected}")


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8"))
            handle.write(b"\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _source_path_for_manifest(source: Path) -> str:
    try:
        return source.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return source.as_posix()


def _builder_configuration(source_sha256: str) -> dict[str, Any]:
    return {
        "builder_version": BUILDER_VERSION,
        "candidate_order": {
            "algorithm": "sha256_utf8_hex_then_canonical_entity_id",
            "salt": CANDIDATE_ORDER_SALT,
            "version": CANDIDATE_ORDER_VERSION,
        },
        "implementation_base_commit": IMPLEMENTATION_BASE_COMMIT,
        "history_policy_version": HISTORY_POLICY_VERSION,
        "label_semantics_version": LABEL_SEMANTICS_VERSION,
        "schemas": {
            "dataset": DATASET_SCHEMA_VERSION,
            "manifest": MANIFEST_SCHEMA_VERSION,
            "source_audit": SOURCE_AUDIT_SCHEMA_VERSION,
            "summary": SUMMARY_SCHEMA_VERSION,
        },
        "source_audit_sha256": source_sha256,
        "title_sanitizer_version": TITLE_SANITIZER_VERSION,
    }


def _write_dataset_stream(
    source: Path,
    destination: Path,
) -> tuple[str, int, dict[str, Any]]:
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    digest = hashlib.sha256()
    output_records = 0
    statistics = AuditStatistics()
    try:
        with source.open("r", encoding="utf-8") as source_handle, os.fdopen(
            descriptor, "wb"
        ) as output_handle:
            for line_number, line in enumerate(source_handle, 1):
                if not line.strip():
                    raise ValueError(f"Blank line in source audit at line {line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in source audit at line {line_number}"
                    ) from error
                transformed = transform_audit_record(record, line_number)
                statistics.observe_source(record, transformed=transformed)
                if transformed is None:
                    continue
                encoded = canonical_json_bytes(transformed) + b"\n"
                output_handle.write(encoded)
                digest.update(encoded)
                output_records += 1
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return digest.hexdigest(), output_records, statistics.result()


def scan_source_audit(
    source_audit: str | Path,
    *,
    expected_source_sha256: str = EXPECTED_SOURCE_AUDIT_SHA256,
    expected_accounting: Mapping[str, Any] | None = EXPECTED_ACCOUNTING,
) -> dict[str, Any]:
    """Validate and summarize an audit without creating output artifacts."""

    source = validate_source_audit(
        source_audit,
        expected_sha256=expected_source_sha256,
    )
    statistics = AuditStatistics()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank line in source audit at line {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in source audit at line {line_number}"
                ) from error
            transformed = transform_audit_record(record, line_number)
            statistics.observe_source(record, transformed=transformed)
    result = statistics.result()
    _validate_accounting(result, expected_accounting)
    return result


def build_stage2_v2_dataset(
    *,
    source_audit: str | Path = DEFAULT_SOURCE_AUDIT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    expected_source_sha256: str = EXPECTED_SOURCE_AUDIT_SHA256,
    expected_accounting: Mapping[str, Any] | None = EXPECTED_ACCOUNTING,
) -> dict[str, Any]:
    """Create deterministic Phase-1 dataset, summary, and manifest artifacts."""

    source = validate_source_audit(
        source_audit,
        expected_sha256=expected_source_sha256,
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset": output / DATASET_FILENAME,
        "summary": output / SUMMARY_FILENAME,
        "manifest": output / MANIFEST_FILENAME,
    }
    collisions = [path for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "Phase-1 outputs already exist; use a new output directory: "
            + ", ".join(str(path) for path in collisions)
        )

    dataset_sha, output_records, statistics = _write_dataset_stream(
        source,
        paths["dataset"],
    )
    try:
        _validate_accounting(statistics, expected_accounting)
        expected_records = statistics["accounting"]["global"]["retrieval_completed"]
        if output_records != expected_records:
            raise ValueError(
                f"Output record count mismatch: {output_records} != {expected_records}"
            )
        if len(statistics["source_run_fingerprints"]) != 1:
            raise ValueError("Source audit must contain exactly one run fingerprint")

        configuration = _builder_configuration(expected_source_sha256.lower())
        builder_fingerprint = canonical_json_digest(configuration)
        summary = {
            "accounting": statistics["accounting"],
            "builder_fingerprint": builder_fingerprint,
            "candidate_order": configuration["candidate_order"],
            "conversation_counts": statistics["conversation_counts"],
            "conversation_overlap_count": statistics["conversation_overlap_count"],
            "dataset": {
                "filename": DATASET_FILENAME,
                "records": output_records,
                "sha256": dataset_sha,
            },
            "experiment": "stage2_v2_phase1_dataset",
            "failures": [],
            "normalized_title_collisions": statistics[
                "normalized_title_collisions"
            ],
            "observed_positive_set_cardinality": statistics[
                "observed_positive_set_cardinality"
            ],
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "source": {
                "audit_path": _source_path_for_manifest(source),
                "audit_schema_version": SOURCE_AUDIT_SCHEMA_VERSION,
                "audit_sha256": expected_source_sha256.lower(),
                "run_fingerprint": statistics["source_run_fingerprints"][0],
            },
        }
        _atomic_json(paths["summary"], summary)
        summary_sha = sha256_file(paths["summary"])
        manifest = {
            "accounting": statistics["accounting"],
            "artifacts": {
                "dataset": {
                    "filename": DATASET_FILENAME,
                    "records": output_records,
                    "sha256": dataset_sha,
                },
                "summary": {
                    "filename": SUMMARY_FILENAME,
                    "sha256": summary_sha,
                },
            },
            "builder_configuration": configuration,
            "builder_fingerprint": builder_fingerprint,
            "conversation_overlap_count": statistics[
                "conversation_overlap_count"
            ],
            "implementation_base_commit": IMPLEMENTATION_BASE_COMMIT,
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
        "dataset_path": str(paths["dataset"]),
        "dataset_sha256": dataset_sha,
        "manifest_path": str(paths["manifest"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
        "output_records": output_records,
        "statistics": statistics,
        "summary_path": str(paths["summary"]),
        "summary_sha256": summary_sha,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-audit", type=Path, default=DEFAULT_SOURCE_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = build_stage2_v2_dataset(
        source_audit=args.source_audit,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
