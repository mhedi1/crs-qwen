"""Build the frozen Stage-2-v2 TEST dataset from official ReDial TEST.

Scientific policy:
- TEST source is immutable and SHA-validated.
- Recommendation events use the same pre-target dialogue construction as TRAIN/DEV.
- KBRD uses the same frozen pure-neural Top-50 path.
- TEST no-seed events remain no-seed and are retained in full-denominator accounting.
- CKG uses the frozen TRAIN-built conversation/conditional/support-2 graph directly.
  No leave-one-conversation-out subtraction is applied on TEST.
- RRF uses the frozen k=60, equal-weight, Top-50 configuration.
- Stage-2 candidate serialization is exactly the frozen Stage2-v2 serialization.
- No model selection or parameter tuning is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from my_crs.build_rrf_train_dataset import (
    DEFAULT_KBRD_CHECKPOINT,
    KBRD_FALLBACK_NO_INFERENCE_SEEDS,
    _candidate_digest,
    _diagnostics_confirm_no_inference_seeds,
    _hash_checkpoint_bundle,
    _load_project_extraction_configuration,
    _load_real_dependencies,
    _rank_provenance,
    _validate_kbrd_checkpoint_selection,
    _validate_neural_kbrd_candidates,
    positive_candidate_positions,
    reconstruct_train_instances,
)
from my_crs.build_stage2_v2_dataset import (
    CANDIDATE_ORDER_VERSION,
    DATASET_SCHEMA_VERSION,
    GOLD_ABSENT_EXCLUSION,
    LABEL_ANCHOR_ONLY,
    LABEL_OBSERVED,
    NO_SEED_EXCLUSION,
    UNORDERED_SET_SEMANTICS,
    canonical_json_bytes,
    canonical_json_digest,
    normalize_title,
    serialize_candidates,
)
from my_crs.evaluate_ckg_complementarity import (
    DEFAULT_CACHE_DIR,
    DEFAULT_PARITY_OUTPUT_PATH,
    DEFAULT_VALID_PATH,
    attach_catalogue_titles,
    require_passing_parity_report,
    validate_valid_source_path,
)
from my_crs.evaluate_rrf_fusion import (
    CKG_WEIGHT,
    KBRD_WEIGHT,
    RRF_K,
    TOP_K,
    load_frozen_ckg,
    reciprocal_rank_fusion,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEST_PATH = (
    PROJECT_ROOT
    / "baseline_repo"
    / "KBRD_project"
    / "KBRD"
    / "data"
    / "redial"
    / "test_data.jsonl"
)

EXPECTED_TEST_SHA256 = (
    "d7781750787d104ac005e829cc7e6277d25b14644a223b0c2445aaaded6b19ac"
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "stage2_v2_test_dataset"

AUDIT_SCHEMA_VERSION = "rrf_test_audit_v1"
SUMMARY_SCHEMA_VERSION = "stage2_v2_test_summary_v1"
MANIFEST_SCHEMA_VERSION = "stage2_v2_test_manifest_v1"
BUILDER_VERSION = "stage2_v2_test_builder_v1"

AUDIT_FILENAME = "stage2_v2_test_audit.jsonl"
DATASET_FILENAME = "stage2_v2_test_candidates.jsonl"
SUMMARY_FILENAME = "stage2_v2_test_summary.json"
MANIFEST_FILENAME = "stage2_v2_test_manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_test_source(path: str | Path) -> Path:
    candidate = Path(path).resolve()
    official = DEFAULT_TEST_PATH.resolve()

    if candidate.name != "test_data.jsonl":
        raise ValueError(
            f"TEST builder requires test_data.jsonl, got {candidate.name!r}"
        )

    if candidate != official:
        raise ValueError(
            f"Expected official TEST path {official}, got {candidate}"
        )

    if not candidate.is_file():
        raise FileNotFoundError(candidate)

    observed = sha256_file(candidate)
    if observed != EXPECTED_TEST_SHA256:
        raise ValueError(
            "TEST SHA256 mismatch: "
            f"expected={EXPECTED_TEST_SHA256} observed={observed}"
        )

    return candidate


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        suffix=".tmp",
    )

    digest = hashlib.sha256()

    try:
        with os.fdopen(descriptor, "wb") as handle:
            for record in records:
                encoded = canonical_json_bytes(record) + b"\n"
                handle.write(encoded)
                digest.update(encoded)

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_name, path)

    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise

    return digest.hexdigest()


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(record) + b"\n"

    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not path.is_file():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"Blank/truncated JSONL record at {path}:{line_number}"
                )

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed JSONL record at {path}:{line_number}"
                ) from error

            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL record must be an object at {path}:{line_number}"
                )

            records.append(value)

    return records


def _conversation_key(
    line_number: int,
    conversation: Mapping[str, Any],
) -> str:
    return f"{line_number}:{conversation.get('conversationId')}"


def _stage2_record_from_audit(
    record: Mapping[str, Any],
    audit_record_number: int,
) -> dict[str, Any]:
    if record["category"] == "no_seed":
        raise ValueError("No-seed event cannot enter Stage2 candidate dataset")

    rrf_candidates = record["rrf_top50"]

    if not isinstance(rrf_candidates, list) or len(rrf_candidates) != TOP_K:
        raise ValueError("Retrieval-completed TEST event must contain 50 RRF candidates")

    candidates = serialize_candidates(rrf_candidates)

    positive_rrf_positions = list(record["positive_positions"])

    serialization_by_id = {
        candidate["canonical_entity_id"]:
        candidate["serialization_position"]
        for candidate in candidates
    }

    positive_ids = {
        int(rrf_candidates[position - 1]["id"])
        for position in positive_rrf_positions
    }

    positive_serialization_positions = sorted(
        serialization_by_id[entity_id]
        for entity_id in positive_ids
    )

    positive_position_mappings = sorted(
        (
            {
                "canonical_entity_id": int(
                    rrf_candidates[rrf_position - 1]["id"]
                ),
                "rrf_position": rrf_position,
                "serialization_position": serialization_by_id[
                    int(rrf_candidates[rrf_position - 1]["id"])
                ],
            }
            for rrf_position in positive_rrf_positions
        ),
        key=lambda item: item["canonical_entity_id"],
    )

    label_status = (
        LABEL_OBSERVED
        if positive_rrf_positions
        else LABEL_ANCHOR_ONLY
    )

    return {
        "candidate_count": TOP_K,
        "candidates": candidates,
        "conversation_contribution_digest": None,
        "conversation_id": record["conversation_id"],
        "conversation_key": record["conversation_key"],
        "history": record["history"],
        "history_sha256": record["history_sha256"],
        "instance_key": record["instance_key"],
        "label_set_semantics": UNORDERED_SET_SEMANTICS,
        "label_status": label_status,
        "line_number": int(record["line_number"]),
        "observed_positive_rrf_positions": positive_rrf_positions,
        "observed_positive_serialization_positions":
            positive_serialization_positions,
        "observed_positive_position_mappings":
            positive_position_mappings,
        "schema_version": DATASET_SCHEMA_VERSION,
        "serialization_digest": canonical_json_digest(candidates),
        "serialization_order_version": CANDIDATE_ORDER_VERSION,
        "source_audit_schema_version": AUDIT_SCHEMA_VERSION,
        "source_audit_record_number": audit_record_number,
        "source_candidate_digest": record["candidate_digest"],
        "source_run_fingerprint": record["run_fingerprint"],
        "source_split": "TEST",
        "split": "test",
        "turn_index": int(record["turn_index"]),
    }


def _process_event(
    *,
    event: Any,
    conversation: Mapping[str, Any],
    retriever: Any,
    run_fingerprint: str,
    kbrd_candidate_fn: Any,
    prepare_input_fn: Any,
    title_lookup: Any,
) -> dict[str, Any]:
    from my_crs import evaluate as frozen_evaluator

    history = frozen_evaluator.build_dialogue_up_to(
        conversation,
        event.turn_index - 1,
    )

    extracted = prepare_input_fn(history)

    if not isinstance(extracted, tuple) or not extracted:
        raise RuntimeError(
            "Resolver did not return its frozen tuple result"
        )

    all_extracted_entity_ids = list(extracted[0])

    kbrd_diagnostics: dict[str, Any] = {}

    kbrd_candidates, _decades = kbrd_candidate_fn(
        history,
        top_k=TOP_K,
        diagnostics=kbrd_diagnostics,
        use_fusion=False,
        retrieval_mode="kbrd",
    )

    fallback_reason = kbrd_diagnostics.get("fallback_reason")

    normalized_ground_truth = sorted(
        {
            normalize_title(title)
            for title in event.ground_truth_titles
        }
    )

    conversation_key = _conversation_key(
        event.line_number,
        conversation,
    )

    common = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "instance_key": event.key,
        "source_split": "TEST",
        "split": "test",
        "line_number": event.line_number,
        "conversation_id": event.conversation_id,
        "conversation_key": conversation_key,
        "turn_index": event.turn_index,
        "history": history,
        "history_sha256": hashlib.sha256(
            history.encode("utf-8")
        ).hexdigest(),
        "ground_truth_titles": list(event.ground_truth_titles),
        "normalized_ground_truth_titles":
            normalized_ground_truth,
        "unique_annotated_target_count":
            len(event.unique_target_ids),
        "failures": [],
    }

    if fallback_reason == KBRD_FALLBACK_NO_INFERENCE_SEEDS:
        if not _diagnostics_confirm_no_inference_seeds(
            kbrd_diagnostics
        ):
            raise RuntimeError(
                "KBRD no_inference_seeds diagnostics "
                "contradict inference seed IDs"
            )

        return {
            **common,
            "category": "no_seed",
            "kbrd_top50": [],
            "ckg_top50": [],
            "rrf_top50": [],
            "positive_positions": [],
            "eligible": False,
            "exclusion_reason": NO_SEED_EXCLUSION,
            "candidate_digest": None,
            "diagnostics": {
                "all_extracted_entity_ids":
                    all_extracted_entity_ids,
                "kbrd": kbrd_diagnostics,
                "ckg": None,
            },
        }

    if fallback_reason is not None:
        raise RuntimeError(
            f"Fatal KBRD fallback: {fallback_reason}"
        )

    _validate_neural_kbrd_candidates(
        kbrd_candidates,
        retriever.movie_ids,
    )

    ckg_views = retriever.retrieve_views(
        all_extracted_entity_ids,
        top_k=TOP_K,
    )

    ckg_candidates = attach_catalogue_titles(
        ckg_views["budget_controlled"],
        title_lookup,
    )

    if len(ckg_candidates) != TOP_K:
        raise RuntimeError(
            f"Frozen TEST CKG produced "
            f"{len(ckg_candidates)} candidates instead of {TOP_K}"
        )

    ckg_ids = [int(candidate["id"]) for candidate in ckg_candidates]

    if len(set(ckg_ids)) != TOP_K:
        raise RuntimeError(
            "Frozen TEST CKG contains duplicate canonical IDs"
        )

    rrf_candidates = reciprocal_rank_fusion(
        kbrd_candidates,
        ckg_candidates,
        rrf_k=RRF_K,
        top_k=TOP_K,
    )

    if len(rrf_candidates) != TOP_K:
        raise RuntimeError(
            f"Frozen TEST RRF produced "
            f"{len(rrf_candidates)} candidates instead of {TOP_K}"
        )

    rrf_ids = [int(candidate["id"]) for candidate in rrf_candidates]

    if len(set(rrf_ids)) != TOP_K:
        raise RuntimeError(
            "Frozen TEST RRF contains duplicate canonical IDs"
        )

    positives = positive_candidate_positions(
        rrf_candidates,
        event.ground_truth_titles,
    )

    category = "reachable" if positives else "gold_absent"

    return {
        **common,
        "category": category,
        "kbrd_top50": _rank_provenance(
            kbrd_candidates,
            include_scores=False,
        ),
        "ckg_top50": _rank_provenance(
            ckg_candidates,
            include_scores=True,
        ),
        "rrf_top50": _rank_provenance(
            rrf_candidates,
            include_scores=True,
        ),
        "positive_positions": positives,
        "eligible": bool(positives),
        "exclusion_reason": (
            None
            if positives
            else GOLD_ABSENT_EXCLUSION
        ),
        "candidate_digest":
            _candidate_digest(rrf_candidates),
        "diagnostics": {
            "all_extracted_entity_ids":
                all_extracted_entity_ids,
            "kbrd": kbrd_diagnostics,
            "ckg": ckg_views["diagnostics"],
        },
    }


def _summary(
    audit_records: Sequence[Mapping[str, Any]],
    dataset_records: Sequence[Mapping[str, Any]],
    *,
    reconstruction: Any,
    selected_event_count: int,
    source: Path,
    cache_path: Path,
    checkpoint: Path,
    run_fingerprint: str,
    audit_sha256: str,
    dataset_sha256: str,
) -> dict[str, Any]:
    categories = Counter(
        str(record["category"])
        for record in audit_records
    )

    total = len(audit_records)
    no_seed = categories["no_seed"]
    reachable = categories["reachable"]
    gold_absent = categories["gold_absent"]
    retrieval_completed = reachable + gold_absent

    if total != retrieval_completed + no_seed:
        raise RuntimeError(
            "TEST accounting invariant failed"
        )

    if len(dataset_records) != retrieval_completed:
        raise RuntimeError(
            "TEST dataset record count does not match "
            "retrieval-completed accounting"
        )

    hits_at_1 = 0
    hits_at_10 = 0
    hits_at_50 = 0
    reciprocal_rank_sum = 0.0

    for record in audit_records:
        positives = list(record["positive_positions"])

        if not positives:
            continue

        rank = min(positives)

        hits_at_1 += int(rank <= 1)
        hits_at_10 += int(rank <= 10)
        hits_at_50 += int(rank <= 50)
        reciprocal_rank_sum += 1.0 / rank

    if hits_at_50 != reachable:
        raise RuntimeError(
            "Frozen TEST RRF R@50 reachability invariant failed"
        )

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "stage2_v2_frozen_test_dataset",
        "source_split": "TEST",
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
        },
        "authoritative_reconstruction": {
            "conversations":
                len(reconstruction.conversations),
            "evaluable_conversations":
                reconstruction.evaluable_conversations,
            "recommendation_instances":
                len(reconstruction.events),
            "unique_suggested_target_occurrences":
                reconstruction.unique_target_occurrences,
            "max_unique_targets_per_instance":
                reconstruction.max_targets_per_instance,
        },
        "processed": {
            "selected_events": selected_event_count,
            "total_events": total,
            "retrieval_completed": retrieval_completed,
            "reachable": reachable,
            "gold_absent": gold_absent,
            "no_seed": no_seed,
        },
        "rrf_full_denominator_metrics": {
            "denominator": total,
            "hits_at_1": hits_at_1,
            "hits_at_10": hits_at_10,
            "hits_at_50": hits_at_50,
            "recall_at_1":
                hits_at_1 / total if total else None,
            "recall_at_10":
                hits_at_10 / total if total else None,
            "recall_at_50":
                hits_at_50 / total if total else None,
            "mrr":
                reciprocal_rank_sum / total
                if total else None,
            "reciprocal_rank_sum":
                reciprocal_rank_sum,
        },
        "frozen_stage1": {
            "kbrd_checkpoint": str(checkpoint),
            "ckg_cache": str(cache_path),
            "rrf_k": RRF_K,
            "weights": {
                "KBRD": KBRD_WEIGHT,
                "CKG": CKG_WEIGHT,
            },
            "candidate_budget": TOP_K,
        },
        "run_fingerprint": run_fingerprint,
        "artifacts": {
            "audit": {
                "filename": AUDIT_FILENAME,
                "sha256": audit_sha256,
                "records": total,
            },
            "dataset": {
                "filename": DATASET_FILENAME,
                "sha256": dataset_sha256,
                "records": len(dataset_records),
            },
        },
        "failures": [],
    }


def build_test_dataset(
    *,
    test_path: str | Path = DEFAULT_TEST_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    kbrd_checkpoint: str | Path = DEFAULT_KBRD_CHECKPOINT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_instances: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:

    if max_instances is not None and max_instances < 1:
        raise ValueError(
            "max_instances must be positive"
        )

    source = validate_test_source(test_path)

    # Preserve the previously established KBRD parity gate.
    validated_valid = validate_valid_source_path(
        DEFAULT_VALID_PATH
    )
    require_passing_parity_report(
        DEFAULT_PARITY_OUTPUT_PATH,
        validated_valid,
    )

    checkpoint = Path(kbrd_checkpoint).resolve()

    _validate_kbrd_checkpoint_selection(
        checkpoint,
        using_real_kbrd=True,
    )

    checkpoint_sha, checkpoint_files = (
        _hash_checkpoint_bundle(checkpoint)
    )

    extraction_configuration = (
        _load_project_extraction_configuration()
    )

    retriever, cache_path = load_frozen_ckg(cache_dir)

    if retriever.metadata.get("source_split") != "TRAIN":
        raise ValueError(
            "Frozen TEST CKG must be TRAIN-derived"
        )

    reconstruction = reconstruct_train_instances(
        source,
        expectations=None,
    )

    selected_events = list(reconstruction.events)

    if max_instances is not None:
        selected_events = selected_events[:max_instances]

    conversation_by_line = dict(
        reconstruction.conversations
    )

    scientific_configuration = {
        "builder_version": BUILDER_VERSION,
        "test_source": {
            "path": str(source),
            "sha256": EXPECTED_TEST_SHA256,
        },
        "event_construction": {
            "function":
                "reconstruct_train_instances",
            "history_policy":
                "pre_target_turn_only",
        },
        "extraction_configuration":
            extraction_configuration,
        "kbrd": {
            "checkpoint": str(checkpoint),
            "checkpoint_bundle_sha256":
                checkpoint_sha,
            "checkpoint_bundle_files":
                checkpoint_files,
            "retrieval_mode": "kbrd",
            "use_fusion": False,
            "top_k": TOP_K,
            "llm_used": False,
            "no_seed_policy":
                "retain_as_full_denominator_miss",
        },
        "ckg": {
            "cache_path": str(cache_path),
            "cache_sha256":
                sha256_file(cache_path),
            "metadata": retriever.metadata,
            "test_policy":
                "frozen_train_graph_no_loo_subtraction",
            "top_k": TOP_K,
        },
        "rrf": {
            "k": RRF_K,
            "weights": {
                "KBRD": KBRD_WEIGHT,
                "CKG": CKG_WEIGHT,
            },
            "top_k": TOP_K,
            "ranks": "1-based",
            "raw_scores_used": False,
            "tie_break":
                "entity_id_ascending",
        },
        "stage2_serialization": {
            "dataset_schema":
                DATASET_SCHEMA_VERSION,
            "candidate_order_version":
                CANDIDATE_ORDER_VERSION,
            "candidate_count": TOP_K,
        },
        "limits": {
            "max_instances": max_instances,
        },
    }

    run_fingerprint = canonical_json_digest(
        scientific_configuration
    )

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "audit": output / AUDIT_FILENAME,
        "dataset": output / DATASET_FILENAME,
        "summary": output / SUMMARY_FILENAME,
        "manifest": output / MANIFEST_FILENAME,
    }

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "scientific_configuration":
            scientific_configuration,
        "artifacts": {
            key: str(path)
            for key, path in paths.items()
        },
    }

    if resume:
        if not paths["manifest"].is_file():
            raise ValueError(
                "Resume requires an existing TEST manifest"
            )

        with paths["manifest"].open(
            "r",
            encoding="utf-8",
        ) as handle:
            stored_manifest = json.load(handle)

        if stored_manifest != manifest:
            raise ValueError(
                "TEST resume fingerprint/configuration mismatch"
            )

    else:
        collisions = [
            path
            for path in paths.values()
            if path.exists()
        ]

        if collisions:
            raise FileExistsError(
                "TEST artifacts already exist; "
                "use --resume or a new output directory: "
                + ", ".join(map(str, collisions))
            )

        _atomic_json(
            paths["manifest"],
            manifest,
        )

    audit_records = _load_jsonl_strict(
        paths["audit"]
    )

    if len(audit_records) > len(selected_events):
        raise ValueError(
            "Resume audit contains more records "
            "than requested TEST events"
        )

    for index, record in enumerate(audit_records):
        expected_event = selected_events[index]

        if record.get("run_fingerprint") != run_fingerprint:
            raise ValueError(
                "Resume audit fingerprint mismatch"
            )

        if record.get("instance_key") != expected_event.key:
            raise ValueError(
                "Resume audit is not the exact expected prefix"
            )

    (
        kbrd_candidate_fn,
        prepare_input_fn,
        title_lookup,
    ) = _load_real_dependencies()

    for event in selected_events[len(audit_records):]:
        conversation = conversation_by_line[
            event.line_number
        ]

        record = _process_event(
            event=event,
            conversation=conversation,
            retriever=retriever,
            run_fingerprint=run_fingerprint,
            kbrd_candidate_fn=kbrd_candidate_fn,
            prepare_input_fn=prepare_input_fn,
            title_lookup=title_lookup,
        )

        _append_jsonl(
            paths["audit"],
            record,
        )

    audit_records = _load_jsonl_strict(
        paths["audit"]
    )

    if len(audit_records) != len(selected_events):
        raise ValueError(
            "Completed TEST audit record count mismatch"
        )

    dataset_records = [
        _stage2_record_from_audit(
            record,
            audit_record_number=index,
        )
        for index, record in enumerate(
            audit_records,
            1,
        )
        if record["category"] != "no_seed"
    ]

    dataset_sha256 = _atomic_jsonl(
        paths["dataset"],
        dataset_records,
    )

    audit_sha256 = sha256_file(
        paths["audit"]
    )

    summary = _summary(
        audit_records,
        dataset_records,
        reconstruction=reconstruction,
        selected_event_count=len(selected_events),
        source=source,
        cache_path=cache_path,
        checkpoint=checkpoint,
        run_fingerprint=run_fingerprint,
        audit_sha256=audit_sha256,
        dataset_sha256=dataset_sha256,
    )

    _atomic_json(
        paths["summary"],
        summary,
    )

    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--test-path",
        type=Path,
        default=DEFAULT_TEST_PATH,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )

    parser.add_argument(
        "--kbrd-checkpoint",
        type=Path,
        default=DEFAULT_KBRD_CHECKPOINT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--max-instances",
        type=int,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser


def main() -> int:
    args = _parser().parse_args()

    result = build_test_dataset(
        test_path=args.test_path,
        cache_dir=args.cache_dir,
        kbrd_checkpoint=args.kbrd_checkpoint,
        output_dir=args.output_dir,
        max_instances=args.max_instances,
        resume=args.resume,
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
