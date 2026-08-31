"""VALID-only local Hugging Face reranking over frozen Stage-1 RRF artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from my_crs.evaluate_rrf_fusion import PROJECT_ROOT, _atomic_json
from my_crs.evaluate_rrf_zeroshot import (
    DEFAULT_RRF_INSTANCES_PATH,
    DEFAULT_RRF_SUMMARY_PATH,
    EXPECTED_CONVERSATIONS,
    EXPECTED_INSTANCES,
    EXPECTED_RRF_METRICS,
    EXPECTED_SOURCE_SHA256,
    METRIC_TOLERANCE,
    OFFICIAL_VALID_PATH,
    PROMPT_VERSION,
    _append_jsonl,
    _assert_metric_sets_equal,
    _file_sha256,
    _metrics_from_records,
    _normalized_title_set,
    _ranked_order,
    _require_resume_field,
    _validated_stored_order,
    build_dialogue_up_to,
    get_rank,
    get_recommended_movies_at_turn,
    instance_key,
    load_frozen_rrf_instances,
    load_resume_records,
    reconstruct_valid_event_index,
    validate_complete_reranked_recall_at_50,
    validate_frozen_instances_against_valid,
    validate_frozen_rrf_summary,
    validate_full_run_accounting,
    validate_official_valid_path,
    validate_output_paths,
    validate_resume_subset,
)
from my_crs.hf_list_reranker import (
    BACKEND_NAME,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    HFGenerationError,
    HFGenerationSettings,
    HFListReranker,
    validate_hf_generation_settings,
)
from my_crs.rrf_list_reranker import (
    FALLBACK_INVALID_MODEL_OUTPUT,
    RankedPositionsError,
    build_list_rerank_prompt,
    complete_ranking,
    parse_ranked_positions,
    prompt_template_digest,
)


EXPERIMENT_NAME = "stage2_hf_rrf_list_reranking"
FALLBACK_GENERATION_FAILURE = "generation_failure"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "rrf_hf_valid.json"
DEFAULT_INSTANCE_OUTPUT_PATH = (
    PROJECT_ROOT / "experiments" / "rrf_hf_valid_instances.jsonl"
)


def hf_run_fingerprint(
    *,
    summary_path: str | Path,
    instances_path: str | Path,
    backend_provenance: Mapping[str, Any],
    generation_provenance: Mapping[str, Any],
) -> str:
    """Hash every scientifically relevant local-HF run input."""
    material = {
        "experiment": EXPERIMENT_NAME,
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": prompt_template_digest(),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "summary_sha256": _file_sha256(summary_path),
        "instances_sha256": _file_sha256(instances_path),
        "backend_provenance": backend_provenance,
        "generation": generation_provenance,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _preflight_resume_jsonl(path: Path) -> None:
    """Reject structurally invalid resume input before loading model weights."""
    if not path.exists():
        return
    seen: set[str] = set()
    observed_fingerprint: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Malformed HF resume JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"HF resume record at line {line_number} must be an object"
                )
            key = record.get("instance_key")
            fingerprint = record.get("run_fingerprint")
            if not isinstance(key, str) or not key:
                raise ValueError(
                    f"HF resume record at line {line_number} lacks an instance key"
                )
            if not isinstance(fingerprint, str) or not fingerprint:
                raise ValueError(
                    f"HF resume record at line {line_number} lacks a run fingerprint"
                )
            if observed_fingerprint is None:
                observed_fingerprint = fingerprint
            elif fingerprint != observed_fingerprint:
                raise ValueError("HF resume contains inconsistent run fingerprints")
            if key in seen:
                raise ValueError(f"Duplicate HF resume instance key: {key}")
            seen.add(key)


def evaluate_hf_instance(
    frozen_record: Mapping[str, Any],
    conversation: Mapping[str, Any],
    backend: HFListReranker,
) -> dict[str, Any]:
    """Generate, strictly parse, complete, and score one frozen VALID instance."""
    line_number = int(frozen_record["line_number"])
    turn_index = int(frozen_record["turn_index"])
    messages = conversation.get("messages", [])
    if turn_index >= len(messages):
        raise ValueError(f"Turn {turn_index} is absent from VALID line {line_number}")
    respondent = conversation.get("respondentWorkerId", -1)
    if messages[turn_index].get("senderWorkerId", -1) != respondent:
        raise ValueError(f"Turn {turn_index} is not a respondent turn")
    if str(conversation.get("conversationId")) != str(frozen_record["conversation_id"]):
        raise ValueError(f"Conversation ID mismatch at VALID line {line_number}")

    ground_truth = get_recommended_movies_at_turn(conversation, turn_index)
    if _normalized_title_set(ground_truth) != _normalized_title_set(
        frozen_record["ground_truth_titles"]
    ):
        raise ValueError(f"Ground-truth mismatch at instance {line_number}:{turn_index}")

    history = build_dialogue_up_to(conversation, turn_index - 1)
    original_candidates = [
        dict(candidate) for candidate in frozen_record["rrf_candidates"]
    ]
    if len(original_candidates) != 50:
        raise ValueError("Frozen Stage-2 instance must contain exactly 50 candidates")
    prompt_messages = build_list_rerank_prompt(history, original_candidates)

    raw_output: str | None
    parsed_positions: list[int]
    fallback: bool
    fallback_reason: str | None
    fallback_detail: str | None
    successful_generations: int
    try:
        raw_output = backend.generate(prompt_messages)
    except HFGenerationError as error:
        raw_output = None
        parsed_positions = []
        fallback = True
        fallback_reason = FALLBACK_GENERATION_FAILURE
        fallback_detail = str(error)
        successful_generations = 0
    else:
        successful_generations = 1
        try:
            parsed_positions = parse_ranked_positions(
                raw_output,
                candidate_count=len(original_candidates),
            )
        except RankedPositionsError as error:
            parsed_positions = []
            fallback = True
            fallback_reason = FALLBACK_INVALID_MODEL_OUTPUT
            fallback_detail = str(error)
        else:
            fallback = False
            fallback_reason = None
            fallback_detail = None

    if fallback:
        final_candidates = [dict(candidate) for candidate in original_candidates]
        selected: list[dict[str, Any]] = []
    else:
        final_candidates = complete_ranking(original_candidates, parsed_positions)
        selected = [original_candidates[position - 1] for position in parsed_positions]

    original_ids = [candidate["id"] for candidate in original_candidates]
    final_ids = [candidate["id"] for candidate in final_candidates]
    if len(final_ids) != 50 or len(set(final_ids)) != 50:
        raise RuntimeError("HF reranker did not produce exactly 50 unique candidates")
    if set(final_ids) != set(original_ids):
        raise RuntimeError("HF reranker changed the frozen candidate set")

    original_rank = get_rank(original_candidates, ground_truth)
    reranked_rank = get_rank(final_candidates, ground_truth)
    original_hit_50 = 0 < original_rank <= 50
    reranked_hit_50 = 0 < reranked_rank <= 50
    if original_hit_50 != reranked_hit_50:
        raise RuntimeError("HF per-instance Recall@50 invariant failed")

    history_hash = hashlib.sha256(history.encode("utf-8")).hexdigest()
    return {
        "instance_key": instance_key(line_number, turn_index),
        "line_number": line_number,
        "conversation_id": frozen_record["conversation_id"],
        "turn_index": turn_index,
        "ground_truth_titles": ground_truth,
        "dialogue_history": history,
        "dialogue_history_sha256": history_hash,
        "original_rrf_candidate_order": _ranked_order(original_candidates),
        "prompt_candidate_positions": _ranked_order(original_candidates),
        "raw_model_output": raw_output,
        "parsed_top10_local_positions": parsed_positions,
        "parsed_canonical_candidate_ids": [
            int(candidate["id"]) for candidate in selected
        ],
        "parsed_candidate_titles": [
            str(candidate.get("title", "")) for candidate in selected
        ],
        "final_complete_top50_order": _ranked_order(final_candidates),
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "fallback_detail": fallback_detail,
        "generation_calls": 1,
        "successful_generations": successful_generations,
        "original_rrf_target_rank": original_rank,
        "reranked_target_rank": reranked_rank,
        "hit_at_1": 0 < reranked_rank <= 1,
        "hit_at_10": 0 < reranked_rank <= 10,
        "hit_at_50": reranked_hit_50,
        "reciprocal_rank": 1.0 / reranked_rank if reranked_rank else 0.0,
    }


def validate_hf_resume_record(
    record: Mapping[str, Any],
    frozen_record: Mapping[str, Any],
    valid_event: Mapping[str, Any],
    conversation: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless a local-HF resume record is self-consistent."""
    line_number = int(frozen_record["line_number"])
    turn_index = int(frozen_record["turn_index"])
    key = instance_key(line_number, turn_index)
    _require_resume_field(record, "instance_key", key)
    _require_resume_field(record, "line_number", line_number)
    _require_resume_field(record, "turn_index", turn_index)
    if str(record.get("conversation_id")) != str(frozen_record["conversation_id"]):
        raise ValueError(f"HF resume conversation ID is inconsistent at {key}")
    if str(record.get("conversation_id")) != str(valid_event.get("conversation_id")):
        raise ValueError(f"HF resume VALID conversation ID is inconsistent at {key}")
    stored_ground_truth = record.get("ground_truth_titles")
    if not isinstance(stored_ground_truth, list) or _normalized_title_set(
        stored_ground_truth
    ) != valid_event["normalized_ground_truth_titles"]:
        raise ValueError(f"HF resume ground truth is inconsistent at {key}")

    history = build_dialogue_up_to(conversation, turn_index - 1)
    expected_history_hash = hashlib.sha256(history.encode("utf-8")).hexdigest()
    _require_resume_field(record, "dialogue_history", history)
    _require_resume_field(record, "dialogue_history_sha256", expected_history_hash)

    original_candidates = [
        dict(candidate) for candidate in frozen_record["rrf_candidates"]
    ]
    expected_original_order = _ranked_order(original_candidates)
    stored_original_order = _validated_stored_order(
        record.get("original_rrf_candidate_order"),
        "original_rrf_candidate_order",
    )
    if stored_original_order != expected_original_order:
        raise ValueError(f"HF resume original candidate order is inconsistent at {key}")
    if record.get("prompt_candidate_positions") != expected_original_order:
        raise ValueError(f"HF resume prompt candidate positions are inconsistent at {key}")

    final_order = _validated_stored_order(
        record.get("final_complete_top50_order"),
        "final_complete_top50_order",
    )
    original_ids = [candidate["id"] for candidate in expected_original_order]
    final_ids = [candidate["id"] for candidate in final_order]
    if set(final_ids) != set(original_ids):
        raise ValueError(f"HF resume changed the candidate set at {key}")

    fallback = record.get("fallback")
    if type(fallback) is not bool:
        raise ValueError(f"HF resume fallback flag is invalid at {key}")
    fallback_reason = record.get("fallback_reason")
    fallback_detail = record.get("fallback_detail")
    raw_output = record.get("raw_model_output")
    positions = record.get("parsed_top10_local_positions")
    parsed_ids = record.get("parsed_canonical_candidate_ids")
    parsed_titles = record.get("parsed_candidate_titles")

    if fallback:
        if fallback_reason not in {
            FALLBACK_INVALID_MODEL_OUTPUT,
            FALLBACK_GENERATION_FAILURE,
        }:
            raise ValueError(f"HF resume fallback reason is unknown at {key}")
        if positions != [] or parsed_ids != [] or parsed_titles != []:
            raise ValueError(f"HF resume fallback parsed fields are inconsistent at {key}")
        if final_order != expected_original_order:
            raise ValueError(f"HF resume fallback changed candidate order at {key}")
        if fallback_reason == FALLBACK_INVALID_MODEL_OUTPUT:
            if not isinstance(raw_output, str) or not isinstance(fallback_detail, str):
                raise ValueError(f"HF resume invalid output lacks model content at {key}")
            try:
                parse_ranked_positions(raw_output, candidate_count=50)
            except RankedPositionsError as error:
                if fallback_detail != str(error):
                    raise ValueError(
                        f"HF resume invalid-output detail is inconsistent at {key}"
                    ) from error
            else:
                raise ValueError(f"HF resume fallback contradicts model output at {key}")
        elif raw_output is not None or not isinstance(fallback_detail, str):
            raise ValueError(f"HF resume generation-failure provenance is invalid at {key}")
    else:
        if fallback_reason is not None or fallback_detail is not None:
            raise ValueError(f"HF resume non-fallback provenance is inconsistent at {key}")
        if not isinstance(raw_output, str) or not isinstance(positions, list):
            raise ValueError(f"HF resume successful output is invalid at {key}")
        try:
            validated_positions = parse_ranked_positions(
                raw_output,
                candidate_count=50,
            )
        except RankedPositionsError as error:
            raise ValueError(f"HF resume raw output is invalid at {key}: {error}") from error
        if positions != validated_positions:
            raise ValueError(f"HF resume parsed positions are inconsistent at {key}")
        selected = [original_candidates[position - 1] for position in positions]
        expected_ids = [int(candidate["id"]) for candidate in selected]
        expected_titles = [str(candidate.get("title", "")) for candidate in selected]
        if parsed_ids != expected_ids or parsed_titles != expected_titles:
            raise ValueError(f"HF resume parsed candidate mapping is inconsistent at {key}")
        expected_final_order = _ranked_order(
            complete_ranking(original_candidates, positions)
        )
        if final_order != expected_final_order:
            raise ValueError(f"HF resume completion rule is inconsistent at {key}")

    calls = record.get("generation_calls")
    successes = record.get("successful_generations")
    if calls != 1 or type(calls) is not int:
        raise ValueError(f"HF resume generation call count is invalid at {key}")
    if type(successes) is not int or successes not in (0, 1):
        raise ValueError(f"HF resume successful generation count is invalid at {key}")
    expected_successes = (
        0
        if fallback and fallback_reason == FALLBACK_GENERATION_FAILURE
        else 1
    )
    if successes != expected_successes:
        raise ValueError(f"HF resume generation provenance is inconsistent at {key}")

    ground_truth = valid_event["ground_truth_titles"]
    final_candidates = [
        {"id": candidate["id"], "title": candidate["title"]}
        for candidate in final_order
    ]
    original_rank = get_rank(original_candidates, ground_truth)
    reranked_rank = get_rank(final_candidates, ground_truth)
    expected_metrics = {
        "original_rrf_target_rank": original_rank,
        "reranked_target_rank": reranked_rank,
        "hit_at_1": 0 < reranked_rank <= 1,
        "hit_at_10": 0 < reranked_rank <= 10,
        "hit_at_50": 0 < reranked_rank <= 50,
    }
    for field, expected in expected_metrics.items():
        _require_resume_field(record, field, expected)
    expected_reciprocal_rank = 1.0 / reranked_rank if reranked_rank else 0.0
    reciprocal_rank = record.get("reciprocal_rank")
    if type(reciprocal_rank) not in (int, float) or abs(
        float(reciprocal_rank) - expected_reciprocal_rank
    ) > METRIC_TOLERANCE:
        raise ValueError(f"HF resume reciprocal rank is inconsistent at {key}")
    if (0 < original_rank <= 50) != (0 < reranked_rank <= 50):
        raise ValueError(f"HF resume Recall@50 invariant failed at {key}")
    return dict(record)


def evaluate_rrf_hf(
    *,
    rrf_summary_path: str | Path = DEFAULT_RRF_SUMMARY_PATH,
    rrf_instances_path: str | Path = DEFAULT_RRF_INSTANCES_PATH,
    valid_path: str | Path = OFFICIAL_VALID_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    instance_output_path: str | Path = DEFAULT_INSTANCE_OUTPUT_PATH,
    settings: HFGenerationSettings | None = None,
    max_instances: int | None = None,
    resume: bool = False,
    backend: HFListReranker | None = None,
) -> dict[str, Any]:
    """Evaluate one loaded local HF model without recomputing Stage-1 retrieval."""
    if max_instances is not None and max_instances < 0:
        raise ValueError("max_instances must be non-negative")
    active_settings = settings or HFGenerationSettings()
    validate_hf_generation_settings(active_settings)
    source_path = validate_official_valid_path(valid_path)
    (
        source_path,
        summary_path,
        instances_path,
        result_path,
        instance_path,
    ) = validate_output_paths(
        valid_path=source_path,
        rrf_summary_path=rrf_summary_path,
        rrf_instances_path=rrf_instances_path,
        output_path=output_path,
        instance_output_path=instance_output_path,
    )
    if not resume and instance_path.exists():
        raise FileExistsError(
            f"Instance output already exists; use --resume or a new path: {instance_path}"
        )
    if resume:
        _preflight_resume_jsonl(instance_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        frozen_summary = json.load(handle)
    validate_frozen_rrf_summary(frozen_summary)
    frozen_instances = load_frozen_rrf_instances(instances_path)
    valid_index = reconstruct_valid_event_index(source_path)
    frozen_by_key = validate_frozen_instances_against_valid(
        frozen_instances,
        valid_index,
    )
    conversations = valid_index.conversations

    local_backend = backend or HFListReranker(active_settings)
    backend_provenance = local_backend.provenance()
    generation_provenance = local_backend.generation_provenance()
    fingerprint = hf_run_fingerprint(
        summary_path=summary_path,
        instances_path=instances_path,
        backend_provenance=backend_provenance,
        generation_provenance=generation_provenance,
    )

    target_total = (
        len(frozen_instances)
        if max_instances is None
        else min(max_instances, len(frozen_instances))
    )
    intended_records = frozen_instances[:target_total]
    intended_keys = [
        instance_key(record["line_number"], record["turn_index"])
        for record in intended_records
    ]
    if resume:
        loaded_completed = load_resume_records(
            instance_path,
            expected_fingerprint=fingerprint,
        )
    else:
        instance_path.parent.mkdir(parents=True, exist_ok=True)
        instance_path.open("x", encoding="utf-8").close()
        loaded_completed = {}
    unknown_completed = set(loaded_completed) - set(frozen_by_key)
    if unknown_completed:
        raise ValueError(
            f"HF resume contains unknown instance keys: {sorted(unknown_completed)}"
        )
    completed_in_frozen_order = validate_resume_subset(
        set(loaded_completed),
        intended_keys,
    )

    completed: dict[str, dict[str, Any]] = {}
    for key in completed_in_frozen_order:
        frozen_record = frozen_by_key[key]
        line_number = int(frozen_record["line_number"])
        conversation = conversations.get(line_number)
        if conversation is None:
            raise ValueError(f"VALID line {line_number} is absent for HF resume {key}")
        completed[key] = validate_hf_resume_record(
            loaded_completed[key],
            frozen_record,
            valid_index.events[key],
            conversation,
        )

    for frozen_record in intended_records:
        if len(completed) >= target_total:
            break
        key = instance_key(frozen_record["line_number"], frozen_record["turn_index"])
        if key in completed:
            continue
        line_number = int(frozen_record["line_number"])
        conversation = conversations.get(line_number)
        if conversation is None:
            raise ValueError(f"VALID line {line_number} is absent")
        evaluated = evaluate_hf_instance(
            frozen_record,
            conversation,
            local_backend,
        )
        evaluated["run_fingerprint"] = fingerprint
        _append_jsonl(instance_path, evaluated)
        completed[key] = evaluated

    ordered_completed = [completed[key] for key in intended_keys if key in completed]
    original_subset_metrics = _metrics_from_records(
        ordered_completed,
        "original_rrf_target_rank",
    )
    reranked_metrics = _metrics_from_records(
        ordered_completed,
        "reranked_target_rank",
    )
    if abs(
        original_subset_metrics["Recall@50"] - reranked_metrics["Recall@50"]
    ) > METRIC_TOLERANCE:
        raise RuntimeError("HF reranked Recall@50 changed on processed candidate sets")

    processed_instances = len(ordered_completed)
    processed_conversations = len(
        {int(record["line_number"]) for record in ordered_completed}
    )
    full_universe_requested = target_total == EXPECTED_INSTANCES
    complete = validate_full_run_accounting(
        processed_instances=processed_instances,
        processed_conversations=processed_conversations,
        full_universe_requested=full_universe_requested,
    )
    if complete:
        _assert_metric_sets_equal(
            original_subset_metrics,
            EXPECTED_RRF_METRICS,
            context="Frozen full-run RRF",
        )
    validate_complete_reranked_recall_at_50(
        reranked_metrics,
        complete=complete,
    )

    fallback_reasons = Counter(
        str(record["fallback_reason"])
        for record in ordered_completed
        if record.get("fallback")
    )
    model_provenance = backend_provenance.get("model", {})
    result = {
        "experiment": EXPERIMENT_NAME,
        "source_split": "VALID",
        "source_path": str(source_path),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "complete": complete,
        "expected_conversations": EXPECTED_CONVERSATIONS,
        "expected_instances": EXPECTED_INSTANCES,
        "processed_conversations": processed_conversations,
        "processed_instances": processed_instances,
        "max_instances": max_instances,
        "resume": resume,
        "run_fingerprint": fingerprint,
        "model": model_provenance.get("model_id"),
        "backend": backend_provenance.get("backend", BACKEND_NAME),
        "model_provenance": model_provenance,
        "tokenizer_provenance": backend_provenance.get("tokenizer", {}),
        "runtime": backend_provenance.get("runtime", {}),
        "generation": generation_provenance,
        "prompt": {
            "version": PROMPT_VERSION,
            "template_sha256": prompt_template_digest(),
        },
        "adapter": backend_provenance.get("adapter", {"enabled": False}),
        "frozen_rrf_artifacts": {
            "summary_path": str(summary_path),
            "summary_sha256": _file_sha256(summary_path),
            "instances_path": str(instances_path),
            "instances_sha256": _file_sha256(instances_path),
        },
        "frozen_rrf_metrics": dict(EXPECTED_RRF_METRICS),
        "frozen_stage1_configuration": {
            "extraction_configuration": frozen_summary["extraction_configuration"],
            "kbrd_configuration": frozen_summary["kbrd_configuration"],
            "ckg_configuration": frozen_summary["ckg_configuration"],
            "rrf_parameters": frozen_summary["rrf_parameters"],
        },
        "processed_subset_original_rrf_metrics": original_subset_metrics,
        "reranked_metrics": reranked_metrics,
        "recall_at_50_invariant_passed": True,
        "generations": {
            "calls": sum(record["generation_calls"] for record in ordered_completed),
            "successful": sum(
                record["successful_generations"] for record in ordered_completed
            ),
        },
        "fallbacks": {
            "count": sum(bool(record.get("fallback")) for record in ordered_completed),
            "reasons": dict(sorted(fallback_reasons.items())),
        },
        "instance_provenance_path": str(instance_path),
        "failures": [],
    }
    _atomic_json(result_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rrf-summary", type=Path, default=DEFAULT_RRF_SUMMARY_PATH)
    parser.add_argument(
        "--rrf-instances",
        type=Path,
        default=DEFAULT_RRF_INSTANCES_PATH,
    )
    parser.add_argument("--valid-path", type=Path, default=OFFICIAL_VALID_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--instance-output-path",
        type=Path,
        default=DEFAULT_INSTANCE_OUTPUT_PATH,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--adapter-path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = HFGenerationSettings(
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        adapter_path=(str(args.adapter_path) if args.adapter_path else None),
    )
    result = evaluate_rrf_hf(
        rrf_summary_path=args.rrf_summary,
        rrf_instances_path=args.rrf_instances,
        valid_path=args.valid_path,
        output_path=args.output_path,
        instance_output_path=args.instance_output_path,
        settings=settings,
        max_instances=args.max_instances,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
