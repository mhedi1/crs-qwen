"""VALID-only zero-shot list reranking over frozen Stage-1 RRF artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from my_crs.evaluate_rrf_fusion import (
    DEFAULT_VALID_PATH,
    PROJECT_ROOT,
    _atomic_json,
    _sha256,
)
from my_crs.rrf_list_reranker import (
    FALLBACK_EMPTY_API_RESPONSE,
    FALLBACK_INVALID_CANDIDATE_COUNT,
    FALLBACK_INVALID_MODEL_OUTPUT,
    FALLBACK_MALFORMED_API_RESPONSE,
    FALLBACK_REQUEST_FAILURE,
    PROMPT_VERSION,
    QwenRerankSettings,
    RankedPositionsError,
    complete_ranking,
    parse_ranked_positions,
    prompt_template_digest,
    rerank_rrf_candidates,
)


EXPECTED_SOURCE_SHA256 = "c8b7ba32d95a85330eb6c129b65916fc7032515ed110d02bdcb0e41bd482a5d7"
EXPECTED_CONVERSATIONS = 797
EXPECTED_INSTANCES = 2588
EXPECTED_RRF_METRICS = {
    "Recall@1": 0.04095826893353941,
    "Recall@10": 0.22372488408037094,
    "Recall@50": 0.4385625965996909,
    "MRR": 0.0969580444098772,
}
METRIC_TOLERANCE = 1e-12
OFFICIAL_VALID_PATH = DEFAULT_VALID_PATH
DEFAULT_RRF_SUMMARY_PATH = PROJECT_ROOT / "experiments" / "rrf_valid_k60_equal_clean.json"
DEFAULT_RRF_INSTANCES_PATH = (
    PROJECT_ROOT / "experiments" / "rrf_valid_k60_equal_clean_instances.jsonl"
)
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "rrf_zeroshot_valid.json"
DEFAULT_INSTANCE_OUTPUT_PATH = (
    PROJECT_ROOT / "experiments" / "rrf_zeroshot_valid_instances.jsonl"
)


def normalize_title(title: str) -> str:
    """Frozen evaluator normalization: remove year/punctuation and lowercase."""
    title = re.sub(r"\(\d{4}\)", "", title)
    title = re.sub(r"[^\w\s]", "", title)
    title = re.sub(r"\s+", " ", title)
    return title.lower().strip()


def strict_title_match(title_a: str, title_b: str) -> bool:
    return normalize_title(title_a) == normalize_title(title_b)


def is_hit(
    candidates: Sequence[Mapping[str, Any]], ground_truth: Sequence[str], cutoff: int
) -> bool:
    return any(
        strict_title_match(str(candidate.get("title", "")), str(gold))
        for candidate in candidates[:cutoff]
        for gold in ground_truth
    )


def get_rank(
    candidates: Sequence[Mapping[str, Any]], ground_truth: Sequence[str]
) -> int:
    for rank, candidate in enumerate(candidates, start=1):
        if any(
            strict_title_match(str(candidate.get("title", "")), str(gold))
            for gold in ground_truth
        ):
            return rank
    return 0


def build_dialogue_up_to(conversation: Mapping[str, Any], turn_index: int) -> str:
    """Mirror the frozen evaluator's dialogue representation exactly."""
    movie_mentions = conversation.get("movieMentions", {})
    messages = conversation.get("messages", [])
    initiator = conversation.get("initiatorWorkerId", -1)
    turns: list[str] = []
    for index, message in enumerate(messages):
        if index > turn_index:
            break
        role = "User" if message.get("senderWorkerId") == initiator else "System"
        text = message.get("text", "").strip()
        for movie_id, movie_name in movie_mentions.items():
            text = text.replace(f"@{movie_id}", str(movie_name).strip())
        text = text.replace("&quot;", '"').replace("&amp;", "&")
        if text:
            turns.append(f"{role}: {text}")
    return "\n".join(turns)


def get_recommended_movies_at_turn(
    conversation: Mapping[str, Any], turn_index: int
) -> list[str]:
    """Mirror the frozen evaluator's ReDial target extraction."""
    messages = conversation.get("messages", [])
    if turn_index >= len(messages):
        return []
    text = messages[turn_index].get("text", "")
    movie_mentions = conversation.get("movieMentions", {})
    respondent_questions = conversation.get("respondentQuestions", {})
    if isinstance(respondent_questions, list):
        respondent_questions = {}
    recommended: list[str] = []
    for movie_id in re.findall(r"@(\d+)", text):
        if respondent_questions.get(movie_id, {}).get("suggested", 0) != 1:
            continue
        movie_name = movie_mentions.get(movie_id, "")
        if movie_name:
            recommended.append(str(movie_name).strip().lower())
    return recommended


def _normalized_title_set(titles: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted({normalize_title(str(title)) for title in titles}))


@dataclass(frozen=True)
class ValidEventIndex:
    conversations: dict[int, dict[str, Any]]
    events: dict[str, dict[str, Any]]
    evaluated_conversations: int


def reconstruct_valid_event_index(path: str | Path) -> ValidEventIndex:
    """Independently reconstruct every evaluable recommendation event in VALID."""
    conversations = _load_valid_conversations(Path(path).resolve())
    events: dict[str, dict[str, Any]] = {}
    evaluated_conversations = 0
    for line_number, conversation in conversations.items():
        respondent = conversation.get("respondentWorkerId", -1)
        conversation_has_event = False
        for turn_index, message in enumerate(conversation.get("messages", [])):
            if message.get("senderWorkerId", -1) != respondent:
                continue
            ground_truth = get_recommended_movies_at_turn(conversation, turn_index)
            if not ground_truth:
                continue
            key = instance_key(line_number, turn_index)
            if key in events:
                raise ValueError(f"Duplicate independently reconstructed VALID key {key}")
            events[key] = {
                "instance_key": key,
                "line_number": line_number,
                "conversation_id": conversation.get("conversationId"),
                "turn_index": turn_index,
                "ground_truth_titles": list(ground_truth),
                "normalized_ground_truth_titles": _normalized_title_set(ground_truth),
            }
            conversation_has_event = True
        evaluated_conversations += int(conversation_has_event)
    return ValidEventIndex(
        conversations=conversations,
        events=events,
        evaluated_conversations=evaluated_conversations,
    )


class RankingMetrics:
    def __init__(self) -> None:
        self.instances = 0
        self.hits = {1: 0, 10: 0, 50: 0}
        self.reciprocal_rank_sum = 0.0

    def add_rank(self, rank: int) -> None:
        self.instances += 1
        for cutoff in self.hits:
            self.hits[cutoff] += int(0 < rank <= cutoff)
        self.reciprocal_rank_sum += 1.0 / rank if rank else 0.0

    def result(self) -> dict[str, Any]:
        count = self.instances
        return {
            "instances": count,
            "Recall@1": self.hits[1] / count if count else 0.0,
            "Recall@10": self.hits[10] / count if count else 0.0,
            "Recall@50": self.hits[50] / count if count else 0.0,
            "MRR": self.reciprocal_rank_sum / count if count else 0.0,
        }


def validate_official_valid_path(path: str | Path) -> Path:
    """Reject every path except this checkout's official ReDial VALID file."""
    candidate = Path(path).resolve()
    if candidate.name != "valid_data.jsonl":
        raise ValueError(f"Stage-2 evaluation is VALID-only, got {candidate.name!r}")
    official = Path(OFFICIAL_VALID_PATH).resolve()
    if candidate != official:
        raise ValueError(f"Expected official ReDial VALID path {official}, got {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    observed_sha = _sha256(candidate)
    if observed_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"VALID SHA mismatch: expected {EXPECTED_SOURCE_SHA256}, got {observed_sha}"
        )
    return candidate


def _require_equal(observed: Any, expected: Any, field: str) -> None:
    if observed != expected:
        raise ValueError(f"Frozen RRF provenance mismatch for {field}: {observed!r} != {expected!r}")


def _require_metric(observed: Any, expected: float, field: str) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Frozen RRF metric {field} is not numeric: {observed!r}") from error
    if abs(value - expected) > METRIC_TOLERANCE:
        raise ValueError(
            f"Frozen RRF metric mismatch for {field}: {value!r} != {expected!r}"
        )


def validate_frozen_rrf_summary(summary: Mapping[str, Any]) -> None:
    """Fail closed unless every frozen Stage-1 configuration invariant matches."""
    _require_equal(summary.get("source_split"), "VALID", "source_split")
    _require_equal(summary.get("source_sha256"), EXPECTED_SOURCE_SHA256, "source_sha256")
    _require_equal(
        summary.get("evaluated_conversations"),
        EXPECTED_CONVERSATIONS,
        "evaluated_conversations",
    )
    _require_equal(
        summary.get("evaluation_instances"), EXPECTED_INSTANCES, "evaluation_instances"
    )
    _require_equal(summary.get("failures"), [], "failures")

    extraction = summary.get("extraction_configuration", {})
    expected_extraction = {
        "resolver_version": "v3",
        "use_legacy_non_movie_entities": True,
        "use_aux_dbpedia_uri_matching": True,
        "use_aux_genre_mapping": True,
        "use_aux_person_matching": False,
        "seed_selection": "all",
    }
    for key, expected in expected_extraction.items():
        _require_equal(extraction.get(key), expected, f"extraction_configuration.{key}")

    kbrd = summary.get("kbrd_configuration", {})
    for key, expected in {
        "retrieval_mode": "kbrd",
        "top_k": 50,
        "use_fusion": False,
        "llm_qwen_used": False,
    }.items():
        _require_equal(kbrd.get(key), expected, f"kbrd_configuration.{key}")

    ckg = summary.get("ckg_configuration", {})
    for key, expected in {
        "graph_type": "conversation",
        "weighting_method": "conditional",
        "min_support": 2,
        "view": "budget_controlled",
        "top_k": 50,
    }.items():
        _require_equal(ckg.get(key), expected, f"ckg_configuration.{key}")

    rrf = summary.get("rrf_parameters", {})
    for key, expected in {
        "k": 60,
        "absent_source_contribution": 0.0,
        "raw_scores_used": False,
        "final_candidate_budget": 50,
    }.items():
        _require_equal(rrf.get(key), expected, f"rrf_parameters.{key}")
    weights = rrf.get("weights", {})
    _require_equal(weights.get("KBRD"), 1.0, "rrf_parameters.weights.KBRD")
    _require_equal(weights.get("CKG"), 1.0, "rrf_parameters.weights.CKG")

    metrics = summary.get("metrics", {}).get("RRF", {})
    for name, expected in EXPECTED_RRF_METRICS.items():
        _require_metric(metrics.get(name), expected, f"metrics.RRF.{name}")


def _validate_frozen_instance(record: Mapping[str, Any], record_number: int) -> None:
    prefix = f"frozen instance record {record_number}"
    if type(record.get("line_number")) is not int or record["line_number"] < 1:
        raise ValueError(f"{prefix} has invalid line_number")
    if type(record.get("turn_index")) is not int or record["turn_index"] < 0:
        raise ValueError(f"{prefix} has invalid turn_index")
    if record.get("conversation_id") is None:
        raise ValueError(f"{prefix} is missing conversation_id")
    ground_truth = record.get("ground_truth_titles")
    if not isinstance(ground_truth, list) or not all(
        isinstance(title, str) for title in ground_truth
    ) or not ground_truth:
        raise ValueError(f"{prefix} has invalid ground_truth_titles")
    candidates = record.get("rrf_candidates")
    if not isinstance(candidates, list) or len(candidates) != 50:
        raise ValueError(f"{prefix} must contain exactly 50 RRF candidates")
    candidate_ids: list[int] = []
    for candidate in candidates:
        if not isinstance(candidate, dict) or type(candidate.get("id")) is not int:
            raise ValueError(f"{prefix} contains a candidate without an integer ID")
        if not isinstance(candidate.get("title"), str) or not candidate["title"].strip():
            raise ValueError(f"{prefix} contains a candidate without a title")
        candidate_ids.append(candidate["id"])
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError(f"{prefix} contains duplicate canonical candidate IDs")


def load_frozen_rrf_instances(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    with Path(path).resolve().open("r", encoding="utf-8") as handle:
        for record_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Malformed frozen instance JSONL at record {record_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Frozen instance record {record_number} is not an object")
            _validate_frozen_instance(record, record_number)
            key = instance_key(record["line_number"], record["turn_index"])
            if key in seen_keys:
                raise ValueError(f"Duplicate frozen instance key {key}")
            seen_keys.add(key)
            records.append(record)
    if len(records) != EXPECTED_INSTANCES:
        raise ValueError(
            f"Frozen instance count mismatch: {len(records)} != {EXPECTED_INSTANCES}"
        )
    return records


def _load_valid_conversations(path: Path) -> dict[int, dict[str, Any]]:
    conversations: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                conversations[line_number] = json.loads(line)
    return conversations


def validate_frozen_instances_against_valid(
    frozen_instances: Sequence[Mapping[str, Any]],
    valid_index: ValidEventIndex,
    *,
    expected_conversations: int = EXPECTED_CONVERSATIONS,
    expected_instances: int = EXPECTED_INSTANCES,
) -> dict[str, Mapping[str, Any]]:
    """Require exact semantic alignment between frozen RRF records and VALID."""
    if valid_index.evaluated_conversations != expected_conversations:
        raise ValueError(
            "Independently reconstructed VALID conversation count mismatch: "
            f"{valid_index.evaluated_conversations} != {expected_conversations}"
        )
    if len(valid_index.events) != expected_instances:
        raise ValueError(
            "Independently reconstructed VALID event count mismatch: "
            f"{len(valid_index.events)} != {expected_instances}"
        )

    artifact_by_key: dict[str, Mapping[str, Any]] = {}
    for record in frozen_instances:
        key = instance_key(int(record["line_number"]), int(record["turn_index"]))
        if key in artifact_by_key:
            raise ValueError(f"Duplicate frozen instance key {key}")
        artifact_by_key[key] = record
    expected_keys = set(valid_index.events)
    artifact_keys = set(artifact_by_key)
    missing = sorted(expected_keys - artifact_keys)
    unexpected = sorted(artifact_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "Frozen artifact/VALID event-key mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(artifact_by_key) != expected_instances:
        raise ValueError(
            f"Frozen artifact event count mismatch: {len(artifact_by_key)} != "
            f"{expected_instances}"
        )

    for key in sorted(expected_keys):
        expected = valid_index.events[key]
        artifact = artifact_by_key[key]
        artifact_ground_truth = artifact.get("ground_truth_titles")
        if not isinstance(artifact_ground_truth, list) or not artifact_ground_truth:
            raise ValueError(f"Frozen artifact event {key} has empty ground truth")
        if _normalized_title_set(artifact_ground_truth) != expected[
            "normalized_ground_truth_titles"
        ]:
            raise ValueError(f"Frozen artifact/VALID ground-truth mismatch at {key}")
        expected_conversation_id = expected.get("conversation_id")
        artifact_conversation_id = artifact.get("conversation_id")
        if expected_conversation_id is not None and str(artifact_conversation_id) != str(
            expected_conversation_id
        ):
            raise ValueError(f"Frozen artifact/VALID conversation ID mismatch at {key}")
        if int(artifact["line_number"]) != expected["line_number"] or int(
            artifact["turn_index"]
        ) != expected["turn_index"]:
            raise ValueError(f"Frozen artifact/VALID identity mismatch at {key}")
    return artifact_by_key


def instance_key(line_number: int, turn_index: int) -> str:
    return f"{line_number}:{turn_index}"


def _ranked_order(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "position": position,
            "id": int(candidate["id"]),
            "title": str(candidate.get("title", "")),
        }
        for position, candidate in enumerate(candidates, start=1)
    ]


def evaluate_instance(
    frozen_record: Mapping[str, Any],
    conversation: Mapping[str, Any],
    settings: QwenRerankSettings,
    *,
    post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Rerank and score one provenance-aligned VALID instance."""
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
    original_candidates = [dict(candidate) for candidate in frozen_record["rrf_candidates"]]

    if post is None:
        rerank_result = rerank_rrf_candidates(history, original_candidates, settings)
    else:
        rerank_result = rerank_rrf_candidates(
            history, original_candidates, settings, post=post
        )
    final_candidates = rerank_result.final_candidates
    original_ids = [candidate["id"] for candidate in original_candidates]
    final_ids = [candidate["id"] for candidate in final_candidates]
    if len(final_ids) != 50 or len(set(final_ids)) != 50:
        raise RuntimeError("Reranker did not produce exactly 50 unique candidates")
    if set(final_ids) != set(original_ids):
        raise RuntimeError("Reranker changed the frozen candidate set")

    original_rank = get_rank(original_candidates, ground_truth)
    reranked_rank = get_rank(final_candidates, ground_truth)
    original_hit_50 = 0 < original_rank <= 50
    reranked_hit_50 = 0 < reranked_rank <= 50
    if original_hit_50 != reranked_hit_50:
        raise RuntimeError("Per-instance Recall@50 invariant failed")

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
        "raw_qwen_output": rerank_result.raw_output,
        "parsed_top10_local_positions": rerank_result.ranked_positions,
        "parsed_canonical_candidate_ids": rerank_result.ranked_candidate_ids,
        "parsed_candidate_titles": rerank_result.ranked_candidate_titles,
        "final_complete_top50_order": _ranked_order(final_candidates),
        "fallback": rerank_result.fallback,
        "fallback_reason": rerank_result.fallback_reason,
        "fallback_detail": rerank_result.fallback_detail,
        "request_attempts": rerank_result.request_attempts,
        "successful_requests": rerank_result.successful_requests,
        "original_rrf_target_rank": original_rank,
        "reranked_target_rank": reranked_rank,
        "hit_at_1": 0 < reranked_rank <= 1,
        "hit_at_10": 0 < reranked_rank <= 10,
        "hit_at_50": reranked_hit_50,
        "reciprocal_rank": 1.0 / reranked_rank if reranked_rank else 0.0,
    }


def _file_sha256(path: str | Path) -> str:
    return _sha256(Path(path).resolve())


def run_fingerprint(
    *,
    summary_path: str | Path,
    instances_path: str | Path,
    settings: QwenRerankSettings,
) -> str:
    material = {
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": prompt_template_digest(),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "summary_sha256": _file_sha256(summary_path),
        "instances_sha256": _file_sha256(instances_path),
        "qwen": settings.provenance(),
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_output_paths(
    *,
    valid_path: str | Path,
    rrf_summary_path: str | Path,
    rrf_instances_path: str | Path,
    output_path: str | Path,
    instance_output_path: str | Path,
) -> tuple[Path, Path, Path, Path, Path]:
    resolved = {
        "VALID dataset": Path(valid_path).resolve(),
        "frozen RRF summary": Path(rrf_summary_path).resolve(),
        "frozen RRF instances": Path(rrf_instances_path).resolve(),
        "summary output": Path(output_path).resolve(),
        "instance output": Path(instance_output_path).resolve(),
    }
    for output_name in ("summary output", "instance output"):
        for protected_name in (
            "VALID dataset",
            "frozen RRF summary",
            "frozen RRF instances",
        ):
            if resolved[output_name] == resolved[protected_name]:
                raise ValueError(
                    f"Path collision: {output_name} equals {protected_name}: "
                    f"{resolved[output_name]}"
                )
    if resolved["summary output"] == resolved["instance output"]:
        raise ValueError(
            "Path collision: summary output equals instance output: "
            f"{resolved['summary output']}"
        )
    return (
        resolved["VALID dataset"],
        resolved["frozen RRF summary"],
        resolved["frozen RRF instances"],
        resolved["summary output"],
        resolved["instance output"],
    )


def load_resume_records(
    path: str | Path,
    *,
    expected_fingerprint: str,
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    output_path = Path(path)
    if not output_path.exists():
        return completed
    with output_path.open("r", encoding="utf-8") as handle:
        for record_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Malformed resume JSONL record {record_number}") from error
            key = record.get("instance_key")
            if not isinstance(key, str):
                raise ValueError(f"Resume record {record_number} is missing instance_key")
            if record.get("run_fingerprint") != expected_fingerprint:
                raise ValueError(f"Resume fingerprint mismatch at record {record_number}")
            if key in completed:
                raise ValueError(f"Duplicate resume instance key {key}")
            completed[key] = record
    return completed


def validate_resume_subset(
    completed_keys: set[str],
    intended_keys: Sequence[str],
) -> list[str]:
    """Require resume records to be an exact prefix inside the requested subset."""
    intended_key_set = set(intended_keys)
    outside_requested_subset = completed_keys - intended_key_set
    if outside_requested_subset:
        raise ValueError(
            "Resume contains records outside the requested --max-instances subset: "
            f"{sorted(outside_requested_subset)}"
        )
    completed_in_frozen_order = [key for key in intended_keys if key in completed_keys]
    if completed_in_frozen_order != list(intended_keys[: len(completed_keys)]):
        raise ValueError("Resume records must form an exact prefix of the requested subset")
    return completed_in_frozen_order


def _validated_stored_order(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != 50:
        raise ValueError(f"Resume record {field} must contain exactly 50 candidates")
    validated: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for expected_position, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Resume record {field} contains a non-object candidate")
        if set(item) != {"position", "id", "title"}:
            raise ValueError(f"Resume record {field} has an invalid candidate schema")
        if item.get("position") != expected_position:
            raise ValueError(f"Resume record {field} has invalid positions")
        if type(item.get("id")) is not int or not isinstance(item.get("title"), str):
            raise ValueError(f"Resume record {field} has an invalid candidate")
        if item["id"] in seen_ids:
            raise ValueError(f"Resume record {field} contains duplicate candidate IDs")
        seen_ids.add(item["id"])
        validated.append(dict(item))
    return validated


def _require_resume_field(record: Mapping[str, Any], field: str, expected: Any) -> None:
    observed = record.get(field)
    if observed != expected or type(observed) is not type(expected):
        raise ValueError(
            f"Resume record field {field} is inconsistent: {observed!r} != "
            f"{expected!r}"
        )


def validate_resume_record(
    record: Mapping[str, Any],
    frozen_record: Mapping[str, Any],
    valid_event: Mapping[str, Any],
    conversation: Mapping[str, Any],
    settings: QwenRerankSettings,
) -> dict[str, Any]:
    """Fail closed unless a completed record is semantically self-consistent."""
    line_number = int(frozen_record["line_number"])
    turn_index = int(frozen_record["turn_index"])
    key = instance_key(line_number, turn_index)
    _require_resume_field(record, "instance_key", key)
    _require_resume_field(record, "line_number", line_number)
    _require_resume_field(record, "turn_index", turn_index)
    if str(record.get("conversation_id")) != str(frozen_record["conversation_id"]):
        raise ValueError(f"Resume record conversation ID is inconsistent at {key}")
    if str(record.get("conversation_id")) != str(valid_event.get("conversation_id")):
        raise ValueError(f"Resume record VALID conversation ID is inconsistent at {key}")
    stored_ground_truth = record.get("ground_truth_titles")
    if not isinstance(stored_ground_truth, list) or _normalized_title_set(
        stored_ground_truth
    ) != valid_event["normalized_ground_truth_titles"]:
        raise ValueError(f"Resume record ground truth is inconsistent at {key}")

    history = build_dialogue_up_to(conversation, turn_index - 1)
    expected_history_hash = hashlib.sha256(history.encode("utf-8")).hexdigest()
    _require_resume_field(record, "dialogue_history", history)
    _require_resume_field(record, "dialogue_history_sha256", expected_history_hash)

    original_candidates = [dict(candidate) for candidate in frozen_record["rrf_candidates"]]
    expected_original_order = _ranked_order(original_candidates)
    stored_original_order = _validated_stored_order(
        record.get("original_rrf_candidate_order"),
        "original_rrf_candidate_order",
    )
    if stored_original_order != expected_original_order:
        raise ValueError(f"Resume record original candidate order is inconsistent at {key}")
    if record.get("prompt_candidate_positions") != expected_original_order:
        raise ValueError(f"Resume record prompt candidate positions are inconsistent at {key}")

    final_order = _validated_stored_order(
        record.get("final_complete_top50_order"),
        "final_complete_top50_order",
    )
    original_ids = [candidate["id"] for candidate in expected_original_order]
    final_ids = [candidate["id"] for candidate in final_order]
    if set(final_ids) != set(original_ids):
        raise ValueError(f"Resume record changed the candidate set at {key}")

    fallback = record.get("fallback")
    if type(fallback) is not bool:
        raise ValueError(f"Resume record fallback flag is invalid at {key}")
    fallback_reason = record.get("fallback_reason")
    fallback_detail = record.get("fallback_detail")
    positions = record.get("parsed_top10_local_positions")
    parsed_ids = record.get("parsed_canonical_candidate_ids")
    parsed_titles = record.get("parsed_candidate_titles")
    if fallback:
        allowed_fallbacks = {
            FALLBACK_INVALID_MODEL_OUTPUT,
            FALLBACK_REQUEST_FAILURE,
            FALLBACK_MALFORMED_API_RESPONSE,
            FALLBACK_EMPTY_API_RESPONSE,
        }
        if fallback_reason not in allowed_fallbacks:
            if fallback_reason == FALLBACK_INVALID_CANDIDATE_COUNT:
                raise ValueError(
                    f"Resume candidate-count fallback is impossible for validated instance {key}"
                )
            raise ValueError(f"Resume fallback reason is unknown at {key}")
        if positions != [] or parsed_ids != [] or parsed_titles != []:
            raise ValueError(f"Resume fallback parsed fields are inconsistent at {key}")
        if final_order != expected_original_order:
            raise ValueError(f"Resume fallback changed candidate order at {key}")
        raw_output = record.get("raw_qwen_output")
        if fallback_reason == FALLBACK_INVALID_MODEL_OUTPUT:
            if not isinstance(raw_output, str) or not isinstance(fallback_detail, str):
                raise ValueError(f"Resume invalid-output fallback lacks raw output at {key}")
            try:
                parse_ranked_positions(raw_output, candidate_count=50)
            except RankedPositionsError as error:
                if fallback_detail != str(error):
                    raise ValueError(
                        f"Resume invalid-output detail is inconsistent at {key}"
                    ) from error
            else:
                raise ValueError(f"Resume fallback reason contradicts raw output at {key}")
        elif fallback_reason == FALLBACK_REQUEST_FAILURE:
            if raw_output is not None or not isinstance(fallback_detail, str):
                raise ValueError(f"Resume request-failure provenance is inconsistent at {key}")
        elif raw_output is not None or fallback_detail is not None:
            raise ValueError(f"Resume API-response fallback provenance is inconsistent at {key}")
    else:
        if fallback_reason is not None or fallback_detail is not None:
            raise ValueError(f"Resume non-fallback provenance is inconsistent at {key}")
        if not isinstance(positions, list):
            raise ValueError(f"Resume parsed positions are invalid at {key}")
        try:
            validated_positions = parse_ranked_positions(
                json.dumps({"ranked_ids": positions}),
                candidate_count=50,
            )
        except RankedPositionsError as error:
            raise ValueError(f"Resume parsed positions are invalid at {key}: {error}") from error
        selected = [original_candidates[position - 1] for position in validated_positions]
        expected_ids = [int(candidate["id"]) for candidate in selected]
        expected_titles = [str(candidate.get("title", "")) for candidate in selected]
        if parsed_ids != expected_ids or parsed_titles != expected_titles:
            raise ValueError(f"Resume parsed candidate mapping is inconsistent at {key}")
        expected_final_order = _ranked_order(
            complete_ranking(original_candidates, validated_positions)
        )
        if final_order != expected_final_order:
            raise ValueError(f"Resume completion rule is inconsistent at {key}")
        raw_output = record.get("raw_qwen_output")
        if not isinstance(raw_output, str):
            raise ValueError(f"Resume non-fallback raw output is invalid at {key}")
        try:
            if parse_ranked_positions(raw_output, candidate_count=50) != validated_positions:
                raise ValueError(f"Resume raw output mapping is inconsistent at {key}")
        except RankedPositionsError as error:
            raise ValueError(f"Resume raw output is invalid at {key}: {error}") from error

    attempts = record.get("request_attempts")
    successes = record.get("successful_requests")
    if (
        type(attempts) is not int
        or attempts < 1
        or attempts > settings.max_retries
    ):
        raise ValueError(f"Resume request attempts are invalid at {key}")
    if type(successes) is not int or successes not in (0, 1) or attempts < successes:
        raise ValueError(f"Resume successful request count is invalid at {key}")
    if not fallback and successes != 1:
        raise ValueError(f"Resume non-fallback request count is inconsistent at {key}")
    if fallback and fallback_reason == FALLBACK_INVALID_MODEL_OUTPUT and successes != 1:
        raise ValueError(f"Resume invalid-output request count is inconsistent at {key}")
    if fallback and fallback_reason == FALLBACK_REQUEST_FAILURE:
        if successes != 0 or attempts != settings.max_retries:
            raise ValueError(f"Resume failed-request count is inconsistent at {key}")
    if (
        fallback
        and fallback_reason
        in (FALLBACK_MALFORMED_API_RESPONSE, FALLBACK_EMPTY_API_RESPONSE)
        and successes != 1
    ):
        raise ValueError(f"Resume API-response request count is inconsistent at {key}")

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
        raise ValueError(f"Resume reciprocal rank is inconsistent at {key}")
    if (0 < original_rank <= 50) != (0 < reranked_rank <= 50):
        raise ValueError(f"Resume Recall@50 invariant failed at {key}")
    return dict(record)


def _append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _metrics_from_records(
    records: Sequence[Mapping[str, Any]], rank_field: str
) -> dict[str, Any]:
    accumulator = RankingMetrics()
    for record in records:
        rank = record.get(rank_field)
        if type(rank) is not int or rank < 0:
            raise ValueError(f"Invalid {rank_field} in completed record")
        accumulator.add_rank(rank)
    return accumulator.result()


def _assert_metric_sets_equal(
    left: Mapping[str, Any], right: Mapping[str, Any], *, context: str
) -> None:
    for name in ("Recall@1", "Recall@10", "Recall@50", "MRR"):
        if abs(float(left[name]) - float(right[name])) > METRIC_TOLERANCE:
            raise RuntimeError(
                f"{context} metric mismatch for {name}: {left[name]} != {right[name]}"
            )


def validate_full_run_accounting(
    *,
    processed_instances: int,
    processed_conversations: int,
    full_universe_requested: bool,
) -> bool:
    if not full_universe_requested:
        return False
    if processed_instances != EXPECTED_INSTANCES:
        raise RuntimeError(
            f"Full-run instance count mismatch: {processed_instances} != "
            f"{EXPECTED_INSTANCES}"
        )
    if processed_conversations != EXPECTED_CONVERSATIONS:
        raise RuntimeError(
            f"Full-run conversation count mismatch: {processed_conversations} != "
            f"{EXPECTED_CONVERSATIONS}"
        )
    return True


def validate_complete_reranked_recall_at_50(
    reranked_metrics: Mapping[str, Any],
    *,
    complete: bool,
) -> None:
    """Enforce the frozen candidate-set Recall@50 scalar on complete runs."""
    if complete:
        _require_metric(
            reranked_metrics.get("Recall@50"),
            EXPECTED_RRF_METRICS["Recall@50"],
            "reranked_metrics.Recall@50",
        )


def evaluate_rrf_zeroshot(
    *,
    rrf_summary_path: str | Path = DEFAULT_RRF_SUMMARY_PATH,
    rrf_instances_path: str | Path = DEFAULT_RRF_INSTANCES_PATH,
    valid_path: str | Path = OFFICIAL_VALID_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    instance_output_path: str | Path = DEFAULT_INSTANCE_OUTPUT_PATH,
    settings: QwenRerankSettings | None = None,
    max_instances: int | None = None,
    resume: bool = False,
    post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Evaluate zero-shot Top-10 list reranking without recomputing retrieval."""
    if max_instances is not None and max_instances < 0:
        raise ValueError("max_instances must be non-negative")
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

    qwen_settings = settings or QwenRerankSettings()
    fingerprint = run_fingerprint(
        summary_path=summary_path,
        instances_path=instances_path,
        settings=qwen_settings,
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
        if instance_path.exists():
            raise FileExistsError(
                f"Instance output already exists; use --resume or a new path: {instance_path}"
            )
        instance_path.parent.mkdir(parents=True, exist_ok=True)
        instance_path.open("x", encoding="utf-8").close()
        loaded_completed = {}
    unknown_completed = set(loaded_completed) - set(frozen_by_key)
    if unknown_completed:
        raise ValueError(f"Resume contains unknown instance keys: {sorted(unknown_completed)}")
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
            raise ValueError(f"VALID line {line_number} is absent for resume record {key}")
        completed[key] = validate_resume_record(
            loaded_completed[key],
            frozen_record,
            valid_index.events[key],
            conversation,
            qwen_settings,
        )

    for frozen_record in intended_records:
        if len(completed) >= target_total:
            break
        key = instance_key(frozen_record["line_number"], frozen_record["turn_index"])
        if key in completed:
            continue
        line_number = frozen_record["line_number"]
        conversation = conversations.get(line_number)
        if conversation is None:
            raise ValueError(f"VALID line {line_number} is absent")
        evaluated = evaluate_instance(
            frozen_record,
            conversation,
            qwen_settings,
            post=post,
        )
        evaluated["run_fingerprint"] = fingerprint
        _append_jsonl(instance_path, evaluated)
        completed[key] = evaluated

    ordered_completed = [
        completed[key]
        for key in intended_keys
        if key in completed
    ]
    original_subset_metrics = _metrics_from_records(
        ordered_completed, "original_rrf_target_rank"
    )
    reranked_metrics = _metrics_from_records(ordered_completed, "reranked_target_rank")
    if abs(
        original_subset_metrics["Recall@50"] - reranked_metrics["Recall@50"]
    ) > METRIC_TOLERANCE:
        raise RuntimeError("Reranked Recall@50 changed on the processed candidate sets")

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
    result = {
        "experiment": "stage2_zero_shot_rrf_list_reranking",
        "source_split": "VALID",
        "source_path": str(source_path),
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "prompt_version": PROMPT_VERSION,
        "prompt_template_sha256": prompt_template_digest(),
        "complete": complete,
        "expected_conversations": EXPECTED_CONVERSATIONS,
        "expected_instances": EXPECTED_INSTANCES,
        "processed_conversations": processed_conversations,
        "processed_instances": processed_instances,
        "max_instances": max_instances,
        "resume": resume,
        "run_fingerprint": fingerprint,
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
        "qwen_configuration": qwen_settings.provenance(),
        "requests": {
            "attempts": sum(record["request_attempts"] for record in ordered_completed),
            "successful_requests": sum(
                record["successful_requests"] for record in ordered_completed
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
    defaults = QwenRerankSettings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rrf-summary", type=Path, default=DEFAULT_RRF_SUMMARY_PATH)
    parser.add_argument("--rrf-instances", type=Path, default=DEFAULT_RRF_INSTANCES_PATH)
    parser.add_argument("--valid-path", type=Path, default=OFFICIAL_VALID_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--instance-output-path", type=Path, default=DEFAULT_INSTANCE_OUTPUT_PATH
    )
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--server-url", default=defaults.server_url)
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--top-p", type=float, default=defaults.top_p)
    parser.add_argument(
        "--max-output-tokens", type=int, default=defaults.max_output_tokens
    )
    parser.add_argument("--max-retries", type=int, default=defaults.max_retries)
    parser.add_argument("--timeout", type=float, default=defaults.timeout)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = QwenRerankSettings(
        server_url=args.server_url,
        model=args.model,
        temperature=0.0,
        top_p=args.top_p,
        max_output_tokens=args.max_output_tokens,
        think=False,
        stream=False,
        max_retries=args.max_retries,
        timeout=args.timeout,
    )
    result = evaluate_rrf_zeroshot(
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
