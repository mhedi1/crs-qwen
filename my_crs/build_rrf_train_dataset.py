"""Build leakage-safe TRAIN candidates and Stage-2 list-reranker SFT data.

This module is intentionally TRAIN-only.  It reuses the frozen resolver, pure
KBRD path, RRF implementation, prompt builder, parser, and title matching while
replacing only the TRAIN CKG view with exact leave-one-conversation-out counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from my_crs import movie_catalogue
from my_crs.ckg_retriever import (
    DEFAULT_CACHE_DIR,
    DEFAULT_TRAIN_PATH,
    KBRD_DATA_DIR,
    ReDialKBRDMapping,
    _suggested_target_ids,
)
from my_crs.evaluate_ckg_complementarity import (
    attach_catalogue_titles,
    extraction_configuration,
)
from my_crs.evaluate_rrf_fusion import (
    CKG_WEIGHT,
    FROZEN_CKG_GRAPH_TYPE,
    FROZEN_CKG_MIN_SUPPORT,
    FROZEN_CKG_WEIGHTING,
    KBRD_WEIGHT,
    RRF_K,
    TOP_K,
    reciprocal_rank_fusion,
)
from my_crs.evaluate_rrf_zeroshot import (
    build_dialogue_up_to,
    get_recommended_movies_at_turn,
    instance_key,
    normalize_title,
)
from my_crs.loo_ckg_retriever import (
    CONTRIBUTION_SCHEMA_VERSION,
    LOO_FORMULA_VERSION,
    POPULARITY_SUBTRACTION_VERSION,
    ConversationContribution,
    LazyLOOCKGRetriever,
    build_conversation_contributions,
    canonical_json_digest,
)
from my_crs.rrf_list_reranker import (
    PROMPT_VERSION,
    TOP_N,
    build_list_rerank_prompt,
    parse_ranked_positions,
    prompt_template_digest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_TRAIN_PATH = DEFAULT_TRAIN_PATH.resolve()
EXPECTED_TRAIN_SHA256 = "5be379d7d23e7ffc2f64c635bbed9a3323a3bd7d4ec3562196c0a82f1bdbdb80"
EXPECTED_TRAIN_CONVERSATIONS = 7293
EXPECTED_EVALUABLE_CONVERSATIONS = 7161
EXPECTED_TRAIN_INSTANCES = 23686
EXPECTED_UNIQUE_TARGET_OCCURRENCES = 26708
EXPECTED_MAX_TARGETS_PER_INSTANCE = 7
EXPECTED_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EXPECTED_BASE_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
DEFAULT_COUNT_PATH = DEFAULT_CACHE_DIR / "train_counts.pkl"
DEFAULT_KBRD_CHECKPOINT = (
    PROJECT_ROOT
    / "baseline_repo"
    / "KBRD_project"
    / "KBRD"
    / "saved"
    / "kbrd_model_retrained"
)
DEFAULT_MOVIE_CATALOGUE_PATH = (
    Path(movie_catalogue.KBRD_REPO_PATH) / "data" / "redial" / "movies_with_mentions.csv"
).resolve()
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "rrf_train_peft"

AUDIT_SCHEMA_VERSION = "rrf_train_audit_v2"
SFT_SCHEMA_VERSION = "rrf_train_sft_v1"
SUMMARY_SCHEMA_VERSION = "rrf_train_summary_v2"
RUN_SCHEMA_VERSION = "rrf_train_run_v1"
TARGET_CONSTRUCTION_VERSION = "positive_first_rrf_stable_top10_v1"
SPLIT_VERSION = "stage2_peft_split_v1"
SPLIT_SALT = "stage2_peft_split_v1"
SPLIT_TRAIN_THRESHOLD = 9000
SPLIT_MODULUS = 10000

CONTRIBUTIONS_FILENAME = "train_conversation_contributions.jsonl"
AUDIT_FILENAME = "train_rrf_candidates.audit.jsonl"
SFT_FILENAME = "train_rrf_sft.jsonl"
SUMMARY_FILENAME = "train_rrf_dataset.summary.json"
RUN_FILENAME = "train_rrf_dataset.run.json"

KBRD_FALLBACK_NO_INFERENCE_SEEDS = "no_inference_seeds"
KBRD_FALLBACK_POLICY_VERSION = "kbrd_fallback_no_seed_exclusion_v1"
EXCLUSION_NO_NEURAL_KBRD_SEEDS = "no_neural_kbrd_inference_seeds"

EXPECTED_EXTRACTION_CONFIGURATION = {
    "resolver_version": "v3",
    "use_legacy_non_movie_entities": True,
    "use_aux_dbpedia_uri_matching": True,
    "use_aux_genre_mapping": True,
    "use_aux_person_matching": False,
    "seed_selection": "all",
    "spacy_model": "en_core_web_sm",
    "fuzzy_cutoff_entity": 0.92,
    "fuzzy_cutoff_title": 0.85,
    "person_match_threshold": 85,
    "weak_seed_threshold": 4,
    "genre_boost_factor": 15,
}


@dataclass(frozen=True)
class ReconstructionExpectations:
    conversations: int
    evaluable_conversations: int
    instances: int
    unique_target_occurrences: int
    max_targets_per_instance: int


OFFICIAL_EXPECTATIONS = ReconstructionExpectations(
    EXPECTED_TRAIN_CONVERSATIONS,
    EXPECTED_EVALUABLE_CONVERSATIONS,
    EXPECTED_TRAIN_INSTANCES,
    EXPECTED_UNIQUE_TARGET_OCCURRENCES,
    EXPECTED_MAX_TARGETS_PER_INSTANCE,
)


@dataclass(frozen=True)
class TrainEvent:
    line_number: int
    conversation_id: Any
    turn_index: int
    ground_truth_titles: tuple[str, ...]
    unique_target_ids: tuple[int, ...]

    @property
    def key(self) -> str:
        return instance_key(self.line_number, self.turn_index)


@dataclass(frozen=True)
class TrainReconstruction:
    conversations: tuple[tuple[int, dict[str, Any]], ...]
    events: tuple[TrainEvent, ...]
    evaluable_conversations: int
    unique_target_occurrences: int
    max_targets_per_instance: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl_strict(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank/truncated JSONL record at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed/truncated JSONL record at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(value)
    return records


def validate_official_train_path(
    path: str | Path,
    *,
    official_path: str | Path = OFFICIAL_TRAIN_PATH,
    expected_sha256: str = EXPECTED_TRAIN_SHA256,
) -> Path:
    """Accept only the exact official TRAIN path and immutable source content."""

    candidate = Path(path).resolve()
    official = Path(official_path).resolve()
    if candidate.name != "train_data.jsonl":
        raise ValueError(f"TRAIN builder requires train_data.jsonl, got {candidate.name!r}")
    if candidate != official:
        raise ValueError(f"Expected official TRAIN path {official}, got {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    observed = _sha256(candidate)
    if observed != expected_sha256:
        raise ValueError(
            f"TRAIN SHA mismatch: expected {expected_sha256}, got {observed}"
        )
    return candidate


def reconstruct_train_instances(
    path: str | Path,
    *,
    expectations: ReconstructionExpectations | None = OFFICIAL_EXPECTATIONS,
) -> TrainReconstruction:
    """Reconstruct every respondent suggested-target event with frozen semantics."""

    conversations: list[tuple[int, dict[str, Any]]] = []
    events: list[TrainEvent] = []
    evaluable_conversations = 0
    target_occurrences = 0
    max_targets = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                conversation = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid TRAIN JSON at line {line_number}") from error
            conversations.append((line_number, conversation))
            respondent = conversation.get("respondentWorkerId", -1)
            conversation_has_event = False
            for turn_index, message in enumerate(conversation.get("messages", [])):
                if message.get("senderWorkerId", -1) != respondent:
                    continue
                ground_truth = get_recommended_movies_at_turn(conversation, turn_index)
                if not ground_truth:
                    continue
                target_ids = tuple(_suggested_target_ids(conversation, turn_index))
                if not target_ids:
                    raise ValueError(
                        f"Frozen title event lacks authoritative target IDs at {line_number}:{turn_index}"
                    )
                event = TrainEvent(
                    line_number=line_number,
                    conversation_id=conversation.get("conversationId"),
                    turn_index=turn_index,
                    ground_truth_titles=tuple(ground_truth),
                    unique_target_ids=target_ids,
                )
                events.append(event)
                conversation_has_event = True
                target_occurrences += len(target_ids)
                max_targets = max(max_targets, len(target_ids))
            evaluable_conversations += int(conversation_has_event)

    reconstruction = TrainReconstruction(
        conversations=tuple(conversations),
        events=tuple(events),
        evaluable_conversations=evaluable_conversations,
        unique_target_occurrences=target_occurrences,
        max_targets_per_instance=max_targets,
    )
    if expectations is not None:
        observed = (
            len(reconstruction.conversations),
            reconstruction.evaluable_conversations,
            len(reconstruction.events),
            reconstruction.unique_target_occurrences,
            reconstruction.max_targets_per_instance,
        )
        expected = (
            expectations.conversations,
            expectations.evaluable_conversations,
            expectations.instances,
            expectations.unique_target_occurrences,
            expectations.max_targets_per_instance,
        )
        if observed != expected:
            raise ValueError(f"Authoritative TRAIN reconstruction mismatch: {observed} != {expected}")
    return reconstruction


def assign_conversation_split(conversation_key: str) -> str:
    material = f"{SPLIT_SALT}|{conversation_key}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % SPLIT_MODULUS
    return "train" if bucket < SPLIT_TRAIN_THRESHOLD else "dev"


def positive_candidate_positions(
    candidates: Sequence[Mapping[str, Any]], ground_truth_titles: Sequence[str]
) -> list[int]:
    normalized_ground_truth = {normalize_title(str(title)) for title in ground_truth_titles}
    return [
        position
        for position, candidate in enumerate(candidates, 1)
        if normalize_title(str(candidate.get("title", ""))) in normalized_ground_truth
    ]


def construct_assistant_target(
    positive_positions: Sequence[int],
    *,
    candidate_count: int = TOP_K,
    top_n: int = TOP_N,
) -> tuple[list[int], bool, str]:
    positives = sorted({int(position) for position in positive_positions})
    if any(position < 1 or position > candidate_count for position in positives):
        raise ValueError("Positive position outside candidate list")
    negatives = [
        position for position in range(1, candidate_count + 1) if position not in set(positives)
    ]
    target_positions = (positives + negatives)[:top_n]
    truncated = len(positives) > top_n
    target = json.dumps(
        {"ranked_ids": target_positions},
        separators=(",", ":"),
    )
    parsed = parse_ranked_positions(
        target,
        candidate_count=candidate_count,
        top_n=top_n,
        allow_code_fence=False,
    )
    if parsed != target_positions:
        raise AssertionError("Generated target does not round-trip through strict parser")
    return target_positions, truncated, target


def _candidate_digest(candidates: Sequence[Mapping[str, Any]]) -> str:
    return canonical_json_digest(
        [
            {
                "position": position,
                "id": int(candidate["id"]),
                "title": str(candidate.get("title", "")),
            }
            for position, candidate in enumerate(candidates, 1)
        ]
    )


def _rank_provenance(
    candidates: Sequence[Mapping[str, Any]], *, include_scores: bool
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, 1):
        item: dict[str, Any] = {
            "rank": rank,
            "id": int(candidate["id"]),
            "title": str(candidate.get("title", "")),
            "source": candidate.get("source"),
        }
        if include_scores:
            for key in (
                "score",
                "rrf_score",
                "kbrd_rank",
                "ckg_rank",
                "kbrd_contribution",
                "ckg_contribution",
            ):
                if key in candidate:
                    item[key] = candidate[key]
        records.append(item)
    return records


def _validate_neural_kbrd_candidates(
    candidates: Sequence[Mapping[str, Any]], movie_ids: frozenset[int]
) -> None:
    if len(candidates) != TOP_K:
        raise RuntimeError(f"Neural KBRD must return exactly {TOP_K} candidates")
    ids = [int(candidate["id"]) for candidate in candidates]
    if len(set(ids)) != TOP_K:
        raise RuntimeError("Neural KBRD candidates contain duplicate canonical IDs")
    if not set(ids) <= movie_ids:
        raise RuntimeError("Neural KBRD returned IDs outside movie_ids.pkl")
    if any(candidate.get("source") != "KBRD_NEURAL" for candidate in candidates):
        raise RuntimeError("KBRD static/debug fallback detected; aborting TRAIN artifact build")


def _diagnostics_confirm_no_inference_seeds(
    diagnostics: Mapping[str, Any],
) -> bool:
    return (
        diagnostics.get("seed_entity_ids") == []
        and diagnostics.get("dialogue_seed_entity_ids") == []
        and diagnostics.get("qwen_seed_entity_ids") == []
        and diagnostics.get("num_matched_seeds") == 0
    )


def _hash_checkpoint_bundle(path: Path) -> tuple[str, list[str]]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Required retrained KBRD checkpoint is missing: {path}")
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        root = path
    else:
        files = sorted(
            item for item in path.parent.glob(path.name + "*") if item.is_file()
        )
        root = path.parent
    if not files:
        raise FileNotFoundError(f"KBRD checkpoint bundle has no files: {path}")
    digest = hashlib.sha256()
    names: list[str] = []
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        names.append(relative)
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest(), names


def _validate_kbrd_checkpoint_selection(path: Path, *, using_real_kbrd: bool) -> None:
    if path.name != "kbrd_model_retrained":
        raise ValueError("TRAIN artifacts require the retrained KBRD checkpoint")
    if using_real_kbrd and path.resolve() != DEFAULT_KBRD_CHECKPOINT.resolve():
        raise ValueError(
            "The production KBRD adapter is hard-wired to the repository's "
            "saved/kbrd_model_retrained checkpoint"
        )


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _training_extraction_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    observed = extraction_configuration(config)
    extraction = config["extraction"]
    pipeline = config["pipeline"]
    observed.update(
        {
            "spacy_model": extraction.get("spacy_model"),
            "fuzzy_cutoff_entity": extraction.get("fuzzy_cutoff_entity"),
            "fuzzy_cutoff_title": extraction.get("fuzzy_cutoff_title"),
            "person_match_threshold": extraction.get("person_match_threshold"),
            "weak_seed_threshold": pipeline.get("weak_seed_threshold"),
            "genre_boost_factor": pipeline.get("genre_boost_factor"),
        }
    )
    return observed


def _load_project_extraction_configuration() -> dict[str, Any]:
    with (PROJECT_ROOT / "my_crs" / "config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    observed = _training_extraction_configuration(config)
    if observed != EXPECTED_EXTRACTION_CONFIGURATION:
        raise ValueError(
            f"Frozen extraction configuration mismatch: {observed} != "
            f"{EXPECTED_EXTRACTION_CONFIGURATION}"
        )
    return observed


def _load_count_data(
    count_path: Path,
    source_sha256: str,
    expected_conversations: int | None = EXPECTED_TRAIN_CONVERSATIONS,
) -> dict[str, Any]:
    with count_path.open("rb") as handle:
        count_data = pickle.load(handle)
    metadata = count_data.get("metadata", {})
    if metadata.get("source_split") != "TRAIN":
        raise ValueError("Global count artifact is not TRAIN-derived")
    if metadata.get("source_sha256") != source_sha256:
        raise ValueError("Global count artifact TRAIN SHA mismatch")
    if expected_conversations is not None and int(
        count_data["conversation"]["N"]
    ) != int(expected_conversations):
        raise ValueError("Global count artifact conversation count mismatch")
    return count_data


def _validate_contributions(
    contributions: Sequence[ConversationContribution],
    count_data: Mapping[str, Any],
) -> None:
    if len(contributions) != int(count_data["conversation"]["N"]):
        raise ValueError("Conversation contribution count does not match global counts")
    keys = [item.conversation_key for item in contributions]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate conversation-contribution key")
    node_counts: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    popularity: Counter[int] = Counter()
    for contribution in contributions:
        node_counts.update(contribution.movie_ids)
        for left, right in combinations(sorted(contribution.movie_ids), 2):
            pair_counts[(left, right)] += 1
        popularity.update(contribution.popularity_counter())
    if node_counts != Counter(count_data["conversation"]["node_counts"]):
        raise ValueError("Conversation contributions do not reconstruct global node counts")
    if pair_counts != Counter(count_data["conversation"]["pair_counts"]):
        raise ValueError("Conversation contributions do not reconstruct global pair counts")
    if popularity != Counter(count_data["suggested_target_popularity"]):
        raise ValueError("Conversation contributions do not reconstruct global popularity")


def _load_contributions(path: Path) -> list[ConversationContribution]:
    return [ConversationContribution.from_record(record) for record in _load_jsonl_strict(path)]


def _load_real_dependencies() -> tuple[
    Callable[..., tuple[list[dict[str, Any]], list[Any]]],
    Callable[[str], tuple[Any, ...]],
    Callable[[int], str | None],
]:
    my_crs_path = str(PROJECT_ROOT / "my_crs")
    if my_crs_path not in sys.path:
        sys.path.insert(0, my_crs_path)
    import kbrd_adapter
    from my_crs import movie_catalogue

    movie_catalogue.load_catalogue()
    return (
        kbrd_adapter.get_kbrd_candidates,
        kbrd_adapter.prepare_input,
        movie_catalogue.get_title,
    )


def _event_record(
    *,
    event: TrainEvent,
    conversation: Mapping[str, Any],
    contribution: ConversationContribution,
    loo_view: Any,
    split: str,
    run_fingerprint: str,
    kbrd_candidate_fn: Callable[..., tuple[list[dict[str, Any]], list[Any]]],
    prepare_input_fn: Callable[[str], tuple[Any, ...]],
    title_lookup: Callable[[int], str | None],
) -> dict[str, Any]:
    history = build_dialogue_up_to(conversation, event.turn_index - 1)
    extracted = prepare_input_fn(history)
    if not isinstance(extracted, tuple) or not extracted:
        raise RuntimeError("Resolver did not return its frozen tuple result")
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
    if fallback_reason == KBRD_FALLBACK_NO_INFERENCE_SEEDS:
        if not _diagnostics_confirm_no_inference_seeds(kbrd_diagnostics):
            raise RuntimeError(
                "KBRD no_inference_seeds diagnostics contradict inference seed IDs"
            )
        normalized_ground_truth = sorted(
            {normalize_title(title) for title in event.ground_truth_titles}
        )
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "instance_key": event.key,
            "source_split": "TRAIN",
            "line_number": event.line_number,
            "conversation_id": event.conversation_id,
            "turn_index": event.turn_index,
            "history": history,
            "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
            "ground_truth_titles": list(event.ground_truth_titles),
            "normalized_ground_truth_titles": normalized_ground_truth,
            "unique_annotated_target_count": len(event.unique_target_ids),
            "kbrd_top50": [],
            "loo_ckg_top50": [],
            "rrf_top50": [],
            "positive_positions": [],
            "positive_positions_truncated": False,
            "target_positions": [],
            "eligible": False,
            "exclusion_reason": EXCLUSION_NO_NEURAL_KBRD_SEEDS,
            "assistant_target": None,
            "candidate_digest": None,
            "conversation_key": contribution.conversation_key,
            "conversation_contribution_digest": contribution.digest,
            "split": split,
            "diagnostics": {
                "all_extracted_entity_ids": all_extracted_entity_ids,
                "kbrd": kbrd_diagnostics,
                "loo_ckg": None,
            },
            "failures": [],
        }
    if fallback_reason is not None:
        raise RuntimeError(f"Fatal KBRD fallback: {fallback_reason}")
    _validate_neural_kbrd_candidates(
        kbrd_candidates, loo_view.retriever.movie_ids
    )

    ckg_views = loo_view.retrieve_views(all_extracted_entity_ids, top_k=TOP_K)
    ckg_candidates = attach_catalogue_titles(
        ckg_views["budget_controlled"], title_lookup
    )
    if len(ckg_candidates) != TOP_K or len(
        {int(item["id"]) for item in ckg_candidates}
    ) != TOP_K:
        raise RuntimeError("LOO-CKG did not produce exactly 50 unique candidates")
    rrf_candidates = reciprocal_rank_fusion(
        kbrd_candidates,
        ckg_candidates,
        rrf_k=RRF_K,
        top_k=TOP_K,
    )
    if len(rrf_candidates) != TOP_K or len(
        {int(item["id"]) for item in rrf_candidates}
    ) != TOP_K:
        raise RuntimeError("Frozen RRF did not produce exactly 50 unique candidates")

    positives = positive_candidate_positions(rrf_candidates, event.ground_truth_titles)
    eligible = bool(positives)
    target_positions: list[int] = []
    positive_truncated = False
    assistant_target: str | None = None
    if eligible:
        target_positions, positive_truncated, assistant_target = construct_assistant_target(
            positives
        )
    normalized_ground_truth = sorted(
        {normalize_title(title) for title in event.ground_truth_titles}
    )
    candidate_digest = _candidate_digest(rrf_candidates)
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "instance_key": event.key,
        "source_split": "TRAIN",
        "line_number": event.line_number,
        "conversation_id": event.conversation_id,
        "turn_index": event.turn_index,
        "history": history,
        "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
        "ground_truth_titles": list(event.ground_truth_titles),
        "normalized_ground_truth_titles": normalized_ground_truth,
        "unique_annotated_target_count": len(event.unique_target_ids),
        "kbrd_top50": _rank_provenance(kbrd_candidates, include_scores=False),
        "loo_ckg_top50": _rank_provenance(ckg_candidates, include_scores=True),
        "rrf_top50": _rank_provenance(rrf_candidates, include_scores=True),
        "positive_positions": positives,
        "positive_positions_truncated": positive_truncated,
        "target_positions": target_positions,
        "eligible": eligible,
        "exclusion_reason": None if eligible else "ground_truth_absent_from_rrf_top50",
        "assistant_target": assistant_target,
        "candidate_digest": candidate_digest,
        "conversation_key": contribution.conversation_key,
        "conversation_contribution_digest": contribution.digest,
        "split": split,
        "diagnostics": {
            "all_extracted_entity_ids": all_extracted_entity_ids,
            "kbrd": kbrd_diagnostics,
            "loo_ckg": ckg_views["diagnostics"],
        },
        "failures": [],
    }


def _minimal_sft_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if not record["eligible"] or not record.get("assistant_target"):
        raise ValueError("Only eligible records can enter the SFT artifact")
    messages = build_list_rerank_prompt(record["history"], record["rrf_top50"])
    return {
        "schema_version": SFT_SCHEMA_VERSION,
        "instance_key": record["instance_key"],
        "split": record["split"],
        "messages": messages,
        "assistant_target": record["assistant_target"],
        "candidate_digest": record["candidate_digest"],
    }


def _validate_resume_record(
    record: Mapping[str, Any],
    event: TrainEvent,
    contribution: ConversationContribution,
    fingerprint: str,
) -> None:
    if record.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("Resume audit schema mismatch")
    if record.get("run_fingerprint") != fingerprint:
        raise ValueError("Resume record fingerprint mismatch")
    if record.get("instance_key") != event.key:
        raise ValueError("Resume records are not the exact expected prefix")
    if (
        int(record.get("line_number", -1)) != event.line_number
        or int(record.get("turn_index", -1)) != event.turn_index
        or str(record.get("conversation_id")) != str(event.conversation_id)
    ):
        raise ValueError("Resume event identity mismatch")
    if list(record.get("ground_truth_titles", [])) != list(event.ground_truth_titles):
        raise ValueError("Resume ground-truth provenance mismatch")
    if record.get("split") != assign_conversation_split(contribution.conversation_key):
        raise ValueError("Resume conversation split mismatch")
    if record.get("conversation_contribution_digest") != contribution.digest:
        raise ValueError("Resume conversation-contribution digest mismatch")
    history = str(record.get("history", ""))
    if record.get("history_sha256") != hashlib.sha256(history.encode("utf-8")).hexdigest():
        raise ValueError("Resume history digest mismatch")
    if record.get("exclusion_reason") == EXCLUSION_NO_NEURAL_KBRD_SEEDS:
        diagnostics = record.get("diagnostics")
        kbrd_diagnostics = (
            diagnostics.get("kbrd") if isinstance(diagnostics, Mapping) else None
        )
        if (
            record.get("eligible") is not False
            or record.get("assistant_target") is not None
            or record.get("target_positions") != []
            or record.get("positive_positions") != []
            or record.get("positive_positions_truncated") is not False
            or record.get("kbrd_top50") != []
            or record.get("loo_ckg_top50") != []
            or record.get("rrf_top50") != []
            or record.get("candidate_digest") is not None
            or record.get("failures") != []
            or not isinstance(kbrd_diagnostics, Mapping)
            or kbrd_diagnostics.get("fallback_reason")
            != KBRD_FALLBACK_NO_INFERENCE_SEEDS
            or not _diagnostics_confirm_no_inference_seeds(kbrd_diagnostics)
            or diagnostics.get("loo_ckg") is not None
        ):
            raise ValueError("Resume no-inference-seed exclusion provenance mismatch")
        return
    candidates = record.get("rrf_top50")
    if not isinstance(candidates, list) or len(candidates) != TOP_K:
        raise ValueError("Resume RRF candidate list is incomplete")
    if record.get("candidate_digest") != _candidate_digest(candidates):
        raise ValueError("Resume candidate digest mismatch")
    positives = positive_candidate_positions(candidates, record.get("ground_truth_titles", []))
    if positives != record.get("positive_positions"):
        raise ValueError("Resume positive positions mismatch")
    eligible = bool(positives)
    if record.get("eligible") is not eligible:
        raise ValueError("Resume eligibility mismatch")
    if eligible:
        target_positions, truncated, target = construct_assistant_target(positives)
        if (
            record.get("target_positions") != target_positions
            or record.get("positive_positions_truncated") is not truncated
            or record.get("assistant_target") != target
            or record.get("exclusion_reason") is not None
        ):
            raise ValueError("Resume assistant-target provenance mismatch")
    elif (
        record.get("assistant_target") is not None
        or record.get("target_positions") != []
        or record.get("exclusion_reason") != "ground_truth_absent_from_rrf_top50"
    ):
        raise ValueError("Resume unreachable-instance provenance mismatch")


def _percentile(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = round((len(ordered) - 1) * percentile)
    return int(ordered[index])


def analyze_sft_token_lengths(
    records: Sequence[Mapping[str, Any]], tokenizer: Any
) -> dict[str, Any]:
    """Tokenize the exact prompt+assistant chat format without truncation."""

    lengths: list[int] = []
    for record in records:
        messages = [dict(message) for message in record["messages"]]
        messages.append({"role": "assistant", "content": record["assistant_target"]})
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        shape = getattr(encoded, "shape", None)
        length = int(shape[-1]) if shape is not None else len(encoded)
        lengths.append(length)
    context_limit = getattr(tokenizer, "model_max_length", None)
    if not isinstance(context_limit, int) or context_limit <= 0 or context_limit >= 10**9:
        context_limit = None
    return {
        "model_id": EXPECTED_BASE_MODEL,
        "model_revision": EXPECTED_BASE_REVISION,
        "truncation": False,
        "instances": len(lengths),
        "min": min(lengths) if lengths else 0,
        "p50": _percentile(lengths, 0.50),
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "max": max(lengths) if lengths else 0,
        "count_above_2048": sum(value > 2048 for value in lengths),
        "count_above_3072": sum(value > 3072 for value in lengths),
        "count_above_4096": sum(value > 4096 for value in lengths),
        "model_context_limit": context_limit,
        "count_above_model_context_limit": (
            sum(value > context_limit for value in lengths)
            if context_limit is not None
            else None
        ),
    }


def _summary(
    records: Sequence[Mapping[str, Any]],
    *,
    reconstruction: TrainReconstruction,
    selected_conversation_count: int,
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
    token_lengths: Mapping[str, Any] | None,
) -> dict[str, Any]:
    eligible = [record for record in records if record["eligible"]]
    exclusion_reasons = Counter(
        str(record["exclusion_reason"])
        for record in records
        if not record["eligible"]
    )
    retrieval_records = [
        record
        for record in records
        if record.get("exclusion_reason") != EXCLUSION_NO_NEURAL_KBRD_SEEDS
    ]
    first_positive_ranks = Counter(
        str(record["positive_positions"][0]) for record in eligible
    )
    target_multiplicity = Counter(
        str(record["unique_annotated_target_count"]) for record in records
    )
    kbrd_hits = 0
    ckg_hits = 0
    both = 0
    kbrd_only = 0
    ckg_only = 0
    neither = 0
    for record in retrieval_records:
        ground_truth = record["ground_truth_titles"]
        kbrd_hit = bool(positive_candidate_positions(record["kbrd_top50"], ground_truth))
        ckg_hit = bool(positive_candidate_positions(record["loo_ckg_top50"], ground_truth))
        kbrd_hits += int(kbrd_hit)
        ckg_hits += int(ckg_hit)
        if kbrd_hit and ckg_hit:
            both += 1
        elif kbrd_hit:
            kbrd_only += 1
        elif ckg_hit:
            ckg_only += 1
        else:
            neither += 1
    split_before = Counter(record["split"] for record in records)
    split_eligible = Counter(record["split"] for record in eligible)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "experiment": "leakage_safe_rrf_train_dataset",
        "source_split": "TRAIN",
        "run_fingerprint": manifest["run_fingerprint"],
        "provenance": manifest["scientific_configuration"],
        "git_commit": manifest.get("git_commit"),
        "authoritative_reconstruction": {
            "conversations": len(reconstruction.conversations),
            "evaluable_conversations": reconstruction.evaluable_conversations,
            "recommendation_instances": len(reconstruction.events),
            "unique_suggested_target_occurrences": reconstruction.unique_target_occurrences,
            "max_unique_targets_per_instance": reconstruction.max_targets_per_instance,
        },
        "processed": {
            "selected_conversations": selected_conversation_count,
            "instances": len(records),
            "eligible": len(eligible),
            "excluded": len(records) - len(eligible),
            "eligible_percentage": 100.0 * len(eligible) / len(records) if records else 0.0,
            "eligible_percentage_of_retrieval_completed": (
                100.0 * len(eligible) / len(retrieval_records)
                if retrieval_records
                else 0.0
            ),
            "eligible_conversations": len(
                {record["conversation_key"] for record in eligible}
            ),
            "retrieval_completed_instances": len(retrieval_records),
            "retrieval_not_run_instances": len(records) - len(retrieval_records),
            "excluded_by_reason": dict(sorted(exclusion_reasons.items())),
            "positive_positions_truncated": sum(
                bool(record["positive_positions_truncated"]) for record in records
            ),
        },
        "split_counts_before_eligibility_filtering": dict(sorted(split_before.items())),
        "eligible_split_counts": dict(sorted(split_eligible.items())),
        "first_positive_rank_distribution": dict(sorted(first_positive_ranks.items(), key=lambda x: int(x[0]))),
        "target_multiplicity": dict(sorted(target_multiplicity.items(), key=lambda x: int(x[0]))),
        "reachability": {
            "evaluated_instances": len(retrieval_records),
            "KBRD_Top50": kbrd_hits,
            "LOO_CKG_Top50": ckg_hits,
            "RRF_Top50": len(eligible),
            "KBRD_only": kbrd_only,
            "CKG_only": ckg_only,
            "both_sources": both,
            "neither_source": neither,
            "source_union_but_RRF_miss": both + kbrd_only + ckg_only - len(eligible),
        },
        "artifacts": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
            if path.is_file()
        },
        "token_length_statistics": token_lengths,
        "failures": [],
    }


def build_rrf_train_dataset(
    *,
    train_path: str | Path = OFFICIAL_TRAIN_PATH,
    count_path: str | Path = DEFAULT_COUNT_PATH,
    mapping_dir: str | Path = KBRD_DATA_DIR,
    movie_catalogue_path: str | Path = DEFAULT_MOVIE_CATALOGUE_PATH,
    kbrd_checkpoint: str | Path = DEFAULT_KBRD_CHECKPOINT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_conversations: int | None = None,
    max_instances: int | None = None,
    resume: bool = False,
    analyze_token_lengths: bool = False,
    tokenizer: Any | None = None,
    official_path: str | Path = OFFICIAL_TRAIN_PATH,
    expected_train_sha256: str = EXPECTED_TRAIN_SHA256,
    reconstruction_expectations: ReconstructionExpectations | None = OFFICIAL_EXPECTATIONS,
    kbrd_candidate_fn: Callable[..., tuple[list[dict[str, Any]], list[Any]]] | None = None,
    prepare_input_fn: Callable[[str], tuple[Any, ...]] | None = None,
    title_lookup: Callable[[int], str | None] | None = None,
) -> dict[str, Any]:
    """Build deterministic rich and minimal SFT artifacts, with strict resume."""

    if max_conversations is not None and max_conversations < 1:
        raise ValueError("max_conversations must be positive")
    if max_instances is not None and max_instances < 1:
        raise ValueError("max_instances must be positive")
    source = validate_official_train_path(
        train_path,
        official_path=official_path,
        expected_sha256=expected_train_sha256,
    )
    source_sha = _sha256(source)
    count_file = Path(count_path).resolve()
    checkpoint = Path(kbrd_checkpoint).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "conversation_contributions": output / CONTRIBUTIONS_FILENAME,
        "rich_audit": output / AUDIT_FILENAME,
        "minimal_sft": output / SFT_FILENAME,
        "summary": output / SUMMARY_FILENAME,
        "run_manifest": output / RUN_FILENAME,
    }
    _validate_kbrd_checkpoint_selection(
        checkpoint,
        using_real_kbrd=kbrd_candidate_fn is None,
    )
    checkpoint_sha, checkpoint_files = _hash_checkpoint_bundle(checkpoint)
    extraction = _load_project_extraction_configuration()
    reconstruction = reconstruct_train_instances(
        source,
        expectations=reconstruction_expectations,
    )
    expected_count_conversations = (
        reconstruction_expectations.conversations
        if reconstruction_expectations is not None
        else len(reconstruction.conversations)
    )
    count_data = _load_count_data(
        count_file,
        source_sha,
        expected_conversations=expected_count_conversations,
    )
    mapping = ReDialKBRDMapping.load(mapping_dir)
    mapping_path = Path(mapping_dir).resolve()
    catalogue_file = Path(movie_catalogue_path).resolve()
    if title_lookup is None and catalogue_file != DEFAULT_MOVIE_CATALOGUE_PATH:
        raise ValueError(
            "The production movie catalogue is hard-wired to movies_with_mentions.csv"
        )
    mapping_artifacts = {
        name: {
            "path": str(mapping_path / name),
            "sha256": _sha256(mapping_path / name),
        }
        for name in ("id2entity.pkl", "entity2entityId.pkl", "movie_ids.pkl")
    }
    mapping_artifacts["movies_with_mentions.csv"] = {
        "path": str(catalogue_file),
        "sha256": _sha256(catalogue_file),
    }

    if resume:
        if not paths["run_manifest"].is_file() or not paths[
            "conversation_contributions"
        ].is_file():
            raise ValueError("Resume requires existing run and contribution artifacts")
        contributions = _load_contributions(paths["conversation_contributions"])
    else:
        collisions = [path for path in paths.values() if path.exists()]
        if collisions:
            raise FileExistsError(
                "Output artifacts already exist; use --resume or a new directory: "
                + ", ".join(map(str, collisions))
            )
        contributions = build_conversation_contributions(
            reconstruction.conversations,
            mapping,
        )
        _atomic_jsonl(
            paths["conversation_contributions"],
            (item.record() for item in contributions),
        )
    _validate_contributions(contributions, count_data)
    contribution_sha = _sha256(paths["conversation_contributions"])
    contribution_by_line = {item.line_number: item for item in contributions}

    scientific_configuration = {
        "train_source": {"path": str(source), "sha256": source_sha},
        "schemas": {
            "run": RUN_SCHEMA_VERSION,
            "audit": AUDIT_SCHEMA_VERSION,
            "sft": SFT_SCHEMA_VERSION,
            "summary": SUMMARY_SCHEMA_VERSION,
            "conversation_contribution": CONTRIBUTION_SCHEMA_VERSION,
        },
        "extraction_configuration": extraction,
        "mapping_artifacts": mapping_artifacts,
        "kbrd": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_bundle_sha256": checkpoint_sha,
            "checkpoint_bundle_files": checkpoint_files,
            "retrieval_mode": "kbrd",
            "use_fusion": False,
            "top_k": TOP_K,
            "llm_used": False,
            "fallback_policy_version": KBRD_FALLBACK_POLICY_VERSION,
            "nonfatal_fallback_reasons": [KBRD_FALLBACK_NO_INFERENCE_SEEDS],
        },
        "ckg": {
            "global_count_path": str(count_file),
            "global_count_sha256": _sha256(count_file),
            "conversation_contribution_sha256": contribution_sha,
            "graph_type": FROZEN_CKG_GRAPH_TYPE,
            "weighting": FROZEN_CKG_WEIGHTING,
            "min_support": FROZEN_CKG_MIN_SUPPORT,
            "top_k": TOP_K,
            "loo_formula_version": LOO_FORMULA_VERSION,
            "node_formula": "C_-i(a)=C(a)-1[a in S_i]",
            "pair_formula": "C_-i(a,b)=C(a,b)-1[a in S_i and b in S_i]",
            "weight_formula": "W_-i(a->b)=C_-i(a,b)/C_-i(a)",
            "popularity_subtraction_version": POPULARITY_SUBTRACTION_VERSION,
        },
        "rrf": {
            "k": RRF_K,
            "weights": {"KBRD": KBRD_WEIGHT, "CKG": CKG_WEIGHT},
            "top_k": TOP_K,
            "ranks": "1-based",
            "deduplication": "canonical_entity_id",
            "tie_break": "entity_id_ascending",
            "raw_scores_used": False,
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "digest": prompt_template_digest(),
            "builder": "build_list_rerank_prompt",
            "required_top_n": TOP_N,
        },
        "target": {
            "version": TARGET_CONSTRUCTION_VERSION,
            "matching": "frozen_normalized_title",
            "unreachable_policy": "exclude_from_sft_without_gold_insertion",
        },
        "split": {
            "version": SPLIT_VERSION,
            "algorithm": "sha256_first8_mod_10000",
            "salt": SPLIT_SALT,
            "train_threshold": SPLIT_TRAIN_THRESHOLD,
            "modulus": SPLIT_MODULUS,
            "unit": "conversation",
            "assigned_before_eligibility": True,
        },
        "base_model_for_future_sft": {
            "model_id": EXPECTED_BASE_MODEL,
            "revision": EXPECTED_BASE_REVISION,
        },
        "limits": {
            "max_conversations": max_conversations,
            "max_instances": max_instances,
        },
    }
    fingerprint = canonical_json_digest(scientific_configuration)
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "git_commit": _git_commit(),
        "scientific_configuration": scientific_configuration,
        "artifact_paths": {
            "conversation_contributions": str(paths["conversation_contributions"]),
            "rich_audit": str(paths["rich_audit"]),
            "minimal_sft": str(paths["minimal_sft"]),
            "summary": str(paths["summary"]),
        },
    }
    if resume:
        with paths["run_manifest"].open("r", encoding="utf-8") as handle:
            stored_manifest = json.load(handle)
        if stored_manifest != manifest:
            raise ValueError("Resume run fingerprint/configuration mismatch")
    else:
        _atomic_json(paths["run_manifest"], manifest)

    selected_conversations = list(reconstruction.conversations)
    if max_conversations is not None:
        selected_conversations = selected_conversations[:max_conversations]
    selected_lines = {line_number for line_number, _ in selected_conversations}
    selected_events = [
        event for event in reconstruction.events if event.line_number in selected_lines
    ]
    if max_instances is not None:
        selected_events = selected_events[:max_instances]
    conversation_by_line = dict(reconstruction.conversations)

    existing: list[dict[str, Any]] = []
    if resume and paths["rich_audit"].is_file():
        existing = _load_jsonl_strict(paths["rich_audit"])
    if len(existing) > len(selected_events):
        raise ValueError("Resume audit has more records than the requested subset")
    for index, record in enumerate(existing):
        event = selected_events[index]
        _validate_resume_record(
            record,
            event,
            contribution_by_line[event.line_number],
            fingerprint,
        )

    if kbrd_candidate_fn is None or prepare_input_fn is None or title_lookup is None:
        real_kbrd, real_prepare, real_title = _load_real_dependencies()
        kbrd_candidate_fn = kbrd_candidate_fn or real_kbrd
        prepare_input_fn = prepare_input_fn or real_prepare
        title_lookup = title_lookup or real_title

    loo_retriever = LazyLOOCKGRetriever(
        count_data,
        mapping.movie_ids,
        min_support=FROZEN_CKG_MIN_SUPPORT,
    )
    loo_views = {
        line_number: loo_retriever.for_conversation(contribution_by_line[line_number])
        for line_number in selected_lines
    }
    for event in selected_events[len(existing) :]:
        contribution = contribution_by_line[event.line_number]
        record = _event_record(
            event=event,
            conversation=conversation_by_line[event.line_number],
            contribution=contribution,
            loo_view=loo_views[event.line_number],
            split=assign_conversation_split(contribution.conversation_key),
            run_fingerprint=fingerprint,
            kbrd_candidate_fn=kbrd_candidate_fn,
            prepare_input_fn=prepare_input_fn,
            title_lookup=title_lookup,
        )
        _append_jsonl(paths["rich_audit"], record)

    records = _load_jsonl_strict(paths["rich_audit"]) if selected_events else []
    if len(records) != len(selected_events):
        raise ValueError("Completed audit record count does not match selected TRAIN instances")
    eligible_records = [_minimal_sft_record(record) for record in records if record["eligible"]]
    _atomic_jsonl(paths["minimal_sft"], eligible_records)

    token_statistics = None
    if analyze_token_lengths:
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                EXPECTED_BASE_MODEL,
                revision=EXPECTED_BASE_REVISION,
            )
        token_statistics = analyze_sft_token_lengths(eligible_records, tokenizer)

    summary = _summary(
        records,
        reconstruction=reconstruction,
        selected_conversation_count=len(selected_conversations),
        manifest=manifest,
        paths={
            "conversation_contributions": paths["conversation_contributions"],
            "rich_audit": paths["rich_audit"],
            "minimal_sft": paths["minimal_sft"],
            "run_manifest": paths["run_manifest"],
        },
        token_lengths=token_statistics,
    )
    _atomic_json(paths["summary"], summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=OFFICIAL_TRAIN_PATH)
    parser.add_argument("--count-path", type=Path, default=DEFAULT_COUNT_PATH)
    parser.add_argument("--mapping-dir", type=Path, default=KBRD_DATA_DIR)
    parser.add_argument("--kbrd-checkpoint", type=Path, default=DEFAULT_KBRD_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument("--max-instances", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-token-lengths", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = build_rrf_train_dataset(
        train_path=args.train_path,
        count_path=args.count_path,
        mapping_dir=args.mapping_dir,
        kbrd_checkpoint=args.kbrd_checkpoint,
        output_dir=args.output_dir,
        max_conversations=args.max_conversations,
        max_instances=args.max_instances,
        resume=args.resume,
        analyze_token_lengths=args.analyze_token_lengths,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
