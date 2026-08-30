"""Isolated TRAIN-only conversational knowledge-graph construction and retrieval.

The module deliberately does not import or modify ResolverV3 or KBRD model code.
ReDial movie annotations are mapped with the deterministic artifacts produced by
the KBRD data build:

    ReDial movie ID -> id2entity.pkl value (or the integer ID when None)
                    -> entity2entityId.pkl value

The conversation graph is structurally undirected.  Its conditional retrieval
weight is source-conditioned, W(i -> j) = C(i,j) / C(i), so the two traversal
directions of one undirected pair may have different weights.  PPMI is symmetric.
The causal graph uses the directed definitions specified in the experiment plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KBRD_DATA_DIR = (
    PROJECT_ROOT / "baseline_repo" / "KBRD_project" / "KBRD" / "data" / "redial"
)
DEFAULT_TRAIN_PATH = KBRD_DATA_DIR / "train_data.jsonl"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "experiments" / "ckg_cache"
MOVIE_ANNOTATION_RE = re.compile(r"@(\d+)")
GRAPH_TYPES = ("conversation", "causal")
WEIGHTING_METHODS = ("conditional", "ppmi")
CACHE_VERSION = 1


@dataclass(frozen=True)
class ReDialKBRDMapping:
    """Audited deterministic ReDial movie ID to KBRD movie entity ID mapping."""

    redial_to_entity: dict[int, int]
    movie_ids: frozenset[int]

    @classmethod
    def load(cls, data_dir: os.PathLike[str] | str = KBRD_DATA_DIR) -> "ReDialKBRDMapping":
        data_path = Path(data_dir)
        with (data_path / "id2entity.pkl").open("rb") as handle:
            id2entity = pickle.load(handle)
        with (data_path / "entity2entityId.pkl").open("rb") as handle:
            entity2id = pickle.load(handle)
        with (data_path / "movie_ids.pkl").open("rb") as handle:
            movie_ids = frozenset(int(value) for value in pickle.load(handle))

        redial_to_entity: dict[int, int] = {}
        missing_artifact_keys: list[int] = []
        for raw_movie_id, entity in id2entity.items():
            movie_id = int(raw_movie_id)
            entity_key = movie_id if entity is None else entity
            if entity_key not in entity2id:
                missing_artifact_keys.append(movie_id)
                continue
            redial_to_entity[movie_id] = int(entity2id[entity_key])

        if missing_artifact_keys:
            raise ValueError(
                "Mapping artifacts are internally incomplete for ReDial IDs: "
                + ", ".join(map(str, sorted(missing_artifact_keys)))
            )
        outside_universe = sorted(set(redial_to_entity.values()) - movie_ids)
        if outside_universe:
            raise ValueError(
                "Mapped entity IDs are absent from movie_ids.pkl: "
                + ", ".join(map(str, outside_universe))
            )
        if len(redial_to_entity) != len(set(redial_to_entity.values())):
            raise ValueError("The audited ReDial-to-KBRD movie mapping is not one-to-one")
        return cls(redial_to_entity=redial_to_entity, movie_ids=movie_ids)

    def map_id(self, redial_movie_id: int | str) -> int | None:
        return self.redial_to_entity.get(int(redial_movie_id))


def validate_train_source_path(path: os.PathLike[str] | str) -> Path:
    """Reject accidental VALID/TEST graph sources before opening any data."""

    source_path = Path(path).resolve()
    if source_path.name != "train_data.jsonl":
        raise ValueError(
            f"Graph construction requires train_data.jsonl, got {source_path.name!r}"
        )
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    return source_path


def _annotation_ids(text: str) -> list[int]:
    return [int(value) for value in MOVIE_ANNOTATION_RE.findall(text or "")]


def _suggested_target_ids(conversation: Mapping[str, Any], turn_index: int) -> list[int]:
    """Return unique suggested movie IDs in one genuine recommender message."""

    messages = conversation.get("messages", [])
    if not (0 <= turn_index < len(messages)):
        return []
    message = messages[turn_index]
    if message.get("senderWorkerId") != conversation.get("respondentWorkerId"):
        return []
    questions = conversation.get("respondentQuestions", {})
    if not isinstance(questions, Mapping):
        if isinstance(questions, list) and not questions:
            # Canonical ReDial contains a small, measurable set of empty-list
            # records.  They carry no recommendation labels, so there are no
            # genuine suggested targets to recover from them.
            return []
        raise ValueError("respondentQuestions must be a mapping in canonical ReDial TRAIN")

    targets: list[int] = []
    seen: set[int] = set()
    for movie_id in _annotation_ids(message.get("text", "")):
        info = questions.get(str(movie_id), questions.get(movie_id, {}))
        if isinstance(info, Mapping) and info.get("suggested", 0) == 1 and movie_id not in seen:
            targets.append(movie_id)
            seen.add(movie_id)
    return targets


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from error


def audit_train_mapping(
    conversations: Iterable[Mapping[str, Any]], mapping: ReDialKBRDMapping
) -> dict[str, Any]:
    """Audit unique-ID and occurrence-level coverage without constructing a graph."""

    annotation_ids: set[int] = set()
    mapped_annotation_ids: set[int] = set()
    suggested_ids: set[int] = set()
    mapped_suggested_ids: set[int] = set()
    annotation_occurrences_total = 0
    annotation_occurrences_mapped = 0
    suggested_occurrences_total = 0
    suggested_occurrences_mapped = 0
    num_conversations = 0
    empty_list_respondent_questions_conversations = 0

    for conversation in conversations:
        num_conversations += 1
        questions = conversation.get("respondentQuestions", {})
        if isinstance(questions, list) and not questions:
            empty_list_respondent_questions_conversations += 1
        for turn_index, message in enumerate(conversation.get("messages", [])):
            for movie_id in _annotation_ids(message.get("text", "")):
                annotation_ids.add(movie_id)
                annotation_occurrences_total += 1
                if mapping.map_id(movie_id) is not None:
                    mapped_annotation_ids.add(movie_id)
                    annotation_occurrences_mapped += 1

            for movie_id in _suggested_target_ids(conversation, turn_index):
                suggested_ids.add(movie_id)
                suggested_occurrences_total += 1
                if mapping.map_id(movie_id) is not None:
                    mapped_suggested_ids.add(movie_id)
                    suggested_occurrences_mapped += 1

    unmapped_annotation_ids = sorted(annotation_ids - mapped_annotation_ids)
    unmapped_suggested_ids = sorted(suggested_ids - mapped_suggested_ids)
    return {
        "num_train_conversations": num_conversations,
        "empty_list_respondent_questions_conversations": (
            empty_list_respondent_questions_conversations
        ),
        "total_unique_train_annotation_movie_ids": len(annotation_ids),
        "mapped_unique_train_annotation_movie_ids": len(mapped_annotation_ids),
        "unmapped_unique_train_annotation_movie_ids": len(unmapped_annotation_ids),
        "unmapped_train_annotation_movie_ids": unmapped_annotation_ids,
        "total_unique_train_suggested_target_movie_ids": len(suggested_ids),
        "mapped_unique_train_suggested_target_movie_ids": len(mapped_suggested_ids),
        "unmapped_unique_train_suggested_target_movie_ids": len(unmapped_suggested_ids),
        "unmapped_train_suggested_target_movie_ids": unmapped_suggested_ids,
        "annotation_occurrences_total": annotation_occurrences_total,
        "annotation_occurrences_mapped": annotation_occurrences_mapped,
        "annotation_occurrences_unmapped": (
            annotation_occurrences_total - annotation_occurrences_mapped
        ),
        "annotation_occurrence_mapping_coverage_pct": _percentage(
            annotation_occurrences_mapped, annotation_occurrences_total
        ),
        "suggested_target_occurrences_total": suggested_occurrences_total,
        "suggested_target_occurrences_mapped": suggested_occurrences_mapped,
        "suggested_target_occurrences_unmapped": (
            suggested_occurrences_total - suggested_occurrences_mapped
        ),
        "suggested_target_occurrence_mapping_coverage_pct": _percentage(
            suggested_occurrences_mapped, suggested_occurrences_total
        ),
    }


def assert_complete_mapping(audit: Mapping[str, Any]) -> None:
    annotation_failures = int(audit["unmapped_unique_train_annotation_movie_ids"])
    target_failures = int(audit["unmapped_unique_train_suggested_target_movie_ids"])
    if annotation_failures or target_failures:
        raise AssertionError(
            "TRAIN mapping is incomplete: "
            f"{annotation_failures} annotation IDs and {target_failures} suggested-target IDs unmapped"
        )


def build_count_data(
    conversations: Iterable[Mapping[str, Any]], mapping: ReDialKBRDMapping
) -> dict[str, Any]:
    """Build conversation-level and causal target-event counts from TRAIN records."""

    conversation_node_counts: Counter[int] = Counter()
    conversation_pair_counts: Counter[tuple[int, int]] = Counter()
    causal_source_counts: Counter[int] = Counter()
    causal_target_counts: Counter[int] = Counter()
    causal_pair_counts: Counter[tuple[int, int]] = Counter()
    suggested_target_popularity: Counter[int] = Counter()
    observed_mapped_movies: set[int] = set()
    num_conversations = 0
    num_target_events = 0

    for conversation in conversations:
        num_conversations += 1
        conversation_movies = {
            entity_id
            for message in conversation.get("messages", [])
            for movie_id in _annotation_ids(message.get("text", ""))
            if (entity_id := mapping.map_id(movie_id)) is not None
        }
        observed_mapped_movies.update(conversation_movies)
        conversation_node_counts.update(conversation_movies)
        for left, right in combinations(sorted(conversation_movies), 2):
            conversation_pair_counts[(left, right)] += 1

        prior_context: set[int] = set()
        for turn_index, message in enumerate(conversation.get("messages", [])):
            targets = _suggested_target_ids(conversation, turn_index)
            for redial_target_id in targets:
                target_id = mapping.map_id(redial_target_id)
                if target_id is None:
                    continue
                num_target_events += 1
                causal_target_counts[target_id] += 1
                suggested_target_popularity[target_id] += 1
                for source_id in prior_context:
                    causal_source_counts[source_id] += 1
                    causal_pair_counts[(source_id, target_id)] += 1

            current_movies = {
                entity_id
                for movie_id in _annotation_ids(message.get("text", ""))
                if (entity_id := mapping.map_id(movie_id)) is not None
            }
            observed_mapped_movies.update(current_movies)
            prior_context.update(current_movies)

    return {
        "cache_version": CACHE_VERSION,
        "num_conversations": num_conversations,
        "observed_mapped_movie_ids": observed_mapped_movies,
        "conversation": {
            "N": num_conversations,
            "node_counts": conversation_node_counts,
            "pair_counts": conversation_pair_counts,
        },
        "causal": {
            "N": num_target_events,
            "source_counts": causal_source_counts,
            "target_counts": causal_target_counts,
            "pair_counts": causal_pair_counts,
        },
        "suggested_target_popularity": suggested_target_popularity,
    }


def _percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def support_distribution(
    pair_counts: Mapping[tuple[int, int], int], min_support: int
) -> dict[str, Any]:
    retained = {pair for pair, count in pair_counts.items() if count >= min_support}
    nodes_before = {node for pair in pair_counts for node in pair}
    nodes_after = {node for pair in retained for node in pair}
    return {
        "total_observed_pairs": len(pair_counts),
        "pairs_with_count_eq_1": sum(count == 1 for count in pair_counts.values()),
        "pairs_with_count_ge_2": sum(count >= 2 for count in pair_counts.values()),
        "nodes_with_edges_before_filtering": len(nodes_before),
        "nodes_with_edges_after_support_filtering": len(nodes_after),
        "retained_pairs": len(retained),
        "edge_retention_pct": _percentage(len(retained), len(pair_counts)),
        "min_support": min_support,
    }


def conversation_weight(
    pair_count: int,
    source_count: int,
    target_count: int,
    num_conversations: int,
    method: str,
) -> float:
    if method == "conditional":
        return pair_count / source_count
    if method == "ppmi":
        ratio = (pair_count * num_conversations) / (source_count * target_count)
        return max(0.0, math.log2(ratio))
    raise ValueError(f"Unknown weighting method: {method}")


def causal_weight(
    pair_count: int,
    source_count: int,
    target_count: int,
    num_target_events: int,
    method: str,
) -> float:
    if method == "conditional":
        return pair_count / source_count
    if method == "ppmi":
        ratio = (pair_count * num_target_events) / (source_count * target_count)
        return max(0.0, math.log2(ratio))
    raise ValueError(f"Unknown weighting method: {method}")


def build_weighted_graph(
    count_data: Mapping[str, Any],
    mapping: ReDialKBRDMapping,
    graph_type: str,
    weighting_method: str,
    min_support: int,
    common_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if graph_type not in GRAPH_TYPES:
        raise ValueError(f"Unknown graph type: {graph_type}")
    if weighting_method not in WEIGHTING_METHODS:
        raise ValueError(f"Unknown weighting method: {weighting_method}")
    if min_support < 1:
        raise ValueError("min_support must be at least 1")

    graph_counts = count_data[graph_type]
    pair_counts = graph_counts["pair_counts"]
    adjacency: defaultdict[int, list[tuple[int, float]]] = defaultdict(list)
    positive_edges = 0

    for (source_id, target_id), pair_count in sorted(pair_counts.items()):
        if pair_count < min_support:
            continue
        if graph_type == "conversation":
            node_counts = graph_counts["node_counts"]
            forward = conversation_weight(
                pair_count,
                node_counts[source_id],
                node_counts[target_id],
                graph_counts["N"],
                weighting_method,
            )
            reverse = conversation_weight(
                pair_count,
                node_counts[target_id],
                node_counts[source_id],
                graph_counts["N"],
                weighting_method,
            )
            if forward > 0.0 or reverse > 0.0:
                if forward > 0.0:
                    adjacency[source_id].append((target_id, forward))
                if reverse > 0.0:
                    adjacency[target_id].append((source_id, reverse))
                positive_edges += 1
        else:
            weight = causal_weight(
                pair_count,
                graph_counts["source_counts"][source_id],
                graph_counts["target_counts"][target_id],
                graph_counts["N"],
                weighting_method,
            )
            if weight > 0.0:
                adjacency[source_id].append((target_id, weight))
                positive_edges += 1

    ordered_adjacency = {
        int(source): sorted(neighbours, key=lambda item: item[0])
        for source, neighbours in sorted(adjacency.items())
    }
    incident_nodes = set(ordered_adjacency)
    for neighbours in ordered_adjacency.values():
        incident_nodes.update(target for target, _ in neighbours)

    popularity = count_data["suggested_target_popularity"]
    fallback_ranking = sorted(
        mapping.movie_ids,
        key=lambda entity_id: (-int(popularity.get(entity_id, 0)), entity_id),
    )
    metadata = dict(common_metadata or {})
    metadata.update(
        {
            "cache_version": CACHE_VERSION,
            "graph_type": graph_type,
            "weighting_method": weighting_method,
            "min_support": min_support,
            "candidate_universe_size": len(mapping.movie_ids),
            "observed_mapped_movie_count": len(count_data["observed_mapped_movie_ids"]),
            "num_observations": int(graph_counts["N"]),
            "nodes_with_edges": len(incident_nodes),
            "edge_count": positive_edges,
            "zero_degree_movie_count": len(mapping.movie_ids - incident_nodes),
        }
    )
    return {
        "metadata": metadata,
        "movie_ids": mapping.movie_ids,
        "adjacency": ordered_adjacency,
        "suggested_target_popularity": Counter(popularity),
        "fallback_ranking": fallback_ranking,
    }


class CKGRetriever:
    """Movie-only CKG retrieval with deterministic aggregation and fallback."""

    def __init__(self, graph: Mapping[str, Any]):
        self.metadata = dict(graph["metadata"])
        self.movie_ids = frozenset(int(value) for value in graph["movie_ids"])
        self.adjacency = graph["adjacency"]
        self.suggested_target_popularity = Counter(graph["suggested_target_popularity"])
        self.fallback_ranking = [int(value) for value in graph["fallback_ranking"]]

    @classmethod
    def load(cls, path: os.PathLike[str] | str) -> "CKGRetriever":
        with Path(path).open("rb") as handle:
            return cls(pickle.load(handle))

    def retrieve(
        self, all_extracted_entity_ids: Sequence[int], top_k: int = 50
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return the original graph-only view, with zero-seed TRAIN fallback.

        New evaluation code should use :meth:`retrieve_views` to obtain both the
        unfilled graph view and the budget-controlled view explicitly.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        movie_seeds = self._movie_seeds(all_extracted_entity_ids)
        diagnostics = {
            "fallback_used": not movie_seeds,
            "num_movie_seeds": len(movie_seeds),
            "movie_seed_ids": movie_seeds,
        }
        if not movie_seeds:
            candidates = [
                {
                    "id": entity_id,
                    "score": float(self.suggested_target_popularity.get(entity_id, 0)),
                    "source": "CKG_ZERO_SEED_FALLBACK",
                }
                for entity_id in self.fallback_ranking[:top_k]
            ]
            return candidates, diagnostics

        ranked = self._rank_graph_candidates(movie_seeds, top_k)
        return ranked, diagnostics

    def retrieve_views(
        self, all_extracted_entity_ids: Sequence[int], top_k: int = 50
    ) -> dict[str, Any]:
        """Return separate graph-only and budget-controlled candidate views.

        The graph-only list is never popularity-filled.  The budget-controlled
        list is deterministically filled to ``top_k`` from TRAIN suggested-target
        popularity whenever the candidate universe permits it.  Zero-seed
        popularity candidates use distinct provenance from nonzero-seed fill.
        """
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        movie_seeds = self._movie_seeds(all_extracted_entity_ids)
        graph_candidates = self._rank_graph_candidates(movie_seeds, top_k)
        graph_ids = {candidate["id"] for candidate in graph_candidates}

        fallback_used = not movie_seeds
        fill_used = bool(movie_seeds) and len(graph_candidates) < top_k
        if fallback_used:
            budget_candidates = self._popularity_candidates(
                top_k=top_k,
                excluded_ids=set(),
                source="CKG_ZERO_SEED_FALLBACK",
            )
        else:
            budget_candidates = list(graph_candidates)
            if fill_used:
                budget_candidates.extend(
                    self._popularity_candidates(
                        top_k=top_k - len(budget_candidates),
                        excluded_ids=graph_ids,
                        source="CKG_POPULARITY_FILL",
                    )
                )

        diagnostics = {
            "fallback_used": fallback_used,
            "fill_used": fill_used,
            "num_movie_seeds": len(movie_seeds),
            "movie_seed_ids": movie_seeds,
            "pre_fill_candidate_count": len(graph_candidates),
            "post_fill_candidate_count": len(budget_candidates),
            "graph_candidate_ids": [candidate["id"] for candidate in graph_candidates],
            "candidate_origins": [candidate["source"] for candidate in budget_candidates],
        }
        return {
            "graph_only": graph_candidates,
            "budget_controlled": budget_candidates,
            "diagnostics": diagnostics,
        }

    def _movie_seeds(self, all_extracted_entity_ids: Sequence[int]) -> list[int]:
        return sorted(
            {
                int(entity_id)
                for entity_id in all_extracted_entity_ids
                if int(entity_id) in self.movie_ids
            }
        )

    def _rank_graph_candidates(
        self, movie_seeds: Sequence[int], top_k: int
    ) -> list[dict[str, Any]]:
        scores: defaultdict[int, float] = defaultdict(float)
        for seed_id in movie_seeds:
            for target_id, weight in self.adjacency.get(seed_id, ()):
                scores[int(target_id)] += float(weight)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            {"id": entity_id, "score": score, "source": "CKG_GRAPH"}
            for entity_id, score in ranked
        ]

    def _popularity_candidates(
        self, top_k: int, excluded_ids: set[int], source: str
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for entity_id in self.fallback_ranking:
            if entity_id in excluded_ids:
                continue
            candidates.append(
                {
                    "id": entity_id,
                    "score": float(self.suggested_target_popularity.get(entity_id, 0)),
                    "source": source,
                }
            )
            if len(candidates) >= top_k:
                break
        return candidates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def build_train_artifacts(
    train_path: os.PathLike[str] | str = DEFAULT_TRAIN_PATH,
    mapping_dir: os.PathLike[str] | str = KBRD_DATA_DIR,
    cache_dir: os.PathLike[str] | str = DEFAULT_CACHE_DIR,
    min_support: int = 2,
) -> dict[str, Any]:
    """Audit TRAIN first, record raw statistics, then create four graph caches."""

    source_path = validate_train_source_path(train_path)
    cache_path = Path(cache_dir).resolve()
    mapping = ReDialKBRDMapping.load(mapping_dir)

    audit = audit_train_mapping(_iter_jsonl(source_path), mapping)
    print(json.dumps({"mapping_coverage": audit}, indent=2, sort_keys=True))
    assert_complete_mapping(audit)

    count_data = build_count_data(_iter_jsonl(source_path), mapping)
    if count_data["num_conversations"] != audit["num_train_conversations"]:
        raise AssertionError("TRAIN conversation count changed between audit and graph passes")

    build_timestamp = datetime.now(timezone.utc).isoformat()
    common_metadata = {
        "source_split": "TRAIN",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "num_train_conversations": count_data["num_conversations"],
        "graph_build_timestamp": build_timestamp,
        "git_commit": _git_commit(),
        "mapping_coverage": audit,
    }
    count_metadata = dict(common_metadata)
    raw_conversation_nodes = {
        node for pair in count_data["conversation"]["pair_counts"] for node in pair
    }
    raw_causal_nodes = {
        node for pair in count_data["causal"]["pair_counts"] for node in pair
    }
    count_metadata.update(
        {
            "graph_type": "conversation_and_causal",
            "weighting_method": "count-cache",
            "min_support": None,
            "candidate_universe_size": len(mapping.movie_ids),
            "observed_mapped_movie_count": len(count_data["observed_mapped_movie_ids"]),
            "num_observations": {
                "conversation": int(count_data["conversation"]["N"]),
                "causal": int(count_data["causal"]["N"]),
            },
            "nodes_with_edges": {
                "conversation": len(raw_conversation_nodes),
                "causal": len(raw_causal_nodes),
            },
            "edge_count": {
                "conversation": len(count_data["conversation"]["pair_counts"]),
                "causal": len(count_data["causal"]["pair_counts"]),
            },
            "zero_degree_movie_count": {
                "conversation": len(mapping.movie_ids - raw_conversation_nodes),
                "causal": len(mapping.movie_ids - raw_causal_nodes),
            },
        }
    )
    count_cache = dict(count_data)
    count_cache["metadata"] = count_metadata
    _atomic_pickle(cache_path / "train_counts.pkl", count_cache)
    _atomic_json(cache_path / "train_counts.metadata.json", count_metadata)

    graph_statistics = {
        "source_split": "TRAIN",
        "source_path": str(source_path),
        "source_sha256": common_metadata["source_sha256"],
        "min_support": min_support,
        "mapping_coverage": audit,
        "conversation": support_distribution(
            count_data["conversation"]["pair_counts"], min_support
        ),
        "causal": support_distribution(count_data["causal"]["pair_counts"], min_support),
    }
    # The raw support report is persisted and printed before weighted graphs exist.
    _atomic_json(cache_path / "train_graph_statistics.json", graph_statistics)
    print(json.dumps({"graph_statistics": graph_statistics}, indent=2, sort_keys=True))

    graph_files: dict[str, str] = {}
    for graph_type in GRAPH_TYPES:
        for method in WEIGHTING_METHODS:
            graph = build_weighted_graph(
                count_data,
                mapping,
                graph_type,
                method,
                min_support,
                common_metadata,
            )
            filename = f"{graph_type}_{method}_support{min_support}.pkl"
            _atomic_pickle(cache_path / filename, graph)
            _atomic_json(
                cache_path / f"{graph_type}_{method}_support{min_support}.metadata.json",
                graph["metadata"],
            )
            graph_files[f"{graph_type}_{method}"] = str(cache_path / filename)

    return {
        "mapping_coverage": audit,
        "graph_statistics": graph_statistics,
        "graph_files": graph_files,
        "count_cache": str(cache_path / "train_counts.pkl"),
    }


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build", help="Build TRAIN-only count and graph caches")
    build_parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    build_parser.add_argument("--mapping-dir", type=Path, default=KBRD_DATA_DIR)
    build_parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    build_parser.add_argument("--min-support", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_cli().parse_args(argv)
    if args.command == "build":
        result = build_train_artifacts(
            train_path=args.train_path,
            mapping_dir=args.mapping_dir,
            cache_dir=args.cache_dir,
            min_support=args.min_support,
        )
        print(json.dumps({"artifacts": result["graph_files"]}, indent=2, sort_keys=True))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
