"""TRAIN-only lazy leave-one-conversation-out views of the frozen CKG.

The global counts are those produced by :mod:`my_crs.ckg_retriever`.  This
module never changes the frozen VALID graph.  Instead, it subtracts one TRAIN
conversation's node, pair, and suggested-target-popularity contributions at
query time and mirrors ``CKGRetriever.retrieve_views``.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

from my_crs.ckg_retriever import (
    ReDialKBRDMapping,
    _annotation_ids,
    _suggested_target_ids,
)


CONTRIBUTION_SCHEMA_VERSION = "train_conversation_contribution_v1"
LOO_FORMULA_VERSION = "conversation_conditional_support2_loo_v1"
POPULARITY_SUBTRACTION_VERSION = "suggested_target_per_message_loo_v1"


def canonical_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pair(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


@dataclass(frozen=True)
class ConversationContribution:
    """Everything one TRAIN conversation contributed to frozen CKG retrieval."""

    line_number: int
    conversation_id: Any
    movie_ids: frozenset[int]
    popularity_contribution: tuple[tuple[int, int], ...]

    @property
    def conversation_key(self) -> str:
        return f"{self.line_number}:{self.conversation_id}"

    def popularity_counter(self) -> Counter[int]:
        return Counter(dict(self.popularity_contribution))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": CONTRIBUTION_SCHEMA_VERSION,
            "line_number": self.line_number,
            "conversation_id": self.conversation_id,
            "conversation_key": self.conversation_key,
            "movie_ids": sorted(self.movie_ids),
            "popularity_contribution": [
                {"id": entity_id, "count": count}
                for entity_id, count in self.popularity_contribution
            ],
        }

    @property
    def digest(self) -> str:
        return canonical_json_digest(self.payload())

    def record(self) -> dict[str, Any]:
        return {**self.payload(), "contribution_digest": self.digest}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ConversationContribution":
        if record.get("schema_version") != CONTRIBUTION_SCHEMA_VERSION:
            raise ValueError("Unsupported conversation-contribution schema")
        contribution = cls(
            line_number=int(record["line_number"]),
            conversation_id=record.get("conversation_id"),
            movie_ids=frozenset(int(value) for value in record["movie_ids"]),
            popularity_contribution=tuple(
                sorted(
                    (
                        int(item["id"]),
                        int(item["count"]),
                    )
                    for item in record["popularity_contribution"]
                )
            ),
        )
        if record.get("conversation_key") != contribution.conversation_key:
            raise ValueError("Conversation-contribution key mismatch")
        if record.get("contribution_digest") != contribution.digest:
            raise ValueError("Conversation-contribution digest mismatch")
        return contribution


def build_conversation_contribution(
    conversation: Mapping[str, Any],
    line_number: int,
    mapping: ReDialKBRDMapping,
) -> ConversationContribution:
    """Reproduce this conversation's exact frozen graph/popularity contribution."""

    movie_ids = {
        entity_id
        for message in conversation.get("messages", [])
        for movie_id in _annotation_ids(message.get("text", ""))
        if (entity_id := mapping.map_id(movie_id)) is not None
    }
    popularity: Counter[int] = Counter()
    for turn_index, _message in enumerate(conversation.get("messages", [])):
        for redial_movie_id in _suggested_target_ids(conversation, turn_index):
            entity_id = mapping.map_id(redial_movie_id)
            if entity_id is not None:
                popularity[entity_id] += 1

    return ConversationContribution(
        line_number=int(line_number),
        conversation_id=conversation.get("conversationId"),
        movie_ids=frozenset(movie_ids),
        popularity_contribution=tuple(sorted(popularity.items())),
    )


def build_conversation_contributions(
    conversations: Iterable[tuple[int, Mapping[str, Any]]],
    mapping: ReDialKBRDMapping,
) -> list[ConversationContribution]:
    return [
        build_conversation_contribution(conversation, line_number, mapping)
        for line_number, conversation in conversations
    ]


class LazyLOOCKGRetriever:
    """Global raw counts indexed once for cheap conversation-specific views."""

    def __init__(
        self,
        count_data: Mapping[str, Any],
        movie_ids: Iterable[int],
        *,
        min_support: int = 2,
    ) -> None:
        if min_support < 1:
            raise ValueError("min_support must be at least 1")
        conversation = count_data["conversation"]
        self.node_counts = Counter(
            {int(key): int(value) for key, value in conversation["node_counts"].items()}
        )
        self.pair_counts = Counter(
            {
                _pair(int(pair[0]), int(pair[1])): int(value)
                for pair, value in conversation["pair_counts"].items()
            }
        )
        self.popularity = Counter(
            {
                int(key): int(value)
                for key, value in count_data["suggested_target_popularity"].items()
            }
        )
        self.movie_ids = frozenset(int(value) for value in movie_ids)
        self.min_support = int(min_support)
        adjacency: defaultdict[int, list[tuple[int, int]]] = defaultdict(list)
        for (left, right), support in sorted(self.pair_counts.items()):
            if support < self.min_support:
                continue
            adjacency[left].append((right, support))
            adjacency[right].append((left, support))
        self.raw_adjacency = {
            source: tuple(sorted(neighbours, key=lambda item: item[0]))
            for source, neighbours in sorted(adjacency.items())
        }
        self.global_fallback_ranking = tuple(
            sorted(
                self.movie_ids,
                key=lambda entity_id: (-self.popularity.get(entity_id, 0), entity_id),
            )
        )

    def for_conversation(
        self, contribution: ConversationContribution
    ) -> "LOOConversationView":
        unknown = contribution.movie_ids - self.movie_ids
        if unknown:
            raise ValueError(f"Conversation contribution contains non-movie IDs: {sorted(unknown)}")
        for entity_id, count in contribution.popularity_contribution:
            if entity_id not in self.movie_ids:
                raise ValueError("Conversation popularity contains a non-movie ID")
            if count < 0 or count > self.popularity.get(entity_id, 0):
                raise ValueError("Conversation popularity contribution exceeds global count")
        return LOOConversationView(self, contribution)


class LOOConversationView:
    """One reusable lazy CKG_{-i} view for all target turns in conversation i."""

    def __init__(
        self,
        retriever: LazyLOOCKGRetriever,
        contribution: ConversationContribution,
    ) -> None:
        self.retriever = retriever
        self.contribution = contribution
        self._movie_set = contribution.movie_ids
        self._popularity_delta = contribution.popularity_counter()

    def adjusted_node_count(self, entity_id: int) -> int:
        entity_id = int(entity_id)
        return self.retriever.node_counts.get(entity_id, 0) - int(
            entity_id in self._movie_set
        )

    def adjusted_pair_count(self, left: int, right: int) -> int:
        left, right = int(left), int(right)
        return self.retriever.pair_counts.get(_pair(left, right), 0) - int(
            left in self._movie_set and right in self._movie_set
        )

    def adjusted_popularity(self, entity_id: int) -> int:
        entity_id = int(entity_id)
        return self.retriever.popularity.get(entity_id, 0) - self._popularity_delta.get(
            entity_id, 0
        )

    def _movie_seeds(self, all_extracted_entity_ids: Sequence[int]) -> list[int]:
        return sorted(
            {
                int(entity_id)
                for entity_id in all_extracted_entity_ids
                if int(entity_id) in self.retriever.movie_ids
            }
        )

    def _rank_graph_candidates(
        self, movie_seeds: Sequence[int], top_k: int
    ) -> list[dict[str, Any]]:
        scores: defaultdict[int, float] = defaultdict(float)
        for source_id in movie_seeds:
            source_count = self.adjusted_node_count(source_id)
            for target_id, _global_support in self.retriever.raw_adjacency.get(
                source_id, ()
            ):
                pair_count = self.adjusted_pair_count(source_id, target_id)
                if pair_count < self.retriever.min_support:
                    continue
                if source_count <= 0:
                    raise ValueError("Positive LOO edge has a non-positive source count")
                scores[target_id] += pair_count / source_count
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [
            {"id": entity_id, "score": score, "source": "CKG_GRAPH"}
            for entity_id, score in ranked
        ]

    def _adjusted_fallback_ids(self) -> Iterator[int]:
        """Merge unchanged global order with the few popularity-adjusted IDs."""

        modified = set(self._popularity_delta)
        unchanged = (
            entity_id
            for entity_id in self.retriever.global_fallback_ranking
            if entity_id not in modified
        )
        changed = iter(
            sorted(
                modified,
                key=lambda entity_id: (-self.adjusted_popularity(entity_id), entity_id),
            )
        )
        unchanged_value = next(unchanged, None)
        changed_value = next(changed, None)
        while unchanged_value is not None or changed_value is not None:
            if unchanged_value is None:
                assert changed_value is not None
                yield changed_value
                changed_value = next(changed, None)
                continue
            if changed_value is None:
                yield unchanged_value
                unchanged_value = next(unchanged, None)
                continue
            unchanged_key = (
                -self.adjusted_popularity(unchanged_value),
                unchanged_value,
            )
            changed_key = (-self.adjusted_popularity(changed_value), changed_value)
            if changed_key < unchanged_key:
                yield changed_value
                changed_value = next(changed, None)
            else:
                yield unchanged_value
                unchanged_value = next(unchanged, None)

    def adjusted_fallback_ranking(self) -> list[int]:
        return list(self._adjusted_fallback_ids())

    def _popularity_candidates(
        self, top_k: int, excluded_ids: set[int], source: str
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for entity_id in self._adjusted_fallback_ids():
            if entity_id in excluded_ids:
                continue
            candidates.append(
                {
                    "id": entity_id,
                    "score": float(self.adjusted_popularity(entity_id)),
                    "source": source,
                }
            )
            if len(candidates) >= top_k:
                break
        return candidates

    def retrieve_views(
        self, all_extracted_entity_ids: Sequence[int], top_k: int = 50
    ) -> dict[str, Any]:
        """Mirror frozen budget-controlled retrieval using exact LOO counts."""

        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        movie_seeds = self._movie_seeds(all_extracted_entity_ids)
        graph_candidates = self._rank_graph_candidates(movie_seeds, top_k)
        graph_ids = {candidate["id"] for candidate in graph_candidates}
        fallback_used = not movie_seeds
        fill_used = bool(movie_seeds) and len(graph_candidates) < top_k
        if fallback_used:
            budget_candidates = self._popularity_candidates(
                top_k, set(), "CKG_ZERO_SEED_FALLBACK"
            )
        else:
            budget_candidates = list(graph_candidates)
            if fill_used:
                budget_candidates.extend(
                    self._popularity_candidates(
                        top_k - len(budget_candidates),
                        graph_ids,
                        "CKG_POPULARITY_FILL",
                    )
                )
        diagnostics = {
            "fallback_used": fallback_used,
            "fill_used": fill_used,
            "num_movie_seeds": len(movie_seeds),
            "movie_seed_ids": movie_seeds,
            "pre_fill_candidate_count": len(graph_candidates),
            "post_fill_candidate_count": len(budget_candidates),
            "graph_candidate_ids": [item["id"] for item in graph_candidates],
            "candidate_origins": [item["source"] for item in budget_candidates],
            "loo_conversation_key": self.contribution.conversation_key,
            "conversation_contribution_digest": self.contribution.digest,
        }
        return {
            "graph_only": graph_candidates,
            "budget_controlled": budget_candidates,
            "diagnostics": diagnostics,
        }
