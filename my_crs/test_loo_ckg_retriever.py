import unittest
from collections import Counter

from my_crs.ckg_retriever import (
    CKGRetriever,
    ReDialKBRDMapping,
    build_count_data,
    build_weighted_graph,
)
from my_crs.evaluate_rrf_zeroshot import build_dialogue_up_to
from my_crs.loo_ckg_retriever import (
    LazyLOOCKGRetriever,
    build_conversation_contribution,
)


def _mapping(*redial_ids: int) -> ReDialKBRDMapping:
    return ReDialKBRDMapping(
        redial_to_entity={movie_id: 100 + movie_id for movie_id in redial_ids},
        movie_ids=frozenset(100 + movie_id for movie_id in redial_ids),
    )


def _conversation(
    conversation_id: str,
    message_movie_sets: list[list[int]],
    *,
    suggested_by_message: dict[int, list[int]] | None = None,
) -> dict:
    suggested_by_message = suggested_by_message or {}
    respondent = 2
    messages = []
    all_ids = {movie_id for values in message_movie_sets for movie_id in values}
    questions = {
        str(movie_id): {"suggested": 1 if any(movie_id in ids for ids in suggested_by_message.values()) else 0}
        for movie_id in all_ids
    }
    for index, values in enumerate(message_movie_sets):
        sender = respondent if index in suggested_by_message else 1
        messages.append(
            {
                "senderWorkerId": sender,
                "text": " ".join(f"@{movie_id}" for movie_id in values),
            }
        )
    return {
        "conversationId": conversation_id,
        "respondentWorkerId": respondent,
        "initiatorWorkerId": 1,
        "movieMentions": {str(movie_id): f"Movie {movie_id}" for movie_id in all_ids},
        "respondentQuestions": questions,
        "messages": messages,
    }


class TestConversationContribution(unittest.TestCase):
    def test_movie_mentions_are_deduplicated_and_popularity_is_per_message(self):
        mapping = _mapping(1, 2)
        conversation = _conversation(
            "c",
            [[1, 1, 1], [2, 2], [2]],
            suggested_by_message={1: [2], 2: [2]},
        )
        contribution = build_conversation_contribution(conversation, 7, mapping)
        self.assertEqual(contribution.movie_ids, frozenset({101, 102}))
        # Frozen _suggested_target_ids deduplicates within a message, but the
        # same suggested target in a later message contributes again.
        self.assertEqual(contribution.popularity_counter(), Counter({102: 2}))
        self.assertEqual(contribution, type(contribution).from_record(contribution.record()))


class TestLazyLOOCKGRetriever(unittest.TestCase):
    def setUp(self):
        self.mapping = _mapping(1, 2, 3, 4, 5)
        self.removed = _conversation(
            "removed",
            [[1, 3], [2]],
            suggested_by_message={1: [2]},
        )
        # (1,2) has global support 2 and must disappear after removing c0.
        self.second_pair = _conversation("pair2", [[1, 2]])
        # (1,3) has global support 3. Removing c0 makes it support 2 and also
        # changes C(1) from 4 to 3, so both numerator and denominator adjust.
        self.third = _conversation("third", [[1, 3]], suggested_by_message={0: [3]})
        self.fourth = _conversation(
            "fourth", [[1, 3, 4, 5]], suggested_by_message={0: [3]}
        )
        self.conversations = [self.removed, self.second_pair, self.third, self.fourth]
        self.count_data = build_count_data(self.conversations, self.mapping)
        self.contribution = build_conversation_contribution(
            self.removed, 1, self.mapping
        )
        self.lazy = LazyLOOCKGRetriever(
            self.count_data, self.mapping.movie_ids, min_support=2
        )
        self.view = self.lazy.for_conversation(self.contribution)

    def test_support2_drops_to_support1_and_support3_or_more_remains(self):
        self.assertEqual(self.count_data["conversation"]["pair_counts"][(101, 102)], 2)
        self.assertEqual(self.view.adjusted_pair_count(101, 102), 1)
        candidates = self.view.retrieve_views([101], top_k=5)["graph_only"]
        self.assertNotIn(102, [candidate["id"] for candidate in candidates])

        self.assertEqual(self.count_data["conversation"]["pair_counts"][(101, 103)], 3)
        self.assertEqual(self.view.adjusted_pair_count(101, 103), 2)
        self.assertIn(103, [candidate["id"] for candidate in candidates])

    def test_numerator_denominator_and_unaffected_counts_are_exact(self):
        self.assertEqual(self.count_data["conversation"]["node_counts"][101], 4)
        self.assertEqual(self.view.adjusted_node_count(101), 3)
        self.assertEqual(self.view.adjusted_pair_count(101, 103), 2)
        candidate = self.view.retrieve_views([101], top_k=5)["graph_only"][0]
        self.assertEqual(candidate["id"], 103)
        self.assertAlmostEqual(candidate["score"], 2 / 3)
        self.assertEqual(self.view.adjusted_node_count(104), 1)
        self.assertEqual(self.view.adjusted_pair_count(104, 105), 1)

    def test_lazy_view_exactly_matches_brute_force_rebuild(self):
        remaining = self.conversations[1:]
        brute_counts = build_count_data(remaining, self.mapping)
        brute_graph = build_weighted_graph(
            brute_counts,
            self.mapping,
            graph_type="conversation",
            weighting_method="conditional",
            min_support=2,
        )
        brute = CKGRetriever(brute_graph)

        for entity_id in self.mapping.movie_ids:
            self.assertEqual(
                self.view.adjusted_node_count(entity_id),
                brute_counts["conversation"]["node_counts"].get(entity_id, 0),
            )
        for left in self.mapping.movie_ids:
            for right in self.mapping.movie_ids:
                if left < right:
                    self.assertEqual(
                        self.view.adjusted_pair_count(left, right),
                        brute_counts["conversation"]["pair_counts"].get((left, right), 0),
                    )

        for seeds in ([], [101], [103], [101, 103]):
            lazy_views = self.view.retrieve_views(seeds, top_k=5)
            brute_views = brute.retrieve_views(seeds, top_k=5)
            self.assertEqual(lazy_views["graph_only"], brute_views["graph_only"])
            self.assertEqual(
                lazy_views["budget_controlled"], brute_views["budget_controlled"]
            )

    def test_zero_seed_and_nonzero_fill_popularity_match_brute_force(self):
        brute_counts = build_count_data(self.conversations[1:], self.mapping)
        brute = CKGRetriever(
            build_weighted_graph(
                brute_counts,
                self.mapping,
                "conversation",
                "conditional",
                2,
            )
        )
        self.assertEqual(self.view.adjusted_popularity(102), 0)
        self.assertEqual(
            self.view.retrieve_views([], top_k=5)["budget_controlled"],
            brute.retrieve_views([], top_k=5)["budget_controlled"],
        )
        self.assertEqual(
            self.view.retrieve_views([102], top_k=5)["budget_controlled"],
            brute.retrieve_views([102], top_k=5)["budget_controlled"],
        )

    def test_ties_use_ascending_entity_id_deterministically(self):
        first = self.view.retrieve_views([], top_k=5)["budget_controlled"]
        second = self.view.retrieve_views([], top_k=5)["budget_controlled"]
        self.assertEqual(first, second)
        zero_popularity = [
            candidate["id"] for candidate in first if candidate["score"] == 0.0
        ]
        self.assertEqual(zero_popularity, sorted(zero_popularity))

    def test_denominator_only_adjustment_matches_brute_force_retrieval(self):
        mapping = _mapping(1, 2, 3)
        removed = _conversation("removed", [[1, 3]])
        first_edge = _conversation("edge-one", [[1, 2]])
        second_edge = _conversation("edge-two", [[1, 2]])
        conversations = [removed, first_edge, second_edge]
        global_counts = build_count_data(conversations, mapping)
        contribution = build_conversation_contribution(removed, 1, mapping)
        view = LazyLOOCKGRetriever(
            global_counts, mapping.movie_ids, min_support=2
        ).for_conversation(contribution)

        self.assertNotIn(102, contribution.movie_ids)
        self.assertEqual(global_counts["conversation"]["pair_counts"][(101, 102)], 2)
        self.assertEqual(view.adjusted_pair_count(101, 102), 2)
        self.assertEqual(global_counts["conversation"]["node_counts"][101], 3)
        self.assertEqual(view.adjusted_node_count(101), 2)
        lazy_candidate = view.retrieve_views([101], top_k=3)["graph_only"][0]
        self.assertEqual(lazy_candidate["id"], 102)
        self.assertAlmostEqual(lazy_candidate["score"], 1.0)

        brute_counts = build_count_data(conversations[1:], mapping)
        brute = CKGRetriever(
            build_weighted_graph(
                brute_counts,
                mapping,
                "conversation",
                "conditional",
                2,
            )
        )
        self.assertEqual(
            view.retrieve_views([101], top_k=3)["graph_only"],
            brute.retrieve_views([101], top_k=3)["graph_only"],
        )

    def test_future_annotation_is_absent_from_history_but_removed_from_full_graph(self):
        mapping = _mapping(1, 2, 3)
        removed = {
            "conversationId": "future",
            "respondentWorkerId": 2,
            "initiatorWorkerId": 1,
            "movieMentions": {"1": "Seed", "2": "Target", "3": "Future Movie"},
            "respondentQuestions": {"2": {"suggested": 1}},
            "messages": [
                {"senderWorkerId": 1, "text": "I liked @1"},
                {"senderWorkerId": 2, "text": "Try @2"},
                {"senderWorkerId": 1, "text": "Only later @3"},
            ],
        }
        others = [
            _conversation("edge-one", [[1, 3]]),
            _conversation("edge-two", [[1, 3]]),
        ]
        history = build_dialogue_up_to(removed, 0)
        self.assertNotIn("Future Movie", history)

        contribution = build_conversation_contribution(removed, 1, mapping)
        self.assertIn(103, contribution.movie_ids)
        counts = build_count_data([removed, *others], mapping)
        view = LazyLOOCKGRetriever(
            counts, mapping.movie_ids, min_support=2
        ).for_conversation(contribution)
        self.assertEqual(counts["conversation"]["pair_counts"][(101, 103)], 3)
        self.assertEqual(view.adjusted_pair_count(101, 103), 2)

        brute_counts = build_count_data(others, mapping)
        brute = CKGRetriever(
            build_weighted_graph(
                brute_counts,
                mapping,
                "conversation",
                "conditional",
                2,
            )
        )
        self.assertEqual(
            view.retrieve_views([101], top_k=3)["graph_only"],
            brute.retrieve_views([101], top_k=3)["graph_only"],
        )


if __name__ == "__main__":
    unittest.main()
