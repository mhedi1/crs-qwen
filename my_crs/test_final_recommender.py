from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from my_crs.evaluate_ckg_complementarity import attach_catalogue_titles
from my_crs.evaluate_rrf_fusion import reciprocal_rank_fusion
from my_crs.final_recommender import (
    FROZEN_RRF_K,
    FinalRecommendationUnavailableError,
    FinalRecommender,
    FinalRecommenderReadinessError,
)
from my_crs.stage2_v2_runtime import FROZEN_TOP_K


class _FakeRetriever:
    metadata = {
        "graph_type": "conversation",
        "weighting_method": "conditional",
        "min_support": 2,
    }
    movie_ids = frozenset(range(1, 1000))

    def retrieve_views(self, entity_ids, top_k=50):
        candidates = [
            {"id": entity_id, "score": 1.0, "source": "CKG_GRAPH"}
            for entity_id in range(26, 26 + top_k)
        ]
        return {
            "graph_only": candidates,
            "budget_controlled": candidates,
            "diagnostics": {
                "movie_seed_ids": list(entity_ids),
                "post_fill_candidate_count": top_k,
            },
        }


class _FakeStage2Runtime:
    def __init__(self):
        self.rank_calls = 0

    @staticmethod
    def ensure_ready():
        return {"checkpoint_step": 1254, "beta": 1.0}

    def rank(self, history, rrf_top50):
        self.rank_calls += 1
        ranked = []
        for final_rank, candidate in enumerate(reversed(rrf_top50), 1):
            item = dict(candidate)
            item["stage1_rrf_rank"] = item["rank"]
            item["rank"] = final_rank
            item["stage2_rank"] = final_rank
            item["stage2_final_score"] = float(-final_rank)
            ranked.append(item)
        return {
            "ranked_candidates": ranked,
            "selected_candidate": ranked[0],
            "diagnostics": {"candidate_membership_preserved": True},
        }


class _FakeRetrievalBindings:
    top_k = FROZEN_TOP_K
    rrf_k = FROZEN_RRF_K
    kbrd_weight = 1.0
    ckg_weight = 1.0
    attach_catalogue_titles = staticmethod(attach_catalogue_titles)
    reciprocal_rank_fusion = staticmethod(reciprocal_rank_fusion)
    title_lookup = staticmethod(lambda entity_id: f"Movie {entity_id}")

    def __init__(self, fallback_reason=None):
        self.fallback_reason = fallback_reason
        self.kbrd_calls = []
        self.retriever = _FakeRetriever()

    def load_frozen_ckg(self, cache_dir):
        return self.retriever, Path(cache_dir) / "conversation_conditional_support2.pkl"

    @staticmethod
    def prepare_input(history):
        return [26, 27], [], ["Movie 26"], [], {}

    def get_kbrd_candidates(self, history, **kwargs):
        self.kbrd_calls.append(kwargs)
        diagnostics = kwargs["diagnostics"]
        diagnostics.update(
            {
                "fallback_reason": self.fallback_reason,
                "qwen_fallback_executed": False,
                "qwen_seed_entity_ids": [],
                "num_fused_seed_candidates": 0,
                "num_fused_qwen_candidates": 0,
            }
        )
        if self.fallback_reason is not None:
            return [{"id": 101, "title": "Static fallback"}], []
        return (
            [
                {
                    "id": entity_id,
                    "title": f"Movie {entity_id}",
                    "source": "KBRD_NEURAL",
                }
                for entity_id in range(1, FROZEN_TOP_K + 1)
            ],
            ["1990s"],
        )


class FinalRecommenderTests(unittest.TestCase):
    def test_smoke_uses_pure_kbrd_exact_rrf_and_fixed_membership(self):
        bindings = _FakeRetrievalBindings()
        stage2 = _FakeStage2Runtime()
        recommender = FinalRecommender(
            stage2_runtime=stage2,
            _bindings=bindings,
        )

        result = recommender.recommend("User: recommend something like Movie 26")

        self.assertEqual(stage2.rank_calls, 1)
        self.assertEqual(len(bindings.kbrd_calls), 1)
        self.assertEqual(bindings.kbrd_calls[0]["top_k"], FROZEN_TOP_K)
        self.assertIs(bindings.kbrd_calls[0]["use_fusion"], False)
        self.assertEqual(bindings.kbrd_calls[0]["retrieval_mode"], "kbrd")
        self.assertEqual(len(result["stage1_rrf_top50"]), FROZEN_TOP_K)
        self.assertEqual(len(result["ranked_candidates"]), FROZEN_TOP_K)
        self.assertEqual(
            {candidate["id"] for candidate in result["stage1_rrf_top50"]},
            {candidate["id"] for candidate in result["ranked_candidates"]},
        )
        self.assertEqual(result["diagnostics"]["configuration"]["rrf_k"], 60)
        self.assertEqual(
            result["diagnostics"]["configuration"]["kbrd_weight"], 1.0
        )
        self.assertEqual(
            result["diagnostics"]["configuration"]["ckg_weight"], 1.0
        )

    def test_rejects_kbrd_hardcoded_no_seed_fallback(self):
        stage2 = _FakeStage2Runtime()
        recommender = FinalRecommender(
            stage2_runtime=stage2,
            _bindings=_FakeRetrievalBindings("no_inference_seeds"),
        )
        with self.assertRaisesRegex(
            FinalRecommendationUnavailableError,
            "static fallback rejected",
        ):
            recommender.recommend("User: surprise me")
        self.assertEqual(stage2.rank_calls, 0)

    def test_missing_default_artifacts_fail_readiness_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            recommender = FinalRecommender(
                ckg_cache_dir=directory,
                stage2_runtime=_FakeStage2Runtime(),
            )
            with self.assertRaisesRegex(
                FinalRecommenderReadinessError,
                "retrieval artifacts are missing",
            ):
                recommender.ensure_ready()


if __name__ == "__main__":
    unittest.main()
