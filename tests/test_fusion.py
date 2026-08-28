import sys
import os
import unittest
from unittest.mock import patch

# Add my_crs to path to import kbrd_adapter
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'my_crs')))
import kbrd_adapter

class DummyKBRDModel:
    def eval(self): pass
    def __call__(self, seed_sets, labels):
        import torch
        scores = torch.zeros(1, 100)
        # 10 is ranked high (KBRD preserved)
        scores[0, 10] = 5.0
        # 20 is ranked low (KBRD tail)
        scores[0, 20] = -5.0
        return {"scores": scores}

class DummyAgent:
    def __init__(self):
        self.model = DummyKBRDModel()
        self.use_cuda = False
        self.movie_ids = list(range(100))

class TestCandidateFusion(unittest.TestCase):
    def setUp(self):
        kbrd_adapter._data_loaded = True
        kbrd_adapter._has_error = False
        kbrd_adapter._id2entity = {i: f"<http://dbpedia.org/resource/Movie_{i}>" for i in range(100)}
        kbrd_adapter._id2entity.update({
            10: "<http://dbpedia.org/resource/Preserved_Movie_(2010_film)>",
            30: "<http://dbpedia.org/resource/Dialogue_Seed_Movie_(2015_film)>",
            40: "<http://dbpedia.org/resource/Qwen_Seed_Movie_(2018_film)>",
        })
        kbrd_adapter._movie_ids = list(range(100))
        kbrd_adapter._entity2id = {v: k for k, v in kbrd_adapter._id2entity.items()}
        kbrd_adapter._movie_title_to_id = {f"movie {i}": i for i in range(100)}
        kbrd_adapter._movie_title_to_id.update({
            "preserved movie": 10,
            "dialogue seed movie": 30,
            "qwen seed movie": 40,
        })
        kbrd_adapter._kbrd_agent = DummyAgent()

        kbrd_adapter._MAX_FUSED_CANDIDATES = 15
        kbrd_adapter._KBRD_TOP_PRESERVED = 1
        kbrd_adapter._WEAK_SEED_THRESHOLD = 4
        kbrd_adapter._N_QWEN_FUSION_TITLES = 3
        # Baseline YAML config (legacy default)
        kbrd_adapter._USE_SEED_FUSION = True
        kbrd_adapter._USE_QWEN_SEED_FALLBACK = True
        kbrd_adapter._USE_QWEN_CANDIDATE_FUSION = True

    @patch('kbrd_adapter.prepare_input')
    @patch('kbrd_adapter.call_qwen')
    def test_pure_kbrd_mode(self, mock_qwen, mock_prepare):
        # retrieval_mode="kbrd"
        mock_prepare.return_value = ([30], [], [], [])

        diags = {}
        cands, _ = kbrd_adapter.get_kbrd_candidates(
            "test dialog", top_k=5, diagnostics=diags, retrieval_mode="kbrd"
        )

        mock_qwen.assert_not_called()
        sources = {c["source"] for c in cands}
        self.assertEqual(sources, {"KBRD_NEURAL"})
        self.assertEqual(diags["num_fused_seed_candidates"], 0)
        self.assertEqual(diags["num_fused_qwen_candidates"], 0)

    @patch('kbrd_adapter.prepare_input')
    @patch('kbrd_adapter.call_qwen')
    def test_seed_fusion_mode(self, mock_qwen, mock_prepare):
        # retrieval_mode="seed_fusion"
        mock_prepare.return_value = ([30], [], [], [])

        diags = {}
        cands, _ = kbrd_adapter.get_kbrd_candidates(
            "test dialog", top_k=5, diagnostics=diags, retrieval_mode="seed_fusion"
        )

        mock_qwen.assert_not_called()
        sources = {c["source"] for c in cands}
        self.assertIn("SEED_FUSION", sources)
        self.assertNotIn("QWEN_FUSION", sources)

        ids = [c["id"] for c in cands]
        self.assertEqual(ids[:2], [10, 30])  # Top-1 preserved, then Seed injected
        self.assertEqual(diags["num_fused_seed_candidates"], 1)

    @patch('kbrd_adapter.prepare_input')
    @patch('kbrd_adapter.call_qwen')
    def test_full_mode(self, mock_qwen, mock_prepare):
        # retrieval_mode="full"
        mock_prepare.return_value = ([30], [], [], [])
        mock_qwen.return_value = "Qwen Seed Movie"

        diags = {}
        cands, _ = kbrd_adapter.get_kbrd_candidates(
            "test dialog", top_k=5, diagnostics=diags, retrieval_mode="full"
        )

        mock_qwen.assert_called_once()
        sources = {c["source"] for c in cands}
        self.assertIn("SEED_FUSION", sources)
        self.assertIn("QWEN_FUSION", sources)

        ids = [c["id"] for c in cands]
        self.assertEqual(ids[:3], [10, 30, 40])
        self.assertEqual(diags["num_fused_seed_candidates"], 1)
        self.assertEqual(diags["num_fused_qwen_candidates"], 1)

    @patch('kbrd_adapter.prepare_input')
    @patch('kbrd_adapter.call_qwen')
    def test_legacy_disable_fusion(self, mock_qwen, mock_prepare):
        # legacy disable fusion: use_fusion=False
        mock_prepare.return_value = ([30], [], [], [])
        mock_qwen.return_value = "Qwen Seed Movie"

        diags = {}
        cands, _ = kbrd_adapter.get_kbrd_candidates(
            "test dialog", top_k=5, diagnostics=diags, use_fusion=False, retrieval_mode="legacy"
        )

        # In legacy fallback mode, Qwen is STILL called to help KBRD inference
        mock_qwen.assert_called_once()
        # But candidate injection is skipped
        sources = {c["source"] for c in cands}
        self.assertEqual(sources, {"KBRD_NEURAL"})

        # Ensure 40 (Qwen seed) did not get injected, it's just in KBRD tail if ranked
        self.assertEqual(diags["num_qwen_seed_ids"], 1)
        self.assertEqual(diags["num_fused_seed_candidates"], 0)
        self.assertEqual(diags["num_fused_qwen_candidates"], 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
