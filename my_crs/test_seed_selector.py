import unittest
from my_crs import seed_selector
from my_crs import movie_catalogue
import copy

class TestSeedSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        movie_catalogue.load_catalogue()
        
    def test_policy_all_preserves_exactly(self):
        dialogue = "I liked The Matrix and The Matrix"
        # Seed list with duplicates and non-movies
        seed_list = [101, 202, 101, 303]
        movie_ids = {101, 202} # 303 is non-movie
        metadata = [
            {"entity_id": 101, "start_char": 8},
            {"entity_id": 202, "start_char": 20},
            {"entity_id": 101, "start_char": 40}, # duplicate
        ]
        
        orig_seed_list = copy.deepcopy(seed_list)
        
        new_list, removed, diags = seed_selector.apply_selection(
            dialogue, seed_list, metadata, "all", movie_ids
        )
        
        self.assertEqual(new_list, orig_seed_list)
        self.assertEqual(removed, [])
        self.assertEqual(diags["num_seeds_after_selection"], len(orig_seed_list))
        
    def test_recent_3_preserves_order_and_keeps_recent(self):
        dialogue = "movie1 movie2 movie3 movie4"
        movie_ids = {1, 2, 3, 4}
        seed_list = [1, 5, 2, 3, 4] # 5 is non-movie
        # 4 is most recent, 3 is second, 2 is third. 1 should be dropped.
        metadata = [
            {"entity_id": 1, "start_char": 10},
            {"entity_id": 2, "start_char": 20},
            {"entity_id": 3, "start_char": 30},
            {"entity_id": 4, "start_char": 40},
        ]
        
        new_list, removed, diags = seed_selector.apply_selection(
            dialogue, seed_list, metadata, "recent_3", movie_ids
        )
        
        # 1 is dropped. Order should be exactly [5, 2, 3, 4]
        self.assertEqual(new_list, [5, 2, 3, 4])
        self.assertEqual(removed, [1])
        
    def test_recent_3_duplicate_handling(self):
        # Even if a movie appears early in seed_list, if its LAST mention is recent, it survives.
        dialogue = "movie1 movie2 movie3 movie1"
        movie_ids = {1, 2, 3}
        seed_list = [1, 2, 3, 1] 
        metadata = [
            {"entity_id": 1, "start_char": 10},
            {"entity_id": 2, "start_char": 20},
            {"entity_id": 3, "start_char": 30},
            {"entity_id": 1, "start_char": 40}, # most recent!
        ]
        
        new_list, removed, diags = seed_selector.apply_selection(
            dialogue, seed_list, metadata, "recent_3", movie_ids
        )
        
        # All 3 are within top 3 unique movies. 
        self.assertEqual(new_list, [1, 2, 3, 1])
        self.assertEqual(removed, [])
        
    def test_recent_5_preserves_order_and_keeps_recent(self):
        dialogue = "movie1 movie2 movie3 movie4 movie5 movie6 movie7"
        movie_ids = {1, 2, 3, 4, 5, 6, 7}
        # 101, 102 are non-movies
        seed_list = [1, 101, 2, 3, 4, 102, 5, 6, 7]
        # Most recent are 7, 6, 5, 4, 3 (positions 70..30)
        # 1 and 2 should be dropped
        metadata = [
            {"entity_id": 1, "start_char": 10},
            {"entity_id": 2, "start_char": 20},
            {"entity_id": 3, "start_char": 30},
            {"entity_id": 4, "start_char": 40},
            {"entity_id": 5, "start_char": 50},
            {"entity_id": 6, "start_char": 60},
            {"entity_id": 7, "start_char": 70},
        ]
        
        new_list, removed, diags = seed_selector.apply_selection(
            dialogue, seed_list, metadata, "recent_5", movie_ids
        )
        
        # Dropped 1 and 2. Order of the rest should be exactly preserved!
        self.assertEqual(new_list, [101, 3, 4, 102, 5, 6, 7])
        self.assertEqual(set(removed), {1, 2})
        
    def test_no_contextual_year(self):
        # Let's say movie 2012 has ID 999
        import my_crs.movie_catalogue as mc
        orig_get_title = mc.get_title
        mc.get_title = lambda x: "2012" if x == 999 else "The Matrix"
        
        try:
            # Contextual year
            dialogue1 = "I watched The Matrix (2012) today"
            seed_list1 = [999, 101]
            metadata1 = [{"entity_id": 999, "start_char": 22}, {"entity_id": 101, "start_char": 10}]
            
            new_list1, removed1, _ = seed_selector.apply_selection(
                dialogue1, seed_list1, metadata1, "no_contextual_year_titles", {999, 101}
            )
            self.assertEqual(new_list1, [101]) # 999 removed
            self.assertEqual(removed1, [999])
            
            # Non-contextual year
            dialogue2 = "2012 was a great movie"
            seed_list2 = [999]
            metadata2 = [{"entity_id": 999, "start_char": 0}]
            
            new_list2, removed2, _ = seed_selector.apply_selection(
                dialogue2, seed_list2, metadata2, "no_contextual_year_titles", {999}
            )
            self.assertEqual(new_list2, [999]) # 999 kept
            self.assertEqual(removed2, [])
        finally:
            mc.get_title = orig_get_title

class TestAdapterIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from my_crs import kbrd_adapter
        kbrd_adapter._cfg["extraction"]["resolver_version"] = "v3"
        kbrd_adapter._cfg["extraction"]["seed_selection"] = "all"
        kbrd_adapter._load_kbrd_resources()
        cls.adapter = kbrd_adapter
        
    def test_adapter_policy_all_reproduces_exact_seed_list(self):
        dialogues = [
            "I liked The Matrix and The Matrix", # duplicates
            "I watched Saw today, it was scary.", # V3 exact match + genre
            "Have you seen 2012 (2009)?", # contextual year
            "The director Christopher Nolan is great", # person entity
        ]
        
        for diag in dialogues:
            seed_list, _, _, _, metadata = self.adapter.prepare_input(diag)
            
            new_list, removed, diags = seed_selector.apply_selection(
                diag, seed_list, metadata, "all", self.adapter._movie_ids
            )
            
            # Policy 'all' must preserve exact order and content
            self.assertEqual(new_list, seed_list)
            self.assertEqual(removed, [])
            
    def test_adapter_diagnostics_propagation(self):
        # We need to test if prepare_input + get_kbrd_candidates correctly includes sel_diags
        # Since get_kbrd_candidates expects the actual KBRD model to be loaded (which is heavy),
        # we will mock the KBRD execution and just verify the dict.
        # Actually, get_kbrd_candidates handles `if _has_error or not _data_loaded`, but wait, 
        # the fallback path ALSO propagates sel_diags because I patched it previously...
        # Wait, the fallback path propagates sel_diags at line 679.
        # Let's just run get_kbrd_candidates with a mocked _kbrd_agent
        
        orig_cfg = self.adapter._cfg["extraction"]["seed_selection"]
        self.adapter._cfg["extraction"]["seed_selection"] = "recent_3"
        
        diagnostics = {}
        # We don't have model loaded, so it will hit fallback and return fallback candidates
        # Let's ensure sel_diags is present in diagnostics.
        # Wait, fallback is handled at the VERY top (line 573): `if _has_error or not _data_loaded:`
        # It returns fallback candidates BEFORE calling prepare_input!
        # So fallback doesn't even run seed selection!
        # Thus, if _data_loaded is False, seed_selection is NOT executed.
        # Let's temporarily mock `_data_loaded` and `_kbrd_agent` to bypass the early exit.
        
        orig_data_loaded = self.adapter._data_loaded
        orig_agent = self.adapter._kbrd_agent
        
        class DummyModel:
            def eval(self): pass
            def __call__(self, *args, **kwargs):
                import torch
                return {"scores": torch.zeros((1, max(self.movie_ids) + 1 if self.movie_ids else 1000))}
                
        class DummyAgent:
            def __init__(self, movie_ids):
                self.movie_ids = movie_ids
                self.model = DummyModel()
                self.model.movie_ids = movie_ids
                
        self.adapter._data_loaded = True
        self.adapter._has_error = False
        self.adapter._kbrd_agent = DummyAgent(list(self.adapter._movie_ids)[:100])
        
        try:
            self.adapter.get_kbrd_candidates("I watched The Matrix today.", top_k=5, diagnostics=diagnostics)
            
            # Verify that seed_selection diagnostics are present
            self.assertIn("seed_selection_policy", diagnostics)
            self.assertEqual(diagnostics["seed_selection_policy"], "recent_3")
            self.assertIn("num_seeds_before_selection", diagnostics)
            self.assertIn("num_seeds_after_selection", diagnostics)
            self.assertIn("removed_seed_ids", diagnostics)
            self.assertIn("selected_movie_seed_ids", diagnostics)
            
        finally:
            self.adapter._data_loaded = orig_data_loaded
            self.adapter._kbrd_agent = orig_agent
            self.adapter._cfg["extraction"]["seed_selection"] = orig_cfg

if __name__ == "__main__":
    unittest.main()
