import os
import sys
import pickle
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import movie_catalogue

class TestMovieCatalogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        movie_catalogue.load_catalogue()
        
        data_dir = os.path.join(movie_catalogue.KBRD_REPO_PATH, "data", "redial")
        with open(os.path.join(data_dir, "movie_ids.pkl"), "rb") as f:
            cls.mids = pickle.load(f)
            
        with open(os.path.join(data_dir, "entity2entityId.pkl"), "rb") as f:
            cls.e2id = pickle.load(f)
            
        cls.id2e = {v: k for k, v in cls.e2id.items()}

    def test_all_movies_have_titles(self):
        titles = movie_catalogue.get_all_movies()
        # Ensure exact match in length to original movie_ids
        self.assertEqual(len(titles), len(self.mids), "Catalogue must contain exactly the KBRD movie IDs")
        
        # Ensure all movie IDs have a valid non-empty title
        for mid in self.mids:
            title = titles.get(mid)
            self.assertIsNotNone(title, f"Movie ID {mid} has None title")
            self.assertNotEqual(title, "Unknown Title", f"Movie ID {mid} was unable to resolve a title")
            self.assertGreater(len(title.strip()), 0, f"Movie ID {mid} has an empty title")
            
    def test_uri_backed_movies_match_legacy(self):
        # Pick a few known URI-backed movies
        uri_backed_mid = None
        for mid in self.mids:
            uri = self.id2e.get(mid)
            if uri and isinstance(uri, str) and 'dbpedia' in str(uri):
                uri_backed_mid = mid
                break
                
        self.assertIsNotNone(uri_backed_mid)
        legacy_title = movie_catalogue._clean_title(self.id2e.get(uri_backed_mid))
        new_title = movie_catalogue.get_title(uri_backed_mid)
        self.assertEqual(legacy_title, new_title, "URI-backed title should exactly match the legacy clean_title output")
        
    def test_uri_less_movies_resolve_from_csv(self):
        # Find an integer-keyed movie
        uri_less_mid = None
        for mid in self.mids:
            uri = self.id2e.get(mid)
            if not uri or not isinstance(uri, str) or 'dbpedia' not in str(uri):
                uri_less_mid = mid
                break
                
        self.assertIsNotNone(uri_less_mid)
        new_title = movie_catalogue.get_title(uri_less_mid)
        self.assertNotEqual(new_title, "Unknown Title")
        self.assertGreater(len(new_title), 1)

if __name__ == '__main__':
    unittest.main()
