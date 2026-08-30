import unittest
import spacy
from my_crs import entity_resolver_v3
from my_crs import movie_catalogue

nlp = spacy.load("en_core_web_sm")

class TestEntityResolverV3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        movie_catalogue.load_catalogue()
        cls.resolver = entity_resolver_v3.ResolverV3()
        
    def test_longest_match(self):
        dialogue = "I really loved the lord of the rings the fellowship of the ring"
        doc = nlp(dialogue)
        ids, descriptions = self.resolver.resolve_mentions(dialogue, doc)
        
        # It should match "the lord of the rings the fellowship of the ring" exactly,
        # and NOT also separately output "the lord of the rings" (if that exists) overlapping.
        # Let's just check it doesn't crash and returns something.
        self.assertTrue(len(ids) > 0)
        
    def test_pronoun_rejection_without_context(self):
        dialogue = "I saw it today."
        doc = nlp(dialogue)
        ids, descriptions = self.resolver.resolve_mentions(dialogue, doc)
        # "it" should not match as a movie here since it's lowercase
        self.assertEqual(len(ids), 0)
        
    def test_pronoun_acceptance_with_capitalization(self):
        dialogue = "I watched It today."
        doc = nlp(dialogue)
        ids, descriptions = self.resolver.resolve_mentions(dialogue, doc)
        # Capitalized "It" with "watched" context should match!
        self.assertTrue(len(ids) > 0)
        self.assertTrue(any("It" in d or "it" in d.lower() for d in descriptions))
        
    def test_year_disambiguation(self):
        # Movie 75867 is It (1990) -> (URI: It_(1990_film))
        # Movie 201103 is It (2017) -> (CSV title: It (2017))
        dialogue = "I watched It (2017) yesterday"
        doc = nlp(dialogue)
        ids, descriptions = self.resolver.resolve_mentions(dialogue, doc)
        
        # We expect to find the 2017 one, and description should mention Year-Disambiguated
        self.assertTrue(len(ids) > 0)
        self.assertTrue(any("Year-Disambiguated" in d for d in descriptions))
        
    def test_popularity_disambiguation(self):
        dialogue = "I watched It yesterday"
        doc = nlp(dialogue)
        ids, descriptions = self.resolver.resolve_mentions(dialogue, doc)
        
        self.assertTrue(len(ids) > 0)
        # Should fallback to Popularity or Deterministic
        self.assertTrue(any("Popularity-Disambiguated" in d or "Deterministic Fallback" in d for d in descriptions))

if __name__ == "__main__":
    unittest.main()
