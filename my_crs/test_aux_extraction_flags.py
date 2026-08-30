import copy
import os
import sys
import unittest
from unittest.mock import patch


MY_CRS_DIR = os.path.dirname(os.path.abspath(__file__))
if MY_CRS_DIR not in sys.path:
    sys.path.insert(0, MY_CRS_DIR)

import kbrd_adapter
import my_crs.entity_resolver_v3 as entity_resolver_v3


class DummyEntity:
    def __init__(self, text, label):
        self.text = text
        self.label_ = label


class DummyDoc:
    def __init__(self):
        self.ents = [DummyEntity("Mary", "PERSON")]
        self.noun_chunks = []


class TestAuxiliaryExtractionFlags(unittest.TestCase):
    DIALOGUE = "Cinemax horror Mary"
    V3_MOVIE_ID = 900
    DBPEDIA_ID = 101
    GENRE_ID = 102
    PERSON_ID = 103
    FLAG_NAMES = (
        "use_aux_dbpedia_uri_matching",
        "use_aux_genre_mapping",
        "use_aux_person_matching",
    )

    def setUp(self):
        self.original_extraction_config = copy.deepcopy(
            kbrd_adapter._cfg["extraction"]
        )
        self.original_resolver_version = kbrd_adapter._RESOLVER_VERSION
        self.original_state = {
            "_data_loaded": kbrd_adapter._data_loaded,
            "_has_error": kbrd_adapter._has_error,
            "_entity2id": kbrd_adapter._entity2id,
            "_id2entity": kbrd_adapter._id2entity,
        }

        kbrd_adapter._RESOLVER_VERSION = "v3"
        kbrd_adapter._cfg["extraction"]["resolver_version"] = "v3"
        kbrd_adapter._cfg["extraction"]["use_legacy_non_movie_entities"] = True
        for name in self.FLAG_NAMES:
            kbrd_adapter._cfg["extraction"][name] = True

        kbrd_adapter._data_loaded = True
        kbrd_adapter._has_error = False
        kbrd_adapter._entity2id = {
            "<http://dbpedia.org/resource/Cinemax>": self.DBPEDIA_ID,
            "<http://dbpedia.org/resource/Horror_film>": self.GENRE_ID,
            "<http://dbpedia.org/resource/Marty_(TV_series)>": self.PERSON_ID,
        }
        kbrd_adapter._id2entity = {
            self.DBPEDIA_ID: "<http://dbpedia.org/resource/Cinemax>",
            self.GENRE_ID: "<http://dbpedia.org/resource/Horror_film>",
            self.PERSON_ID: "<http://dbpedia.org/resource/Marty_(TV_series)>",
        }

    def tearDown(self):
        kbrd_adapter._cfg["extraction"] = self.original_extraction_config
        kbrd_adapter._RESOLVER_VERSION = self.original_resolver_version
        for name, value in self.original_state.items():
            setattr(kbrd_adapter, name, value)

    def _run_extraction(self):
        v3_result = (
            [self.V3_MOVIE_ID],
            ["[V3: Exact] Fixture Movie"],
            [
                {
                    "entity_id": self.V3_MOVIE_ID,
                    "surface_text": "Fixture Movie",
                    "start_char": -1,
                    "end_char": -1,
                    "provenance": "[V3: Exact] Fixture Movie",
                }
            ],
        )
        with patch.object(kbrd_adapter, "_load_kbrd_resources"), patch.object(
            kbrd_adapter, "_get_spacy_nlp", return_value=lambda _: DummyDoc()
        ), patch.object(
            kbrd_adapter, "_is_valid_one_word_seed", return_value=True
        ), patch.object(
            entity_resolver_v3, "resolve_mentions", return_value=v3_result
        ):
            return kbrd_adapter.prepare_input(self.DIALOGUE)

    @staticmethod
    def _provenance(result):
        return {item["provenance"] for item in result[4]}

    def test_master_legacy_flag_false_preserves_v3_only_early_return(self):
        kbrd_adapter._cfg["extraction"]["use_legacy_non_movie_entities"] = False
        result = self._run_extraction()

        self.assertEqual(
            result,
            (
                [self.V3_MOVIE_ID],
                ["[V3: Exact] Fixture Movie"],
                [],
                [],
                [
                    {
                        "entity_id": self.V3_MOVIE_ID,
                        "surface_text": "Fixture Movie",
                        "start_char": -1,
                        "end_char": -1,
                        "provenance": "[V3: Exact] Fixture Movie",
                    }
                ],
            ),
        )

    def test_all_true_matches_old_config_without_component_keys(self):
        for name in self.FLAG_NAMES:
            self.assertIs(self.original_extraction_config.get(name), True)

        explicit_true_result = self._run_extraction()
        for name in self.FLAG_NAMES:
            del kbrd_adapter._cfg["extraction"][name]
        historical_default_result = self._run_extraction()

        self.assertEqual(explicit_true_result, historical_default_result)
        self.assertTrue(
            {
                "DBpedia URI Match",
                "Genre Mapping",
                "Person Actor/Director",
            }.issubset(self._provenance(explicit_true_result))
        )

    def test_dbpedia_false_prevents_dbpedia_uri_match(self):
        kbrd_adapter._cfg["extraction"]["use_aux_dbpedia_uri_matching"] = False
        result = self._run_extraction()

        self.assertNotIn("DBpedia URI Match", self._provenance(result))
        self.assertNotIn(self.DBPEDIA_ID, result[0])

    def test_genre_false_prevents_genre_mapping_and_boost(self):
        kbrd_adapter._cfg["extraction"]["use_aux_genre_mapping"] = False
        result = self._run_extraction()

        self.assertNotIn("Genre Mapping", self._provenance(result))
        self.assertNotIn(self.GENRE_ID, result[0])

    def test_person_false_prevents_person_actor_director_match(self):
        kbrd_adapter._cfg["extraction"]["use_aux_person_matching"] = False
        result = self._run_extraction()

        self.assertNotIn("Person Actor/Director", self._provenance(result))
        self.assertNotIn(self.PERSON_ID, result[0])

    def test_disabling_one_component_preserves_the_other_two(self):
        cases = {
            "use_aux_dbpedia_uri_matching": {
                "Genre Mapping",
                "Person Actor/Director",
            },
            "use_aux_genre_mapping": {
                "DBpedia URI Match",
                "Person Actor/Director",
            },
            "use_aux_person_matching": {
                "DBpedia URI Match",
                "Genre Mapping",
            },
        }
        for disabled_flag, expected_provenance in cases.items():
            with self.subTest(disabled_flag=disabled_flag):
                for name in self.FLAG_NAMES:
                    kbrd_adapter._cfg["extraction"][name] = True
                kbrd_adapter._cfg["extraction"][disabled_flag] = False
                self.assertTrue(
                    expected_provenance.issubset(
                        self._provenance(self._run_extraction())
                    )
                )


if __name__ == "__main__":
    unittest.main()
