from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_response_generator():
    fake_prompts = types.ModuleType("prompts")
    fake_prompts.truncate_history = lambda history: history

    fake_reranker = types.ModuleType("reranker")
    fake_reranker.USE_FAKE_MODE = False
    fake_reranker.call_qwen = lambda messages: ""

    module_path = Path(__file__).resolve().with_name("response_generator.py")
    spec = importlib.util.spec_from_file_location(
        "response_generator_under_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"prompts": fake_prompts, "reranker": fake_reranker},
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


response_generator = _load_response_generator()


class ResponseGeneratorGroundingTests(unittest.TestCase):
    selected_movie = {
        "title": "It",
        "genre": "Horror",
        "decade": "2010s",
    }

    def generate_with(self, qwen_output: str, previously_recommended=None):
        with patch.object(response_generator, "USE_FAKE_MODE", False), patch.object(
            response_generator,
            "call_qwen",
            return_value=qwen_output,
        ) as call_qwen:
            result = response_generator.generate_response(
                "User: I want a tense horror movie with a strong story.",
                self.selected_movie,
                previously_recommended=previously_recommended,
            )
        return result, call_qwen

    def test_rejects_substituted_movie(self):
        result, _ = self.generate_with(
            "SELECTED_TITLE: The Babadook\n"
            "RESPONSE: I think The Babadook would be a great fit for you."
        )

        self.assertEqual(
            result,
            response_generator._fallback_response(self.selected_movie),
        )
        self.assertNotIn("The Babadook", result)

    def test_lowercase_it_and_fit_do_not_satisfy_grounding(self):
        result, _ = self.generate_with(
            "SELECTED_TITLE: It\n"
            "RESPONSE: I think it would fit your request."
        )

        self.assertEqual(
            result,
            response_generator._fallback_response(self.selected_movie),
        )

    def test_current_selection_is_excluded_from_previous_title_veto(self):
        generated = "I recommend It because its suspense matches your request."
        result, call_qwen = self.generate_with(
            f"SELECTED_TITLE: It\nRESPONSE: {generated}",
            previously_recommended=["It", "The Shining"],
        )

        self.assertEqual(result, generated)
        system_prompt = call_qwen.call_args.args[0][0]["content"]
        self.assertIn("recommendations: ['The Shining']", system_prompt)
        self.assertNotIn("recommendations: ['It'", system_prompt)
        self.assertNotIn("Do not recommend any", system_prompt)

    def test_preserves_valid_grounded_response(self):
        generated = "I recommend It because its suspense matches your request."
        result, _ = self.generate_with(
            f"SELECTED_TITLE: It\nRESPONSE: {generated}"
        )

        self.assertEqual(result, generated)

    def test_malformed_response_fails_closed(self):
        result, _ = self.generate_with(
            "I think The Babadook would be a great fit for you."
        )

        self.assertEqual(
            result,
            response_generator._fallback_response(self.selected_movie),
        )


if __name__ == "__main__":
    unittest.main()
