from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


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

    def generate_with(
        self,
        qwen_output: str,
        previously_recommended=None,
        is_followup: bool = False,
    ):
        with patch.object(response_generator, "USE_FAKE_MODE", False), patch.object(
            response_generator,
            "call_qwen",
            return_value=qwen_output,
        ) as call_qwen:
            result = response_generator.generate_response(
                "User: I want a tense horror movie with a strong story.",
                self.selected_movie,
                previously_recommended=previously_recommended,
                is_followup=is_followup,
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

    def test_initial_accepts_natural_selected_title_opening(self):
        generated = "It would be a strong choice because it fits your request."
        result, _ = self.generate_with(
            "SELECTED_TITLE: It\n"
            f"RESPONSE: {generated}"
        )

        self.assertEqual(result, generated)

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

    def test_followup_accepts_direct_grounded_answer(self):
        generated = "Yes, it is more suspenseful than frightening."
        result, call_qwen = self.generate_with(
            f"SELECTED_TITLE: It\nRESPONSE: {generated}",
            is_followup=True,
        )

        self.assertEqual(result, generated)
        user_prompt = call_qwen.call_args.args[0][1]["content"]
        self.assertIn("MOST RECENT follow-up question directly", user_prompt)
        self.assertNotIn("must begin exactly with: I recommend It", user_prompt)

    def test_followup_rejects_wrong_selected_title(self):
        result, _ = self.generate_with(
            "SELECTED_TITLE: The Babadook\n"
            "RESPONSE: It is more suspenseful than frightening.",
            is_followup=True,
        )

        self.assertEqual(
            result,
            response_generator._fallback_response(self.selected_movie),
        )

    def test_malformed_response_fails_closed(self):
        result, _ = self.generate_with(
            "I think The Babadook would be a great fit for you."
        )

        self.assertEqual(
            result,
            response_generator._fallback_response(self.selected_movie),
        )


class ResponseGeneratorFactualGroundingTests(unittest.TestCase):
    """NLG must not invent metadata it was never given."""

    def prompts_for(self, selected_movie, *, is_followup=False):
        """Return (system_prompt, user_prompt) actually sent to the model."""
        with patch.object(response_generator, "USE_FAKE_MODE", False), patch.object(
            response_generator,
            "call_qwen",
            return_value="",
        ) as call_qwen:
            response_generator.generate_response(
                "User: I want something starring Cillian Murphy, not by Nolan.",
                selected_movie,
                is_followup=is_followup,
            )
        messages = call_qwen.call_args.args[0]
        return messages[0]["content"], messages[1]["content"]

    # ── Test 4 ──────────────────────────────────────────────────────────────
    def test_initial_mode_carries_the_anti_fabrication_rule(self):
        system_prompt, user_prompt = self.prompts_for({"title": "Notes on a Scandal"})

        self.assertIn(
            "Only state factual information that appears in "
            "[Selected Recommendation] or [Dialogue History].",
            system_prompt,
        )
        for forbidden in ("cast", "director", "release year", "rating", "awards", "plot details"):
            with self.subTest(field=forbidden):
                self.assertIn(forbidden, system_prompt)
        self.assertIn("cannot be confirmed rather than guessing", system_prompt)
        self.assertIn("never assert cast, director, year, rating", user_prompt)

    # ── Test 5 ──────────────────────────────────────────────────────────────
    def test_followup_mode_carries_the_same_rule(self):
        system_prompt, user_prompt = self.prompts_for(
            {"title": "Notes on a Scandal"},
            is_followup=True,
        )

        self.assertIn(
            "Only state factual information that appears in "
            "[Selected Recommendation] or [Dialogue History].",
            system_prompt,
        )
        self.assertIn("cannot be confirmed rather than guessing", system_prompt)
        self.assertIn("never assert cast, director, year, rating", user_prompt)

    # ── Test 6 ──────────────────────────────────────────────────────────────
    def test_enriched_metadata_is_rendered_into_the_prompt(self):
        _, user_prompt = self.prompts_for(
            {
                "title": "Notes on a Scandal",
                "genre": "Drama",
                "decade": "2000s",
                "year": 2006,
                "rating": 7.4,
                "overview": "A teacher's affair is discovered by a colleague.",
            },
            is_followup=True,
        )

        block = user_prompt.split("[Selected Recommendation]\n", 1)[1]
        self.assertIn("Title: Notes on a Scandal", block)
        self.assertIn("Genre: Drama", block)
        self.assertIn("Decade: 2000s", block)
        self.assertIn("Year: 2006", block)
        self.assertIn("Rating: 7.4", block)
        self.assertIn(
            "Overview: A teacher's affair is discovered by a colleague.", block
        )

    # ── Test 7 ──────────────────────────────────────────────────────────────
    def test_bare_title_produces_a_clean_prompt(self):
        _, user_prompt = self.prompts_for({"title": "Notes on a Scandal"})

        block = user_prompt.split("[Selected Recommendation]\n", 1)[1].strip()

        self.assertEqual(block, "Title: Notes on a Scandal")
        self.assertNotIn("|", block)
        self.assertNotIn("Unknown", block)
        for absent in ("Genre:", "Decade:", "Year:", "Rating:", "Overview:"):
            with self.subTest(field=absent):
                self.assertNotIn(absent, block)

    def test_unknown_and_empty_fields_are_omitted(self):
        _, user_prompt = self.prompts_for(
            {
                "title": "Notes on a Scandal",
                "genre": "Unknown",
                "decade": "",
                "rating": None,
                "overview": "   ",
            }
        )

        block = user_prompt.split("[Selected Recommendation]\n", 1)[1].strip()

        self.assertEqual(block, "Title: Notes on a Scandal")

    def test_stage2_score_fields_are_never_shown_to_the_model(self):
        """The real remote payload carries ranking internals; they must not leak."""
        _, user_prompt = self.prompts_for(
            {
                "title": "Notes on a Scandal",
                "id": 4211,
                "source": "RRF",
                "rrf_score": 0.0312,
                "stage2_rank": 1,
                "stage2_final_score": -3.91,
            }
        )

        block = user_prompt.split("[Selected Recommendation]\n", 1)[1].strip()

        self.assertEqual(block, "Title: Notes on a Scandal")
        for leaked in ("rrf_score", "stage2_rank", "stage2_final_score", "0.0312"):
            with self.subTest(field=leaked):
                self.assertNotIn(leaked, block)


class FlaskFollowupCallSiteTests(unittest.TestCase):
    def test_followup_helper_enables_followup_mode(self):
        app_path = Path(__file__).resolve().parents[1] / "web_app" / "app.py"
        tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
        function_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_generate_followup_response"
        )
        isolated_module = ast.Module(body=[function_node], type_ignores=[])
        ast.fix_missing_locations(isolated_module)
        namespace = {"os": os, "sys": sys, "__file__": str(app_path)}
        exec(compile(isolated_module, str(app_path), "exec"), namespace)

        fake_response_module = types.ModuleType("my_crs.response_generator")
        fake_generate_response = Mock(return_value="grounded follow-up")
        fake_response_module.generate_response = fake_generate_response
        history = [{"role": "user", "content": "Why would I like it?"}]
        selected_movie = {"title": "It"}

        with patch.object(sys, "path", list(sys.path)), patch.dict(
            sys.modules,
            {"my_crs.response_generator": fake_response_module},
        ):
            result = namespace["_generate_followup_response"](
                history,
                selected_movie,
                ["It"],
            )

        self.assertEqual(result, "grounded follow-up")
        fake_generate_response.assert_called_once_with(
            "User: Why would I like it?",
            selected_movie,
            previously_recommended=["It"],
            is_followup=True,
        )


if __name__ == "__main__":
    unittest.main()
