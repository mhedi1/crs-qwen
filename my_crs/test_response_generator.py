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
                # A broad-preference turn, so the deterministic factual-constraint
                # guard does not intercept and the prompt is actually built.
                "User: I want a tense character drama with a strong story.",
                selected_movie,
                is_followup=is_followup,
            )
        messages = call_qwen.call_args.args[0]
        return messages[0]["content"], messages[1]["content"]

    # ── Test 4 ──────────────────────────────────────────────────────────────
    def test_initial_mode_carries_the_anti_fabrication_rule(self):
        system_prompt, user_prompt = self.prompts_for({"title": "Notes on a Scandal"})

        self.assertIn(
            "Factual claims about the selected movie may come ONLY from "
            "[Selected Recommendation].",
            system_prompt,
        )
        for forbidden in ("cast", "director", "release year", "rating", "awards", "plot details"):
            with self.subTest(field=forbidden):
                self.assertIn(forbidden, system_prompt)
        self.assertIn("cannot confirm it rather than guessing", system_prompt)
        self.assertIn("never assert or imply that a requested actor", user_prompt)

    # ── Test 5 ──────────────────────────────────────────────────────────────
    def test_followup_mode_carries_the_same_rule(self):
        system_prompt, user_prompt = self.prompts_for(
            {"title": "Notes on a Scandal"},
            is_followup=True,
        )

        self.assertIn(
            "Factual claims about the selected movie may come ONLY from "
            "[Selected Recommendation].",
            system_prompt,
        )
        self.assertIn("cannot confirm it rather than guessing", system_prompt)
        self.assertIn("never assert or imply that a requested actor", user_prompt)

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


class DialogueConstraintsAreNotEvidenceTests(unittest.TestCase):
    """Prompt-level defence for the case the deterministic guard lets through.

    When the selected movie DOES carry verified cast and director, the guard
    passes and the prompt is built — but the verified facts may still contradict
    what the user asked for. The prompt rule must keep the model from treating
    the request itself as evidence. (The unverified case is covered
    deterministically by FactualConstraintGuardTests.)
    """

    DIALOGUE = (
        "User: I'm looking for a movie starring Cillian Murphy, but I don't want "
        "anything directed by Christopher Nolan."
    )

    # Verified metadata that contradicts the request: the guard passes, so the
    # model is reached, and the rule must stop it claiming the request is met.
    SELECTED = {
        "title": "Notes on a Scandal",
        "cast": ["Judi Dench", "Cate Blanchett"],
        "director": "Richard Eyre",
    }

    def prompts_for(self, selected_movie=None, *, is_followup=False):
        selected_movie = self.SELECTED if selected_movie is None else selected_movie
        with patch.object(response_generator, "USE_FAKE_MODE", False), patch.object(
            response_generator,
            "call_qwen",
            return_value="",
        ) as call_qwen:
            response_generator.generate_response(
                self.DIALOGUE,
                selected_movie,
                is_followup=is_followup,
            )
        messages = call_qwen.call_args.args[0]
        return messages[0]["content"], messages[1]["content"]

    # ── Test A ──────────────────────────────────────────────────────────────
    def test_dialogue_history_is_declared_non_evidential(self):
        system_prompt, user_prompt = self.prompts_for()

        self.assertIn(
            "[Dialogue History] tells you what the user is looking for; it is "
            "never evidence about the selected movie's attributes.",
            system_prompt,
        )
        self.assertIn(
            "treat [Dialogue History] as the user's preferences only, never as "
            "facts about the selected movie",
            user_prompt,
        )

        # The constraints are visible only as preferences. The verified block
        # carries different people entirely, so nothing there supports the
        # requested actor or the excluded director.
        self.assertIn("Cillian Murphy", user_prompt)
        block = user_prompt.split("[Selected Recommendation]\n", 1)[1].strip()
        self.assertIn("Cast: Judi Dench, Cate Blanchett", block)
        self.assertIn("Director: Richard Eyre", block)
        self.assertNotIn("Cillian Murphy", block)
        self.assertNotIn("Christopher Nolan", block)

    # ── Test B ──────────────────────────────────────────────────────────────
    def test_requested_constraints_may_not_be_claimed_as_satisfied(self):
        system_prompt, user_prompt = self.prompts_for()

        self.assertIn(
            "A requested actor is not verified cast, an excluded director is not "
            "a verified director",
            system_prompt,
        )
        self.assertIn(
            "Never state or imply that the selected movie satisfies a requested "
            "or excluded constraint unless the corresponding fact is explicitly "
            "listed in [Selected Recommendation].",
            system_prompt,
        )
        self.assertIn("cannot confirm it rather than guessing", system_prompt)
        self.assertIn(
            "unless it is explicitly listed in [Selected Recommendation]; "
            "otherwise say it cannot be confirmed",
            user_prompt,
        )

    def test_requested_year_and_rating_are_also_non_evidential(self):
        system_prompt, _ = self.prompts_for()

        self.assertIn("a requested year is not a verified release year", system_prompt)
        self.assertIn("a requested rating is not a verified rating", system_prompt)

    def test_rule_applies_in_followup_mode_too(self):
        system_prompt, user_prompt = self.prompts_for(is_followup=True)

        self.assertIn(
            "[Dialogue History] tells you what the user is looking for; it is "
            "never evidence about the selected movie's attributes.",
            system_prompt,
        )
        self.assertIn(
            "treat [Dialogue History] as the user's preferences only, never as "
            "facts about the selected movie",
            user_prompt,
        )

    # ── Test C ──────────────────────────────────────────────────────────────
    def test_explicitly_present_facts_remain_usable(self):
        _, user_prompt = self.prompts_for(
            {
                **self.SELECTED,
                "genre": "Drama",
                "year": 2006,
                "rating": 7.4,
                "overview": "A teacher's affair is discovered by a colleague.",
            }
        )

        block = user_prompt.split("[Selected Recommendation]\n", 1)[1].strip()

        # Facts that ARE listed stay available to the model; the rule scopes
        # claims to this block rather than suppressing it.
        self.assertIn("Genre: Drama", block)
        self.assertIn("Year: 2006", block)
        self.assertIn("Rating: 7.4", block)
        self.assertIn(
            "Overview: A teacher's affair is discovered by a colleague.", block
        )
        # Still no cast or director, so those remain unconfirmable.
        self.assertNotIn("Cillian Murphy", block)
        self.assertNotIn("Christopher Nolan", block)

    def test_selected_title_grounding_is_unchanged(self):
        """The title contract must survive the stricter factual rule."""
        _, user_prompt = self.prompts_for()

        self.assertIn("SELECTED_TITLE: <copy the exact selected title>", user_prompt)
        self.assertIn("RESPONSE: <natural-language response>", user_prompt)

        with patch.object(response_generator, "USE_FAKE_MODE", False), patch.object(
            response_generator,
            "call_qwen",
            return_value="SELECTED_TITLE: Notes on a Scandal\nRESPONSE: ok",
        ):
            accepted = response_generator.generate_response(
                self.DIALOGUE, self.SELECTED
            )
        self.assertEqual(accepted, "ok")

        with patch.object(response_generator, "USE_FAKE_MODE", False), patch.object(
            response_generator,
            "call_qwen",
            return_value="SELECTED_TITLE: Inception\nRESPONSE: ok",
        ):
            rejected = response_generator.generate_response(
                self.DIALOGUE, self.SELECTED
            )
        self.assertEqual(
            rejected,
            response_generator._fallback_response(self.SELECTED),
        )


class FactualConstraintGuardTests(unittest.TestCase):
    """Deterministic gate: unverifiable factual constraints never reach Qwen3."""

    LIVE_FAILURE_HISTORY = (
        "User: I'm looking for a movie starring Cillian Murphy, but I don't want "
        "anything directed by Christopher Nolan."
    )

    def call(self, history, selected_movie, *, qwen="", is_followup=False):
        """Run generate_response with call_qwen mocked; return (result, mock)."""
        call_qwen = Mock(return_value=qwen)
        with patch.object(response_generator, "USE_FAKE_MODE", False), patch.object(
            response_generator, "call_qwen", call_qwen
        ):
            result = response_generator.generate_response(
                history,
                selected_movie,
                is_followup=is_followup,
            )
        return result, call_qwen

    # ── Test A: the exact confirmed live failure ────────────────────────────
    def test_live_failure_returns_deterministic_answer_without_qwen(self):
        result, call_qwen = self.call(
            self.LIVE_FAILURE_HISTORY,
            {"title": "Notes on a Scandal"},
        )

        call_qwen.assert_not_called()

        # Must not repeat the hallucinated claims, in either direction.
        self.assertNotIn("Cillian Murphy", result)
        self.assertNotIn("Christopher Nolan", result)
        self.assertNotIn("features", result)
        self.assertNotIn("directed by", result)

        self.assertIn("Notes on a Scandal", result)
        self.assertIn("verified cast or director information", result)
        self.assertIn("can't confirm", result)

    def test_live_failure_reports_both_missing_categories_together(self):
        result, _ = self.call(
            self.LIVE_FAILURE_HISTORY,
            {"title": "Notes on a Scandal"},
        )

        self.assertEqual(
            result,
            "Notes on a Scandal was selected by the recommendation pipeline, but "
            "I don't have verified cast or director information for this result, "
            "so I can't confirm that it satisfies those constraints.",
        )

    # ── Test B: explicit year with no year metadata ─────────────────────────
    def test_year_request_without_year_metadata_is_deterministic(self):
        result, call_qwen = self.call(
            "User: Find me a movie from 2014",
            {"title": "Notes on a Scandal"},
        )

        call_qwen.assert_not_called()
        self.assertIn("verified release year information", result)
        self.assertIn("that constraint", result)

    def test_last_n_years_phrase_is_also_a_year_constraint(self):
        result, call_qwen = self.call(
            "User: something from the last five years please",
            {"title": "Notes on a Scandal"},
        )

        call_qwen.assert_not_called()
        self.assertIn("verified release year information", result)

    def test_bare_four_digit_number_in_a_title_is_not_a_year_request(self):
        """"Blade Runner 2049" must not be read as a year constraint."""
        result, call_qwen = self.call(
            "User: I loved Blade Runner 2049, something similar?",
            {"title": "Notes on a Scandal"},
            qwen="SELECTED_TITLE: Notes on a Scandal\nRESPONSE: here you go",
        )

        call_qwen.assert_called_once()
        self.assertEqual(result, "here you go")

    # ── Test C: rating with no rating metadata ──────────────────────────────
    def test_rating_request_without_rating_metadata_is_deterministic(self):
        for phrasing in (
            "User: something highly rated please",
            "User: I want a highly-rated thriller",
            "User: what is its rating?",
            "User: only well rated films",
            "User: what's its IMDb score?",
        ):
            with self.subTest(phrasing=phrasing):
                result, call_qwen = self.call(
                    phrasing, {"title": "Notes on a Scandal"}
                )
                call_qwen.assert_not_called()
                self.assertIn("verified rating information", result)

    # ── Test D: broad preference request is untouched ───────────────────────
    def test_broad_semantic_request_still_uses_qwen(self):
        result, call_qwen = self.call(
            "User: I want a dark science-fiction movie with action",
            {"title": "Notes on a Scandal"},
            qwen="SELECTED_TITLE: Notes on a Scandal\nRESPONSE: a grounded reply",
        )

        call_qwen.assert_called_once()
        self.assertEqual(result, "a grounded reply")

    def test_other_broad_requests_are_not_caught(self):
        for phrasing in (
            "User: something funny for tonight",
            "User: a slow-burn mystery",
            "User: recommend a 90s horror movie",
            "User: why would I like it?",
        ):
            with self.subTest(phrasing=phrasing):
                _, call_qwen = self.call(
                    phrasing,
                    {"title": "Notes on a Scandal"},
                    qwen="SELECTED_TITLE: Notes on a Scandal\nRESPONSE: ok",
                )
                call_qwen.assert_called_once()

    # ── Test E: verified metadata keeps the normal grounded path ────────────
    def test_present_metadata_keeps_the_normal_qwen_path(self):
        selected = {
            "title": "Sunshine",
            "cast": ["Cillian Murphy", "Chris Evans"],
            "director": "Danny Boyle",
        }
        result, call_qwen = self.call(
            self.LIVE_FAILURE_HISTORY,
            selected,
            qwen="SELECTED_TITLE: Sunshine\nRESPONSE: grounded in real metadata",
        )

        call_qwen.assert_called_once()
        self.assertEqual(result, "grounded in real metadata")

        # The facts the guard accepted as verification are shown to the model.
        user_prompt = call_qwen.call_args.args[0][1]["content"]
        block = user_prompt.split("[Selected Recommendation]\n", 1)[1].strip()
        self.assertIn("Cast: Cillian Murphy, Chris Evans", block)
        self.assertIn("Director: Danny Boyle", block)

    def test_actors_and_release_year_aliases_count_as_verification(self):
        _, call_qwen = self.call(
            "User: who stars in it and what year is it from 2006?",
            {"title": "Sunshine", "actors": "Cillian Murphy", "release_year": 2007},
            qwen="SELECTED_TITLE: Sunshine\nRESPONSE: ok",
        )

        call_qwen.assert_called_once()
        block = call_qwen.call_args.args[0][1]["content"].split(
            "[Selected Recommendation]\n", 1
        )[1]
        self.assertIn("Cast: Cillian Murphy", block)
        self.assertIn("Year: 2007", block)

    def test_partially_verified_constraints_still_gate(self):
        """Cast known, director unknown: only the unverified one is reported."""
        result, call_qwen = self.call(
            self.LIVE_FAILURE_HISTORY,
            {"title": "Sunshine", "cast": ["Cillian Murphy"]},
        )

        call_qwen.assert_not_called()
        self.assertIn("verified director information", result)
        self.assertNotIn("cast or director", result)

    def test_empty_and_unknown_metadata_do_not_count_as_verification(self):
        for selected in (
            {"title": "X", "cast": [], "director": ""},
            {"title": "X", "cast": "Unknown", "director": "unknown"},
            {"title": "X", "cast": None, "director": None},
        ):
            with self.subTest(selected=selected):
                _, call_qwen = self.call(self.LIVE_FAILURE_HISTORY, selected)
                call_qwen.assert_not_called()

    def test_genre_decade_overview_never_verify_a_factual_constraint(self):
        _, call_qwen = self.call(
            self.LIVE_FAILURE_HISTORY,
            {
                "title": "Notes on a Scandal",
                "genre": "Drama",
                "decade": "2000s",
                "overview": "Cillian Murphy is not in this film.",
            },
        )

        call_qwen.assert_not_called()

    # ── Scope of detection ──────────────────────────────────────────────────
    def test_only_the_latest_user_turn_is_inspected(self):
        history = (
            "User: who stars in it?\n"
            "System: I can't confirm that.\n"
            "User: ok, just give me something atmospheric"
        )
        _, call_qwen = self.call(
            history,
            {"title": "Notes on a Scandal"},
            qwen="SELECTED_TITLE: Notes on a Scandal\nRESPONSE: ok",
        )

        call_qwen.assert_called_once()

    def test_guard_applies_in_followup_mode(self):
        result, call_qwen = self.call(
            "User: who directed it?",
            {"title": "Notes on a Scandal"},
            is_followup=True,
        )

        call_qwen.assert_not_called()
        self.assertIn("verified director information", result)

    def test_history_without_user_prefix_does_not_trigger_the_guard(self):
        _, call_qwen = self.call(
            "starring Cillian Murphy",
            {"title": "Notes on a Scandal"},
            qwen="SELECTED_TITLE: Notes on a Scandal\nRESPONSE: ok",
        )

        call_qwen.assert_called_once()

    def test_selected_movie_is_never_modified_by_the_guard(self):
        selected = {"title": "Notes on a Scandal"}
        snapshot = dict(selected)

        self.call(self.LIVE_FAILURE_HISTORY, selected)

        self.assertEqual(selected, snapshot)


class ConstraintDetectionPrecisionTests(unittest.TestCase):
    """The detector must not fire on ambiguous keywords.

    Detection is asserted directly: with a bare-title selection every category
    is unverified, so the returned list is exactly what was detected.
    """

    def detected(self, user_turn):
        return response_generator.unverifiable_constraints(
            f"User: {user_turn}",
            {"title": "Notes on a Scandal"},
        )

    # ── Ambiguous keywords must not fire ────────────────────────────────────
    def test_five_stars_is_not_a_cast_request(self):
        detected = self.detected("I want something rated five stars")

        self.assertNotIn("cast", detected)
        # It is still a rating request.
        self.assertIn("rating", detected)

    def test_beautifully_directed_is_not_a_director_request(self):
        self.assertNotIn(
            "director", self.detected("I want a beautifully directed thriller")
        )

    def test_musical_score_is_not_a_rating_request(self):
        self.assertNotIn(
            "rating", self.detected("I want a movie with a great musical score")
        )

    def test_other_ambiguous_phrasings_stay_quiet(self):
        for phrasing, category in (
            ("a film with five stars", "cast"),
            ("give it five stars", "cast"),
            ("a tightly directed heist movie", "director"),
            ("loved the score in that one", "rating"),
            ("the orchestral score was gorgeous", "rating"),
        ):
            with self.subTest(phrasing=phrasing):
                self.assertNotIn(category, self.detected(phrasing))

    # ── Reliable cues must still fire ───────────────────────────────────────
    def test_who_directed_it_detects_director(self):
        self.assertIn("director", self.detected("Who directed it?"))

    def test_starring_detects_cast(self):
        self.assertIn("cast", self.detected("starring Cillian Murphy"))

    def test_highly_rated_detects_rating(self):
        self.assertIn("rating", self.detected("highly rated movie"))

    def test_year_cue_detects_year(self):
        self.assertIn("year", self.detected("Find me a movie from 2014"))

    def test_bare_year_in_a_title_does_not_detect_year(self):
        self.assertNotIn("year", self.detected("I loved Blade Runner 2049"))

    def test_remaining_reliable_cues(self):
        for phrasing, category in (
            ("who is in the cast?", "cast"),
            ("I like that actor", "cast"),
            ("a film with a great actress", "cast"),
            ("who stars in it?", "cast"),
            ("it stars Cillian Murphy", "cast"),
            ("not directed by Christopher Nolan", "director"),
            ("who is the director?", "director"),
            ("a highly-rated thriller", "rating"),
            ("what is its rating?", "rating"),
            ("what's its IMDb score?", "rating"),
            ("something from the last five years", "year"),
            ("a 2014 movie", "year"),
        ):
            with self.subTest(phrasing=phrasing):
                self.assertIn(category, self.detected(phrasing))


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
