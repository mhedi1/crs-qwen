"""Routing tests for the Flask chat endpoint's previous-options guard.

``web_app/app.py`` cannot be imported directly in a test environment: it needs
Flask and a populated ``TMDB_API_KEY`` at import time.  These tests therefore
reuse the AST-isolation approach already used by
``test_response_generator.FlaskFollowupCallSiteTests``: the relevant top-level
definitions are extracted, executed in a controlled namespace, and driven with
explicit fakes.  That keeps the assertions behavioural (real routing code, real
regex) without importing the web stack.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


APP_PATH = Path(__file__).resolve().parents[1] / "web_app" / "app.py"

# Top-level definitions required to exercise api_chat's routing.
_WANTED = {
    "_GENRE_KEYWORD_MAP",
    "_DECADE_PATTERNS",
    "_extract_genres",
    "_extract_decades",
    "_extract_mentioned_movies",
    "_PREVIOUS_OPTIONS_PATTERN",
    "_references_previous_options",
    "_previous_options_response",
    "RemoteInputTooLongError",
    "CONVERSATION_TOO_LONG_MESSAGE",
    "api_chat",
}


def _node_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {
            target.id
            for target in node.targets
            if isinstance(target, ast.Name)
        }
    return set()


def _load_routing_namespace() -> dict:
    """Execute only the routing-relevant parts of app.py in a fake namespace."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    selected = [node for node in tree.body if _node_names(node) & _WANTED]
    missing = _WANTED - {name for node in selected for name in _node_names(node)}
    if missing:
        raise AssertionError(f"app.py no longer defines: {sorted(missing)}")

    for node in selected:
        # Drop @app.route so api_chat is a plain callable.
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict = {"re": re, "print": lambda *a, **k: None}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


class _FakeSession(dict):
    """Dict with the attribute surface Flask's session exposes to api_chat."""

    modified = False


class PreviousOptionsGuardTests(unittest.TestCase):
    """The guard answers locally and must never reach ranking or a model."""

    def build(self, message: str, *, last_movie=None, classified="NEW_PREFERENCE"):
        namespace = _load_routing_namespace()

        session = _FakeSession()
        session["history"] = []
        session["turn"] = 0
        session["previously_recommended"] = []
        session["mentioned_films"] = []
        if last_movie is not None:
            session["last_movie"] = last_movie

        payloads: list = []

        def fake_jsonify(value):
            payloads.append(value)
            return value

        namespace["session"] = session
        namespace["jsonify"] = fake_jsonify
        namespace["request"] = Mock(get_json=Mock(return_value={"message": message}))
        namespace["classify_intent"] = Mock(return_value=classified)
        namespace["_generate_followup_response"] = Mock(return_value="qwen follow-up")
        namespace["get_recommender"] = Mock(
            side_effect=AssertionError("ranking must not be reached")
        )
        namespace["enrich_with_tmdb"] = Mock(return_value={})
        namespace["traceback"] = Mock()

        result = namespace["api_chat"]()
        return result, namespace, session

    # ── Test 1 ──────────────────────────────────────────────────────────────
    def test_comparative_reference_is_answered_locally_without_ranking(self):
        result, namespace, _ = self.build(
            "Out of those, which one has the best plot twist?",
            last_movie={"title": "Notes on a Scandal"},
        )

        self.assertEqual(result["intent"], "FOLLOW_UP")
        self.assertEqual(result["candidates"], [])
        self.assertIsNone(result["selected_candidate"])

        # No ranking, and no Qwen3 call of any kind: neither the classifier
        # nor the follow-up generator may run for this turn.
        namespace["get_recommender"].assert_not_called()
        namespace["classify_intent"].assert_not_called()
        namespace["_generate_followup_response"].assert_not_called()

        self.assertIn("Notes on a Scandal", result["response"])
        self.assertIn("doesn't compare or re-rank", result["response"])

    def test_guard_requires_an_existing_selection(self):
        """Without last_movie the turn is a real request and must route normally."""
        namespace = _load_routing_namespace()
        session = _FakeSession()
        session["history"] = []
        session["turn"] = 0
        namespace["session"] = session
        namespace["jsonify"] = lambda value: value
        namespace["request"] = Mock(
            get_json=Mock(return_value={"message": "Out of those, which is best?"})
        )
        namespace["classify_intent"] = Mock(return_value="NEW_PREFERENCE")
        namespace["_generate_followup_response"] = Mock()
        namespace["get_recommender"] = Mock(return_value=(None, "unavailable"))
        namespace["enrich_with_tmdb"] = Mock(return_value={})
        namespace["traceback"] = Mock()

        result = namespace["api_chat"]()

        self.assertEqual(result["intent"], "NEW_PREFERENCE")
        namespace["classify_intent"].assert_called_once()

    # ── Test 2 ──────────────────────────────────────────────────────────────
    def test_ordinary_followup_still_uses_existing_behaviour(self):
        result, namespace, _ = self.build(
            "why would I like it?",
            last_movie={"title": "Notes on a Scandal"},
            classified="FOLLOW_UP",
        )

        self.assertEqual(result["intent"], "FOLLOW_UP")
        self.assertEqual(result["response"], "qwen follow-up")
        namespace["classify_intent"].assert_called_once()
        namespace["_generate_followup_response"].assert_called_once()
        namespace["get_recommender"].assert_not_called()

    # ── Test 3 ──────────────────────────────────────────────────────────────
    def test_new_preference_message_is_not_caught_by_the_guard(self):
        namespace = _load_routing_namespace()
        session = _FakeSession()
        session["history"] = []
        session["turn"] = 0
        session["previously_recommended"] = []
        session["last_movie"] = {"title": "Notes on a Scandal"}
        namespace["session"] = session
        namespace["jsonify"] = lambda value: value
        namespace["request"] = Mock(
            get_json=Mock(
                return_value={"message": "Recommend me a 90s horror movie instead."}
            )
        )
        namespace["classify_intent"] = Mock(return_value="NEW_PREFERENCE")
        namespace["_generate_followup_response"] = Mock()
        namespace["get_recommender"] = Mock(return_value=(None, "unavailable"))
        namespace["enrich_with_tmdb"] = Mock(return_value={})
        namespace["traceback"] = Mock()

        result = namespace["api_chat"]()

        self.assertEqual(result["intent"], "NEW_PREFERENCE")
        namespace["classify_intent"].assert_called_once()
        namespace["get_recommender"].assert_called_once()


class InputTooLongRoutingTests(unittest.TestCase):
    """Over-length conversations get an actionable message, never a retry."""

    def run_chat(self, make_recommender):
        """Run api_chat once.

        ``make_recommender`` receives the executed namespace so that exception
        types it raises are the *same* class objects api_chat catches — each
        namespace load creates fresh class objects.
        """
        namespace = _load_routing_namespace()
        recommender = make_recommender(namespace)

        session = _FakeSession()
        session["history"] = []
        session["turn"] = 0
        session["previously_recommended"] = []
        session["mentioned_films"] = []

        namespace["session"] = session
        namespace["jsonify"] = lambda value: value
        namespace["request"] = Mock(
            get_json=Mock(return_value={"message": "recommend me something"})
        )
        namespace["classify_intent"] = Mock(return_value="NEW_PREFERENCE")
        namespace["_generate_followup_response"] = Mock(
            side_effect=AssertionError("NLG must not choose a movie on failure")
        )
        get_recommender = Mock(return_value=(recommender, None))
        namespace["get_recommender"] = get_recommender
        namespace["enrich_with_tmdb"] = Mock(return_value={})
        namespace["traceback"] = Mock()

        result = namespace["api_chat"]()
        return result, namespace, session, get_recommender, recommender

    # ── Test 3 + 4 ──────────────────────────────────────────────────────────
    def test_input_too_long_is_reported_without_retry_or_fallback(self):
        result, namespace, session, get_recommender, recommender = self.run_chat(
            lambda ns: Mock(side_effect=ns["RemoteInputTooLongError"]("too long"))
        )

        self.assertEqual(result["response"], namespace["CONVERSATION_TOO_LONG_MESSAGE"])
        self.assertIn("too long for a new recommendation", result["response"])
        self.assertIn("clear the chat", result["response"])
        self.assertIsNone(result["movie"])
        self.assertEqual(result["candidates"], [])

        # No retry, no second recommender resolution, no NLG movie selection.
        self.assertEqual(recommender.call_count, 1)
        self.assertEqual(get_recommender.call_count, 1)
        namespace["_generate_followup_response"].assert_not_called()

        # Nothing is recorded as recommended.
        self.assertNotIn("last_movie", session)
        self.assertEqual(session["previously_recommended"], [])

    def test_input_too_long_message_does_not_blame_the_model(self):
        namespace = _load_routing_namespace()
        message = namespace["CONVERSATION_TOO_LONG_MESSAGE"].casefold()

        for blame in ("issue with the recommendation model", "error", "failed", "broken"):
            with self.subTest(phrase=blame):
                self.assertNotIn(blame, message)

    # ── Test 2 (client half) ────────────────────────────────────────────────
    def test_other_remote_failures_keep_the_generic_model_error(self):
        result, _, session, _, _ = self.run_chat(
            lambda ns: Mock(
                side_effect=RuntimeError("Remote recommender returned HTTP 500")
            )
        )

        self.assertIn("issue with the recommendation model", result["response"])
        self.assertIsNone(result["movie"])
        self.assertEqual(result["candidates"], [])
        self.assertNotIn("last_movie", session)

    # ── Test 5 ──────────────────────────────────────────────────────────────
    def test_successful_recommendation_path_is_unchanged(self):
        movie = {"title": "Blade Runner 2049", "stage2_rank": 1}
        result, _, session, _, recommender = self.run_chat(
            lambda ns: Mock(
                return_value={
                    "response": "You might enjoy Blade Runner 2049.",
                    "movie": movie,
                    "candidates": [movie],
                    "diagnostics": {"kbrd": {"qwen_fallback_executed": False}},
                }
            )
        )

        self.assertEqual(result["response"], "You might enjoy Blade Runner 2049.")
        self.assertEqual(result["intent"], "NEW_PREFERENCE")
        self.assertEqual(result["movie"]["title"], "Blade Runner 2049")
        self.assertEqual(result["selected_candidate"]["final_rank"], 1)
        self.assertTrue(result["selected_candidate"]["in_final_top5"])
        self.assertEqual(session["last_movie"]["title"], "Blade Runner 2049")
        self.assertEqual(session["previously_recommended"], ["Blade Runner 2049"])
        self.assertEqual(recommender.call_count, 1)


_CLIENT_WANTED = {
    "RemoteInputTooLongError",
    "_recommender",
    "_recommender_error",
    "get_recommender",
}


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


def _load_client_namespace(post):
    """Execute get_recommender with the network and NLG stack faked out."""
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"), filename=str(APP_PATH))
    selected = [node for node in tree.body if _node_names(node) & _CLIENT_WANTED]
    missing = _CLIENT_WANTED - {n for node in selected for n in _node_names(node)}
    if missing:
        raise AssertionError(f"app.py no longer defines: {sorted(missing)}")

    for node in selected:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)

    fake_requests = types.ModuleType("requests")
    fake_requests.post = post

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *a, **k: None

    fake_response_module = types.ModuleType("my_crs.response_generator")
    generate_response = Mock(return_value="a grounded sentence")
    fake_response_module.generate_response = generate_response

    namespace: dict = {"print": lambda *a, **k: None}
    with patch.dict(
        sys.modules,
        {
            "requests": fake_requests,
            "dotenv": fake_dotenv,
            "my_crs.response_generator": fake_response_module,
        },
    ), patch.dict(
        os.environ,
        {
            "REMOTE_RECOMMENDER_URL": "https://gpu.example/",
            "REMOTE_RECOMMENDER_TOKEN": "test-token",
        },
    ):
        exec(compile(module, str(APP_PATH), "exec"), namespace)
        recommender, error = namespace["get_recommender"]()

    return namespace, recommender, error, generate_response


class RemoteRecommenderClientTests(unittest.TestCase):
    """The HTTP wrapper must classify 413 precisely and never retry."""

    HISTORY = [{"role": "user", "content": "something long"}]

    # ── Test 3 + 4 ──────────────────────────────────────────────────────────
    def test_input_too_long_response_raises_without_retry_or_nlg(self):
        post = Mock(
            return_value=_FakeResponse(413, {"error": "input_too_long"})
        )
        namespace, recommender, error, generate_response = _load_client_namespace(post)
        self.assertIsNone(error)

        with self.assertRaises(namespace["RemoteInputTooLongError"]):
            recommender(self.HISTORY)

        # Exactly one request: the wrapper must not retry an over-length input.
        self.assertEqual(post.call_count, 1)
        # No Qwen3 call to pick a different movie.
        generate_response.assert_not_called()

    def test_413_without_the_marker_is_a_generic_failure(self):
        post = Mock(return_value=_FakeResponse(413, {"error": "something_else"}))
        namespace, recommender, _, _ = self.load_ok(post)

        with self.assertRaises(RuntimeError) as caught:
            recommender(self.HISTORY)

        self.assertNotIsInstance(
            caught.exception, namespace["RemoteInputTooLongError"]
        )
        self.assertEqual(post.call_count, 1)

    def test_server_error_is_a_generic_failure(self):
        post = Mock(return_value=_FakeResponse(500, {"error": "recommendation_failed"}))
        namespace, recommender, _, _ = self.load_ok(post)

        with self.assertRaises(RuntimeError) as caught:
            recommender(self.HISTORY)

        self.assertNotIsInstance(
            caught.exception, namespace["RemoteInputTooLongError"]
        )
        self.assertEqual(post.call_count, 1)

    # ── Test 5 ──────────────────────────────────────────────────────────────
    def test_successful_remote_call_is_unchanged(self):
        selected = {"title": "Blade Runner 2049", "stage2_rank": 1}
        post = Mock(
            return_value=_FakeResponse(
                200,
                {
                    "selected_candidate": selected,
                    "ranked_candidates": [selected],
                    "stage1_rrf_top50": [selected],
                    "diagnostics": {"kbrd": {}},
                },
            )
        )
        _, recommender, _, generate_response = self.load_ok(post)

        result = recommender(self.HISTORY, previously_recommended=["Arrival"])

        self.assertEqual(result["movie"], selected)
        self.assertEqual(result["selected_candidate"], selected)
        self.assertEqual(result["candidates"], [selected])
        self.assertEqual(result["response"], "a grounded sentence")
        self.assertEqual(post.call_count, 1)
        generate_response.assert_called_once()

        # The history sent to the scientific recommender is unchanged.
        self.assertEqual(
            post.call_args.kwargs["json"], {"history": "User: something long"}
        )

    def load_ok(self, post):
        namespace, recommender, error, generate_response = _load_client_namespace(post)
        self.assertIsNone(error)
        return namespace, recommender, error, generate_response


class PreviousOptionsPatternTests(unittest.TestCase):
    """The trigger set must stay narrow: no generic comparatives."""

    def setUp(self):
        self.matches = _load_routing_namespace()["_references_previous_options"]

    def test_recognises_only_the_approved_phrases(self):
        for phrase in (
            "Out of those, which one has the best plot twist?",
            "out of these, what should I watch?",
            "Which of those is scariest?",
            "which of these would you pick?",
            "Among those, any with a twist?",
            "among these, which is shortest?",
            "Best of those?",
            "best of these please",
            "Between those, which wins?",
            "between them, which is better?",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(self.matches(phrase))

    def test_ignores_generic_comparatives_and_ordinary_requests(self):
        for phrase in (
            "which one is better?",
            "can you compare them for me?",
            "Blade Runner versus The Matrix",
            "Alien vs Predator",
            "Recommend me a 90s horror movie instead.",
            "why would I like it?",
            "tell me more about it",
            "",
        ):
            with self.subTest(phrase=phrase):
                self.assertFalse(self.matches(phrase))


if __name__ == "__main__":
    unittest.main()
