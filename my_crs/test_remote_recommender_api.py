"""Error-mapping tests for the remote recommender API.

``remote_recommender_api.py`` cannot be imported in a test environment: it needs
Flask, a populated ``REMOTE_RECOMMENDER_TOKEN``, and it loads the frozen
FinalRecommender at import time.  The relevant definitions are therefore
extracted and executed with explicit fakes, matching the approach used by
``test_web_app_routing``.

These tests assert only the sanitized error contract.  Nothing here touches the
frozen recommender: the token-ceiling ValueError is produced by the frozen code
unchanged, and is merely classified.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import Mock


API_PATH = Path(__file__).resolve().parents[1] / "remote_recommender_api.py"

_WANTED = {
    "_INPUT_TOO_LONG_MARKERS",
    "_is_input_too_long",
    "recommend",
}

# The exact message the frozen packer raises, as confirmed on the GPU server.
FROZEN_TOKEN_LIMIT_ERROR = (
    "Tokenized scoring input requires 2383 tokens, exceeding 2304"
)
# The frozen belt-and-braces check in stage2_v2_peft.validate_packed_token_count.
FROZEN_PACKED_EVENT_ERROR = (
    "Packed event requires 2383 tokens, exceeding frozen "
    "Phase-3B ceiling 2304; truncation is forbidden"
)


def _node_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    return set()


def _load_api_namespace(engine):
    tree = ast.parse(API_PATH.read_text(encoding="utf-8"), filename=str(API_PATH))
    selected = [node for node in tree.body if _node_names(node) & _WANTED]
    missing = _WANTED - {name for node in selected for name in _node_names(node)}
    if missing:
        raise AssertionError(
            f"remote_recommender_api.py no longer defines: {sorted(missing)}"
        )

    for node in selected:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace: dict = {}
    exec(compile(module, str(API_PATH), "exec"), namespace)
    namespace["authorized"] = lambda: True
    namespace["jsonify"] = lambda value: value
    namespace["request"] = Mock(
        get_json=Mock(return_value={"history": "User: hello"})
    )
    namespace["engine"] = engine
    namespace["app"] = Mock(logger=Mock())
    return namespace


class TokenCeilingClassificationTests(unittest.TestCase):
    """Only the frozen token-ceiling ValueError may be classified as too long."""

    def setUp(self):
        self.is_too_long = _load_api_namespace(Mock())["_is_input_too_long"]

    def test_recognises_both_frozen_token_ceiling_messages(self):
        self.assertTrue(self.is_too_long(ValueError(FROZEN_TOKEN_LIMIT_ERROR)))
        self.assertTrue(self.is_too_long(ValueError(FROZEN_PACKED_EVENT_ERROR)))

    def test_rejects_other_value_errors_and_other_exception_types(self):
        for exc in (
            ValueError("Stage-2 v2 record schema mismatch"),
            ValueError("Every RRF score must be finite and strictly positive"),
            ValueError(""),
            RuntimeError(FROZEN_TOKEN_LIMIT_ERROR),  # right message, wrong type
            KeyError("history"),
            Exception("boom"),
        ):
            with self.subTest(exc=repr(exc)):
                self.assertFalse(self.is_too_long(exc))


class RecommendErrorMappingTests(unittest.TestCase):
    """HTTP contract: 413 for known over-length input, 500 for everything else."""

    def call_recommend(self, side_effect=None, return_value=None):
        engine = Mock()
        engine.recommend = Mock(side_effect=side_effect, return_value=return_value)
        namespace = _load_api_namespace(engine)
        return namespace["recommend"](), namespace, engine

    # ── Test 1 ──────────────────────────────────────────────────────────────
    def test_token_ceiling_error_maps_to_413_input_too_long(self):
        (body, status), namespace, _ = self.call_recommend(
            side_effect=ValueError(FROZEN_TOKEN_LIMIT_ERROR)
        )

        self.assertEqual(status, 413)
        self.assertEqual(body, {"error": "input_too_long"})

        # Nothing sensitive may leak.
        serialized = repr(body)
        for leak in ("2383", "2304", "ValueError", "Tokenized", "Traceback", "token"):
            with self.subTest(leak=leak):
                self.assertNotIn(leak, serialized)

        # The full traceback is still logged server-side.
        namespace["app"].logger.exception.assert_called_once_with(
            "Recommendation failed"
        )

    def test_packed_event_variant_also_maps_to_413(self):
        (body, status), _, _ = self.call_recommend(
            side_effect=ValueError(FROZEN_PACKED_EVENT_ERROR)
        )

        self.assertEqual(status, 413)
        self.assertEqual(body, {"error": "input_too_long"})

    # ── Test 2 ──────────────────────────────────────────────────────────────
    def test_other_value_error_stays_sanitized_500(self):
        (body, status), namespace, _ = self.call_recommend(
            side_effect=ValueError("Stage-2 v2 candidate serialization digest mismatch")
        )

        self.assertEqual(status, 500)
        self.assertEqual(body, {"error": "recommendation_failed"})
        self.assertNotIn("digest", repr(body))
        namespace["app"].logger.exception.assert_called_once_with(
            "Recommendation failed"
        )

    def test_arbitrary_exception_stays_sanitized_500(self):
        for exc in (
            RuntimeError("CUDA out of memory on device cuda:0"),
            KeyError("selected_candidate"),
            Exception("unexpected"),
        ):
            with self.subTest(exc=repr(exc)):
                (body, status), _, _ = self.call_recommend(side_effect=exc)
                self.assertEqual(status, 500)
                self.assertEqual(body, {"error": "recommendation_failed"})

    # ── Test 5 (server half) ────────────────────────────────────────────────
    def test_successful_recommendation_response_is_unchanged(self):
        payload = {
            "selected_candidate": {"title": "Blade Runner 2049", "stage2_rank": 1},
            "ranked_candidates": [{"title": "Blade Runner 2049"}],
            "stage1_rrf_top50": [{"title": "Blade Runner 2049"}],
            "diagnostics": {"kbrd": {}},
        }
        result, namespace, engine = self.call_recommend(return_value=payload)

        # A successful view returns the body alone, with no status tuple.
        self.assertEqual(
            result,
            {
                "selected_candidate": payload["selected_candidate"],
                "ranked_candidates": payload["ranked_candidates"],
                "stage1_rrf_top50": payload["stage1_rrf_top50"],
                "diagnostics": payload["diagnostics"],
            },
        )
        engine.recommend.assert_called_once_with("User: hello")
        namespace["app"].logger.exception.assert_not_called()


if __name__ == "__main__":
    unittest.main()
