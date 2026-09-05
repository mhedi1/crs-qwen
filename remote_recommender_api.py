import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from my_crs.final_recommender import get_final_recommender

load_dotenv()

TOKEN = os.getenv("REMOTE_RECOMMENDER_TOKEN")
if not TOKEN:
    raise RuntimeError("REMOTE_RECOMMENDER_TOKEN is not configured")

app = Flask(__name__)

print("Loading frozen FinalRecommender...")
engine = get_final_recommender()
engine.ensure_ready()
print("FinalRecommender READY")


# The frozen Stage-2 packing refuses inputs above its token ceiling and raises a
# plain ValueError.  Match on those exact frozen messages so the client can tell
# "conversation too long" apart from a genuine model failure.  The frozen code is
# never modified; this only classifies what it already raises.  Anything that does
# not match falls through to the generic sanitized error.
_INPUT_TOO_LONG_MARKERS = (
    "tokenized scoring input requires",  # joint_rrf_ranker.tokenize_scoring_event
    "packed event requires",             # stage2_v2_peft.validate_packed_token_count
)


def _is_input_too_long(exc: BaseException) -> bool:
    """True only for the frozen Stage-2 token-ceiling ValueError."""
    if not isinstance(exc, ValueError):
        return False
    message = str(exc).casefold()
    return any(marker in message for marker in _INPUT_TOO_LONG_MARKERS)


def authorized():
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {TOKEN}"


@app.get("/health")
def health():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401

    return jsonify({
        "status": "ok",
        "recommender": "KBRD+CKG+RRF+Stage2-v2"
    })


@app.post("/recommend")
def recommend():
    if not authorized():
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    history = (data.get("history") or "").strip()

    if not history:
        return jsonify({"error": "history is required"}), 400

    try:
        result = engine.recommend(history)

        return jsonify({
            "selected_candidate": result["selected_candidate"],
            "ranked_candidates": result["ranked_candidates"],
            "stage1_rrf_top50": result["stage1_rrf_top50"],
            "diagnostics": result["diagnostics"],
        })

    except Exception as exc:
        app.logger.exception("Recommendation failed")

        # Known, expected out-of-range input: report it as such, still without
        # exposing the message, traceback, token counts, or exception class.
        if _is_input_too_long(exc):
            return jsonify({
                "error": "input_too_long",
            }), 413

        return jsonify({
            "error": "recommendation_failed",
        }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=9000,
        debug=False,
        threaded=False,
    )
