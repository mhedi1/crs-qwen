import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, session, jsonify
import traceback

app = Flask(__name__)
app.secret_key = "crs_thesis_demo_2024"

_recommender = None
_recommender_error = None


def get_recommender():
    global _recommender, _recommender_error
    if _recommender is not None:
        return _recommender, None
    if _recommender_error is not None:
        return None, _recommender_error
    try:
        from my_crs.recommender import get_recommendation
        _recommender = get_recommendation
        return _recommender, None
    except Exception as e:
        _recommender_error = str(e)
        print(f"[ERROR] Failed to load recommender: {e}")
        traceback.print_exc()
        return None, _recommender_error


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/chat")
def chat():
    if "history" not in session:
        session["history"] = []
    return render_template("chat.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if "history" not in session:
        session["history"] = []

    history = list(session["history"])
    history.append({"role": "user", "content": user_message})

    recommender, load_error = get_recommender()

    if recommender is None:
        error_msg = (
            "I encountered an issue with the recommendation model. "
            "Please try again or rephrase your request."
        )
        print(f"[ERROR] Recommender not available: {load_error}")
        history.append({"role": "system", "content": error_msg})
        session["history"] = history
        session.modified = True
        turn_number = sum(1 for m in history if m["role"] == "system")
        return jsonify({
            "response": error_msg,
            "movie": None,
            "candidates": [],
            "selected_candidate": None,
            "turn_number": turn_number
        })

    selected_candidate = None
    try:
        result = recommender(history)
        response_text = result.get("response", "")
        movie = result.get("movie", None)
        candidates = result.get("candidates", [])  # pipeline already returns top-5

        # Find selected movie's rank within the top-5 KBRD list
        if movie and movie.get("title"):
            sel_title = movie["title"].lower()
            kbrd_rank = None
            in_top5 = False
            for i, c in enumerate(candidates):
                if c.get("title", "").lower() == sel_title:
                    kbrd_rank = i + 1  # 1-indexed
                    in_top5 = True
                    break
            selected_candidate = {
                "title": movie.get("title"),
                "genre": movie.get("genre"),
                "decade": movie.get("decade"),
                "kbrd_rank": kbrd_rank,
                "in_top5": in_top5,
            }
    except Exception as e:
        print(f"[ERROR] Recommender call failed: {e}")
        traceback.print_exc()
        response_text = (
            "I encountered an issue with the recommendation model. "
            "Please try again or rephrase your request."
        )
        movie = None
        candidates = []

    history.append({"role": "system", "content": response_text})
    session["history"] = history
    session.modified = True

    turn_number = sum(1 for m in history if m["role"] == "system")

    return jsonify({
        "response": response_text,
        "movie": movie,
        "candidates": candidates,
        "selected_candidate": selected_candidate,
        "turn_number": turn_number
    })


@app.route("/api/clear", methods=["POST"])
def api_clear():
    session["history"] = []
    session.modified = True
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
