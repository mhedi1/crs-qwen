import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, session, jsonify
import traceback
import requests

app = Flask(__name__)
app.secret_key = "crs_thesis_demo_2024"
TMDB_API_KEY = "945e20b8c7b2e5046a046a6e2a1b910c"

TMDB_GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}

_recommender = None
_recommender_error = None


def enrich_with_tmdb(title):
    try:
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={"query": title, "api_key": TMDB_API_KEY},
            timeout=2,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return {}
        hit = results[0]

        genre_ids = hit.get("genre_ids", [])
        genre = ", ".join(
            TMDB_GENRE_MAP[gid] for gid in genre_ids if gid in TMDB_GENRE_MAP
        )

        release_date = hit.get("release_date") or ""
        year_str = release_date[:4]
        decade = (str((int(year_str) // 10) * 10) + "s") if year_str.isdigit() else ""

        poster_path = hit.get("poster_path")
        poster_url = f"https://image.tmdb.org/t/p/w300{poster_path}" if poster_path else None

        return {"genre": genre or None, "decade": decade or None, "poster_url": poster_url}
    except Exception as e:
        print(f"[TMDB] Enrichment failed for '{title}': {e}")
        return {}


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

    # Enrich movie with TMDB poster and metadata (override only if pipeline value missing/Unknown)
    if movie:
        movie.setdefault("poster_url", None)
        if movie.get("title"):
            tmdb = enrich_with_tmdb(movie["title"])
            cur_genre = (movie.get("genre") or "").strip()
            cur_decade = (movie.get("decade") or "").strip()
            if tmdb.get("genre") and (not cur_genre or cur_genre.lower() == "unknown"):
                movie["genre"] = tmdb["genre"]
            if tmdb.get("decade") and (not cur_decade or cur_decade.lower() == "unknown"):
                movie["decade"] = tmdb["decade"]
            movie["poster_url"] = tmdb.get("poster_url")

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
