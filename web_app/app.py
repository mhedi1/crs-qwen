import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, session, jsonify
import traceback
import re
import requests
import yaml
import secrets

from dotenv import load_dotenv
load_dotenv()

_cfg_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "my_crs", "config.yaml"
)
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(16))
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise EnvironmentError(
        "TMDB_API_KEY environment variable is not set. "
        "See .env.example at the project root."
    )

TMDB_GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}

_GENRE_KEYWORD_MAP = {
    "horror": "Horror",       "scary": "Horror",
    "comedy": "Comedy",       "comedic": "Comedy",
    "action": "Action",
    "thriller": "Thriller",
    "romance": "Romance",     "romantic": "Romance",
    "drama": "Drama",         "dramatic": "Drama",
    "sci-fi": "Sci-Fi",       "scifi": "Sci-Fi",
    "fantasy": "Fantasy",
    "mystery": "Mystery",
    "animation": "Animation", "animated": "Animation",
    "documentary": "Documentary",
    "western": "Western",
    "war": "War",
    "crime": "Crime",
    "adventure": "Adventure",
}

_DECADE_PATTERNS = [
    (r"\b50s\b|\b1950s\b|\bfifties\b",               "1950s"),
    (r"\b60s\b|\b1960s\b|\bsixties\b",               "1960s"),
    (r"\b70s\b|\b1970s\b|\bseventies\b",             "1970s"),
    (r"\b80s\b|\b1980s\b|\brighties\b|\beighties\b", "1980s"),
    (r"\b90s\b|\b1990s\b|\bnineties\b",              "1990s"),
    (r"\b00s\b|\b2000s\b",                            "2000s"),
    (r"\b2010s\b",                                    "2010s"),
    (r"\b2020s\b",                                    "2020s"),
]


def _extract_genres(text: str) -> list:
    found = []
    t = text.lower()
    for phrase in ("science fiction", "sci fi", "sci-fi"):
        if phrase in t and "Sci-Fi" not in found:
            found.append("Sci-Fi")
    for token in re.findall(r"[\w'-]+", t):
        genre = _GENRE_KEYWORD_MAP.get(token)
        if genre and genre not in found:
            found.append(genre)
    return found


def _extract_decades(text: str) -> list:
    found = []
    t = text.lower()
    for pattern, decade in _DECADE_PATTERNS:
        if re.search(pattern, t) and decade not in found:
            found.append(decade)
    return found


def _extract_mentioned_movies(text: str) -> list:
    found = []
    for m in re.finditer(r'["“”‘’]([^"\']{3,60})["“”‘’]', text):
        title = m.group(1).strip().title()
        if title not in found:
            found.append(title)
    for m in re.finditer(
        r'\b(?:like|loved?|watch(?:ed)?|seen|saw|enjoy(?:ed)?|similar to|such as|films? like|movies? like)'
        r'\s+([A-Z][A-Za-z0-9 :\'!&\-]{1,50}?)(?=[,\.!?\n]|$|\s+(?:and|but|or|which|because)\b)',
        text,
    ):
        title = m.group(1).strip()
        if len(title) > 2 and title not in found:
            found.append(title)
    for m in re.finditer(r'\b([A-Z][a-z0-9]*(?:\s+(?:[a-z]{1,3}\s+)?[A-Z][a-z0-9]*)+)\b', text):
        title = m.group(1).strip()
        if len(title) > 2 and title not in found:
            found.append(title)
    return found


_recommender = None
_recommender_error = None


def enrich_with_tmdb(title, year=None):
    try:
        # Strip any parenthetical disambiguation before searching TMDB
        search_title = re.sub(r'\s*\(.*?\)', '', title).strip()
        params = {"query": search_title, "api_key": TMDB_API_KEY}
        if year and str(year).isdigit() and len(str(year)) == 4:
            params["primary_release_year"] = str(year)

        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params,
            timeout=_cfg["tmdb"]["timeout"],
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return {}

        # Filter out obscure / wrong-language matches
        results = [r for r in results if r.get("vote_count", 0) > 10]
        if not results:
            return {}

        title_lower = title.strip().lower()

        # 1. Exact title match (case-insensitive)
        hit = next(
            (r for r in results if r.get("title", "").strip().lower() == title_lower),
            None,
        )
        # 2. Partial match — title contained in result or vice versa
        if hit is None:
            hit = next(
                (
                    r for r in results
                    if title_lower in r.get("title", "").strip().lower()
                    or r.get("title", "").strip().lower() in title_lower
                ),
                None,
            )
        # 3. Fallback: highest-popularity result that passed the vote filter
        if hit is None:
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

        raw_rating = hit.get("vote_average")
        rating = round(float(raw_rating), 1) if raw_rating else None
        
        overview = hit.get("overview")

        return {"genre": genre or None, "decade": decade or None, "poster_url": poster_url, "rating": rating, "overview": overview}
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
        from kbrd_adapter import get_kbrd_candidates
        from reranker import rerank
        from response_generator import generate_response

        def custom_recommender(dialogue_history, previously_recommended=None):
            turns = []
            for turn in dialogue_history:
                role = "User" if turn["role"] == "user" else "System"
                turns.append(f"{role}: {turn['content']}")
            dialogue_str = "\n".join(turns)
            
            diagnostics = {}
            # Safely pass diagnostics dictionary so it gets populated
            try:
                candidates, detected_decades = get_kbrd_candidates(dialogue_str, top_k=_cfg["pipeline"]["top_k_candidates"], diagnostics=diagnostics)
            except TypeError:
                # If diagnostics argument is removed in future
                candidates, detected_decades = get_kbrd_candidates(dialogue_str, top_k=_cfg["pipeline"]["top_k_candidates"])
                
            diagnostics["detected_decades"] = detected_decades
            
            selected_movie, _ = rerank(dialogue_str, candidates, era_hints=detected_decades, previously_recommended=previously_recommended)
            response = generate_response(dialogue_str, selected_movie, previously_recommended=previously_recommended)
            
            return {
                "response": response,
                "movie": {
                    "title": selected_movie.get("title", "Unknown"),
                    "genre": selected_movie.get("genre", "Unknown"),
                    "decade": selected_movie.get("decade", "Unknown")
                },
                "candidates": candidates[:5],
                "diagnostics": diagnostics
            }

        _recommender = custom_recommender
        return _recommender, None
    except Exception as custom_e:
        print(f"[WARN] custom_recommender failed to load: {custom_e}. Falling back to default.")
        try:
            from my_crs.recommender import get_recommendation
            _recommender = get_recommendation
            return _recommender, None
        except Exception as e:
            _recommender_error = str(e)
            print(f"[ERROR] Failed to load recommender: {e}")
            traceback.print_exc()
            return None, _recommender_error


_qwen_cfg_cache: dict = {}


def _get_qwen_cfg() -> dict:
    global _qwen_cfg_cache
    if _qwen_cfg_cache:
        return _qwen_cfg_cache
    try:
        cfg_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "my_crs", "config.yaml"
        )
        with open(cfg_path) as f:
            _qwen_cfg_cache = yaml.safe_load(f).get("qwen", {})
    except Exception:
        pass
    return _qwen_cfg_cache


def classify_intent(message: str, history: list) -> str:
    """Return 'FOLLOW_UP' or 'NEW_PREFERENCE'. Defaults to NEW_PREFERENCE on any failure."""
    try:
        cfg = _get_qwen_cfg()
        url = cfg.get("server_url", "")
        model = cfg.get("model", "")
        if not url:
            return "NEW_PREFERENCE"

        recent_lines = []
        for turn in history[-4:]:
            role = "User" if turn["role"] == "user" else "System"
            recent_lines.append(f"{role}: {turn['content'][:200]}")
        recent = "\n".join(recent_lines)

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a dialogue intent classifier for a movie recommender system. "
                    "Classify the user message as either NEW_PREFERENCE or FOLLOW_UP.\n\n"
                    "NEW_PREFERENCE means: the user wants a new movie recommendation. This includes "
                    "requests like 'recommend something', 'give me another', 'I want something different', "
                    "'suggest a movie', 'something else', 'another film'.\n\n"
                    "FOLLOW_UP means: the user is asking about the CURRENT recommendation, "
                    "not requesting a new one. This includes 'why', 'tell me more about it', "
                    "'what is it about', 'who directed it', 'I watched it', 'I liked it'.\n\n"
                    "Reply with exactly one word: NEW_PREFERENCE or FOLLOW_UP"
                )
            },
            {
                "role": "user",
                "content": message
            }
        ]

        resp = requests.post(
            url,
            json={
                "model": model,
                "messages": prompt,
                "stream": cfg.get("stream", False),
                "think": cfg.get("think", False),
                "temperature": cfg.get("temperature", 0),
            },
            timeout=cfg.get("intent_timeout", 10),
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip().upper()
        if "FOLLOW_UP" in text:
            return "FOLLOW_UP"
        return "NEW_PREFERENCE"
    except Exception as e:
        print(f"[Intent] classify_intent failed, defaulting to NEW_PREFERENCE: {e}")
        return "NEW_PREFERENCE"


def _generate_followup_response(dialogue_history: list, last_movie: dict, previously_recommended: list = None) -> str:
    """Generate a conversational follow-up response using the response generator."""
    try:
        my_crs_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_crs"
        )
        if my_crs_path not in sys.path:
            sys.path.insert(0, my_crs_path)
        from response_generator import generate_response
        turns = []
        for turn in dialogue_history:
            role = "User" if turn["role"] == "user" else "System"
            turns.append(f"{role}: {turn['content']}")
        return generate_response("\n".join(turns), last_movie, previously_recommended)
    except Exception as e:
        print(f"[FOLLOW_UP] Response generation failed: {e}")
        title = last_movie.get("title", "the movie I recommended")
        return (
            f"Regarding **{title}** — feel free to ask anything else about it, "
            "or let me know if you'd like a different recommendation."
        )


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
    
    if "turn" not in session:
        session["turn"] = 0
    session["turn"] += 1

    intent = classify_intent(user_message, session["history"])

    history = list(session["history"])
    history.append({"role": "user", "content": user_message})

    # ── FOLLOW_UP path: skip KBRD retrieval and Qwen reranker ──────────────
    if intent == "FOLLOW_UP" and session.get("last_movie"):
        last_movie = session["last_movie"]
        response_text = _generate_followup_response(history, last_movie, session.get("previously_recommended", []))

        history.append({"role": "system", "content": response_text})
        session["history"] = history

        turn_number = session["turn"]

        if "profile" not in session:
            session["profile"] = {
                "genres": [], "decades": [], "mentioned_movies": [],
                "turn": 0, "seed_count": 0, "fallback_used": False
            }
        if "mentioned_films" not in session:
            session["mentioned_films"] = []
        if "previously_recommended" not in session:
            session["previously_recommended"] = []

        profile = session["profile"]
        for g in _extract_genres(user_message):
            if g not in profile["genres"]:
                profile["genres"].append(g)
        for d in _extract_decades(user_message):
            if d not in profile["decades"]:
                profile["decades"].append(d)
        for mv in _extract_mentioned_movies(user_message):
            if mv not in session["mentioned_films"]:
                session["mentioned_films"].append(mv)
        profile["mentioned_movies"] = session["mentioned_films"]
        profile["seed_count"] = len(session["mentioned_films"])
        profile["turn"] = session["turn"]
        session["profile"] = profile
        session.modified = True

        return jsonify({
            "response": response_text,
            "movie": last_movie,
            "candidates": [],
            "selected_candidate": None,
            "turn_number": turn_number,
            "profile": profile,
            "intent": "FOLLOW_UP"
        })

    # ── NEW_PREFERENCE path: run full pipeline ──────────────────────────────
    recommender, load_error = get_recommender()
    if "previously_recommended" not in session:
        session["previously_recommended"] = []

    if recommender is None:
        error_msg = (
            "I encountered an issue with the recommendation model. "
            "Please try again or rephrase your request."
        )
        print(f"[ERROR] Recommender not available: {load_error}")
        history.append({"role": "system", "content": error_msg})
        session["history"] = history
        session.modified = True
        turn_number = session["turn"]
        return jsonify({
            "response": error_msg,
            "movie": None,
            "candidates": [],
            "selected_candidate": None,
            "turn_number": turn_number,
            "intent": "NEW_PREFERENCE"
        })

    selected_candidate = None
    result = {}
    try:
        try:
            result = recommender(history, previously_recommended=session["previously_recommended"])
        except TypeError:
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
            movie["rating"] = tmdb.get("rating")
            movie["overview"] = tmdb.get("overview")

    history.append({"role": "system", "content": response_text})
    session["history"] = history

    turn_number = session["turn"]

    # Profile Update
    if "profile" not in session:
        session["profile"] = {
            "genres": [],
            "decades": [],
            "mentioned_movies": [],
            "turn": 0,
            "seed_count": 0,
            "fallback_used": False
        }

    profile = session["profile"]
    diagnostics = result.get("diagnostics", {})

    for g in _extract_genres(user_message):
        if g not in profile["genres"]:
            profile["genres"].append(g)
    for d in _extract_decades(user_message):
        if d not in profile["decades"]:
            profile["decades"].append(d)
            
    if "mentioned_films" not in session:
        session["mentioned_films"] = []
    for mv in _extract_mentioned_movies(user_message):
        if mv not in session["mentioned_films"]:
            session["mentioned_films"].append(mv)
    profile["mentioned_movies"] = session["mentioned_films"]

    profile["turn"] = session["turn"]
    profile["seed_count"] = len(session["mentioned_films"])
    profile["fallback_used"] = diagnostics.get("weak_seed_fallback", False)

    session["profile"] = profile
    if movie:
        session["last_movie"] = movie
        if movie.get("title") and movie["title"] not in session["previously_recommended"]:
            session["previously_recommended"].append(movie["title"])
    session.modified = True

    return jsonify({
        "response": response_text,
        "movie": movie,
        "candidates": candidates,
        "selected_candidate": selected_candidate,
        "turn_number": turn_number,
        "profile": profile,
        "intent": "NEW_PREFERENCE"
    })


@app.route("/api/classify", methods=["POST"])
def api_classify():
    try:
        data = request.get_json() or {}
        message = (data.get("message") or "").strip()
        history_data = data.get("history")
        
        if history_data is None:
            history_data = session.get("history", [])
            
        if not message:
            return jsonify({"intent": "NEW_PREFERENCE"})
            
        intent = classify_intent(message, history_data)
        return jsonify({"intent": intent})
    except Exception as e:
        print(f"[API Classify Error] {e}")
        return jsonify({"intent": "NEW_PREFERENCE"})


@app.route("/api/clear", methods=["POST"])
def api_clear():
    session["history"] = []
    session["turn"] = 0
    session["mentioned_films"] = []
    session["previously_recommended"] = []
    session["profile"] = {
        "genres": [],
        "decades": [],
        "mentioned_movies": [],
        "turn": 0,
        "seed_count": 0,
        "fallback_used": False,
    }
    session["last_movie"] = None
    session.modified = True
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    _webapp = _cfg.get("webapp", {})
    app.run(
        host=_webapp.get("host", "0.0.0.0"),
        port=_webapp.get("port", 5000),
        debug=_webapp.get("debug", False),
    )
