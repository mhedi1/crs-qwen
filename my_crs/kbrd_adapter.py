import os
import pickle
import json
import re
import difflib
import spacy
import requests
from typing import List, Dict, Any
import logging
import warnings
import yaml
from rapidfuzz import fuzz
from reranker import call_qwen

from dotenv import load_dotenv
load_dotenv()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")) as _f:
    _cfg = yaml.safe_load(_f)

_TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
if not _TMDB_API_KEY:
    raise EnvironmentError(
        "TMDB_API_KEY environment variable is not set. "
        "See .env.example at the project root."
    )
_TMDB_TIMEOUT = _cfg["tmdb"]["timeout"]
_TMDB_GENRE_MAP = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}

_tmdb_cache = {}
_TMDB_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "experiments", "tmdb_cache.json"
)

try:
    with open(_TMDB_CACHE_PATH, "r") as f:
        _tmdb_cache = json.load(f)
except Exception:
    _tmdb_cache = {}


def _tmdb_enrich(title, year=None):
    if not _TMDB_API_KEY:
        return {}
    cache_key = f"{title}_{year or ''}"
    if cache_key in _tmdb_cache:
        return _tmdb_cache[cache_key]
    try:
        params = {"query": title, "api_key": _TMDB_API_KEY}
        if year and str(year).isdigit() and len(str(year)) == 4:
            params["primary_release_year"] = str(year)
        resp = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params=params, timeout=_TMDB_TIMEOUT
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        results = [r for r in results if r.get("vote_count", 0) > 10]
        if not results:
            return {}
        title_lower = title.strip().lower()
        hit = next(
            (r for r in results
             if r.get("title", "").strip().lower() == title_lower),
            None
        )
        if hit is None:
            hit = next(
                (r for r in results
                 if title_lower in r.get("title", "").strip().lower()
                 or r.get("title", "").strip().lower() in title_lower),
                None
            )
        if hit is None:
            hit = results[0]
        genre_ids = hit.get("genre_ids", [])
        genre = ", ".join(
            _TMDB_GENRE_MAP[gid] for gid in genre_ids
            if gid in _TMDB_GENRE_MAP
        )
        year_str = hit.get("release_date", "")[:4]
        decade = ""
        if year_str.isdigit():
            decade = str((int(year_str) // 10) * 10) + "s"
        result = {"genre": genre, "decade": decade}
        _tmdb_cache[cache_key] = result
        try:
            with open(_TMDB_CACHE_PATH, "w") as f:
                json.dump(_tmdb_cache, f)
        except Exception:
            pass
        return result
    except Exception:
        return {}


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

def extract_year_from_uri(uri: str) -> str:
    """
    Extract year from DBpedia URI.
    Examples:
    It_(2017_film) -> '2017'
    Scream_(1996_film) -> '1996'
    The_Conjuring -> None
    """
    import re
    if not uri:
        return None
    match = re.search(r'_\((\d{4})_film\)', uri)
    if match:
        return match.group(1)
    match = re.search(r'_\((\d{4})\)', uri)
    if match:
        return match.group(1)
    return None

GENRE_KEYWORDS = {
    "Horror": ["horror", "scary", "ghost", "zombie", "vampire", "slasher", "haunting", "exorcist", "amityville"],
    "Comedy": ["comedy", "funny", "humor", "laugh"],
    "Animation": ["animated", "animation", "pixar", "disney"],
    "Action": ["action", "superhero", "marvel", "avengers", "batman", "spider"],
    "Drama": ["drama"],
    "Sci-Fi": ["science_fiction", "sci-fi", "space", "alien", "robot"],
    "Crime": ["crime", "gangster", "mafia", "thriller"],
    "Romance": ["romance", "romantic", "love"]
}

KNOWN_MOVIE_GENRES = {
    "halloween": "Horror",
    "carrie": "Horror",
    "psycho": "Horror",
    "jaws": "Horror",
    "the thing": "Horror",
    "alien": "Sci-Fi",
    "blade runner": "Sci-Fi",
    "star wars": "Sci-Fi",
    "the godfather": "Crime",
    "goodfellas": "Crime",
    "scarface": "Crime",
    "toy story": "Animation",
    "finding nemo": "Animation",
    "the lion king": "Animation",
    "titanic": "Romance",
    "pretty woman": "Romance",
    "die hard": "Action",
    "terminator": "Action",
    "rocky": "Drama",
    "forrest gump": "Drama"
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KBRD_REPO_PATH = os.path.normpath(
    os.path.join(CURRENT_DIR, "..", "baseline_repo", "KBRD_project", "KBRD")
)

GENRE_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
    'experiments', 'improved_ekg', 'genre_cache.json'
)

_genre_cache = {}

def _load_genre_cache():
    global _genre_cache
    if _genre_cache:
        return
    if os.path.exists(GENRE_CACHE_PATH):
        with open(GENRE_CACHE_PATH, 'r') as f:
            _genre_cache = json.load(f)
        logger.info(
            f"[KBRD Adapter] Loaded genre cache: "
            f"{len(_genre_cache)} entries"
        )
    else:
        logger.warning(
            "[KBRD Adapter] Genre cache not found. "
            "Genre will be Unknown for all candidates."
        )

_data_loaded = False
_has_error = False
_id2entity = None
_movie_ids = None
_entity2id = None
_movie_title_to_id = {}
_movie_title_to_ids = {}
_RESOLVER_VERSION = _cfg["extraction"].get("resolver_version", "v1")


def get_fallback_candidates(top_k: int) -> List[Dict[str, Any]]:
    return [
        {"id": 101, "title": "The Matrix (1999)", "genre": "Sci-Fi", "decade": "1990s"},
        {"id": 102, "title": "The Exorcist (1973)", "genre": "Horror", "decade": "1970s"},
        {"id": 103, "title": "Shrek (2001)", "genre": "Animation", "decade": "2000s"},
        {"id": 104, "title": "Die Hard (1988)", "genre": "Action", "decade": "1980s"},
        {"id": 105, "title": "The Hangover (2009)", "genre": "Comedy", "decade": "2000s"},
    ][:top_k]


def _load_kbrd_resources() -> None:
    global _data_loaded, _has_error, _id2entity, _movie_ids, _entity2id, _movie_title_to_id, _movie_title_to_ids

    if _data_loaded or _has_error:
        return

    logger.info("[KBRD Adapter] Loading processed KBRD resources...")

    data_dir = os.path.join(KBRD_REPO_PATH, "data", "redial")
    entity2id_path = os.path.join(data_dir, "entity2entityId.pkl")
    movie_ids_path = os.path.join(data_dir, "movie_ids.pkl")

    try:
        with open(entity2id_path, "rb") as f:
            _entity2id = pickle.load(f)

        with open(movie_ids_path, "rb") as f:
            _movie_ids = pickle.load(f)

        _id2entity = {v: k for k, v in _entity2id.items()}

        # Build fast lowercased lookup for movies to enable exact and fuzzy matching
        for mid in _movie_ids:
            uri = _id2entity.get(mid)
            if uri:
                clean_t = _clean_title(uri).lower()
                if _is_valid_movie_title(clean_t):
                    clean_lookup = re.sub(r"[^\w\s]", "", clean_t).strip()
                    _movie_title_to_id[clean_lookup] = mid
                    _movie_title_to_ids.setdefault(clean_lookup, []).append(mid)

        _data_loaded = True

        logger.info(f"[KBRD Adapter] Loaded {len(_id2entity)} entities and {len(_movie_ids)} movie ids.")
        _load_genre_cache()

    except Exception as e:
        logger.error(f"[KBRD Adapter ERROR] Could not load KBRD resources: {e}")
        _has_error = True


def _clean_title(entity_uri: str) -> str:
    match = re.search(r"resource/(.+)>", str(entity_uri))
    title = match.group(1) if match else str(entity_uri)

    title = title.replace("_", " ")
    title = re.sub(r"\s*\(.*?\)", "", title)
    title = title.strip()

    return title


def _is_valid_movie_title(title: str) -> bool:
    t_lower = title.lower()

    if t_lower.startswith("list of ") or t_lower.startswith("category: "):
        return False

    if "(film series)" in t_lower or "(tv series)" in t_lower:
        return False

    if "(franchise)" in t_lower:
        return False

    blocklist = ["carriers", "films", "movies", "cinema"]
    if t_lower in blocklist:
        return False

    return True


def _extract_year(text: str) -> str:
    """
    Extracts the release year from a DBpedia URI or title string based on a strict priority order
    to prevent false positives from generic numbers in the URL.
    Valid years are between 1880 and 2030.
    """
    text_str = str(text)

    text_lower = text_str.lower()

    # 1. Skip if URI contains novel, book, or character
    if "novel" in text_lower or "book" in text_lower or "character" in text_lower:
        return ""

    year = ""

    # 2. If it contains "_film", strictly require the year to be adjacent to it
    if "_film" in text_lower:
        match_1 = re.search(r"\((18[8-9]\d|19\d{2}|20[0-2]\d|2030)_film\)", text_str, re.IGNORECASE)
        if match_1:
            year = match_1.group(1)
    else:
        # 3. Existing priority order for all other cases
        match_2 = re.search(r"\((18[8-9]\d|19\d{2}|20[0-2]\d|2030)\)", text_str)
        if match_2:
            year = match_2.group(1)
        else:
            match_3 = re.search(r"_(18[8-9]\d|19\d{2}|20[0-2]\d|2030)_", text_str)
            if match_3:
                year = match_3.group(1)

    # 4. Minimum threshold check: if year is before 1900, discard it
    if year and int(year) >= 1900:
        return year

    return ""


def _year_to_decade(year: str) -> str:
    if not year:
        return "Unknown"
    decade = int(year[:3]) * 10
    return f"{decade}s"


def _infer_genre(uri: str, title: str) -> str:
    """
    Infers the genre of a movie.
    First checks a hardcoded dictionary for exact title matches (ignoring years).
    Then falls back to keyword matching in the DBpedia URI and clean title.
    Returns the first matching genre, or 'Unknown' if nothing matches.
    """
    # 1. First layer: Check known movie genres by cleaned title
    # Clean the title: lowercase and strip out any trailing (YYYY) years
    clean_title = title.lower()
    clean_title = re.sub(r"\s*\(\d{4}\)", "", clean_title).strip()

    if clean_title in KNOWN_MOVIE_GENRES:
        return KNOWN_MOVIE_GENRES[clean_title]

    # 2. Second layer: Keyword matching
    combined_str = (str(uri) + " " + str(title)).lower()

    for genre, keywords in GENRE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in combined_str:
                return genre

    return "Unknown"


_kbrd_agent = None

def _load_kbrd_model():
    global _kbrd_agent, _has_error
    if _kbrd_agent is not None or _has_error:
        return

    try:
        import sys
        import torch
        if KBRD_REPO_PATH not in sys.path:
            sys.path.insert(0, KBRD_REPO_PATH)

        from parlai.core.agents import create_agent

        no_cuda = not torch.cuda.is_available()
        if no_cuda:
            logger.info("[KBRD Neural] CUDA not available â€” loading model on CPU")
        logger.info("[KBRD Neural] Loading model from saved/kbrd_model_retrained")
        opt = {
            'model_file': os.path.join(KBRD_REPO_PATH, 'saved', 'kbrd_model_retrained'),
            'datatype': 'test',
            'datapath': os.path.join(KBRD_REPO_PATH, 'data'),
            'no_cuda': no_cuda,
            'override': {
                'model_file': os.path.join(KBRD_REPO_PATH, 'saved', 'kbrd_model_retrained'),
                'datapath': os.path.join(KBRD_REPO_PATH, 'data'),
                'no_cuda': no_cuda,
            }
        }
        _kbrd_agent = create_agent(opt, requireModelExists=True)
    except Exception as e:
        logger.error(f"[KBRD Neural ERROR] Failed to load model: {e}")
        _has_error = True


def _enrich_candidate(candidate):
    """Augment a candidate dict with genre, director, and year from the cache.

    Args:
        candidate: Candidate dict with at least an "id" key.

    Returns:
        The same dict updated in place with any available cache data.
    """
    eid = str(candidate.get('id', ''))
    if eid in _genre_cache:
        entry = _genre_cache[eid]
        genres_clean = entry.get('genres_clean', [])
        if genres_clean and \
           candidate.get('genre', 'Unknown') == 'Unknown':
            candidate['genre'] = ', '.join(genres_clean)
        if entry.get('directors') and \
           not candidate.get('director'):
            candidate['director'] = \
                entry['directors'][0]
        if entry.get('year') and \
           not candidate.get('year'):
            candidate['year'] = str(entry['year'])

    if not candidate.get('genre') or candidate.get('genre') == 'Unknown':
        tmdb = _tmdb_enrich(
            candidate.get('title', ''),
            candidate.get('year')
        )
        if tmdb.get('genre'):
            candidate['genre'] = tmdb['genre']

    if not candidate.get('decade') or candidate.get('decade') == 'Unknown':
        tmdb_data = _tmdb_enrich(
            candidate.get('title', ''),
            candidate.get('year')
        )
        if tmdb_data.get('decade'):
            candidate['decade'] = tmdb_data['decade']

    return candidate


_MAX_FUSED_CANDIDATES = _cfg["pipeline"]["max_fused_candidates"]
_KBRD_TOP_PRESERVED   = _cfg["pipeline"]["kbrd_top_preserved"]
_WEAK_SEED_THRESHOLD  = _cfg["pipeline"]["weak_seed_threshold"]
_GENRE_BOOST_FACTOR   = _cfg["pipeline"]["genre_boost_factor"]
_USE_SEED_FUSION = _cfg["pipeline"].get("use_seed_fusion", True)
_USE_QWEN_SEED_FALLBACK = _cfg["pipeline"].get("use_qwen_seed_fallback", True)
_USE_QWEN_CANDIDATE_FUSION = _cfg["pipeline"].get("use_qwen_candidate_fusion", True)

_FUZZY_CUTOFF_ENTITY  = _cfg["extraction"]["fuzzy_cutoff_entity"]
_FUZZY_CUTOFF_TITLE   = _cfg["extraction"]["fuzzy_cutoff_title"]
_PERSON_MATCH_THRESH  = _cfg["extraction"]["person_match_threshold"]
_SPACY_MODEL          = _cfg["extraction"]["spacy_model"]
_N_QWEN_FUSION_TITLES = _cfg["extraction"]["n_qwen_fusion_titles"]


def _seed_id_to_movie_candidate(seed_id: int, source_label: str) -> dict:
    """Convert a single entity/seed ID to a movie candidate dict, or return None.

    Verifies that the seed_id belongs to the KBRD movie candidate space
    (_movie_ids), then builds the candidate dict using the same helpers
    used for normal KBRD neural candidates.

    Args:
        seed_id: The DBpedia entity integer ID.
        source_label: One of 'SEED_FUSION' or 'QWEN_FUSION'.

    Returns:
        A candidate dict on success, or None if the seed does not correspond
        to a valid movie candidate.
    """
    # Gate 1: must be in KBRD movie candidate space (not genre/person/generic).
    if _movie_ids is None or seed_id not in _movie_ids:
        return None

    entity_uri = _id2entity.get(seed_id)
    if not entity_uri:
        return None

    title = _clean_title(entity_uri)
    # Gate 2: title must be non-trivial and pass the validity filter.
    if not title or title.strip().isdigit() or len(title.strip()) < 2:
        return None
    if not _is_valid_movie_title(title):
        return None

    year = _extract_year(entity_uri)
    uri_string = entity_uri  # already the full URI string
    c = {
        "id": int(seed_id),
        "title": title,
        "genre": _infer_genre(entity_uri, title),
        "decade": _year_to_decade(year),
        "source": source_label,
        "uri": uri_string,
        "year": extract_year_from_uri(uri_string),
    }
    c = _enrich_candidate(c)
    return c


def _qwen_title_to_movie_candidate(title: str) -> dict:
    """Resolve a Qwen-suggested title string to a verified movie candidate dict.

    Performs exact lookup then fuzzy matching against _movie_title_to_id,
    then verifies the resolved ID is in _movie_ids before building the dict.

    Args:
        title: A raw movie title string as returned by Qwen.

    Returns:
        A candidate dict on success, or None if no valid movie can be resolved.
    """
    clean_t = re.sub(r"[^\w\s]", "", title.lower()).strip()

    # Exact lookup first.
    mid = _movie_title_to_id.get(clean_t)
    if mid is None:
        # Fuzzy fallback with a high cutoff to avoid false positives.
        matches = difflib.get_close_matches(
            clean_t, _movie_title_to_id.keys(), n=1, cutoff=_FUZZY_CUTOFF_TITLE
        )
        if matches:
            mid = _movie_title_to_id[matches[0]]

    if mid is None:
        return None

    return _seed_id_to_movie_candidate(mid, source_label="QWEN_FUSION")


def get_kbrd_candidates(
    dialogue: str,
    top_k: int = 5,
    diagnostics: dict = None,
    use_fusion: bool = True,
    retrieval_mode: str = "legacy",
) -> tuple:
    """Extract seeds from dialogue and get top-K movie candidates via KBRD neural.

    Args:
        dialogue: The conversation history text.
        top_k: Number of candidates to return.
        diagnostics: Optional dictionary to update with metadata/instrumentation.
        use_fusion: Legacy flag to disable direct candidate fusion.
        retrieval_mode: V2 retrieval mode ('legacy', 'kbrd', 'seed_fusion', 'full').

    Returns:
        tuple: (list of candidate dicts, list of detected decades)
    """

    # Resolve runtime overrides
    run_qwen_fallback = _USE_QWEN_SEED_FALLBACK
    run_seed_fusion = _USE_SEED_FUSION
    run_qwen_fusion = _USE_QWEN_CANDIDATE_FUSION

    if retrieval_mode == "kbrd":
        run_qwen_fallback = False
        run_seed_fusion = False
        run_qwen_fusion = False
    elif retrieval_mode == "seed_fusion":
        run_qwen_fallback = False
        run_seed_fusion = True
        run_qwen_fusion = False
    elif retrieval_mode == "full":
        run_qwen_fallback = True
        run_seed_fusion = True
        run_qwen_fusion = True

    if not use_fusion:
        # Legacy compatibility: only disables Candidate Fusion injection.
        run_seed_fusion = False
        run_qwen_fusion = False

    logger.info(f"\n{'=' * 50}")
    logger.info("[KBRD Neural] Starting Neural KBRD Candidate Generation")
    logger.info(f"{'=' * 50}")

    _load_kbrd_resources()
    _load_kbrd_model()

    if _has_error or not _data_loaded or _kbrd_agent is None:
        logger.warning("[KBRD Neural] Falling back because model or resources are unavailable.")
        if diagnostics is not None:
            diagnostics.update({
                "extracted_seeds": [], "qwen_fallback_seeds": [],
                # Historical compatibility keys
                "seed_entity_ids": [],
                "num_extracted_seeds": 0,
                "num_matched_seeds": 0,
                # New explicit provenance keys
                "num_dialogue_seed_ids": 0, "num_qwen_seed_ids": 0,
                "dialogue_seed_entity_ids": [], "qwen_seed_entity_ids": [],
                "qwen_fallback_titles": [],
                "qwen_fallback_executed": False,
                "weak_seed_fallback": False,
                "filtered_noisy_seeds": [],
                "num_filtered_noisy_seeds": 0,
                "num_fused_seed_candidates": 0,
                "num_fused_qwen_candidates": 0,
                "fused_candidate_titles": [],
                "candidate_sources": {},
            })
        return get_fallback_candidates(top_k), []

    seed_list, detected_decades, detected_phrases, filtered_1grams = prepare_input(dialogue)
    dialogue_seed_ids = list(seed_list)

    _seeds_before_fallback = len(dialogue_seed_ids)
    _weak_seed_fallback = _seeds_before_fallback < _WEAK_SEED_THRESHOLD
    _qwen_fallback_executed = False
    qwen_seed_ids: list = []
    _qwen_titles: list = []

    if _weak_seed_fallback and run_qwen_fallback:
        _qwen_fallback_executed = True
        logger.warning("[KBRD Adapter] Weak seeds detected, using Qwen fallback")
        try:
            prompt = (
                "Based on this movie recommendation conversation, \n"
                f"name exactly {_N_QWEN_FUSION_TITLES} well-known movies that match what \n"
                "the user is looking for. Return only movie titles, \n"
                "one per line, nothing else.\n"
                "\n"
                "Conversation:\n"
                f"{dialogue}"
            )
            content = call_qwen(prompt)
            titles = [t.strip() for t in content.split('\n') if t.strip()][:_N_QWEN_FUSION_TITLES]
            _qwen_titles = titles
            logger.debug(f"[KBRD Adapter] Qwen suggested seeds: {', '.join(titles)}")

            added_count = 0
            for title in titles:
                clean_t = re.sub(r"[^\w\s]", "", title.lower()).strip()
                if clean_t in _movie_title_to_id:
                    mid = _movie_title_to_id[clean_t]
                    if mid not in qwen_seed_ids and mid not in dialogue_seed_ids:
                        qwen_seed_ids.append(mid)
                        added_count += 1
                else:
                    matches = difflib.get_close_matches(clean_t, _movie_title_to_id.keys(), n=1, cutoff=_FUZZY_CUTOFF_TITLE)
                    if matches:
                        mid = _movie_title_to_id[matches[0]]
                        if mid not in qwen_seed_ids and mid not in dialogue_seed_ids:
                            qwen_seed_ids.append(mid)
                            added_count += 1

            if added_count > 0:
                logger.debug(f"[KBRD Adapter] Added {added_count} semantic seed entities")
        except Exception as e:
            logger.error(f"[KBRD Adapter] Qwen fallback error: {e}")

    inference_seed_ids = dialogue_seed_ids + qwen_seed_ids

    if not inference_seed_ids:
        logger.warning("[KBRD Neural] No entities detected in dialogue. Using fallback.")
        if diagnostics is not None:
            diagnostics.update({
                "extracted_seeds": detected_phrases,
                "qwen_fallback_seeds": _qwen_titles,
                # Historical compatibility keys
                "seed_entity_ids": inference_seed_ids if inference_seed_ids else [],
                "num_extracted_seeds": _seeds_before_fallback,
                "num_matched_seeds": len(inference_seed_ids) if inference_seed_ids else 0,
                # New explicit provenance keys
                "num_dialogue_seed_ids": len(dialogue_seed_ids),
                "num_qwen_seed_ids": len(qwen_seed_ids),
                "dialogue_seed_entity_ids": dialogue_seed_ids,
                "qwen_seed_entity_ids": qwen_seed_ids,
                "qwen_fallback_titles": _qwen_titles,
                "qwen_fallback_executed": _qwen_fallback_executed,
                "weak_seed_fallback": _weak_seed_fallback,
                "filtered_noisy_seeds": filtered_1grams,
                "num_filtered_noisy_seeds": len(filtered_1grams),
                "num_fused_seed_candidates": 0,
                "num_fused_qwen_candidates": 0,
                "fused_candidate_titles": [],
                "candidate_sources": {},
            })
        return get_fallback_candidates(top_k), detected_decades

    logger.info("[KBRD Neural] Running inference...")

    import torch
    use_cuda = getattr(_kbrd_agent, 'use_cuda', False) and torch.cuda.is_available()
    seed_sets = [inference_seed_ids]
    labels = torch.zeros(1, dtype=torch.long)
    if use_cuda:
        labels = labels.cuda()

    with torch.no_grad():
        _kbrd_agent.model.eval()
        return_dict = _kbrd_agent.model(seed_sets, labels)
        scores = return_dict["scores"].cpu()[0]

    movie_ids = _kbrd_agent.movie_ids
    movie_scores = scores[torch.LongTensor(movie_ids)]

    # Over-sample generously to have a buffer for both KBRD top-30 and tail slots.
    # top_k * 4 ensures enough raw indices even after title-validity filtering.
    fetch_k = min(top_k * 4, len(movie_ids))
    topk_scores, topk_indices = torch.topk(movie_scores, k=fetch_k)

    # --- Collect all valid KBRD candidates (up to top_k) in ranked order ---
    kbrd_candidates: list = []
    for score, idx in zip(topk_scores.tolist(), topk_indices.tolist()):
        if len(kbrd_candidates) >= top_k:
            break

        movie_id = movie_ids[idx]
        entity_uri = _id2entity.get(movie_id)
        if not entity_uri:
            continue

        title = _clean_title(entity_uri)
        if not title or title.strip().isdigit() or len(title.strip()) < 2:
            continue

        if not _is_valid_movie_title(title):
            continue

        year = _extract_year(entity_uri)
        uri_string = _id2entity.get(movie_id, '')
        c = {
            "id": int(movie_id),
            "title": title,
            "genre": _infer_genre(entity_uri, title),
            "decade": _year_to_decade(year),
            "source": "KBRD_NEURAL",
            "uri": uri_string,
            "year": extract_year_from_uri(uri_string),
        }
        c = _enrich_candidate(c)
        kbrd_candidates.append(c)

        if len(kbrd_candidates) == 1:
            logger.info(f"[KBRD Neural] Top candidate: {title} (score: {score:.4f})")

    if not kbrd_candidates:
        logger.warning("[KBRD Neural] No valid candidates after filtering. Using fallback.")
        if diagnostics is not None:
            diagnostics.update({
                "extracted_seeds": detected_phrases,
                "qwen_fallback_seeds": _qwen_titles,
                # Historical compatibility keys
                "seed_entity_ids": inference_seed_ids,
                "num_extracted_seeds": _seeds_before_fallback,
                "num_matched_seeds": len(inference_seed_ids),
                # New explicit provenance keys
                "num_dialogue_seed_ids": len(dialogue_seed_ids),
                "num_qwen_seed_ids": len(qwen_seed_ids),
                "dialogue_seed_entity_ids": dialogue_seed_ids,
                "qwen_seed_entity_ids": qwen_seed_ids,
                "qwen_fallback_titles": _qwen_titles,
                "qwen_fallback_executed": _qwen_fallback_executed,
                "weak_seed_fallback": _weak_seed_fallback,
                "filtered_noisy_seeds": filtered_1grams,
                "num_filtered_noisy_seeds": len(filtered_1grams),
                "num_fused_seed_candidates": 0,
                "num_fused_qwen_candidates": 0,
                "fused_candidate_titles": [],
                "candidate_sources": {},
            })
        return get_fallback_candidates(top_k), detected_decades

    # -----------------------------------------------------------------------
    # Candidate Fusion (skipped when use_fusion=False or mode=kbrd)
    # -----------------------------------------------------------------------
    if not run_seed_fusion and not run_qwen_fusion:
        logger.info("[KBRD Neural] Fusion disabled â€” returning pure KBRD candidates.")
        pure_candidates = kbrd_candidates[:top_k]
        if diagnostics is not None:
            candidate_sources_pure: dict = {}
            for c in pure_candidates:
                src = c.get("source", "UNKNOWN")
                candidate_sources_pure[src] = candidate_sources_pure.get(src, 0) + 1
            diagnostics.update({
                "extracted_seeds": detected_phrases,
                "qwen_fallback_seeds": _qwen_titles,
                # Historical compatibility keys
                "seed_entity_ids": inference_seed_ids,
                "num_extracted_seeds": _seeds_before_fallback,
                "num_matched_seeds": len(inference_seed_ids),
                # New explicit provenance keys
                "num_dialogue_seed_ids": len(dialogue_seed_ids),
                "num_qwen_seed_ids": len(qwen_seed_ids),
                "dialogue_seed_entity_ids": dialogue_seed_ids,
                "qwen_seed_entity_ids": qwen_seed_ids,
                "qwen_fallback_titles": _qwen_titles,
                "qwen_fallback_executed": _qwen_fallback_executed,
                "weak_seed_fallback": _weak_seed_fallback,
                "filtered_noisy_seeds": filtered_1grams,
                "num_filtered_noisy_seeds": len(filtered_1grams),
                "num_fused_seed_candidates": 0,
                "num_fused_qwen_candidates": 0,
                "fused_candidate_titles": [],
                "candidate_sources": candidate_sources_pure,
            })
        return pure_candidates, detected_decades

    # -----------------------------------------------------------------------
    # Candidate Fusion
    # Strategy: KBRD top-30 (preserved) + fused candidates + KBRD tail = top_k
    # -----------------------------------------------------------------------

    # Build lookup sets from ALL kbrd_candidates (not just top-30) to allow
    # proper deduplication across the full KBRD list before fusion.
    seen_ids: set = {c["id"] for c in kbrd_candidates}
    seen_norm_titles: set = {
        re.sub(r"[^\w\s]", "", c["title"].lower()).strip()
        for c in kbrd_candidates
    }

    fused_candidates: list = []
    num_fused_seed = 0
    num_fused_qwen = 0

    # --- Fuse movie-only seeds (extracted from dialogue history only) ---
    if run_seed_fusion:
        for seed_id in dialogue_seed_ids:
            if len(fused_candidates) >= _MAX_FUSED_CANDIDATES:
                break
            # Skip if already in KBRD candidate list (keep original KBRD position).
            if seed_id in seen_ids:
                continue
            candidate = _seed_id_to_movie_candidate(seed_id, source_label="SEED_FUSION")
            if candidate is None:
                continue
            norm_t = re.sub(r"[^\w\s]", "", candidate["title"].lower()).strip()
            if norm_t in seen_norm_titles:
                continue  # deduplicate by normalized title as well
            fused_candidates.append(candidate)
            seen_ids.add(seed_id)
            seen_norm_titles.add(norm_t)
            num_fused_seed += 1
            logger.info(f"[Fusion] Injecting SEED_FUSION candidate: {candidate['title']}")

    # --- Fuse Qwen-suggested titles (only when weak-seed fallback triggered) ---
    if run_qwen_fusion and run_qwen_fallback and _weak_seed_fallback:
        for qwen_title in _qwen_titles:
            if len(fused_candidates) >= _MAX_FUSED_CANDIDATES:
                break
            candidate = _qwen_title_to_movie_candidate(qwen_title)
            if candidate is None:
                continue
            if candidate["id"] in seen_ids:
                continue  # already present in KBRD or seed fusion
            norm_t = re.sub(r"[^\w\s]", "", candidate["title"].lower()).strip()
            if norm_t in seen_norm_titles:
                continue
            fused_candidates.append(candidate)
            seen_ids.add(candidate["id"])
            seen_norm_titles.add(norm_t)
            num_fused_qwen += 1
            logger.info(f"[Fusion] Injecting QWEN_FUSION candidate: {candidate['title']}")

    fused_titles = [c["title"] for c in fused_candidates]
    logger.info(
        f"[Fusion] Fused {num_fused_seed} seed + {num_fused_qwen} Qwen candidates"
        f" (total fused: {len(fused_candidates)})"
    )

    # --- Assemble final list: KBRD top-preserved + fused + KBRD tail ---
    # Top-K preserved candidates kept exactly in original KBRD order.
    kbrd_top30 = kbrd_candidates[:_KBRD_TOP_PRESERVED]

    # KBRD tail: candidates beyond kbrd_top_preserved that were NOT fused (by id).
    fused_ids = {c["id"] for c in fused_candidates}
    kbrd_tail = [
        c for c in kbrd_candidates[_KBRD_TOP_PRESERVED:]
        if c["id"] not in fused_ids
    ]

    # Remaining slots after top-30 and fused block.
    n_fused = len(fused_candidates)
    tail_slots = max(0, top_k - len(kbrd_top30) - n_fused)
    kbrd_tail_fill = kbrd_tail[:tail_slots]

    final_candidates = kbrd_top30 + fused_candidates + kbrd_tail_fill

    # Safety assertion: never exceed top_k.
    if len(final_candidates) > top_k:
        final_candidates = final_candidates[:top_k]

    # Build candidate_sources summary dict.
    candidate_sources: dict = {}
    for c in final_candidates:
        src = c.get("source", "UNKNOWN")
        candidate_sources[src] = candidate_sources.get(src, 0) + 1

    logger.info(
        f"[Fusion] Final candidate list: {len(final_candidates)} total | "
        + ", ".join(f"{s}={n}" for s, n in candidate_sources.items())
    )

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    if diagnostics is not None:
        candidate_sources: dict = {}
        for c in final_candidates:
            src = c.get("source", "UNKNOWN")
            candidate_sources[src] = candidate_sources.get(src, 0) + 1

        diagnostics.update({
            "extracted_seeds": detected_phrases,
            "qwen_fallback_seeds": _qwen_titles,
            # Historical compatibility keys
            "seed_entity_ids": inference_seed_ids,
            "num_extracted_seeds": _seeds_before_fallback,
            "num_matched_seeds": len(inference_seed_ids),
            # New explicit provenance keys
            "num_dialogue_seed_ids": len(dialogue_seed_ids),
            "num_qwen_seed_ids": len(qwen_seed_ids),
            "dialogue_seed_entity_ids": dialogue_seed_ids,
            "qwen_seed_entity_ids": qwen_seed_ids,
            "qwen_fallback_titles": _qwen_titles,
            "qwen_fallback_executed": _qwen_fallback_executed,
            "weak_seed_fallback": _weak_seed_fallback,
            "filtered_noisy_seeds": filtered_1grams,
            "num_filtered_noisy_seeds": len(filtered_1grams),
            "num_fused_seed_candidates": num_fused_seed,
            "num_fused_qwen_candidates": num_fused_qwen,
            "fused_candidate_titles": fused_titles,
            "candidate_sources": candidate_sources,
        })

    return final_candidates, detected_decades


def _get_ngrams(words: List[str], n: int) -> List[str]:
    """Helper function to generate n-grams from a list of words."""
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


_nlp = None

def _get_spacy_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(_SPACY_MODEL)
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", _SPACY_MODEL])
            _nlp = spacy.load(_SPACY_MODEL)
    return _nlp

def _is_valid_one_word_seed(phrase: str, doc) -> bool:
    """
    High-precision resolver for one-word movie titles.

    Ordinary stopwords are never accepted. A small set of legitimate
    ambiguous movie titles (e.g. It, Her, Up, Us) requires explicit
    movie-related context.
    """
    from spacy.lang.en.stop_words import STOP_WORDS

    phrase = phrase.lower().strip()

    hard_block = {
        "a", "an", "the", "to", "of", "in", "on", "at", "for",
        "and", "or", "but", "yes", "no", "lol", "ok", "okay",
        "hi", "hello", "thanks", "thank",
        "black", "white", "girls", "girl", "boys", "boy",
        "star", "stars", "weekend", "time", "home",
    }

    if phrase in hard_block:
        return False

    # Real movie titles that are also highly ambiguous English words.
    ambiguous_movie_titles = {
        "it", "her", "up", "us"
    }

    strong_context = {
        "movie", "movies", "film", "films",
        "called", "titled", "named",
        "watch", "watched", "watching",
        "seen", "saw",
        "recommend", "recommended",
    }

    for token in doc:
        if token.text.lower() != phrase:
            continue

        nearby_tokens = [
            doc[j].text.lower()
            for j in range(max(0, token.i - 3),
                           min(len(doc), token.i + 4))
            if j != token.i
        ]

        has_movie_context = any(
            word in strong_context for word in nearby_tokens
        )

        # Titles such as "It" or "Her" must have explicit movie context.
        if phrase in ambiguous_movie_titles:
            if has_movie_context:
                return True
            continue

        # All remaining stopwords/function words are rejected.
        if phrase in STOP_WORDS:
            continue

        if token.pos_ in {"PRON", "VERB", "AUX", "DET", "ADP", "CCONJ"}:
            continue

        # PERSON-like words (e.g. "Amy") require movie context;
        # otherwise they are more likely a person's name.
        if token.ent_type_ == "PERSON":
            if has_movie_context:
                return True
            continue

        if has_movie_context:
            return True

        # Proper-name usage is acceptable for non-ambiguous titles.
        if token.pos_ == "PROPN" and token.is_title:
            return True

    return False


def _normalize_movie_text(text: str) -> str:
    """Normalize text for deterministic movie-title resolution."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _resolve_ambiguous_movie_ids(title: str, ids: List[int], context: str):
    """
    Resolve title collisions such as remakes using an explicit year when possible.
    Returns one entity ID or None when resolution would be unsafe.
    """
    if not ids:
        return None

    if len(ids) == 1:
        return ids[0]

    years = set(re.findall(r"\b(?:19|20)\d{2}\b", context))

    if years:
        year_matches = []
        for eid in ids:
            uri = _id2entity.get(eid, "")
            entity_year = extract_year_from_uri(uri)
            if entity_year and entity_year in years:
                year_matches.append(eid)

        if len(year_matches) == 1:
            return year_matches[0]

    # Do not silently choose between ambiguous remakes/entities.
    return None



def _is_safe_direct_movie_phrase(phrase: str) -> bool:
    """
    Reject obvious conversational phrases that happen to share a title
    with a movie in the KBRD catalogue.
    """
    from spacy.lang.en.stop_words import STOP_WORDS

    tokens = phrase.lower().split()

    if not tokens:
        return False

    conversational_block = {
        "are you here",
        "how are you",
        "what about",
        "do you know",
        "thank you",
        "i know",
        "i do",
        "you know",
    }

    if phrase.lower() in conversational_block:
        return False

    # If a multi-word candidate consists entirely of stopwords /
    # conversational function words, it is unsafe.
    if len(tokens) >= 2 and all(t in STOP_WORDS for t in tokens):
        return False

    return True


def _extract_movie_titles_v2(dialogue: str, doc):
    """
    Deterministic movie-title resolver used by Entity Resolver V2.

    Returns:
        movie_ids: linked KBRD movie entity IDs
        descriptions: provenance strings for diagnostics
    """
    normalized = _normalize_movie_text(dialogue)
    words = normalized.split()

    found_ids = []
    descriptions = []
    consumed_spans = set()

    # Longest titles first. ReDial/KBRD contains titles longer than 3 words,
    # so V2 searches up to 8 tokens.
    max_n = min(8, len(words))

    for n in range(max_n, 0, -1):
        for i in range(len(words) - n + 1):
            span = (i, i + n)

            # Avoid detecting fragments inside a title already resolved.
            if any(i >= a and i + n <= b for a, b in consumed_spans):
                continue

            phrase = " ".join(words[i:i+n])

            if n > 1 and not _is_safe_direct_movie_phrase(phrase):
                continue

            ids = _movie_title_to_ids.get(phrase)
            if not ids:
                continue

            # One-word titles require additional contextual safety.
            if n == 1 and not _is_valid_one_word_seed(phrase, doc):
                continue

            eid = _resolve_ambiguous_movie_ids(phrase, ids, dialogue)

            if eid is None:
                logger.debug(
                    f"[Entity V2] Ambiguous movie title skipped: "
                    f"{phrase!r} -> {len(ids)} entities"
                )
                continue

            if eid not in found_ids:
                found_ids.append(eid)
                descriptions.append(
                    f"'{phrase}' (V2 Direct Movie Match)"
                )

            consumed_spans.add(span)

    return found_ids, descriptions


def prepare_input(dialogue: str) -> tuple:
    """
    Hybrid entity extraction & linking module (spaCy + N-grams).
    1. Extracts spaCy named entities and noun chunks.
    2. Generates 1, 2, and 3-grams from dialogue.
    3. Matches against known movies and generic DBpedia URIs.
    4. Uses exact, case-insensitive, and fuzzy matching.
    5. Maps genres dynamically.
    """
    logger.info("[KBRD Adapter] -> STAGE 2: Preparing input from dialogue...")
    _load_kbrd_resources()

    if _has_error or not _entity2id:
        logger.warning("[KBRD Adapter WARNING] Skipping input preparation due to prior errors.")
        return [], [], [], []

    # Step A: Preprocessing
    clean_dialogue = re.sub(r"[^\w\s]", "", dialogue.lower()).strip()
    words = clean_dialogue.split()

    seed_set = set()
    detected_phrases = []

    # Step B: spaCy Extraction
    nlp = _get_spacy_nlp()
    doc = nlp(dialogue)

    # Entity Resolver V2: deterministic longest-title-first movie resolution.
    if _RESOLVER_VERSION == "v2":
        v2_movie_ids, v2_movie_descriptions = _extract_movie_titles_v2(
            dialogue, doc
        )
        seed_set.update(v2_movie_ids)
        detected_phrases.extend(v2_movie_descriptions)
    spacy_phrases = [ent.text.lower() for ent in doc.ents]
    spacy_phrases.extend([chunk.text.lower() for chunk in doc.noun_chunks])

    # Clean spaCy phrases
    spacy_phrases = [re.sub(r"[^\w\s]", "", p).strip() for p in spacy_phrases]
    spacy_phrases = [p for p in spacy_phrases if p]

    # Step C: N-Gram Fallback
    all_ngrams = []
    for n in [3, 2, 1]:
        all_ngrams.extend(_get_ngrams(words, n))

    # Combine and remove duplicates, preserving order (longest/spaCy first)
    candidate_phrases = []
    for p in spacy_phrases + all_ngrams:
        if p not in candidate_phrases:
            candidate_phrases.append(p)

    # Sort candidate phrases by length descending to prioritize longer phrases
    candidate_phrases.sort(key=len, reverse=True)

    # Step D: Matching pipeline
    unmatched_phrases = []
    filtered_1grams = []

    ENTITY_BLOCKLIST = {
        "something", "anything", "nothing", "everything", "someone", "anyone",
        "the", "this", "that", "these", "those", "yes", "no", "not", "and",
        "but", "or", "if", "in", "on", "at", "to", "for", "of", "with",
        "film", "films", "movie", "movies", "watch", "love", "like", "want",
        "looking", "prefer", "enjoy", "seen", "tonight", "please", "maybe",
        "really", "just", "would", "could", "should", "from", "about",
        "action", "fiction", "crime", "drama", "comedy", "thriller", "fun",
        "old", "new", "classic", "modern", "good", "great", "interesting",
        "real", "epic", "light", "family", "kids", "funny", "scary"
    }

    for phrase in candidate_phrases:
        if phrase in ENTITY_BLOCKLIST:
            continue

        # Apply strict filtering for 1-grams
        if len(phrase.split()) == 1:
            if not _is_valid_one_word_seed(phrase, doc):
                if phrase not in filtered_1grams:
                    filtered_1grams.append(phrase)
                continue

        matched = False

        # Stage 1: Exact Match
        if phrase in _movie_title_to_id:
            if _RESOLVER_VERSION == "v2":
                mid = _resolve_ambiguous_movie_ids(
                    phrase,
                    _movie_title_to_ids.get(phrase, []),
                    dialogue,
                )
            else:
                mid = _movie_title_to_id[phrase]

            if mid is not None:
                if mid not in seed_set:
                    seed_set.add(mid)
                    detected_phrases.append(
                        f"'{phrase}' (Exact Movie Match)"
                    )
                matched = True

        # Stage 2: DBpedia URI Match
        if not matched:
            capitalized = "_".join([w.capitalize() for w in phrase.split()])
            potential_uri = f"<http://dbpedia.org/resource/{capitalized}>"
            if potential_uri in _entity2id:
                eid = _entity2id[potential_uri]
                if eid not in seed_set:
                    seed_set.add(eid)
                    detected_phrases.append(f"'{phrase}' (DBpedia URI Match)")
                matched = True

        if not matched:
            unmatched_phrases.append(phrase)

    # Stage 3: Fuzzy Matching on unmatched phrases (length >= 6)
    long_unmatched = [p for p in unmatched_phrases if len(p) >= 6]
    movie_titles = list(_movie_title_to_id.keys())

    for phrase in long_unmatched:
        # Prevent redundant fuzzy matching if an exact match already snagged this concept
        if any(phrase in dp for dp in detected_phrases):
            continue

        matches = difflib.get_close_matches(phrase, movie_titles, n=3, cutoff=_FUZZY_CUTOFF_ENTITY)
        if matches:
            matched_title = matches[0]
            mid = _movie_title_to_id[matched_title]
            if mid not in seed_set:
                seed_set.add(mid)
                detected_phrases.append(f"'{phrase}' -> '{matched_title}' (Fuzzy Movie Match)")

    # Get last user turn text
    last_turn_idx = dialogue.rfind("User: ")
    if last_turn_idx != -1:
        last_turn_text = dialogue[last_turn_idx:]
    else:
        last_turn_text = dialogue
    last_turn_clean = re.sub(r"[^\w\s]", "", last_turn_text.lower()).strip()
    last_turn_words = last_turn_clean.split()

    # Step E: Genre Detection
    genre_map = {
        "horror": "Horror_film",
        "comedy": "Comedy_film",
        "action": "Action_film",
        "animation": "Animated_film",
        "sci fi": "Science_fiction_films",
        "scifi": "Science_fiction_films",
        "sci-fi": "Science_fiction_films",
        "thriller": "Thriller_(genre)",
        "romance": "Romance_film",
        "documentary": "Documentary_film",
        "family": "Children's_film",
    }

    last_turn_genres = set()
    for word in last_turn_words:
        if word in genre_map:
            last_turn_genres.add(word)
    all_genres = set()
    for word in words:
        if word in genre_map:
            all_genres.add(word)

    active_genres = last_turn_genres if last_turn_genres else all_genres

    for word in active_genres:
        genre_uri = f"<http://dbpedia.org/resource/{genre_map[word]}>"
        if genre_uri in _entity2id:
            eid = _entity2id[genre_uri]
            if eid not in seed_set:
                seed_set.add(eid)
                detected_phrases.append(f"'{word}' (Genre Mapping)")

    # ADD BLOCK 1 â€” Person detection (actors/directors)
    entity_map = {eid: _clean_title(uri) for eid, uri in _id2entity.items()}
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            person_name = ent.text.strip()
            person_name_lower = person_name.lower()
            best_match = None
            best_score = 0
            for entity_id, entity_name in entity_map.items():
                entity_lower = entity_name.lower()
                if person_name_lower == entity_lower:
                    best_match = entity_id
                    best_score = 100
                    break
                score = fuzz.ratio(person_name_lower,
                                  entity_lower)
                if score > _PERSON_MATCH_THRESH and score > best_score:
                    best_score = score
                    best_match = entity_id
            if best_match is not None:
                seed_set.add(best_match)
                detected_phrases.append(f"'{person_name}' (Person Actor/Director)")
                logger.info(
                    f"[KBRD Adapter] Person detected: "
                    f"'{person_name}'"
                )

    # ADD BLOCK 2 â€” Temporal clue detection
    # Detect explicit decade mentions
    decade_patterns = [
        (r'\b(192\d)s?\b', '1920s'),
        (r'\b(193\d)s?\b', '1930s'),
        (r'\b(194\d)s?\b', '1940s'),
        (r'\b(195\d)s?\b', '1950s'),
        (r'\b(196\d)s?\b', '1960s'),
        (r'\b(197\d)s?\b', '1970s'),
        (r'\b(198\d)s?\b', '1980s'),
        (r'\b(199\d)s?\b', '1990s'),
        (r'\b(200\d)s?\b', '2000s'),
        (r'\b(201\d)s?\b', '2010s'),
        (r'\b20s\b|twenties', '1920s'),
        (r'\b30s\b|thirties', '1930s'),
        (r'\b40s\b|forties', '1940s'),
        (r'\b50s\b|fifties', '1950s'),
        (r'\b60s\b|sixties', '1960s'),
        (r'\b70s\b|seventies', '1970s'),
        (r'\b80s\b|eighties', '1980s'),
        (r'\b90s\b|nineties', '1990s'),
    ]

    dialogue_lower = dialogue.lower()
    last_turn_lower = last_turn_text.lower()
    detected_decades = []
    last_turn_decades = []
    for pattern, decade in decade_patterns:
        if re.search(pattern, last_turn_lower):
            last_turn_decades.append(decade)

    if last_turn_decades:
        detected_decades = last_turn_decades
        for d in detected_decades:
            logger.info(f"[KBRD Adapter] Temporal clue detected in last turn: {d}")
    else:
        for pattern, decade in decade_patterns:
            if re.search(pattern, dialogue_lower):
                detected_decades.append(decade)
                logger.info(f"[KBRD Adapter] Temporal clue detected: {decade}")

    # Store detected decades for use in reranking hint
    if detected_decades:
        # Add to context so Qwen knows user preference
        logger.info(
            f"[KBRD Adapter] User era preference: "
            f"{detected_decades}"
        )

    # Step F & G: Deduplication and Logging
    seed_list = list(seed_set)

    if last_turn_genres:
        for word in last_turn_genres:
            genre_uri = f"<http://dbpedia.org/resource/{genre_map[word]}>"
            if genre_uri in _entity2id:
                eid = _entity2id[genre_uri]
                seed_list.extend([eid] * _GENRE_BOOST_FACTOR)
                logger.info(f"[KBRD Adapter] Boosted '{word}' genre weight by {_GENRE_BOOST_FACTOR}x in seed list.")
    if not seed_list:
        logger.warning("[KBRD Adapter WARNING] No matching entities or movies found in dialogue.")
    else:
        logger.debug(f"[KBRD Adapter] Detected Entities:")
        for dp in detected_phrases:
            logger.debug(f"  - {dp}")
        logger.debug(f"[KBRD Adapter] Found {len(seed_list)} DBpedia entities linked to dialogue.")
        if filtered_1grams:
            logger.debug(f"[KBRD Adapter] Filtered noisy 1-grams: {filtered_1grams}")

    return seed_list, detected_decades, detected_phrases, filtered_1grams
