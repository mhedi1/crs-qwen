import os
import pickle
import json
import re
import difflib
import spacy
from typing import List, Dict, Any
import logging
import warnings
from fuzzywuzzy import fuzz
from reranker import call_qwen

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


def get_fallback_candidates(top_k: int) -> List[Dict[str, Any]]:
    return [
        {"id": 101, "title": "The Matrix (1999)", "genre": "Sci-Fi", "decade": "1990s"},
        {"id": 102, "title": "The Exorcist (1973)", "genre": "Horror", "decade": "1970s"},
        {"id": 103, "title": "Shrek (2001)", "genre": "Animation", "decade": "2000s"},
        {"id": 104, "title": "Die Hard (1988)", "genre": "Action", "decade": "1980s"},
        {"id": 105, "title": "The Hangover (2009)", "genre": "Comedy", "decade": "2000s"},
    ][:top_k]


def _load_kbrd_resources() -> None:
    global _data_loaded, _has_error, _id2entity, _movie_ids, _entity2id, _movie_title_to_id

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
    title = re.sub(r"\s*\(film\)", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(\d{4} film\)", "", title, flags=re.IGNORECASE)
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
            logger.info("[KBRD Neural] CUDA not available — loading model on CPU")
        logger.info("[KBRD Neural] Loading model from saved/kbrd_model")
        opt = {
            'model_file': os.path.join(KBRD_REPO_PATH, 'saved', 'kbrd_model'),
            'datatype': 'test',
            'datapath': os.path.join(KBRD_REPO_PATH, 'data'),
            'no_cuda': no_cuda,
            'override': {
                'model_file': os.path.join(KBRD_REPO_PATH, 'saved', 'kbrd_model'),
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
    return candidate


# Maximum number of fused candidates that can be injected after KBRD top-30.
_MAX_FUSED_CANDIDATES = 15


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
            clean_t, _movie_title_to_id.keys(), n=1, cutoff=0.85
        )
        if matches:
            mid = _movie_title_to_id[matches[0]]

    if mid is None:
        return None

    return _seed_id_to_movie_candidate(mid, source_label="QWEN_FUSION")


def get_kbrd_candidates(dialogue: str, top_k: int = 5, diagnostics: dict = None) -> tuple:
    """Generate ranked movie candidates from the KBRD neural model with Candidate Fusion.

    Loads model and resources on first call, extracts seed entities from the
    dialogue, runs KBRD inference, then performs conservative Candidate Fusion:
    - KBRD top-30 are kept exactly as-is.
    - Verified movie-only seeds and resolved Qwen titles are injected after rank 30.
    - Remaining KBRD candidates fill slots up to top_k (max 50).
    - Fusion is capped at _MAX_FUSED_CANDIDATES to prevent domination.
    - No ground-truth or future-turn data is used.

    Args:
        dialogue: Full conversation history as a formatted string.
        top_k: Maximum number of candidates to return.
        diagnostics: Optional dict to populate with per-call diagnostic data.

    Returns:
        Tuple of (candidates, detected_decades). candidates is a list of dicts;
        detected_decades is a list of decade strings found in the dialogue.
    """
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
                "seed_entity_ids": [], "weak_seed_fallback": False,
                "num_extracted_seeds": 0, "num_matched_seeds": 0,
                "filtered_noisy_seeds": [],
                "num_filtered_noisy_seeds": 0,
                "num_fused_seed_candidates": 0,
                "num_fused_qwen_candidates": 0,
                "fused_candidate_titles": [],
                "candidate_sources": {},
            })
        return get_fallback_candidates(top_k), []

    seed_list, detected_decades, detected_phrases, filtered_1grams = prepare_input(dialogue)

    _seeds_before_fallback = len(seed_list)
    _weak_seed_fallback = _seeds_before_fallback < 4
    _qwen_titles: list = []

    if _weak_seed_fallback:
        logger.warning("[KBRD Adapter] Weak seeds detected, using Qwen fallback")
        try:
            prompt = (
                "Based on this movie recommendation conversation, \n"
                "name exactly 3 well-known movies that match what \n"
                "the user is looking for. Return only movie titles, \n"
                "one per line, nothing else.\n"
                "\n"
                "Conversation:\n"
                f"{dialogue}"
            )
            content = call_qwen(prompt)
            titles = [t.strip() for t in content.split('\n') if t.strip()]
            _qwen_titles = titles
            logger.debug(f"[KBRD Adapter] Qwen suggested seeds: {', '.join(titles)}")
            
            added_count = 0
            for title in titles:
                clean_t = re.sub(r"[^\w\s]", "", title.lower()).strip()
                if clean_t in _movie_title_to_id:
                    mid = _movie_title_to_id[clean_t]
                    if mid not in seed_list:
                        seed_list.append(mid)
                        added_count += 1
                else:
                    matches = difflib.get_close_matches(clean_t, _movie_title_to_id.keys(), n=1, cutoff=0.85)
                    if matches:
                        mid = _movie_title_to_id[matches[0]]
                        if mid not in seed_list:
                            seed_list.append(mid)
                            added_count += 1
                            
            if added_count > 0:
                logger.debug(f"[KBRD Adapter] Added {added_count} semantic seed entities")
        except Exception as e:
            logger.error(f"[KBRD Adapter] Qwen fallback error: {e}")

    if not seed_list:
        logger.warning("[KBRD Neural] No entities detected in dialogue. Using fallback.")
        if diagnostics is not None:
            diagnostics.update({
                "extracted_seeds": detected_phrases,
                "qwen_fallback_seeds": _qwen_titles,
                "seed_entity_ids": [],
                "weak_seed_fallback": _weak_seed_fallback,
                "num_extracted_seeds": _seeds_before_fallback,
                "num_matched_seeds": 0,
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
    seed_sets = [seed_list]
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
                "seed_entity_ids": list(seed_list),
                "weak_seed_fallback": _weak_seed_fallback,
                "num_extracted_seeds": _seeds_before_fallback,
                "num_matched_seeds": len(seed_list),
                "filtered_noisy_seeds": filtered_1grams,
                "num_filtered_noisy_seeds": len(filtered_1grams),
                "num_fused_seed_candidates": 0,
                "num_fused_qwen_candidates": 0,
                "fused_candidate_titles": [],
                "candidate_sources": {},
            })
        return get_fallback_candidates(top_k), detected_decades

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
    # Only accept seeds that are confirmed movie IDs (seed_id in _movie_ids).
    # Genre IDs, person/actor IDs, and generic DBpedia entity IDs are excluded
    # by the _seed_id_to_movie_candidate() gate: `seed_id not in _movie_ids`.
    for seed_id in seed_list:
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

    # --- Assemble final list: KBRD top-30 + fused + KBRD tail ---
    # KBRD top-30: preserved exactly, in original order.
    kbrd_top30 = kbrd_candidates[:30]

    # KBRD tail: candidates ranked 31+ that were NOT fused (by id).
    fused_ids = {c["id"] for c in fused_candidates}
    kbrd_tail = [
        c for c in kbrd_candidates[30:]
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
        diagnostics.update({
            "extracted_seeds": detected_phrases,
            "qwen_fallback_seeds": _qwen_titles,
            "seed_entity_ids": list(seed_list),
            "weak_seed_fallback": _weak_seed_fallback,
            "num_extracted_seeds": _seeds_before_fallback,
            "num_matched_seeds": len(seed_list),
            "filtered_noisy_seeds": filtered_1grams,
            "num_filtered_noisy_seeds": len(filtered_1grams),
            # Fusion-specific diagnostics
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
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp

def _is_valid_one_word_seed(phrase: str, doc) -> bool:
    """
    Stricter filter for 1-gram movie candidates to avoid noisy matches like 'it', 'saw', 'time'.
    """
    from spacy.lang.en.stop_words import STOP_WORDS
    strong_context = {"movie", "film", "called", "titled", "named"}
    
    is_stopword = phrase in STOP_WORDS
    
    for token in doc:
        if token.text.lower() == phrase:
            # 1. Capitalization check (not sentence initial)
            if token.is_title and token.i > 0:
                if not token.nbor(-1).is_punct:
                    return True
                
            # 2. Strong context keywords nearby
            prev_token = token.nbor(-1).text.lower() if token.i > 0 else ""
            next_token = token.nbor(1).text.lower() if token.i < len(doc) - 1 else ""
            
            if prev_token in strong_context or next_token in strong_context:
                return True
                
            # 3. Stopword / Pronoun / Verb check
            if is_stopword or token.pos_ in ["PRON", "VERB", "AUX", "DET"]:
                continue
                
            # Otherwise, it's generally safe (e.g. valid noun/name)
            return True
            
    return False


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
        return [], [], []

    # Step A: Preprocessing
    clean_dialogue = re.sub(r"[^\w\s]", "", dialogue.lower()).strip()
    words = clean_dialogue.split()

    seed_set = set()
    detected_phrases = []

    # Step B: spaCy Extraction
    nlp = _get_spacy_nlp()
    doc = nlp(dialogue)
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
            mid = _movie_title_to_id[phrase]
            if mid not in seed_set:
                seed_set.add(mid)
                detected_phrases.append(f"'{phrase}' (Exact Movie Match)")
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
            
        matches = difflib.get_close_matches(phrase, movie_titles, n=3, cutoff=0.92)
        if matches:
            matched_title = matches[0]
            mid = _movie_title_to_id[matched_title]
            if mid not in seed_set:
                seed_set.add(mid)
                detected_phrases.append(f"'{phrase}' -> '{matched_title}' (Fuzzy Movie Match)")

    # Step E: Genre Detection
    genre_map = {
        "horror": "Horror_film",
        "comedy": "Comedy_film",
        "action": "Action_film",
        "animation": "Animated_film",
        "sci fi": "Science_fiction_film",
        "scifi": "Science_fiction_film",
        "thriller": "Thriller_(genre)",
        "romance": "Romance_film",
        "documentary": "Documentary_film",
        "family": "Children's_film",
    }

    # Match individual words against genre map
    for word in words:
        if word in genre_map:
            genre_uri = f"<http://dbpedia.org/resource/{genre_map[word]}>"
            if genre_uri in _entity2id:
                eid = _entity2id[genre_uri]
                if eid not in seed_set:
                    seed_set.add(eid)
                    detected_phrases.append(f"'{word}' (Genre Mapping)")

    # ADD BLOCK 1 — Person detection (actors/directors)
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
                if score > 85 and score > best_score:
                    best_score = score
                    best_match = entity_id
            if best_match is not None:
                seed_set.add(best_match)
                detected_phrases.append(f"'{person_name}' (Person Actor/Director)")
                logger.info(
                    f"[KBRD Adapter] Person detected: "
                    f"'{person_name}'"
                )

    # ADD BLOCK 2 — Temporal clue detection
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
    detected_decades = []
    for pattern, decade in decade_patterns:
        if re.search(pattern, dialogue_lower):
            detected_decades.append(decade)
            logger.info(
                f"[KBRD Adapter] Temporal clue detected:"
                f" {decade}"
            )

    # Store detected decades for use in reranking hint
    if detected_decades:
        # Add to context so Qwen knows user preference
        logger.info(
            f"[KBRD Adapter] User era preference: "
            f"{detected_decades}"
        )

    # Step F & G: Deduplication and Logging
    seed_list = list(seed_set)
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