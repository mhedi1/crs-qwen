import logging
from prompts import truncate_history
from reranker import call_qwen, USE_FAKE_MODE

logger = logging.getLogger(__name__)


def _fallback_response(movie: dict) -> str:
    title = movie.get("title", "this film")
    genre = movie.get("genre", "")
    decade = movie.get("decade", "")
    parts = []
    if genre and genre != "Unknown":
        parts.append(genre.lower())
    if decade and decade != "Unknown":
        parts.append(f"from the {decade}")
    description = " ".join(parts)
    if description:
        return (f"I would recommend {title}. "
                f"It is a great film {description} that "
                f"matches what you are looking for.")
    else:
        return (f"I would recommend {title}. "
                f"I think it fits well with what you described.")


def generate_response(history: str, selected_movie: dict, previously_recommended: list = None) -> str:
    if USE_FAKE_MODE:
        return _fallback_response(selected_movie)

    history = truncate_history(history)
    
    sys_content = "You are a helpful and natural conversational movie recommender."
    sys_content += " CRITICAL: You must evaluate the user's MOST RECENT request (e.g., a specific genre like 'Comedy'). If the retrieved KBRD candidate list does not contain a strong match for this new constraint, you must acknowledge the gap in your response, explain why you are recommending the selected film, and attempt to bridge the thematic gap."
    if previously_recommended:
        sys_content += f" CRITICAL: Do not recommend any of the following movies as they have already been suggested this session: {previously_recommended}"

    messages = [
        {
            "role": "system",
            "content": sys_content
        },
        {
            "role": "user", 
            "content": (
                "Write one conversational response using "
                "the selected recommendation.\n"
                "The reply must:\n"
                "- mention the movie naturally\n"
                "- justify it with the user's preferences\n"
                "- sound concise and friendly\n"
                "- avoid mentioning unselected candidates\n\n"
                f"[Dialogue History]\n{history}\n\n"
                f"[Selected Recommendation]\n"
                f"{selected_movie.get('title', '')}"
                f"{' | ' + selected_movie['genre'] if selected_movie.get('genre') and selected_movie['genre'] != 'Unknown' else ''}"
                f"{' | ' + selected_movie['decade'] if selected_movie.get('decade') and selected_movie['decade'] != 'Unknown' else ''}"
            )
        }
    ]

    try:
        response = call_qwen(messages)
        if not response or not response.strip():
            logger.warning("[Response Generator] Empty output from Qwen.")
            return _fallback_response(selected_movie)
        return response.strip()
    except Exception as e:
        logger.error(f"[Response Generator ERROR] {e}")
        return _fallback_response(selected_movie)