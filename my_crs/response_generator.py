import logging
from prompts import truncate_history
from reranker import call_qwen, USE_FAKE_MODE

logger = logging.getLogger(__name__)

_SELECTED_TITLE_PREFIX = "SELECTED_TITLE:"
_RESPONSE_PREFIX = "RESPONSE:"


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


def _parse_grounded_response(raw_response: str, selected_title: str) -> str | None:
    lines = raw_response.strip().splitlines()
    if len(lines) < 2 or not lines[0].startswith(_SELECTED_TITLE_PREFIX):
        return None
    if not lines[1].startswith(_RESPONSE_PREFIX):
        return None

    returned_title = lines[0][len(_SELECTED_TITLE_PREFIX):].strip()
    if returned_title != selected_title:
        return None

    response_lines = [lines[1][len(_RESPONSE_PREFIX):].strip(), *lines[2:]]
    response = "\n".join(response_lines).strip()
    if not response:
        return None

    required_opening = f"I recommend {selected_title}"
    if not response.startswith(required_opening):
        return None
    if len(response) > len(required_opening):
        boundary = response[len(required_opening)]
        if not boundary.isspace() and boundary not in ".,!?:;—-":
            return None
    return response


def generate_response(history: str, selected_movie: dict, previously_recommended: list = None) -> str:
    if USE_FAKE_MODE:
        return _fallback_response(selected_movie)

    history = truncate_history(history)
    
    selected_title = str(selected_movie.get("title", ""))
    sys_content = (
        "You are a response renderer and explainer for a movie recommender system. "
        "You do not choose or rank movies. The upstream selected recommendation is "
        "authoritative and immutable. Discuss that exact movie and never replace it "
        "with or recommend another movie. If it is an imperfect match, acknowledge "
        "the limitation but still discuss the selected movie. On a follow-up, answer "
        "about the selected movie only."
    )
    other_previously_recommended = [
        str(title)
        for title in (previously_recommended or [])
        if str(title).strip().casefold() != selected_title.strip().casefold()
    ]
    if other_previously_recommended:
        sys_content += (
            " Previously suggested titles are context only and must not override "
            "the authoritative selection. Do not introduce these titles as additional "
            f"recommendations: {other_previously_recommended}"
        )

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
                "Return exactly this structured format:\n"
                "SELECTED_TITLE: <copy the exact selected title>\n"
                "RESPONSE: <natural-language response>\n"
                "Do not add text before SELECTED_TITLE.\n"
                f"The RESPONSE field must begin exactly with: I recommend {selected_title}\n"
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
        grounded_response = _parse_grounded_response(response, selected_title)
        if grounded_response is None:
            logger.warning(
                "[Response Generator] Malformed or ungrounded output from Qwen."
            )
            return _fallback_response(selected_movie)
        return grounded_response
    except Exception as e:
        logger.error(f"[Response Generator ERROR] {e}")
        return _fallback_response(selected_movie)
