import logging
from prompts import truncate_history
from reranker import call_qwen, USE_FAKE_MODE

logger = logging.getLogger(__name__)

_SELECTED_TITLE_PREFIX = "SELECTED_TITLE:"
_RESPONSE_PREFIX = "RESPONSE:"

_ANTI_FABRICATION_RULE = (
    " Only state factual information that appears in [Selected Recommendation] "
    "or [Dialogue History]. Never invent or assume cast, director, release year, "
    "rating, awards, or plot details. If a fact was not provided to you, say it "
    "cannot be confirmed rather than guessing."
)
_ANTI_FABRICATION_REQUIREMENT = (
    "- never assert cast, director, year, rating, awards, or plot details "
    "that were not provided; say they cannot be confirmed instead\n"
)


def _selected_recommendation_block(selected_movie: dict) -> str:
    """Render only the metadata actually present on the selected movie.

    Missing, empty, and "Unknown" fields are omitted entirely so the model is
    never shown a blank or placeholder value it might try to fill in.
    """

    def _clean(key: str) -> str:
        value = selected_movie.get(key)
        if value is None:
            return ""
        text = str(value).strip()
        if not text or text.casefold() == "unknown":
            return ""
        return text

    lines = [f"Title: {_clean('title')}"] if _clean("title") else []
    for label, key in (
        ("Genre", "genre"),
        ("Decade", "decade"),
        ("Year", "year"),
        ("Rating", "rating"),
        ("Overview", "overview"),
    ):
        value = _clean(key)
        if value:
            lines.append(f"{label}: {value}")
    if not lines:
        return "(no metadata available)"
    return "\n".join(lines)


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
    return response


def generate_response(
    history: str,
    selected_movie: dict,
    previously_recommended: list = None,
    is_followup: bool = False,
) -> str:
    if USE_FAKE_MODE:
        return _fallback_response(selected_movie)

    history = truncate_history(history)
    
    selected_title = str(selected_movie.get("title", ""))
    sys_content = (
        "You are a response renderer and explainer for a movie recommender system. "
        "You do not choose or rank movies. The upstream selected recommendation is "
        "authoritative and immutable. Discuss that exact movie and never replace it "
        "with or recommend another movie. If it is an imperfect match, acknowledge "
        "the limitation but still discuss the selected movie."
    )
    sys_content += _ANTI_FABRICATION_RULE
    if is_followup:
        sys_content += (
            " This is a follow-up turn. Answer the user's MOST RECENT follow-up "
            "question directly about the selected movie. You may use pronouns "
            "naturally, but must not substitute or recommend another movie."
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

    if is_followup:
        response_instruction = (
            "Answer the user's MOST RECENT follow-up question directly about the "
            "selected recommendation. You may use pronouns naturally.\n"
        )
        reply_requirements = (
            "- answer the most recent follow-up question directly\n"
            "- discuss the selected movie only\n"
            "- do not recommend or substitute another movie\n"
            + _ANTI_FABRICATION_REQUIREMENT
            + "- sound concise and friendly\n"
        )
    else:
        response_instruction = (
            "Write one conversational response using the selected recommendation.\n"
        )
        reply_requirements = (
            "- discuss exactly the authoritative selected movie\n"
            "- mention the selected movie naturally\n"
            "- justify it with the user's preferences\n"
            "- do not substitute or recommend another movie\n"
            + _ANTI_FABRICATION_REQUIREMENT
            + "- remain concise and conversational\n"
        )

    messages = [
        {
            "role": "system",
            "content": sys_content
        },
        {
            "role": "user", 
            "content": (
                response_instruction
                + "Return exactly this structured format:\n"
                "SELECTED_TITLE: <copy the exact selected title>\n"
                "RESPONSE: <natural-language response>\n"
                "Do not add text before SELECTED_TITLE.\n"
                "The reply must:\n"
                f"{reply_requirements}\n"
                f"[Dialogue History]\n{history}\n\n"
                f"[Selected Recommendation]\n"
                f"{_selected_recommendation_block(selected_movie)}"
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
