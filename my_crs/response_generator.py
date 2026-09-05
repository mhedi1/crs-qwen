import logging
import re

from prompts import truncate_history
from reranker import call_qwen, USE_FAKE_MODE

logger = logging.getLogger(__name__)

_SELECTED_TITLE_PREFIX = "SELECTED_TITLE:"
_RESPONSE_PREFIX = "RESPONSE:"

# Factual claims are scoped to [Selected Recommendation] alone. [Dialogue
# History] states what the user wants, which is not evidence about what the
# selected movie actually is: a requested actor is not verified cast, and an
# excluded director is not a verified director.
_ANTI_FABRICATION_RULE = (
    " Factual claims about the selected movie may come ONLY from "
    "[Selected Recommendation]. [Dialogue History] tells you what the user is "
    "looking for; it is never evidence about the selected movie's attributes. "
    "A requested actor is not verified cast, an excluded director is not a "
    "verified director, a requested year is not a verified release year, and a "
    "requested rating is not a verified rating. Never state or imply that the "
    "selected movie satisfies a requested or excluded constraint unless the "
    "corresponding fact is explicitly listed in [Selected Recommendation]. "
    "Never invent or assume cast, director, release year, rating, awards, or "
    "plot details. If the user asked about such a constraint and that fact is "
    "not listed, say plainly that you cannot confirm it rather than guessing or "
    "implying that it holds."
)
_ANTI_FABRICATION_REQUIREMENT = (
    "- treat [Dialogue History] as the user's preferences only, never as facts "
    "about the selected movie\n"
    "- never assert or imply that a requested actor, excluded director, release "
    "year, rating, awards, or plot detail applies to the selected movie unless "
    "it is explicitly listed in [Selected Recommendation]; otherwise say it "
    "cannot be confirmed\n"
)


# ── Deterministic factual-constraint guard ─────────────────────────────────
# Prompt-only enforcement proved insufficient in live use: Qwen3 still asserted
# a requested actor and director for a selection that carried neither. When the
# latest user turn names an explicit factual constraint and the selected movie
# has no verified metadata for it, the reply is produced here instead of by the
# model. Detection is deliberately narrow keyword matching, not entity
# recognition.
_CONSTRAINT_PATTERNS = {
    # Bare "stars" is excluded: "rated five stars" is a rating, not a cast.
    # "stars" counts only in an actor-like construction — followed by a
    # capitalised name, or in "who stars".
    "cast": re.compile(
        r"\bstarring\b"
        r"|\bcast\b"
        r"|\bactors?\b"
        r"|\bactress(?:es)?\b"
        r"|\bwho\s+stars\b"
        r"|\bstars\s+(?-i:[A-Z])",
        re.IGNORECASE,
    ),
    # Bare "directed" is excluded: "beautifully directed thriller" is a stylistic
    # preference, not a request about who directed the film.
    "director": re.compile(
        r"\bdirected\s+by\b|\bdirectors?\b|\bwho\s+directed\b",
        re.IGNORECASE,
    ),
    # A bare four-digit number is NOT treated as a year: it appears inside real
    # titles ("Blade Runner 2049"). A year cue word or an explicit "N years"
    # window is required.
    "year": re.compile(
        r"\b(?:from|in|released\s+in|since|after|before|around)\s+(?:19|20)\d{2}\b"
        r"|\b(?:19|20)\d{2}\s+(?:movie|film|release)\b"
        r"|\blast\s+(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
        r"\s+years?\b",
        re.IGNORECASE,
    ),
    # Bare "score" is excluded: a "great musical score" is not a rating. Only
    # explicit rating-source scores count.
    "rating": re.compile(
        r"\bhighly[\s-]rated\b"
        r"|\bratings?\b"
        r"|\brated\b"
        r"|\b(?:imdb|metacritic|rotten\s+tomatoes|review|user)\s+scores?\b",
        re.IGNORECASE,
    ),
}

# Metadata that can verify each constraint. Title, genre, decade and overview
# are deliberately excluded: none of them verifies cast, director, release year,
# or rating. Dialogue History is never consulted here.
_CONSTRAINT_FIELDS = {
    "cast": ("cast", "actors"),
    "director": ("director",),
    "year": ("year", "release_year"),
    "rating": ("rating",),
}

_CONSTRAINT_LABELS = {
    "cast": "cast",
    "director": "director",
    "year": "release year",
    "rating": "rating",
}


def _has_verified_value(selected_movie: dict, key: str) -> bool:
    """True when the selected movie carries a usable value for ``key``."""
    value = selected_movie.get(key)
    if value is None:
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    text = str(value).strip()
    return bool(text) and text.casefold() != "unknown"


def _latest_user_turn(history: str) -> str:
    """Return the most recent ``User:`` line, or "" when there is none."""
    for line in reversed(str(history).splitlines()):
        stripped = line.strip()
        if stripped[:5].casefold() == "user:":
            return stripped[5:].strip()
    return ""


def unverifiable_constraints(history: str, selected_movie: dict) -> list:
    """Constraint categories the latest user turn asks about but we cannot verify.

    Only the most recent user turn is inspected, and only the selected movie's
    own metadata counts as verification.
    """
    latest = _latest_user_turn(history)
    if not latest:
        return []
    return [
        name
        for name, pattern in _CONSTRAINT_PATTERNS.items()
        if pattern.search(latest)
        and not any(
            _has_verified_value(selected_movie, key)
            for key in _CONSTRAINT_FIELDS[name]
        )
    ]


def _join_labels(labels: list) -> str:
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} or {labels[1]}"
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _unverified_constraint_response(selected_title: str, missing: list) -> str:
    """Deterministic, truthful reply. Never asserts the constraint either way."""
    labels = [_CONSTRAINT_LABELS[name] for name in missing]
    title = str(selected_title).strip() or "The selected movie"
    subject = "that constraint" if len(labels) == 1 else "those constraints"
    return (
        f"{title} was selected by the recommendation pipeline, but I don't have "
        f"verified {_join_labels(labels)} information for this result, so I "
        f"can't confirm that it satisfies {subject}."
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
        if isinstance(value, (list, tuple)):
            text = ", ".join(
                str(item).strip() for item in value if str(item).strip()
            )
        else:
            text = str(value).strip()
        if not text or text.casefold() == "unknown":
            return ""
        return text

    lines = [f"Title: {_clean('title')}"] if _clean("title") else []
    # Keys mirror _CONSTRAINT_FIELDS so that whatever the guard accepts as
    # verification is also what the model is actually shown.
    for label, keys in (
        ("Genre", ("genre",)),
        ("Decade", ("decade",)),
        ("Year", ("year", "release_year")),
        ("Rating", ("rating",)),
        ("Director", ("director",)),
        ("Cast", ("cast", "actors")),
        ("Overview", ("overview",)),
    ):
        value = next((cleaned for cleaned in map(_clean, keys) if cleaned), "")
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

    # Deterministic guard: the model is never asked to talk about a factual
    # constraint we cannot verify from the selected movie's own metadata. The
    # selection itself is untouched; only the wording is produced here.
    missing_constraints = unverifiable_constraints(history, selected_movie)
    if missing_constraints:
        logger.info(
            "[Response Generator] Unverifiable constraints %s for %r; "
            "answering deterministically without Qwen.",
            missing_constraints,
            selected_title,
        )
        return _unverified_constraint_response(selected_title, missing_constraints)

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
