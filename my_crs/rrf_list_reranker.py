"""Isolated zero-shot list reranking for frozen RRF candidate lists."""

from __future__ import annotations

import json
import hashlib
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import requests
import yaml


logger = logging.getLogger(__name__)
MY_CRS_DIR = Path(__file__).resolve().parent
with (MY_CRS_DIR / "config.yaml").open("r", encoding="utf-8") as handle:
    _PROJECT_CONFIG = yaml.safe_load(handle)

PROMPT_VERSION = "rrf_top10_local_positions_v1"
TOP_N = 10
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_OUTPUT_TOKENS = 128

SYSTEM_PROMPT = (
    "/no_think\n"
    "You are a movie recommendation list reranker. Return only the "
    "requested JSON object and no hidden reasoning or explanation."
)
USER_PROMPT_TEMPLATE = (
    "Rank exactly 10 unique movies from the supplied candidate list in "
    "best-to-worst order for the user.\n"
    "Use only local candidate positions from the list.\n"
    "Respect explicit genre, era, actor, and director preferences when "
    "they appear in the dialogue.\n"
    "Respect exclusions, already-seen movies, dislikes, and requests for "
    "something different.\n"
    "Do not add any movie outside the candidate list.\n"
    "Return JSON only, with no Markdown and no explanation, in exactly "
    "this shape:\n"
    '{{"ranked_ids":[17,3,42,8,1,29,10,5,31,14]}}\n\n'
    "[Pre-target Dialogue History]\n"
    "{history}\n\n"
    "[Candidate List]\n"
    "{candidate_block}"
)

FALLBACK_INVALID_CANDIDATE_COUNT = "invalid_candidate_count"
FALLBACK_INVALID_MODEL_OUTPUT = "invalid_model_output"
FALLBACK_REQUEST_FAILURE = "request_failure"
FALLBACK_MALFORMED_API_RESPONSE = "malformed_api_response"
FALLBACK_EMPTY_API_RESPONSE = "empty_api_response"


@dataclass(frozen=True)
class QwenRerankSettings:
    server_url: str = _PROJECT_CONFIG["qwen"]["server_url"]
    model: str = _PROJECT_CONFIG["qwen"]["model"]
    temperature: float = 0.0
    top_p: float = DEFAULT_TOP_P
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    think: bool = False
    stream: bool = False
    max_retries: int = _PROJECT_CONFIG["qwen"]["max_retries"]
    timeout: float = _PROJECT_CONFIG["qwen"]["timeout"]

    def provenance(self) -> dict[str, Any]:
        return asdict(self)


class RankedPositionsError(ValueError):
    """Raised when an LLM ranking violates the strict output contract."""


class QwenRerankError(RuntimeError):
    """Raised when the isolated Qwen call cannot provide parseable content."""

    def __init__(
        self,
        code: str,
        *,
        detail: str | None = None,
        attempts: int,
        successful_requests: int,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.attempts = attempts
        self.successful_requests = successful_requests


@dataclass(frozen=True)
class QwenCallResult:
    content: str
    attempts: int
    successful_requests: int = 1


@dataclass(frozen=True)
class ListRerankResult:
    final_candidates: list[dict[str, Any]]
    raw_output: str | None
    ranked_positions: list[int]
    ranked_candidate_ids: list[int]
    ranked_candidate_titles: list[str]
    fallback: bool
    fallback_reason: str | None
    fallback_detail: str | None
    request_attempts: int
    successful_requests: int


def _single_line_title(value: Any) -> str:
    sanitized = "".join(
        " " if ord(character) < 32 or ord(character) == 127 else character
        for character in str(value)
    )
    return " ".join(sanitized.split())


def _serialize_candidate_line(position: int, title: Any) -> str:
    return f"{position}. {_single_line_title(title)}"


def build_list_rerank_prompt(
    history: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build the fixed zero-shot prompt using local positions and titles only."""
    candidate_lines = [
        _serialize_candidate_line(
            position,
            candidate.get("title", "Unknown Title"),
        )
        for position, candidate in enumerate(candidates, start=1)
    ]
    candidate_block = "\n".join(candidate_lines)
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                history=history,
                candidate_block=candidate_block,
            ),
        },
    ]


def prompt_template_digest() -> str:
    """Hash a canonical prompt rendered by the real prompt-construction path."""
    sentinel_candidates = [
        {"title": f"SENTINEL_TITLE_{position:02d}"}
        for position in range(1, 51)
    ]
    sentinel_candidates[0]["title"] = "Ordinary ASCII"
    sentinel_candidates[1]["title"] = "Unicode Amélie – 千と千尋の神隠し"
    sentinel_candidates[2]["title"] = "Newline\nSentinel"
    sentinel_candidates[3]["title"] = "Tab\tSentinel"
    sentinel_candidates[4]["title"] = "Control\x1bSentinel"
    material = {
        "messages": build_list_rerank_prompt(
            "User: SENTINEL_HISTORY",
            sentinel_candidates,
        ),
        "required_top_n": TOP_N,
    }
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_CODE_FENCE = re.compile(
    r"\A```(?:json)?[ \t]*\r?\n?(.*?)\r?\n?```\Z",
    flags=re.IGNORECASE | re.DOTALL,
)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise RankedPositionsError(f"duplicate_json_key:{key}")
        parsed[key] = value
    return parsed


def parse_ranked_positions(
    raw_output: str,
    *,
    candidate_count: int,
    top_n: int = TOP_N,
    allow_code_fence: bool = True,
) -> list[int]:
    """Parse an exact ``ranked_ids`` JSON object.

    Unknown keys are rejected. One surrounding JSON/unspecified Markdown code
    fence is accepted; all other free-form text is rejected.
    """
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise RankedPositionsError("empty_output")
    if candidate_count < top_n:
        raise RankedPositionsError("candidate_count_below_top_n")

    payload = raw_output.strip()
    if allow_code_fence:
        fence_match = _CODE_FENCE.fullmatch(payload)
        if fence_match:
            payload = fence_match.group(1).strip()

    try:
        parsed = json.loads(payload, object_pairs_hook=_reject_duplicate_json_keys)
    except RankedPositionsError:
        raise
    except (TypeError, ValueError) as error:
        raise RankedPositionsError("malformed_json") from error

    if not isinstance(parsed, dict):
        raise RankedPositionsError("output_must_be_json_object")
    if set(parsed) != {"ranked_ids"}:
        raise RankedPositionsError("object_must_contain_only_ranked_ids")
    positions = parsed["ranked_ids"]
    if not isinstance(positions, list):
        raise RankedPositionsError("ranked_ids_must_be_list")
    if len(positions) != top_n:
        raise RankedPositionsError(f"ranked_ids_must_contain_exactly_{top_n}_positions")
    if any(type(position) is not int for position in positions):
        raise RankedPositionsError("ranked_ids_must_be_integers")
    if any(position < 1 or position > candidate_count for position in positions):
        raise RankedPositionsError("ranked_id_out_of_range")
    if len(set(positions)) != len(positions):
        raise RankedPositionsError("ranked_ids_must_be_unique")
    return positions


def complete_ranking(
    candidates: Sequence[Mapping[str, Any]],
    ranked_positions: Sequence[int],
) -> list[dict[str, Any]]:
    """Place selected local positions first and retain the remaining RRF order."""
    copied = [dict(candidate) for candidate in candidates]
    selected_indexes = [position - 1 for position in ranked_positions]
    selected_index_set = set(selected_indexes)
    return [copied[index] for index in selected_indexes] + [
        candidate
        for index, candidate in enumerate(copied)
        if index not in selected_index_set
    ]


def _fallback_result(
    candidates: Sequence[Mapping[str, Any]],
    *,
    reason: str,
    detail: str | None,
    raw_output: str | None,
    attempts: int,
    successful_requests: int,
) -> ListRerankResult:
    return ListRerankResult(
        final_candidates=[dict(candidate) for candidate in candidates],
        raw_output=raw_output,
        ranked_positions=[],
        ranked_candidate_ids=[],
        ranked_candidate_titles=[],
        fallback=True,
        fallback_reason=reason,
        fallback_detail=detail,
        request_attempts=attempts,
        successful_requests=successful_requests,
    )


def call_qwen_list_reranker(
    messages: Sequence[Mapping[str, str]],
    settings: QwenRerankSettings,
    *,
    post: Callable[..., Any] = requests.post,
) -> QwenCallResult:
    """Call the Ollama-compatible chat endpoint, retrying request failures only."""
    if settings.max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    if settings.max_output_tokens < 1:
        raise ValueError("max_output_tokens must be at least 1")

    payload = {
        "model": settings.model,
        "messages": [dict(message) for message in messages],
        "stream": settings.stream,
        "think": settings.think,
        "options": {
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "num_predict": settings.max_output_tokens,
        },
    }
    for attempt in range(1, settings.max_retries + 1):
        try:
            response = post(
                settings.server_url,
                json=payload,
                timeout=settings.timeout,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as error:
            if attempt < settings.max_retries:
                logger.warning(
                    "Qwen request failed; retrying (%s/%s)",
                    attempt,
                    settings.max_retries,
                )
                continue
            raise QwenRerankError(
                FALLBACK_REQUEST_FAILURE,
                detail=str(error),
                attempts=attempt,
                successful_requests=0,
            ) from error

        try:
            data = response.json()
            content = data["message"]["content"]
        except (KeyError, TypeError, ValueError) as error:
            raise QwenRerankError(
                FALLBACK_MALFORMED_API_RESPONSE,
                attempts=attempt,
                successful_requests=1,
            ) from error
        if not isinstance(content, str) or not content.strip():
            raise QwenRerankError(
                FALLBACK_EMPTY_API_RESPONSE,
                attempts=attempt,
                successful_requests=1,
            )
        return QwenCallResult(content=content, attempts=attempt)

    raise AssertionError("unreachable Qwen retry state")


def rerank_rrf_candidates(
    history: str,
    candidates: Sequence[Mapping[str, Any]],
    settings: QwenRerankSettings,
    *,
    post: Callable[..., Any] = requests.post,
) -> ListRerankResult:
    """Run one list-reranking request or deterministically retain RRF order."""
    if len(candidates) != 50:
        return _fallback_result(
            candidates,
            reason=FALLBACK_INVALID_CANDIDATE_COUNT,
            detail="expected exactly 50 candidates",
            raw_output=None,
            attempts=0,
            successful_requests=0,
        )

    messages = build_list_rerank_prompt(history, candidates)
    try:
        call_result = call_qwen_list_reranker(messages, settings, post=post)
    except QwenRerankError as error:
        return _fallback_result(
            candidates,
            reason=error.code,
            detail=error.detail,
            raw_output=None,
            attempts=error.attempts,
            successful_requests=error.successful_requests,
        )

    try:
        positions = parse_ranked_positions(
            call_result.content,
            candidate_count=len(candidates),
        )
    except RankedPositionsError as error:
        return _fallback_result(
            candidates,
            reason=FALLBACK_INVALID_MODEL_OUTPUT,
            detail=str(error),
            raw_output=call_result.content,
            attempts=call_result.attempts,
            successful_requests=call_result.successful_requests,
        )

    final_candidates = complete_ranking(candidates, positions)
    selected = [candidates[position - 1] for position in positions]
    return ListRerankResult(
        final_candidates=final_candidates,
        raw_output=call_result.content,
        ranked_positions=positions,
        ranked_candidate_ids=[int(candidate["id"]) for candidate in selected],
        ranked_candidate_titles=[str(candidate.get("title", "")) for candidate in selected],
        fallback=False,
        fallback_reason=None,
        fallback_detail=None,
        request_attempts=call_result.attempts,
        successful_requests=call_result.successful_requests,
    )
