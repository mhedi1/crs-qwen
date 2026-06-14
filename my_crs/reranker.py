import os
import yaml
import requests
import logging
from prompts import build_rerank_prompt
from utils import parse_answer_id

logger = logging.getLogger(__name__)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")) as _f:
    _cfg = yaml.safe_load(_f)

LLM_URL = _cfg["qwen"]["server_url"]
MODEL_NAME = _cfg["qwen"]["model"]
_TIMEOUT = _cfg["qwen"]["timeout"]
_MAX_RETRIES = _cfg["qwen"]["max_retries"]
_THINK = _cfg["qwen"]["think"]
_TEMPERATURE = _cfg["qwen"]["temperature"]
_STREAM = _cfg["qwen"]["stream"]
USE_FAKE_MODE = False


def call_qwen(messages) -> str:
    """Send a prompt to the Qwen LLM and return its text response.

    Args:
        messages: A string prompt or a list of OpenAI-style message dicts.

    Returns:
        The model's response content as a string. Raises
        RequestException if all retries are exhausted.
    """
    if USE_FAKE_MODE:
        if isinstance(messages, list):
            prompt = messages[-1]["content"]
        else:
            prompt = messages
        prompt_lower = prompt.lower()

        if "superhero" in prompt_lower or "marvel" in prompt_lower or "modern" in prompt_lower:
            return "ANSWER: 8"

        if "funny" in prompt_lower or "family" in prompt_lower or "animation" in prompt_lower:
            return "ANSWER: 7"

        if "horror" in prompt_lower or "old" in prompt_lower or "classic" in prompt_lower:
            return "ANSWER: 4"

        return "ANSWER: 1"

    if isinstance(messages, str):
        payload_messages = [{"role": "user", "content": messages}]
    else:
        payload_messages = messages

    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.post(
                LLM_URL,
                json={
                    "model": MODEL_NAME,
                    "messages": payload_messages,
                    "stream": _STREAM,
                    "think": _THINK,
                    "temperature": _TEMPERATURE,
                },
                timeout=_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except requests.exceptions.RequestException as e:
            if attempt < _MAX_RETRIES - 1:
                logger.warning(f"Qwen timeout or error, retrying ({attempt + 1}/{_MAX_RETRIES})...")
            else:
                logger.error(f"[QWEN ERROR] {e}")
                raise


def rerank(history: str, candidates: list[dict], era_hints: list = None, serialization_format: int = 1, previously_recommended: list = None) -> tuple[dict, bool]:
    """Select the best candidate movie using Qwen as a reranker.

    Args:
        history: Dialogue history string passed as context to Qwen.
        candidates: Ranked list of candidate dicts from KBRD.
        era_hints: Optional decade strings detected in the dialogue.
        serialization_format: Integer (1-4) controlling the prompt format.

    Returns:
        Tuple of (selected_movie_dict, is_fallback). is_fallback is True when
        Qwen fails or returns an unparseable answer.
    """
    prompt = build_rerank_prompt(history, candidates, era_hints=era_hints, serialization_format=serialization_format, previously_recommended=previously_recommended)
    
    try:
        raw_output = call_qwen(prompt)
    except Exception as e:
        logger.warning("[Reranker] Qwen call failed after retries. Using fallback.")
        return (candidates[0] if candidates else {"title": "Unknown", "genre": "Unknown", "decade": "Unknown"}, True)

    answer_idx = parse_answer_id(raw_output)

    if answer_idx is None:
        logger.warning("[Reranker] Could not parse answer. Using fallback.")
        return (candidates[0] if candidates else {"title": "Unknown", "genre": "Unknown", "decade": "Unknown"}, True)

    if answer_idx < 1 or answer_idx > len(candidates):
        logger.warning(f"[Reranker] Answer index {answer_idx} out of range. Using fallback.")
        return (candidates[0] if candidates else {"title": "Unknown", "genre": "Unknown", "decade": "Unknown"}, True)

    selected = candidates[answer_idx - 1]
    logger.info(f"[Reranker] Qwen selected position {answer_idx}: {selected['title']}")
    return (selected, False)