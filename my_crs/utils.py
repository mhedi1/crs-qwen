import json
import re
from pathlib import Path


def parse_answer_id(text: str, max_val: int = 50) -> int | None:
    """
    Extract an answer like: ANSWER: 3
    Returns the first integer id in the range [1, max_val] if found, otherwise None.
    """
    for match in re.finditer(r"ANSWER:\s*(\d+)", text):
        val = int(match.group(1))
        if 1 <= val <= max_val:
            return val
    return None


def save_json(data, path: str) -> None:
    """
    Save data as JSON with UTF-8 encoding.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    """
    Load JSON file and return Python object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)