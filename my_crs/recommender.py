import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "baseline_repo", "KBRD_project", "KBRD"))

from kbrd_adapter import get_kbrd_candidates
from reranker import rerank
from response_generator import generate_response


def get_recommendation(dialogue_history: list) -> dict:
    """
    Main entry point for the CRS pipeline.
    
    Args:
        dialogue_history: list of dicts like:
        [
            {"role": "user", "content": "I want horror"},
            {"role": "system", "content": "Did you see The Shining?"},
            {"role": "user", "content": "Yes, I want something old"}
        ]
    
    Returns:
        {
            "response": "natural language response",
            "movie": {
                "title": "...",
                "genre": "...", 
                "decade": "..."
            },
            "candidates": [top 5 candidates]
        }
    """
    # Convert history list to dialogue string
    turns = []
    for turn in dialogue_history:
        role = "User" if turn["role"] == "user" else "System"
        turns.append(f"{role}: {turn['content']}")
    dialogue_str = "\n".join(turns)
    
    # Stage 1: Get candidates from KBRD
    candidates, detected_decades = get_kbrd_candidates(dialogue_str, top_k=50)
    
    # Stage 2: Rerank with Qwen
    selected_movie, _ = rerank(dialogue_str, candidates, era_hints=detected_decades)
    
    # Stage 3: Generate response with Qwen
    response = generate_response(dialogue_str, selected_movie)
    
    return {
        "response": response,
        "movie": {
            "title": selected_movie.get("title", "Unknown"),
            "genre": selected_movie.get("genre", "Unknown"),
            "decade": selected_movie.get("decade", "Unknown")
        },
        "candidates": candidates[:5]
    }


if __name__ == "__main__":
    
    # Test 1 — Original horror test
    print("\n" + "="*60)
    print("TEST 1: Old horror film")
    print("="*60)
    test1 = [
        {"role": "user",
         "content": "I am looking for a horror film"},
        {"role": "system",
         "content": "Did you see A Nightmare on Elm Street?"},
        {"role": "user",
         "content": "Yes I have seen it. I want something old."}
    ]
    result1 = get_recommendation(test1)
    print("Movie:", result1["movie"]["title"])
    print("Genre:", result1["movie"]["genre"])
    print("Decade:", result1["movie"]["decade"])
    print("Response:", result1["response"])

    # Test 2 — Actor name test
    print("\n" + "="*60)
    print("TEST 2: Actor name (Tom Hanks)")
    print("="*60)
    test2 = [
        {"role": "user",
         "content": "I love Tom Hanks movies"},
        {"role": "system",
         "content": "Have you seen Forrest Gump?"},
        {"role": "user",
         "content": "Yes loved it, something similar please"}
    ]
    result2 = get_recommendation(test2)
    print("Movie:", result2["movie"]["title"])
    print("Genre:", result2["movie"]["genre"])
    print("Decade:", result2["movie"]["decade"])
    print("Response:", result2["response"])

    # Test 3 — Temporal clue test
    print("\n" + "="*60)
    print("TEST 3: Decade preference (80s horror)")
    print("="*60)
    test3 = [
        {"role": "user",
         "content": "I want something from the 80s"},
        {"role": "system",
         "content": "Do you like horror from that era?"},
        {"role": "user",
         "content": "Yes exactly, old classic horror"}
    ]
    result3 = get_recommendation(test3)
    print("Movie:", result3["movie"]["title"])
    print("Genre:", result3["movie"]["genre"])
    print("Decade:", result3["movie"]["decade"])
    print("Response:", result3["response"])
