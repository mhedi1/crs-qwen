import os
import json
import re
import sys
from evaluate import normalize_title, strict_title_match
from reranker import call_qwen
from evaluate import get_recommended_movies_at_turn, build_dialogue_up_to

_MY_CRS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MY_CRS_DIR)

def build_zeroshot_prompt(dialogue_history: str) -> str:
    prompt = (
        "You are a movie recommender system. Based on the user's dialogue history below, "
        "recommend exactly 10 movies that fit their preferences.\n\n"
        "RULES:\n"
        "- Output exactly a numbered list from 1 to 10.\n"
        "- Only include the movie title on each line.\n"
        "- Do NOT include release years, descriptions, or any conversational text.\n\n"
        "Dialogue History:\n"
        f"{dialogue_history}\n\n"
        "Recommendations:\n"
    )
    return prompt

def parse_zeroshot_output(output: str) -> list:
    """Parse the numbered list from Qwen into a list of candidate dicts."""
    candidates = []
    lines = output.strip().split('\n')
    for line in lines:
        line = line.strip()
        # Remove leading numbers and punctuation (e.g. "1.", "1)", "1 - ")
        title = re.sub(r'^\d+[\.\)\-\*]*\s*', '', line).strip()
        
        # Sometimes LLM puts titles in quotes
        title = re.sub(r'^["\']|["\']$', '', title)
        
        if title:
            candidates.append({"title": title})
            
    # LLM might return more or fewer than 10, but we cap it at 10 for evaluation
    return candidates[:10]

def evaluate_zeroshot(test_path: str, max_samples: int = 1500):
    print(f"\nEvaluating Zero-Shot LLM baseline on test set (Max {max_samples} conversations)...")
    hits = {1: [], 10: []}
    mrrs = []
    
    total_conversations = 0
    total_instances = 0
    fallbacks = 0
    
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            if total_conversations >= max_samples:
                break
                
            sample = json.loads(line)
            messages = sample.get("messages", [])
            respondent = sample.get("respondentWorkerId", -1)
            
            conv_has_instances = False
            
            for turn_index, msg in enumerate(messages):
                if msg.get("senderWorkerId", -1) == respondent:
                    recommended_movies = get_recommended_movies_at_turn(sample, turn_index)
                    if not recommended_movies:
                        continue
                        
                    dialogue_up_to = build_dialogue_up_to(sample, turn_index - 1)
                    prompt = build_zeroshot_prompt(dialogue_up_to)
                    
                    try:
                        raw_output = call_qwen(prompt)
                        candidates = parse_zeroshot_output(raw_output)
                    except Exception as e:
                        print(f"[ERROR] LLM call failed at conv turn {turn_index}: {e}")
                        candidates = []
                        fallbacks += 1
                        
                    # If empty, fill with dummy so metrics don't crash
                    if not candidates:
                        candidates = [{"title": "Unknown"}] * 10
                        
                    total_instances += 1
                    conv_has_instances = True
                    
                    # Calculate hits
                    for k in [1, 10]:
                        top_k = candidates[:k]
                        hit = False
                        for c in top_k:
                            for gt in recommended_movies:
                                if strict_title_match(c["title"], gt):
                                    hit = True
                                    break
                            if hit: break
                        hits[k].append(hit)
                        
                    # Calculate MRR
                    rank = 0
                    for r, c in enumerate(candidates, 1):
                        found = False
                        for gt in recommended_movies:
                            if strict_title_match(c["title"], gt):
                                found = True
                                break
                        if found:
                            rank = r
                            break
                    mrrs.append(1.0 / rank if rank > 0 else 0.0)
                    
                    if total_instances % 10 == 0:
                        r1 = sum(hits[1]) / len(hits[1]) if hits[1] else 0
                        r10 = sum(hits[10]) / len(hits[10]) if hits[10] else 0
                        print(f"[{total_instances} instances] Recall@1={r1:.4f} Recall@10={r10:.4f}")
                        
            if conv_has_instances:
                total_conversations += 1

    print(f"\n==========================================")
    print(f"ZERO-SHOT LLM BASELINE RESULTS")
    print(f"==========================================")
    print(f"Conversations: {total_conversations}")
    print(f"Instances:     {total_instances}")
    print(f"Fallbacks:     {fallbacks}")
    for k in [1, 10]:
        score = sum(hits[k]) / len(hits[k]) if hits[k] else 0.0
        print(f"Recall@{k}:     {score:.4f}")
    mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
    print(f"MRR:           {mrr:.4f}")
    print(f"==========================================\n")

if __name__ == "__main__":
    test_path = os.path.join(_PROJECT_ROOT, "baseline_repo", "KBRD_project", "KBRD", "data", "redial", "test_data.jsonl")
    
    if not os.path.exists(test_path):
        print("Data paths not found. Please ensure script is run from project root or my_crs dir.")
        sys.exit(1)
        
    evaluate_zeroshot(test_path, max_samples=1500)
