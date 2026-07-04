import os
import json
import re
import math
import sys
from collections import Counter

_MY_CRS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_MY_CRS_DIR)

def normalize_title(title: str) -> str:
    title = re.sub(r'\(\d{4}\)', '', title)
    title = re.sub(r'[^\w\s]', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title.lower().strip()

def strict_title_match(title_a: str, title_b: str) -> bool:
    return normalize_title(title_a) == normalize_title(title_b)

def get_recommended_movies_at_turn(sample: dict, turn_index: int) -> list:
    messages = sample.get("messages", [])
    if turn_index >= len(messages):
        return []
        
    msg = messages[turn_index]
    text = msg.get("text", "")
    movie_mentions = sample.get("movieMentions", {})
    respondent_questions = sample.get("respondentQuestions", {})

    if isinstance(respondent_questions, list):
        respondent_questions = {}

    movie_ids_in_text = re.findall(r'@(\d+)', text)
    recommended_movies = []
    for movie_id in movie_ids_in_text:
        info = respondent_questions.get(movie_id, {})
        if info.get("suggested", 0) == 1:
            movie_name = movie_mentions.get(movie_id, "")
            if movie_name:
                recommended_movies.append(movie_name.strip().lower())
                
    return recommended_movies

def build_popularity_list(train_path: str, top_n: int = 50) -> list:
    print("Scanning training set to build popularity baseline...")
    movie_counter = Counter()
    
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            messages = sample.get("messages", [])
            respondent = sample.get("respondentWorkerId", -1)
            
            for turn_index, msg in enumerate(messages):
                if msg.get("senderWorkerId", -1) == respondent:
                    recommended = get_recommended_movies_at_turn(sample, turn_index)
                    for movie in recommended:
                        norm = normalize_title(movie)
                        if norm:
                            movie_counter[norm] += 1
                            
    most_common = movie_counter.most_common(top_n)
    print(f"Top 5 most popular movies: {[m[0] for m in most_common[:5]]}")
    return [{"title": m[0]} for m in most_common]

def evaluate_popularity(test_path: str, pop_candidates: list):
    print("\nEvaluating static popularity baseline on test set...")
    hits = {1: [], 10: [], 50: []}
    mrrs = []
    total_instances = 0
    
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            messages = sample.get("messages", [])
            respondent = sample.get("respondentWorkerId", -1)
            
            for turn_index, msg in enumerate(messages):
                if msg.get("senderWorkerId", -1) == respondent:
                    recommended_movies = get_recommended_movies_at_turn(sample, turn_index)
                    if not recommended_movies:
                        continue
                        
                    total_instances += 1
                    
                    # Calculate hits
                    for k in [1, 10, 50]:
                        top_k = pop_candidates[:k]
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
                    for r, c in enumerate(pop_candidates, 1):
                        found = False
                        for gt in recommended_movies:
                            if strict_title_match(c["title"], gt):
                                found = True
                                break
                        if found:
                            rank = r
                            break
                    mrrs.append(1.0 / rank if rank > 0 else 0.0)

    print(f"\n==========================================")
    print(f"POPULARITY BASELINE RESULTS")
    print(f"==========================================")
    print(f"Instances Evaluated: {total_instances}")
    for k in [1, 10, 50]:
        score = sum(hits[k]) / len(hits[k]) if hits[k] else 0.0
        print(f"Recall@{k}: {score:.4f}")
    mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
    print(f"MRR:       {mrr:.4f}")
    print(f"==========================================\n")

if __name__ == "__main__":
    train_path = os.path.join(_PROJECT_ROOT, "baseline_repo", "KBRD_project", "KBRD", "data", "redial", "train_data.jsonl")
    test_path = os.path.join(_PROJECT_ROOT, "baseline_repo", "KBRD_project", "KBRD", "data", "redial", "test_data.jsonl")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Data paths not found. Please ensure script is run from project root or my_crs dir.")
        sys.exit(1)
        
    pop_candidates = build_popularity_list(train_path, top_n=50)
    evaluate_popularity(test_path, pop_candidates)
