import os
import json
import yaml
import sys
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
_MY_CRS_DIR = os.path.join(_PROJECT_ROOT, "my_crs")

# Add my_crs to path to import your exact pipeline
sys.path.insert(0, _MY_CRS_DIR)

from evaluate import (
    build_dialogue_up_to, 
    get_recommended_movies_at_turn, 
    get_rank
)
from kbrd_adapter import get_kbrd_candidates
from prompts import build_rerank_prompt

def build_lora_dataset():
    # 1. Load config
    with open(os.path.join(_MY_CRS_DIR, "config.yaml")) as _f:
        _cfg = yaml.safe_load(_f)
        
    format_id = _cfg["pipeline"]["serialisation_format"]
    top_k = _cfg["pipeline"]["top_k_candidates"]
    
    # 2. Strict paths
    train_file = os.path.join(_PROJECT_ROOT, "baseline_repo", "KBRD_project", "KBRD", "data", "redial", "train_data.jsonl")
    output_file = os.path.join(_EXPERIMENTS_DIR, "lora_train.jsonl")
    
    print(f"=== STEP 3: BUILDING LORA TRAINING DATASET ===")
    print(f"Reading from strictly: {train_file}")
    print(f"Using Serialization Format: {format_id}")
    
    dataset_records = []
    skipped_no_hit = 0
    total_processed = 0
    
    with open(train_file, 'r', encoding='utf-8') as f:
        # For time reasons, we will process a max of 2000 conversations for the fine-tuning dataset
        # which yields roughly 3000-4000 recommendation turns (plenty for LoRA).
        max_convos = 2000 
        
        for i, line in enumerate(tqdm(f, total=max_convos, desc="Processing Conversations")):
            if i >= max_convos:
                break
                
            sample = json.loads(line)
            messages = sample.get("messages", [])
            respondent = sample.get("respondentWorkerId", -1)
            
            for turn_index, msg in enumerate(messages):
                sender = msg.get("senderWorkerId", -1)
                
                # If it's a system turn, check if a movie was recommended
                if sender == respondent:
                    recommended_movies = get_recommended_movies_at_turn(sample, turn_index)
                    if not recommended_movies:
                        continue
                        
                    try:
                        # Replicate EXACT inference logic
                        dialogue_up_to = build_dialogue_up_to(sample, turn_index - 1)
                        candidates, detected_decades = get_kbrd_candidates(
                            dialogue_up_to,
                            top_k=top_k,
                            diagnostics=None,
                            use_fusion=True
                        )
                        
                        # Find where the gold movie is in the top-K list
                        gold_rank = get_rank(candidates, recommended_movies)
                        
                        # Only train on instances where KBRD actually found the movie in the top 50
                        # Otherwise, we can't teach the reranker to pick it!
                        if gold_rank > 0:
                            # Build the EXACT prompt used in inference
                            prompt_messages = build_rerank_prompt(
                                history=dialogue_up_to,
                                candidates=candidates,
                                era_hints=detected_decades,
                                serialization_format=format_id
                            )
                            
                            # Append the expected correct answer
                            # Exactly matches parse_answer_id() format
                            prompt_messages.append({
                                "role": "assistant",
                                "content": f"ANSWER: {gold_rank}"
                            })
                            
                            dataset_records.append({"messages": prompt_messages})
                        else:
                            skipped_no_hit += 1
                            
                        total_processed += 1
                        
                    except Exception as e:
                        print(f"Error processing turn: {e}")
                        continue

    # Save out in HuggingFace dataset format
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for record in dataset_records:
            out_f.write(json.dumps(record) + "\n")
            
    print(f"\n--- DATASET GENERATION COMPLETE ---")
    print(f"Total recommendation turns processed: {total_processed}")
    print(f"Turns skipped (gold movie not in top {top_k}): {skipped_no_hit}")
    print(f"Valid Training Examples Generated: {len(dataset_records)}")
    print(f"Dataset saved to: {output_file}")

if __name__ == "__main__":
    build_lora_dataset()
