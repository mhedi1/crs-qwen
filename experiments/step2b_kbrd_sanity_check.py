import os
import json
import yaml
import sys
from dotenv import load_dotenv

load_dotenv()

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_EXPERIMENTS_DIR)
_MY_CRS_DIR = os.path.join(_PROJECT_ROOT, "my_crs")

sys.path.insert(0, _MY_CRS_DIR)

from evaluate import build_dialogue_up_to
from kbrd_adapter import get_kbrd_candidates

def sanity_check():
    print("=== STEP 2b: KBRD INFERENCE SANITY CHECK ===")
    
    test_file = os.path.join(_PROJECT_ROOT, "baseline_repo", "KBRD_project", "KBRD", "data", "redial", "test_data.jsonl")
    
    conversations_to_check = 2
    checked = 0
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if checked >= conversations_to_check:
                break
                
            sample = json.loads(line)
            messages = sample.get("messages", [])
            respondent = sample.get("respondentWorkerId", -1)
            
            for turn_index, msg in enumerate(messages):
                sender = msg.get("senderWorkerId", -1)
                
                # Check at the first respondent recommendation turn
                if sender == respondent and "@" in msg.get("text", ""):
                    
                    dialogue_up_to = build_dialogue_up_to(sample, turn_index - 1)
                    
                    print(f"--- CONVERSATION {sample.get('conversationId')} ---")
                    print("[DIALOGUE HISTORY]")
                    print(dialogue_up_to)
                    print("\n[RUNNING KBRD INFERENCE...]")
                    
                    try:
                        candidates, detected_decades = get_kbrd_candidates(
                            dialogue_up_to,
                            top_k=10,
                            diagnostics=None,
                            use_fusion=True
                        )
                        
                        print("\n[TOP 10 CANDIDATES]")
                        for i, c in enumerate(candidates):
                            title = c.get('title', 'Unknown')
                            genre = c.get('genre', 'Unknown')
                            source = c.get('source', 'Unknown')
                            print(f"{i+1}. {title} | Genre: {genre} | Source: {source}")
                            
                        print("\n" + "="*50 + "\n")
                        checked += 1
                        break # Move to next conversation
                        
                    except Exception as e:
                        print(f"[ERROR] Inference crashed! PyTorch compatibility issue?: {e}")
                        sys.exit(1)

if __name__ == "__main__":
    sanity_check()
