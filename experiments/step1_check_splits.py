import os
import json

def check_splits():
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            "baseline_repo", "KBRD_project", "KBRD", "data", "redial")
    
    train_file = os.path.join(base_dir, "train_data.jsonl")
    test_file = os.path.join(base_dir, "test_data.jsonl")
    
    print("=== STEP 1: DATA SPLIT VERIFICATION ===")
    
    for name, path in [("TRAIN", train_file), ("TEST (Held-out)", test_file)]:
        if not os.path.exists(path):
            print(f"[ERROR] {name} file not found at: {path}")
            continue
            
        count = 0
        sample_ids = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                count += 1
                if count <= 5:
                    sample_ids.append(str(data.get("conversationId")))
                    
        print(f"\n{name} SPLIT:")
        print(f"Path: {path}")
        print(f"Total Conversations: {count}")
        print(f"First 5 Conversation IDs: {', '.join(sample_ids)}")

if __name__ == "__main__":
    check_splits()
