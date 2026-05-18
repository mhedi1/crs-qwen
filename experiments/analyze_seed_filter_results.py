import json
from pathlib import Path
from collections import Counter

files = sorted(Path("experiments/error_analysis").glob("error_analysis_redial_format1_reranked_*.jsonl"))
path = files[-1]

records = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

total = len(records)

def pct(x):
    return f"{x / total * 100:.2f}%" if total else "0.00%"

gold_top50 = sum(1 for r in records if r.get("gold_in_top_50"))
gold_not_top50 = sum(1 for r in records if not r.get("gold_in_top_50"))
weak = sum(1 for r in records if r.get("weak_seed_fallback"))
no_extracted = sum(1 for r in records if r.get("num_extracted_seeds", 0) == 0)
no_matched = sum(1 for r in records if r.get("num_matched_seeds", 0) == 0)

filtered_instances = sum(1 for r in records if r.get("num_filtered_noisy_seeds", 0) > 0)
total_filtered = sum(r.get("num_filtered_noisy_seeds", 0) for r in records)

filtered_counter = Counter()
for r in records:
    for seed in r.get("filtered_noisy_seeds", []):
        filtered_counter[str(seed).lower()] += 1

extracted_counter = Counter()
for r in records:
    for seed in r.get("extracted_seeds", []):
        extracted_counter[str(seed).lower()] += 1

print("Using:", path)
print()
print("=== Seed Filter Diagnostic Summary ===")
print("Total instances:", total)
print("Gold in top-50:", gold_top50, pct(gold_top50))
print("Gold not in top-50:", gold_not_top50, pct(gold_not_top50))
print("Weak seed fallback used:", weak, pct(weak))
print("No extracted seeds:", no_extracted, pct(no_extracted))
print("No matched seeds:", no_matched, pct(no_matched))
print("Instances with filtered noisy seeds:", filtered_instances, pct(filtered_instances))
print("Total filtered noisy seeds:", total_filtered)

print("\n=== Most Common Filtered Noisy Seeds ===")
for seed, count in filtered_counter.most_common(30):
    print(f"{count:4d}  {seed}")

print("\n=== Most Common Remaining Extracted Seeds ===")
for seed, count in extracted_counter.most_common(30):
    print(f"{count:4d}  {seed}")

print("\n=== Examples where noisy seeds were filtered ===")
shown = 0
for r in records:
    if r.get("num_filtered_noisy_seeds", 0) > 0:
        print(json.dumps({
            "conversation_id": r.get("conversation_id"),
            "turn_index": r.get("turn_index"),
            "ground_truth_movies": r.get("ground_truth_movies"),
            "filtered_noisy_seeds": r.get("filtered_noisy_seeds"),
            "extracted_seeds": r.get("extracted_seeds"),
            "num_filtered_noisy_seeds": r.get("num_filtered_noisy_seeds"),
            "num_extracted_seeds": r.get("num_extracted_seeds"),
            "num_matched_seeds": r.get("num_matched_seeds"),
            "gold_in_top_50": r.get("gold_in_top_50"),
            "kbrd_top1": r.get("kbrd_top1"),
            "selected_movie": r.get("selected_movie"),
        }, ensure_ascii=False, indent=2))
        shown += 1
        if shown == 5:
            break
