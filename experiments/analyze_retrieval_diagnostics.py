import json
from pathlib import Path
from collections import Counter

path = Path("experiments/error_analysis/error_analysis_redial_format1_reranked_20260518_160735.jsonl")

records = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

total = len(records)

def pct(x):
    return f"{x / total * 100:.2f}%" if total else "0.00%"

weak = sum(1 for r in records if r.get("weak_seed_fallback"))
no_extracted = sum(1 for r in records if r.get("num_extracted_seeds", 0) == 0)
no_matched = sum(1 for r in records if r.get("num_matched_seeds", 0) == 0)
gold_not_top50 = sum(1 for r in records if not r.get("gold_in_top_50"))
gold_top50 = sum(1 for r in records if r.get("gold_in_top_50"))

matched_but_gold_missing = sum(
    1 for r in records
    if r.get("num_matched_seeds", 0) > 0 and not r.get("gold_in_top_50")
)

weak_and_gold_missing = sum(
    1 for r in records
    if r.get("weak_seed_fallback") and not r.get("gold_in_top_50")
)

weak_and_gold_available = sum(
    1 for r in records
    if r.get("weak_seed_fallback") and r.get("gold_in_top_50")
)

qwen_fallback_with_suggestions = sum(
    1 for r in records
    if len(r.get("qwen_fallback_seeds", [])) > 0
)

print("\n=== Retrieval Diagnostics Summary ===")
print("File:", path)
print("Total instances:", total)
print("Weak seed fallback used:", weak, pct(weak))
print("Qwen fallback produced suggestions:", qwen_fallback_with_suggestions, pct(qwen_fallback_with_suggestions))
print("No extracted seeds:", no_extracted, pct(no_extracted))
print("No matched seeds:", no_matched, pct(no_matched))
print("Gold in top-50:", gold_top50, pct(gold_top50))
print("Gold not in top-50:", gold_not_top50, pct(gold_not_top50))
print("Matched seeds but gold missing:", matched_but_gold_missing, pct(matched_but_gold_missing))
print("Weak fallback + gold missing:", weak_and_gold_missing, pct(weak_and_gold_missing))
print("Weak fallback + gold available:", weak_and_gold_available, pct(weak_and_gold_available))

print("\n=== Most Common Extracted Seeds ===")
seed_counter = Counter()
for r in records:
    for seed in r.get("extracted_seeds", []):
        seed_counter[str(seed).lower()] += 1

for seed, count in seed_counter.most_common(30):
    print(f"{count:4d}  {seed}")

print("\n=== Example Failures: Matched Seeds But Gold Missing ===")
shown = 0
for r in records:
    if r.get("num_matched_seeds", 0) > 0 and not r.get("gold_in_top_50"):
        print(json.dumps({
            "conversation_id": r.get("conversation_id"),
            "turn_index": r.get("turn_index"),
            "ground_truth_movies": r.get("ground_truth_movies"),
            "extracted_seeds": r.get("extracted_seeds"),
            "qwen_fallback_seeds": r.get("qwen_fallback_seeds"),
            "num_extracted_seeds": r.get("num_extracted_seeds"),
            "num_matched_seeds": r.get("num_matched_seeds"),
            "weak_seed_fallback": r.get("weak_seed_fallback"),
            "kbrd_top1": r.get("kbrd_top1"),
            "selected_movie": r.get("selected_movie"),
        }, ensure_ascii=False, indent=2))
        shown += 1
        if shown == 10:
            break
