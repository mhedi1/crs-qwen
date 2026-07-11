import json

path = 'experiments/error_analysis/error_analysis_redial_format1_reranked_20260703_162233.jsonl'

reranking_example = None
retrieval_example = None

with open(path) as f:
    for line in f:
        d = json.loads(line)
        is_correct = d.get('correct', False)
        gold_in_top_50 = d.get('gold_in_top_50', False)
        if not is_correct:
            if gold_in_top_50 and reranking_example is None:
                reranking_example = d
            if not gold_in_top_50 and retrieval_example is None:
                retrieval_example = d
        if reranking_example and retrieval_example:
            break

print('=' * 60)
print('RETRIEVAL FAILURE EXAMPLE')
print('=' * 60)
print()
print('Dialogue:')
print(retrieval_example.get('dialogue_history', '')[-300:])
print()
print('Extracted Seeds:')
for s in retrieval_example.get('extracted_seeds', []):
    print(f'  - {s}')
print()
print('Ground Truth:', retrieval_example.get('ground_truth_movies'))
print('Gold in Top-50:', retrieval_example.get('gold_in_top_50'))
print('Selected Movie:', retrieval_example.get('selected_movie'))
print()
print('=' * 60)
print('RERANKING FAILURE EXAMPLE')
print('=' * 60)
print()
print('Dialogue:')
print(reranking_example.get('dialogue_history', '')[-300:])
print()
print('Extracted Seeds:')
for s in reranking_example.get('extracted_seeds', []):
    print(f'  - {s}')
print()
print('Ground Truth:', reranking_example.get('ground_truth_movies'))
print('Gold in Top-50:', reranking_example.get('gold_in_top_50'))
print('Gold Rank:', reranking_example.get('gold_rank'))
print('Selected Movie:', reranking_example.get('selected_movie'))