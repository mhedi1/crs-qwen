# One-Word Seed Quality Filter Results

## Motivation

Retrieval diagnostics showed that many extracted KBRD seeds were noisy one-word movie-title matches. Common dialogue words such as "it", "saw", "time", "help", and "loved" were sometimes matched as movie titles because they exist in the KBRD/DBpedia vocabulary.

This polluted the KBRD seed list and reduced candidate retrieval quality.

## Method

A conservative one-word seed quality filter was added before KBRD retrieval.

The filter:
- applies only to one-word candidate phrases
- keeps multi-word movie titles unchanged
- removes ambiguous stopwords, pronouns, verbs, auxiliaries, and determiners
- allows one-word titles only when context is strong enough
- keeps Qwen fallback behavior unchanged
- keeps KBRD ranking logic unchanged
- keeps Qwen reranking logic unchanged

## Full ReDial Evaluation After Filter

Configuration:
- Dataset: ReDial
- Format: 1
- Conversations: 1301
- Recommendation instances: 3898
- Result file: experiments/eval_format1_redial_20260518_193503.json
- Error analysis file: experiments/error_analysis/error_analysis_redial_format1_reranked_20260518_193503.jsonl

| Metric | Before Filter | After Filter |
|---|---:|---:|
| Recall@1 | 0.0462 | 0.0485 |
| Recall@10 | 0.1647 | 0.1875 |
| Recall@50 | 0.3261 | 0.3386 |
| MRR | 0.0831 | 0.0897 |
| Reranker@1 | 0.1393 | 0.1539 |

## Diagnostic Summary After Filter

- Gold in top-50: 1320 / 3898 = 33.86%
- Gold not in top-50: 2578 / 3898 = 66.14%
- Weak seed fallback used: 1133 / 3898 = 29.07%
- No extracted seeds: 47 / 3898 = 1.21%
- No matched seeds: 9 / 3898 = 0.23%
- Instances with filtered noisy seeds: 3896 / 3898 = 99.95%
- Total filtered noisy seeds: 65,764

## Interpretation

The one-word seed quality filter improved all main recommendation metrics. The largest improvement was Recall@10, which increased from 0.1647 to 0.1875.

This confirms that noisy one-word entity matches were hurting KBRD retrieval and downstream Qwen reranking. The improvement is thesis-defensible because it addresses a general entity-linking disambiguation problem: ambiguous one-word dialogue tokens can be incorrectly matched as movie titles. The filter does not target specific test conversations, and it preserves multi-word movie titles and the original KBRD/Qwen ranking logic.

Some noisy seeds still remain, such as "war", "it", "time", "nice", and "lol", so further retrieval improvement is possible. However, the conservative filter already improves the system without changing KBRD ranking or reranker logic.

## Official Extended-Metrics ReDial Run

After adding extended recommendation metrics, the full ReDial evaluation was rerun using the seed-filtered system.

Configuration:
- Dataset: ReDial
- Format: 1
- Conversations processed: 1301
- Evaluation instances: 3898
- Result file: experiments/eval_format1_redial_20260519_175558.json
- Error analysis file: experiments/error_analysis/error_analysis_redial_format1_reranked_20260519_175558.jsonl
- MLflow run ID: 5427961d37af4d628b32dbaf62f6504c

| Metric | Value |
|---|---:|
| Recall@1 | 0.0511 |
| Recall@10 | 0.1893 |
| Recall@50 | 0.3397 |
| Precision@1 | 0.0511 |
| Precision@10 | 0.0198 |
| Precision@50 | 0.0074 |
| NDCG@10 | 0.0994 |
| NDCG@50 | 0.1331 |
| MRR | 0.0919 |
| Reranker@1 | 0.1503 |
| AvgGoldRank | 13.43 |
| MedianGoldRank | 9.0 |

Conversation-level metrics:

| Metric | Value |
|---|---:|
| ConvSuccess@1 | 0.3697 |
| ConvRecall@10 | 0.4374 |
| ConvRecall@50 | 0.6626 |

Interpretation:
This run is the official main ReDial result for the seed-filtered system. It confirms that the one-word seed quality filter improves candidate retrieval and downstream reranking while also providing stronger ranking-quality metrics such as NDCG and conversation-level success.