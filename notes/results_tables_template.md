# CRS Evaluation Results Tables

## Table 1 — Main ReDial Evaluation

| System / Mode | Conversations | Instances | Recall@1 | Recall@10 | Recall@50 | MRR | Reranker@1 | Skipped Instances |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| KBRD + Qwen, Format 3 | 200 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Table 2 — Serialization Format Ablation

| Format | Description | Conversations | Recall@1 | Recall@10 | Recall@50 | MRR | Reranker@1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Format 1 | Basic candidate list | 50/200 | TBD | TBD | TBD | TBD | TBD |
| Format 2 | Candidate list with metadata | 50/200 | TBD | TBD | TBD | TBD | TBD |
| Format 3 | Structured enriched format | 50/200 | TBD | TBD | TBD | TBD | TBD |
| Format 4 | Alternative prompt format | 50/200 | TBD | TBD | TBD | TBD | TBD |

## Table 3 — Response Generation Metrics

| Conversations | Instances | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | Distinct-2 | Distinct-3 | Avg Length |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| System | Recall@1 / Final@1 | Recall@10 | Recall@50 | Precision@1 | NDCG@10 | NDCG@50 | MRR | Reranker@1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Pure KBRD, no fusion, no Qwen | 0.0231 | 0.0980 | 0.2188 | 0.0231 | 0.0487 | 0.0744 | 0.0457 | 0.0231 |
| KBRD + Candidate Fusion, no Qwen | 0.0221 | 0.0967 | 0.2453 | 0.0221 | 0.0481 | 0.0789 | 0.0457 | 0.0221 |
| KBRD + Candidate Fusion + Qwen reranker | 0.0228 | 0.0983 | 0.2447 | 0.0228 | 0.0493 | 0.0799 | 0.0470 | 0.0274 |

These are the final no-leak ReDial results. The dialogue history excludes the current recommendation turn. Therefore, these values are lower than the earlier pre-fix results, but they are methodologically cleaner and should be used for the final thesis comparison.