\# Final No-Leak ReDial Ablation Results



Evaluation protocol:

\- Dataset: ReDial test set

\- Conversations processed: 1301

\- Evaluation instances: 3898

\- Recommendation-only mode

\- No-leak setting: dialogue history excludes the current recommendation turn



| System | Recall@1 / Final@1 | Recall@10 | Recall@50 | Precision@1 | NDCG@10 | NDCG@50 | MRR | Reranker@1 |

|---|---:|---:|---:|---:|---:|---:|---:|---:|

| Pure KBRD, no fusion, no Qwen | 0.0231 | 0.0980 | 0.2188 | 0.0231 | 0.0487 | 0.0744 | 0.0457 | 0.0231 |

| KBRD + Candidate Fusion, no Qwen | 0.0221 | 0.0967 | 0.2453 | 0.0221 | 0.0481 | 0.0789 | 0.0457 | 0.0221 |

| KBRD + Candidate Fusion + Qwen reranker | 0.0228 | 0.0983 | 0.2447 | 0.0228 | 0.0493 | 0.0799 | 0.0470 | 0.0274 |



Main conclusions:

\- Candidate Fusion improves Recall@50 from 0.2188 to about 0.245, showing better candidate coverage.

\- Fusion alone does not improve top-1 because fused candidates are inserted after the preserved KBRD top-30.

\- Qwen reranking improves final selected recommendation accuracy under fusion, from 0.0221 to 0.0274.

\- The main remaining bottleneck is retrieval coverage and ranking under the strict no-leak evaluation.

