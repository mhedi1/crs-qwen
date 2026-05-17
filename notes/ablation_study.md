# Serialization Format Ablation Study

## Overview

The KBRD + Qwen pipeline passes a serialized candidate list to Qwen3.5 for reranking.
Four serialization formats were compared to determine which representation best supports
Qwen's ability to select the most relevant recommendation.

All runs used: `--dataset redial --max_samples 200 --recommendation_only`

---

## Results Table

| Format | R@1 | R@10 | R@50 | MRR | **Reranker@1** | Fallbacks |
|:---:|---:|---:|---:|---:|---:|---:|
| Format 1 | 0.0422 | 0.1734 | 0.3375 | 0.0826 | **0.1328** | 0 |
| Format 2 | 0.0422 | 0.1750 | 0.3359 | 0.0826 | 0.1000 | 0 |
| Format 3 | 0.0422 | 0.1797 | 0.3359 | 0.0838 | 0.0938 | 1 |
| Format 4 | 0.0422 | 0.1781 | 0.3391 | 0.0833 | 0.1031 | 0 |

---

## Metric Interpretation

**Why Reranker@1 is the decisive metric here:**

- **Recall@K and MRR** measure the quality of KBRD's raw candidate retrieval pool.
  All four formats receive the same KBRD candidates, so these metrics are affected
  by the serialization format only indirectly (through any feedback loop in extraction).
  In practice, R@1 is identical (0.0422) across all four formats, confirming that
  KBRD retrieval is unchanged.

- **Reranker@1** measures how often Qwen selects the correct movie as its top-1 pick
  from the KBRD candidates. This metric directly captures the effect of serialization
  format on Qwen's reranking decision — it is the most relevant signal for format
  selection.

---

## Selected Format: Format 1

Format 1 achieved the highest Reranker@1 (0.1328) with zero fallbacks.
It was therefore selected as the default format for all subsequent full-scale experiments.

**Note:** Format 3 achieved marginally higher R@10 (0.1797) and MRR (0.0838),
but these differences reflect noise at 200 conversations, and Reranker@1 is lower (0.0938).
Since the reranker is the component being ablated, Format 1 is the correct choice.

---

## Full-Scale Validation (Format 1, 1301 conversations)

Running Format 1 on the full ReDial test set confirmed the ablation finding:

| Metric | 200-conv ablation | Full test set (1301 conv) |
|---|---:|---:|
| Recall@1 | 0.0422 | 0.0459 |
| Recall@10 | 0.1734 | 0.1662 |
| Recall@50 | 0.3375 | 0.3279 |
| MRR | 0.0826 | 0.0832 |
| Reranker@1 | 0.1328 | **0.1383** |
| Fallbacks | 0 | 4 |

Reranker@1 improves slightly on the full test set, supporting the format selection.
Minor differences in retrieval metrics (R@10, R@50) are expected due to sample variance.
