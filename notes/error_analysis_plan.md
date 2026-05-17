# Error Analysis Plan

## Purpose

Identify and categorize failures in the KBRD + Qwen pipeline to understand
where the system breaks down and what improvements would have the most impact.

---

## Output Format

For each evaluation instance that is a failure (ground-truth movie not in top-1),
save a record to a JSONL file alongside the standard eval JSON. Suggested path:
`experiments/errors_format{N}_{dataset}_{timestamp}.jsonl`

---

## Fields to Save per Instance

| Field | Description |
|---|---|
| `conversation_id` | Dialogue identifier (e.g., ReDial `conversationId`) |
| `turn_index` | Index of the recommendation turn within the dialogue |
| `dialogue_context` | Full dialogue text up to and including the recommendation turn |
| `gold_movie` | Ground-truth recommended movie title(s) |
| `kbrd_top1` | Title of KBRD's rank-1 candidate (before reranking) |
| `qwen_selected` | Title selected by Qwen after reranking |
| `gold_in_top50` | Boolean — whether any gold movie appears in the KBRD top-50 |
| `extracted_seeds` | Movie seeds extracted from the dialogue context (used by KBRD) |
| `detected_decade_hints` | Era/decade hints passed to the reranker |
| `qwen_fallback_used` | Boolean — whether the reranker fell back to KBRD top-1 |
| `format` | Serialization format number |
| `error_type` | Assigned error category (see below) |

---

## Error Categories

| Category | Definition |
|---|---|
| `retrieval_failure` | Gold movie not in KBRD top-50; the retriever never surfaces it. No reranker can fix this. |
| `reranking_failure` | Gold movie IS in top-50, but Qwen selects a different movie. The retriever is correct; the reranker is wrong. |
| `entity_extraction_failure` | No seeds or very few seeds extracted from the dialogue; KBRD is likely querying on weak signal. |
| `parsing_api_failure` | Qwen returned an unparseable response or triggered a fallback; Qwen's output format was invalid. |

**Assignment logic:**

```
if gold_in_top50 is False:
    error_type = "retrieval_failure"
elif qwen_fallback_used is True:
    error_type = "parsing_api_failure"
elif len(extracted_seeds) == 0:
    error_type = "entity_extraction_failure"
else:
    error_type = "reranking_failure"
```

---

## Analysis Protocol

1. Run error analysis on the Format 1 full ReDial run (3898 instances).
2. Count instances by error type; compute percentage of total failures.
3. Sample 20 `reranking_failure` cases for qualitative inspection:
   - Does the dialogue context contain enough signal?
   - Is Qwen selecting a plausible but wrong movie?
   - Is the gold movie semantically similar to Qwen's pick?
4. Sample 10 `retrieval_failure` cases:
   - Is the gold movie in KBRD's vocabulary at all?
   - Is the dialogue too short or too generic for KBRD to retrieve it?
5. Document findings in thesis limitations/error analysis section.
