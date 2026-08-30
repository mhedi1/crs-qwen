"""VALID-only offline Reciprocal Rank Fusion for frozen KBRD and CKG rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from my_crs.ckg_retriever import CKGRetriever
from my_crs.evaluate_ckg_complementarity import (
    DEFAULT_CACHE_DIR,
    DEFAULT_PARITY_OUTPUT_PATH,
    DEFAULT_VALID_PATH,
    PROJECT_ROOT,
    _atomic_json,
    _atomic_jsonl,
    _sha256,
    attach_catalogue_titles,
    extraction_configuration,
    require_passing_parity_report,
    validate_valid_source_path,
)


RRF_K = 60
TOP_K = 50
KBRD_WEIGHT = 1.0
CKG_WEIGHT = 1.0
FROZEN_CKG_GRAPH_TYPE = "conversation"
FROZEN_CKG_WEIGHTING = "conditional"
FROZEN_CKG_MIN_SUPPORT = 2
FROZEN_CKG_CACHE_NAME = "conversation_conditional_support2.pkl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "rrf_fusion_valid.json"
DEFAULT_INSTANCE_OUTPUT_PATH = (
    PROJECT_ROOT / "experiments" / "rrf_fusion_valid.instances.jsonl"
)


HitFunction = Callable[[list[dict[str, Any]], list[str], int], bool]
RankFunction = Callable[[list[dict[str, Any]], list[str]], int]


def _deduplicate_ranking(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy a ranking while retaining only the first occurrence of each entity ID."""
    unique: list[dict[str, Any]] = []
    seen: set[int] = set()
    for candidate in candidates:
        entity_id = int(candidate["id"])
        if entity_id in seen:
            continue
        seen.add(entity_id)
        copied = dict(candidate)
        copied["id"] = entity_id
        unique.append(copied)
    return unique


def reciprocal_rank_fusion(
    kbrd_candidates: Sequence[Mapping[str, Any]],
    ckg_candidates: Sequence[Mapping[str, Any]],
    *,
    rrf_k: int = RRF_K,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    """Fuse rankings by canonical entity ID using rank positions only.

    Each source is deduplicated before assigning 1-based ranks. Raw candidate
    scores are intentionally ignored. Equal RRF scores are ordered by ascending
    entity ID.
    """
    if rrf_k < 0:
        raise ValueError("rrf_k must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    kbrd_unique = _deduplicate_ranking(kbrd_candidates)
    ckg_unique = _deduplicate_ranking(ckg_candidates)
    kbrd_by_id = {candidate["id"]: candidate for candidate in kbrd_unique}
    ckg_by_id = {candidate["id"]: candidate for candidate in ckg_unique}
    kbrd_ranks = {
        candidate["id"]: rank for rank, candidate in enumerate(kbrd_unique, start=1)
    }
    ckg_ranks = {
        candidate["id"]: rank for rank, candidate in enumerate(ckg_unique, start=1)
    }

    fused: list[dict[str, Any]] = []
    for entity_id in kbrd_by_id.keys() | ckg_by_id.keys():
        kbrd_rank = kbrd_ranks.get(entity_id)
        ckg_rank = ckg_ranks.get(entity_id)
        kbrd_contribution = (
            KBRD_WEIGHT / (rrf_k + kbrd_rank) if kbrd_rank is not None else 0.0
        )
        ckg_contribution = (
            CKG_WEIGHT / (rrf_k + ckg_rank) if ckg_rank is not None else 0.0
        )
        kbrd_title = kbrd_by_id.get(entity_id, {}).get("title")
        ckg_title = ckg_by_id.get(entity_id, {}).get("title")
        fused.append(
            {
                "id": entity_id,
                "title": kbrd_title or ckg_title or "Unknown Title",
                "source": "RRF",
                "rrf_score": kbrd_contribution + ckg_contribution,
                "kbrd_rank": kbrd_rank,
                "ckg_rank": ckg_rank,
                "kbrd_contribution": kbrd_contribution,
                "ckg_contribution": ckg_contribution,
            }
        )

    return sorted(fused, key=lambda candidate: (-candidate["rrf_score"], candidate["id"]))[
        :top_k
    ]


class RankingMetricAccumulator:
    """Frozen normalized-title Recall@K and MRR for one ranked source."""

    def __init__(self) -> None:
        self.instances = 0
        self.recall = {1: 0, 10: 0, 50: 0}
        self.reciprocal_rank_sum = 0.0

    def add(
        self,
        candidates: list[dict[str, Any]],
        ground_truth_titles: list[str],
        hit_function: HitFunction,
        rank_function: RankFunction,
    ) -> None:
        self.instances += 1
        for cutoff in self.recall:
            self.recall[cutoff] += int(
                hit_function(candidates, ground_truth_titles, cutoff)
            )
        rank = rank_function(candidates, ground_truth_titles)
        self.reciprocal_rank_sum += 1.0 / rank if rank else 0.0

    def result(self) -> dict[str, Any]:
        instances = self.instances
        return {
            "instances": instances,
            "Recall@1": self.recall[1] / instances if instances else 0.0,
            "Recall@10": self.recall[10] / instances if instances else 0.0,
            "Recall@50": self.recall[50] / instances if instances else 0.0,
            "MRR": self.reciprocal_rank_sum / instances if instances else 0.0,
        }


def load_frozen_ckg(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[CKGRetriever, Path]:
    """Load and validate the selected conversation/conditional/support-2 cache."""
    cache_path = Path(cache_dir).resolve() / FROZEN_CKG_CACHE_NAME
    retriever = CKGRetriever.load(cache_path)
    expected = {
        "graph_type": FROZEN_CKG_GRAPH_TYPE,
        "weighting_method": FROZEN_CKG_WEIGHTING,
        "min_support": FROZEN_CKG_MIN_SUPPORT,
    }
    mismatches = {
        key: {"expected": value, "actual": retriever.metadata.get(key)}
        for key, value in expected.items()
        if retriever.metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen CKG cache metadata mismatch: {mismatches}")
    return retriever, cache_path


def _source_hits(
    candidates: list[dict[str, Any]],
    ground_truth: list[str],
    hit_function: HitFunction,
) -> dict[str, bool]:
    return {
        f"Recall@{cutoff}": hit_function(candidates, ground_truth, cutoff)
        for cutoff in (1, 10, 50)
    }


def evaluate_rrf_valid(
    valid_path: str | Path = DEFAULT_VALID_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    instance_output_path: str | Path = DEFAULT_INSTANCE_OUTPUT_PATH,
    parity_report_path: str | Path = DEFAULT_PARITY_OUTPUT_PATH,
    max_conversations: int | None = None,
) -> dict[str, Any]:
    """Evaluate fixed KBRD, CKG, and RRF rankings on VALID only."""
    source_path = validate_valid_source_path(valid_path)
    require_passing_parity_report(parity_report_path, source_path)
    retriever, cache_path = load_frozen_ckg(cache_dir)

    from my_crs import evaluate as frozen_evaluator
    import kbrd_adapter as frozen_kbrd_adapter
    from my_crs import movie_catalogue

    movie_catalogue.load_catalogue()
    metrics = {
        "KBRD": RankingMetricAccumulator(),
        "CKG": RankingMetricAccumulator(),
        "RRF": RankingMetricAccumulator(),
    }
    provenance: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    input_conversations_seen = 0
    evaluated_conversations = 0
    evaluation_instances = 0
    skipped_instances = 0
    skipped_conversations = 0
    oracle_union_hits_at_50 = 0

    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if max_conversations is not None and evaluated_conversations >= max_conversations:
                break
            input_conversations_seen += 1
            try:
                conversation = json.loads(line)
                respondent = conversation.get("respondentWorkerId", -1)
                conversation_has_instances = False
                for turn_index, message in enumerate(conversation.get("messages", [])):
                    if message.get("senderWorkerId", -1) != respondent:
                        continue
                    ground_truth = frozen_evaluator.get_recommended_movies_at_turn(
                        conversation, turn_index
                    )
                    if not ground_truth:
                        continue
                    try:
                        dialogue = frozen_evaluator.build_dialogue_up_to(
                            conversation, turn_index - 1
                        )
                        all_extracted_entity_ids = frozen_kbrd_adapter.prepare_input(dialogue)[0]
                        kbrd_candidates, _ = frozen_evaluator.get_kbrd_candidates(
                            dialogue,
                            top_k=TOP_K,
                            diagnostics=None,
                            use_fusion=False,
                            retrieval_mode="kbrd",
                        )
                        if not kbrd_candidates:
                            raise ValueError("Frozen KBRD returned no candidates")

                        ckg_views = retriever.retrieve_views(
                            all_extracted_entity_ids, top_k=TOP_K
                        )
                        ckg_candidates = attach_catalogue_titles(
                            ckg_views["budget_controlled"], movie_catalogue.get_title
                        )
                        rrf_candidates = reciprocal_rank_fusion(
                            kbrd_candidates, ckg_candidates, rrf_k=RRF_K, top_k=TOP_K
                        )

                        source_candidates = {
                            "KBRD": kbrd_candidates,
                            "CKG": ckg_candidates,
                            "RRF": rrf_candidates,
                        }
                        source_hits = {
                            name: _source_hits(
                                candidates, ground_truth, frozen_evaluator.is_hit
                            )
                            for name, candidates in source_candidates.items()
                        }
                        source_ranks = {
                            name: frozen_evaluator.get_rank(candidates, ground_truth)
                            for name, candidates in source_candidates.items()
                        }
                        for name, candidates in source_candidates.items():
                            metrics[name].add(
                                candidates,
                                ground_truth,
                                frozen_evaluator.is_hit,
                                frozen_evaluator.get_rank,
                            )

                        oracle_union_hit = (
                            source_hits["KBRD"]["Recall@50"]
                            or source_hits["CKG"]["Recall@50"]
                        )
                        oracle_union_hits_at_50 += int(oracle_union_hit)
                        diagnostics = ckg_views["diagnostics"]
                        provenance.append(
                            {
                                "line_number": line_number,
                                "conversation_id": conversation.get("conversationId"),
                                "turn_index": turn_index,
                                "ground_truth_titles": ground_truth,
                                "all_extracted_entity_ids": all_extracted_entity_ids,
                                "kbrd_candidate_ids": [
                                    int(candidate["id"])
                                    for candidate in kbrd_candidates[:TOP_K]
                                ],
                                "ckg_candidate_ids": [
                                    int(candidate["id"]) for candidate in ckg_candidates
                                ],
                                "ckg_candidate_origins": [
                                    {
                                        "id": int(candidate["id"]),
                                        "origin": candidate["source"],
                                    }
                                    for candidate in ckg_candidates
                                ],
                                "rrf_candidates": rrf_candidates,
                                "hits": source_hits,
                                "ranks": source_ranks,
                                "oracle_union_hit_at_50_upper_bound": oracle_union_hit,
                                "ckg_diagnostics": diagnostics,
                            }
                        )
                        evaluation_instances += 1
                        conversation_has_instances = True
                    except Exception as error:
                        skipped_instances += 1
                        failures.append(
                            {
                                "line_number": line_number,
                                "conversation_id": conversation.get("conversationId"),
                                "turn_index": turn_index,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
                if conversation_has_instances:
                    evaluated_conversations += 1
            except Exception as error:
                skipped_conversations += 1
                failures.append(
                    {
                        "line_number": line_number,
                        "conversation_id": None,
                        "turn_index": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    result = {
        "experiment": "offline_reciprocal_rank_fusion",
        "source_split": "VALID",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "input_conversations_seen": input_conversations_seen,
        "evaluated_conversations": evaluated_conversations,
        "evaluation_instances": evaluation_instances,
        "skipped_instances": skipped_instances,
        "skipped_conversations": skipped_conversations,
        "primary_scoring": "frozen normalized-title semantics",
        "extraction_configuration": extraction_configuration(frozen_evaluator._cfg),
        "kbrd_configuration": {
            "retrieval_mode": "kbrd",
            "top_k": TOP_K,
            "use_fusion": False,
            "llm_qwen_used": False,
        },
        "ckg_configuration": {
            "graph_type": FROZEN_CKG_GRAPH_TYPE,
            "weighting_method": FROZEN_CKG_WEIGHTING,
            "min_support": FROZEN_CKG_MIN_SUPPORT,
            "view": "budget_controlled",
            "top_k": TOP_K,
            "cache_path": str(cache_path),
            "graph_metadata": retriever.metadata,
        },
        "rrf_parameters": {
            "formula": "1/(k + rank_KBRD) + 1/(k + rank_CKG)",
            "k": RRF_K,
            "weights": {"KBRD": KBRD_WEIGHT, "CKG": CKG_WEIGHT},
            "ranks": "1-based",
            "absent_source_contribution": 0.0,
            "deduplication_key": "canonical KBRD entity/movie ID",
            "raw_scores_used": False,
            "tie_break": "entity_id_ascending",
            "final_candidate_budget": TOP_K,
        },
        "metrics": {name: accumulator.result() for name, accumulator in metrics.items()},
        "oracle_union_coverage_upper_bound": {
            "label": "UPPER BOUND ONLY - not a realizable fused ranking",
            "definition": (
                "A hit occurs when either KBRD Top-50 or CKG Top-50 contains "
                "a normalized-title match."
            ),
            "cutoff_per_source": TOP_K,
            "hits": oracle_union_hits_at_50,
            "instances": evaluation_instances,
            "coverage": (
                oracle_union_hits_at_50 / evaluation_instances
                if evaluation_instances
                else 0.0
            ),
        },
        "instance_provenance_path": str(Path(instance_output_path).resolve()),
        "failures": failures,
    }
    _atomic_jsonl(Path(instance_output_path).resolve(), provenance)
    _atomic_json(Path(output_path).resolve(), result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--instance-output-path", type=Path, default=DEFAULT_INSTANCE_OUTPUT_PATH
    )
    parser.add_argument(
        "--parity-report-path", type=Path, default=DEFAULT_PARITY_OUTPUT_PATH
    )
    parser.add_argument("--max-conversations", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = evaluate_rrf_valid(
        valid_path=args.valid_path,
        cache_dir=args.cache_dir,
        output_path=args.output_path,
        instance_output_path=args.instance_output_path,
        parity_report_path=args.parity_report_path,
        max_conversations=args.max_conversations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
