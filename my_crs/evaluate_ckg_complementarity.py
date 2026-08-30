"""VALID-only frozen-KBRD parity and isolated CKG complementarity evaluation.

``--kbrd-parity-only`` branches before graph loading and reproduces the frozen
evaluator's normalized-title, recommendation-only semantics. The CKG path is
implemented for a later run but is not required by parity mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


MY_CRS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MY_CRS_DIR.parent
KBRD_DATA_DIR = (
    PROJECT_ROOT / "baseline_repo" / "KBRD_project" / "KBRD" / "data" / "redial"
)
DEFAULT_VALID_PATH = KBRD_DATA_DIR / "valid_data.jsonl"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "experiments" / "ckg_cache"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ckg_complementarity_valid.json"
DEFAULT_INSTANCE_OUTPUT_PATH = (
    PROJECT_ROOT / "experiments" / "ckg_complementarity_valid.instances.jsonl"
)
DEFAULT_PARITY_OUTPUT_PATH = PROJECT_ROOT / "experiments" / "ckg_kbrd_parity_valid.json"
EXPECTED_PARITY = {
    "evaluated_conversations": 797,
    "evaluation_instances": 2588,
    "Recall@1": 0.0232,
    "Recall@10": 0.1700,
    "Recall@50": 0.3841,
    "MRR": 0.0687,
}
EXPECTED_FROZEN_CONFIGURATION = {
    "resolver_version": "v3",
    "use_legacy_non_movie_entities": True,
    "seed_selection": "all",
    "retrieval_mode": "kbrd",
    "top_k": 50,
    "skip_reranker": True,
    "recommendation_only": True,
    "split": "valid",
    "fusion": False,
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_crs.ckg_retriever import (  # noqa: E402
    CKGRetriever,
    GRAPH_TYPES,
    WEIGHTING_METHODS,
)


HitFunction = Callable[[list[dict[str, Any]], list[str], int], bool]
RankFunction = Callable[[list[dict[str, Any]], list[str]], int]


def validate_valid_source_path(path: os.PathLike[str] | str) -> Path:
    valid_path = Path(path).resolve()
    if valid_path.name != "valid_data.jsonl":
        raise ValueError(f"Complementarity evaluation is VALID-only, got {valid_path.name!r}")
    if not valid_path.is_file():
        raise FileNotFoundError(valid_path)
    return valid_path


def strict_id_is_hit(candidate_ids: Sequence[int], target_ids: set[int], k: int) -> bool:
    """Secondary diagnostic only; frozen primary metrics use normalized titles."""
    return bool(target_ids.intersection(candidate_ids[:k]))


def attach_catalogue_titles(
    candidates: Sequence[Mapping[str, Any]], title_lookup: Callable[[int], str | None]
) -> list[dict[str, Any]]:
    titled: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        item["id"] = int(item["id"])
        item["title"] = title_lookup(item["id"]) or "Unknown Title"
        titled.append(item)
    return titled


def pre_fill_bucket(candidate_count: int) -> str:
    if candidate_count == 0:
        return "0"
    if candidate_count <= 9:
        return "1-9"
    if candidate_count <= 49:
        return "10-49"
    return "50"


def extraction_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    extraction = config["extraction"]
    return {
        "resolver_version": extraction.get("resolver_version"),
        "use_legacy_non_movie_entities": extraction.get("use_legacy_non_movie_entities"),
        "use_aux_dbpedia_uri_matching": extraction.get("use_aux_dbpedia_uri_matching", True),
        "use_aux_genre_mapping": extraction.get("use_aux_genre_mapping", True),
        "use_aux_person_matching": extraction.get("use_aux_person_matching", True),
        "seed_selection": extraction.get("seed_selection"),
    }


class MetricAccumulator:
    """Primary normalized-title metrics plus KBRD complementarity counts."""

    def __init__(self) -> None:
        self.instances = 0
        self.recall = {1: 0, 10: 0, 50: 0}
        self.reciprocal_rank_sum = 0.0
        self.both_hit = 0
        self.kbrd_only_hit = 0
        self.ckg_only_hit = 0
        self.neither = 0
        self.union_hits = 0
        self.kbrd_misses = 0
        self.overlap_sum = 0.0
        self.jaccard_sum = 0.0

    def add(
        self,
        ckg_candidates: list[dict[str, Any]],
        kbrd_candidates: list[dict[str, Any]],
        ground_truth_titles: list[str],
        hit_function: HitFunction,
        rank_function: RankFunction,
    ) -> None:
        self.instances += 1
        for k in self.recall:
            self.recall[k] += int(hit_function(ckg_candidates, ground_truth_titles, k))
        rank = rank_function(ckg_candidates, ground_truth_titles)
        self.reciprocal_rank_sum += 1.0 / rank if rank else 0.0

        ckg_hit = hit_function(ckg_candidates, ground_truth_titles, 50)
        kbrd_hit = hit_function(kbrd_candidates, ground_truth_titles, 50)
        if ckg_hit and kbrd_hit:
            self.both_hit += 1
        elif kbrd_hit:
            self.kbrd_only_hit += 1
        elif ckg_hit:
            self.ckg_only_hit += 1
        else:
            self.neither += 1
        self.union_hits += int(ckg_hit or kbrd_hit)
        self.kbrd_misses += int(not kbrd_hit)

        ckg_ids = {int(candidate["id"]) for candidate in ckg_candidates[:50]}
        kbrd_ids = {int(candidate["id"]) for candidate in kbrd_candidates[:50]}
        intersection = ckg_ids & kbrd_ids
        union = ckg_ids | kbrd_ids
        self.overlap_sum += len(intersection)
        self.jaccard_sum += len(intersection) / len(union) if union else 0.0

    def result(self) -> dict[str, Any]:
        n = self.instances
        partition_total = self.both_hit + self.kbrd_only_hit + self.ckg_only_hit + self.neither
        return {
            "instances": n,
            "standalone_ckg": {
                "Recall@1": self.recall[1] / n if n else 0.0,
                "Recall@10": self.recall[10] / n if n else 0.0,
                "Recall@50": self.recall[50] / n if n else 0.0,
                "MRR": self.reciprocal_rank_sum / n if n else 0.0,
            },
            "against_frozen_kbrd_at_50": {
                "both_hit": self.both_hit,
                "KBRD_only_hit": self.kbrd_only_hit,
                "CKG_only_hit": self.ckg_only_hit,
                "neither": self.neither,
                "partition_total": partition_total,
                "partition_invariant_passed": partition_total == n,
                "union_coverage": self.union_hits / n if n else 0.0,
                "average_candidate_overlap": self.overlap_sum / n if n else 0.0,
                "average_jaccard_overlap": self.jaccard_sum / n if n else 0.0,
                "CKG_recovery_among_KBRD_failures": (
                    self.ckg_only_hit / self.kbrd_misses if self.kbrd_misses else 0.0
                ),
                "total_KBRD_misses": self.kbrd_misses,
            },
        }


def count_frozen_evaluable_turns(
    valid_path: os.PathLike[str] | str,
    get_recommended_movies_at_turn: Callable[[dict[str, Any], int], list[str]],
) -> dict[str, int]:
    """Count frozen-evaluator input conversations, evaluated conversations, and turns."""
    source_path = validate_valid_source_path(valid_path)
    input_conversations_seen = 0
    evaluated_conversations = 0
    evaluation_instances = 0
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            input_conversations_seen += 1
            conversation = json.loads(line)
            respondent = conversation.get("respondentWorkerId", -1)
            conversation_has_instances = False
            for turn_index, message in enumerate(conversation.get("messages", [])):
                if message.get("senderWorkerId", -1) != respondent:
                    continue
                if get_recommended_movies_at_turn(conversation, turn_index):
                    evaluation_instances += 1
                    conversation_has_instances = True
            if conversation_has_instances:
                evaluated_conversations += 1
    return {
        "input_conversations_seen": input_conversations_seen,
        "evaluated_conversations": evaluated_conversations,
        "evaluation_instances": evaluation_instances,
    }


def parity_verification(result: Mapping[str, Any]) -> dict[str, Any]:
    """Compare independently computed full-VALID output with frozen expectations."""
    checks: dict[str, Any] = {}
    configuration = result.get("configuration", {})
    for name, expected in EXPECTED_FROZEN_CONFIGURATION.items():
        observed = configuration.get(name)
        checks[f"configuration.{name}"] = {
            "observed": observed,
            "expected": expected,
            "passed": observed == expected,
        }
    for name in ("evaluated_conversations", "evaluation_instances"):
        observed = int(result[name])
        expected = int(EXPECTED_PARITY[name])
        checks[name] = {"observed": observed, "expected": expected, "passed": observed == expected}
    metrics = result["recommendation"]
    for name in ("Recall@1", "Recall@10", "Recall@50", "MRR"):
        observed = float(metrics[name])
        expected = float(EXPECTED_PARITY[name])
        checks[name] = {
            "observed": observed,
            "observed_rounded_4dp": round(observed, 4),
            "expected_rounded_4dp": expected,
            "difference": observed - expected,
            "passed": round(observed, 4) == expected,
        }
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def require_passing_parity_report(
    parity_report_path: os.PathLike[str] | str, valid_path: Path
) -> dict[str, Any]:
    """Block CKG loading unless verified KBRD parity passed on the same VALID file."""
    report_path = Path(parity_report_path).resolve()
    if not report_path.is_file():
        raise RuntimeError(
            "CKG evaluation is gated on a passing full-VALID KBRD parity report. "
            f"Run --kbrd-parity-only --verify-parity first; missing {report_path}"
        )
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    verification = parity_verification(report)
    if not verification["passed"]:
        failed = {
            name: check
            for name, check in verification["checks"].items()
            if not check["passed"]
        }
        raise RuntimeError(f"KBRD parity failed; CKG evaluation blocked: {failed}")
    expected_sha256 = _sha256(valid_path)
    if report.get("source_sha256") != expected_sha256:
        raise RuntimeError(
            "KBRD parity report was produced from a different VALID file; "
            "CKG evaluation blocked"
        )
    return report


@contextmanager
def isolate_adapter_tmdb_cache(adapter_module: Any):
    """Redirect optional enrichment writes without changing recommendation ranking."""
    if adapter_module is None or not hasattr(adapter_module, "_TMDB_CACHE_PATH"):
        yield False
        return
    original_path = adapter_module._TMDB_CACHE_PATH
    with tempfile.TemporaryDirectory(prefix="ckg-parity-tmdb-") as directory:
        adapter_module._TMDB_CACHE_PATH = str(Path(directory) / "tmdb_cache.json")
        try:
            yield True
        finally:
            adapter_module._TMDB_CACHE_PATH = original_path


def run_kbrd_parity(
    valid_path: os.PathLike[str] | str = DEFAULT_VALID_PATH,
    output_path: os.PathLike[str] | str = DEFAULT_PARITY_OUTPUT_PATH,
    verify: bool = False,
    max_conversations: int | None = None,
) -> dict[str, Any]:
    """Run frozen pure-KBRD@50 only; no CKG cache is loaded or retrieved."""
    if verify and max_conversations is not None:
        raise ValueError("Parity verification requires the full VALID split")
    source_path = validate_valid_source_path(valid_path)

    from my_crs import evaluate as frozen_evaluator

    frozen_adapter = sys.modules.get(frozen_evaluator.get_kbrd_candidates.__module__)

    frozen_config = {
        **extraction_configuration(frozen_evaluator._cfg),
        "retrieval_mode": "kbrd",
        "top_k": 50,
        "skip_reranker": True,
        "recommendation_only": True,
        "split": "valid",
        "fusion": False,
    }
    input_conversations_seen = 0
    evaluated_conversations = 0
    evaluation_instances = 0
    skipped_instances = 0
    skipped_conversations = 0
    hits = {1: [], 10: [], 50: []}
    reciprocal_ranks: list[float] = []
    failures: list[dict[str, Any]] = []

    with isolate_adapter_tmdb_cache(frozen_adapter) as tmdb_cache_isolated, source_path.open(
        "r", encoding="utf-8"
    ) as handle:
        for line_number, line in enumerate(handle, 1):
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
                        candidates, _ = frozen_evaluator.get_kbrd_candidates(
                            dialogue,
                            top_k=50,
                            diagnostics=None,
                            use_fusion=False,
                            retrieval_mode="kbrd",
                        )
                        if not candidates:
                            raise ValueError("Frozen KBRD returned no candidates")
                        for k in hits:
                            hits[k].append(
                                frozen_evaluator.is_hit(candidates, ground_truth, k)
                            )
                        rank = frozen_evaluator.get_rank(candidates, ground_truth)
                        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
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

    result: dict[str, Any] = {
        "mode": "KBRD_PARITY_ONLY",
        "source_split": "VALID",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "input_conversations_seen": input_conversations_seen,
        "evaluated_conversations": evaluated_conversations,
        "evaluation_instances": evaluation_instances,
        "skipped_instances": skipped_instances,
        "skipped_conversations": skipped_conversations,
        "configuration": frozen_config,
        "recommendation": {
            "Recall@1": sum(hits[1]) / len(hits[1]) if hits[1] else 0.0,
            "Recall@10": sum(hits[10]) / len(hits[10]) if hits[10] else 0.0,
            "Recall@50": sum(hits[50]) / len(hits[50]) if hits[50] else 0.0,
            "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        },
        "failures": failures,
        "ckg_loaded": False,
        "ckg_retrieved": False,
        "tmdb_cache_isolated": tmdb_cache_isolated,
    }
    if verify:
        result["parity_verification"] = parity_verification(result)
    _atomic_json(Path(output_path).resolve(), result)
    return result


def evaluate_valid(
    valid_path: os.PathLike[str] | str = DEFAULT_VALID_PATH,
    cache_dir: os.PathLike[str] | str = DEFAULT_CACHE_DIR,
    output_path: os.PathLike[str] | str = DEFAULT_OUTPUT_PATH,
    instance_output_path: os.PathLike[str] | str = DEFAULT_INSTANCE_OUTPUT_PATH,
    parity_report_path: os.PathLike[str] | str = DEFAULT_PARITY_OUTPUT_PATH,
    min_support: int = 2,
    max_conversations: int | None = None,
) -> dict[str, Any]:
    """Run later CKG evaluation with frozen normalized-title primary scoring."""
    source_path = validate_valid_source_path(valid_path)
    require_passing_parity_report(parity_report_path, source_path)
    cache_path = Path(cache_dir).resolve()
    retrievers: dict[str, CKGRetriever] = {}
    for graph_type in GRAPH_TYPES:
        for weighting_method in WEIGHTING_METHODS:
            key = f"{graph_type}_{weighting_method}"
            retrievers[key] = CKGRetriever.load(
                cache_path / f"{key}_support{min_support}.pkl"
            )

    from my_crs import evaluate as frozen_evaluator
    import kbrd_adapter as frozen_kbrd_adapter
    from my_crs import movie_catalogue

    movie_catalogue.load_catalogue()
    accumulators = {
        key: {
            view: {
                "all_instances": MetricAccumulator(),
                "with_movie_seed": MetricAccumulator(),
                "zero_movie_seed": MetricAccumulator(),
            }
            for view in ("graph_only", "budget_controlled")
        }
        for key in retrievers
    }
    contribution = {
        key: {
            "graph_only_hit_count": 0,
            "popularity_fill_only_hit_count": 0,
            "zero_seed_fallback_only_hit_count": 0,
            "pre_fill_candidate_count_sum": 0,
            "pre_fill_candidate_count_distribution": Counter(
                {"0": 0, "1-9": 0, "10-49": 0, "50": 0}
            ),
            "fallback_used_count": 0,
            "fill_used_count": 0,
        }
        for key in retrievers
    }
    provenance: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    input_conversations_seen = 0
    evaluated_conversations = 0
    evaluation_instances = 0
    skipped_instances = 0
    skipped_conversations = 0

    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
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
                        prepared = frozen_kbrd_adapter.prepare_input(dialogue)
                        all_extracted_entity_ids = prepared[0]
                        kbrd_candidates, _ = frozen_evaluator.get_kbrd_candidates(
                            dialogue,
                            top_k=50,
                            diagnostics=None,
                            use_fusion=False,
                            retrieval_mode="kbrd",
                        )
                        if not kbrd_candidates:
                            raise ValueError("Frozen KBRD returned no candidates")
                        kbrd_hit = frozen_evaluator.is_hit(kbrd_candidates, ground_truth, 50)

                        prepared_configs: dict[str, Any] = {}
                        for key, retriever in retrievers.items():
                            views = retriever.retrieve_views(all_extracted_entity_ids, top_k=50)
                            graph_candidates = attach_catalogue_titles(
                                views["graph_only"], movie_catalogue.get_title
                            )
                            budget_candidates = attach_catalogue_titles(
                                views["budget_controlled"], movie_catalogue.get_title
                            )
                            prepared_configs[key] = {
                                "graph_candidates": graph_candidates,
                                "budget_candidates": budget_candidates,
                                "diagnostics": views["diagnostics"],
                                "graph_hit": frozen_evaluator.is_hit(
                                    graph_candidates, ground_truth, 50
                                ),
                                "budget_hit": frozen_evaluator.is_hit(
                                    budget_candidates, ground_truth, 50
                                ),
                            }

                        for key, values in prepared_configs.items():
                            diagnostics = values["diagnostics"]
                            group = (
                                "with_movie_seed"
                                if diagnostics["num_movie_seeds"] >= 1
                                else "zero_movie_seed"
                            )
                            for view, candidates in (
                                ("graph_only", values["graph_candidates"]),
                                ("budget_controlled", values["budget_candidates"]),
                            ):
                                for subset in ("all_instances", group):
                                    accumulators[key][view][subset].add(
                                        candidates,
                                        kbrd_candidates,
                                        ground_truth,
                                        frozen_evaluator.is_hit,
                                        frozen_evaluator.get_rank,
                                    )

                            stats = contribution[key]
                            stats["graph_only_hit_count"] += int(values["graph_hit"])
                            pre_fill_count = diagnostics["pre_fill_candidate_count"]
                            stats["pre_fill_candidate_count_sum"] += pre_fill_count
                            stats["pre_fill_candidate_count_distribution"][
                                pre_fill_bucket(pre_fill_count)
                            ] += 1
                            stats["fallback_used_count"] += int(diagnostics["fallback_used"])
                            stats["fill_used_count"] += int(diagnostics["fill_used"])

                            if values["budget_hit"] and not values["graph_hit"]:
                                matching_origins = {
                                    candidate["source"]
                                    for candidate in values["budget_candidates"]
                                    if any(
                                        frozen_evaluator.strict_title_match(
                                            candidate.get("title", ""), gold
                                        )
                                        for gold in ground_truth
                                    )
                                }
                                if "CKG_POPULARITY_FILL" in matching_origins:
                                    stats["popularity_fill_only_hit_count"] += 1
                                if "CKG_ZERO_SEED_FALLBACK" in matching_origins:
                                    stats["zero_seed_fallback_only_hit_count"] += 1

                            graph_type, weighting_method = key.split("_", 1)
                            provenance.append(
                                {
                                    "conversation_id": conversation.get("conversationId"),
                                    "turn_index": turn_index,
                                    "num_movie_seeds": diagnostics["num_movie_seeds"],
                                    "movie_seed_ids": diagnostics["movie_seed_ids"],
                                    "graph_type": graph_type,
                                    "weighting_method": weighting_method,
                                    "pre_fill_candidate_count": pre_fill_count,
                                    "post_fill_candidate_count": diagnostics[
                                        "post_fill_candidate_count"
                                    ],
                                    "fallback_used": diagnostics["fallback_used"],
                                    "fill_used": diagnostics["fill_used"],
                                    "graph_candidate_ids": diagnostics["graph_candidate_ids"],
                                    "candidate_origins": [
                                        {"id": candidate["id"], "origin": candidate["source"]}
                                        for candidate in values["budget_candidates"]
                                    ],
                                    "graph_only_hit": values["graph_hit"],
                                    "filled_list_hit": values["budget_hit"],
                                    "KBRD_hit": kbrd_hit,
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

    configurations: dict[str, Any] = {}
    for key, views in accumulators.items():
        stats = contribution[key]
        configurations[key] = {
            "graph_metadata": retrievers[key].metadata,
            "views": {
                view: {
                    "subsets": {
                        name: accumulator.result()
                        for name, accumulator in groups.items()
                    }
                }
                for view, groups in views.items()
            },
            "graph_contribution": {
                "graph_only_hit_count": stats["graph_only_hit_count"],
                "popularity_fill_only_hit_count": stats["popularity_fill_only_hit_count"],
                "zero_seed_fallback_only_hit_count": stats[
                    "zero_seed_fallback_only_hit_count"
                ],
                "average_pre_fill_candidate_count": (
                    stats["pre_fill_candidate_count_sum"] / evaluation_instances
                    if evaluation_instances
                    else 0.0
                ),
                "pre_fill_candidate_count_distribution": dict(
                    stats["pre_fill_candidate_count_distribution"]
                ),
                "fallback_used_count": stats["fallback_used_count"],
                "fill_used_count": stats["fill_used_count"],
            },
        }

    result = {
        "source_split": "VALID",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "input_conversations_seen": input_conversations_seen,
        "evaluated_conversations": evaluated_conversations,
        "evaluation_instances": evaluation_instances,
        "skipped_instances": skipped_instances,
        "skipped_conversations": skipped_conversations,
        "primary_scoring": "frozen normalized-title semantics",
        "kbrd_baseline": "pure KBRD@50 (retrieval_mode=kbrd, fusion disabled)",
        "fusion_implemented": False,
        "extraction_configuration": extraction_configuration(frozen_evaluator._cfg),
        "configurations": configurations,
        "instance_provenance_path": str(Path(instance_output_path).resolve()),
        "failures": failures,
    }
    _atomic_jsonl(Path(instance_output_path).resolve(), provenance)
    _atomic_json(Path(output_path).resolve(), result)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valid-path", type=Path, default=DEFAULT_VALID_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--instance-output-path", type=Path, default=DEFAULT_INSTANCE_OUTPUT_PATH
    )
    parser.add_argument("--parity-output-path", type=Path, default=DEFAULT_PARITY_OUTPUT_PATH)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-conversations", type=int)
    parser.add_argument(
        "--kbrd-parity-only",
        action="store_true",
        help="Run frozen KBRD@50 only; do not load or retrieve any CKG graph",
    )
    parser.add_argument(
        "--verify-parity",
        action="store_true",
        help="Compare full-VALID parity output with the accepted frozen result",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verify_parity and not args.kbrd_parity_only:
        raise ValueError("--verify-parity requires --kbrd-parity-only")
    if args.kbrd_parity_only:
        result = run_kbrd_parity(
            valid_path=args.valid_path,
            output_path=args.parity_output_path,
            verify=args.verify_parity,
            max_conversations=args.max_conversations,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.verify_parity and not result["parity_verification"]["passed"]:
            return 1
        return 0


    result = evaluate_valid(
        valid_path=args.valid_path,
        cache_dir=args.cache_dir,
        output_path=args.output_path,
        instance_output_path=args.instance_output_path,
        parity_report_path=args.parity_output_path,
        min_support=args.min_support,
        max_conversations=args.max_conversations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
