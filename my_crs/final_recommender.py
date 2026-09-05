"""Reusable final CRS ranking pipeline, intentionally not connected to the web app.

The pipeline is fixed to pure KBRD + conversational CKG + equal-weight RRF
(k=60, Top 50), followed by the selected Stage2-v2 runtime.  Missing artifacts
and fallback outputs are fatal: this module never substitutes static movies.
"""

from __future__ import annotations

import importlib
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from my_crs.stage2_v2_runtime import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DEVICE,
    FROZEN_TOP_K,
    Stage2V2ReadinessError,
    Stage2V2Runtime,
    get_stage2_v2_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KBRD_ROOT = PROJECT_ROOT / "baseline_repo" / "KBRD_project" / "KBRD"
KBRD_DATA_DIR = KBRD_ROOT / "data" / "redial"
KBRD_CHECKPOINT_PATH = KBRD_ROOT / "saved" / "kbrd_model_retrained"
DEFAULT_CKG_CACHE_DIR = PROJECT_ROOT / "experiments" / "ckg_cache"
FROZEN_CKG_CACHE_NAME = "conversation_conditional_support2.pkl"
FROZEN_RRF_K = 60


class FinalRecommenderError(RuntimeError):
    """Base error for the final reusable recommendation pipeline."""


class FinalRecommenderReadinessError(FinalRecommenderError):
    """Raised when required artifacts, dependencies, or hardware are unavailable."""


class FinalRecommendationUnavailableError(FinalRecommenderError):
    """Raised when the frozen pipeline cannot form an inference event."""


class FinalRecommenderInvariantError(FinalRecommenderError):
    """Raised when a frozen candidate-set or provenance invariant is violated."""


@dataclass(frozen=True)
class FinalRecommenderState:
    ckg_cache_path: Path
    ckg_metadata: dict[str, Any]
    stage2: dict[str, Any]


def _required_retrieval_artifacts(ckg_cache_dir: Path) -> tuple[Path, ...]:
    return (
        KBRD_DATA_DIR / "entity2entityId.pkl",
        KBRD_DATA_DIR / "movie_ids.pkl",
        KBRD_DATA_DIR / "subkg.pkl",
        KBRD_DATA_DIR / "movies_with_mentions.csv",
        KBRD_DATA_DIR / "train_data.jsonl",
        KBRD_CHECKPOINT_PATH,
        ckg_cache_dir / FROZEN_CKG_CACHE_NAME,
    )


def _require_retrieval_artifacts(ckg_cache_dir: Path) -> None:
    missing = [path for path in _required_retrieval_artifacts(ckg_cache_dir) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FinalRecommenderReadinessError(
            "Final CRS retrieval artifacts are missing:\n" + formatted
        )


def _load_default_retrieval_bindings() -> Any:
    """Load the exact frozen retrieval helpers and legacy KBRD adapter lazily."""

    try:
        from my_crs.evaluate_ckg_complementarity import attach_catalogue_titles
        from my_crs.evaluate_rrf_fusion import (
            CKG_WEIGHT,
            KBRD_WEIGHT,
            RRF_K,
            TOP_K,
            load_frozen_ckg,
            reciprocal_rank_fusion,
        )
        from my_crs import movie_catalogue

        # The existing KBRD adapter uses historical top-level sibling imports.
        # Match the frozen dataset builders without changing that adapter here.
        my_crs_path = str(PROJECT_ROOT / "my_crs")
        if my_crs_path not in sys.path:
            sys.path.insert(0, my_crs_path)
        kbrd_adapter = importlib.import_module("kbrd_adapter")
        movie_catalogue.load_catalogue()
    except Exception as error:
        raise FinalRecommenderReadinessError(
            "Unable to load the frozen KBRD/CKG retrieval stack: " f"{error}"
        ) from error

    if TOP_K != FROZEN_TOP_K or RRF_K != FROZEN_RRF_K:
        raise FinalRecommenderReadinessError(
            "Frozen retrieval modules disagree with RRF k=60 or Top 50"
        )
    if float(KBRD_WEIGHT) != 1.0 or float(CKG_WEIGHT) != 1.0:
        raise FinalRecommenderReadinessError(
            "Frozen retrieval modules do not use equal KBRD/CKG RRF weights"
        )
    return SimpleNamespace(
        attach_catalogue_titles=attach_catalogue_titles,
        ckg_weight=float(CKG_WEIGHT),
        get_kbrd_candidates=kbrd_adapter.get_kbrd_candidates,
        kbrd_weight=float(KBRD_WEIGHT),
        load_frozen_ckg=load_frozen_ckg,
        prepare_input=kbrd_adapter.prepare_input,
        reciprocal_rank_fusion=reciprocal_rank_fusion,
        rrf_k=RRF_K,
        title_lookup=movie_catalogue.get_title,
        top_k=TOP_K,
    )


def _validate_unique_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    source_name: str,
    required_source: str | None = None,
) -> list[dict[str, Any]]:
    if len(candidates) != FROZEN_TOP_K:
        raise FinalRecommenderInvariantError(
            f"{source_name} returned {len(candidates)} candidates; expected 50"
        )
    copied: list[dict[str, Any]] = []
    ids: list[int] = []
    for position, candidate in enumerate(candidates, 1):
        if not isinstance(candidate, Mapping):
            raise FinalRecommenderInvariantError(
                f"{source_name} candidate {position} is not an object"
            )
        item = dict(candidate)
        if type(item.get("id")) is not int:
            raise FinalRecommenderInvariantError(
                f"{source_name} candidate {position} has a non-integer ID"
            )
        if required_source is not None and item.get("source") != required_source:
            raise FinalRecommenderInvariantError(
                f"{source_name} candidate {position} has invalid provenance "
                f"{item.get('source')!r}"
            )
        ids.append(int(item["id"]))
        copied.append(item)
    if len(set(ids)) != FROZEN_TOP_K:
        raise FinalRecommenderInvariantError(
            f"{source_name} must contain 50 unique canonical IDs"
        )
    return copied


class FinalRecommender:
    """Orchestrate the frozen Stage 1 and Stage 2 ranking pipeline."""

    def __init__(
        self,
        *,
        ckg_cache_dir: str | Path = DEFAULT_CKG_CACHE_DIR,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device: str = DEFAULT_DEVICE,
        stage2_runtime: Stage2V2Runtime | Any | None = None,
        _bindings: Any | None = None,
    ) -> None:
        self.ckg_cache_dir = Path(ckg_cache_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = str(device)
        self._bindings = _bindings
        self._stage2 = stage2_runtime or get_stage2_v2_runtime(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        )
        self._retriever: Any | None = None
        self._state: FinalRecommenderState | None = None
        self._load_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._state is not None

    def ensure_ready(self) -> FinalRecommenderState:
        if self._state is not None:
            return self._state
        with self._load_lock:
            if self._state is not None:
                return self._state
            if self._bindings is None:
                _require_retrieval_artifacts(self.ckg_cache_dir)
                bindings = _load_default_retrieval_bindings()
            else:
                bindings = self._bindings
            if int(bindings.top_k) != FROZEN_TOP_K or int(bindings.rrf_k) != FROZEN_RRF_K:
                raise FinalRecommenderReadinessError(
                    "Retrieval bindings must enforce RRF k=60 and Top 50"
                )
            if (
                float(bindings.kbrd_weight) != 1.0
                or float(bindings.ckg_weight) != 1.0
            ):
                raise FinalRecommenderReadinessError(
                    "Retrieval bindings must use equal KBRD/CKG weights"
                )
            try:
                retriever, cache_path = bindings.load_frozen_ckg(
                    self.ckg_cache_dir
                )
                stage2_state = self._stage2.ensure_ready()
            except (FinalRecommenderError, Stage2V2ReadinessError):
                raise
            except Exception as error:
                raise FinalRecommenderReadinessError(
                    f"Unable to initialize final recommender: {error}"
                ) from error
            stage2_diagnostics = (
                stage2_state.diagnostics()
                if hasattr(stage2_state, "diagnostics")
                else dict(stage2_state)
            )
            self._bindings = bindings
            self._retriever = retriever
            self._state = FinalRecommenderState(
                ckg_cache_path=Path(cache_path).resolve(),
                ckg_metadata=dict(retriever.metadata),
                stage2=stage2_diagnostics,
            )
            return self._state

    def recommend(self, history: str) -> dict[str, Any]:
        """Return the final Stage2-ranked version of one frozen RRF Top-50."""

        if not isinstance(history, str):
            raise FinalRecommenderInvariantError("Conversation history must be a string")
        state = self.ensure_ready()
        assert self._retriever is not None
        bindings = self._bindings

        extracted = bindings.prepare_input(history)
        if not isinstance(extracted, tuple) or not extracted:
            raise FinalRecommenderInvariantError(
                "Entity resolver did not return its frozen tuple result"
            )
        all_extracted_entity_ids = [int(entity_id) for entity_id in extracted[0]]

        kbrd_diagnostics: dict[str, Any] = {}
        kbrd_candidates, detected_decades = bindings.get_kbrd_candidates(
            history,
            top_k=FROZEN_TOP_K,
            diagnostics=kbrd_diagnostics,
            use_fusion=False,
            retrieval_mode="kbrd",
        )
        fallback_reason = kbrd_diagnostics.get("fallback_reason")
        if fallback_reason == "model_or_resources_unavailable":
            raise FinalRecommenderReadinessError(
                "Pure KBRD model or resources are unavailable; static fallback rejected"
            )
        if fallback_reason == "no_inference_seeds":
            raise FinalRecommendationUnavailableError(
                "Pure KBRD has no inference seeds for this conversation; static fallback rejected"
            )
        if fallback_reason is not None:
            raise FinalRecommenderInvariantError(
                f"Pure KBRD fallback rejected: {fallback_reason}"
            )
        if (
            kbrd_diagnostics.get("qwen_fallback_executed") is True
            or kbrd_diagnostics.get("qwen_seed_entity_ids") not in (None, [])
            or kbrd_diagnostics.get("num_fused_seed_candidates") not in (None, 0)
            or kbrd_diagnostics.get("num_fused_qwen_candidates") not in (None, 0)
        ):
            raise FinalRecommenderInvariantError(
                "Pure KBRD retrieval unexpectedly used Qwen or candidate fusion"
            )
        kbrd = _validate_unique_candidates(
            kbrd_candidates,
            source_name="KBRD",
            required_source="KBRD_NEURAL",
        )
        if not {int(item["id"]) for item in kbrd} <= set(
            self._retriever.movie_ids
        ):
            raise FinalRecommenderInvariantError(
                "KBRD returned IDs outside the frozen CKG movie universe"
            )

        ckg_views = self._retriever.retrieve_views(
            all_extracted_entity_ids,
            top_k=FROZEN_TOP_K,
        )
        if not isinstance(ckg_views, Mapping):
            raise FinalRecommenderInvariantError("CKG did not return frozen views")
        ckg_candidates = bindings.attach_catalogue_titles(
            ckg_views.get("budget_controlled", []),
            bindings.title_lookup,
        )
        ckg = _validate_unique_candidates(
            ckg_candidates,
            source_name="CKG budget-controlled view",
        )

        rrf_candidates = bindings.reciprocal_rank_fusion(
            kbrd,
            ckg,
            rrf_k=FROZEN_RRF_K,
            top_k=FROZEN_TOP_K,
        )
        rrf = _validate_unique_candidates(
            rrf_candidates,
            source_name="RRF",
            required_source="RRF",
        )
        rrf_top50 = [
            {**candidate, "rank": rank}
            for rank, candidate in enumerate(rrf, 1)
        ]

        stage2_result = self._stage2.rank(history, rrf_top50)
        if not isinstance(stage2_result, Mapping):
            raise FinalRecommenderInvariantError(
                "Stage2-v2 runtime did not return a result object"
            )
        ranked_candidates = _validate_unique_candidates(
            stage2_result.get("ranked_candidates", []),
            source_name="Stage2-v2",
        )
        rrf_ids = {int(candidate["id"]) for candidate in rrf_top50}
        stage2_ids = {int(candidate["id"]) for candidate in ranked_candidates}
        if stage2_ids != rrf_ids:
            raise FinalRecommenderInvariantError(
                "Stage2-v2 changed the frozen Stage1 RRF Top-50 membership"
            )

        return {
            "selected_candidate": dict(ranked_candidates[0]),
            "ranked_candidates": ranked_candidates,
            "stage1_rrf_top50": [dict(candidate) for candidate in rrf_top50],
            "diagnostics": {
                "configuration": {
                    "ckg_weight": float(bindings.ckg_weight),
                    "kbrd_retrieval_mode": "kbrd",
                    "kbrd_weight": float(bindings.kbrd_weight),
                    "rrf_k": FROZEN_RRF_K,
                    "stage1_top_k": FROZEN_TOP_K,
                    "stage2_candidate_membership": "fixed_stage1_rrf_top50",
                },
                "extraction": {
                    "all_extracted_entity_ids": all_extracted_entity_ids,
                    "detected_decades": list(detected_decades),
                },
                "kbrd": dict(kbrd_diagnostics),
                "ckg": dict(ckg_views.get("diagnostics", {})),
                "rrf": {
                    "candidate_ids": [int(item["id"]) for item in rrf_top50],
                    "candidate_membership_count": len(rrf_top50),
                },
                "stage2": dict(stage2_result.get("diagnostics", {})),
                "readiness": {
                    "ckg_cache_path": str(state.ckg_cache_path),
                    "ckg_metadata": dict(state.ckg_metadata),
                    "stage2": dict(state.stage2),
                },
            },
        }


_singleton_lock = threading.Lock()
_singleton: FinalRecommender | None = None


def get_final_recommender(
    *,
    ckg_cache_dir: str | Path = DEFAULT_CKG_CACHE_DIR,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: str = DEFAULT_DEVICE,
) -> FinalRecommender:
    """Return the process-wide final recommender without connecting it to Flask."""

    global _singleton
    requested_cache = Path(ckg_cache_dir).resolve()
    requested_checkpoint = Path(checkpoint_path).resolve()
    with _singleton_lock:
        if _singleton is None:
            _singleton = FinalRecommender(
                ckg_cache_dir=requested_cache,
                checkpoint_path=requested_checkpoint,
                device=device,
            )
        elif (
            _singleton.ckg_cache_dir != requested_cache
            or _singleton.checkpoint_path != requested_checkpoint
            or _singleton.device != str(device)
        ):
            raise FinalRecommenderReadinessError(
                "Final recommender is already configured with different artifacts "
                "or device"
            )
        return _singleton


__all__ = [
    "DEFAULT_CKG_CACHE_DIR",
    "FROZEN_CKG_CACHE_NAME",
    "FROZEN_RRF_K",
    "FinalRecommendationUnavailableError",
    "FinalRecommender",
    "FinalRecommenderError",
    "FinalRecommenderInvariantError",
    "FinalRecommenderReadinessError",
    "FinalRecommenderState",
    "KBRD_CHECKPOINT_PATH",
    "KBRD_DATA_DIR",
    "KBRD_ROOT",
    "get_final_recommender",
]
