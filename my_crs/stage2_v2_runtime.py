"""Reusable inference runtime for the frozen Stage2-v2 ranker.

This module intentionally contains no alternative ranking implementation.  It
loads and calls the exact serializer, tokenizer, model builder, checkpoint
validator, float64 score combiner, and deterministic ranker used by the frozen
Stage2-v2 evaluation code.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_TOP_K = 50
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "stage2_v2_3b_beta100_seed42"
    / "checkpoints"
    / "checkpoint_step_00001254.pt"
)
DEFAULT_DEVICE = "cuda:0"


class Stage2V2RuntimeError(RuntimeError):
    """Base error for the final Stage2-v2 runtime."""


class Stage2V2ReadinessError(Stage2V2RuntimeError):
    """Raised when frozen artifacts, dependencies, or hardware are unavailable."""


class Stage2V2InvariantError(Stage2V2RuntimeError):
    """Raised when an inference request violates a frozen Stage2-v2 invariant."""


@dataclass(frozen=True)
class Stage2V2RuntimeState:
    checkpoint_path: Path
    checkpoint_sha256: str
    scientific_fingerprint: str
    checkpoint_step: int
    beta: float
    device: str
    model_identity: dict[str, Any]
    resolved_tokenizer_commit: str | None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_step": self.checkpoint_step,
            "device": self.device,
            "model_identity": dict(self.model_identity),
            "resolved_tokenizer_commit": self.resolved_tokenizer_commit,
            "scientific_fingerprint": self.scientific_fingerprint,
        }


@dataclass
class _LoadedStage2V2:
    tokenizer: Any
    ranker: Any
    peft_module: Any
    device: Any
    state: Stage2V2RuntimeState


def _load_frozen_bindings() -> Any:
    """Import the frozen scientific surface lazily.

    Lazy imports let readiness checks fail clearly on installations where the
    CUDA/PyTorch/Transformers stack has not been installed yet.
    """

    try:
        import torch

        from my_crs.build_stage2_v2_dataset import (
            CANDIDATE_ORDER_VERSION,
            DATASET_SCHEMA_VERSION,
            TOP_K as DATASET_TOP_K,
            canonical_json_digest,
            serialize_candidates,
        )
        from my_crs.evaluate_stage2_v2_dev import load_inference_stack
        from my_crs.evaluate_stage2_v2_test import validate_selected_checkpoint
        from my_crs.joint_rrf_ranker import (
            TOP_K as RANKER_TOP_K,
            canonicalize_phase1_candidates,
            combine_rrf_prior,
            rank_candidate_ids,
        )
        from my_crs.stage2_v2_peft import (
            require_single_cuda_device,
            tokenize_single_smoke_event,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise Stage2V2ReadinessError(
            "The frozen Stage2-v2 Python stack is unavailable. Install the exact "
            "PyTorch, Transformers, and PEFT runtime used by the selected model."
        ) from error

    if DATASET_TOP_K != FROZEN_TOP_K or RANKER_TOP_K != FROZEN_TOP_K:
        raise Stage2V2ReadinessError(
            "Frozen Stage2-v2 modules disagree with the required Top-50 policy"
        )

    def infer_residuals(
        record: Mapping[str, Any],
        tokenizer: Any,
        ranker: Any,
        device: Any,
    ) -> tuple[Any, int]:
        candidates = canonicalize_phase1_candidates(record)
        event, batch, actual_tokens = tokenize_single_smoke_event(record, tokenizer)
        expected_ids = tuple(
            int(candidate["canonical_entity_id"]) for candidate in candidates
        )
        expected_ranks = tuple(int(candidate["rrf_rank"]) for candidate in candidates)
        expected_scores = tuple(
            float(candidate["rrf_score"]) for candidate in candidates
        )
        if (
            event.canonical_entity_ids != expected_ids
            or event.rrf_ranks != expected_ranks
            or event.rrf_scores != expected_scores
        ):
            raise Stage2V2InvariantError(
                "Tokenized candidate order disagrees with frozen serialization order"
            )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            residuals = ranker(batch.to(device))
        if not isinstance(residuals, torch.Tensor) or residuals.shape != (
            1,
            FROZEN_TOP_K,
        ):
            raise Stage2V2InvariantError(
                "JointRRFRanker must return one residual for each of 50 candidates"
            )
        return residuals[0], actual_tokens

    def combine_prior(rrf_scores: Sequence[float], residuals: Any) -> Any:
        prior = torch.tensor(
            rrf_scores,
            dtype=torch.float64,
            device=residuals.device,
        )
        return combine_rrf_prior(prior, residuals)

    return SimpleNamespace(
        candidate_order_version=CANDIDATE_ORDER_VERSION,
        canonical_json_digest=canonical_json_digest,
        canonicalize_phase1_candidates=canonicalize_phase1_candidates,
        combine_rrf_prior=combine_prior,
        dataset_schema_version=DATASET_SCHEMA_VERSION,
        infer_residuals=infer_residuals,
        load_inference_stack=load_inference_stack,
        rank_candidate_ids=rank_candidate_ids,
        require_single_cuda_device=require_single_cuda_device,
        serialize_candidates=serialize_candidates,
        top_k=DATASET_TOP_K,
        validate_selected_checkpoint=validate_selected_checkpoint,
    )


def _numeric_values(value: Any, field: str) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != FROZEN_TOP_K:
        raise Stage2V2InvariantError(f"{field} must contain exactly 50 values")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as error:
        raise Stage2V2InvariantError(f"{field} contains a non-numeric value") from error


class Stage2V2Runtime:
    """Load the selected ranker once and score frozen RRF Top-50 events."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        device: str = DEFAULT_DEVICE,
        *,
        _bindings: Any | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = str(device)
        self._bindings = _bindings
        self._loaded: _LoadedStage2V2 | None = None
        self._load_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._loaded is not None

    def ensure_ready(self) -> Stage2V2RuntimeState:
        if self._loaded is not None:
            return self._loaded.state
        with self._load_lock:
            if self._loaded is not None:
                return self._loaded.state
            if not self.checkpoint_path.is_file():
                raise Stage2V2ReadinessError(
                    "Selected Stage2-v2 checkpoint is missing: "
                    f"{self.checkpoint_path}"
                )
            bindings = self._bindings or _load_frozen_bindings()
            if int(bindings.top_k) != FROZEN_TOP_K:
                raise Stage2V2ReadinessError(
                    "Stage2-v2 runtime bindings do not enforce Top 50"
                )
            try:
                checkpoint = bindings.validate_selected_checkpoint(
                    self.checkpoint_path
                )
                selected_device = bindings.require_single_cuda_device(self.device)
                (
                    tokenizer,
                    ranker,
                    peft_module,
                    model_identity,
                    resolved_tokenizer_commit,
                ) = bindings.load_inference_stack(checkpoint, selected_device)
                loss = checkpoint.scientific_configuration.get("loss", {})
                state = Stage2V2RuntimeState(
                    checkpoint_path=Path(checkpoint.path).resolve(),
                    checkpoint_sha256=str(checkpoint.sha256),
                    scientific_fingerprint=str(
                        checkpoint.scientific_fingerprint
                    ),
                    checkpoint_step=int(checkpoint.optimizer_step),
                    beta=float(loss["beta"]),
                    device=str(selected_device),
                    model_identity=dict(model_identity),
                    resolved_tokenizer_commit=resolved_tokenizer_commit,
                )
            except Stage2V2RuntimeError:
                raise
            except Exception as error:
                raise Stage2V2ReadinessError(
                    "Unable to initialize the selected Stage2-v2 runtime: "
                    f"{error}"
                ) from error
            self._bindings = bindings
            self._loaded = _LoadedStage2V2(
                tokenizer=tokenizer,
                ranker=ranker,
                peft_module=peft_module,
                device=selected_device,
                state=state,
            )
            return state

    def _prepare_rrf_top50(
        self, rrf_top50: Sequence[Mapping[str, Any]]
    ) -> list[dict[str, Any]]:
        if not isinstance(rrf_top50, Sequence) or isinstance(
            rrf_top50, (str, bytes)
        ):
            raise Stage2V2InvariantError("RRF candidates must be a sequence")
        if len(rrf_top50) != FROZEN_TOP_K:
            raise Stage2V2InvariantError("Stage2-v2 requires exactly 50 RRF candidates")
        prepared: list[dict[str, Any]] = []
        ids: list[int] = []
        for rrf_rank, candidate in enumerate(rrf_top50, 1):
            if not isinstance(candidate, Mapping):
                raise Stage2V2InvariantError(
                    f"RRF candidate {rrf_rank} must be an object"
                )
            item = dict(candidate)
            entity_id = item.get("id")
            if type(entity_id) is not int:
                raise Stage2V2InvariantError(
                    f"RRF candidate {rrf_rank} ID must be an integer"
                )
            if item.get("source") != "RRF":
                raise Stage2V2InvariantError(
                    f"RRF candidate {rrf_rank} has non-RRF provenance"
                )
            declared_rank = item.get("rank")
            if declared_rank is not None and declared_rank != rrf_rank:
                raise Stage2V2InvariantError(
                    "RRF candidates must be supplied in exact rank order"
                )
            item["rank"] = rrf_rank
            ids.append(entity_id)
            prepared.append(item)
        if len(set(ids)) != FROZEN_TOP_K:
            raise Stage2V2InvariantError(
                "Stage2-v2 input contains duplicate canonical entity IDs"
            )
        return prepared

    def rank(
        self,
        history: str,
        rrf_top50: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Rerank one RRF Top-50 without adding or removing candidates."""

        if not isinstance(history, str):
            raise Stage2V2InvariantError("Stage2-v2 history must be a string")
        state = self.ensure_ready()
        assert self._loaded is not None
        bindings = self._bindings
        prepared = self._prepare_rrf_top50(rrf_top50)
        serialized = bindings.serialize_candidates(prepared)
        record = {
            "candidate_count": FROZEN_TOP_K,
            "candidates": serialized,
            "history": history,
            "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
            "schema_version": bindings.dataset_schema_version,
            "serialization_digest": bindings.canonical_json_digest(serialized),
            "serialization_order_version": bindings.candidate_order_version,
        }
        canonical = bindings.canonicalize_phase1_candidates(record)
        raw_residuals, actual_tokens = bindings.infer_residuals(
            record,
            self._loaded.tokenizer,
            self._loaded.ranker,
            self._loaded.device,
        )
        canonical_ids = [
            int(candidate["canonical_entity_id"]) for candidate in canonical
        ]
        rrf_ranks = [int(candidate["rrf_rank"]) for candidate in canonical]
        rrf_scores = [float(candidate["rrf_score"]) for candidate in canonical]
        combination = bindings.combine_rrf_prior(rrf_scores, raw_residuals)
        ranked_ids = bindings.rank_candidate_ids(
            combination.final_scores,
            canonical_ids,
            rrf_ranks,
        )
        if len(ranked_ids) != FROZEN_TOP_K or set(ranked_ids) != set(canonical_ids):
            raise Stage2V2InvariantError(
                "Stage2-v2 changed the frozen RRF Top-50 membership"
            )

        raw_values = _numeric_values(raw_residuals, "raw residuals")
        centered_values = _numeric_values(
            combination.centered_residuals, "centered residuals"
        )
        log_prior_values = _numeric_values(combination.log_prior, "log-RRF prior")
        final_values = _numeric_values(combination.final_scores, "final scores")
        score_by_id = {
            entity_id: {
                "stage2_centered_residual": centered_values[index],
                "stage2_final_score": final_values[index],
                "stage2_log_rrf_prior": log_prior_values[index],
                "stage2_raw_residual": raw_values[index],
            }
            for index, entity_id in enumerate(canonical_ids)
        }
        source_by_id = {int(candidate["id"]): dict(candidate) for candidate in prepared}
        ranked_candidates: list[dict[str, Any]] = []
        for stage2_rank, entity_id in enumerate(ranked_ids, 1):
            item = source_by_id[entity_id]
            item["stage1_rrf_rank"] = int(item["rank"])
            item["rank"] = stage2_rank
            item["stage2_rank"] = stage2_rank
            item.update(score_by_id[entity_id])
            ranked_candidates.append(item)

        return {
            "ranked_candidates": ranked_candidates,
            "selected_candidate": dict(ranked_candidates[0]),
            "diagnostics": {
                "actual_packed_tokens": int(actual_tokens),
                "candidate_membership_preserved": True,
                "input_rrf_candidate_ids": [int(item["id"]) for item in prepared],
                "output_stage2_candidate_ids": [
                    int(item["id"]) for item in ranked_candidates
                ],
                "runtime": state.diagnostics(),
                "top_k": FROZEN_TOP_K,
            },
        }


_singleton_lock = threading.Lock()
_singleton: Stage2V2Runtime | None = None


def get_stage2_v2_runtime(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: str = DEFAULT_DEVICE,
) -> Stage2V2Runtime:
    """Return the process-wide Stage2-v2 runtime for one exact configuration."""

    global _singleton
    requested_path = Path(checkpoint_path).resolve()
    with _singleton_lock:
        if _singleton is None:
            _singleton = Stage2V2Runtime(requested_path, device)
        elif (
            _singleton.checkpoint_path != requested_path
            or _singleton.device != str(device)
        ):
            raise Stage2V2ReadinessError(
                "Stage2-v2 runtime is already configured with a different "
                "checkpoint or device"
            )
        return _singleton


__all__ = [
    "DEFAULT_CHECKPOINT_PATH",
    "DEFAULT_DEVICE",
    "FROZEN_TOP_K",
    "Stage2V2InvariantError",
    "Stage2V2ReadinessError",
    "Stage2V2Runtime",
    "Stage2V2RuntimeError",
    "Stage2V2RuntimeState",
    "get_stage2_v2_runtime",
]
