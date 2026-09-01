"""Frozen Stage-2 v2 partial-label plus RRF-anchor scientific objective."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch.nn import functional as F

from my_crs.analyze_stage2_v2_tokens import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
)
from my_crs.build_stage2_v2_dataset import TOP_K, canonical_json_bytes
from my_crs.joint_rrf_ranker import canonicalize_phase1_candidates
from my_crs.stage2_v2_peft import (
    EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT,
    MODEL_ID,
    REQUESTED_MODEL_REVISION,
    phase3b_integration_fingerprint,
)


LOSS_VERSION = "stage2_v2_partial_set_likelihood_rrf_anchor_v1"
EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT = (
    "ee755d8860b48992b5eac5067c2d463964f7b3e5b21d2766db807b72a8148d36"
)

TRAIN_RETRIEVAL_COMPLETED_EVENTS = 20055
TRAIN_POSITIVE_EVENTS = 12970
POSITIVE_ESTIMATOR_WEIGHT = (
    TRAIN_RETRIEVAL_COMPLETED_EVENTS / TRAIN_POSITIVE_EVENTS
)
PREREGISTERED_BETA_GRID = (0.03, 0.10, 0.30, 1.00, 3.00)


def _fingerprint(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()


def _validated_beta(beta: float) -> float:
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise ValueError("beta must be a finite nonnegative number")
    value = float(beta)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("beta must be a finite nonnegative number")
    return value


def loss_scientific_configuration(beta: float) -> dict[str, Any]:
    observed_phase3b = phase3b_integration_fingerprint()
    if observed_phase3b != EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT:
        raise RuntimeError(
            "Frozen Phase-3B integration fingerprint mismatch: "
            f"expected={EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT} "
            f"observed={observed_phase3b}"
        )
    return {
        "anchor": {
            "definition": "kl_fixed_rrf_prior_to_contextual_distribution",
            "events": "all_train_retrieval_completed",
        },
        "beta": _validated_beta(beta),
        "candidate_count": TOP_K,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "loss_arithmetic_dtype": "float64",
        "loss_version": LOSS_VERSION,
        "model_id": MODEL_ID,
        "partial_labels": {
            "definition": "negative_log_observed_positive_set_probability",
            "nonpositive_candidates": "unlabeled_not_hard_negatives",
            "positive_set_order": "unordered",
        },
        "phase2_architecture_fingerprint": EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
        "phase3a_analysis_fingerprint": EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT,
        "phase3b_integration_fingerprint": observed_phase3b,
        "positive_estimator": {
            "computed_weight": POSITIVE_ESTIMATOR_WEIGHT,
            "denominator_positive_events": TRAIN_POSITIVE_EVENTS,
            "numerator_retrieval_completed_events": TRAIN_RETRIEVAL_COMPLETED_EVENTS,
            "weight_definition": "retrieval_completed_divided_by_positive_events",
        },
        "prior": {
            "autograd": "detached",
            "definition": "positive_rrf_score_normalized_over_50",
        },
        "requested_model_revision": REQUESTED_MODEL_REVISION,
        "residual_centering": "per_event_arithmetic_mean_over_50",
    }


def loss_scientific_fingerprint(beta: float) -> str:
    return _fingerprint(loss_scientific_configuration(beta))


@dataclass(frozen=True)
class ContextualProbabilityState:
    q: torch.Tensor
    log_q: torch.Tensor
    centered_residuals: torch.Tensor
    logits: torch.Tensor
    log_p: torch.Tensor
    p: torch.Tensor


@dataclass(frozen=True)
class V2BatchLoss:
    total_loss: torch.Tensor
    per_event_losses: torch.Tensor
    partial_losses: torch.Tensor
    anchor_losses: torch.Tensor
    positive_mask: torch.Tensor
    positive_event_rate: torch.Tensor
    mean_partial_over_positive_events: torch.Tensor
    mean_anchor_over_all_events: torch.Tensor
    probability_state: ContextualProbabilityState
    beta: float
    positive_estimator_weight: float = POSITIVE_ESTIMATOR_WEIGHT


@dataclass(frozen=True)
class TrainingLossInputs:
    rrf_scores: tuple[float, ...]
    positive_serialization_positions: tuple[int, ...]
    instance_key: str


def contextual_probability_state(
    rrf_scores: torch.Tensor | Sequence[Sequence[float]],
    raw_residuals: torch.Tensor,
) -> ContextualProbabilityState:
    if not isinstance(raw_residuals, torch.Tensor):
        raise ValueError("raw_residuals must be a tensor preserving model autograd")
    if raw_residuals.ndim != 2 or raw_residuals.shape[1] != TOP_K:
        raise ValueError("raw_residuals must have shape [batch, 50]")
    if not torch.is_floating_point(raw_residuals):
        raise ValueError("raw_residuals must be floating point")
    if not torch.isfinite(raw_residuals).all():
        raise ValueError("raw_residuals must be finite")

    prior = (
        rrf_scores
        if isinstance(rrf_scores, torch.Tensor)
        else torch.tensor(rrf_scores, dtype=torch.float64)
    )
    if prior.shape != raw_residuals.shape:
        raise ValueError("rrf_scores and raw_residuals must both have shape [batch, 50]")
    prior = prior.detach().to(device=raw_residuals.device, dtype=torch.float64)
    if not torch.isfinite(prior).all() or not torch.all(prior > 0.0):
        raise ValueError("Every RRF score must be finite and strictly positive")

    q = prior / prior.sum(dim=-1, keepdim=True)
    log_q = torch.log(q)
    residuals_64 = raw_residuals.to(torch.float64)
    centered = residuals_64 - residuals_64.mean(dim=-1, keepdim=True)
    logits = log_q + centered
    log_p = F.log_softmax(logits, dim=-1)
    p = torch.exp(log_p)
    tensors = (q, log_q, centered, logits, log_p, p)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("Stage-2 v2 probability construction became non-finite")
    return ContextualProbabilityState(q, log_q, centered, logits, log_p, p)


def _validate_positive_positions(
    values: Sequence[int],
    *,
    context: str,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{context} positive positions must be a sequence")
    positions: list[int] = []
    for value in values:
        if type(value) is not int or not 1 <= value <= TOP_K:
            raise ValueError(f"{context} positive positions must be integers in 1..50")
        positions.append(value)
    if len(positions) != len(set(positions)):
        raise ValueError(f"{context} positive positions contain duplicates")
    return tuple(positions)


def compute_v2_batch_loss(
    rrf_scores: torch.Tensor | Sequence[Sequence[float]],
    raw_residuals: torch.Tensor,
    positive_serialization_positions: Sequence[Sequence[int]],
    *,
    beta: float,
) -> V2BatchLoss:
    """Compute the unbiased per-event estimator of the frozen v2 objective."""

    beta_value = _validated_beta(beta)
    state = contextual_probability_state(rrf_scores, raw_residuals)
    batch_size = int(raw_residuals.shape[0])
    if len(positive_serialization_positions) != batch_size:
        raise ValueError("Exactly one positive-position set is required per event")

    partial_losses: list[torch.Tensor] = []
    positive_flags: list[bool] = []
    for batch_index, raw_positions in enumerate(positive_serialization_positions):
        positions = _validate_positive_positions(
            raw_positions,
            context=f"event {batch_index}",
        )
        if positions:
            indices = torch.tensor(
                [position - 1 for position in positions],
                dtype=torch.long,
                device=state.log_p.device,
            )
            partial = -torch.logsumexp(state.log_p[batch_index, indices], dim=0)
            positive_flags.append(True)
        else:
            partial = state.log_p[batch_index].sum() * 0.0
            positive_flags.append(False)
        partial_losses.append(partial)

    partial_tensor = torch.stack(partial_losses)
    anchor_tensor = torch.sum(
        state.q * (state.log_q - state.log_p),
        dim=-1,
    )
    positive_mask = torch.tensor(
        positive_flags,
        dtype=torch.bool,
        device=raw_residuals.device,
    )
    weighted_partial = partial_tensor * positive_mask.to(torch.float64)
    weighted_partial = weighted_partial * POSITIVE_ESTIMATOR_WEIGHT
    per_event = weighted_partial + beta_value * anchor_tensor
    total = per_event.mean()
    positive_count = int(positive_mask.sum().item())
    if positive_count:
        mean_partial = partial_tensor[positive_mask].mean()
    else:
        mean_partial = partial_tensor.sum() * 0.0
    mean_anchor = anchor_tensor.mean()
    positive_rate = positive_mask.to(torch.float64).mean()
    for name, value in (
        ("partial", partial_tensor),
        ("anchor", anchor_tensor),
        ("per-event", per_event),
        ("total", total),
    ):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Stage-2 v2 {name} loss became non-finite")
    return V2BatchLoss(
        total_loss=total,
        per_event_losses=per_event,
        partial_losses=partial_tensor,
        anchor_losses=anchor_tensor,
        positive_mask=positive_mask,
        positive_event_rate=positive_rate,
        mean_partial_over_positive_events=mean_partial,
        mean_anchor_over_all_events=mean_anchor,
        probability_state=state,
        beta=beta_value,
    )


def compute_v2_event_loss(
    rrf_scores: torch.Tensor | Sequence[float],
    raw_residuals: torch.Tensor,
    positive_serialization_positions: Sequence[int],
    *,
    beta: float,
) -> V2BatchLoss:
    if raw_residuals.ndim != 1 or raw_residuals.shape[0] != TOP_K:
        raise ValueError("Event residuals must have shape [50]")
    prior = rrf_scores.unsqueeze(0) if isinstance(rrf_scores, torch.Tensor) else [rrf_scores]
    return compute_v2_batch_loss(
        prior,
        raw_residuals.unsqueeze(0),
        [positive_serialization_positions],
        beta=beta,
    )


def training_loss_inputs_from_record(record: Mapping[str, Any]) -> TrainingLossInputs:
    if record.get("split") != "train":
        raise ValueError("Stage-2 v2 training accepts TRAIN records only")
    candidates = canonicalize_phase1_candidates(record)
    raw_positions = record.get("observed_positive_serialization_positions")
    positions = _validate_positive_positions(
        raw_positions,
        context=str(record.get("instance_key", "record")),
    )
    instance_key = record.get("instance_key")
    if not isinstance(instance_key, str) or not instance_key:
        raise ValueError("Stage-2 v2 training record lacks an instance key")
    scores = tuple(float(candidate["rrf_score"]) for candidate in candidates)
    if len(scores) != TOP_K or any(
        not math.isfinite(score) or score <= 0.0 for score in scores
    ):
        raise ValueError("Training record RRF scores must be 50 finite positive values")
    return TrainingLossInputs(scores, positions, instance_key)


__all__ = [
    "EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT",
    "LOSS_VERSION",
    "POSITIVE_ESTIMATOR_WEIGHT",
    "PREREGISTERED_BETA_GRID",
    "TRAIN_POSITIVE_EVENTS",
    "TRAIN_RETRIEVAL_COMPLETED_EVENTS",
    "ContextualProbabilityState",
    "TrainingLossInputs",
    "V2BatchLoss",
    "compute_v2_batch_loss",
    "compute_v2_event_loss",
    "contextual_probability_state",
    "loss_scientific_configuration",
    "loss_scientific_fingerprint",
    "training_loss_inputs_from_record",
]
