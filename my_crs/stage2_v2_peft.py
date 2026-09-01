"""Stage-2 v2 PEFT integration and smoke-only validation utilities.

This module deliberately contains no scientific training loss, dataset loop,
beta selection, scheduler, or recommendation evaluation.  It only binds the
frozen Stage-2 v2 scorer to the exact Qwen2.5-3B base and LoRA policy, then
provides fail-closed checks used by the one-record GPU smoke.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import transformers
from torch import nn
from torch.nn import functional as F

from my_crs.analyze_stage2_v2_tokens import (
    DEFAULT_DATASET_PATH,
    EXPECTED_DATASET_SHA256,
    EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
    MODEL_ID,
    REQUESTED_MODEL_REVISION,
    analysis_configuration,
    load_production_tokenizer,
    validate_tokenizer,
)
from my_crs.build_stage2_v2_dataset import (
    PROJECT_ROOT,
    TOP_K,
    canonical_json_bytes,
    sha256_file,
)
from my_crs.joint_rrf_ranker import (
    REQUIRED_ATTENTION_BACKEND,
    REQUIRED_MODEL_TYPE,
    JointRRFRanker,
    PackedScoringBatch,
    PackedScoringEvent,
    collate_scoring_events,
    mean_center_residuals,
    phase2_architecture_fingerprint,
    tokenize_scoring_event,
)


PHASE3B_INTEGRATION_VERSION = "stage2_v2_qwen25_3b_peft_integration_v1"
REPORT_SCHEMA_VERSION = "stage2_v2_qwen25_3b_real_model_smoke_v1"
EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT = (
    "f1b72fb7318880c04e9e810afd6c3dbbe343e808422ecd0c125e534191047f0a"
)

MAX_PACKED_TOKENS = 2304
TRUNCATION_ALLOWED = False
MODEL_DTYPE = torch.bfloat16
MODEL_DTYPE_NAME = "bfloat16"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
LORA_TASK_TYPE = "FEATURE_EXTRACTION"
QUANTIZATION_ENABLED = False

GRADIENT_CHECKPOINTING_ENABLED = True
GRADIENT_CHECKPOINTING_USE_REENTRANT = False
USE_CACHE = False

SMOKE_OBJECTIVE_VERSION = "smoke_only_centered_residual_c1_minus_c2_v1"
SMOKE_OPTIMIZER = "AdamW"
SMOKE_LEARNING_RATE = 1e-4
SMOKE_WEIGHT_DECAY = 0.0
FIRST_BACKWARD_LORA_ZERO_TOLERANCE = 1e-12

PERMUTATION_CHECK_VERSION = "hidden_state_reverse_physical_blocks_v1"
PERMUTATION_MINIMUM_COSINE = 0.999

DEFAULT_SMOKE_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "stage2_v2_real_model_smoke"
)
SMOKE_REPORT_FILENAME = "stage2_v2_qwen25_3b_smoke.json"


def _fingerprint(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()


def _phase3a_analysis_fingerprint() -> str:
    return _fingerprint(analysis_configuration())


def phase3b_scientific_configuration() -> dict[str, Any]:
    """Return environment-independent integration and smoke configuration."""

    observed_phase2 = phase2_architecture_fingerprint()
    if observed_phase2 != EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT:
        raise RuntimeError(
            "Frozen Phase-2 architecture fingerprint mismatch: "
            f"expected={EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT} "
            f"observed={observed_phase2}"
        )
    observed_phase3a = _phase3a_analysis_fingerprint()
    if observed_phase3a != EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT:
        raise RuntimeError(
            "Frozen Phase-3A analysis fingerprint mismatch: "
            f"expected={EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT} "
            f"observed={observed_phase3a}"
        )
    return {
        "attention_backend": REQUIRED_ATTENTION_BACKEND,
        "base_model_policy": {
            "all_non_lora_parameters_frozen": True,
            "model_type": REQUIRED_MODEL_TYPE,
        },
        "candidate_count": TOP_K,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "dtype": MODEL_DTYPE_NAME,
        "gradient_checkpointing": {
            "enabled": GRADIENT_CHECKPOINTING_ENABLED,
            "use_reentrant": GRADIENT_CHECKPOINTING_USE_REENTRANT,
        },
        "integration_version": PHASE3B_INTEGRATION_VERSION,
        "lora": {
            "bias": LORA_BIAS,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "quantization": QUANTIZATION_ENABLED,
            "r": LORA_R,
            "target_modules": list(LORA_TARGET_MODULES),
            "task_type": LORA_TASK_TYPE,
        },
        "max_packed_tokens": MAX_PACKED_TOKENS,
        "model_id": MODEL_ID,
        "permutation_check": {
            "minimum_cosine_similarity": PERMUTATION_MINIMUM_COSINE,
            "version": PERMUTATION_CHECK_VERSION,
        },
        "phase2_architecture_fingerprint": observed_phase2,
        "phase3a_analysis_fingerprint": observed_phase3a,
        "requested_model_revision": REQUESTED_MODEL_REVISION,
        "scorer_head_policy": {
            "implementation": "frozen_joint_rrf_ranker_shared_head",
            "only_scorer_head_trainable_outside_lora": True,
            "zero_initialized_projection": True,
        },
        "smoke_gradient_protocol": {
            "learning_rate": SMOKE_LEARNING_RATE,
            "objective_version": SMOKE_OBJECTIVE_VERSION,
            "optimizer": SMOKE_OPTIMIZER,
            "optimizer_steps": 1,
            "scientific_training_objective": False,
            "weight_decay": SMOKE_WEIGHT_DECAY,
        },
        "truncation": TRUNCATION_ALLOWED,
        "use_cache": USE_CACHE,
    }


def phase3b_integration_fingerprint() -> str:
    return _fingerprint(phase3b_scientific_configuration())


def _load_peft_module() -> ModuleType:
    try:
        import peft
    except ImportError as error:
        raise RuntimeError("Phase-3B real-model smoke requires PEFT") from error
    return peft


def create_lora_config(*, peft_module: ModuleType | Any | None = None) -> Any:
    peft_module = peft_module or _load_peft_module()
    task_type = getattr(getattr(peft_module, "TaskType", None), LORA_TASK_TYPE, None)
    if task_type is None:
        raise RuntimeError("PEFT does not expose TaskType.FEATURE_EXTRACTION")
    return peft_module.LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=list(LORA_TARGET_MODULES),
        task_type=task_type,
    )


def _resolved_model_commit(model: nn.Module) -> str | None:
    config = getattr(model, "config", None)
    candidates = [
        getattr(model, "_commit_hash", None),
        getattr(config, "_commit_hash", None),
    ]
    commits = {str(value) for value in candidates if value}
    if len(commits) > 1:
        raise RuntimeError("Qwen model exposes conflicting resolved commit hashes")
    return next(iter(commits), None)


def validate_loaded_base_qwen(model: nn.Module) -> dict[str, Any]:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != REQUIRED_MODEL_TYPE:
        raise RuntimeError("Phase-3B requires the Qwen2 base/encoder model")
    if getattr(config, "_attn_implementation", None) != REQUIRED_ATTENTION_BACKEND:
        raise RuntimeError("Phase-3B requires explicit SDPA attention")
    if getattr(config, "use_cache", None) is not False:
        raise RuntimeError("Phase-3B Qwen config must have use_cache=False")
    resolved_commit = _resolved_model_commit(model)
    if resolved_commit is not None and resolved_commit != REQUESTED_MODEL_REVISION:
        raise RuntimeError(
            "Qwen resolved commit mismatch: "
            f"requested={REQUESTED_MODEL_REVISION} resolved={resolved_commit}"
        )
    floating_dtypes = {
        parameter.dtype
        for parameter in model.parameters()
        if torch.is_floating_point(parameter)
    }
    if not floating_dtypes:
        raise RuntimeError("Qwen base exposes no floating-point parameters")
    if floating_dtypes != {MODEL_DTYPE}:
        names = sorted(str(dtype).removeprefix("torch.") for dtype in floating_dtypes)
        raise RuntimeError(f"Qwen base parameters must all be BF16; observed {names}")
    return {
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "model_type": str(config.model_type),
        "requested_model_id": MODEL_ID,
        "requested_revision": REQUESTED_MODEL_REVISION,
        "resolved_commit": resolved_commit,
    }


def load_base_qwen(
    device: torch.device | str,
    *,
    auto_model_class: Any | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the exact non-generative Qwen2 base without sharding/quantization."""

    if auto_model_class is None:
        from transformers import AutoModel

        auto_model_class = AutoModel
    model = auto_model_class.from_pretrained(
        MODEL_ID,
        revision=REQUESTED_MODEL_REVISION,
        dtype=MODEL_DTYPE,
        attn_implementation=REQUIRED_ATTENTION_BACKEND,
    )
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("Loaded Qwen base has no configuration")
    config.use_cache = USE_CACHE
    model = model.to(torch.device(device))
    return model, validate_loaded_base_qwen(model)


def enable_required_gradient_checkpointing(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("PEFT-wrapped Qwen model has no configuration")
    config.use_cache = USE_CACHE
    enable = getattr(model, "gradient_checkpointing_enable", None)
    if not callable(enable):
        raise RuntimeError("PEFT-wrapped Qwen does not support gradient checkpointing")
    try:
        enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": GRADIENT_CHECKPOINTING_USE_REENTRANT
            }
        )
    except TypeError as error:
        raise RuntimeError(
            "Runtime cannot honor required non-reentrant gradient checkpointing"
        ) from error
    require_inputs = getattr(model, "enable_input_require_grads", None)
    if not callable(require_inputs):
        raise RuntimeError("PEFT-wrapped Qwen cannot require embedding outputs to grad")
    require_inputs()
    status = getattr(model, "is_gradient_checkpointing", None)
    if status is not True:
        base = getattr(model, "get_base_model", lambda: None)()
        status = getattr(base, "is_gradient_checkpointing", None)
    if status is not True:
        raise RuntimeError("Gradient checkpointing did not become active")
    if getattr(config, "use_cache", None) is not False:
        raise RuntimeError("Gradient-checkpointed Qwen must retain use_cache=False")


def _parameter_group(name: str) -> str:
    if name.startswith("scoring_head."):
        return "scorer_head"
    if ".lora_A." in name or ".lora_B." in name:
        return "lora"
    return "base"


def _actual_lora_modules(ranker: JointRRFRanker) -> list[str]:
    names: list[str] = []
    target_counts: Counter[str] = Counter()
    for name, module in ranker.base_model.named_modules():
        has_a = hasattr(module, "lora_A")
        has_b = hasattr(module, "lora_B")
        if has_a != has_b:
            raise RuntimeError(f"Incomplete LoRA A/B module at {name}")
        if not has_a:
            continue
        target = name.rsplit(".", 1)[-1]
        if target not in LORA_TARGET_MODULES:
            raise RuntimeError(f"Unexpected LoRA target module: {name}")
        names.append(name)
        target_counts[target] += 1
    if not names:
        raise RuntimeError("No LoRA target modules were installed")
    if set(target_counts) != set(LORA_TARGET_MODULES):
        raise RuntimeError(
            "LoRA target set mismatch: "
            f"expected={sorted(LORA_TARGET_MODULES)} observed={sorted(target_counts)}"
        )
    counts = set(target_counts.values())
    if len(counts) != 1:
        raise RuntimeError(f"LoRA target module counts are inconsistent: {target_counts}")
    return sorted(names)


def validate_trainability(ranker: JointRRFRanker) -> dict[str, Any]:
    """Independently enforce frozen-base + trainable-LoRA + trainable-head."""

    totals = Counter()
    trainable = Counter()
    names_by_group: dict[str, list[str]] = {
        "base": [],
        "lora": [],
        "scorer_head": [],
    }
    for name, parameter in ranker.named_parameters():
        group = _parameter_group(name)
        count = int(parameter.numel())
        totals[group] += count
        if parameter.requires_grad:
            trainable[group] += count
            names_by_group[group].append(name)

    if totals["base"] <= 0 or totals["lora"] <= 0 or totals["scorer_head"] <= 0:
        raise RuntimeError(f"Incomplete trainability groups: {dict(totals)}")
    unexpected_base = names_by_group["base"]
    if unexpected_base:
        raise RuntimeError(
            "Unexpected trainable non-LoRA base parameters: "
            + ", ".join(unexpected_base[:8])
        )
    missing_lora = [
        name
        for name, parameter in ranker.named_parameters()
        if _parameter_group(name) == "lora" and not parameter.requires_grad
    ]
    if missing_lora:
        raise RuntimeError(
            "LoRA parameters are unexpectedly frozen: " + ", ".join(missing_lora[:8])
        )
    missing_head = [
        name
        for name, parameter in ranker.named_parameters()
        if _parameter_group(name) == "scorer_head" and not parameter.requires_grad
    ]
    if missing_head:
        raise RuntimeError(
            "Scorer-head parameters are unexpectedly frozen: "
            + ", ".join(missing_head[:8])
        )
    actual_modules = _actual_lora_modules(ranker)
    total_parameters = sum(totals.values())
    total_trainable = sum(trainable.values())
    if total_trainable != trainable["lora"] + trainable["scorer_head"]:
        raise RuntimeError("Unexpected trainable parameter group")
    return {
        "actual_lora_module_names": actual_modules,
        "frozen_base_parameters": totals["base"],
        "lora_trainable_parameters": trainable["lora"],
        "scorer_head_trainable_parameters": trainable["scorer_head"],
        "total_parameters": total_parameters,
        "total_trainable_parameters": total_trainable,
        "trainable_percentage": 100.0 * total_trainable / total_parameters,
    }


def apply_peft_and_build_ranker(
    base_model: nn.Module,
    device: torch.device | str,
    *,
    peft_module: ModuleType | Any | None = None,
) -> tuple[JointRRFRanker, dict[str, Any]]:
    peft_module = peft_module or _load_peft_module()
    lora_config = create_lora_config(peft_module=peft_module)
    peft_model = peft_module.get_peft_model(base_model, lora_config)
    enable_required_gradient_checkpointing(peft_model)
    ranker = JointRRFRanker(peft_model).to(torch.device(device))
    report = validate_trainability(ranker)
    return ranker, report


def validate_packed_token_count(actual_tokens: int) -> int:
    if type(actual_tokens) is not int or actual_tokens <= 0:
        raise ValueError("Packed token count must be a positive integer")
    if actual_tokens > MAX_PACKED_TOKENS:
        raise ValueError(
            f"Packed event requires {actual_tokens} tokens, exceeding frozen "
            f"Phase-3B ceiling {MAX_PACKED_TOKENS}; truncation is forbidden"
        )
    return actual_tokens


def tokenize_single_smoke_event(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    physical_block_positions: Sequence[int] | None = None,
) -> tuple[PackedScoringEvent, PackedScoringBatch, int]:
    event = tokenize_scoring_event(
        record,
        tokenizer,
        physical_block_positions=physical_block_positions,
        max_sequence_length=MAX_PACKED_TOKENS,
    )
    actual_tokens = validate_packed_token_count(int(event.input_ids.shape[0]))
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("Production tokenizer must expose a nonnegative pad token ID")
    batch = collate_scoring_events([event], pad_token_id=pad_token_id)
    if batch.input_ids.shape != (1, actual_tokens):
        raise RuntimeError("Batch-size-1 smoke event was unexpectedly padded")
    return event, batch, actual_tokens


def stream_record_by_instance_key(
    dataset_path: str | Path,
    instance_key: str,
    *,
    expected_sha256: str = EXPECTED_DATASET_SHA256,
) -> tuple[dict[str, Any], str]:
    source = Path(dataset_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    observed_sha = sha256_file(source)
    if observed_sha.lower() != expected_sha256.lower():
        raise ValueError(
            "Frozen Stage-2 v2 dataset SHA256 mismatch: "
            f"expected={expected_sha256.lower()} observed={observed_sha.lower()}"
        )
    if not isinstance(instance_key, str) or not instance_key:
        raise ValueError("instance_key must be a nonempty string")
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank dataset line at {line_number}")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at dataset line {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Dataset line {line_number} is not an object")
            if record.get("instance_key") == instance_key:
                return record, observed_sha
    raise KeyError(f"Instance key not found in frozen dataset: {instance_key}")


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _candidate_hidden_states(
    base_model: nn.Module,
    batch: PackedScoringBatch,
) -> torch.Tensor:
    outputs = base_model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
        use_cache=False,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise RuntimeError("Qwen base did not return last_hidden_state")
    gather = batch.score_token_indices.to(hidden.device).unsqueeze(-1)
    gather = gather.expand(-1, -1, hidden.shape[-1])
    selected = hidden.gather(1, gather)
    if selected.shape[:2] != (1, TOP_K) or not torch.isfinite(selected).all():
        raise RuntimeError("Permutation check recovered invalid scoring representations")
    return selected


def real_model_permutation_check(
    ranker: JointRRFRanker,
    normal_batch: PackedScoringBatch,
    reversed_batch: PackedScoringBatch,
) -> dict[str, Any]:
    """Compare non-vacuous hidden states, never zero-head residuals."""

    if normal_batch.input_ids.device != reversed_batch.input_ids.device:
        raise ValueError("Permutation batches must share a device")
    device = normal_batch.input_ids.device
    was_training = ranker.training
    ranker.eval()
    try:
        with torch.no_grad(), _autocast_context(device):
            normal = _candidate_hidden_states(ranker.base_model, normal_batch)
            reversed_hidden = _candidate_hidden_states(
                ranker.base_model,
                reversed_batch,
            )
        normal = normal.float().cpu()
        reversed_hidden = reversed_hidden.float().cpu()
    finally:
        ranker.train(was_training)
    differences = (normal - reversed_hidden).abs()
    cosine = F.cosine_similarity(normal, reversed_hidden, dim=-1)
    if not torch.isfinite(differences).all() or not torch.isfinite(cosine).all():
        raise RuntimeError("Permutation comparison produced non-finite metrics")
    result = {
        "maximum_absolute_difference": float(differences.max().item()),
        "mean_absolute_difference": float(differences.mean().item()),
        "mean_cosine_similarity": float(cosine.mean().item()),
        "minimum_cosine_similarity": float(cosine.min().item()),
        "passed": bool(float(cosine.min().item()) >= PERMUTATION_MINIMUM_COSINE),
        "required_minimum_cosine_similarity": PERMUTATION_MINIMUM_COSINE,
        "version": PERMUTATION_CHECK_VERSION,
    }
    if not result["passed"]:
        raise RuntimeError(f"Real-model physical permutation check failed: {result}")
    return result


def smoke_only_centered_contrast_objective(residuals: torch.Tensor) -> torch.Tensor:
    """Return a plumbing-only scalar; this is NOT the scientific v2 loss."""

    centered = mean_center_residuals(residuals)
    if centered.ndim != 2 or centered.shape[0] != 1:
        raise ValueError("Smoke-only objective requires one event with 50 residuals")
    objective = centered[0, 0] - centered[0, 1]
    if not torch.isfinite(objective):
        raise RuntimeError("Smoke-only objective is non-finite")
    return objective


def _gradient_group(name: str) -> str:
    if name.startswith("scoring_head."):
        return "scoring_head"
    if ".lora_A." in name:
        return "lora_A"
    if ".lora_B." in name:
        return "lora_B"
    return "frozen_base"


def gradient_norm_report(ranker: JointRRFRanker) -> dict[str, Any]:
    groups = {
        name: {
            "finite": True,
            "nonzero_parameter_tensors": 0,
            "parameter_tensors_with_grad": 0,
            "squared_norm": 0.0,
        }
        for name in ("scoring_head", "lora_A", "lora_B", "frozen_base")
    }
    for name, parameter in ranker.named_parameters():
        group = _gradient_group(name)
        gradient = parameter.grad
        if gradient is None:
            continue
        groups[group]["parameter_tensors_with_grad"] += 1
        finite = bool(torch.isfinite(gradient).all().item())
        groups[group]["finite"] = groups[group]["finite"] and finite
        norm = float(torch.linalg.vector_norm(gradient.detach().float()).item())
        if not math.isfinite(norm):
            groups[group]["finite"] = False
        if norm > 0.0:
            groups[group]["nonzero_parameter_tensors"] += 1
        groups[group]["squared_norm"] += norm * norm
    result: dict[str, Any] = {}
    for name, values in groups.items():
        result[name] = {
            "finite": values["finite"],
            "l2_norm": math.sqrt(values["squared_norm"]),
            "nonzero_parameter_tensors": values["nonzero_parameter_tensors"],
            "parameter_tensors_with_grad": values["parameter_tensors_with_grad"],
        }
    lora_squared = (
        result["lora_A"]["l2_norm"] ** 2 + result["lora_B"]["l2_norm"] ** 2
    )
    result["all_lora"] = {
        "finite": result["lora_A"]["finite"] and result["lora_B"]["finite"],
        "l2_norm": math.sqrt(lora_squared),
        "nonzero_parameter_tensors": (
            result["lora_A"]["nonzero_parameter_tensors"]
            + result["lora_B"]["nonzero_parameter_tensors"]
        ),
        "parameter_tensors_with_grad": (
            result["lora_A"]["parameter_tensors_with_grad"]
            + result["lora_B"]["parameter_tensors_with_grad"]
        ),
    }
    projection_gradient = ranker.scoring_head.projection.weight.grad
    if projection_gradient is None:
        projection_finite = True
        projection_norm = 0.0
    else:
        projection_finite = bool(torch.isfinite(projection_gradient).all().item())
        projection_norm = float(
            torch.linalg.vector_norm(projection_gradient.detach().float()).item()
        )
        projection_finite = projection_finite and math.isfinite(projection_norm)
    result["scoring_head_projection"] = {
        "finite": projection_finite,
        "l2_norm": projection_norm,
        "nonzero": projection_norm > 0.0,
    }
    return result


def validate_first_backward_gradients(report: Mapping[str, Any]) -> None:
    head = report["scoring_head"]
    projection = report["scoring_head_projection"]
    lora = report["all_lora"]
    base = report["frozen_base"]
    if not head["finite"] or float(head["l2_norm"]) <= 0.0:
        raise RuntimeError("Backward 1 must produce a finite nonzero scorer-head gradient")
    if not projection["finite"] or float(projection["l2_norm"]) <= 0.0:
        raise RuntimeError(
            "Backward 1 must produce a finite nonzero projection-weight gradient"
        )
    if not lora["finite"] or float(lora["l2_norm"]) > FIRST_BACKWARD_LORA_ZERO_TOLERANCE:
        raise RuntimeError("Backward 1 LoRA gradient violates zero-head expectation")
    if not base["finite"] or float(base["l2_norm"]) != 0.0:
        raise RuntimeError("Frozen base received a gradient during backward 1")


def validate_second_backward_gradients(report: Mapping[str, Any]) -> None:
    head = report["scoring_head"]
    projection = report["scoring_head_projection"]
    lora = report["all_lora"]
    base = report["frozen_base"]
    if not head["finite"] or float(head["l2_norm"]) <= 0.0:
        raise RuntimeError("Backward 2 must retain a finite nonzero head gradient")
    if not projection["finite"] or float(projection["l2_norm"]) <= 0.0:
        raise RuntimeError(
            "Backward 2 must retain a finite nonzero projection-weight gradient"
        )
    if not lora["finite"] or float(lora["l2_norm"]) <= 0.0:
        raise RuntimeError("Backward 2 must reach at least one LoRA parameter")
    if int(lora["nonzero_parameter_tensors"]) <= 0:
        raise RuntimeError("Backward 2 has no nonzero LoRA gradient tensor")
    if not base["finite"] or float(base["l2_norm"]) != 0.0:
        raise RuntimeError("Frozen base received a gradient during backward 2")


def run_gradient_plumbing_smoke(
    ranker: JointRRFRanker,
    batch: PackedScoringBatch,
    *,
    memory_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run exactly one smoke optimizer step and two plumbing backpropagations."""

    if torch.count_nonzero(ranker.scoring_head.projection.weight.detach()).item() != 0:
        raise RuntimeError("Smoke requires the frozen zero-initialized scoring head")
    validate_trainability(ranker)
    ranker.train()
    parameters = [parameter for parameter in ranker.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=SMOKE_LEARNING_RATE,
        weight_decay=SMOKE_WEIGHT_DECAY,
    )
    optimizer.zero_grad(set_to_none=True)
    device = batch.input_ids.device

    with _autocast_context(device):
        residuals_1 = ranker(batch)
        objective_1 = smoke_only_centered_contrast_objective(residuals_1)
    objective_1.backward()
    if memory_callback is not None:
        memory_callback("after_backward_1")
    gradients_1 = gradient_norm_report(ranker)
    validate_first_backward_gradients(gradients_1)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if memory_callback is not None:
        memory_callback("after_optimizer_step")
    projection_nonzero = bool(
        torch.count_nonzero(ranker.scoring_head.projection.weight.detach()).item() > 0
    )
    if not projection_nonzero:
        raise RuntimeError("Smoke optimizer step did not update the scorer projection")

    with _autocast_context(device):
        residuals_2 = ranker(batch)
        objective_2 = smoke_only_centered_contrast_objective(residuals_2)
    objective_2.backward()
    if memory_callback is not None:
        memory_callback("after_backward_2")
    gradients_2 = gradient_norm_report(ranker)
    validate_second_backward_gradients(gradients_2)
    validate_trainability(ranker)

    return {
        "backward_1": {
            "gradient_norms": gradients_1,
            "objective_value": float(objective_1.detach().float().cpu().item()),
            "raw_residual_dtype": str(residuals_1.dtype).removeprefix("torch."),
        },
        "backward_2": {
            "gradient_norms": gradients_2,
            "objective_value": float(objective_2.detach().float().cpu().item()),
            "raw_residual_dtype": str(residuals_2.dtype).removeprefix("torch."),
        },
        "optimizer": {
            "learning_rate": SMOKE_LEARNING_RATE,
            "name": SMOKE_OPTIMIZER,
            "projection_nonzero_after_step": projection_nonzero,
            "steps": 1,
            "weight_decay": SMOKE_WEIGHT_DECAY,
        },
        "purpose": "gradient_plumbing_only_not_scientific_training",
        "smoke_objective_version": SMOKE_OBJECTIVE_VERSION,
    }


def parameter_dtype_report(ranker: JointRRFRanker) -> dict[str, Any]:
    dtypes: dict[str, set[str]] = {
        "base": set(),
        "lora": set(),
        "scorer_head": set(),
    }
    for name, parameter in ranker.named_parameters():
        dtypes[_parameter_group(name)].add(str(parameter.dtype).removeprefix("torch."))
    return {
        "base_parameter_dtypes": sorted(dtypes["base"]),
        "lora_parameter_dtypes": sorted(dtypes["lora"]),
        "scorer_head_parameter_dtypes": sorted(dtypes["scorer_head"]),
    }


def require_single_cuda_device(device: str | torch.device) -> torch.device:
    selected = torch.device(device)
    if selected.type != "cuda" or selected.index not in {None, 0}:
        raise RuntimeError("Real-model smoke requires the single visible device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Real-model smoke requires exactly one visible GPU; set CUDA_VISIBLE_DEVICES=0"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible GPU does not support BF16")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def cuda_memory_snapshot(device: torch.device) -> dict[str, int]:
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
    }


def runtime_provenance(
    *,
    device: torch.device,
    peft_module: ModuleType | Any,
    tokenizer: Any,
    resolved_tokenizer_commit: str | None,
) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(properties.total_memory),
        "resolved_tokenizer_commit": resolved_tokenizer_commit,
        "runtime_peft_version": str(getattr(peft_module, "__version__", "unknown")),
        "runtime_torch_version": torch.__version__,
        "runtime_transformers_version": transformers.__version__,
        "tokenizer_class": (
            f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}"
        ),
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "visible_cuda_device_count": int(torch.cuda.device_count()),
    }


_FORBIDDEN_REPORT_KEYS = {
    "candidate_titles",
    "candidates",
    "conversation",
    "dialogue",
    "ground_truth",
    "ground_truth_titles",
    "history",
    "label",
    "labels",
    "target_response",
    "title",
    "titles",
}
_FORBIDDEN_REPORT_KEY_FRAGMENTS = (
    "conversation",
    "dialogue",
    "ground_truth",
    "history",
    "label",
    "target_response",
    "title",
)


def validate_smoke_report_privacy(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).lower()
            if normalized_key in _FORBIDDEN_REPORT_KEYS or any(
                fragment in normalized_key
                for fragment in _FORBIDDEN_REPORT_KEY_FRAGMENTS
            ):
                raise ValueError(f"Sensitive field is forbidden in smoke report: {key}")
            validate_smoke_report_privacy(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_smoke_report_privacy(child)


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode(
                    "utf-8"
                )
            )
            handle.write(b"\n")
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def write_smoke_report(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    validate_smoke_report_privacy(report)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / SMOKE_REPORT_FILENAME
    if path.exists():
        raise FileExistsError(f"Smoke report already exists: {path}")
    _atomic_json(path, report)
    return path


__all__ = [
    "DEFAULT_DATASET_PATH",
    "DEFAULT_SMOKE_OUTPUT_DIR",
    "EXPECTED_DATASET_SHA256",
    "EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT",
    "EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT",
    "GRADIENT_CHECKPOINTING_USE_REENTRANT",
    "LORA_ALPHA",
    "LORA_BIAS",
    "LORA_DROPOUT",
    "LORA_R",
    "LORA_TARGET_MODULES",
    "MAX_PACKED_TOKENS",
    "MODEL_ID",
    "QUANTIZATION_ENABLED",
    "REQUESTED_MODEL_REVISION",
    "SMOKE_REPORT_FILENAME",
    "TRUNCATION_ALLOWED",
    "apply_peft_and_build_ranker",
    "create_lora_config",
    "cuda_memory_snapshot",
    "gradient_norm_report",
    "load_base_qwen",
    "load_production_tokenizer",
    "parameter_dtype_report",
    "phase3b_integration_fingerprint",
    "phase3b_scientific_configuration",
    "real_model_permutation_check",
    "require_single_cuda_device",
    "run_gradient_plumbing_smoke",
    "runtime_provenance",
    "smoke_only_centered_contrast_objective",
    "stream_record_by_instance_key",
    "tokenize_single_smoke_event",
    "validate_first_backward_gradients",
    "validate_loaded_base_qwen",
    "validate_packed_token_count",
    "validate_second_backward_gradients",
    "validate_smoke_report_privacy",
    "validate_tokenizer",
    "validate_trainability",
    "write_smoke_report",
]
