"""Minimal production-capable Stage-2 v2 trainer and few-step smoke CLI.

This module trains only on the frozen Phase-1 TRAIN records.  It contains no
DEV/VALID/TEST evaluation, model selection, beta search, or final experiment
defaults; scientific hyperparameters are explicit CLI inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any, BinaryIO

import torch
import transformers
from torch import nn

from my_crs.analyze_stage2_v2_tokens import (
    DEFAULT_DATASET_PATH,
    EXPECTED_DATASET_SHA256,
)
from my_crs.build_stage2_v2_dataset import PROJECT_ROOT, TOP_K, canonical_json_bytes
from my_crs.joint_rrf_ranker import (
    JointRRFRanker,
    PackedScoringBatch,
    load_scorer_head_state_dict,
    scorer_head_state_dict,
)
from my_crs.stage2_v2_loss import (
    EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT,
    compute_v2_batch_loss,
    loss_scientific_configuration,
    loss_scientific_fingerprint,
    training_loss_inputs_from_record,
)
from my_crs.stage2_v2_peft import (
    MAX_PACKED_TOKENS,
    MODEL_ID,
    REQUESTED_MODEL_REVISION,
    TRUNCATION_ALLOWED,
    apply_peft_and_build_ranker,
    cuda_memory_snapshot,
    gradient_norm_report,
    load_base_qwen,
    load_production_tokenizer,
    phase3b_integration_fingerprint,
    require_single_cuda_device,
    tokenize_single_smoke_event,
    validate_packed_token_count,
    validate_tokenizer,
    validate_trainability,
)


TRAINER_VERSION = "stage2_v2_direct_pytorch_trainer_v1"
TRAINING_MANIFEST_SCHEMA = "stage2_v2_training_manifest_v1"
TRAINING_METRIC_SCHEMA = "stage2_v2_training_metric_v1"
CHECKPOINT_SCHEMA = "stage2_v2_training_checkpoint_v1"
SHUFFLE_POLICY_VERSION = "sha256_seed_epoch_fisher_yates_v1"

EXPECTED_DATASET_COUNTS = {"all": 22199, "train": 20055, "dev": 2144}
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "stage2_v2_training"
MANIFEST_FILENAME = "stage2_v2_training_manifest.json"
METRICS_FILENAME = "stage2_v2_training_metrics.jsonl"
CHECKPOINT_DIRECTORY = "checkpoints"


def _fingerprint(configuration: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()


def _finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be finite and nonnegative")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return number


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def training_scientific_configuration(
    *,
    beta: float,
    seed: int,
    learning_rate: float,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float,
    max_optimizer_steps: int,
) -> dict[str, Any]:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    learning_rate_value = _finite_nonnegative(learning_rate, "learning_rate")
    if learning_rate_value <= 0.0:
        raise ValueError("learning_rate must be strictly positive")
    clip_value = _finite_nonnegative(gradient_clip_norm, "gradient_clip_norm")
    if clip_value <= 0.0:
        raise ValueError("gradient_clip_norm must be strictly positive")
    accumulation = _positive_int(
        gradient_accumulation_steps,
        "gradient_accumulation_steps",
    )
    maximum_steps = _positive_int(max_optimizer_steps, "max_optimizer_steps")
    observed_phase3b = phase3b_integration_fingerprint()
    if observed_phase3b != EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT:
        raise RuntimeError("Frozen Phase-3B integration fingerprint mismatch")
    return {
        "batch_size": 1,
        "dataset": {
            "counts": dict(EXPECTED_DATASET_COUNTS),
            "sha256": EXPECTED_DATASET_SHA256,
            "training_split": "train",
        },
        "gradient_accumulation_steps": accumulation,
        "gradient_clip_norm": clip_value,
        "loss": loss_scientific_configuration(beta),
        "loss_fingerprint": loss_scientific_fingerprint(beta),
        "max_optimizer_steps": maximum_steps,
        "max_packed_tokens": MAX_PACKED_TOKENS,
        "model_id": MODEL_ID,
        "optimizer": {
            "learning_rate": learning_rate_value,
            "name": "AdamW",
            "parameters": "trainable_only",
            "weight_decay": 0.0,
        },
        "phase3b_integration_fingerprint": observed_phase3b,
        "requested_model_revision": REQUESTED_MODEL_REVISION,
        "scheduler": None,
        "seed": seed,
        "shuffle_policy_version": SHUFFLE_POLICY_VERSION,
        "trainer_version": TRAINER_VERSION,
        "truncation": TRUNCATION_ALLOWED,
    }


def training_scientific_fingerprint(**kwargs: Any) -> str:
    return _fingerprint(training_scientific_configuration(**kwargs))


@dataclass(frozen=True)
class TrainOffsetEntry:
    byte_offset: int
    byte_length: int
    instance_key: str
    split: str = "train"


@dataclass(frozen=True)
class TrainJsonlOffsetIndex:
    path: Path
    dataset_sha256: str
    counts: dict[str, int]
    entries: tuple[TrainOffsetEntry, ...]

    def read_record(
        self,
        index: int,
        *,
        handle: BinaryIO | None = None,
    ) -> dict[str, Any]:
        if type(index) is not int or not 0 <= index < len(self.entries):
            raise IndexError("TRAIN offset index is outside the dataset")
        entry = self.entries[index]
        owns_handle = handle is None
        stream = handle or self.path.open("rb")
        try:
            stream.seek(entry.byte_offset)
            line = stream.read(entry.byte_length)
        finally:
            if owns_handle:
                stream.close()
        if len(line) != entry.byte_length:
            raise RuntimeError("Indexed TRAIN record is truncated")
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Indexed TRAIN record is not valid UTF-8 JSON") from error
        if not isinstance(record, dict):
            raise RuntimeError("Indexed TRAIN record is not an object")
        if record.get("instance_key") != entry.instance_key:
            raise RuntimeError("Indexed TRAIN instance identity changed")
        if record.get("split") != "train":
            raise RuntimeError("Indexed training record no longer has split=train")
        return record


def build_train_offset_index(
    dataset_path: str | Path,
    *,
    expected_sha256: str = EXPECTED_DATASET_SHA256,
    expected_counts: Mapping[str, int] = EXPECTED_DATASET_COUNTS,
) -> TrainJsonlOffsetIndex:
    """Hash, validate, and index the JSONL in one binary streaming pass."""

    source = Path(dataset_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    counts = {"all": 0, "train": 0, "dev": 0}
    entries: list[TrainOffsetEntry] = []
    instance_keys: set[str] = set()
    with source.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                raise ValueError(f"Blank dataset line at byte offset {offset}")
            digest.update(line)
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid dataset JSON at byte offset {offset}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Dataset record at byte offset {offset} is not an object")
            split = record.get("split")
            if split not in {"train", "dev"}:
                raise ValueError(f"Dataset record at byte offset {offset} has invalid split")
            instance_key = record.get("instance_key")
            if not isinstance(instance_key, str) or not instance_key:
                raise ValueError(f"Dataset record at byte offset {offset} lacks instance key")
            if instance_key in instance_keys:
                raise ValueError(f"Duplicate dataset instance key: {instance_key}")
            instance_keys.add(instance_key)
            candidates = record.get("candidates")
            if (
                record.get("candidate_count") != TOP_K
                or not isinstance(candidates, list)
                or len(candidates) != TOP_K
            ):
                raise ValueError(f"Dataset instance {instance_key} must contain 50 candidates")
            counts["all"] += 1
            counts[str(split)] += 1
            if split == "train":
                entries.append(
                    TrainOffsetEntry(
                        byte_offset=offset,
                        byte_length=len(line),
                        instance_key=instance_key,
                    )
                )
    observed_sha = digest.hexdigest()
    if observed_sha.lower() != expected_sha256.lower():
        raise ValueError(
            "Frozen Stage-2 v2 dataset SHA256 mismatch: "
            f"expected={expected_sha256.lower()} observed={observed_sha.lower()}"
        )
    required_counts = {
        scope: int(expected_counts[scope]) for scope in ("all", "train", "dev")
    }
    if counts != required_counts:
        raise ValueError(f"Stage-2 v2 dataset accounting mismatch: {counts} != {required_counts}")
    if len(entries) != counts["train"]:
        raise RuntimeError("TRAIN offset count disagrees with split accounting")
    return TrainJsonlOffsetIndex(source, observed_sha, counts, tuple(entries))


def deterministic_epoch_order(length: int, *, seed: int, epoch: int) -> list[int]:
    if type(length) is not int or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if type(seed) is not int or seed < 0 or type(epoch) is not int or epoch < 0:
        raise ValueError("seed and epoch must be nonnegative integers")
    material = f"{SHUFFLE_POLICY_VERSION}|{seed}|{epoch}".encode("utf-8")
    epoch_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    order = list(range(length))
    random.Random(epoch_seed).shuffle(order)
    return order


@dataclass
class TrainingState:
    epoch: int = 0
    next_epoch_position: int = 0
    optimizer_step: int = 0
    events_processed: int = 0
    nonzero_lora_gradient_observed: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingState":
        fields: dict[str, int] = {}
        for name in ("epoch", "next_epoch_position", "optimizer_step", "events_processed"):
            item = value.get(name)
            if type(item) is not int or item < 0:
                raise ValueError(f"Checkpoint training state has invalid {name}")
            fields[name] = item
        observed = value.get("nonzero_lora_gradient_observed")
        if type(observed) is not bool:
            raise ValueError(
                "Checkpoint training state has invalid nonzero_lora_gradient_observed"
            )
        return cls(**fields, nonzero_lora_gradient_observed=observed)


@dataclass
class AccumulationWindow:
    target_events: int
    event_count: int = 0
    positive_count: int = 0
    total_loss_sum: float = 0.0
    partial_loss_sum: float = 0.0
    anchor_loss_sum: float = 0.0
    token_count_sum: int = 0
    maximum_tokens: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.target_events, "target_events")

    def observe(
        self,
        *,
        total_loss: float,
        partial_loss: float,
        anchor_loss: float,
        positive: bool,
        actual_tokens: int,
    ) -> None:
        if self.event_count >= self.target_events:
            raise RuntimeError("Gradient accumulation window is already full")
        values = (total_loss, partial_loss, anchor_loss)
        if any(not math.isfinite(float(value)) for value in values):
            raise RuntimeError("Accumulation window received a non-finite loss")
        validate_packed_token_count(actual_tokens)
        self.event_count += 1
        self.total_loss_sum += float(total_loss)
        self.anchor_loss_sum += float(anchor_loss)
        self.token_count_sum += actual_tokens
        self.maximum_tokens = max(self.maximum_tokens, actual_tokens)
        if positive:
            self.positive_count += 1
            self.partial_loss_sum += float(partial_loss)

    @property
    def ready(self) -> bool:
        return self.event_count == self.target_events

    def tail_gradient_rescale(self) -> float:
        if self.event_count <= 0:
            raise RuntimeError("Cannot step an empty accumulation window")
        return self.target_events / self.event_count

    def metrics(self) -> dict[str, int | float]:
        if self.event_count <= 0:
            raise RuntimeError("Cannot summarize an empty accumulation window")
        return {
            "anchor_loss": self.anchor_loss_sum / self.event_count,
            "events": self.event_count,
            "maximum_packed_tokens": self.maximum_tokens,
            "mean_packed_tokens": self.token_count_sum / self.event_count,
            "partial_loss": (
                self.partial_loss_sum / self.positive_count
                if self.positive_count
                else 0.0
            ),
            "positive_event_rate": self.positive_count / self.event_count,
            "total_loss": self.total_loss_sum / self.event_count,
        }


@dataclass(frozen=True)
class PreparedTrainingEvent:
    batch: PackedScoringBatch
    rrf_scores: tuple[float, ...]
    positive_positions: tuple[int, ...]
    actual_tokens: int
    instance_key: str


def prepare_training_event(record: Mapping[str, Any], tokenizer: Any) -> PreparedTrainingEvent:
    inputs = training_loss_inputs_from_record(record)
    event, batch, actual_tokens = tokenize_single_smoke_event(record, tokenizer)
    validate_packed_token_count(actual_tokens)
    if tuple(float(value) for value in event.rrf_scores) != inputs.rrf_scores:
        raise RuntimeError("Tokenized event RRF scores disagree with frozen loss inputs")
    return PreparedTrainingEvent(
        batch=batch,
        rrf_scores=inputs.rrf_scores,
        positive_positions=inputs.positive_serialization_positions,
        actual_tokens=actual_tokens,
        instance_key=inputs.instance_key,
    )


def load_phase3b_training_stack(
    device: torch.device,
) -> tuple[Any, JointRRFRanker, dict[str, Any], dict[str, Any], str | None]:
    tokenizer = load_production_tokenizer()
    resolved_tokenizer_commit = validate_tokenizer(tokenizer)
    base_model, model_identity = load_base_qwen(device)
    ranker, trainability = apply_peft_and_build_ranker(base_model, device)
    validate_trainability(ranker)
    return tokenizer, ranker, model_identity, trainability, resolved_tokenizer_commit


def create_trainable_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
) -> torch.optim.AdamW:
    rate = _finite_nonnegative(learning_rate, "learning_rate")
    if rate <= 0.0:
        raise ValueError("learning_rate must be strictly positive")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("Training model exposes no trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=rate, weight_decay=0.0)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {id(parameter) for parameter in trainable}
    if optimizer_ids != expected_ids:
        raise RuntimeError("Optimizer parameter set is not exactly trainable parameters")
    return optimizer


def validate_finite_trainable_gradients(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite trainable gradient: {name}")


def _scale_trainable_gradients(model: nn.Module, factor: float) -> None:
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("Gradient rescale factor must be finite and positive")
    if factor == 1.0:
        return
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            parameter.grad.mul_(factor)


def _cpu_tensor_mapping(value: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().cpu().clone() for key, tensor in value.items()}


def _rng_state() -> dict[str, Any]:
    return {
        "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    torch_state = value.get("torch")
    python_state = value.get("python")
    cuda_states = value.get("cuda")
    if not isinstance(torch_state, torch.Tensor) or not isinstance(cuda_states, list):
        raise ValueError("Checkpoint RNG state is malformed")
    random.setstate(python_state)
    torch.set_rng_state(torch_state.cpu())
    torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _atomic_torch_save(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        torch.save(value, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def save_checkpoint_payload(
    path: str | Path,
    *,
    adapter_state: Mapping[str, torch.Tensor],
    scorer_head_state: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
    training_state: TrainingState,
    scientific_configuration: Mapping[str, Any],
    scientific_fingerprint: str,
    rng_state: Mapping[str, Any],
) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    payload = {
        "adapter_state": _cpu_tensor_mapping(adapter_state),
        "checkpoint_contents": [
            "lora_adapter",
            "shared_scorer_head",
            "optimizer",
            "rng",
            "training_state",
            "scientific_configuration",
        ],
        "optimizer_state": dict(optimizer_state),
        "rng_state": dict(rng_state),
        "schema_version": CHECKPOINT_SCHEMA,
        "scientific_configuration": dict(scientific_configuration),
        "scientific_fingerprint": scientific_fingerprint,
        "scorer_head_state": _cpu_tensor_mapping(scorer_head_state),
        "training_state": asdict(training_state),
    }
    _atomic_torch_save(destination, payload)
    return destination


def load_checkpoint_payload(
    path: str | Path,
    *,
    expected_scientific_fingerprint: str,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("Stage-2 v2 checkpoint schema mismatch")
    if payload.get("scientific_fingerprint") != expected_scientific_fingerprint:
        raise ValueError("Stage-2 v2 checkpoint scientific fingerprint mismatch")
    required = {
        "adapter_state",
        "optimizer_state",
        "rng_state",
        "scientific_configuration",
        "scorer_head_state",
        "training_state",
    }
    if not required.issubset(payload):
        raise ValueError("Stage-2 v2 checkpoint is incomplete")
    TrainingState.from_mapping(payload["training_state"])
    return payload


def save_training_checkpoint(
    *,
    path: Path,
    ranker: JointRRFRanker,
    optimizer: torch.optim.Optimizer,
    training_state: TrainingState,
    scientific_configuration: Mapping[str, Any],
    scientific_fingerprint: str,
    peft_module: Any,
) -> Path:
    adapter_state = peft_module.get_peft_model_state_dict(ranker.base_model)
    return save_checkpoint_payload(
        path,
        adapter_state=adapter_state,
        scorer_head_state=scorer_head_state_dict(ranker),
        optimizer_state=optimizer.state_dict(),
        training_state=training_state,
        scientific_configuration=scientific_configuration,
        scientific_fingerprint=scientific_fingerprint,
        rng_state=_rng_state(),
    )


def restore_training_checkpoint(
    *,
    path: str | Path,
    ranker: JointRRFRanker,
    optimizer: torch.optim.Optimizer,
    expected_scientific_configuration: Mapping[str, Any],
    expected_scientific_fingerprint: str,
    device: torch.device,
    peft_module: Any,
) -> TrainingState:
    payload = load_checkpoint_payload(
        path,
        expected_scientific_fingerprint=expected_scientific_fingerprint,
        map_location=device,
    )
    if payload["scientific_configuration"] != dict(expected_scientific_configuration):
        raise ValueError("Checkpoint scientific configuration changed")
    peft_module.set_peft_model_state_dict(
        ranker.base_model,
        payload["adapter_state"],
    )
    load_scorer_head_state_dict(ranker, payload["scorer_head_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    state = TrainingState.from_mapping(payload["training_state"])
    _restore_rng_state(payload["rng_state"])
    validate_trainability(ranker)
    return state


_FORBIDDEN_ARTIFACT_KEYS = {
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
_FORBIDDEN_ARTIFACT_KEY_FRAGMENTS = (
    "conversation_text",
    "dialogue_text",
    "ground_truth",
    "history_text",
    "target_response",
    "title_text",
)


def validate_training_artifact_privacy(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_ARTIFACT_KEYS or any(
                fragment in normalized
                for fragment in _FORBIDDEN_ARTIFACT_KEY_FRAGMENTS
            ):
                raise ValueError(f"Sensitive field is forbidden in training artifact: {key}")
            validate_training_artifact_privacy(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            validate_training_artifact_privacy(child)


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


def write_training_metric(path: Path, metric: Mapping[str, Any]) -> None:
    validate_training_artifact_privacy(metric)
    encoded = canonical_json_bytes(metric) + b"\n"
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _last_metric_step(path: Path, expected_fingerprint: str) -> int:
    last_step = 0
    if not path.exists():
        return last_step
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank training metric at line {line_number}")
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"Training metric at line {line_number} is not an object")
            if record.get("scientific_fingerprint") != expected_fingerprint:
                raise ValueError("Training metric scientific fingerprint mismatch")
            step = record.get("optimizer_step")
            if type(step) is not int or step != last_step + 1:
                raise ValueError("Training metric steps are not contiguous")
            last_step = step
    return last_step


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _checkpoint_path(output_dir: Path, optimizer_step: int) -> Path:
    return output_dir / CHECKPOINT_DIRECTORY / f"checkpoint_step_{optimizer_step:08d}.pt"


def train_stage2_v2(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    beta: float,
    max_optimizer_steps: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    gradient_clip_norm: float,
    checkpoint_every_steps: int,
    device_name: str,
    seed: int,
    resume_checkpoint: str | Path | None = None,
    require_lora_gradient: bool = False,
) -> dict[str, Any]:
    """Run direct PyTorch training; callers control the explicit step budget."""

    checkpoint_interval = _positive_int(checkpoint_every_steps, "checkpoint_every_steps")
    scientific_configuration = training_scientific_configuration(
        beta=beta,
        seed=seed,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip_norm=gradient_clip_norm,
        max_optimizer_steps=max_optimizer_steps,
    )
    scientific_fingerprint = _fingerprint(scientific_configuration)
    destination = Path(output_dir).resolve()
    manifest_path = destination / MANIFEST_FILENAME
    metrics_path = destination / METRICS_FILENAME
    checkpoint_dir = destination / CHECKPOINT_DIRECTORY
    existing_manifest: dict[str, Any] | None = None
    if resume_checkpoint is None:
        collisions = [
            path
            for path in (manifest_path, metrics_path, checkpoint_dir)
            if path.exists()
        ]
        if collisions:
            raise FileExistsError(f"Training output already exists: {collisions}")
    else:
        if not manifest_path.is_file():
            raise FileNotFoundError("Resume requires the existing training manifest")
        loaded_manifest = json.loads(manifest_path.read_text("utf-8"))
        if not isinstance(loaded_manifest, dict):
            raise ValueError("Resume training manifest must be an object")
        existing_manifest = loaded_manifest
        if existing_manifest.get("scientific_fingerprint") != scientific_fingerprint:
            raise ValueError("Resume manifest scientific fingerprint mismatch")
        if existing_manifest.get("scientific_configuration") != scientific_configuration:
            raise ValueError("Resume manifest scientific configuration mismatch")

    index = build_train_offset_index(dataset_path)
    device = require_single_cuda_device(device_name)
    _set_deterministic_seed(seed)
    tokenizer, ranker, model_identity, trainability, resolved_tokenizer_commit = (
        load_phase3b_training_stack(device)
    )
    optimizer = create_trainable_optimizer(ranker, learning_rate=learning_rate)
    import peft

    if resume_checkpoint is None:
        destination.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        runtime = {
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "resolved_tokenizer_commit": resolved_tokenizer_commit,
            "runtime_peft_version": str(getattr(peft, "__version__", "unknown")),
            "runtime_torch_version": torch.__version__,
            "runtime_transformers_version": transformers.__version__,
        }
        manifest = {
            "checkpoint_contents": [
                "lora_adapter",
                "shared_scorer_head",
                "optimizer",
                "rng",
                "training_state",
                "scientific_configuration",
            ],
            "checkpoint_every_steps": checkpoint_interval,
            "dataset": {
                "counts": index.counts,
                "path": str(index.path),
                "sha256": index.dataset_sha256,
            },
            "model_identity": model_identity,
            "runtime_provenance": runtime,
            "schema_version": TRAINING_MANIFEST_SCHEMA,
            "scientific_configuration": scientific_configuration,
            "scientific_fingerprint": scientific_fingerprint,
            "trainability": trainability,
        }
        validate_training_artifact_privacy(manifest)
        _atomic_json(manifest_path, manifest)
        state = TrainingState()
    else:
        state = restore_training_checkpoint(
            path=resume_checkpoint,
            ranker=ranker,
            optimizer=optimizer,
            expected_scientific_configuration=scientific_configuration,
            expected_scientific_fingerprint=scientific_fingerprint,
            device=device,
            peft_module=peft,
        )
        if _last_metric_step(metrics_path, scientific_fingerprint) != state.optimizer_step:
            raise ValueError("Resume metrics do not end at checkpoint optimizer step")

    maximum_steps = int(scientific_configuration["max_optimizer_steps"])
    if state.optimizer_step > maximum_steps:
        raise ValueError("Checkpoint optimizer step exceeds configured maximum")
    accumulation_target = int(scientific_configuration["gradient_accumulation_steps"])
    clip_norm = float(scientific_configuration["gradient_clip_norm"])
    beta_value = float(scientific_configuration["loss"]["beta"])
    optimizer.zero_grad(set_to_none=True)
    latest_checkpoint = (
        Path(resume_checkpoint).resolve() if resume_checkpoint is not None else None
    )

    with index.path.open("rb") as dataset_handle:
        while state.optimizer_step < maximum_steps:
            order = deterministic_epoch_order(
                len(index.entries),
                seed=seed,
                epoch=state.epoch,
            )
            if state.next_epoch_position > len(order):
                raise ValueError("Checkpoint epoch position exceeds TRAIN index")
            window = AccumulationWindow(accumulation_target)
            while (
                state.next_epoch_position < len(order)
                and state.optimizer_step < maximum_steps
            ):
                record_index = order[state.next_epoch_position]
                record = index.read_record(record_index, handle=dataset_handle)
                prepared = prepare_training_event(record, tokenizer)
                batch = prepared.batch.to(device)
                rrf_scores = torch.tensor(
                    [prepared.rrf_scores],
                    dtype=torch.float64,
                    device=device,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    residuals = ranker(batch)
                    loss = compute_v2_batch_loss(
                        rrf_scores,
                        residuals,
                        [prepared.positive_positions],
                        beta=beta_value,
                    )
                    scaled_loss = loss.total_loss / accumulation_target
                if not torch.isfinite(loss.total_loss):
                    raise RuntimeError("Stage-2 v2 total loss is non-finite")
                scaled_loss.backward()
                validate_finite_trainable_gradients(ranker)

                positive = bool(prepared.positive_positions)
                window.observe(
                    total_loss=float(loss.total_loss.detach().cpu()),
                    partial_loss=float(
                        loss.mean_partial_over_positive_events.detach().cpu()
                    ),
                    anchor_loss=float(loss.mean_anchor_over_all_events.detach().cpu()),
                    positive=positive,
                    actual_tokens=prepared.actual_tokens,
                )
                state.next_epoch_position += 1
                state.events_processed += 1
                end_of_epoch = state.next_epoch_position == len(order)
                if not window.ready and not end_of_epoch:
                    continue

                _scale_trainable_gradients(ranker, window.tail_gradient_rescale())
                validate_finite_trainable_gradients(ranker)
                gradient_groups = gradient_norm_report(ranker)
                if not all(
                    bool(group["finite"])
                    for group in gradient_groups.values()
                    if isinstance(group, Mapping) and "finite" in group
                ):
                    raise RuntimeError("Stage-2 v2 gradient group report is non-finite")
                if float(gradient_groups["frozen_base"]["l2_norm"]) != 0.0:
                    raise RuntimeError("Frozen Qwen base received a training gradient")
                if float(gradient_groups["all_lora"]["l2_norm"]) > 0.0:
                    state.nonzero_lora_gradient_observed = True
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in ranker.parameters() if parameter.requires_grad],
                    max_norm=clip_norm,
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Stage-2 v2 gradient norm is non-finite")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state.optimizer_step += 1
                if end_of_epoch:
                    state.epoch += 1
                    state.next_epoch_position = 0

                should_checkpoint = (
                    state.optimizer_step % checkpoint_interval == 0
                    or state.optimizer_step == maximum_steps
                )
                checkpoint_reference: str | None = None
                if should_checkpoint:
                    latest_checkpoint = save_training_checkpoint(
                        path=_checkpoint_path(destination, state.optimizer_step),
                        ranker=ranker,
                        optimizer=optimizer,
                        training_state=state,
                        scientific_configuration=scientific_configuration,
                        scientific_fingerprint=scientific_fingerprint,
                        peft_module=peft,
                    )
                    checkpoint_reference = latest_checkpoint.relative_to(destination).as_posix()
                metric = {
                    **window.metrics(),
                    "checkpoint_reference": checkpoint_reference,
                    "cuda_memory": cuda_memory_snapshot(device),
                    "epoch": state.epoch,
                    "gradient_norm_before_clipping": float(
                        gradient_norm.detach().float().cpu()
                    ),
                    "gradient_norms": gradient_groups,
                    "next_epoch_position": state.next_epoch_position,
                    "optimizer_step": state.optimizer_step,
                    "schema_version": TRAINING_METRIC_SCHEMA,
                    "scientific_fingerprint": scientific_fingerprint,
                }
                write_training_metric(metrics_path, metric)
                window = AccumulationWindow(accumulation_target)
                if end_of_epoch:
                    break

            if state.next_epoch_position == len(order):
                state.epoch += 1
                state.next_epoch_position = 0

    if require_lora_gradient and not state.nonzero_lora_gradient_observed:
        raise RuntimeError(
            "Few-step smoke did not yet observe a nonzero LoRA gradient; "
            "rerun in a new output directory with a larger explicit smoke budget"
        )
    return {
        "events_processed": state.events_processed,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "optimizer_steps": state.optimizer_step,
        "nonzero_lora_gradient_observed": state.nonzero_lora_gradient_observed,
        "scientific_fingerprint": scientific_fingerprint,
        "status": "step_budget_complete",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--max-optimizer-steps", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--gradient-clip-norm", type=float, required=True)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--require-lora-gradient", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_stage2_v2(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        beta=args.beta,
        max_optimizer_steps=args.max_optimizer_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        gradient_clip_norm=args.gradient_clip_norm,
        checkpoint_every_steps=args.checkpoint_every_steps,
        device_name=args.device,
        seed=args.seed,
        resume_checkpoint=args.resume_checkpoint,
        require_lora_gradient=args.require_lora_gradient,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
