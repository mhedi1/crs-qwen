"""Stage-2 v2 independent-block Qwen scorer and frozen-RRF helpers.

This module is isolated from retrieval, training, and evaluation.  It consumes
only frozen Phase-1 records and deliberately keeps RRF provenance outside the
model-visible text and contextual scoring head.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from my_crs.build_stage2_v2_dataset import (
    CANDIDATE_ORDER_VERSION,
    DATASET_SCHEMA_VERSION,
    TOP_K,
    canonical_json_digest,
    sanitize_title,
)


INPUT_SERIALIZATION_VERSION = "stage2_v2_joint_input_v1"
TOKEN_BOUNDARY_POLICY_VERSION = "segmentwise_offsets_no_special_tokens_v1"
MASK_POLICY_VERSION = "qwen2_sdpa_4d_independent_candidate_blocks_v1"
POSITION_ID_POLICY_VERSION = "common_causal_blocks_restart_after_prefix_v1"
SCORER_ARCHITECTURE_VERSION = "shared_layernorm_linear_zero_init_v1"
RRF_PRIOR_POLICY_VERSION = "normalized_log_rrf_plus_mean_centered_residual_v1"
RANKING_POLICY_VERSION = "score_desc_rrf_rank_entity_id_v1"

REQUIRED_MODEL_TYPE = "qwen2"
REQUIRED_ATTENTION_BACKEND = "sdpa"
TESTED_TRANSFORMERS_VERSION = "5.8.0"
TESTED_TORCH_VERSION = "2.6.0+cu124"

SYSTEM_INSTRUCTION = """You are an internal contextual movie-candidate scorer.
Candidate identifiers are arbitrary and do not indicate recommendation quality.
Use only the pre-target dialogue and candidate titles.
No natural-language response is required."""

SCORING_MARKER = "Contextual fit:"


def phase2_architecture_configuration() -> dict[str, Any]:
    """Return the versioned, fingerprintable Phase-2 architecture contract."""

    return {
        "input_serialization_version": INPUT_SERIALIZATION_VERSION,
        "token_boundary_policy_version": TOKEN_BOUNDARY_POLICY_VERSION,
        "mask_policy_version": MASK_POLICY_VERSION,
        "position_id_policy_version": POSITION_ID_POLICY_VERSION,
        "scorer_architecture_version": SCORER_ARCHITECTURE_VERSION,
        "rrf_prior_policy_version": RRF_PRIOR_POLICY_VERSION,
        "ranking_policy_version": RANKING_POLICY_VERSION,
        "candidate_count": TOP_K,
        "required_model_type": REQUIRED_MODEL_TYPE,
        "required_attention_backend": REQUIRED_ATTENTION_BACKEND,
        "tested_transformers_version": TESTED_TRANSFORMERS_VERSION,
        "tested_torch_version": TESTED_TORCH_VERSION,
        "gamma_gate": False,
        "rrf_in_model_text": False,
    }


def phase2_architecture_fingerprint() -> str:
    payload = json.dumps(
        phase2_architecture_configuration(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer")
    return value


def _finite_float(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{field} must be strictly positive")
    return number


def canonicalize_phase1_candidates(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and canonicalize a frozen Phase-1 event's 50 candidates."""

    if record.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("Stage-2 v2 record schema mismatch")
    if record.get("candidate_count") != TOP_K:
        raise ValueError(f"Stage-2 v2 record must declare {TOP_K} candidates")
    if record.get("serialization_order_version") != CANDIDATE_ORDER_VERSION:
        raise ValueError("Stage-2 v2 candidate-order version mismatch")
    history = record.get("history")
    if not isinstance(history, str):
        raise ValueError("Stage-2 v2 history must be a string")
    expected_history_sha = hashlib.sha256(history.encode("utf-8")).hexdigest()
    if record.get("history_sha256") != expected_history_sha:
        raise ValueError("Stage-2 v2 history SHA256 mismatch")

    value = record.get("candidates")
    if not isinstance(value, list) or len(value) != TOP_K:
        raise ValueError(f"Stage-2 v2 record must contain exactly {TOP_K} candidates")
    candidates: list[dict[str, Any]] = []
    for input_index, candidate in enumerate(value, 1):
        if not isinstance(candidate, Mapping):
            raise ValueError(f"candidate {input_index} must be an object")
        item = dict(candidate)
        position = _strict_int(
            item.get("serialization_position"),
            f"candidate {input_index}.serialization_position",
        )
        if not 1 <= position <= TOP_K:
            raise ValueError("Candidate serialization position is outside 1..50")
        expected_local_id = f"C{position:02d}"
        if item.get("local_id") != expected_local_id:
            raise ValueError(f"Candidate local ID must be {expected_local_id}")
        _strict_int(item.get("canonical_entity_id"), "candidate.canonical_entity_id")
        rrf_rank = _strict_int(item.get("rrf_rank"), "candidate.rrf_rank")
        if not 1 <= rrf_rank <= TOP_K:
            raise ValueError("Candidate RRF rank is outside 1..50")
        _finite_float(item.get("rrf_score"), "candidate.rrf_score", positive=True)
        original_title = item.get("title_original")
        sanitized_title = item.get("title_sanitized")
        if not isinstance(original_title, str) or not original_title.strip():
            raise ValueError("Candidate original title must be nonempty")
        if not isinstance(sanitized_title, str) or not sanitized_title:
            raise ValueError("Candidate sanitized title must be nonempty")
        if sanitized_title != sanitize_title(original_title):
            raise ValueError("Candidate title sanitizer provenance mismatch")
        candidates.append(item)

    candidates.sort(key=lambda item: int(item["serialization_position"]))
    positions = [int(item["serialization_position"]) for item in candidates]
    if positions != list(range(1, TOP_K + 1)):
        raise ValueError("Candidate serialization positions must be exactly 1..50")
    canonical_ids = [int(item["canonical_entity_id"]) for item in candidates]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("Candidate canonical entity IDs must be unique")
    rrf_ranks = [int(item["rrf_rank"]) for item in candidates]
    if sorted(rrf_ranks) != list(range(1, TOP_K + 1)):
        raise ValueError("Original RRF ranks must be exactly 1..50")
    if record.get("serialization_digest") != canonical_json_digest(candidates):
        raise ValueError("Stage-2 v2 candidate serialization digest mismatch")
    return candidates


def _common_prefix(history: str, candidates: Sequence[Mapping[str, Any]]) -> str:
    candidate_lines = "\n".join(
        f'{candidate["local_id"]} | {candidate["title_sanitized"]}'
        for candidate in candidates
    )
    return (
        "SYSTEM:\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        "USER:\n"
        "[PRE-TARGET DIALOGUE]\n"
        f"{history}\n"
        "[END PRE-TARGET DIALOGUE]\n\n"
        "[CANDIDATE SET]\n"
        f"{candidate_lines}\n"
        "[END CANDIDATE SET]\n\n"
        "[INDEPENDENT SCORING QUERIES]\n"
    )


def _scoring_block(candidate: Mapping[str, Any]) -> str:
    return (
        "\n<SCORING_QUERY>\n"
        f'Candidate title: {candidate["title_sanitized"]}\n'
        f"{SCORING_MARKER}"
    )


@dataclass(frozen=True)
class LogicalScoringInput:
    common_prefix: str
    scoring_blocks: tuple[str, ...]
    physical_block_positions: tuple[int, ...]
    full_text: str
    candidates: tuple[dict[str, Any], ...]


def build_logical_scoring_input(
    record: Mapping[str, Any],
    *,
    physical_block_positions: Sequence[int] | None = None,
) -> LogicalScoringInput:
    """Create model-visible text without retrieval or label provenance."""

    candidates = canonicalize_phase1_candidates(record)
    if physical_block_positions is None:
        physical_order = tuple(range(1, TOP_K + 1))
    else:
        physical_order = tuple(
            _strict_int(position, "physical_block_positions[]")
            for position in physical_block_positions
        )
    if sorted(physical_order) != list(range(1, TOP_K + 1)):
        raise ValueError("Physical scoring-block positions must be a permutation of 1..50")
    prefix = _common_prefix(str(record["history"]), candidates)
    blocks = tuple(_scoring_block(candidates[position - 1]) for position in physical_order)
    return LogicalScoringInput(
        common_prefix=prefix,
        scoring_blocks=blocks,
        physical_block_positions=physical_order,
        full_text=prefix + "".join(blocks),
        candidates=tuple(candidates),
    )


def _flat_tokenizer_field(value: Any, field: str) -> list[Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError(f"Tokenizer {field} must be a list or tensor")
    if len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    return value


def _tokenize_segment(
    tokenizer: Any,
    text: str,
    *,
    score_marker: bool,
) -> tuple[list[int], int | None]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            truncation=False,
        )
    except Exception as error:
        raise ValueError("Tokenizer must provide deterministic offset mappings") from error
    if not isinstance(encoded, Mapping):
        raise ValueError("Tokenizer output must be a mapping")
    token_ids = _flat_tokenizer_field(encoded.get("input_ids"), "input_ids")
    offsets = _flat_tokenizer_field(encoded.get("offset_mapping"), "offset_mapping")
    if not token_ids or len(token_ids) != len(offsets):
        raise ValueError("Tokenizer IDs and offsets are empty or inconsistent")
    if any(type(token_id) is not int or token_id < 0 for token_id in token_ids):
        raise ValueError("Tokenizer input IDs must be nonnegative integers")
    parsed_offsets: list[tuple[int, int]] = []
    for offset in offsets:
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("Tokenizer offset entries must be start/end pairs")
        start, end = offset
        if type(start) is not int or type(end) is not int:
            raise ValueError("Tokenizer offsets must be integers")
        if start < 0 or end < start or end > len(text):
            raise ValueError("Tokenizer offset is outside its segment")
        parsed_offsets.append((start, end))

    if not score_marker:
        return [int(token_id) for token_id in token_ids], None
    marker_start = text.rfind(SCORING_MARKER)
    marker_end = marker_start + len(SCORING_MARKER)
    if marker_start < 0 or text[marker_start:marker_end] != SCORING_MARKER:
        raise ValueError("Scoring marker is missing from candidate block")
    overlapping = [
        index
        for index, (start, end) in enumerate(parsed_offsets)
        if end > marker_start and start < marker_end
    ]
    if not overlapping:
        raise ValueError("Tokenizer did not recover the scoring-marker boundary")
    score_index = overlapping[-1]
    score_start, score_end = parsed_offsets[score_index]
    if score_end != marker_end or score_start >= marker_end:
        raise ValueError("Final scoring-marker token boundary is ambiguous")
    if any(
        start < marker_end < end for start, end in parsed_offsets
    ):
        raise ValueError("A token crosses the scoring-marker boundary")
    return [int(token_id) for token_id in token_ids], score_index


def _independent_attention_mask(
    *,
    sequence_length: int,
    common_prefix_length: int,
    block_spans: Sequence[tuple[int, int]],
) -> torch.Tensor:
    mask = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)
    mask[:common_prefix_length, :common_prefix_length] = torch.tril(
        torch.ones(
            (common_prefix_length, common_prefix_length),
            dtype=torch.bool,
        )
    )
    for start, end in block_spans:
        if not common_prefix_length <= start < end <= sequence_length:
            raise ValueError("Scoring-block token span is invalid")
        mask[start:end, :common_prefix_length] = True
        block_length = end - start
        mask[start:end, start:end] = torch.tril(
            torch.ones((block_length, block_length), dtype=torch.bool)
        )
    covered = sorted(index for start, end in block_spans for index in range(start, end))
    if covered != list(range(common_prefix_length, sequence_length)):
        raise ValueError("Scoring blocks do not exactly partition the post-prefix tokens")
    return mask


@dataclass(frozen=True)
class PackedScoringEvent:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    score_token_indices: torch.Tensor
    common_prefix_length: int
    block_spans: tuple[tuple[int, int], ...]
    physical_block_positions: tuple[int, ...]
    canonical_entity_ids: tuple[int, ...]
    rrf_ranks: tuple[int, ...]
    rrf_scores: tuple[float, ...]
    local_ids: tuple[str, ...]
    full_text: str
    mask_policy_version: str = MASK_POLICY_VERSION
    position_id_policy_version: str = POSITION_ID_POLICY_VERSION


def tokenize_scoring_event(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    physical_block_positions: Sequence[int] | None = None,
    max_sequence_length: int | None = None,
) -> PackedScoringEvent:
    """Tokenize prefix/blocks separately and construct exact mask boundaries."""

    logical = build_logical_scoring_input(
        record,
        physical_block_positions=physical_block_positions,
    )
    prefix_ids, prefix_score = _tokenize_segment(
        tokenizer,
        logical.common_prefix,
        score_marker=False,
    )
    if prefix_score is not None:
        raise AssertionError("Common prefix unexpectedly produced a score token")
    common_prefix_length = len(prefix_ids)
    all_ids = list(prefix_ids)
    all_position_ids = list(range(common_prefix_length))
    block_spans_by_logical_position: list[tuple[int, int] | None] = [None] * TOP_K
    score_indices_by_logical_position: list[int | None] = [None] * TOP_K
    physical_spans: list[tuple[int, int]] = []

    for logical_position, block in zip(
        logical.physical_block_positions,
        logical.scoring_blocks,
        strict=True,
    ):
        block_ids, local_score_index = _tokenize_segment(
            tokenizer,
            block,
            score_marker=True,
        )
        if local_score_index is None:
            raise ValueError("Candidate block has no score-token boundary")
        start = len(all_ids)
        end = start + len(block_ids)
        span = (start, end)
        physical_spans.append(span)
        block_spans_by_logical_position[logical_position - 1] = span
        score_indices_by_logical_position[logical_position - 1] = (
            start + local_score_index
        )
        all_ids.extend(block_ids)
        all_position_ids.extend(
            range(common_prefix_length, common_prefix_length + len(block_ids))
        )

    if len(physical_spans) != TOP_K or any(
        span is None for span in block_spans_by_logical_position
    ):
        raise ValueError("Exactly 50 scoring blocks were not recovered")
    if any(index is None for index in score_indices_by_logical_position):
        raise ValueError("Exactly 50 score-token indices were not recovered")
    sequence_length = len(all_ids)
    inferred_limit = getattr(tokenizer, "model_max_length", None)
    if max_sequence_length is None and type(inferred_limit) is int and inferred_limit < 10**9:
        max_sequence_length = inferred_limit
    if max_sequence_length is not None:
        limit = _strict_int(max_sequence_length, "max_sequence_length")
        if sequence_length > limit:
            raise ValueError(
                f"Tokenized scoring input requires {sequence_length} tokens, exceeding {limit}"
            )

    mask = _independent_attention_mask(
        sequence_length=sequence_length,
        common_prefix_length=common_prefix_length,
        block_spans=physical_spans,
    )
    candidates = logical.candidates
    event = PackedScoringEvent(
        input_ids=torch.tensor(all_ids, dtype=torch.long),
        attention_mask=mask,
        position_ids=torch.tensor(all_position_ids, dtype=torch.long),
        score_token_indices=torch.tensor(
            [int(index) for index in score_indices_by_logical_position],
            dtype=torch.long,
        ),
        common_prefix_length=common_prefix_length,
        block_spans=tuple(
            span for span in block_spans_by_logical_position if span is not None
        ),
        physical_block_positions=logical.physical_block_positions,
        canonical_entity_ids=tuple(int(item["canonical_entity_id"]) for item in candidates),
        rrf_ranks=tuple(int(item["rrf_rank"]) for item in candidates),
        rrf_scores=tuple(float(item["rrf_score"]) for item in candidates),
        local_ids=tuple(str(item["local_id"]) for item in candidates),
        full_text=logical.full_text,
    )
    validate_packed_event(event)
    return event


def validate_packed_event(event: PackedScoringEvent) -> None:
    if event.mask_policy_version != MASK_POLICY_VERSION:
        raise ValueError("Packed event mask-policy version mismatch")
    if event.position_id_policy_version != POSITION_ID_POLICY_VERSION:
        raise ValueError("Packed event position-ID-policy version mismatch")
    if event.input_ids.ndim != 1 or event.input_ids.dtype != torch.long:
        raise ValueError("Packed event input IDs must be a one-dimensional long tensor")
    sequence_length = int(event.input_ids.shape[0])
    if event.attention_mask.shape != (sequence_length, sequence_length):
        raise ValueError("Packed event attention mask has an invalid shape")
    if event.attention_mask.dtype != torch.bool:
        raise ValueError("Packed event attention mask must be boolean")
    if event.position_ids.shape != (sequence_length,):
        raise ValueError("Packed event position IDs have an invalid shape")
    if event.score_token_indices.shape != (TOP_K,):
        raise ValueError("Packed event must expose exactly 50 score-token indices")
    if len(event.block_spans) != TOP_K:
        raise ValueError("Packed event must expose exactly 50 block spans")
    physical_spans = [
        event.block_spans[logical_position - 1]
        for logical_position in event.physical_block_positions
    ]
    expected_mask = _independent_attention_mask(
        sequence_length=sequence_length,
        common_prefix_length=event.common_prefix_length,
        block_spans=physical_spans,
    )
    if not torch.equal(event.attention_mask.cpu(), expected_mask):
        raise ValueError("Packed event does not contain the required independent-block mask")
    expected_positions = list(range(event.common_prefix_length))
    for start, end in physical_spans:
        expected_positions.extend(
            range(event.common_prefix_length, event.common_prefix_length + end - start)
        )
    if event.position_ids.tolist() != expected_positions:
        raise ValueError("Packed event does not contain the required repeated position IDs")
    for logical_index, (start, end) in enumerate(event.block_spans):
        score_index = int(event.score_token_indices[logical_index])
        if not start <= score_index < end:
            raise ValueError("Score-token index is outside its candidate block")
    if len(event.canonical_entity_ids) != TOP_K or len(set(event.canonical_entity_ids)) != TOP_K:
        raise ValueError("Packed event candidate IDs are not exactly 50 unique values")


@dataclass(frozen=True)
class PackedScoringBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    score_token_indices: torch.Tensor
    sequence_lengths: tuple[int, ...]
    events: tuple[PackedScoringEvent, ...]
    mask_policy_version: str = MASK_POLICY_VERSION
    position_id_policy_version: str = POSITION_ID_POLICY_VERSION

    def to(self, device: torch.device | str) -> "PackedScoringBatch":
        return PackedScoringBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            position_ids=self.position_ids.to(device),
            score_token_indices=self.score_token_indices.to(device),
            sequence_lengths=self.sequence_lengths,
            events=self.events,
            mask_policy_version=self.mask_policy_version,
            position_id_policy_version=self.position_id_policy_version,
        )


def collate_scoring_events(
    events: Sequence[PackedScoringEvent],
    *,
    pad_token_id: int,
) -> PackedScoringBatch:
    if not events:
        raise ValueError("At least one packed scoring event is required")
    pad_id = _strict_int(pad_token_id, "pad_token_id")
    if pad_id < 0:
        raise ValueError("pad_token_id must be nonnegative")
    for event in events:
        validate_packed_event(event)
    batch_size = len(events)
    maximum_length = max(int(event.input_ids.shape[0]) for event in events)
    input_ids = torch.full((batch_size, maximum_length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros(
        (batch_size, 1, maximum_length, maximum_length), dtype=torch.bool
    )
    position_ids = torch.zeros((batch_size, maximum_length), dtype=torch.long)
    score_indices = torch.empty((batch_size, TOP_K), dtype=torch.long)
    sequence_lengths: list[int] = []
    for batch_index, event in enumerate(events):
        length = int(event.input_ids.shape[0])
        sequence_lengths.append(length)
        input_ids[batch_index, :length] = event.input_ids
        attention_mask[batch_index, 0, :length, :length] = event.attention_mask
        position_ids[batch_index, :length] = event.position_ids
        score_indices[batch_index] = event.score_token_indices
        for padding_index in range(length, maximum_length):
            attention_mask[batch_index, 0, padding_index, padding_index] = True
    return PackedScoringBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        score_token_indices=score_indices,
        sequence_lengths=tuple(sequence_lengths),
        events=tuple(events),
    )


def validate_packed_batch(batch: PackedScoringBatch) -> None:
    if batch.mask_policy_version != MASK_POLICY_VERSION:
        raise ValueError("Batch mask-policy version mismatch")
    if batch.position_id_policy_version != POSITION_ID_POLICY_VERSION:
        raise ValueError("Batch position-ID-policy version mismatch")
    if batch.input_ids.ndim != 2:
        raise ValueError("Batch input IDs must have shape [batch, sequence]")
    batch_size, maximum_length = batch.input_ids.shape
    if batch.attention_mask.shape != (batch_size, 1, maximum_length, maximum_length):
        raise ValueError("Batch attention mask must have shape [batch, 1, L, L]")
    if batch.attention_mask.dtype != torch.bool:
        raise ValueError("Batch attention mask must be boolean")
    if batch.position_ids.shape != (batch_size, maximum_length):
        raise ValueError("Batch position IDs have an invalid shape")
    if batch.score_token_indices.shape != (batch_size, TOP_K):
        raise ValueError("Batch must contain exactly 50 score indices per event")
    if len(batch.events) != batch_size or len(batch.sequence_lengths) != batch_size:
        raise ValueError("Batch event metadata count mismatch")

    for batch_index, event in enumerate(batch.events):
        validate_packed_event(event)
        length = int(event.input_ids.shape[0])
        if batch.sequence_lengths[batch_index] != length or length > maximum_length:
            raise ValueError("Batch sequence-length provenance mismatch")
        expected_mask = torch.zeros(
            (maximum_length, maximum_length), dtype=torch.bool
        )
        expected_mask[:length, :length] = event.attention_mask
        for padding_index in range(length, maximum_length):
            expected_mask[padding_index, padding_index] = True
        if not torch.equal(
            batch.attention_mask[batch_index, 0].detach().cpu(), expected_mask
        ):
            raise ValueError("Batch does not contain the required independent-block mask")
        if not torch.equal(
            batch.input_ids[batch_index, :length].detach().cpu(), event.input_ids
        ):
            raise ValueError("Batch input IDs do not match packed-event provenance")
        if not torch.equal(
            batch.position_ids[batch_index, :length].detach().cpu(), event.position_ids
        ):
            raise ValueError("Batch position IDs do not match packed-event provenance")
        if not torch.equal(
            batch.score_token_indices[batch_index].detach().cpu(),
            event.score_token_indices,
        ):
            raise ValueError("Batch score-token indices do not match packed-event provenance")


class SharedContextualScoringHead(nn.Module):
    """One shared LayerNorm-plus-linear scorer for all candidate states."""

    def __init__(self, hidden_size: int, *, layer_norm_eps: float = 1e-6) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.projection = nn.Linear(hidden_size, 1, bias=True)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, candidate_hidden_states: torch.Tensor) -> torch.Tensor:
        if candidate_hidden_states.ndim != 3 or candidate_hidden_states.shape[1] != TOP_K:
            raise ValueError("Candidate hidden states must have shape [batch, 50, hidden]")
        return self.projection(self.layer_norm(candidate_hidden_states)).squeeze(-1)


def _require_qwen2_sdpa(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if config is None or getattr(config, "model_type", None) != REQUIRED_MODEL_TYPE:
        raise RuntimeError("Joint scorer requires a Qwen2 base model")
    backend = getattr(config, "_attn_implementation", None)
    if backend != REQUIRED_ATTENTION_BACKEND:
        raise RuntimeError(
            "Joint scorer requires explicit Qwen2 SDPA attention; "
            f"observed {backend!r}"
        )
    if not hasattr(model, "forward"):
        raise RuntimeError("Qwen2 base model has no forward method")


class JointRRFRanker(nn.Module):
    """Qwen2 contextual residual scorer with a single shared zero-init head."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        _require_qwen2_sdpa(base_model)
        hidden_size = _strict_int(
            getattr(base_model.config, "hidden_size", None),
            "base_model.config.hidden_size",
        )
        self.base_model = base_model
        self.scoring_head = SharedContextualScoringHead(
            hidden_size,
            layer_norm_eps=float(getattr(base_model.config, "rms_norm_eps", 1e-6)),
        )

    def forward(self, batch: PackedScoringBatch) -> torch.Tensor:
        _require_qwen2_sdpa(self.base_model)
        validate_packed_batch(batch)
        maximum_position = int(batch.position_ids.max().item())
        configured_maximum = getattr(self.base_model.config, "max_position_embeddings", None)
        if type(configured_maximum) is int and maximum_position >= configured_maximum:
            raise ValueError("Repeated position IDs exceed Qwen2 maximum positions")

        outputs = self.base_model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
            raise RuntimeError("Qwen2 base model did not return last_hidden_state")
        gather_indices = batch.score_token_indices.to(hidden_states.device).unsqueeze(-1)
        gather_indices = gather_indices.expand(-1, -1, hidden_states.shape[-1])
        candidate_hidden_states = hidden_states.gather(1, gather_indices)
        residuals = self.scoring_head(candidate_hidden_states)
        if residuals.shape != (batch.input_ids.shape[0], TOP_K):
            raise RuntimeError("Joint scorer did not produce [batch, 50] residuals")
        if not torch.isfinite(residuals).all():
            raise RuntimeError("Joint scorer produced non-finite residuals")
        return residuals


def mean_center_residuals(residuals: torch.Tensor) -> torch.Tensor:
    if not isinstance(residuals, torch.Tensor) or residuals.shape[-1:] != (TOP_K,):
        raise ValueError("Residuals must have final dimension 50")
    if not torch.is_floating_point(residuals) or not torch.isfinite(residuals).all():
        raise ValueError("Residuals must be finite floating-point values")
    return residuals - residuals.mean(dim=-1, keepdim=True)


@dataclass(frozen=True)
class RRFCombination:
    normalized_prior: torch.Tensor
    log_prior: torch.Tensor
    centered_residuals: torch.Tensor
    final_scores: torch.Tensor


def combine_rrf_prior(
    rrf_scores: torch.Tensor | Sequence[float],
    residuals: torch.Tensor | Sequence[float],
) -> RRFCombination:
    prior = (
        rrf_scores
        if isinstance(rrf_scores, torch.Tensor)
        else torch.tensor(rrf_scores, dtype=torch.float64)
    )
    raw_residuals = (
        residuals
        if isinstance(residuals, torch.Tensor)
        else torch.tensor(residuals, dtype=torch.float64)
    )
    if prior.shape[-1:] != (TOP_K,) or raw_residuals.shape[-1:] != (TOP_K,):
        raise ValueError("RRF scores and residuals must have final dimension 50")
    if prior.shape != raw_residuals.shape:
        raise ValueError("RRF scores and residuals must have identical shapes")
    if not torch.is_floating_point(prior):
        prior = prior.to(torch.float64)
    if not torch.is_floating_point(raw_residuals):
        raw_residuals = raw_residuals.to(torch.float64)
    if not torch.isfinite(prior).all() or not torch.all(prior > 0):
        raise ValueError("Every RRF score must be finite and strictly positive")
    if not torch.isfinite(raw_residuals).all():
        raise ValueError("Every contextual residual must be finite")
    normalized = prior / prior.sum(dim=-1, keepdim=True)
    log_prior = torch.log(normalized)
    centered = mean_center_residuals(raw_residuals)
    final_scores = log_prior + centered
    if not torch.isfinite(final_scores).all():
        raise ValueError("Combined final scores are non-finite")
    return RRFCombination(normalized, log_prior, centered, final_scores)


def rank_candidate_ids(
    final_scores: torch.Tensor | Sequence[float],
    canonical_entity_ids: Sequence[int],
    rrf_ranks: Sequence[int],
) -> list[int]:
    scores = (
        final_scores.detach().cpu().tolist()
        if isinstance(final_scores, torch.Tensor)
        else list(final_scores)
    )
    if len(scores) != TOP_K or len(canonical_entity_ids) != TOP_K or len(rrf_ranks) != TOP_K:
        raise ValueError("Ranking inputs must contain exactly 50 candidates")
    ids = [_strict_int(value, "canonical_entity_ids[]") for value in canonical_entity_ids]
    ranks = [_strict_int(value, "rrf_ranks[]") for value in rrf_ranks]
    numeric_scores = [_finite_float(value, "final_scores[]") for value in scores]
    if len(ids) != len(set(ids)):
        raise ValueError("Canonical entity IDs must be unique")
    if sorted(ranks) != list(range(1, TOP_K + 1)):
        raise ValueError("RRF ranks must be exactly 1..50")
    order = sorted(
        range(TOP_K),
        key=lambda index: (-numeric_scores[index], ranks[index], ids[index]),
    )
    ranked_ids = [ids[index] for index in order]
    if len(ranked_ids) != TOP_K or set(ranked_ids) != set(ids):
        raise RuntimeError("Final ranking did not preserve the candidate set")
    return ranked_ids


def scorer_head_state_dict(model: JointRRFRanker) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.scoring_head.state_dict().items()
    }


def load_scorer_head_state_dict(
    model: JointRRFRanker,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    model.scoring_head.load_state_dict(dict(state_dict), strict=True)
