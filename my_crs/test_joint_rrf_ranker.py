from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import random
import re
import unittest
from dataclasses import replace
from unittest import mock

import torch
import transformers
from transformers import Qwen2Config, Qwen2Model

from my_crs.build_stage2_v2_dataset import (
    CANDIDATE_ORDER_VERSION,
    DATASET_SCHEMA_VERSION,
    canonical_json_digest,
    serialize_candidates,
)
from my_crs.joint_rrf_ranker import (
    MASK_POLICY_VERSION,
    POSITION_ID_POLICY_VERSION,
    REQUIRED_ATTENTION_BACKEND,
    SCORING_MARKER,
    JointRRFRanker,
    SharedContextualScoringHead,
    build_logical_scoring_input,
    canonicalize_phase1_candidates,
    collate_scoring_events,
    combine_rrf_prior,
    load_scorer_head_state_dict,
    mean_center_residuals,
    phase2_architecture_configuration,
    phase2_architecture_fingerprint,
    phase2_runtime_fingerprint,
    phase2_runtime_provenance,
    rank_candidate_ids,
    scorer_head_state_dict,
    tokenize_scoring_event,
    validate_packed_batch,
)


TOP_K = 50
TOKENIZER_VOCAB_SIZE = 521
PEFT_AVAILABLE = importlib.util.find_spec("peft") is not None


class RegexOffsetTokenizer:
    """Small deterministic tokenizer fixture with exact offset mappings."""

    model_max_length = 8192
    pad_token_id = 0

    def __call__(
        self,
        text,
        *,
        add_special_tokens,
        return_attention_mask,
        return_offsets_mapping,
        truncation,
    ):
        if add_special_tokens or return_attention_mask or not return_offsets_mapping or truncation:
            raise AssertionError("Unexpected tokenizer policy")
        ids = []
        offsets = []
        for match in re.finditer(r"\w+|[^\w\s]", text, flags=re.UNICODE):
            token = match.group(0).encode("utf-8")
            token_id = int.from_bytes(hashlib.sha256(token).digest()[:4], "big")
            ids.append(1 + token_id % (TOKENIZER_VOCAB_SIZE - 1))
            offsets.append((match.start(), match.end()))
        return {"input_ids": ids, "offset_mapping": offsets}


class BrokenOffsetTokenizer(RegexOffsetTokenizer):
    def __call__(self, text, **kwargs):
        encoded = super().__call__(text, **kwargs)
        if SCORING_MARKER in text:
            encoded["offset_mapping"] = [(0, 0)] * len(encoded["input_ids"])
        return encoded


def _phase1_record(
    *,
    identical_titles: bool = False,
    ragged_titles: bool = False,
    history_suffix: str = "",
) -> dict:
    raw_candidates = []
    for rank in range(1, TOP_K + 1):
        title = "Identical Movie" if identical_titles else f"Candidate Movie {rank}"
        if ragged_titles:
            word_count = 1 + rank % 5
            title = " ".join(["Ragged"] * word_count) + f" {rank}"
        if not identical_titles and rank == 7:
            title = "Ragged Candidate\nMovie Seven" if ragged_titles else "Candidate\nMovie Seven"
        raw_candidates.append(
            {
                "ckg_contribution": 1.0 / (200 + rank),
                "ckg_rank": TOP_K + 1 - rank,
                "id": 800000 + rank,
                "kbrd_contribution": 1.0 / (100 + rank),
                "kbrd_rank": rank,
                "rank": rank,
                "rrf_score": 1.0 / (60 + rank) + 1.0 / (110 + rank),
                "source": "RRF",
                "title": title,
            }
        )
    candidates = serialize_candidates(raw_candidates)
    history = "SEEKER: I want a thoughtful science-fiction movie." + history_suffix
    return {
        "candidate_count": TOP_K,
        "candidates": candidates,
        "ground_truth_titles": ["SECRET_GROUND_TRUTH_TITLE"],
        "history": history,
        "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
        "instance_key": "fixture:1",
        "kbrd_private_provenance": "SECRET_KBRD_VALUE",
        "observed_positive_rrf_positions": [7],
        "observed_positive_serialization_positions": [3],
        "schema_version": DATASET_SCHEMA_VERSION,
        "serialization_digest": canonical_json_digest(candidates),
        "serialization_order_version": CANDIDATE_ORDER_VERSION,
        "split": "train",
        "target_response": "SECRET_TARGET_RESPONSE",
    }


def _tiny_qwen2() -> Qwen2Model:
    torch.manual_seed(771)
    config = Qwen2Config(
        vocab_size=TOKENIZER_VOCAB_SIZE,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        attention_dropout=0.0,
    )
    config._attn_implementation = REQUIRED_ATTENTION_BACKEND
    return Qwen2Model(config)


class InputSerializationTests(unittest.TestCase):
    def setUp(self):
        self.record = _phase1_record()
        self.tokenizer = RegexOffsetTokenizer()

    def test_exactly_fifty_blocks_and_visible_context_titles(self):
        logical = build_logical_scoring_input(self.record)
        self.assertEqual(len(logical.scoring_blocks), TOP_K)
        self.assertEqual(logical.full_text.count("<SCORING_QUERY>"), TOP_K)
        self.assertIn(self.record["history"], logical.full_text)
        for candidate in canonicalize_phase1_candidates(self.record):
            self.assertIn(candidate["title_sanitized"], logical.full_text)

    def test_model_text_excludes_retrieval_and_label_provenance(self):
        text = build_logical_scoring_input(self.record).full_text
        forbidden_literals = [
            "rrf_rank",
            "rrf_score",
            "RRF rank",
            "KBRD",
            "CKG",
            "SECRET_KBRD_VALUE",
            "SECRET_GROUND_TRUTH_TITLE",
            "SECRET_TARGET_RESPONSE",
            "observed_positive",
        ]
        for value in forbidden_literals:
            with self.subTest(value=value):
                self.assertNotIn(value, text)
        for candidate in self.record["candidates"]:
            self.assertNotIn(str(candidate["canonical_entity_id"]), text)
            self.assertNotIn(repr(candidate["rrf_score"]), text)
        self.assertNotIn("Candidate\nMovie Seven", text)
        self.assertIn("Candidate Movie Seven", text)

    def test_incoming_candidate_array_permutation_is_canonicalized(self):
        shuffled = copy.deepcopy(self.record)
        random.Random(481).shuffle(shuffled["candidates"])
        original = build_logical_scoring_input(self.record)
        permuted = build_logical_scoring_input(shuffled)
        self.assertEqual(original.full_text, permuted.full_text)
        self.assertEqual(
            [item["canonical_entity_id"] for item in original.candidates],
            [item["canonical_entity_id"] for item in permuted.candidates],
        )

    def test_boundary_reconstruction_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "scoring-marker boundary"):
            tokenize_scoring_event(self.record, BrokenOffsetTokenizer())
        with self.assertRaisesRegex(ValueError, "exceeding"):
            tokenize_scoring_event(self.record, self.tokenizer, max_sequence_length=10)
        malformed = copy.deepcopy(self.record)
        malformed["candidates"].pop()
        malformed["candidate_count"] = 49
        with self.assertRaises(ValueError):
            tokenize_scoring_event(malformed, self.tokenizer)


class IndependentMaskAndPositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = RegexOffsetTokenizer()
        cls.event = tokenize_scoring_event(_phase1_record(), cls.tokenizer)

    def test_mask_policy_and_prefix_are_exact(self):
        event = self.event
        self.assertEqual(event.mask_policy_version, MASK_POLICY_VERSION)
        prefix = event.common_prefix_length
        expected_prefix = torch.tril(torch.ones((prefix, prefix), dtype=torch.bool))
        self.assertTrue(torch.equal(event.attention_mask[:prefix, :prefix], expected_prefix))
        self.assertFalse(event.attention_mask[:prefix, prefix:].any())

    def test_each_block_sees_prefix_and_own_causal_tokens_only(self):
        event = self.event
        prefix = event.common_prefix_length
        for logical_index, (start, end) in enumerate(event.block_spans):
            with self.subTest(logical_index=logical_index):
                self.assertTrue(event.attention_mask[start:end, :prefix].all())
                expected = torch.tril(torch.ones((end - start, end - start), dtype=torch.bool))
                self.assertTrue(
                    torch.equal(event.attention_mask[start:end, start:end], expected)
                )
                for other_index, (other_start, other_end) in enumerate(event.block_spans):
                    if other_index != logical_index:
                        self.assertFalse(
                            event.attention_mask[start:end, other_start:other_end].any()
                        )

    def test_blocks_restart_position_ids_after_prefix(self):
        event = self.event
        self.assertEqual(event.position_id_policy_version, POSITION_ID_POLICY_VERSION)
        prefix = event.common_prefix_length
        self.assertEqual(event.position_ids[:prefix].tolist(), list(range(prefix)))
        for start, end in event.block_spans:
            self.assertEqual(
                event.position_ids[start:end].tolist(),
                list(range(prefix, prefix + end - start)),
            )

    def test_ordinary_causal_mask_is_rejected(self):
        batch = collate_scoring_events([self.event], pad_token_id=0)
        length = batch.input_ids.shape[1]
        ordinary_causal = torch.tril(
            torch.ones((1, 1, length, length), dtype=torch.bool)
        )
        corrupted = replace(batch, attention_mask=ordinary_causal)
        with self.assertRaisesRegex(ValueError, "independent-block mask"):
            validate_packed_batch(corrupted)

    def test_identical_blocks_have_identical_position_ids(self):
        event = tokenize_scoring_event(
            _phase1_record(identical_titles=True), self.tokenizer
        )
        first_start, first_end = event.block_spans[0]
        first_positions = event.position_ids[first_start:first_end]
        for start, end in event.block_spans[1:]:
            self.assertTrue(torch.equal(first_positions, event.position_ids[start:end]))

    def test_ragged_blocks_use_only_block_relative_positions_and_attention(self):
        record = _phase1_record(ragged_titles=True)
        event = tokenize_scoring_event(record, self.tokenizer)
        permuted = tokenize_scoring_event(
            record,
            self.tokenizer,
            physical_block_positions=list(reversed(range(1, TOP_K + 1))),
        )
        block_lengths = [end - start for start, end in event.block_spans]
        self.assertGreater(len(set(block_lengths)), 1)
        prefix = event.common_prefix_length
        marker_positions = []
        for logical_index, ((start, end), (other_start, other_end)) in enumerate(
            zip(event.block_spans, permuted.block_spans, strict=True)
        ):
            with self.subTest(logical_index=logical_index):
                self.assertEqual(int(event.position_ids[start]), prefix)
                self.assertEqual(int(permuted.position_ids[other_start]), prefix)
                self.assertTrue(
                    torch.equal(
                        event.position_ids[start:end],
                        permuted.position_ids[other_start:other_end],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        event.input_ids[start:end],
                        permuted.input_ids[other_start:other_end],
                    )
                )
                marker_index = int(event.score_token_indices[logical_index])
                marker_position = int(event.position_ids[marker_index])
                self.assertEqual(marker_position, prefix + block_lengths[logical_index] - 1)
                marker_positions.append(marker_position)
                self.assertTrue(event.attention_mask[start:end, :prefix].all())
                for other_logical, (cross_start, cross_end) in enumerate(event.block_spans):
                    if other_logical != logical_index:
                        self.assertFalse(
                            event.attention_mask[start:end, cross_start:cross_end].any()
                        )
        self.assertGreater(len(set(marker_positions)), 1)


class ScoringAndRRFHelperTests(unittest.TestCase):
    def test_only_one_shared_head_exists_and_is_zero_initialized(self):
        head = SharedContextualScoringHead(12)
        linear_layers = [module for module in head.modules() if isinstance(module, torch.nn.Linear)]
        self.assertEqual(len(linear_layers), 1)
        self.assertIsNone(head.projection.bias)
        self.assertIsNone(head.layer_norm.bias)
        self.assertIsNotNone(head.layer_norm.weight)
        hidden = torch.randn(2, TOP_K, 12)
        residuals = head(hidden)
        self.assertTrue(torch.equal(residuals, torch.zeros_like(residuals)))

    def test_mean_centering_has_zero_mean(self):
        residuals = torch.linspace(-3.0, 7.0, TOP_K).repeat(2, 1)
        centered = mean_center_residuals(residuals)
        self.assertEqual(centered.shape, residuals.shape)
        self.assertTrue(torch.isfinite(centered).all())
        epsilon = torch.finfo(residuals.dtype).eps
        scale = residuals.abs().amax(dim=-1).clamp_min(1.0)
        mean_tolerance = 2.0 * epsilon * scale
        self.assertTrue(
            torch.all(centered.mean(dim=-1).abs() <= mean_tolerance)
        )

        offset = 1.25
        offset_centered = mean_center_residuals(residuals + offset)
        offset_scale = (residuals.abs().amax() + abs(offset)).clamp_min(1.0)
        offset_tolerance = float(2.0 * epsilon * offset_scale)
        self.assertTrue(torch.isfinite(offset_centered).all())
        self.assertTrue(
            torch.allclose(
                centered,
                offset_centered,
                rtol=0.0,
                atol=offset_tolerance,
            )
        )

    def test_zero_residuals_reproduce_exact_rrf_order(self):
        logical_ranks = list(reversed(range(1, TOP_K + 1)))
        rrf_scores = [1.0 / (60 + rank) for rank in logical_ranks]
        ids = [910000 + index for index in range(TOP_K)]
        combination = combine_rrf_prior(rrf_scores, [0.0] * TOP_K)
        ranked = rank_candidate_ids(combination.final_scores, ids, logical_ranks)
        expected = [
            entity_id
            for _, entity_id in sorted(zip(logical_ranks, ids), key=lambda item: item[0])
        ]
        self.assertEqual(ranked, expected)
        self.assertTrue(
            torch.equal(
                combination.centered_residuals,
                torch.zeros_like(combination.centered_residuals),
            )
        )

    def test_final_ranking_preserves_candidate_set_and_breaks_ties(self):
        ids = list(reversed(range(700001, 700001 + TOP_K)))
        ranks = list(range(1, TOP_K + 1))
        ranked = rank_candidate_ids([0.0] * TOP_K, ids, ranks)
        self.assertEqual(ranked, ids)
        self.assertEqual(len(ranked), TOP_K)
        self.assertEqual(set(ranked), set(ids))

    def test_invalid_rrf_scores_fail_closed(self):
        valid = [1.0] * TOP_K
        for invalid in (
            valid[:-1],
            [0.0] + valid[1:],
            [-1.0] + valid[1:],
            [math.nan] + valid[1:],
            [math.inf] + valid[1:],
        ):
            with self.subTest(invalid=invalid[0] if invalid else None):
                with self.assertRaises(ValueError):
                    combine_rrf_prior(invalid, [0.0] * TOP_K)

    def test_nonfinite_residuals_fail_closed(self):
        for invalid_value in (math.nan, math.inf, -math.inf):
            residuals = [0.0] * TOP_K
            residuals[3] = invalid_value
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaises(ValueError):
                    combine_rrf_prior([1.0] * TOP_K, residuals)

    def _assert_rrf_combination_matches_float64_reference(self, dtype):
        prior = torch.linspace(0.009, 0.031, TOP_K, dtype=dtype).requires_grad_()
        residuals = torch.linspace(-0.2, 0.3, TOP_K, dtype=dtype).requires_grad_()
        combination = combine_rrf_prior(prior, residuals)
        reference_prior = prior.detach().to(torch.float64)
        reference_residuals = residuals.to(torch.float64)
        expected_q = reference_prior / reference_prior.sum()
        expected_centered = reference_residuals - reference_residuals.mean()
        expected_final = torch.log(expected_q) + expected_centered
        self.assertEqual(combination.normalized_prior.dtype, torch.float64)
        self.assertEqual(combination.log_prior.dtype, torch.float64)
        self.assertEqual(combination.centered_residuals.dtype, torch.float64)
        self.assertEqual(combination.final_scores.dtype, torch.float64)
        self.assertTrue(torch.equal(combination.normalized_prior, expected_q))
        self.assertTrue(torch.equal(combination.centered_residuals, expected_centered))
        self.assertTrue(torch.equal(combination.final_scores, expected_final))
        combination.final_scores.square().sum().backward()
        self.assertIsNone(prior.grad)
        self.assertIsNotNone(residuals.grad)
        self.assertTrue(torch.isfinite(residuals.grad).all())

    def test_float16_rrf_combination_uses_float64_math(self):
        self._assert_rrf_combination_matches_float64_reference(torch.float16)

    def test_bfloat16_rrf_combination_uses_float64_math(self):
        try:
            self._assert_rrf_combination_matches_float64_reference(torch.bfloat16)
        except RuntimeError as error:
            self.skipTest(f"bfloat16 is unsupported by this Torch backend: {error}")

    def test_float32_rrf_combination_uses_float64_math(self):
        self._assert_rrf_combination_matches_float64_reference(torch.float32)

    def test_head_state_round_trip_reproduces_outputs(self):
        first = JointRRFRanker(_tiny_qwen2())
        second = JointRRFRanker(_tiny_qwen2())
        torch.manual_seed(192)
        with torch.no_grad():
            first.scoring_head.projection.weight.normal_(mean=0.0, std=0.1)
        state = scorer_head_state_dict(first)
        load_scorer_head_state_dict(second, state)
        hidden = torch.randn(3, TOP_K, first.base_model.config.hidden_size)
        self.assertTrue(
            torch.equal(first.scoring_head(hidden), second.scoring_head(hidden))
        )

    def test_scoring_head_receives_gradients_at_zero_initialization(self):
        head = SharedContextualScoringHead(8)
        hidden = torch.randn(2, TOP_K, 8, requires_grad=True)
        loss = head(hidden).sum()
        loss.backward()
        self.assertIsNotNone(head.projection.weight.grad)
        self.assertGreater(float(head.projection.weight.grad.abs().sum()), 0.0)
        self.assertIsNone(head.projection.bias)
        self.assertIsNone(head.layer_norm.bias)
        self.assertIsNotNone(hidden.grad)
        self.assertTrue(torch.equal(hidden.grad, torch.zeros_like(hidden.grad)))


class TinyQwen2CompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = RegexOffsetTokenizer()
        cls.record = _phase1_record()
        cls.event = tokenize_scoring_event(cls.record, cls.tokenizer)

    def test_runtime_versions_are_recorded_separately(self):
        architecture = phase2_architecture_configuration()
        runtime = phase2_runtime_provenance()
        self.assertEqual(architecture["required_attention_backend"], "sdpa")
        self.assertNotIn("runtime_transformers_version", architecture)
        self.assertNotIn("runtime_torch_version", architecture)
        self.assertEqual(
            runtime["runtime_transformers_version"],
            transformers.__version__,
        )
        self.assertEqual(runtime["runtime_torch_version"], torch.__version__)
        self.assertEqual(runtime["required_attention_backend"], "sdpa")
        self.assertEqual(
            architecture["duplicate_title_policy"],
            "evaluation_equivalent_shared_contextual_residual_v1",
        )
        self.assertEqual(
            architecture["position_id_semantics"],
            "prefix_absolute_then_block_relative_left_aligned_ragged",
        )
        self.assertEqual(architecture["rrf_combination_dtype"], "float64")
        self.assertIs(architecture["scoring_head_biases"], False)

    def test_architecture_fingerprint_is_deterministic_and_runtime_independent(self):
        expected = phase2_architecture_fingerprint()
        self.assertEqual(expected, phase2_architecture_fingerprint())
        with (
            mock.patch.object(transformers, "__version__", "5.999.0"),
            mock.patch.object(torch, "__version__", "2.999.0+synthetic"),
        ):
            self.assertEqual(phase2_architecture_fingerprint(), expected)

    def test_runtime_fingerprint_changes_with_runtime_versions(self):
        expected = phase2_runtime_fingerprint()
        with (
            mock.patch.object(transformers, "__version__", "5.999.0"),
            mock.patch.object(torch, "__version__", "2.999.0+synthetic"),
        ):
            self.assertNotEqual(phase2_runtime_fingerprint(), expected)

    def test_tiny_real_qwen2_accepts_custom_4d_mask_and_returns_finite_scores(self):
        ranker = JointRRFRanker(_tiny_qwen2()).eval()
        batch = collate_scoring_events([self.event], pad_token_id=0)
        with torch.no_grad():
            residuals = ranker(batch)
        self.assertEqual(residuals.shape, (1, TOP_K))
        self.assertTrue(torch.isfinite(residuals).all())
        self.assertTrue(torch.equal(residuals, torch.zeros_like(residuals)))

    def test_physical_block_permutation_is_invariant(self):
        record = _phase1_record(ragged_titles=True)
        original = tokenize_scoring_event(record, self.tokenizer)
        permuted = tokenize_scoring_event(
            record,
            self.tokenizer,
            physical_block_positions=list(reversed(range(1, TOP_K + 1))),
        )
        ranker = JointRRFRanker(_tiny_qwen2()).eval()
        torch.manual_seed(933)
        with torch.no_grad():
            ranker.scoring_head.projection.weight.normal_(mean=0.0, std=0.05)
            first = ranker(collate_scoring_events([original], pad_token_id=0))
            second = ranker(collate_scoring_events([permuted], pad_token_id=0))
        self.assertTrue(torch.allclose(first, second, atol=2e-6, rtol=2e-6))

    def test_identical_title_null_has_identical_residuals(self):
        record = _phase1_record(identical_titles=True)
        event = tokenize_scoring_event(
            record,
            self.tokenizer,
        )
        permuted = tokenize_scoring_event(
            record,
            self.tokenizer,
            physical_block_positions=list(reversed(range(1, TOP_K + 1))),
        )
        ranker = JointRRFRanker(_tiny_qwen2()).eval()
        torch.manual_seed(418)
        with torch.no_grad():
            ranker.scoring_head.projection.weight.normal_(mean=0.0, std=0.05)
            residuals = ranker(collate_scoring_events([event], pad_token_id=0))[0]
            permuted_residuals = ranker(
                collate_scoring_events([permuted], pad_token_id=0)
            )[0]
        self.assertTrue(
            torch.allclose(
                residuals,
                residuals[0].expand_as(residuals),
                atol=2e-6,
                rtol=2e-6,
            )
        )
        self.assertTrue(
            torch.allclose(residuals, permuted_residuals, atol=2e-6, rtol=2e-6)
        )

    def test_mixed_length_padded_batch_matches_individual_forwards(self):
        first_event = tokenize_scoring_event(
            _phase1_record(ragged_titles=True), self.tokenizer
        )
        second_event = tokenize_scoring_event(
            _phase1_record(
                ragged_titles=True,
                history_suffix=" Please avoid horror and include a hopeful ending.",
            ),
            self.tokenizer,
        )
        self.assertNotEqual(len(first_event.input_ids), len(second_event.input_ids))
        ranker = JointRRFRanker(_tiny_qwen2()).eval()
        torch.manual_seed(144)
        with torch.no_grad():
            ranker.scoring_head.projection.weight.normal_(mean=0.0, std=0.05)
            first_alone = ranker(collate_scoring_events([first_event], pad_token_id=0))[0]
            second_alone = ranker(collate_scoring_events([second_event], pad_token_id=0))[0]
            together = ranker(
                collate_scoring_events([first_event, second_event], pad_token_id=0)
            )
        self.assertTrue(torch.allclose(together[0], first_alone, atol=2e-6, rtol=2e-6))
        self.assertTrue(torch.allclose(together[1], second_alone, atol=2e-6, rtol=2e-6))

    def test_forward_hot_path_does_not_call_full_batch_validator(self):
        ranker = JointRRFRanker(_tiny_qwen2()).eval()
        batch = collate_scoring_events([self.event], pad_token_id=0)
        with (
            mock.patch(
                "my_crs.joint_rrf_ranker.validate_packed_batch",
                side_effect=AssertionError("full validator entered hot path"),
            ),
            torch.no_grad(),
        ):
            residuals = ranker(batch)
        self.assertEqual(residuals.shape, (1, TOP_K))

    def test_gradients_can_flow_through_shared_head_and_qwen_base(self):
        ranker = JointRRFRanker(_tiny_qwen2()).train()
        with torch.no_grad():
            ranker.scoring_head.projection.weight.fill_(0.01)
        residuals = ranker(collate_scoring_events([self.event], pad_token_id=0))
        residuals.square().sum().backward()
        self.assertGreater(
            float(ranker.scoring_head.projection.weight.grad.abs().sum()), 0.0
        )
        embedding_gradient = ranker.base_model.embed_tokens.weight.grad
        self.assertIsNotNone(embedding_gradient)
        self.assertGreater(float(embedding_gradient.abs().sum()), 0.0)

    def test_non_sdpa_qwen2_backend_fails_closed(self):
        model = _tiny_qwen2()
        model.config._attn_implementation = "eager"
        with self.assertRaisesRegex(RuntimeError, "requires explicit Qwen2 SDPA"):
            JointRRFRanker(model)

    @unittest.skipUnless(PEFT_AVAILABLE, "peft is not installed in this environment")
    def test_tiny_qwen2_peft_preserves_custom_mask_positions_and_permutation(self):
        from peft import LoraConfig, get_peft_model

        record = _phase1_record(ragged_titles=True)
        event = tokenize_scoring_event(record, self.tokenizer)
        permuted = tokenize_scoring_event(
            record,
            self.tokenizer,
            physical_block_positions=list(reversed(range(1, TOP_K + 1))),
        )
        peft_model = get_peft_model(
            _tiny_qwen2(),
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            ),
        )
        ranker = JointRRFRanker(peft_model).eval()
        torch.manual_seed(884)
        with torch.no_grad():
            ranker.scoring_head.projection.weight.normal_(mean=0.0, std=0.05)

        captured = []

        def capture_inputs(_module, _args, kwargs):
            captured.append(dict(kwargs))

        base_qwen = peft_model.get_base_model()
        hook = base_qwen.register_forward_pre_hook(capture_inputs, with_kwargs=True)
        try:
            first_batch = collate_scoring_events([event], pad_token_id=0)
            second_batch = collate_scoring_events([permuted], pad_token_id=0)
            with torch.no_grad():
                first = ranker(first_batch)
                second = ranker(second_batch)
        finally:
            hook.remove()

        self.assertEqual(first.shape, (1, TOP_K))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.allclose(first, second, atol=2e-6, rtol=2e-6))
        self.assertEqual(len(captured), 2)
        for observed, expected_batch in zip(
            captured, (first_batch, second_batch), strict=True
        ):
            self.assertTrue(
                torch.equal(observed["attention_mask"], expected_batch.attention_mask)
            )
            self.assertTrue(
                torch.equal(observed["position_ids"], expected_batch.position_ids)
            )
            self.assertIs(observed["use_cache"], False)


if __name__ == "__main__":
    unittest.main()
