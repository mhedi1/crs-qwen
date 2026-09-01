from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import transformers
from torch import nn

from my_crs.build_stage2_v2_dataset import TOP_K, canonical_json_bytes, sha256_file
from my_crs.joint_rrf_ranker import (
    JointRRFRanker,
    PackedScoringEvent,
    SharedContextualScoringHead,
    collate_scoring_events,
)
from my_crs.stage2_v2_peft import (
    EXPECTED_DATASET_SHA256,
    EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
    EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT,
    GRADIENT_CHECKPOINTING_USE_REENTRANT,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_PACKED_TOKENS,
    MODEL_ID,
    QUANTIZATION_ENABLED,
    REQUESTED_MODEL_REVISION,
    TRUNCATION_ALLOWED,
    apply_peft_and_build_ranker,
    create_lora_config,
    enable_required_gradient_checkpointing,
    gradient_norm_report,
    load_base_qwen,
    phase3b_integration_fingerprint,
    phase3b_scientific_configuration,
    require_single_cuda_device,
    smoke_only_centered_contrast_objective,
    stream_record_by_instance_key,
    tokenize_single_smoke_event,
    validate_first_backward_gradients,
    validate_loaded_base_qwen,
    validate_packed_token_count,
    validate_second_backward_gradients,
    validate_smoke_report_privacy,
    validate_trainability,
    write_smoke_report,
)


PEFT_AVAILABLE = importlib.util.find_spec("peft") is not None


class _FakeLoraProjection(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.base_layer = nn.Linear(width, width, bias=False)
        self.base_layer.requires_grad_(False)
        self.lora_A = nn.ModuleDict(
            {"default": nn.Linear(width, 2, bias=False)}
        )
        self.lora_B = nn.ModuleDict(
            {"default": nn.Linear(2, width, bias=False)}
        )


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for target in LORA_TARGET_MODULES:
            setattr(self, target, _FakeLoraProjection())


class _FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeQwenWithLora(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen2",
            _attn_implementation="sdpa",
            hidden_size=4,
            rms_norm_eps=1e-6,
            max_position_embeddings=4096,
            use_cache=False,
        )
        self.embedding = nn.Embedding(64, 4)
        self.embedding.requires_grad_(False)
        self.layers = nn.ModuleList([_FakeLayer()])


class _LoadedBFloat16Qwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen2",
            _attn_implementation="sdpa",
            _commit_hash=REQUESTED_MODEL_REVISION,
            use_cache=True,
        )
        self.weight = nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))


class _RecordingCheckpointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=True)
        self.is_gradient_checkpointing = False
        self.kwargs = None
        self.inputs_require_grad_enabled = False

    def gradient_checkpointing_enable(self, *, gradient_checkpointing_kwargs):
        self.kwargs = gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = True

    def enable_input_require_grads(self):
        self.inputs_require_grad_enabled = True


def _minimal_packed_event() -> PackedScoringEvent:
    sequence_length = 1 + TOP_K
    attention = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)
    attention[0, 0] = True
    for index in range(1, sequence_length):
        attention[index, 0] = True
        attention[index, index] = True
    return PackedScoringEvent(
        input_ids=torch.tensor([1] + [2 + index % 20 for index in range(TOP_K)]),
        attention_mask=attention,
        position_ids=torch.tensor([0] + [1] * TOP_K),
        score_token_indices=torch.arange(1, sequence_length, dtype=torch.long),
        common_prefix_length=1,
        block_spans=tuple((index, index + 1) for index in range(1, sequence_length)),
        physical_block_positions=tuple(range(1, TOP_K + 1)),
        canonical_entity_ids=tuple(range(1000, 1000 + TOP_K)),
        rrf_ranks=tuple(range(1, TOP_K + 1)),
        rrf_scores=tuple(1.0 / (60 + rank) for rank in range(1, TOP_K + 1)),
        local_ids=tuple(f"C{position:02d}" for position in range(1, TOP_K + 1)),
        full_text="",
    )


def _fake_ranker() -> JointRRFRanker:
    return JointRRFRanker(_FakeQwenWithLora())


class Stage2V2PeftPolicyTests(unittest.TestCase):
    def test_integration_fingerprint_is_deterministic_and_runtime_independent(self):
        expected = "ee755d8860b48992b5eac5067c2d463964f7b3e5b21d2766db807b72a8148d36"
        observed = phase3b_integration_fingerprint()
        self.assertEqual(observed, expected)
        self.assertEqual(observed, phase3b_integration_fingerprint())
        with mock.patch.object(torch, "__version__", "mock-torch"), mock.patch.object(
            transformers,
            "__version__",
            "mock-transformers",
        ):
            self.assertEqual(expected, phase3b_integration_fingerprint())
        serialized = json.dumps(phase3b_scientific_configuration(), sort_keys=True)
        for runtime_key in (
            "runtime_torch_version",
            "runtime_transformers_version",
            "runtime_peft_version",
            "cuda_version",
            "gpu_name",
            "timestamp",
        ):
            with self.subTest(runtime_key=runtime_key):
                self.assertNotIn(f'"{runtime_key}"', serialized)

    def test_scientific_configuration_pins_all_frozen_identities(self):
        configuration = phase3b_scientific_configuration()
        self.assertEqual(configuration["model_id"], MODEL_ID)
        self.assertEqual(
            configuration["requested_model_revision"], REQUESTED_MODEL_REVISION
        )
        self.assertEqual(
            configuration["dataset_sha256"], EXPECTED_DATASET_SHA256
        )
        self.assertEqual(
            configuration["phase2_architecture_fingerprint"],
            EXPECTED_PHASE2_ARCHITECTURE_FINGERPRINT,
        )
        self.assertEqual(
            configuration["phase3a_analysis_fingerprint"],
            EXPECTED_PHASE3A_ANALYSIS_FINGERPRINT,
        )
        self.assertEqual(configuration["max_packed_tokens"], 2304)
        self.assertFalse(configuration["truncation"])
        self.assertEqual(configuration["dtype"], "bfloat16")
        self.assertEqual(configuration["attention_backend"], "sdpa")
        self.assertFalse(configuration["use_cache"])
        self.assertEqual(
            configuration["gradient_checkpointing"],
            {"enabled": True, "use_reentrant": False},
        )

    def test_exact_lora_policy_and_no_quantization(self):
        self.assertEqual(LORA_R, 16)
        self.assertEqual(LORA_ALPHA, 32)
        self.assertEqual(LORA_DROPOUT, 0.05)
        self.assertEqual(LORA_BIAS, "none")
        self.assertEqual(
            LORA_TARGET_MODULES,
            ("q_proj", "k_proj", "v_proj", "o_proj"),
        )
        self.assertFalse(QUANTIZATION_ENABLED)
        self.assertEqual(
            phase3b_scientific_configuration()["lora"],
            {
                "bias": "none",
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "quantization": False,
                "r": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "task_type": "FEATURE_EXTRACTION",
            },
        )
        lora_constructor = mock.Mock(side_effect=lambda **kwargs: kwargs)
        fake_peft = SimpleNamespace(
            LoraConfig=lora_constructor,
            TaskType=SimpleNamespace(FEATURE_EXTRACTION="feature-extraction"),
        )
        result = create_lora_config(peft_module=fake_peft)
        self.assertEqual(result["target_modules"], list(LORA_TARGET_MODULES))
        self.assertEqual(result["task_type"], "feature-extraction")
        self.assertNotIn("modules_to_save", result)
        self.assertNotIn("use_dora", result)
        self.assertNotIn("quantization", result)

    def test_base_loader_uses_exact_nongenerative_unsharded_bf16_policy(self):
        model = _LoadedBFloat16Qwen()
        auto_model = mock.Mock()
        auto_model.from_pretrained.return_value = model
        loaded, identity = load_base_qwen("cpu", auto_model_class=auto_model)
        self.assertIs(loaded, model)
        auto_model.from_pretrained.assert_called_once_with(
            MODEL_ID,
            revision=REQUESTED_MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        kwargs = auto_model.from_pretrained.call_args.kwargs
        self.assertNotIn("device_map", kwargs)
        self.assertNotIn("quantization_config", kwargs)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(identity["resolved_commit"], REQUESTED_MODEL_REVISION)

    def test_verifiable_model_revision_mismatch_fails_closed(self):
        model = _LoadedBFloat16Qwen()
        model.config.use_cache = False
        model.config._commit_hash = "f" * 40
        with self.assertRaisesRegex(RuntimeError, "resolved commit mismatch"):
            validate_loaded_base_qwen(model)

    def test_gradient_checkpointing_requires_nonreentrant_policy(self):
        model = _RecordingCheckpointModel()
        enable_required_gradient_checkpointing(model)
        self.assertEqual(
            model.kwargs,
            {"use_reentrant": GRADIENT_CHECKPOINTING_USE_REENTRANT},
        )
        self.assertFalse(model.kwargs["use_reentrant"])
        self.assertTrue(model.inputs_require_grad_enabled)
        self.assertFalse(model.config.use_cache)

    def test_real_smoke_rejects_multiple_visible_gpus(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), mock.patch.object(
            torch.cuda,
            "device_count",
            return_value=2,
        ):
            with self.assertRaisesRegex(RuntimeError, "exactly one visible GPU"):
                require_single_cuda_device("cuda:0")

    def test_sequence_ceiling_fails_closed_and_does_not_force_padding(self):
        self.assertEqual(MAX_PACKED_TOKENS, 2304)
        self.assertFalse(TRUNCATION_ALLOWED)
        self.assertEqual(validate_packed_token_count(2304), 2304)
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            validate_packed_token_count(2305)

        event = _minimal_packed_event()
        tokenizer = SimpleNamespace(pad_token_id=0)
        with mock.patch(
            "my_crs.stage2_v2_peft.tokenize_scoring_event",
            return_value=event,
        ) as frozen_tokenizer:
            observed_event, batch, actual = tokenize_single_smoke_event(
                {"fixture": True},
                tokenizer,
            )
        self.assertIs(observed_event, event)
        self.assertEqual(actual, 51)
        self.assertEqual(batch.input_ids.shape, (1, 51))
        frozen_tokenizer.assert_called_once_with(
            {"fixture": True},
            tokenizer,
            physical_block_positions=None,
            max_sequence_length=MAX_PACKED_TOKENS,
        )

    def test_overlength_tokenized_event_fails_before_collation(self):
        overlength = SimpleNamespace(input_ids=torch.zeros(2305, dtype=torch.long))
        tokenizer = SimpleNamespace(pad_token_id=0)
        with mock.patch(
            "my_crs.stage2_v2_peft.tokenize_scoring_event",
            return_value=overlength,
        ), mock.patch(
            "my_crs.stage2_v2_peft.collate_scoring_events"
        ) as collate:
            with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
                tokenize_single_smoke_event({}, tokenizer)
        collate.assert_not_called()

    def test_trainability_report_accepts_only_lora_and_shared_head(self):
        ranker = _fake_ranker()
        report = validate_trainability(ranker)
        self.assertGreater(report["frozen_base_parameters"], 0)
        self.assertGreater(report["lora_trainable_parameters"], 0)
        self.assertGreater(report["scorer_head_trainable_parameters"], 0)
        self.assertEqual(
            report["total_trainable_parameters"],
            report["lora_trainable_parameters"]
            + report["scorer_head_trainable_parameters"],
        )
        self.assertEqual(len(report["actual_lora_module_names"]), 4)
        for name, parameter in ranker.named_parameters():
            if ".lora_" not in name and not name.startswith("scoring_head."):
                self.assertFalse(parameter.requires_grad, name)

    def test_trainability_rejects_trainable_base_missing_lora_and_unexpected(self):
        ranker = _fake_ranker()
        ranker.base_model.embedding.weight.requires_grad_(True)
        with self.assertRaisesRegex(RuntimeError, "Unexpected trainable non-LoRA"):
            validate_trainability(ranker)

        ranker = _fake_ranker()
        next(
            parameter
            for name, parameter in ranker.named_parameters()
            if ".lora_A." in name
        ).requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "LoRA parameters are unexpectedly frozen"):
            validate_trainability(ranker)

        ranker = _fake_ranker()
        ranker.unexpected_auxiliary = nn.Parameter(torch.ones(1))
        with self.assertRaisesRegex(RuntimeError, "Unexpected trainable non-LoRA"):
            validate_trainability(ranker)

    def test_trainability_rejects_frozen_scorer_head(self):
        ranker = _fake_ranker()
        ranker.scoring_head.projection.weight.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, "Scorer-head parameters"):
            validate_trainability(ranker)

    def test_zero_head_first_backward_blocks_upstream_gradient(self):
        torch.manual_seed(991)
        head = SharedContextualScoringHead(8)
        hidden = torch.randn(1, TOP_K, 8, requires_grad=True)
        residuals = head(hidden)
        objective = smoke_only_centered_contrast_objective(residuals)
        objective.backward()
        projection_gradient = head.projection.weight.grad
        self.assertIsNotNone(projection_gradient)
        self.assertGreater(float(projection_gradient.abs().sum()), 0.0)
        self.assertIsNotNone(hidden.grad)
        self.assertEqual(float(hidden.grad.abs().sum()), 0.0)

    def test_gradient_validation_distinguishes_first_and_second_backward(self):
        ranker = _fake_ranker()
        for parameter in ranker.parameters():
            parameter.grad = None
        ranker.scoring_head.projection.weight.grad = torch.ones_like(
            ranker.scoring_head.projection.weight
        )
        for name, parameter in ranker.named_parameters():
            if ".lora_" in name:
                parameter.grad = torch.zeros_like(parameter)
        first = gradient_norm_report(ranker)
        validate_first_backward_gradients(first)
        self.assertTrue(first["scoring_head_projection"]["nonzero"])
        with self.assertRaisesRegex(RuntimeError, "reach at least one LoRA"):
            validate_second_backward_gradients(first)

        lora_b = next(
            parameter
            for name, parameter in ranker.named_parameters()
            if ".lora_B." in name
        )
        lora_b.grad = torch.ones_like(lora_b)
        second = gradient_norm_report(ranker)
        validate_second_backward_gradients(second)
        self.assertGreater(second["lora_B"]["l2_norm"], 0.0)

    def test_smoke_objective_is_explicitly_not_scientific_loss(self):
        protocol = phase3b_scientific_configuration()["smoke_gradient_protocol"]
        self.assertFalse(protocol["scientific_training_objective"])
        self.assertIn("smoke_only", protocol["objective_version"])
        self.assertNotIn("beta", protocol)


class Stage2V2PeftDataAndArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "fixture.jsonl"
        records = [
            {"instance_key": "first", "split": "train", "payload": 1},
            {"instance_key": "wanted", "split": "train", "payload": 2},
            {"instance_key": "later", "split": "dev", "payload": 3},
        ]
        with self.dataset.open("wb") as handle:
            for record in records:
                handle.write(canonical_json_bytes(record) + b"\n")
        self.sha = sha256_file(self.dataset)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_dataset_lookup_streams_to_requested_instance(self):
        with mock.patch.object(
            Path,
            "read_text",
            side_effect=AssertionError("dataset must be streamed"),
        ):
            record, observed_sha = stream_record_by_instance_key(
                self.dataset,
                "wanted",
                expected_sha256=self.sha,
            )
        self.assertEqual(record["payload"], 2)
        self.assertEqual(observed_sha, self.sha)

    def test_dataset_sha_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "dataset SHA256 mismatch"):
            stream_record_by_instance_key(
                self.dataset,
                "wanted",
                expected_sha256="0" * 64,
            )

    def test_smoke_output_rejects_sensitive_fields_and_writes_no_content(self):
        report = {
            "actual_packed_tokens": 51,
            "checks": {"passed": True},
            "dataset": {"sha256": self.sha},
            "instance_key": "wanted",
            "scientific_configuration": {"candidate_count": TOP_K},
        }
        validate_smoke_report_privacy(report)
        path = write_smoke_report(report, self.root / "output")
        contents = path.read_text("utf-8")
        self.assertNotIn("dialogue", contents.lower())
        self.assertNotIn("ground_truth", contents.lower())
        self.assertNotIn('"title"', contents.lower())
        self.assertNotIn('"label"', contents.lower())
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            validate_smoke_report_privacy({"history": "PRIVATE"})


@unittest.skipUnless(PEFT_AVAILABLE, "peft is not installed in this environment")
class TinyActualQwen2PeftTests(unittest.TestCase):
    def test_qkv_o_lora_exercises_joint_ranker_custom_mask(self):
        import peft
        from transformers import Qwen2Config, Qwen2Model

        config = Qwen2Config(
            vocab_size=64,
            hidden_size=24,
            intermediate_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            attention_dropout=0.0,
        )
        config._attn_implementation = "sdpa"
        config.use_cache = False
        base = Qwen2Model(config)
        ranker, report = apply_peft_and_build_ranker(
            base,
            "cpu",
            peft_module=peft,
        )
        ranker.eval()
        self.assertEqual(len(report["actual_lora_module_names"]), 4)
        batch = collate_scoring_events([_minimal_packed_event()], pad_token_id=0)
        with torch.no_grad():
            residuals = ranker(batch)
        self.assertEqual(residuals.shape, (1, TOP_K))
        self.assertTrue(torch.isfinite(residuals).all())


if __name__ == "__main__":
    unittest.main()
