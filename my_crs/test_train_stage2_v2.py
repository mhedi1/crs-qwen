from __future__ import annotations

import hashlib
import json
import random
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from my_crs.build_stage2_v2_dataset import (
    CANDIDATE_ORDER_VERSION,
    DATASET_SCHEMA_VERSION,
    TOP_K,
    canonical_json_bytes,
    canonical_json_digest,
    serialize_candidates,
)
from my_crs.stage2_v2_loss import (
    EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT,
    training_loss_inputs_from_record,
)
from my_crs.train_stage2_v2 import (
    CHECKPOINT_SCHEMA,
    MANIFEST_FILENAME,
    AccumulationWindow,
    TrainingState,
    build_train_offset_index,
    create_trainable_optimizer,
    deterministic_epoch_order,
    load_checkpoint_payload,
    load_phase3b_training_stack,
    prepare_training_event,
    save_checkpoint_payload,
    train_stage2_v2,
    training_scientific_configuration,
    training_scientific_fingerprint,
    validate_training_artifact_privacy,
    write_training_metric,
)


def _record(instance_key: str, split: str) -> dict:
    raw_candidates = []
    for rank in range(1, TOP_K + 1):
        raw_candidates.append(
            {
                "ckg_contribution": 1.0 / (200 + rank),
                "ckg_rank": TOP_K + 1 - rank,
                "id": 600000 + rank,
                "kbrd_contribution": 1.0 / (100 + rank),
                "kbrd_rank": rank,
                "rank": rank,
                "rrf_score": 1.0 / (60 + rank) + 1.0 / (110 + rank),
                "source": "RRF",
                "title": f"Fixture Movie {rank}",
            }
        )
    candidates = serialize_candidates(raw_candidates)
    history = f"SEEKER: private fixture for {instance_key}"
    return {
        "candidate_count": TOP_K,
        "candidates": candidates,
        "history": history,
        "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
        "instance_key": instance_key,
        "observed_positive_serialization_positions": [3] if split == "train" else [],
        "schema_version": DATASET_SCHEMA_VERSION,
        "serialization_digest": canonical_json_digest(candidates),
        "serialization_order_version": CANDIDATE_ORDER_VERSION,
        "split": split,
    }


def _write_jsonl(path: Path, records: list[dict]) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for record in records:
            encoded = canonical_json_bytes(record) + b"\n"
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def _configuration() -> dict:
    return training_scientific_configuration(
        beta=0.10,
        seed=42,
        learning_rate=1e-4,
        gradient_accumulation_steps=2,
        gradient_clip_norm=1.0,
        max_optimizer_steps=3,
    )


class TrainOffsetIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "fixture.jsonl"
        self.records = [
            _record("train-b", "train"),
            _record("dev-a", "dev"),
            _record("train-a", "train"),
            _record("dev-b", "dev"),
            _record("train-c", "train"),
        ]
        self.sha = _write_jsonl(self.path, self.records)
        self.counts = {"all": 5, "train": 3, "dev": 2}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_binary_index_streams_once_and_retains_train_only_offsets(self):
        with mock.patch.object(
            Path,
            "read_text",
            side_effect=AssertionError("full-file text loading is forbidden"),
        ), mock.patch.object(
            Path,
            "readlines",
            side_effect=AssertionError("readlines is forbidden"),
            create=True,
        ):
            index = build_train_offset_index(
                self.path,
                expected_sha256=self.sha,
                expected_counts=self.counts,
            )
        self.assertEqual(index.counts, self.counts)
        self.assertEqual(index.dataset_sha256, self.sha)
        self.assertEqual(
            [entry.instance_key for entry in index.entries],
            ["train-b", "train-a", "train-c"],
        )
        self.assertTrue(all(entry.split == "train" for entry in index.entries))
        with index.path.open("rb") as handle:
            self.assertEqual(index.read_record(1, handle=handle)["instance_key"], "train-a")

    def test_sha_and_accounting_mismatches_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "dataset SHA256 mismatch"):
            build_train_offset_index(
                self.path,
                expected_sha256="0" * 64,
                expected_counts=self.counts,
            )
        wrong = dict(self.counts)
        wrong["train"] += 1
        with self.assertRaisesRegex(ValueError, "accounting mismatch"):
            build_train_offset_index(
                self.path,
                expected_sha256=self.sha,
                expected_counts=wrong,
            )

    def test_invalid_split_and_candidate_count_fail_closed(self):
        records = list(self.records)
        records[0] = dict(records[0], split="test")
        sha = _write_jsonl(self.path, records)
        with self.assertRaisesRegex(ValueError, "invalid split"):
            build_train_offset_index(
                self.path,
                expected_sha256=sha,
                expected_counts=self.counts,
            )

        records = list(self.records)
        records[0] = dict(records[0], candidate_count=49)
        sha = _write_jsonl(self.path, records)
        with self.assertRaisesRegex(ValueError, "must contain 50 candidates"):
            build_train_offset_index(
                self.path,
                expected_sha256=sha,
                expected_counts=self.counts,
            )

    def test_epoch_shuffle_is_deterministic_and_epoch_specific(self):
        first = deterministic_epoch_order(100, seed=42, epoch=0)
        second = deterministic_epoch_order(100, seed=42, epoch=0)
        next_epoch = deterministic_epoch_order(100, seed=42, epoch=1)
        self.assertEqual(first, second)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(sorted(first), list(range(100)))


class TrainerPlumbingTests(unittest.TestCase):
    def test_phase3b_model_loading_is_reused_directly(self):
        tokenizer = object()
        base_model = object()
        ranker = mock.Mock()
        with mock.patch(
            "my_crs.train_stage2_v2.load_production_tokenizer",
            return_value=tokenizer,
        ) as load_tokenizer, mock.patch(
            "my_crs.train_stage2_v2.validate_tokenizer",
            return_value="resolved-tokenizer",
        ) as validate_tokenizer, mock.patch(
            "my_crs.train_stage2_v2.load_base_qwen",
            return_value=(base_model, {"identity": True}),
        ) as load_base, mock.patch(
            "my_crs.train_stage2_v2.apply_peft_and_build_ranker",
            return_value=(ranker, {"trainability": True}),
        ) as apply_peft, mock.patch(
            "my_crs.train_stage2_v2.validate_trainability"
        ) as validate_trainability:
            result = load_phase3b_training_stack(torch.device("cpu"))
        self.assertIs(result[0], tokenizer)
        self.assertIs(result[1], ranker)
        self.assertEqual(result[4], "resolved-tokenizer")
        load_tokenizer.assert_called_once_with()
        validate_tokenizer.assert_called_once_with(tokenizer)
        load_base.assert_called_once_with(torch.device("cpu"))
        apply_peft.assert_called_once_with(base_model, torch.device("cpu"))
        validate_trainability.assert_called_once_with(ranker)

    def test_event_preparation_reuses_phase3b_tokenization_and_ceiling(self):
        record = _record("train-event", "train")
        inputs = training_loss_inputs_from_record(record)
        event = SimpleNamespace(rrf_scores=inputs.rrf_scores)
        batch = object()
        tokenizer = object()
        with mock.patch(
            "my_crs.train_stage2_v2.tokenize_single_smoke_event",
            return_value=(event, batch, 51),
        ) as tokenize:
            prepared = prepare_training_event(record, tokenizer)
        tokenize.assert_called_once_with(record, tokenizer)
        self.assertIs(prepared.batch, batch)
        self.assertEqual(prepared.actual_tokens, 51)
        self.assertEqual(prepared.positive_positions, (3,))

    def test_event_preparation_fails_closed_above_2304(self):
        record = _record("train-event", "train")
        inputs = training_loss_inputs_from_record(record)
        event = SimpleNamespace(rrf_scores=inputs.rrf_scores)
        with mock.patch(
            "my_crs.train_stage2_v2.tokenize_single_smoke_event",
            return_value=(event, object(), 2305),
        ):
            with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
                prepare_training_event(record, object())

    def test_optimizer_contains_exactly_trainable_parameters(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
        model[0].requires_grad_(False)
        optimizer = create_trainable_optimizer(model, learning_rate=1e-4)
        observed = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        expected = {id(parameter) for parameter in model[1].parameters()}
        self.assertEqual(observed, expected)
        self.assertTrue(all(not parameter.requires_grad for parameter in model[0].parameters()))

    def test_output_collision_fails_before_dataset_or_model_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            (destination / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
            with mock.patch(
                "my_crs.train_stage2_v2.build_train_offset_index",
                side_effect=AssertionError("dataset must not be opened"),
            ), mock.patch(
                "my_crs.train_stage2_v2.load_phase3b_training_stack",
                side_effect=AssertionError("model must not be loaded"),
            ):
                with self.assertRaisesRegex(FileExistsError, "output already exists"):
                    train_stage2_v2(
                        dataset_path=destination / "missing.jsonl",
                        output_dir=destination,
                        beta=0.10,
                        max_optimizer_steps=1,
                        gradient_accumulation_steps=1,
                        learning_rate=1e-4,
                        gradient_clip_norm=1.0,
                        checkpoint_every_steps=1,
                        device_name="cuda:0",
                        seed=42,
                    )

    def test_gradient_accumulation_bookkeeping_and_tail_rescale(self):
        window = AccumulationWindow(3)
        window.observe(
            total_loss=2.0,
            partial_loss=1.5,
            anchor_loss=0.5,
            positive=True,
            actual_tokens=1400,
        )
        window.observe(
            total_loss=1.0,
            partial_loss=0.0,
            anchor_loss=0.25,
            positive=False,
            actual_tokens=1500,
        )
        self.assertFalse(window.ready)
        self.assertEqual(window.tail_gradient_rescale(), 1.5)
        self.assertEqual(
            window.metrics(),
            {
                "anchor_loss": 0.375,
                "events": 2,
                "maximum_packed_tokens": 1500,
                "mean_packed_tokens": 1450.0,
                "partial_loss": 1.5,
                "positive_event_rate": 0.5,
                "total_loss": 1.5,
            },
        )
        window.observe(
            total_loss=3.0,
            partial_loss=2.0,
            anchor_loss=0.75,
            positive=True,
            actual_tokens=1300,
        )
        self.assertTrue(window.ready)
        self.assertEqual(window.tail_gradient_rescale(), 1.0)

    def test_epoch_rollover_recomputes_order_and_resume_matches_fresh_sequence(self):
        class SyntheticBatch:
            def to(self, _device):
                return self

        class TinyRanker(nn.Module):
            def __init__(self):
                super().__init__()
                self.residuals = nn.Parameter(torch.zeros(TOP_K))

            def forward(self, _batch):
                return self.residuals.unsqueeze(0)

        class SyntheticOptimizer:
            def __init__(self, model):
                self.parameters = tuple(model.parameters())

            def zero_grad(self, *, set_to_none):
                for parameter in self.parameters:
                    if set_to_none:
                        parameter.grad = None
                    elif parameter.grad is not None:
                        parameter.grad.zero_()

            def step(self):
                return None

        class SyntheticIndex:
            def __init__(self, path: Path):
                self.path = path
                self.dataset_sha256 = "fixture-sha256"
                self.counts = {"all": 5, "train": 5, "dev": 0}
                self.entries = tuple(range(5))
                self.records = [
                    {"instance_key": f"event-{index}"} for index in range(5)
                ]

            def read_record(self, index, *, handle=None):
                self.assert_handle = handle
                return self.records[index]

        class SimulatedInterruption(RuntimeError):
            pass

        def clone_state(state: TrainingState) -> TrainingState:
            return TrainingState(
                epoch=state.epoch,
                next_epoch_position=state.next_epoch_position,
                optimizer_step=state.optimizer_step,
                events_processed=state.events_processed,
                nonzero_lora_gradient_observed=state.nonzero_lora_gradient_observed,
            )

        def execute(
            *,
            index: SyntheticIndex,
            output_dir: Path,
            completed: list[str],
            saved_states: dict[int, TrainingState],
            interrupt_after_events: int | None = None,
            resume_state: TrainingState | None = None,
        ):
            preparation_calls = 0
            current_instance_key = ""

            def prepare(record, _tokenizer):
                nonlocal preparation_calls, current_instance_key
                preparation_calls += 1
                if (
                    interrupt_after_events is not None
                    and preparation_calls > interrupt_after_events
                ):
                    raise SimulatedInterruption("synthetic interruption")
                current_instance_key = record["instance_key"]
                return SimpleNamespace(
                    actual_tokens=100,
                    batch=SyntheticBatch(),
                    positive_positions=(1,),
                    rrf_scores=(1.0,) * TOP_K,
                )

            def save_checkpoint(**kwargs):
                state = clone_state(kwargs["training_state"])
                saved_states[state.optimizer_step] = state
                completed.append(current_instance_key)
                return Path(kwargs["path"]).resolve()

            def restore_checkpoint(**_kwargs):
                if resume_state is None:
                    raise AssertionError("fresh execution must not restore a checkpoint")
                return clone_state(resume_state)

            gradient_groups = {
                "all_lora": {"finite": True, "l2_norm": 1.0},
                "frozen_base": {"finite": True, "l2_norm": 0.0},
                "scoring_head": {"finite": True, "l2_norm": 1.0},
            }
            checkpoint = (
                output_dir / "checkpoints" / "checkpoint_step_00000004.pt"
                if resume_state is not None
                else None
            )
            with mock.patch.dict(
                "sys.modules",
                {"peft": SimpleNamespace(__version__="fixture")},
            ), mock.patch(
                "my_crs.train_stage2_v2.build_train_offset_index",
                return_value=index,
            ), mock.patch(
                "my_crs.train_stage2_v2.require_single_cuda_device",
                return_value=torch.device("cpu"),
            ), mock.patch(
                "my_crs.train_stage2_v2._set_deterministic_seed",
            ), mock.patch(
                "my_crs.train_stage2_v2.load_phase3b_training_stack",
                side_effect=lambda _device: (
                    object(),
                    TinyRanker(),
                    {"fixture_model": True},
                    {"fixture_trainability": True},
                    "fixture-tokenizer-commit",
                ),
            ), mock.patch(
                "my_crs.train_stage2_v2.create_trainable_optimizer",
                side_effect=lambda model, **_kwargs: SyntheticOptimizer(model),
            ), mock.patch(
                "my_crs.train_stage2_v2.prepare_training_event",
                side_effect=prepare,
            ), mock.patch(
                "my_crs.train_stage2_v2.torch.autocast",
                side_effect=lambda **_kwargs: nullcontext(),
            ), mock.patch(
                "my_crs.train_stage2_v2.gradient_norm_report",
                return_value=gradient_groups,
            ), mock.patch(
                "my_crs.train_stage2_v2.cuda_memory_snapshot",
                return_value={"allocated_bytes": 0},
            ), mock.patch(
                "my_crs.train_stage2_v2.torch.cuda.get_device_name",
                return_value="fixture-gpu",
            ), mock.patch(
                "my_crs.train_stage2_v2.save_training_checkpoint",
                side_effect=save_checkpoint,
            ), mock.patch(
                "my_crs.train_stage2_v2.restore_training_checkpoint",
                side_effect=restore_checkpoint,
            ):
                return train_stage2_v2(
                    dataset_path=index.path,
                    output_dir=output_dir,
                    beta=0.10,
                    max_optimizer_steps=7,
                    gradient_accumulation_steps=1,
                    learning_rate=1e-4,
                    gradient_clip_norm=1.0,
                    checkpoint_every_steps=1,
                    device_name="cuda:0",
                    seed=42,
                    resume_checkpoint=checkpoint,
                )

        epoch_zero = deterministic_epoch_order(5, seed=42, epoch=0)
        epoch_one = deterministic_epoch_order(5, seed=42, epoch=1)
        self.assertNotEqual(epoch_zero, epoch_one)
        expected = [f"event-{index}" for index in epoch_zero + epoch_one[:2]]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "synthetic.jsonl"
            dataset.write_bytes(b"fixture\n")
            index = SyntheticIndex(dataset)

            fresh_completed: list[str] = []
            fresh_states: dict[int, TrainingState] = {}
            fresh_result = execute(
                index=index,
                output_dir=root / "fresh",
                completed=fresh_completed,
                saved_states=fresh_states,
            )

            resumed_completed: list[str] = []
            resumed_states: dict[int, TrainingState] = {}
            with self.assertRaisesRegex(SimulatedInterruption, "synthetic interruption"):
                execute(
                    index=index,
                    output_dir=root / "resumed",
                    completed=resumed_completed,
                    saved_states=resumed_states,
                    interrupt_after_events=4,
                )
            boundary_checkpoint = resumed_states[4]
            resumed_result = execute(
                index=index,
                output_dir=root / "resumed",
                completed=resumed_completed,
                saved_states=resumed_states,
                resume_state=boundary_checkpoint,
            )

        self.assertEqual(fresh_completed, expected)
        self.assertEqual(resumed_completed, expected)
        self.assertEqual(fresh_completed[:5], [f"event-{index}" for index in epoch_zero])
        self.assertEqual(fresh_completed[5:], [f"event-{index}" for index in epoch_one[:2]])
        self.assertCountEqual(fresh_completed[:5], [f"event-{index}" for index in range(5)])
        self.assertEqual(len(fresh_completed), 7)
        self.assertEqual(fresh_states[7].epoch, 1)
        self.assertEqual(fresh_states[7].next_epoch_position, 2)
        self.assertEqual(resumed_states[7], fresh_states[7])
        self.assertEqual(fresh_result["events_processed"], 7)
        self.assertEqual(resumed_result["events_processed"], 7)


class CheckpointAndProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_checkpoint_round_trip_preserves_step_epoch_and_excludes_base(self):
        configuration = _configuration()
        fingerprint = training_scientific_fingerprint(
            beta=0.10,
            seed=42,
            learning_rate=1e-4,
            gradient_accumulation_steps=2,
            gradient_clip_norm=1.0,
            max_optimizer_steps=3,
        )
        state = TrainingState(
            epoch=4,
            next_epoch_position=123,
            optimizer_step=17,
            events_processed=34,
            nonzero_lora_gradient_observed=True,
        )
        path = save_checkpoint_payload(
            self.root / "checkpoint.pt",
            adapter_state={"lora_A.weight": torch.tensor([1.0])},
            scorer_head_state={"projection.weight": torch.tensor([[2.0]])},
            optimizer_state={"state": {}, "param_groups": []},
            training_state=state,
            scientific_configuration=configuration,
            scientific_fingerprint=fingerprint,
            rng_state={
                "cuda": [],
                "python": random.getstate(),
                "torch": torch.get_rng_state(),
            },
        )
        payload = load_checkpoint_payload(
            path,
            expected_scientific_fingerprint=fingerprint,
        )
        restored = TrainingState.from_mapping(payload["training_state"])
        self.assertEqual(restored, state)
        self.assertEqual(payload["schema_version"], CHECKPOINT_SCHEMA)
        self.assertNotIn("base_model_state", payload)
        self.assertNotIn("model_state_dict", payload)
        self.assertEqual(
            payload["checkpoint_contents"],
            [
                "lora_adapter",
                "shared_scorer_head",
                "optimizer",
                "rng",
                "training_state",
                "scientific_configuration",
            ],
        )
        with self.assertRaisesRegex(ValueError, "scientific fingerprint mismatch"):
            load_checkpoint_payload(
                path,
                expected_scientific_fingerprint="0" * 64,
            )

    def test_scientific_configuration_excludes_runtime_and_uses_train_only(self):
        configuration = _configuration()
        self.assertEqual(
            configuration["phase3b_integration_fingerprint"],
            EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT,
        )
        self.assertEqual(configuration["dataset"]["training_split"], "train")
        self.assertEqual(configuration["max_packed_tokens"], 2304)
        self.assertIs(configuration["truncation"], False)
        serialized = json.dumps(configuration, sort_keys=True)
        for forbidden in (
            "runtime_torch_version",
            "runtime_transformers_version",
            "runtime_peft_version",
            "cuda_version",
            "gpu_name",
            "timestamp",
        ):
            self.assertNotIn(forbidden, serialized)

    def test_training_metrics_reject_and_omit_sensitive_content(self):
        metric = {
            "anchor_loss": 0.2,
            "checkpoint_reference": "checkpoints/checkpoint_step_00000001.pt",
            "optimizer_step": 1,
            "partial_loss": 1.1,
            "positive_event_rate": 0.5,
            "scientific_fingerprint": "a" * 64,
            "total_loss": 1.3,
        }
        validate_training_artifact_privacy(metric)
        path = self.root / "metrics.jsonl"
        write_training_metric(path, metric)
        contents = path.read_text("utf-8")
        self.assertNotIn("private", contents.lower())
        self.assertNotIn('"history"', contents.lower())
        self.assertNotIn('"title"', contents.lower())
        self.assertNotIn('"label"', contents.lower())
        self.assertIn('"checkpoint_reference"', contents)
        with self.assertRaisesRegex(ValueError, "Sensitive field"):
            validate_training_artifact_privacy({"dialogue_text": "PRIVATE"})


if __name__ == "__main__":
    unittest.main()
