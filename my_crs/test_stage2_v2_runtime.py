from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from my_crs.stage2_v2_runtime import (
    FROZEN_TOP_K,
    Stage2V2InvariantError,
    Stage2V2ReadinessError,
    Stage2V2Runtime,
    _load_frozen_bindings,
)


try:
    import torch as _torch
except ImportError:
    _torch = None


class _FakeStage2Bindings:
    top_k = FROZEN_TOP_K
    dataset_schema_version = "stage2_v2_candidates_v1"
    candidate_order_version = "stage2_v2_candidate_order_v1"

    def __init__(self) -> None:
        self.checkpoint_validations = 0
        self.stack_loads = 0
        self.last_record = None

    @staticmethod
    def serialize_candidates(candidates):
        serialized = []
        for position, candidate in enumerate(
            sorted(candidates, key=lambda item: -int(item["id"])), 1
        ):
            serialized.append(
                {
                    "canonical_entity_id": int(candidate["id"]),
                    "ckg_contribution": float(candidate["ckg_contribution"]),
                    "ckg_rank": candidate["ckg_rank"],
                    "kbrd_contribution": float(candidate["kbrd_contribution"]),
                    "kbrd_rank": candidate["kbrd_rank"],
                    "local_id": f"C{position:02d}",
                    "rrf_rank": int(candidate["rank"]),
                    "rrf_score": float(candidate["rrf_score"]),
                    "serialization_position": position,
                    "source": "RRF",
                    "title_original": candidate["title"],
                    "title_sanitized": candidate["title"],
                }
            )
        return serialized

    @staticmethod
    def canonical_json_digest(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def canonicalize_phase1_candidates(record):
        return list(record["candidates"])

    def validate_selected_checkpoint(self, path):
        self.checkpoint_validations += 1
        return SimpleNamespace(
            path=Path(path),
            sha256="selected-sha",
            scientific_fingerprint="selected-fingerprint",
            optimizer_step=1254,
            scientific_configuration={"loss": {"beta": 1.0}},
        )

    @staticmethod
    def require_single_cuda_device(device):
        if device != "cuda:0":
            raise RuntimeError("wrong device")
        return device

    def load_inference_stack(self, checkpoint, device):
        self.stack_loads += 1
        return object(), object(), object(), {"requested_model_id": "Qwen/Qwen2.5-3B-Instruct"}, "revision"

    def infer_residuals(self, record, tokenizer, ranker, device):
        self.last_record = record
        return [float(index) / 10.0 for index in range(FROZEN_TOP_K)], 321

    @staticmethod
    def combine_rrf_prior(rrf_scores, residuals):
        total = sum(rrf_scores)
        log_prior = [math.log(score / total) for score in rrf_scores]
        mean = sum(residuals) / len(residuals)
        centered = [value - mean for value in residuals]
        final = [left + right for left, right in zip(log_prior, centered)]
        return SimpleNamespace(
            centered_residuals=centered,
            final_scores=final,
            log_prior=log_prior,
        )

    @staticmethod
    def rank_candidate_ids(final_scores, canonical_ids, rrf_ranks):
        order = sorted(
            range(FROZEN_TOP_K),
            key=lambda index: (-final_scores[index], rrf_ranks[index], canonical_ids[index]),
        )
        return [canonical_ids[index] for index in order]


def _rrf_candidates():
    return [
        {
            "id": entity_id,
            "title": f"Movie {entity_id}",
            "source": "RRF",
            "rrf_score": 1.0 / (60 + entity_id),
            "kbrd_rank": entity_id,
            "ckg_rank": None,
            "kbrd_contribution": 1.0 / (60 + entity_id),
            "ckg_contribution": 0.0,
        }
        for entity_id in range(1, FROZEN_TOP_K + 1)
    ]


class Stage2V2RuntimeTests(unittest.TestCase):
    def test_loads_selected_stack_once_and_preserves_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_step_00001254.pt"
            checkpoint.touch()
            bindings = _FakeStage2Bindings()
            runtime = Stage2V2Runtime(checkpoint, "cuda:0", _bindings=bindings)

            first_state = runtime.ensure_ready()
            second_state = runtime.ensure_ready()
            result = runtime.rank("User: something thoughtful", _rrf_candidates())

        self.assertIs(first_state, second_state)
        self.assertEqual(bindings.checkpoint_validations, 1)
        self.assertEqual(bindings.stack_loads, 1)
        self.assertEqual(first_state.checkpoint_step, 1254)
        self.assertEqual(first_state.beta, 1.0)
        self.assertEqual(len(result["ranked_candidates"]), FROZEN_TOP_K)
        self.assertEqual(
            {candidate["id"] for candidate in result["ranked_candidates"]},
            set(range(1, FROZEN_TOP_K + 1)),
        )
        self.assertTrue(result["diagnostics"]["candidate_membership_preserved"])
        self.assertEqual(result["diagnostics"]["actual_packed_tokens"], 321)
        self.assertEqual(
            bindings.last_record["history_sha256"],
            __import__("hashlib").sha256(b"User: something thoughtful").hexdigest(),
        )

    def test_missing_checkpoint_is_an_explicit_readiness_error(self):
        bindings = _FakeStage2Bindings()
        runtime = Stage2V2Runtime("definitely-missing-step-1254.pt", _bindings=bindings)
        with self.assertRaisesRegex(Stage2V2ReadinessError, "checkpoint is missing"):
            runtime.ensure_ready()
        self.assertEqual(bindings.checkpoint_validations, 0)

    def test_rejects_any_non_top50_input(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint_step_00001254.pt"
            checkpoint.touch()
            runtime = Stage2V2Runtime(
                checkpoint,
                _bindings=_FakeStage2Bindings(),
            )
            with self.assertRaisesRegex(Stage2V2InvariantError, "exactly 50"):
                runtime.rank("history", _rrf_candidates()[:-1])


@unittest.skipUnless(_torch is not None, "real frozen numeric test requires PyTorch")
class RealFrozenNumericPathTests(unittest.TestCase):
    def test_runtime_matches_frozen_numeric_path_centering_and_tie_breaking(self):
        torch = _torch
        transformers_context = nullcontext()
        if importlib.util.find_spec("transformers") is None:
            transformers_stub = ModuleType("transformers")
            transformers_stub.__version__ = "numeric-path-test-stub"
            transformers_context = patch.dict(
                sys.modules,
                {"transformers": transformers_stub},
            )

        with transformers_context:
            bindings = _load_frozen_bindings()
            from my_crs.joint_rrf_ranker import (
                combine_rrf_prior as frozen_combine_rrf_prior,
                rank_candidate_ids as frozen_rank_candidate_ids,
            )

            residual_holder = {
                "value": torch.linspace(
                    -0.75,
                    0.85,
                    FROZEN_TOP_K,
                    dtype=torch.float32,
                )
            }
            record_holder = {}

            bindings.validate_selected_checkpoint = lambda path: SimpleNamespace(
                path=Path(path),
                sha256="selected-sha",
                scientific_fingerprint="selected-fingerprint",
                optimizer_step=1254,
                scientific_configuration={"loss": {"beta": 1.0}},
            )
            bindings.require_single_cuda_device = lambda device: torch.device("cpu")
            bindings.load_inference_stack = lambda checkpoint, device: (
                object(),
                object(),
                object(),
                {"requested_model_id": "Qwen/Qwen2.5-3B-Instruct"},
                "revision",
            )

            def stub_model_inference(record, tokenizer, ranker, device):
                record_holder["value"] = record
                return residual_holder["value"].clone(), 321

            bindings.infer_residuals = stub_model_inference

            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "checkpoint_step_00001254.pt"
                checkpoint.touch()
                runtime = Stage2V2Runtime(
                    checkpoint,
                    "cpu",
                    _bindings=bindings,
                )
                runtime_result = runtime.rank("fixed history", _rrf_candidates())

                canonical = bindings.canonicalize_phase1_candidates(
                    record_holder["value"]
                )
                canonical_ids = [
                    int(candidate["canonical_entity_id"])
                    for candidate in canonical
                ]
                rrf_ranks = [int(candidate["rrf_rank"]) for candidate in canonical]
                rrf_scores = [
                    float(candidate["rrf_score"])
                    for candidate in canonical
                ]
                direct_prior = torch.tensor(
                    rrf_scores,
                    dtype=torch.float64,
                    device=residual_holder["value"].device,
                )
                direct_combination = frozen_combine_rrf_prior(
                    direct_prior,
                    residual_holder["value"],
                )
                direct_ranking = frozen_rank_candidate_ids(
                    direct_combination.final_scores,
                    canonical_ids,
                    rrf_ranks,
                )
                self.assertEqual(
                    [candidate["id"] for candidate in runtime_result["ranked_candidates"]],
                    direct_ranking,
                )
                self.assertEqual(direct_combination.log_prior.dtype, torch.float64)
                self.assertEqual(
                    direct_combination.log_prior.device,
                    residual_holder["value"].device,
                )

                residual_holder["value"] = residual_holder["value"] + 17.0
                shifted_result = runtime.rank("fixed history", _rrf_candidates())
                self.assertEqual(
                    [candidate["id"] for candidate in shifted_result["ranked_candidates"]],
                    direct_ranking,
                )

            tie_scores = torch.zeros(FROZEN_TOP_K, dtype=torch.float64)
            tie_scores[-1] = 2.0
            tie_scores[0] = 1.0
            tie_scores[1] = 1.0
            tie_ids = list(range(1000, 1000 + FROZEN_TOP_K))
            tie_ids[0], tie_ids[1] = 900, 800
            tie_ranks = list(range(1, FROZEN_TOP_K + 1))
            tie_ranks[0], tie_ranks[1] = 2, 1
            tie_ranking = frozen_rank_candidate_ids(
                tie_scores,
                tie_ids,
                tie_ranks,
            )
            self.assertEqual(tie_ranking[:3], [tie_ids[-1], tie_ids[1], tie_ids[0]])

            # Valid records have unique RRF ranks, so the entity-ID tertiary key
            # cannot be isolated numerically; verify that the frozen function's
            # final key still contains it after score and RRF rank.
            compact_source = " ".join(
                inspect.getsource(frozen_rank_candidate_ids).split()
            )
            self.assertIn(
                "(-numeric_scores[index], ranks[index], ids[index])",
                compact_source,
            )


if __name__ == "__main__":
    unittest.main()
