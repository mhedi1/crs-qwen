from __future__ import annotations

import hashlib
import json
import math
import unittest

import torch

from my_crs.build_stage2_v2_dataset import (
    CANDIDATE_ORDER_VERSION,
    DATASET_SCHEMA_VERSION,
    TOP_K,
    canonical_json_digest,
    serialize_candidates,
)
from my_crs.stage2_v2_loss import (
    EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT,
    POSITIVE_ESTIMATOR_WEIGHT,
    TRAIN_POSITIVE_EVENTS,
    TRAIN_RETRIEVAL_COMPLETED_EVENTS,
    compute_v2_batch_loss,
    compute_v2_event_loss,
    contextual_probability_state,
    loss_scientific_configuration,
    training_loss_inputs_from_record,
)


def _scores(*, batch: int = 1) -> torch.Tensor:
    values = torch.linspace(0.01, 0.06, TOP_K, dtype=torch.float64)
    return values.repeat(batch, 1)


def _record(*, split: str = "train", positives: list[int] | None = None) -> dict:
    raw_candidates = []
    for rank in range(1, TOP_K + 1):
        raw_candidates.append(
            {
                "ckg_contribution": 1.0 / (200 + rank),
                "ckg_rank": TOP_K + 1 - rank,
                "id": 700000 + rank,
                "kbrd_contribution": 1.0 / (100 + rank),
                "kbrd_rank": rank,
                "rank": rank,
                "rrf_score": 1.0 / (60 + rank) + 1.0 / (110 + rank),
                "source": "RRF",
                "title": f"Candidate {rank}",
            }
        )
    candidates = serialize_candidates(raw_candidates)
    history = "SEEKER: frozen pre-target history"
    return {
        "candidate_count": TOP_K,
        "candidates": candidates,
        "history": history,
        "history_sha256": hashlib.sha256(history.encode("utf-8")).hexdigest(),
        "instance_key": "fixture:1",
        "observed_positive_serialization_positions": positives or [],
        "schema_version": DATASET_SCHEMA_VERSION,
        "serialization_digest": canonical_json_digest(candidates),
        "serialization_order_version": CANDIDATE_ORDER_VERSION,
        "split": split,
    }


class Stage2V2ProbabilityTests(unittest.TestCase):
    def test_q_and_p_are_normalized_and_zero_residual_recovers_q(self):
        residuals = torch.zeros(2, TOP_K, dtype=torch.float32)
        state = contextual_probability_state(_scores(batch=2), residuals)
        self.assertTrue(
            torch.allclose(
                state.q.sum(dim=-1),
                torch.ones(2, dtype=torch.float64),
                atol=1e-15,
                rtol=0.0,
            )
        )
        self.assertTrue(torch.allclose(state.p, state.q, atol=1e-15, rtol=1e-15))
        self.assertEqual(state.q.dtype, torch.float64)
        self.assertEqual(state.log_p.dtype, torch.float64)

    def test_zero_residual_anchor_is_zero(self):
        result = compute_v2_event_loss(
            _scores()[0],
            torch.zeros(TOP_K),
            [2, 7],
            beta=0.10,
        )
        self.assertAlmostEqual(float(result.anchor_losses[0]), 0.0, delta=1e-15)

    def test_constant_residual_offset_changes_nothing(self):
        residuals = torch.linspace(-2.0, 3.0, TOP_K, dtype=torch.float64)
        first = compute_v2_event_loss(_scores()[0], residuals, [3, 9], beta=0.3)
        second = compute_v2_event_loss(
            _scores()[0],
            residuals + 917.0,
            [3, 9],
            beta=0.3,
        )
        self.assertTrue(
            torch.allclose(
                first.probability_state.p,
                second.probability_state.p,
                atol=2e-14,
                rtol=2e-14,
            )
        )
        self.assertAlmostEqual(
            float(first.total_loss),
            float(second.total_loss),
            delta=2e-13,
        )

    def test_partial_loss_matches_observed_set_probability(self):
        residuals = torch.linspace(-0.7, 0.9, TOP_K)
        positives = [2, 17, 41]
        result = compute_v2_event_loss(
            _scores()[0],
            residuals,
            positives,
            beta=0.0,
        )
        probability = result.probability_state.p[0]
        manual = -torch.log(probability[[position - 1 for position in positives]].sum())
        self.assertAlmostEqual(
            float(result.partial_losses[0]),
            float(manual),
            delta=1e-14,
        )

    def test_multiple_positive_order_is_irrelevant(self):
        residuals = torch.linspace(-1.0, 1.0, TOP_K)
        first = compute_v2_event_loss(_scores()[0], residuals, [1, 11, 50], beta=0.1)
        second = compute_v2_event_loss(_scores()[0], residuals, [50, 1, 11], beta=0.1)
        self.assertEqual(float(first.total_loss), float(second.total_loss))

    def test_absent_positive_set_has_no_partial_but_keeps_anchor(self):
        residuals = torch.linspace(-1.0, 2.0, TOP_K)
        result = compute_v2_event_loss(_scores()[0], residuals, [], beta=0.3)
        self.assertEqual(float(result.partial_losses[0]), 0.0)
        self.assertFalse(bool(result.positive_mask[0]))
        self.assertGreater(float(result.anchor_losses[0]), 0.0)
        self.assertAlmostEqual(
            float(result.total_loss),
            0.3 * float(result.anchor_losses[0]),
            delta=1e-14,
        )

    def test_candidate_permutation_with_labels_preserves_loss(self):
        scores = _scores()[0]
        residuals = torch.linspace(-0.8, 0.6, TOP_K)
        positives = [4, 19, 44]
        original = compute_v2_event_loss(scores, residuals, positives, beta=1.0)
        permutation = torch.randperm(TOP_K, generator=torch.Generator().manual_seed(87))
        inverse = {int(old): new for new, old in enumerate(permutation.tolist())}
        permuted_positives = [inverse[position - 1] + 1 for position in positives]
        permuted = compute_v2_event_loss(
            scores[permutation],
            residuals[permutation],
            permuted_positives,
            beta=1.0,
        )
        self.assertAlmostEqual(
            float(original.total_loss),
            float(permuted.total_loss),
            delta=1e-14,
        )


class Stage2V2EstimatorTests(unittest.TestCase):
    def test_positive_estimator_weight_is_exact_frozen_count_ratio(self):
        self.assertEqual(TRAIN_RETRIEVAL_COMPLETED_EVENTS, 20055)
        self.assertEqual(TRAIN_POSITIVE_EVENTS, 12970)
        self.assertEqual(
            POSITIVE_ESTIMATOR_WEIGHT,
            TRAIN_RETRIEVAL_COMPLETED_EVENTS / TRAIN_POSITIVE_EVENTS,
        )

    def test_batch_estimator_matches_independent_manual_reference(self):
        scores = _scores(batch=3)
        residuals = torch.stack(
            [
                torch.linspace(-0.5, 0.7, TOP_K),
                torch.linspace(0.8, -0.3, TOP_K),
                torch.linspace(-0.1, 0.2, TOP_K),
            ]
        )
        positives = [[2, 8], [], [17]]
        beta = 0.3
        result = compute_v2_batch_loss(scores, residuals, positives, beta=beta)

        manual_events = []
        for event_index in range(3):
            q = scores[event_index] / scores[event_index].sum()
            delta = residuals[event_index].double()
            delta = delta - delta.mean()
            unnormalized = q * torch.exp(delta)
            p = unnormalized / unnormalized.sum()
            anchor = torch.sum(q * (torch.log(q) - torch.log(p)))
            partial = torch.tensor(0.0, dtype=torch.float64)
            if positives[event_index]:
                indices = [position - 1 for position in positives[event_index]]
                partial = -torch.log(p[indices].sum())
            manual_events.append(
                (POSITIVE_ESTIMATOR_WEIGHT * partial if positives[event_index] else 0.0)
                + beta * anchor
            )
        manual = torch.stack(manual_events).mean()
        self.assertAlmostEqual(float(result.total_loss), float(manual), delta=1e-14)
        self.assertAlmostEqual(float(result.positive_event_rate), 2.0 / 3.0, delta=1e-15)

    def test_beta_zero_removes_anchor_and_larger_beta_scales_it_exactly(self):
        scores = _scores(batch=2)
        residuals = torch.stack(
            [torch.linspace(-1.0, 1.0, TOP_K), torch.linspace(0.6, -0.4, TOP_K)]
        )
        positives = [[5], []]
        zero = compute_v2_batch_loss(scores, residuals, positives, beta=0.0)
        one = compute_v2_batch_loss(scores, residuals, positives, beta=1.0)
        three = compute_v2_batch_loss(scores, residuals, positives, beta=3.0)
        anchor_mean = float(one.mean_anchor_over_all_events)
        self.assertAlmostEqual(float(one.total_loss - zero.total_loss), anchor_mean, delta=1e-14)
        self.assertAlmostEqual(
            float(three.total_loss - zero.total_loss),
            3.0 * anchor_mean,
            delta=1e-14,
        )

    def test_gradients_are_finite_prior_is_detached_and_residuals_receive_gradient(self):
        prior = _scores().requires_grad_(True)
        residuals = torch.linspace(-0.3, 0.4, TOP_K).unsqueeze(0).requires_grad_(True)
        result = compute_v2_batch_loss(prior, residuals, [[3, 11]], beta=0.1)
        result.total_loss.backward()
        self.assertIsNone(prior.grad)
        self.assertIsNotNone(residuals.grad)
        self.assertTrue(torch.isfinite(residuals.grad).all())
        self.assertGreater(float(residuals.grad.abs().sum()), 0.0)

    def test_fixed_positive_weight_does_not_depend_on_minibatch_composition(self):
        scores = _scores(batch=2)
        residuals = torch.zeros(2, TOP_K)
        result = compute_v2_batch_loss(scores, residuals, [[1], []], beta=0.0)
        expected_positive = POSITIVE_ESTIMATOR_WEIGHT * float(result.partial_losses[0])
        self.assertAlmostEqual(float(result.per_event_losses[0]), expected_positive, delta=1e-14)
        self.assertEqual(float(result.per_event_losses[1]), 0.0)
        self.assertAlmostEqual(float(result.total_loss), expected_positive / 2.0, delta=1e-14)


class Stage2V2ValidationTests(unittest.TestCase):
    def test_invalid_rrf_scores_fail_closed(self):
        for bad in (0.0, -0.1, float("nan"), float("inf")):
            with self.subTest(bad=bad):
                scores = _scores()
                scores[0, 7] = bad
                with self.assertRaisesRegex(ValueError, "finite and strictly positive"):
                    compute_v2_batch_loss(
                        scores,
                        torch.zeros(1, TOP_K),
                        [[1]],
                        beta=0.1,
                    )

    def test_invalid_or_duplicate_positive_positions_fail_closed(self):
        for positions in ([0], [51], [2, 2], [True]):
            with self.subTest(positions=positions):
                with self.assertRaisesRegex(ValueError, "positive positions"):
                    compute_v2_event_loss(
                        _scores()[0],
                        torch.zeros(TOP_K),
                        positions,
                        beta=0.1,
                    )

    def test_nonfinite_residuals_and_invalid_shapes_fail_closed(self):
        residuals = torch.zeros(1, TOP_K)
        residuals[0, 4] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            contextual_probability_state(_scores(), residuals)
        with self.assertRaisesRegex(ValueError, "shape"):
            contextual_probability_state(
                torch.ones(1, TOP_K - 1),
                torch.zeros(1, TOP_K - 1),
            )

    def test_frozen_record_fields_are_used_without_reinterpreting_text(self):
        record = _record(positives=[2, 37])
        inputs = training_loss_inputs_from_record(record)
        self.assertEqual(inputs.positive_serialization_positions, (2, 37))
        self.assertEqual(len(inputs.rrf_scores), TOP_K)
        self.assertEqual(inputs.instance_key, "fixture:1")
        self.assertTrue(all(score > 0.0 for score in inputs.rrf_scores))

    def test_nontrain_split_and_wrong_candidate_count_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "TRAIN records only"):
            training_loss_inputs_from_record(_record(split="dev", positives=[1]))
        record = _record(positives=[1])
        record["candidate_count"] = TOP_K - 1
        with self.assertRaisesRegex(ValueError, "declare 50 candidates"):
            training_loss_inputs_from_record(record)

    def test_loss_configuration_records_semantics_without_runtime_provenance(self):
        configuration = loss_scientific_configuration(0.10)
        self.assertEqual(
            configuration["phase3b_integration_fingerprint"],
            EXPECTED_PHASE3B_INTEGRATION_FINGERPRINT,
        )
        self.assertEqual(
            configuration["positive_estimator"]["numerator_retrieval_completed_events"],
            20055,
        )
        self.assertEqual(
            configuration["positive_estimator"]["denominator_positive_events"],
            12970,
        )
        self.assertEqual(
            configuration["positive_estimator"]["computed_weight"],
            20055 / 12970,
        )
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


if __name__ == "__main__":
    unittest.main()
