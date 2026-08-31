import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import my_crs.evaluate_rrf_zeroshot as evaluator_module
import my_crs.rrf_list_reranker as reranker_module

from my_crs.evaluate_rrf_zeroshot import (
    EXPECTED_RRF_METRICS,
    EXPECTED_SOURCE_SHA256,
    OFFICIAL_VALID_PATH,
    RankingMetrics,
    ValidEventIndex,
    build_dialogue_up_to,
    evaluate_instance,
    evaluate_rrf_zeroshot,
    get_rank,
    instance_key,
    is_hit,
    load_resume_records,
    normalize_title,
    reconstruct_valid_event_index,
    run_fingerprint,
    validate_frozen_instances_against_valid,
    validate_complete_reranked_recall_at_50,
    validate_full_run_accounting,
    validate_output_paths,
    validate_resume_record,
    validate_resume_subset,
    validate_frozen_rrf_summary,
)
from my_crs.rrf_list_reranker import (
    FALLBACK_INVALID_MODEL_OUTPUT,
    FALLBACK_REQUEST_FAILURE,
    QwenRerankSettings,
    RankedPositionsError,
    build_list_rerank_prompt,
    complete_ranking,
    parse_ranked_positions,
    prompt_template_digest,
    rerank_rrf_candidates,
)


VALID_POSITIONS = [17, 3, 42, 8, 1, 29, 10, 5, 31, 14]
VALID_OUTPUT = json.dumps({"ranked_ids": VALID_POSITIONS}, separators=(",", ":"))


def _candidates(target_position: int | None = None) -> list[dict]:
    candidates = [
        {
            "id": 1000 + position,
            "title": f"Movie {position}",
            "source": "RRF",
            "rrf_score": 1.0 / (60 + position),
        }
        for position in range(1, 51)
    ]
    if target_position is not None:
        candidates[target_position - 1]["title"] = "Target Movie (2000)"
    return candidates


def _settings(**overrides) -> QwenRerankSettings:
    values = {
        "server_url": "http://fixture.invalid/api/chat",
        "model": "fixture-model",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 128,
        "think": False,
        "stream": False,
        "max_retries": 3,
        "timeout": 5,
    }
    values.update(overrides)
    return QwenRerankSettings(**values)


def _successful_post(content: str = VALID_OUTPUT) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"message": {"content": content}}
    return Mock(return_value=response)


def _valid_summary() -> dict:
    return {
        "source_split": "VALID",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "evaluated_conversations": 797,
        "evaluation_instances": 2588,
        "failures": [],
        "extraction_configuration": {
            "resolver_version": "v3",
            "use_legacy_non_movie_entities": True,
            "use_aux_dbpedia_uri_matching": True,
            "use_aux_genre_mapping": True,
            "use_aux_person_matching": False,
            "seed_selection": "all",
        },
        "kbrd_configuration": {
            "retrieval_mode": "kbrd",
            "top_k": 50,
            "use_fusion": False,
            "llm_qwen_used": False,
        },
        "ckg_configuration": {
            "graph_type": "conversation",
            "weighting_method": "conditional",
            "min_support": 2,
            "view": "budget_controlled",
            "top_k": 50,
        },
        "rrf_parameters": {
            "k": 60,
            "weights": {"KBRD": 1.0, "CKG": 1.0},
            "absent_source_contribution": 0.0,
            "raw_scores_used": False,
            "final_candidate_budget": 50,
        },
        "metrics": {"RRF": dict(EXPECTED_RRF_METRICS)},
    }


def _conversation(
    *,
    line_number: int = 1,
    conversation_id: str | None = None,
) -> tuple[dict, dict]:
    conversation_id = conversation_id or f"conversation-{line_number}"
    conversation = {
        "conversationId": conversation_id,
        "initiatorWorkerId": 1,
        "respondentWorkerId": 2,
        "movieMentions": {"99": "Target Movie (2000)"},
        "respondentQuestions": {"99": {"suggested": 1}},
        "messages": [
            {"senderWorkerId": 1, "text": "I want an adventurous movie."},
            {"senderWorkerId": 2, "text": "Do you prefer something modern?"},
            {"senderWorkerId": 1, "text": "Yes, but not a comedy."},
            {"senderWorkerId": 2, "text": "Try @99."},
            {"senderWorkerId": 1, "text": "FUTURE PRIVATE TURN"},
        ],
    }
    frozen_record = {
        "line_number": line_number,
        "conversation_id": conversation_id,
        "turn_index": 3,
        "ground_truth_titles": ["target movie (2000)"],
        "rrf_candidates": _candidates(target_position=17),
    }
    return conversation, frozen_record


def _valid_index(*conversation_records: tuple[dict, dict]) -> ValidEventIndex:
    conversations: dict[int, dict] = {}
    events: dict[str, dict] = {}
    for conversation, record in conversation_records:
        line_number = record["line_number"]
        turn_index = record["turn_index"]
        key = instance_key(line_number, turn_index)
        ground_truth = list(record["ground_truth_titles"])
        conversations[line_number] = conversation
        events[key] = {
            "instance_key": key,
            "line_number": line_number,
            "conversation_id": conversation.get("conversationId"),
            "turn_index": turn_index,
            "ground_truth_titles": ground_truth,
            "normalized_ground_truth_titles": tuple(
                sorted({normalize_title(title) for title in ground_truth})
            ),
        }
    return ValidEventIndex(
        conversations=conversations,
        events=events,
        evaluated_conversations=len(conversations),
    )


def _valid_resume_record(
    conversation: dict,
    frozen_record: dict,
    *,
    fallback: bool = False,
    settings: QwenRerankSettings | None = None,
    post: Mock | None = None,
) -> dict:
    output = "malformed" if fallback else VALID_OUTPUT
    effective_settings = settings or _settings()
    return evaluate_instance(
        frozen_record,
        conversation,
        effective_settings,
        post=post or _successful_post(output),
    )


class TestRRFZeroShotParser(unittest.TestCase):
    def test_valid_exact_top10_json(self):
        self.assertEqual(
            parse_ranked_positions(VALID_OUTPUT, candidate_count=50),
            VALID_POSITIONS,
        )

    def test_optional_single_code_fence(self):
        fenced = f"```json\n{VALID_OUTPUT}\n```"
        self.assertEqual(
            parse_ranked_positions(fenced, candidate_count=50),
            VALID_POSITIONS,
        )

    def test_unknown_keys_are_rejected(self):
        payload = json.dumps({"ranked_ids": VALID_POSITIONS, "explanation": "no"})
        with self.assertRaisesRegex(RankedPositionsError, "only_ranked_ids"):
            parse_ranked_positions(payload, candidate_count=50)

    def test_duplicate_json_ranked_ids_keys_are_rejected(self):
        payload = (
            '{"ranked_ids":[1,2,3,4,5,6,7,8,9,10],'
            '"ranked_ids":[11,12,13,14,15,16,17,18,19,20]}'
        )
        with self.assertRaisesRegex(RankedPositionsError, "duplicate_json_key"):
            parse_ranked_positions(payload, candidate_count=50)

    def test_malformed_json_falls_back(self):
        result = rerank_rrf_candidates(
            "User: fixture",
            _candidates(),
            _settings(),
            post=_successful_post("not-json"),
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, FALLBACK_INVALID_MODEL_OUTPUT)
        self.assertIn("malformed_json", result.fallback_detail)

    def test_wrong_list_length_falls_back(self):
        output = json.dumps({"ranked_ids": VALID_POSITIONS[:9]})
        result = rerank_rrf_candidates(
            "User: fixture", _candidates(), _settings(), post=_successful_post(output)
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, FALLBACK_INVALID_MODEL_OUTPUT)
        self.assertIn("exactly_10", result.fallback_detail)

    def test_duplicate_positions_fall_back(self):
        output = json.dumps({"ranked_ids": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
        result = rerank_rrf_candidates(
            "User: fixture", _candidates(), _settings(), post=_successful_post(output)
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, FALLBACK_INVALID_MODEL_OUTPUT)
        self.assertIn("unique", result.fallback_detail)

    def test_non_integer_position_falls_back(self):
        output = json.dumps({"ranked_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, "10"]})
        result = rerank_rrf_candidates(
            "User: fixture", _candidates(), _settings(), post=_successful_post(output)
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, FALLBACK_INVALID_MODEL_OUTPUT)
        self.assertIn("integers", result.fallback_detail)

    def test_boolean_position_is_not_accepted_as_integer(self):
        output = json.dumps({"ranked_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, True]})
        with self.assertRaisesRegex(RankedPositionsError, "integers"):
            parse_ranked_positions(output, candidate_count=50)

    def test_out_of_range_position_falls_back(self):
        output = json.dumps({"ranked_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 51]})
        result = rerank_rrf_candidates(
            "User: fixture", _candidates(), _settings(), post=_successful_post(output)
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.fallback_reason, FALLBACK_INVALID_MODEL_OUTPUT)
        self.assertIn("out_of_range", result.fallback_detail)

    def test_free_form_unknown_movie_cannot_enter_ranking(self):
        result = rerank_rrf_candidates(
            "User: fixture",
            _candidates(),
            _settings(),
            post=_successful_post("Titanic"),
        )
        self.assertTrue(result.fallback)
        self.assertEqual(
            [candidate["id"] for candidate in result.final_candidates],
            [candidate["id"] for candidate in _candidates()],
        )


class TestRRFZeroShotCompletion(unittest.TestCase):
    def test_deterministic_completion_retains_remaining_rrf_order(self):
        candidates = _candidates()
        final = complete_ranking(candidates, VALID_POSITIONS)
        expected_positions = VALID_POSITIONS + [
            position for position in range(1, 51) if position not in VALID_POSITIONS
        ]
        self.assertEqual(
            [candidate["id"] for candidate in final],
            [1000 + position for position in expected_positions],
        )

    def test_completion_has_exactly_50_unique_candidates(self):
        final = complete_ranking(_candidates(), VALID_POSITIONS)
        ids = [candidate["id"] for candidate in final]
        self.assertEqual(len(ids), 50)
        self.assertEqual(len(ids), len(set(ids)))

    def test_input_list_is_not_mutated(self):
        candidates = _candidates()
        original = copy.deepcopy(candidates)
        complete_ranking(candidates, VALID_POSITIONS)
        self.assertEqual(candidates, original)

    def test_candidate_set_is_preserved(self):
        candidates = _candidates()
        final = complete_ranking(candidates, VALID_POSITIONS)
        self.assertEqual(
            {candidate["id"] for candidate in final},
            {candidate["id"] for candidate in candidates},
        )

    def test_hand_computable_reranked_metrics(self):
        original = _candidates(target_position=17)
        reranked = complete_ranking(original, VALID_POSITIONS)
        ground_truth = ["target movie!"]
        original_rank = get_rank(original, ground_truth)
        reranked_rank = get_rank(reranked, ground_truth)
        metrics = RankingMetrics()
        metrics.add_rank(reranked_rank)

        self.assertEqual(original_rank, 17)
        self.assertEqual(reranked_rank, 1)
        self.assertEqual(
            metrics.result(),
            {
                "instances": 1,
                "Recall@1": 1.0,
                "Recall@10": 1.0,
                "Recall@50": 1.0,
                "MRR": 1.0,
            },
        )

    def test_recall_at_50_invariant_after_valid_reranking(self):
        original = _candidates(target_position=49)
        reranked = complete_ranking(original, VALID_POSITIONS)
        ground_truth = ["Target Movie"]
        self.assertEqual(is_hit(original, ground_truth, 50), is_hit(reranked, ground_truth, 50))

    def test_recall_at_50_invariant_after_fallback(self):
        original = _candidates(target_position=49)
        result = rerank_rrf_candidates(
            "User: fixture",
            original,
            _settings(),
            post=_successful_post("malformed"),
        )
        self.assertTrue(result.fallback)
        self.assertEqual(
            is_hit(original, ["Target Movie"], 50),
            is_hit(result.final_candidates, ["Target Movie"], 50),
        )


class TestRRFZeroShotPromptSanitization(unittest.TestCase):
    def _render_title(self, title: str) -> str:
        return build_list_rerank_prompt("User: fixture", [{"title": title}])[1][
            "content"
        ]

    def test_movie_title_newline_is_sanitized(self):
        rendered = self._render_title("First\nSecond")
        self.assertIn("1. First Second", rendered)
        self.assertNotIn("First\nSecond", rendered)

    def test_movie_title_carriage_return_and_tab_are_sanitized(self):
        rendered = self._render_title("First\rSecond\tThird")
        self.assertIn("1. First Second Third", rendered)
        self.assertNotIn("First\rSecond", rendered)
        self.assertNotIn("Second\tThird", rendered)

    def test_movie_title_non_whitespace_ascii_control_is_sanitized(self):
        rendered = self._render_title("First\x1bSecond\x7fThird")
        self.assertIn("1. First Second Third", rendered)
        self.assertNotIn("\x1b", rendered)
        self.assertNotIn("\x7f", rendered)

    def test_normal_unicode_movie_title_remains_intact(self):
        title = "Amélie – 千と千尋の神隠し"
        self.assertIn(f"1. {title}", self._render_title(title))


class TestRRFZeroShotArtifactAlignment(unittest.TestCase):
    def setUp(self):
        self.conversation_one, self.record_one = _conversation(line_number=1)
        self.conversation_two, self.record_two = _conversation(line_number=2)
        self.index = _valid_index(
            (self.conversation_one, self.record_one),
            (self.conversation_two, self.record_two),
        )

    def _validate(self, records):
        return validate_frozen_instances_against_valid(
            records,
            self.index,
            expected_conversations=2,
            expected_instances=2,
        )

    def test_independently_reconstructed_valid_keys_equal_frozen_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "valid_data.jsonl"
            path.write_text(
                json.dumps(self.conversation_one)
                + "\n"
                + json.dumps(self.conversation_two)
                + "\n",
                encoding="utf-8",
            )
            reconstructed = reconstruct_valid_event_index(path)
        self.assertEqual(set(reconstructed.events), {"1:3", "2:3"})
        validate_frozen_instances_against_valid(
            [self.record_one, self.record_two],
            reconstructed,
            expected_conversations=2,
            expected_instances=2,
        )

    def test_same_count_but_wrong_key_set_is_rejected(self):
        wrong = copy.deepcopy(self.record_two)
        wrong["turn_index"] = 1
        with self.assertRaisesRegex(ValueError, "event-key mismatch"):
            self._validate([self.record_one, wrong])

    def test_missing_expected_key_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing=.*2:3"):
            self._validate([self.record_one])

    def test_unexpected_extra_key_is_rejected(self):
        extra = copy.deepcopy(self.record_two)
        extra["line_number"] = 3
        with self.assertRaisesRegex(ValueError, "unexpected=.*3:3"):
            self._validate([self.record_one, self.record_two, extra])

    def test_empty_artifact_ground_truth_is_rejected(self):
        empty = copy.deepcopy(self.record_one)
        empty["ground_truth_titles"] = []
        with self.assertRaisesRegex(ValueError, "empty ground truth"):
            self._validate([empty, self.record_two])

    def test_mismatching_artifact_valid_ground_truth_is_rejected(self):
        mismatched = copy.deepcopy(self.record_one)
        mismatched["ground_truth_titles"] = ["Different Movie"]
        with self.assertRaisesRegex(ValueError, "ground-truth mismatch"):
            self._validate([mismatched, self.record_two])

    def test_semantic_misalignment_fails_before_qwen(self):
        bad_record = copy.deepcopy(self.record_one)
        bad_record["ground_truth_titles"] = ["Different Movie"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid_data.jsonl"
            valid_path.write_text("fixture", encoding="utf-8")
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(_valid_summary()), encoding="utf-8")
            instances_path = root / "instances.jsonl"
            instances_path.write_text("fixture", encoding="utf-8")
            with patch(
                "my_crs.evaluate_rrf_zeroshot.validate_official_valid_path",
                return_value=valid_path,
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.load_frozen_rrf_instances",
                return_value=[bad_record],
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.reconstruct_valid_event_index",
                return_value=_valid_index((self.conversation_one, self.record_one)),
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.validate_frozen_instances_against_valid",
                side_effect=lambda records, index: validate_frozen_instances_against_valid(
                    records,
                    index,
                    expected_conversations=1,
                    expected_instances=1,
                ),
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.rerank_rrf_candidates"
            ) as rerank:
                with self.assertRaisesRegex(ValueError, "ground-truth mismatch"):
                    evaluate_rrf_zeroshot(
                        rrf_summary_path=summary_path,
                        rrf_instances_path=instances_path,
                        valid_path=valid_path,
                        output_path=root / "output.json",
                        instance_output_path=root / "output.jsonl",
                        max_instances=1,
                    )
            rerank.assert_not_called()

    def test_official_valid_reconstructs_frozen_counts(self):
        reconstructed = reconstruct_valid_event_index(OFFICIAL_VALID_PATH)
        self.assertEqual(reconstructed.evaluated_conversations, 797)
        self.assertEqual(len(reconstructed.events), 2588)


class TestRRFZeroShotResumeValidation(unittest.TestCase):
    def setUp(self):
        self.conversation, self.frozen_record = _conversation()
        self.event = _valid_index((self.conversation, self.frozen_record)).events["1:3"]
        self.settings = _settings()

    def _validate(self, record, settings=None):
        return validate_resume_record(
            record,
            self.frozen_record,
            self.event,
            self.conversation,
            settings or self.settings,
        )

    def test_valid_resume_record_is_accepted(self):
        record = _valid_resume_record(self.conversation, self.frozen_record)
        self.assertEqual(self._validate(record), record)

    def test_valid_parser_fallback_provenance_is_accepted(self):
        record = _valid_resume_record(
            self.conversation,
            self.frozen_record,
            fallback=True,
        )
        self.assertEqual(record["fallback_reason"], FALLBACK_INVALID_MODEL_OUTPUT)
        self.assertEqual(self._validate(record), record)

    def test_arbitrary_fallback_reason_is_rejected(self):
        record = _valid_resume_record(
            self.conversation,
            self.frozen_record,
            fallback=True,
        )
        record["fallback_reason"] = "arbitrary_exception_text"
        with self.assertRaisesRegex(ValueError, "fallback reason is unknown"):
            self._validate(record)

    def test_corrupted_fallback_provenance_cannot_reach_aggregate_summary(self):
        settings = _settings()
        record = _valid_resume_record(
            self.conversation,
            self.frozen_record,
            fallback=True,
        )
        record["fallback_reason"] = "forged_aggregate_category"
        valid_index = _valid_index((self.conversation, self.frozen_record))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid_data.jsonl"
            valid_path.write_text("fixture", encoding="utf-8")
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(_valid_summary()), encoding="utf-8")
            instances_path = root / "instances.jsonl"
            instances_path.write_text("fixture-artifact", encoding="utf-8")
            record["run_fingerprint"] = run_fingerprint(
                summary_path=summary_path,
                instances_path=instances_path,
                settings=settings,
            )
            output_instances = root / "result.instances.jsonl"
            output_instances.write_text(json.dumps(record) + "\n", encoding="utf-8")
            post = _successful_post()

            with patch(
                "my_crs.evaluate_rrf_zeroshot.validate_official_valid_path",
                return_value=valid_path,
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.load_frozen_rrf_instances",
                return_value=[self.frozen_record],
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.reconstruct_valid_event_index",
                return_value=valid_index,
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.validate_frozen_instances_against_valid",
                return_value={"1:3": self.frozen_record},
            ):
                with self.assertRaisesRegex(ValueError, "fallback reason is unknown"):
                    evaluate_rrf_zeroshot(
                        rrf_summary_path=summary_path,
                        rrf_instances_path=instances_path,
                        valid_path=valid_path,
                        output_path=root / "result.json",
                        instance_output_path=output_instances,
                        settings=settings,
                        max_instances=1,
                        resume=True,
                        post=post,
                    )
            post.assert_not_called()

    def test_valid_request_failure_provenance_is_accepted(self):
        settings = _settings(max_retries=2)
        record = _valid_resume_record(
            self.conversation,
            self.frozen_record,
            settings=settings,
            post=Mock(side_effect=requests.exceptions.Timeout("offline")),
        )
        self.assertEqual(record["fallback_reason"], FALLBACK_REQUEST_FAILURE)
        self.assertEqual(self._validate(record, settings), record)

    def test_parser_fallback_without_raw_output_is_rejected(self):
        record = _valid_resume_record(
            self.conversation,
            self.frozen_record,
            fallback=True,
        )
        record["raw_qwen_output"] = None
        with self.assertRaisesRegex(ValueError, "lacks raw output"):
            self._validate(record)

    def test_request_failure_with_raw_output_is_rejected(self):
        settings = _settings(max_retries=2)
        record = _valid_resume_record(
            self.conversation,
            self.frozen_record,
            settings=settings,
            post=Mock(side_effect=requests.exceptions.Timeout("offline")),
        )
        record["raw_qwen_output"] = "impossible model output"
        with self.assertRaisesRegex(ValueError, "request-failure provenance"):
            self._validate(record, settings)

    def test_zero_request_attempts_is_rejected(self):
        record = _valid_resume_record(self.conversation, self.frozen_record)
        record["request_attempts"] = 0
        with self.assertRaisesRegex(ValueError, "request attempts are invalid"):
            self._validate(record)

    def test_attempts_above_retry_maximum_are_rejected(self):
        record = _valid_resume_record(self.conversation, self.frozen_record)
        record["request_attempts"] = self.settings.max_retries + 1
        with self.assertRaisesRegex(ValueError, "request attempts are invalid"):
            self._validate(record)

    def test_valid_maximum_attempt_count_is_accepted(self):
        settings = _settings(max_retries=2)
        success = _successful_post().return_value
        record = _valid_resume_record(
            self.conversation,
            self.frozen_record,
            settings=settings,
            post=Mock(
                side_effect=[requests.exceptions.Timeout("first timeout"), success]
            ),
        )
        self.assertEqual(record["request_attempts"], settings.max_retries)
        self.assertEqual(self._validate(record, settings), record)

    def test_corrupted_resume_original_candidate_order_is_rejected(self):
        record = _valid_resume_record(self.conversation, self.frozen_record)
        record["original_rrf_candidate_order"][0]["title"] = "Corrupted"
        with self.assertRaisesRegex(ValueError, "original candidate order"):
            self._validate(record)

    def test_corrupted_resume_final_candidate_set_is_rejected(self):
        record = _valid_resume_record(self.conversation, self.frozen_record)
        record["final_complete_top50_order"][0]["id"] = 999999
        with self.assertRaisesRegex(ValueError, "changed the candidate set"):
            self._validate(record)

    def test_corrupted_stored_target_rank_is_rejected(self):
        record = _valid_resume_record(self.conversation, self.frozen_record)
        record["reranked_target_rank"] = 2
        with self.assertRaisesRegex(ValueError, "reranked_target_rank"):
            self._validate(record)

    def test_corrupted_hit_flags_are_rejected(self):
        record = _valid_resume_record(self.conversation, self.frozen_record)
        record["hit_at_10"] = False
        with self.assertRaisesRegex(ValueError, "hit_at_10"):
            self._validate(record)

    def test_corrupted_reciprocal_rank_is_rejected(self):
        record = _valid_resume_record(self.conversation, self.frozen_record)
        record["reciprocal_rank"] = 0.5
        with self.assertRaisesRegex(ValueError, "reciprocal rank"):
            self._validate(record)

    def test_corrupted_history_hash_is_rejected(self):
        record = _valid_resume_record(self.conversation, self.frozen_record)
        record["dialogue_history_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "dialogue_history_sha256"):
            self._validate(record)

    def test_fallback_resume_record_with_changed_ranking_is_rejected(self):
        record = _valid_resume_record(
            self.conversation,
            self.frozen_record,
            fallback=True,
        )
        first = record["final_complete_top50_order"][0]
        second = record["final_complete_top50_order"][1]
        record["final_complete_top50_order"][0] = {**second, "position": 1}
        record["final_complete_top50_order"][1] = {**first, "position": 2}
        with self.assertRaisesRegex(ValueError, "fallback changed candidate order"):
            self._validate(record)

    def test_truncated_final_resume_jsonl_line_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.jsonl"
            path.write_text(
                json.dumps({"instance_key": "1:3", "run_fingerprint": "fingerprint"})
                + "\n{\"instance_key\":",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Malformed resume JSONL"):
                load_resume_records(path, expected_fingerprint="fingerprint")

    def test_max_instances_resume_rejects_records_outside_requested_subset(self):
        with self.assertRaisesRegex(ValueError, "outside the requested"):
            validate_resume_subset({"1:3", "2:3"}, ["1:3"])


class TestRRFZeroShotAccountingAndIdentity(unittest.TestCase):
    def _paths(self, root: Path) -> dict:
        return {
            "valid_path": root / "valid_data.jsonl",
            "rrf_summary_path": root / "frozen-summary.json",
            "rrf_instances_path": root / "frozen-instances.jsonl",
            "output_path": root / "output.json",
            "instance_output_path": root / "output.jsonl",
        }

    def test_output_path_colliding_with_valid_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            paths["output_path"] = paths["valid_path"]
            with self.assertRaisesRegex(ValueError, "VALID dataset"):
                validate_output_paths(**paths)

    def test_output_path_colliding_with_rrf_summary_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            paths["output_path"] = paths["rrf_summary_path"]
            with self.assertRaisesRegex(ValueError, "frozen RRF summary"):
                validate_output_paths(**paths)

    def test_output_path_colliding_with_rrf_instances_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            paths["instance_output_path"] = paths["rrf_instances_path"]
            with self.assertRaisesRegex(ValueError, "frozen RRF instances"):
                validate_output_paths(**paths)

    def test_summary_and_instance_outputs_cannot_be_the_same_file(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self._paths(Path(directory))
            paths["instance_output_path"] = paths["output_path"]
            with self.assertRaisesRegex(ValueError, "summary output equals instance output"):
                validate_output_paths(**paths)

    def test_full_run_requires_797_conversations(self):
        with self.assertRaisesRegex(RuntimeError, "conversation count"):
            validate_full_run_accounting(
                processed_instances=2588,
                processed_conversations=796,
                full_universe_requested=True,
            )

    def test_full_run_requires_2588_instances(self):
        with self.assertRaisesRegex(RuntimeError, "instance count"):
            validate_full_run_accounting(
                processed_instances=2587,
                processed_conversations=797,
                full_universe_requested=True,
            )

    def test_subset_does_not_enforce_full_run_counts(self):
        self.assertFalse(
            validate_full_run_accounting(
                processed_instances=5,
                processed_conversations=2,
                full_universe_requested=False,
            )
        )

    def test_complete_run_recall_at_50_frozen_scalar_gate(self):
        changed_metrics = dict(EXPECTED_RRF_METRICS)
        changed_metrics["Recall@50"] += 1e-6
        validate_complete_reranked_recall_at_50(
            changed_metrics,
            complete=False,
        )
        with self.assertRaisesRegex(ValueError, "reranked_metrics.Recall@50"):
            validate_complete_reranked_recall_at_50(
                changed_metrics,
                complete=True,
            )

    def test_prompt_text_change_changes_template_digest_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            instances = Path(directory) / "instances.jsonl"
            summary.write_text("summary", encoding="utf-8")
            instances.write_text("instances", encoding="utf-8")
            original_digest = prompt_template_digest()
            original_fingerprint = run_fingerprint(
                summary_path=summary,
                instances_path=instances,
                settings=_settings(),
            )
            with patch.object(
                reranker_module,
                "USER_PROMPT_TEMPLATE",
                reranker_module.USER_PROMPT_TEMPLATE + "\nChanged prompt text",
            ):
                changed_digest = prompt_template_digest()
                changed_fingerprint = run_fingerprint(
                    summary_path=summary,
                    instances_path=instances,
                    settings=_settings(),
                )
        self.assertNotEqual(original_digest, changed_digest)
        self.assertNotEqual(original_fingerprint, changed_fingerprint)

    def test_candidate_serialization_change_changes_prompt_digest(self):
        original_digest = prompt_template_digest()
        with patch.object(
            reranker_module,
            "_serialize_candidate_line",
            side_effect=lambda position, title: f"[{position}] {title}",
        ):
            changed_digest = prompt_template_digest()
        self.assertNotEqual(original_digest, changed_digest)

    def test_title_sanitization_change_changes_prompt_digest(self):
        original_digest = prompt_template_digest()
        with patch.object(
            reranker_module,
            "_single_line_title",
            side_effect=lambda title: f"UNSANITIZED:{title}",
        ):
            changed_digest = prompt_template_digest()
        self.assertNotEqual(original_digest, changed_digest)

    def test_required_top_n_change_changes_prompt_digest(self):
        original_digest = prompt_template_digest()
        with patch.object(reranker_module, "TOP_N", reranker_module.TOP_N + 1):
            changed_digest = prompt_template_digest()
        self.assertNotEqual(original_digest, changed_digest)

    def test_model_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            instances = Path(directory) / "instances.jsonl"
            summary.write_text("summary", encoding="utf-8")
            instances.write_text("instances", encoding="utf-8")
            first = run_fingerprint(
                summary_path=summary,
                instances_path=instances,
                settings=_settings(model="model-a"),
            )
            second = run_fingerprint(
                summary_path=summary,
                instances_path=instances,
                settings=_settings(model="model-b"),
            )
        self.assertNotEqual(first, second)

    def test_decoding_change_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            instances = Path(directory) / "instances.jsonl"
            summary.write_text("summary", encoding="utf-8")
            instances.write_text("instances", encoding="utf-8")
            first = run_fingerprint(
                summary_path=summary,
                instances_path=instances,
                settings=_settings(top_p=1.0),
            )
            second = run_fingerprint(
                summary_path=summary,
                instances_path=instances,
                settings=_settings(top_p=0.9),
            )
        self.assertNotEqual(first, second)

    def test_multiple_ground_truth_ranking_semantics(self):
        candidates = _candidates()
        candidates[8]["title"] = "Second Target (1999)"
        candidates[3]["title"] = "First Target!"
        self.assertEqual(
            get_rank(candidates, ["missing", "second target", "first target"]),
            4,
        )


class TestRRFZeroShotSafetyAndProvenance(unittest.TestCase):
    def test_full_pre_target_history_without_target_or_future(self):
        conversation, _ = _conversation()
        history = build_dialogue_up_to(conversation, 2)
        self.assertIn("I want an adventurous movie", history)
        self.assertIn("Yes, but not a comedy", history)
        self.assertNotIn("Try Target Movie", history)
        self.assertNotIn("FUTURE PRIVATE TURN", history)

    def test_prompt_does_not_truncate_full_supplied_history(self):
        history = "\n".join(f"User: history line {index}" for index in range(1, 9))
        prompt = build_list_rerank_prompt(history, _candidates())
        self.assertIn("history line 1", prompt[1]["content"])
        self.assertIn("history line 8", prompt[1]["content"])

    def test_ground_truth_is_not_inserted_into_prompt(self):
        history = "User: I want an adventurous movie"
        prompt = build_list_rerank_prompt(history, _candidates())
        rendered = json.dumps(prompt)
        self.assertNotIn("Secret Ground Truth Movie", rendered)

    def test_test_path_rejected_before_artifact_or_qwen_work(self):
        with tempfile.TemporaryDirectory() as directory:
            test_path = Path(directory) / "test_data.jsonl"
            test_path.write_text("DO NOT READ", encoding="utf-8")
            with patch(
                "my_crs.evaluate_rrf_zeroshot.validate_frozen_rrf_summary"
            ) as validate_summary, patch(
                "my_crs.evaluate_rrf_zeroshot.rerank_rrf_candidates"
            ) as rerank:
                with self.assertRaisesRegex(ValueError, "VALID-only"):
                    evaluate_rrf_zeroshot(
                        valid_path=test_path,
                        rrf_summary_path=Path(directory) / "missing-summary.json",
                        rrf_instances_path=Path(directory) / "missing-instances.jsonl",
                        output_path=Path(directory) / "output.json",
                        instance_output_path=Path(directory) / "output.jsonl",
                    )
            validate_summary.assert_not_called()
            rerank.assert_not_called()

    def test_frozen_rrf_summary_provenance_validation(self):
        summary = _valid_summary()
        validate_frozen_rrf_summary(summary)
        summary["source_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            validate_frozen_rrf_summary(summary)

    def test_extraction_configuration_validation(self):
        summary = _valid_summary()
        summary["extraction_configuration"]["use_aux_person_matching"] = True
        with self.assertRaisesRegex(ValueError, "use_aux_person_matching"):
            validate_frozen_rrf_summary(summary)

    def test_rrf_configuration_and_metric_validation(self):
        for field, bad_value in (("k", 61), ("final_candidate_budget", 49)):
            summary = _valid_summary()
            summary["rrf_parameters"][field] = bad_value
            with self.assertRaisesRegex(ValueError, field):
                validate_frozen_rrf_summary(summary)

        summary = _valid_summary()
        summary["rrf_parameters"]["weights"]["CKG"] = 0.5
        with self.assertRaisesRegex(ValueError, "weights.CKG"):
            validate_frozen_rrf_summary(summary)

        summary = _valid_summary()
        summary["metrics"]["RRF"]["Recall@50"] += 1e-6
        with self.assertRaisesRegex(ValueError, "Recall@50"):
            validate_frozen_rrf_summary(summary)

    def test_http_failure_retries_then_falls_back(self):
        post = Mock(side_effect=requests.exceptions.Timeout("offline"))
        candidates = _candidates()
        result = rerank_rrf_candidates(
            "User: fixture",
            candidates,
            _settings(max_retries=2),
            post=post,
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.request_attempts, 2)
        self.assertEqual(result.successful_requests, 0)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(result.final_candidates, candidates)

    def test_timeout_then_success_records_two_attempts_and_one_success(self):
        success = _successful_post().return_value
        post = Mock(side_effect=[requests.exceptions.Timeout("first timeout"), success])
        result = rerank_rrf_candidates(
            "User: fixture",
            _candidates(),
            _settings(max_retries=2),
            post=post,
        )
        self.assertFalse(result.fallback)
        self.assertEqual(result.request_attempts, 2)
        self.assertEqual(result.successful_requests, 1)
        self.assertEqual(post.call_count, 2)

    def test_malformed_api_response_falls_back_without_retry(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"unexpected": "shape"}
        post = Mock(return_value=response)
        result = rerank_rrf_candidates(
            "User: fixture", _candidates(), _settings(), post=post
        )
        self.assertTrue(result.fallback)
        self.assertEqual(result.request_attempts, 1)
        self.assertEqual(result.successful_requests, 1)
        post.assert_called_once()

    def test_exactly_one_successful_qwen_call_and_bounded_payload(self):
        post = _successful_post()
        settings = _settings()
        result = rerank_rrf_candidates(
            "User: fixture", _candidates(), settings, post=post
        )
        self.assertFalse(result.fallback)
        self.assertEqual(result.successful_requests, 1)
        post.assert_called_once()
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "fixture-model")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(
            payload["options"],
            {"temperature": 0.0, "top_p": 1.0, "num_predict": 128},
        )

    def test_response_generator_is_never_called(self):
        poison = types.ModuleType("response_generator")
        poison.generate_response = Mock(side_effect=AssertionError("must not be called"))
        with patch.dict(
            sys.modules,
            {"response_generator": poison, "my_crs.response_generator": poison},
        ):
            result = rerank_rrf_candidates(
                "User: fixture",
                _candidates(),
                _settings(),
                post=_successful_post(),
            )
        self.assertFalse(result.fallback)
        poison.generate_response.assert_not_called()

    def test_per_instance_provenance_fields_and_no_leakage(self):
        conversation, frozen_record = _conversation()
        record = evaluate_instance(
            frozen_record,
            conversation,
            _settings(),
            post=_successful_post(),
        )
        required = {
            "line_number",
            "conversation_id",
            "turn_index",
            "ground_truth_titles",
            "dialogue_history",
            "dialogue_history_sha256",
            "original_rrf_candidate_order",
            "prompt_candidate_positions",
            "raw_qwen_output",
            "parsed_top10_local_positions",
            "parsed_canonical_candidate_ids",
            "parsed_candidate_titles",
            "final_complete_top50_order",
            "fallback",
            "fallback_reason",
            "fallback_detail",
            "request_attempts",
            "original_rrf_target_rank",
            "reranked_target_rank",
            "hit_at_1",
            "hit_at_10",
            "hit_at_50",
            "reciprocal_rank",
        }
        self.assertTrue(required.issubset(record))
        self.assertNotIn("Try Target Movie", record["dialogue_history"])
        self.assertNotIn("FUTURE PRIVATE TURN", record["dialogue_history"])
        self.assertEqual(record["original_rrf_target_rank"], 17)
        self.assertEqual(record["reranked_target_rank"], 1)
        self.assertEqual(record["parsed_top10_local_positions"], VALID_POSITIONS)

    def test_evaluate_instance_prompt_excludes_targets_future_and_metadata(self):
        conversation, frozen_record = _conversation()
        conversation["movieMentions"]["98"] = "Secret Ground Truth (2001)"
        conversation["respondentQuestions"]["98"] = {"suggested": 1}
        conversation["messages"][3]["text"] = "LEAK_TARGET_TURN @99 and @98."
        conversation["messages"][4]["text"] = "LEAK_FUTURE_TURN"
        frozen_record["ground_truth_titles"] = [
            "target movie (2000)",
            "secret ground truth (2001)",
        ]
        frozen_record["rrf_candidates"][0].update(
            {"kbrd_rank": 1, "ckg_rank": 2, "rrf_rank": 1}
        )
        post = _successful_post()

        evaluate_instance(
            frozen_record,
            conversation,
            _settings(),
            post=post,
        )

        rendered_messages = json.dumps(
            post.call_args.kwargs["json"]["messages"],
            ensure_ascii=False,
        )
        self.assertIn("I want an adventurous movie.", rendered_messages)
        self.assertNotIn("LEAK_TARGET_TURN", rendered_messages)
        self.assertNotIn("LEAK_FUTURE_TURN", rendered_messages)
        self.assertNotIn("Secret Ground Truth", rendered_messages)
        for hidden_field in (
            "ground_truth_titles",
            "rrf_score",
            "kbrd_rank",
            "ckg_rank",
            "rrf_rank",
        ):
            self.assertNotIn(hidden_field, rendered_messages)

    def test_max_instances_and_fingerprinted_resume(self):
        conversation_one, record_one = _conversation(line_number=1)
        conversation_two, record_two = _conversation(line_number=2)
        settings = _settings()
        valid_index = _valid_index(
            (conversation_one, record_one),
            (conversation_two, record_two),
        )
        frozen_by_key = {"1:3": record_one, "2:3": record_two}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid_data.jsonl"
            valid_path.write_text("fixture", encoding="utf-8")
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(_valid_summary()), encoding="utf-8")
            instances_path = root / "instances.jsonl"
            instances_path.write_text("fixture-artifact", encoding="utf-8")
            output_path = root / "result.json"
            output_instances = root / "result.instances.jsonl"

            first_post = _successful_post()
            with patch(
                "my_crs.evaluate_rrf_zeroshot.validate_official_valid_path",
                return_value=valid_path,
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.load_frozen_rrf_instances",
                return_value=[record_one, record_two],
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.reconstruct_valid_event_index",
                return_value=valid_index,
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.validate_frozen_instances_against_valid",
                return_value=frozen_by_key,
            ):
                first = evaluate_rrf_zeroshot(
                    rrf_summary_path=summary_path,
                    rrf_instances_path=instances_path,
                    valid_path=valid_path,
                    output_path=output_path,
                    instance_output_path=output_instances,
                    settings=settings,
                    max_instances=1,
                    post=first_post,
                )
            self.assertEqual(first["processed_instances"], 1)
            self.assertEqual(first_post.call_count, 1)
            self.assertEqual(first["prompt_template_sha256"], prompt_template_digest())

            resume_post = _successful_post()
            with patch(
                "my_crs.evaluate_rrf_zeroshot.validate_official_valid_path",
                return_value=valid_path,
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.load_frozen_rrf_instances",
                return_value=[record_one, record_two],
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.reconstruct_valid_event_index",
                return_value=valid_index,
            ), patch(
                "my_crs.evaluate_rrf_zeroshot.validate_frozen_instances_against_valid",
                return_value=frozen_by_key,
            ):
                resumed = evaluate_rrf_zeroshot(
                    rrf_summary_path=summary_path,
                    rrf_instances_path=instances_path,
                    valid_path=valid_path,
                    output_path=output_path,
                    instance_output_path=output_instances,
                    settings=settings,
                    max_instances=2,
                    resume=True,
                    post=resume_post,
                )

            records = [
                json.loads(line)
                for line in output_instances.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(resumed["processed_instances"], 2)
        self.assertEqual(resume_post.call_count, 1)
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record["instance_key"] for record in records}), 2)

    def test_resume_rejects_duplicate_instance_keys(self):
        record = {
            "instance_key": "1:3",
            "run_fingerprint": "fingerprint",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.jsonl"
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate resume"):
                load_resume_records(path, expected_fingerprint="fingerprint")


if __name__ == "__main__":
    unittest.main()
