import copy
import json
import tempfile
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import my_crs.evaluate_rrf_hf as evaluator_module
import my_crs.rrf_list_reranker as shared_reranker
from my_crs.evaluate_rrf_hf import (
    FALLBACK_GENERATION_FAILURE,
    evaluate_hf_instance,
    evaluate_rrf_hf,
    hf_run_fingerprint,
    validate_hf_resume_record,
)
from my_crs.evaluate_rrf_zeroshot import (
    EXPECTED_RRF_METRICS,
    EXPECTED_SOURCE_SHA256,
    ValidEventIndex,
    get_rank,
    instance_key,
    normalize_title,
    validate_frozen_rrf_summary,
)
from my_crs.hf_list_reranker import (
    BACKEND_NAME,
    DEFAULT_MODEL_ID,
    HFGenerationError,
    HFGenerationSettings,
    HFListReranker,
)
from my_crs.rrf_list_reranker import FALLBACK_INVALID_MODEL_OUTPUT


VALID_POSITIONS = [17, 3, 42, 8, 1, 29, 10, 5, 31, 14]
VALID_OUTPUT = json.dumps({"ranked_ids": VALID_POSITIONS}, separators=(",", ":"))
MODEL_SHA = "a" * 40
OTHER_MODEL_SHA = "b" * 40


def _candidates(target_position: int | None = None) -> list[dict]:
    candidates = [
        {
            "id": 1000 + position,
            "title": f"Movie {position}",
            "source": "RRF",
            "rrf_score": 1.0 / (60 + position),
            "kbrd_rank": position,
            "ckg_rank": position + 1,
            "rrf_rank": position,
        }
        for position in range(1, 51)
    ]
    if target_position is not None:
        candidates[target_position - 1]["title"] = "Target Movie (2000)"
    return candidates


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


class _FakeBackend:
    def __init__(
        self,
        outputs: list[str] | None = None,
        *,
        error: Exception | None = None,
        model_id: str = "fixture/model",
        adapter: bool = False,
    ) -> None:
        self.messages: list[list[dict[str, str]]] = []
        self._outputs = iter(outputs or [VALID_OUTPUT])
        self._error = error
        self._provenance = {
            "backend": BACKEND_NAME,
            "model": {
                "model_id": model_id,
                "requested_revision": None,
                "resolved_revision": MODEL_SHA,
                "local_path_sha256": None,
                "architecture": ["FixtureForCausalLM"],
            },
            "tokenizer": {
                "identity": model_id,
                "resolved_revision": MODEL_SHA,
            },
            "runtime": {
                "device": "cuda:0",
                "dtype": "bfloat16",
                "torch_version": "fixture",
                "transformers_version": "fixture",
                "peft_version": "fixture" if adapter else None,
                "huggingface_hub_version": "fixture",
            },
            "adapter": {
                "enabled": adapter,
                "path": "/fixture/adapter" if adapter else None,
                "sha256": "adapter-sha" if adapter else None,
                "config": {"r": 8} if adapter else None,
                "base_model_name_or_path": model_id if adapter else None,
                "base_revision": MODEL_SHA if adapter else None,
                "compatibility_validation": {
                    "status": "passed" if adapter else "not_applicable",
                    "base_model_identity_match": True if adapter else None,
                    "base_revision_match": True if adapter else None,
                },
                "autocast_adapter_dtype": True if adapter else None,
                "base_requested_dtype": "bfloat16" if adapter else None,
                "observed_state_tensor_dtypes": ["float32"] if adapter else [],
            },
        }

    def generate(self, messages):
        self.messages.append(copy.deepcopy(messages))
        if self._error is not None:
            raise self._error
        return next(self._outputs)

    def provenance(self):
        return copy.deepcopy(self._provenance)

    def generation_provenance(self):
        return {
            "do_sample": False,
            "max_new_tokens": 128,
            "decoding": "deterministic_greedy",
        }


class _FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _FakeBatch(dict):
    def to(self, device):
        for value in self.values():
            if hasattr(value, "to"):
                value.to(device)
        return self


class _FakeGenerated:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        rows, columns = key
        if rows != slice(None, None, None):
            raise AssertionError("all generated rows must be retained")
        return [self.values[columns]]


class _FakeTokenizer:
    eos_token_id = 151645

    def __init__(
        self,
        decoded=VALID_OUTPUT,
        *,
        name_or_path=DEFAULT_MODEL_ID,
        commit=MODEL_SHA,
    ):
        self.name_or_path = name_or_path
        self._commit_hash = commit
        self.init_kwargs = {"_commit_hash": commit}
        self.input_ids = _FakeTensor([10, 11, 12])
        self.attention_mask = _FakeTensor([1, 1, 1])
        self.apply_chat_template = Mock(side_effect=self._apply_chat_template)
        self.batch_decode = Mock(return_value=[decoded])
        self.second_tokenization_calls = 0

    def _apply_chat_template(self, messages, **kwargs):
        return _FakeBatch(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
        )

    def __call__(self, *args, **kwargs):
        self.second_tokenization_calls += 1
        raise AssertionError("chat-template output must not be tokenized a second time")


class _FakeConfig:
    architectures = ["Qwen2ForCausalLM"]

    def __init__(self, commit=MODEL_SHA):
        self._commit_hash = commit


class _FakeModel:
    def __init__(self, *, commit=MODEL_SHA):
        self.config = _FakeConfig(commit)
        self.to = Mock(side_effect=self._to)
        self.eval = Mock()
        self.generate = Mock(return_value=_FakeGenerated([10, 11, 12, 91, 92]))
        self.peft_config = None

    def _to(self, device):
        self.device = device
        return self


class _FakeAdapterConfig:
    def __init__(
        self,
        *,
        base_model_name_or_path=DEFAULT_MODEL_ID,
        revision=MODEL_SHA,
    ):
        self.base_model_name_or_path = base_model_name_or_path
        self.revision = revision

    def to_dict(self):
        return {
            "r": 8,
            "base_model_name_or_path": self.base_model_name_or_path,
            "revision": self.revision,
        }


def _fake_dependencies(
    decoded=VALID_OUTPUT,
    *,
    resolved_sha=MODEL_SHA,
    tokenizer_commit=None,
    model_commit=None,
    adapter_base=DEFAULT_MODEL_ID,
    adapter_revision=MODEL_SHA,
    hub_error=None,
):
    tokenizer = _FakeTokenizer(
        decoded,
        commit=tokenizer_commit or resolved_sha,
    )
    model = _FakeModel(commit=model_commit or resolved_sha)
    auto_tokenizer = types.SimpleNamespace(
        from_pretrained=Mock(return_value=tokenizer)
    )
    auto_model = types.SimpleNamespace(from_pretrained=Mock(return_value=model))
    transformers_module = types.SimpleNamespace(
        __version__="5.14.1",
        AutoTokenizer=auto_tokenizer,
        AutoModelForCausalLM=auto_model,
    )
    torch_module = types.SimpleNamespace(
        __version__="2.6.0+cu124",
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
        inference_mode=Mock(side_effect=lambda: nullcontext()),
    )

    adapter_config = _FakeAdapterConfig(
        base_model_name_or_path=adapter_base,
        revision=adapter_revision,
    )

    def load_adapter(
        base_model,
        path,
        *,
        is_trainable,
        config,
        autocast_adapter_dtype,
    ):
        if is_trainable:
            raise AssertionError("adapter must be loaded for inference only")
        if config is not adapter_config:
            raise AssertionError("prevalidated adapter config must be reused")
        base_model.peft_config = {"default": config}
        return base_model

    peft_module = types.SimpleNamespace(
        __version__="0.20.0",
        PeftConfig=types.SimpleNamespace(
            from_pretrained=Mock(return_value=adapter_config)
        ),
        PeftModel=types.SimpleNamespace(
            from_pretrained=Mock(side_effect=load_adapter)
        ),
        get_peft_model_state_dict=Mock(
            return_value={
                "adapter.weight": types.SimpleNamespace(dtype="torch.float32")
            }
        ),
    )
    model_info = Mock()
    if hub_error is not None:
        model_info.side_effect = hub_error
    else:
        model_info.return_value = types.SimpleNamespace(sha=resolved_sha)
    api = types.SimpleNamespace(model_info=model_info)
    hub_module = types.SimpleNamespace(
        __version__="1.3.0",
        HfApi=Mock(return_value=api),
    )
    return types.SimpleNamespace(
        torch=torch_module,
        transformers=transformers_module,
        peft=peft_module,
        hub=hub_module,
        tokenizer=tokenizer,
        model=model,
        adapter_config=adapter_config,
        hub_api=api,
    )


@contextmanager
def _patched_evaluator_fixture(count: int = 2):
    records: list[dict] = []
    conversations: list[tuple[dict, dict]] = []
    for line_number in range(1, count + 1):
        conversation, record = _conversation(line_number=line_number)
        records.append(record)
        conversations.append((conversation, record))
    valid_index = _valid_index(*conversations)
    frozen_by_key = {
        instance_key(record["line_number"], record["turn_index"]): record
        for record in records
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        valid_path = root / "valid_data.jsonl"
        valid_path.write_text("fixture", encoding="utf-8")
        summary_path = root / "summary.json"
        summary_path.write_text(json.dumps(_valid_summary()), encoding="utf-8")
        instances_path = root / "instances.jsonl"
        instances_path.write_text("fixture-artifact", encoding="utf-8")
        paths = {
            "root": root,
            "valid_path": valid_path,
            "summary_path": summary_path,
            "instances_path": instances_path,
            "output_path": root / "result.json",
            "instance_output_path": root / "result.instances.jsonl",
        }
        with patch.object(
            evaluator_module,
            "validate_official_valid_path",
            return_value=valid_path,
        ), patch.object(
            evaluator_module,
            "load_frozen_rrf_instances",
            return_value=records,
        ), patch.object(
            evaluator_module,
            "reconstruct_valid_event_index",
            return_value=valid_index,
        ), patch.object(
            evaluator_module,
            "validate_frozen_instances_against_valid",
            return_value=frozen_by_key,
        ):
            yield paths, records, valid_index


class TestHFListRerankerBackend(unittest.TestCase):
    def _backend(self, dependencies, settings=None):
        return HFListReranker(
            settings or HFGenerationSettings(),
            torch_module=dependencies.torch,
            transformers_module=dependencies.transformers,
            peft_module=dependencies.peft,
            huggingface_hub_module=dependencies.hub,
        )

    def test_model_and_tokenizer_load_once_for_multiple_generations(self):
        dependencies = _fake_dependencies()
        backend = self._backend(dependencies)
        first = backend.generate([{"role": "user", "content": "one"}])
        second = backend.generate([{"role": "user", "content": "two"}])

        self.assertEqual(first, VALID_OUTPUT)
        self.assertEqual(second, VALID_OUTPUT)
        dependencies.transformers.AutoTokenizer.from_pretrained.assert_called_once()
        dependencies.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
        self.assertEqual(dependencies.model.generate.call_count, 2)
        self.assertEqual(dependencies.tokenizer.apply_chat_template.call_count, 2)
        self.assertEqual(dependencies.tokenizer.batch_decode.call_count, 2)
        self.assertEqual(dependencies.tokenizer.second_tokenization_calls, 0)

    def test_generation_is_greedy_inference_and_decodes_new_tokens_only(self):
        dependencies = _fake_dependencies()
        backend = self._backend(
            dependencies,
            HFGenerationSettings(max_new_tokens=77),
        )
        backend.generate([{"role": "user", "content": "fixture"}])

        dependencies.torch.inference_mode.assert_called_once()
        dependencies.model.generate.assert_called_once()
        kwargs = dependencies.model.generate.call_args.kwargs
        self.assertFalse(kwargs["do_sample"])
        self.assertEqual(kwargs["max_new_tokens"], 77)
        self.assertIs(kwargs["attention_mask"], dependencies.tokenizer.attention_mask)
        self.assertNotIn("pad_token_id", kwargs)
        dependencies.tokenizer.batch_decode.assert_called_once_with(
            [[91, 92]],
            skip_special_tokens=True,
        )
        dependencies.tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "fixture"}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        self.assertEqual(dependencies.tokenizer.input_ids.device, "cuda:0")
        self.assertEqual(dependencies.tokenizer.attention_mask.device, "cuda:0")

    def test_adapter_loader_is_invoked_only_when_adapter_is_configured(self):
        dependencies = _fake_dependencies()
        self._backend(dependencies)
        dependencies.peft.PeftConfig.from_pretrained.assert_not_called()
        dependencies.peft.PeftModel.from_pretrained.assert_not_called()

        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            adapted = self._backend(
                dependencies,
                HFGenerationSettings(adapter_path=str(adapter)),
            )
        dependencies.peft.PeftConfig.from_pretrained.assert_called_once_with(
            str(adapter)
        )
        dependencies.peft.PeftModel.from_pretrained.assert_called_once()
        self.assertFalse(
            dependencies.peft.PeftModel.from_pretrained.call_args.kwargs[
                "is_trainable"
            ]
        )
        self.assertTrue(
            dependencies.peft.PeftModel.from_pretrained.call_args.kwargs[
                "autocast_adapter_dtype"
            ]
        )
        self.assertTrue(adapted.provenance()["adapter"]["enabled"])
        self.assertIsNotNone(adapted.provenance()["adapter"]["sha256"])

    def test_backend_provenance_records_model_runtime_and_revisions(self):
        dependencies = _fake_dependencies()
        backend = self._backend(
            dependencies,
            HFGenerationSettings(model_revision="requested-revision"),
        )
        provenance = backend.provenance()
        self.assertEqual(provenance["backend"], BACKEND_NAME)
        self.assertEqual(provenance["model"]["resolved_revision"], MODEL_SHA)
        self.assertEqual(provenance["tokenizer"]["resolved_revision"], MODEL_SHA)
        self.assertEqual(provenance["runtime"]["dtype"], "bfloat16")
        self.assertEqual(provenance["runtime"]["device"], "cuda:0")
        dependencies.hub_api.model_info.assert_called_once_with(
            DEFAULT_MODEL_ID,
            revision="requested-revision",
        )

    def test_remote_revision_resolves_once_and_is_shared_by_tokenizer_and_model(self):
        for requested in (None, "main"):
            with self.subTest(requested=requested):
                dependencies = _fake_dependencies()
                self._backend(
                    dependencies,
                    HFGenerationSettings(model_revision=requested),
                )
                dependencies.hub_api.model_info.assert_called_once_with(
                    DEFAULT_MODEL_ID,
                    revision=requested,
                )
                dependencies.transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
                    DEFAULT_MODEL_ID,
                    revision=MODEL_SHA,
                )
                model_kwargs = dependencies.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
                self.assertEqual(model_kwargs["revision"], MODEL_SHA)

    def test_explicit_immutable_revision_is_preserved_without_hub_resolution(self):
        dependencies = _fake_dependencies()
        backend = self._backend(
            dependencies,
            HFGenerationSettings(model_revision=MODEL_SHA.upper()),
        )
        dependencies.hub.HfApi.assert_not_called()
        self.assertEqual(
            backend.provenance()["model"]["resolved_revision"],
            MODEL_SHA.upper(),
        )
        tokenizer_kwargs = dependencies.transformers.AutoTokenizer.from_pretrained.call_args.kwargs
        model_kwargs = dependencies.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
        self.assertEqual(tokenizer_kwargs["revision"], MODEL_SHA.upper())
        self.assertEqual(model_kwargs["revision"], MODEL_SHA.upper())

    def test_remote_resolution_fails_closed_before_loading(self):
        cases = (
            _fake_dependencies(resolved_sha="main"),
            _fake_dependencies(hub_error=RuntimeError("offline")),
        )
        for dependencies in cases:
            with self.subTest(side_effect=dependencies.hub_api.model_info.side_effect):
                with self.assertRaisesRegex(ValueError, "immutable"):
                    self._backend(dependencies)
                dependencies.transformers.AutoTokenizer.from_pretrained.assert_not_called()
                dependencies.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_loaded_tokenizer_or_model_revision_mismatch_is_rejected(self):
        tokenizer_mismatch = _fake_dependencies(tokenizer_commit=OTHER_MODEL_SHA)
        with self.assertRaisesRegex(ValueError, "tokenizer revision"):
            self._backend(tokenizer_mismatch)
        tokenizer_mismatch.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

        model_mismatch = _fake_dependencies(model_commit=OTHER_MODEL_SHA)
        with self.assertRaisesRegex(ValueError, "model revision"):
            self._backend(model_mismatch)

    def test_local_model_uses_content_hash_without_hub_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            model_directory = Path(directory) / "model"
            model_directory.mkdir()
            (model_directory / "config.json").write_text("fixture", encoding="utf-8")
            dependencies = _fake_dependencies()
            backend = self._backend(
                dependencies,
                HFGenerationSettings(model_id=str(model_directory)),
            )
        dependencies.hub.HfApi.assert_not_called()
        provenance = backend.provenance()
        self.assertIsNone(provenance["model"]["resolved_revision"])
        self.assertIsNotNone(provenance["model"]["local_path_sha256"])
        self.assertNotIn(
            "revision",
            dependencies.transformers.AutoTokenizer.from_pretrained.call_args.kwargs,
        )

    def test_wrong_adapter_identity_and_revision_fail_before_weight_loading(self):
        cases = (
            (_fake_dependencies(adapter_base="Qwen/Qwen2.5-3B"), "base model"),
            (_fake_dependencies(adapter_revision=OTHER_MODEL_SHA), "base revision"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, (dependencies, message) in enumerate(cases):
                adapter = Path(directory) / f"adapter-{index}"
                adapter.mkdir()
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    self._backend(
                        dependencies,
                        HFGenerationSettings(adapter_path=str(adapter)),
                    )
                dependencies.transformers.AutoTokenizer.from_pretrained.assert_not_called()
                dependencies.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
                dependencies.peft.PeftModel.from_pretrained.assert_not_called()

    def test_matching_adapter_records_compatibility_and_dtype_provenance(self):
        dependencies = _fake_dependencies()
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            (adapter / "weights.bin").write_bytes(b"adapter")
            backend = self._backend(
                dependencies,
                HFGenerationSettings(adapter_path=str(adapter)),
            )
        adapter_provenance = backend.provenance()["adapter"]
        self.assertEqual(adapter_provenance["base_model_name_or_path"], DEFAULT_MODEL_ID)
        self.assertEqual(adapter_provenance["base_revision"], MODEL_SHA)
        self.assertEqual(
            adapter_provenance["compatibility_validation"],
            {
                "status": "passed",
                "base_model_identity_match": True,
                "base_revision_match": True,
            },
        )
        self.assertTrue(adapter_provenance["autocast_adapter_dtype"])
        self.assertEqual(adapter_provenance["base_requested_dtype"], "bfloat16")
        self.assertEqual(
            adapter_provenance["observed_state_tensor_dtypes"],
            ["float32"],
        )
        dependencies.peft.get_peft_model_state_dict.assert_called_once_with(
            dependencies.model
        )

    def test_matching_adapter_without_revision_is_accepted_without_invention(self):
        dependencies = _fake_dependencies(adapter_revision=None)
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / "adapter"
            adapter.mkdir()
            backend = self._backend(
                dependencies,
                HFGenerationSettings(adapter_path=str(adapter)),
            )
        adapter_provenance = backend.provenance()["adapter"]
        self.assertIsNone(adapter_provenance["base_revision"])
        self.assertIsNone(
            adapter_provenance["compatibility_validation"]["base_revision_match"]
        )


class TestHFInstanceEvaluation(unittest.TestCase):
    def test_shared_prompt_parser_and_valid_top10_completion(self):
        conversation, frozen_record = _conversation()
        backend = _FakeBackend([VALID_OUTPUT])
        with patch.object(
            evaluator_module,
            "build_list_rerank_prompt",
            wraps=shared_reranker.build_list_rerank_prompt,
        ) as prompt_builder, patch.object(
            evaluator_module,
            "parse_ranked_positions",
            wraps=shared_reranker.parse_ranked_positions,
        ) as parser:
            result = evaluate_hf_instance(frozen_record, conversation, backend)

        prompt_builder.assert_called_once()
        parser.assert_called_once_with(VALID_OUTPUT, candidate_count=50)
        self.assertFalse(result["fallback"])
        self.assertEqual(result["parsed_top10_local_positions"], VALID_POSITIONS)
        self.assertEqual(result["reranked_target_rank"], 1)
        self.assertEqual(len(result["final_complete_top50_order"]), 50)
        self.assertEqual(
            len({item["id"] for item in result["final_complete_top50_order"]}),
            50,
        )

    def test_malformed_duplicate_and_out_of_range_outputs_fall_back_exactly(self):
        invalid_outputs = [
            "not-json",
            json.dumps({"ranked_ids": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9]}),
            json.dumps({"ranked_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 51]}),
        ]
        for output in invalid_outputs:
            with self.subTest(output=output):
                conversation, frozen_record = _conversation()
                original = copy.deepcopy(frozen_record["rrf_candidates"])
                result = evaluate_hf_instance(
                    frozen_record,
                    conversation,
                    _FakeBackend([output]),
                )
                self.assertTrue(result["fallback"])
                self.assertEqual(
                    result["fallback_reason"], FALLBACK_INVALID_MODEL_OUTPUT
                )
                self.assertEqual(
                    [item["id"] for item in result["final_complete_top50_order"]],
                    [item["id"] for item in original],
                )
                self.assertEqual(result["original_rrf_target_rank"], 17)
                self.assertEqual(result["reranked_target_rank"], 17)
                self.assertTrue(result["hit_at_50"])

    def test_generation_failure_preserves_original_candidates(self):
        conversation, frozen_record = _conversation()
        result = evaluate_hf_instance(
            frozen_record,
            conversation,
            _FakeBackend(error=HFGenerationError("CUDA unavailable")),
        )
        self.assertTrue(result["fallback"])
        self.assertEqual(result["fallback_reason"], FALLBACK_GENERATION_FAILURE)
        self.assertIsNone(result["raw_model_output"])
        self.assertEqual(result["successful_generations"], 0)
        self.assertEqual(result["original_rrf_target_rank"], 17)
        self.assertEqual(result["reranked_target_rank"], 17)

    def test_prompt_has_no_target_future_ground_truth_scores_or_ranks(self):
        conversation, frozen_record = _conversation()
        conversation["movieMentions"]["98"] = "Secret Ground Truth (2001)"
        conversation["respondentQuestions"]["98"] = {"suggested": 1}
        conversation["messages"][3]["text"] = "LEAK_TARGET @99 and @98."
        conversation["messages"][4]["text"] = "LEAK_FUTURE"
        frozen_record["ground_truth_titles"] = [
            "target movie (2000)",
            "secret ground truth (2001)",
        ]
        backend = _FakeBackend([VALID_OUTPUT])
        evaluate_hf_instance(frozen_record, conversation, backend)
        prompt = json.dumps(backend.messages[0], ensure_ascii=False)

        self.assertIn("I want an adventurous movie.", prompt)
        self.assertNotIn("LEAK_TARGET", prompt)
        self.assertNotIn("LEAK_FUTURE", prompt)
        self.assertNotIn("Secret Ground Truth", prompt)
        for hidden in (
            "ground_truth_titles",
            "rrf_score",
            "kbrd_rank",
            "ckg_rank",
            "rrf_rank",
        ):
            self.assertNotIn(hidden, prompt)

    def test_metrics_use_frozen_normalized_title_semantics(self):
        candidates = _candidates()
        candidates[6]["title"] = "Target, Movie! (2000)"
        self.assertEqual(get_rank(candidates, ["target movie"]), 7)


class TestHFFingerprintAndResume(unittest.TestCase):
    def _fingerprint(self, root: Path, backend: _FakeBackend) -> str:
        summary = root / "summary.json"
        instances = root / "instances.jsonl"
        if not summary.exists():
            summary.write_text("summary", encoding="utf-8")
            instances.write_text("instances", encoding="utf-8")
        return hf_run_fingerprint(
            summary_path=summary,
            instances_path=instances,
            backend_provenance=backend.provenance(),
            generation_provenance=backend.generation_provenance(),
        )

    def test_zero_shot_adapter_and_model_fingerprints_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            zero_shot = self._fingerprint(root, _FakeBackend())
            adapted = self._fingerprint(root, _FakeBackend(adapter=True))
            other_model = self._fingerprint(
                root,
                _FakeBackend(model_id="fixture/other-model"),
            )
        self.assertNotEqual(zero_shot, adapted)
        self.assertNotEqual(zero_shot, other_model)

    def test_prompt_digest_change_changes_hf_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self._fingerprint(root, _FakeBackend())
            with patch.object(
                evaluator_module,
                "prompt_template_digest",
                return_value="changed-prompt-digest",
            ):
                changed = self._fingerprint(root, _FakeBackend())
        self.assertNotEqual(original, changed)

    def test_dtype_and_generation_changes_change_hf_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = _FakeBackend()
            original = self._fingerprint(root, backend)
            changed_dtype = backend.provenance()
            changed_dtype["runtime"]["dtype"] = "float16"
            dtype_fingerprint = hf_run_fingerprint(
                summary_path=root / "summary.json",
                instances_path=root / "instances.jsonl",
                backend_provenance=changed_dtype,
                generation_provenance=backend.generation_provenance(),
            )
            changed_generation = backend.generation_provenance()
            changed_generation["max_new_tokens"] = 64
            generation_fingerprint = hf_run_fingerprint(
                summary_path=root / "summary.json",
                instances_path=root / "instances.jsonl",
                backend_provenance=backend.provenance(),
                generation_provenance=changed_generation,
            )
        self.assertNotEqual(original, dtype_fingerprint)
        self.assertNotEqual(original, generation_fingerprint)

    def test_immutable_model_revision_changes_hf_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = _FakeBackend()
            original = self._fingerprint(root, backend)
            changed_provenance = backend.provenance()
            changed_provenance["model"]["resolved_revision"] = OTHER_MODEL_SHA
            changed_provenance["tokenizer"]["resolved_revision"] = OTHER_MODEL_SHA
            changed = hf_run_fingerprint(
                summary_path=root / "summary.json",
                instances_path=root / "instances.jsonl",
                backend_provenance=changed_provenance,
                generation_provenance=backend.generation_provenance(),
            )
        self.assertNotEqual(original, changed)

    def test_adapter_content_and_dtype_policy_change_hf_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            weights = adapter / "adapter_model.bin"
            weights.write_bytes(b"first")

            first_dependencies = _fake_dependencies()
            first = HFListReranker(
                HFGenerationSettings(adapter_path=str(adapter)),
                torch_module=first_dependencies.torch,
                transformers_module=first_dependencies.transformers,
                peft_module=first_dependencies.peft,
                huggingface_hub_module=first_dependencies.hub,
            )
            first_fingerprint = self._fingerprint(root, first)

            weights.write_bytes(b"second")
            second_dependencies = _fake_dependencies()
            second = HFListReranker(
                HFGenerationSettings(adapter_path=str(adapter)),
                torch_module=second_dependencies.torch,
                transformers_module=second_dependencies.transformers,
                peft_module=second_dependencies.peft,
                huggingface_hub_module=second_dependencies.hub,
            )
            content_fingerprint = self._fingerprint(root, second)

            policy_dependencies = _fake_dependencies()
            other_policy = HFListReranker(
                HFGenerationSettings(
                    adapter_path=str(adapter),
                    autocast_adapter_dtype=False,
                ),
                torch_module=policy_dependencies.torch,
                transformers_module=policy_dependencies.transformers,
                peft_module=policy_dependencies.peft,
                huggingface_hub_module=policy_dependencies.hub,
            )
            policy_fingerprint = self._fingerprint(root, other_policy)

        self.assertNotEqual(first_fingerprint, content_fingerprint)
        self.assertNotEqual(content_fingerprint, policy_fingerprint)
        self.assertFalse(
            policy_dependencies.peft.PeftModel.from_pretrained.call_args.kwargs[
                "autocast_adapter_dtype"
            ]
        )

    def test_valid_and_corrupted_resume_records(self):
        conversation, frozen_record = _conversation()
        backend = _FakeBackend([VALID_OUTPUT])
        record = evaluate_hf_instance(frozen_record, conversation, backend)
        event = _valid_index((conversation, frozen_record)).events["1:3"]
        self.assertEqual(
            validate_hf_resume_record(record, frozen_record, event, conversation),
            record,
        )
        record["parsed_top10_local_positions"] = list(reversed(VALID_POSITIONS))
        with self.assertRaisesRegex(ValueError, "parsed positions"):
            validate_hf_resume_record(record, frozen_record, event, conversation)

    def test_invalid_output_resume_reproduces_parser_failure(self):
        conversation, frozen_record = _conversation()
        record = evaluate_hf_instance(
            frozen_record,
            conversation,
            _FakeBackend(["malformed"]),
        )
        event = _valid_index((conversation, frozen_record)).events["1:3"]
        self.assertEqual(
            validate_hf_resume_record(record, frozen_record, event, conversation),
            record,
        )
        record["fallback_detail"] = "forged detail"
        with self.assertRaisesRegex(ValueError, "invalid-output detail"):
            validate_hf_resume_record(record, frozen_record, event, conversation)

    def test_generation_failure_resume_provenance_is_fail_closed(self):
        conversation, frozen_record = _conversation()
        record = evaluate_hf_instance(
            frozen_record,
            conversation,
            _FakeBackend(error=HFGenerationError("CUDA unavailable")),
        )
        event = _valid_index((conversation, frozen_record)).events["1:3"]
        self.assertEqual(
            validate_hf_resume_record(record, frozen_record, event, conversation),
            record,
        )
        record["successful_generations"] = 1
        with self.assertRaisesRegex(ValueError, "generation provenance"):
            validate_hf_resume_record(record, frozen_record, event, conversation)


class TestHFFullEvaluator(unittest.TestCase):
    def test_deterministic_first_n_and_backend_constructed_once(self):
        backend = _FakeBackend([VALID_OUTPUT, VALID_OUTPUT, VALID_OUTPUT])
        with _patched_evaluator_fixture(count=3) as (paths, _, _), patch.object(
            evaluator_module,
            "HFListReranker",
            return_value=backend,
        ) as backend_class:
            result = evaluate_rrf_hf(
                rrf_summary_path=paths["summary_path"],
                rrf_instances_path=paths["instances_path"],
                valid_path=paths["valid_path"],
                output_path=paths["output_path"],
                instance_output_path=paths["instance_output_path"],
                settings=HFGenerationSettings(),
                max_instances=2,
            )
            records = [
                json.loads(line)
                for line in paths["instance_output_path"].read_text(
                    encoding="utf-8"
                ).splitlines()
            ]

        backend_class.assert_called_once()
        self.assertEqual(len(backend.messages), 2)
        self.assertEqual([record["instance_key"] for record in records], ["1:3", "2:3"])
        self.assertEqual(result["processed_instances"], 2)
        self.assertEqual(result["model"], "fixture/model")
        self.assertEqual(result["backend"], BACKEND_NAME)
        self.assertEqual(result["reranked_metrics"]["Recall@50"], 1.0)
        self.assertTrue(result["recall_at_50_invariant_passed"])

    def test_valid_only_rejected_before_model_loading(self):
        backend_class = Mock(side_effect=AssertionError("model must not load"))
        with patch.object(evaluator_module, "HFListReranker", backend_class):
            with self.assertRaisesRegex(ValueError, "VALID-only"):
                evaluate_rrf_hf(valid_path=Path("test_data.jsonl"))
        backend_class.assert_not_called()

    def test_static_generation_settings_rejected_before_model_loading(self):
        backend_class = Mock(side_effect=AssertionError("model must not load"))
        with patch.object(evaluator_module, "HFListReranker", backend_class):
            with self.assertRaisesRegex(ValueError, "max_new_tokens"):
                evaluate_rrf_hf(
                    settings=HFGenerationSettings(max_new_tokens=0),
                )
        backend_class.assert_not_called()

    def test_frozen_summary_validation_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid_data.jsonl"
            valid_path.write_text("fixture", encoding="utf-8")
            summary_path = root / "summary.json"
            bad_summary = _valid_summary()
            bad_summary["source_sha256"] = "wrong"
            summary_path.write_text(json.dumps(bad_summary), encoding="utf-8")
            instances_path = root / "instances.jsonl"
            instances_path.write_text("fixture", encoding="utf-8")
            backend_class = Mock(side_effect=AssertionError("model must not load"))
            with patch.object(
                evaluator_module,
                "validate_official_valid_path",
                return_value=valid_path,
            ), patch.object(evaluator_module, "HFListReranker", backend_class):
                with self.assertRaisesRegex(ValueError, "source_sha256"):
                    evaluate_rrf_hf(
                        rrf_summary_path=summary_path,
                        rrf_instances_path=instances_path,
                        valid_path=valid_path,
                        output_path=root / "output.json",
                        instance_output_path=root / "output.jsonl",
                    )
            backend_class.assert_not_called()

    def test_output_collision_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid_data.jsonl"
            backend_class = Mock(side_effect=AssertionError("model must not load"))
            with patch.object(
                evaluator_module,
                "validate_official_valid_path",
                return_value=valid_path,
            ), patch.object(evaluator_module, "HFListReranker", backend_class):
                with self.assertRaisesRegex(ValueError, "Path collision"):
                    evaluate_rrf_hf(
                        valid_path=valid_path,
                        output_path=valid_path,
                    )
            backend_class.assert_not_called()

    def test_existing_instance_output_rejected_before_model_loading(self):
        backend_class = Mock(side_effect=AssertionError("model must not load"))
        with _patched_evaluator_fixture(count=1) as (paths, _, _), patch.object(
            evaluator_module,
            "HFListReranker",
            backend_class,
        ):
            paths["instance_output_path"].write_text("existing", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                evaluate_rrf_hf(
                    rrf_summary_path=paths["summary_path"],
                    rrf_instances_path=paths["instances_path"],
                    valid_path=paths["valid_path"],
                    output_path=paths["output_path"],
                    instance_output_path=paths["instance_output_path"],
                )
        backend_class.assert_not_called()

    def test_malformed_resume_rejected_before_model_loading(self):
        backend_class = Mock(side_effect=AssertionError("model must not load"))
        with _patched_evaluator_fixture(count=1) as (paths, _, _), patch.object(
            evaluator_module,
            "HFListReranker",
            backend_class,
        ):
            paths["instance_output_path"].write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Malformed HF resume JSON"):
                evaluate_rrf_hf(
                    rrf_summary_path=paths["summary_path"],
                    rrf_instances_path=paths["instances_path"],
                    valid_path=paths["valid_path"],
                    output_path=paths["output_path"],
                    instance_output_path=paths["instance_output_path"],
                    resume=True,
                )
        backend_class.assert_not_called()

    def test_internally_mismatched_resume_fingerprints_fail_before_model_loading(self):
        backend_class = Mock(side_effect=AssertionError("model must not load"))
        with _patched_evaluator_fixture(count=2) as (paths, _, _), patch.object(
            evaluator_module,
            "HFListReranker",
            backend_class,
        ):
            resume_records = (
                {"instance_key": "1:3", "run_fingerprint": "first"},
                {"instance_key": "2:3", "run_fingerprint": "second"},
            )
            paths["instance_output_path"].write_text(
                "".join(json.dumps(record) + "\n" for record in resume_records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inconsistent run fingerprints"):
                evaluate_rrf_hf(
                    rrf_summary_path=paths["summary_path"],
                    rrf_instances_path=paths["instances_path"],
                    valid_path=paths["valid_path"],
                    output_path=paths["output_path"],
                    instance_output_path=paths["instance_output_path"],
                    resume=True,
                )
        backend_class.assert_not_called()

    def test_resume_fingerprint_mismatch_rejected_before_generation(self):
        backend = _FakeBackend([VALID_OUTPUT])
        with _patched_evaluator_fixture(count=1) as (paths, _, _):
            paths["instance_output_path"].write_text(
                json.dumps(
                    {"instance_key": "1:3", "run_fingerprint": "wrong"}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                evaluate_rrf_hf(
                    rrf_summary_path=paths["summary_path"],
                    rrf_instances_path=paths["instances_path"],
                    valid_path=paths["valid_path"],
                    output_path=paths["output_path"],
                    instance_output_path=paths["instance_output_path"],
                    max_instances=1,
                    resume=True,
                    backend=backend,
                )
        self.assertEqual(backend.messages, [])


if __name__ == "__main__":
    unittest.main()
