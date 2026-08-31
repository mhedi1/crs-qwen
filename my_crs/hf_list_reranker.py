"""Reusable local Hugging Face backend for frozen Stage-2 list reranking."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_MAX_NEW_TOKENS = 128
BACKEND_NAME = "huggingface_transformers"
IMMUTABLE_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class HFGenerationSettings:
    """Scientifically relevant local-generation settings."""

    model_id: str = DEFAULT_MODEL_ID
    model_revision: str | None = None
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    adapter_path: str | None = None
    autocast_adapter_dtype: bool = True


class HFGenerationError(RuntimeError):
    """Raised when a loaded local backend cannot generate model output."""


def _module_version(module: ModuleType, distribution: str) -> str | None:
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _path_sha256(path_value: str | None) -> str | None:
    """Hash a local file/directory deterministically; remote IDs return ``None``."""
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.exists():
        return None
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(
        item for item in path.rglob("*") if item.is_file()
    )
    root = path.parent if path.is_file() else path
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _resolved_reference(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def _canonical_dtype_name(value: str) -> str:
    aliases = {
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp16": "float16",
        "float16": "float16",
        "fp32": "float32",
        "float32": "float32",
    }
    canonical = aliases.get(str(value).lower())
    if canonical is None:
        raise ValueError(
            "dtype must be one of bfloat16/bf16, float16/fp16, or "
            "float32/fp32"
        )
    return canonical


def validate_hf_generation_settings(settings: HFGenerationSettings) -> None:
    """Validate settings without importing or loading large model dependencies."""
    if settings.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if not settings.model_id:
        raise ValueError("model_id must not be empty")
    if not settings.device:
        raise ValueError("device must not be empty")
    if type(settings.autocast_adapter_dtype) is not bool:
        raise ValueError("autocast_adapter_dtype must be a boolean")
    _canonical_dtype_name(settings.dtype)


def _normalized_model_identity(value: str) -> str:
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return value.strip().rstrip("/")


def _adapter_tensor_dtypes(peft_module: ModuleType, model: Any) -> list[str]:
    helper = getattr(peft_module, "get_peft_model_state_dict", None)
    if not callable(helper):
        return []
    adapter_state = helper(model)
    dtype_names = {
        str(dtype).removeprefix("torch.")
        for tensor in adapter_state.values()
        if (dtype := getattr(tensor, "dtype", None)) is not None
    }
    del adapter_state
    return sorted(dtype_names)


class HFListReranker:
    """Load one local causal LM and generate strict list-reranking responses."""

    def __init__(
        self,
        settings: HFGenerationSettings | None = None,
        *,
        torch_module: ModuleType | None = None,
        transformers_module: ModuleType | None = None,
        peft_module: ModuleType | None = None,
        huggingface_hub_module: ModuleType | None = None,
    ) -> None:
        self.settings = settings or HFGenerationSettings()
        validate_hf_generation_settings(self.settings)

        self._torch = torch_module or importlib.import_module("torch")
        self._transformers = transformers_module or importlib.import_module(
            "transformers"
        )
        self._peft = peft_module
        self._huggingface_hub = huggingface_hub_module
        self._dtype_name, self._torch_dtype = self._resolve_dtype(self.settings.dtype)
        self._local_model_sha256 = _path_sha256(self.settings.model_id)
        self._model_is_local = Path(self.settings.model_id).expanduser().is_dir()
        self._resolved_model_revision = self._resolve_model_revision()

        self._adapter_config: Any = None
        self._adapter_base_model: str | None = None
        self._adapter_base_revision: str | None = None
        self._adapter_compatibility: dict[str, Any] = {
            "status": "not_applicable",
            "base_model_identity_match": None,
            "base_revision_match": None,
        }
        if self.settings.adapter_path:
            if self._peft is None:
                self._peft = importlib.import_module("peft")
            self._adapter_config = self._peft.PeftConfig.from_pretrained(
                self.settings.adapter_path
            )
            self._validate_adapter_compatibility()

        load_kwargs: dict[str, Any] = {"torch_dtype": self._torch_dtype}
        tokenizer_kwargs: dict[str, Any] = {}
        if self._resolved_model_revision:
            load_kwargs["revision"] = self._resolved_model_revision
            tokenizer_kwargs["revision"] = self._resolved_model_revision

        self.tokenizer = self._transformers.AutoTokenizer.from_pretrained(
            self.settings.model_id,
            **tokenizer_kwargs,
        )
        tokenizer_init_kwargs = getattr(self.tokenizer, "init_kwargs", {})
        observed_tokenizer_commit = getattr(self.tokenizer, "_commit_hash", None)
        if observed_tokenizer_commit is None and isinstance(
            tokenizer_init_kwargs, Mapping
        ):
            observed_tokenizer_commit = tokenizer_init_kwargs.get("_commit_hash")
        if (
            not self._model_is_local
            and observed_tokenizer_commit is not None
            and str(observed_tokenizer_commit).lower()
            != str(self._resolved_model_revision).lower()
        ):
            raise ValueError(
                "Loaded tokenizer revision does not match the resolved immutable "
                f"revision: {observed_tokenizer_commit} != "
                f"{self._resolved_model_revision}"
            )
        base_model = self._transformers.AutoModelForCausalLM.from_pretrained(
            self.settings.model_id,
            **load_kwargs,
        )
        base_model = base_model.to(self.settings.device)
        base_config = getattr(base_model, "config", None)
        observed_model_commit = getattr(base_config, "_commit_hash", None)
        if (
            not self._model_is_local
            and observed_model_commit is not None
            and str(observed_model_commit).lower()
            != str(self._resolved_model_revision).lower()
        ):
            raise ValueError(
                "Loaded model revision does not match the resolved immutable revision: "
                f"{observed_model_commit} != {self._resolved_model_revision}"
            )

        if self.settings.adapter_path:
            self.model = self._peft.PeftModel.from_pretrained(
                base_model,
                self.settings.adapter_path,
                is_trainable=False,
                config=self._adapter_config,
                autocast_adapter_dtype=self.settings.autocast_adapter_dtype,
            )
        else:
            self.model = base_model
        self.model.eval()
        self._adapter_state_dtypes = (
            _adapter_tensor_dtypes(self._peft, self.model)
            if self.settings.adapter_path
            else []
        )
        self._provenance = self._build_provenance(base_config)

    def _resolve_model_revision(self) -> str | None:
        if self._model_is_local:
            return None
        requested = self.settings.model_revision
        if requested and IMMUTABLE_REVISION_RE.fullmatch(requested):
            return requested
        if self._huggingface_hub is None:
            self._huggingface_hub = importlib.import_module("huggingface_hub")
        try:
            model_info = self._huggingface_hub.HfApi().model_info(
                self.settings.model_id,
                revision=requested,
            )
        except Exception as error:
            raise ValueError(
                "Could not resolve the remote model to an immutable Hugging Face revision"
            ) from error
        resolved = getattr(model_info, "sha", None)
        if not isinstance(resolved, str) or not IMMUTABLE_REVISION_RE.fullmatch(resolved):
            raise ValueError(
                "Remote model revision did not resolve to an immutable 40-character SHA"
            )
        return resolved

    def _validate_adapter_compatibility(self) -> None:
        adapter_base = getattr(self._adapter_config, "base_model_name_or_path", None)
        adapter_revision = getattr(self._adapter_config, "revision", None)
        self._adapter_base_model = str(adapter_base) if adapter_base else None
        self._adapter_base_revision = (
            str(adapter_revision) if adapter_revision is not None else None
        )

        expected_identity = _normalized_model_identity(self.settings.model_id)
        identity_match = bool(adapter_base) and (
            _normalized_model_identity(str(adapter_base)) == expected_identity
        )
        if not identity_match:
            raise ValueError(
                "PEFT adapter base model does not match the requested base model: "
                f"{adapter_base!r} != {self.settings.model_id!r}"
            )

        revision_match: bool | None = None
        if adapter_revision is not None:
            if self._resolved_model_revision is None:
                raise ValueError(
                    "PEFT adapter records a base revision that cannot be validated "
                    "against a local model directory"
                )
            revision_match = (
                str(adapter_revision).lower()
                == str(self._resolved_model_revision).lower()
            )
            if not revision_match:
                raise ValueError(
                    "PEFT adapter base revision does not match the resolved base SHA: "
                    f"{adapter_revision} != {self._resolved_model_revision}"
                )

        self._adapter_compatibility = {
            "status": "passed",
            "base_model_identity_match": True,
            "base_revision_match": revision_match,
        }

    def _resolve_dtype(self, value: str) -> tuple[str, Any]:
        canonical = _canonical_dtype_name(value)
        return canonical, getattr(self._torch, canonical)

    def _build_provenance(self, base_config: Any) -> dict[str, Any]:
        architectures = getattr(base_config, "architectures", None) or []

        adapter_enabled = bool(self.settings.adapter_path)
        adapter_config = (
            getattr(self.model, "peft_config", self._adapter_config)
            if adapter_enabled
            else None
        )
        adapter_path = _resolved_reference(self.settings.adapter_path)

        return {
            "backend": BACKEND_NAME,
            "model": {
                "model_id": self.settings.model_id,
                "requested_revision": self.settings.model_revision,
                "resolved_revision": self._resolved_model_revision,
                "local_path_sha256": self._local_model_sha256,
                "architecture": [str(item) for item in architectures],
            },
            "tokenizer": {
                "identity": str(
                    getattr(self.tokenizer, "name_or_path", self.settings.model_id)
                ),
                "resolved_revision": self._resolved_model_revision,
            },
            "runtime": {
                "device": self.settings.device,
                "dtype": self._dtype_name,
                "torch_version": _module_version(self._torch, "torch"),
                "transformers_version": _module_version(
                    self._transformers, "transformers"
                ),
                "peft_version": (
                    _module_version(self._peft, "peft") if adapter_enabled else None
                ),
                "huggingface_hub_version": (
                    _module_version(self._huggingface_hub, "huggingface-hub")
                    if self._huggingface_hub is not None
                    else None
                ),
            },
            "adapter": {
                "enabled": adapter_enabled,
                "path": adapter_path,
                "sha256": _path_sha256(self.settings.adapter_path),
                "config": _json_safe(adapter_config),
                "base_model_name_or_path": self._adapter_base_model,
                "base_revision": self._adapter_base_revision,
                "compatibility_validation": copy.deepcopy(
                    self._adapter_compatibility
                ),
                "autocast_adapter_dtype": (
                    self.settings.autocast_adapter_dtype
                    if adapter_enabled
                    else None
                ),
                "base_requested_dtype": self._dtype_name if adapter_enabled else None,
                "observed_state_tensor_dtypes": list(self._adapter_state_dtypes),
            },
        }

    def provenance(self) -> dict[str, Any]:
        return copy.deepcopy(self._provenance)

    def generation_provenance(self) -> dict[str, Any]:
        return {
            "do_sample": False,
            "max_new_tokens": self.settings.max_new_tokens,
            "decoding": "deterministic_greedy",
        }

    def generate(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Generate and decode only tokens following the rendered input prompt."""
        try:
            model_inputs = self.tokenizer.apply_chat_template(
                [dict(message) for message in messages],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            if hasattr(model_inputs, "to"):
                model_inputs = model_inputs.to(self.settings.device)
            else:
                model_inputs = {
                    key: (
                        value.to(self.settings.device)
                        if hasattr(value, "to")
                        else value
                    )
                    for key, value in model_inputs.items()
                }
            input_length = model_inputs["input_ids"].shape[-1]
            generation_kwargs: dict[str, Any] = {
                "do_sample": False,
                "max_new_tokens": self.settings.max_new_tokens,
            }
            with self._torch.inference_mode():
                generated = self.model.generate(
                    **model_inputs,
                    **generation_kwargs,
                )
            new_tokens = generated[:, input_length:]
            return self.tokenizer.batch_decode(
                new_tokens,
                skip_special_tokens=True,
            )[0].strip()
        except Exception as error:
            raise HFGenerationError(str(error)) from error


def canonical_provenance_digest(provenance: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(provenance),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
