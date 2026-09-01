"""Run the one-record Stage-2 v2 real-Qwen PEFT/GPU integration smoke."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from my_crs.build_stage2_v2_dataset import TOP_K
from my_crs.stage2_v2_peft import (
    DEFAULT_DATASET_PATH,
    DEFAULT_SMOKE_OUTPUT_DIR,
    EXPECTED_DATASET_SHA256,
    MAX_PACKED_TOKENS,
    REPORT_SCHEMA_VERSION,
    apply_peft_and_build_ranker,
    cuda_memory_snapshot,
    load_base_qwen,
    load_production_tokenizer,
    parameter_dtype_report,
    phase3b_integration_fingerprint,
    phase3b_scientific_configuration,
    real_model_permutation_check,
    require_single_cuda_device,
    run_gradient_plumbing_smoke,
    runtime_provenance,
    stream_record_by_instance_key,
    tokenize_single_smoke_event,
    validate_tokenizer,
    write_smoke_report,
)


def run_real_model_smoke(
    *,
    dataset_path: str | Path,
    instance_key: str,
    output_dir: str | Path,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """Execute integration checks only; this is not a training entry point."""

    device = require_single_cuda_device(device_name)
    torch.cuda.reset_peak_memory_stats(device)
    scientific_configuration = phase3b_scientific_configuration()
    integration_fingerprint = phase3b_integration_fingerprint()

    record, observed_dataset_sha = stream_record_by_instance_key(
        dataset_path,
        instance_key,
        expected_sha256=EXPECTED_DATASET_SHA256,
    )
    tokenizer = load_production_tokenizer()
    resolved_tokenizer_commit = validate_tokenizer(tokenizer)

    _, normal_batch_cpu, actual_tokens = tokenize_single_smoke_event(
        record,
        tokenizer,
    )
    _, reversed_batch_cpu, reversed_tokens = tokenize_single_smoke_event(
        record,
        tokenizer,
        physical_block_positions=list(reversed(range(1, TOP_K + 1))),
    )
    if reversed_tokens != actual_tokens:
        raise RuntimeError(
            "Physical block reversal changed total packed length: "
            f"normal={actual_tokens} reversed={reversed_tokens}"
        )

    import peft

    memory: dict[str, dict[str, int]] = {}
    base_model, model_identity = load_base_qwen(device)
    memory["after_base_model_load"] = cuda_memory_snapshot(device)

    ranker, trainability = apply_peft_and_build_ranker(
        base_model,
        device,
        peft_module=peft,
    )
    memory["after_peft_and_ranker_construction"] = cuda_memory_snapshot(device)
    dtypes = parameter_dtype_report(ranker)

    normal_batch = normal_batch_cpu.to(device)
    reversed_batch = reversed_batch_cpu.to(device)
    if normal_batch.input_ids.shape[1] != actual_tokens:
        raise RuntimeError("GPU smoke batch was padded beyond its actual token length")
    memory["after_tokenized_events_move_to_gpu"] = cuda_memory_snapshot(device)

    permutation = real_model_permutation_check(
        ranker,
        normal_batch,
        reversed_batch,
    )
    memory["after_permutation_forward_check"] = cuda_memory_snapshot(device)

    def capture_memory(checkpoint: str) -> None:
        memory[checkpoint] = cuda_memory_snapshot(device)

    gradient_smoke = run_gradient_plumbing_smoke(
        ranker,
        normal_batch,
        memory_callback=capture_memory,
    )
    dtypes["raw_residual_dtype_backward_1"] = gradient_smoke["backward_1"][
        "raw_residual_dtype"
    ]
    dtypes["raw_residual_dtype_backward_2"] = gradient_smoke["backward_2"][
        "raw_residual_dtype"
    ]

    report = {
        "actual_packed_tokens": actual_tokens,
        "checks": {
            "backward_1": True,
            "backward_2_reaches_lora": True,
            "base_parameters_frozen": True,
            "bf16_base": dtypes["base_parameter_dtypes"] == ["bfloat16"],
            "custom_mask_and_position_permutation": permutation["passed"],
            "gradient_checkpointing_non_reentrant": True,
            "optimizer_step": gradient_smoke["optimizer"][
                "projection_nonzero_after_step"
            ],
            "packed_length_within_ceiling": actual_tokens <= MAX_PACKED_TOKENS,
            "peft_lora_trainable": trainability["lora_trainable_parameters"] > 0,
            "scorer_head_trainable": (
                trainability["scorer_head_trainable_parameters"] > 0
            ),
            "sdpa": True,
        },
        "dataset": {
            "path": str(Path(dataset_path)),
            "sha256": observed_dataset_sha,
        },
        "dtype_report": dtypes,
        "gpu_memory": memory,
        "gradient_smoke": gradient_smoke,
        "instance_key": instance_key,
        "integration_fingerprint": integration_fingerprint,
        "lora_modules": trainability["actual_lora_module_names"],
        "model_identity": model_identity,
        "permutation_check": permutation,
        "runtime_provenance": runtime_provenance(
            device=device,
            peft_module=peft,
            tokenizer=tokenizer,
            resolved_tokenizer_commit=resolved_tokenizer_commit,
        ),
        "schema_version": REPORT_SCHEMA_VERSION,
        "scientific_configuration": scientific_configuration,
        "trainability": trainability,
    }
    if not all(report["checks"].values()):
        raise RuntimeError(f"One or more Phase-3B smoke checks failed: {report['checks']}")
    output_path = write_smoke_report(report, output_dir)
    return {
        "actual_packed_tokens": actual_tokens,
        "integration_fingerprint": integration_fingerprint,
        "report_path": str(output_path),
        "status": "passed",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--instance-key", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SMOKE_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_real_model_smoke(
        dataset_path=args.dataset_path,
        instance_key=args.instance_key,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
