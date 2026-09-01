from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

SFT_PATH = Path(
    "experiments/rrf_train_peft_full_bcdacb14/train_rrf_sft.jsonl"
)

EXPECTED_SFT_SHA256 = (
    "35164492f536172dcd065b2222646406d067c23ebd441abc31a6418ca50b6471"
)

TOKEN_ANALYSIS_PATH = Path(
    "experiments/stage2_token_analysis/"
    "qwen25_3b_aa8e7253_rrf_train_lengths_v2.json"
)

EXPECTED_TOKEN_ANALYSIS_SHA256 = (
    "b8dc897c123a049b5f6ab76a3d6d31481db80c65733e1e38160cd85cf9004bf8"
)

MAX_SEQ_LENGTH = 1280

EXPECTED_SPLITS = {
    "train": 12970,
    "dev": 1364,
}

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

EPOCHS = 2.0
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_REFERENCE_RATIO = 0.03
FULL_TOTAL_OPTIMIZER_STEPS = 812
FULL_WARMUP_STEPS = 25
MAX_GRAD_NORM = 1.0

PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32
PER_DEVICE_EVAL_BATCH_SIZE = 1

SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def extract_input_ids(obj) -> list[int]:
    if isinstance(obj, Mapping):
        if "input_ids" not in obj:
            raise TypeError(
                f"Tokenizer mapping has no input_ids: "
                f"{list(obj.keys())}"
            )
        ids = obj["input_ids"]

    elif hasattr(obj, "input_ids"):
        ids = obj.input_ids

    else:
        ids = obj

    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    if (
        isinstance(ids, list)
        and ids
        and isinstance(ids[0], list)
    ):
        if len(ids) != 1:
            raise ValueError(
                f"Expected one sequence, got {len(ids)}"
            )

        ids = ids[0]

    if not isinstance(ids, list):
        ids = list(ids)

    if not ids:
        raise ValueError("Empty token sequence.")

    if not all(isinstance(x, int) for x in ids):
        raise TypeError(
            "input_ids is not a flat list[int]."
        )

    return ids


def render_ids(
    tokenizer,
    messages,
    *,
    add_generation_prompt: bool,
) -> list[int]:
    obj = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )

    return extract_input_ids(obj)


def tokenize_record(tokenizer, rec: dict) -> dict:
    if rec["schema_version"] != "rrf_train_sft_v1":
        raise RuntimeError(
            f"Unexpected SFT schema: "
            f"{rec['schema_version']}"
        )

    if rec["split"] not in {"train", "dev"}:
        raise RuntimeError(
            f"Unexpected split: {rec['split']}"
        )

    messages = rec["messages"]
    target = rec["assistant_target"]

    if (
        len(messages) != 2
        or messages[0]["role"] != "system"
        or messages[1]["role"] != "user"
    ):
        raise RuntimeError(
            f"Unexpected message structure: "
            f"{rec['instance_key']}"
        )

    prompt_ids = render_ids(
        tokenizer,
        messages,
        add_generation_prompt=True,
    )

    full_ids = render_ids(
        tokenizer,
        messages
        + [{
            "role": "assistant",
            "content": target,
        }],
        add_generation_prompt=False,
    )

    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            f"Prompt/full prefix mismatch: "
            f"{rec['instance_key']}"
        )

    if len(full_ids) > MAX_SEQ_LENGTH:
        raise RuntimeError(
            f"Sequence exceeds frozen limit: "
            f"{rec['instance_key']} "
            f"length={len(full_ids)} "
            f"max={MAX_SEQ_LENGTH}"
        )

    completion_ids = full_ids[len(prompt_ids):]

    if not completion_ids:
        raise RuntimeError(
            f"Empty supervised completion: "
            f"{rec['instance_key']}"
        )

    decoded_completion = tokenizer.decode(
        completion_ids,
        skip_special_tokens=False,
    )

    if target not in decoded_completion:
        raise RuntimeError(
            f"Target missing from decoded completion: "
            f"{rec['instance_key']}"
        )

    labels = (
        [-100] * len(prompt_ids)
        + completion_ids
    )

    if len(labels) != len(full_ids):
        raise RuntimeError(
            f"Label length mismatch: "
            f"{rec['instance_key']}"
        )

    return {
        "instance_key": rec["instance_key"],
        "split": rec["split"],
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
        "prompt_length": len(prompt_ids),
        "completion_length": len(completion_ids),
    }


class Stage2Dataset(Dataset):
    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "labels": example["labels"],
        }


class CompletionOnlyCollator:
    def __init__(
        self,
        *,
        pad_token_id: int,
        pad_to_multiple_of: int = 8,
    ):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(
            len(x["input_ids"])
            for x in features
        )

        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of

            max_len = (
                (max_len + m - 1) // m
            ) * m

        if max_len > MAX_SEQ_LENGTH:
            raise RuntimeError(
                f"Collated length {max_len} exceeds "
                f"{MAX_SEQ_LENGTH}"
            )

        input_ids = []
        attention_masks = []
        labels = []

        for x in features:
            n = len(x["input_ids"])
            pad = max_len - n

            input_ids.append(
                x["input_ids"]
                + [self.pad_token_id] * pad
            )

            attention_masks.append(
                x["attention_mask"]
                + [0] * pad
            )

            labels.append(
                x["labels"]
                + [-100] * pad
            )

        batch = {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }

        if not torch.any(
            batch["labels"] != -100
        ):
            raise RuntimeError(
                "Batch contains no supervised tokens."
            )

        return batch


def load_and_tokenize(tokenizer):
    if not SFT_PATH.exists():
        raise FileNotFoundError(SFT_PATH)

    actual_sft_sha = sha256_file(SFT_PATH)

    if actual_sft_sha != EXPECTED_SFT_SHA256:
        raise RuntimeError(
            "SFT SHA256 mismatch.\n"
            f"expected={EXPECTED_SFT_SHA256}\n"
            f"actual={actual_sft_sha}"
        )

    if not TOKEN_ANALYSIS_PATH.exists():
        raise FileNotFoundError(
            TOKEN_ANALYSIS_PATH
        )

    actual_analysis_sha = sha256_file(
        TOKEN_ANALYSIS_PATH
    )

    if (
        actual_analysis_sha
        != EXPECTED_TOKEN_ANALYSIS_SHA256
    ):
        raise RuntimeError(
            "Token-analysis SHA256 mismatch.\n"
            f"expected="
            f"{EXPECTED_TOKEN_ANALYSIS_SHA256}\n"
            f"actual={actual_analysis_sha}"
        )

    examples = {
        "train": [],
        "dev": [],
    }

    print("Tokenizing authoritative SFT artifact...")

    with SFT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            rec = json.loads(line)

            try:
                example = tokenize_record(
                    tokenizer,
                    rec,
                )

            except Exception as exc:
                raise RuntimeError(
                    f"Tokenization failed at "
                    f"JSONL line {line_no}"
                ) from exc

            examples[example["split"]].append(
                example
            )

    actual_counts = {
        split: len(items)
        for split, items in examples.items()
    }

    if actual_counts != EXPECTED_SPLITS:
        raise RuntimeError(
            f"Split mismatch: {actual_counts}"
        )

    all_examples = (
        examples["train"]
        + examples["dev"]
    )

    lengths = [
        x["length"]
        for x in all_examples
    ]

    completions = [
        x["completion_length"]
        for x in all_examples
    ]

    print(
        "TRAIN examples:",
        len(examples["train"]),
    )
    print(
        "DEV examples:",
        len(examples["dev"]),
    )
    print(
        "min/full/max:",
        min(lengths),
        max(lengths),
    )
    print(
        "completion min/max:",
        min(completions),
        max(completions),
    )

    if max(lengths) != 1275:
        raise RuntimeError(
            f"Unexpected maximum sequence "
            f"length: {max(lengths)}"
        )

    if max(lengths) > MAX_SEQ_LENGTH:
        raise RuntimeError(
            "Frozen max sequence length violated."
        )

    return examples


def write_json(path: Path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
    ):
        raise RuntimeError(
            f"Output directory is not empty: "
            f"{output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "This frozen trainer expects exactly "
            "one visible GPU. Launch with "
            "CUDA_VISIBLE_DEVICES=0."
        )

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Visible GPU does not support BF16."
        )

    print("GPU:", torch.cuda.get_device_name(0))
    print("torch:", torch.__version__)
    print(
        "transformers:",
        transformers.__version__,
    )
    print("peft:", peft.__version__)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )

    if not tokenizer.chat_template:
        raise RuntimeError(
            "Tokenizer has no chat template."
        )

    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    examples = load_and_tokenize(
        tokenizer
    )

    if args.smoke:
        train_examples = (
            examples["train"][:8]
        )
        dev_examples = (
            examples["dev"][:4]
        )

        gradient_accumulation = 2
        max_steps = 2
        epochs = 1.0
        warmup_steps = 1

        print(
            "SMOKE MODE:",
            len(train_examples),
            "train /",
            len(dev_examples),
            "dev",
        )

    else:
        train_examples = examples["train"]
        dev_examples = examples["dev"]

        gradient_accumulation = (
            GRADIENT_ACCUMULATION_STEPS
        )
        max_steps = -1
        epochs = EPOCHS
        warmup_steps = FULL_WARMUP_STEPS

    manifest = {
        "experiment":
            "stage2_qwen25_3b_lora_reranker",
        "smoke": args.smoke,
        "git_commit": git_commit(),
        "source": {
            "sft_path": str(SFT_PATH),
            "sft_sha256":
                EXPECTED_SFT_SHA256,
            "token_analysis_path":
                str(TOKEN_ANALYSIS_PATH),
            "token_analysis_sha256":
                EXPECTED_TOKEN_ANALYSIS_SHA256,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers":
                transformers.__version__,
            "peft": peft.__version__,
            "gpu":
                torch.cuda.get_device_name(0),
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "attention":
                "sdpa",
        },
        "data": {
            "train_examples":
                len(train_examples),
            "dev_examples":
                len(dev_examples),
            "max_seq_length":
                MAX_SEQ_LENGTH,
            "completion_only_loss":
                True,
            "padding":
                "dynamic_right_multiple_of_8",
            "truncation":
                False,
        },
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "bias": "none",
            "targets": LORA_TARGETS,
        },
        "optimization": {
            "epochs": epochs,
            "max_steps": max_steps,
            "learning_rate":
                LEARNING_RATE,
            "weight_decay":
                WEIGHT_DECAY,
            "scheduler":
                "linear",
            "warmup_steps":
                warmup_steps,
            "warmup_reference_ratio":
                WARMUP_REFERENCE_RATIO,
            "planned_full_optimizer_steps":
                FULL_TOTAL_OPTIMIZER_STEPS,
            "full_warmup_steps":
                FULL_WARMUP_STEPS,
            "max_grad_norm":
                MAX_GRAD_NORM,
            "per_device_train_batch_size":
                PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps":
                gradient_accumulation,
            "effective_batch_size":
                PER_DEVICE_TRAIN_BATCH_SIZE
                * gradient_accumulation,
            "seed": SEED,
        },
    }

    write_json(
        output_dir / "run_manifest.json",
        manifest,
    )

    train_dataset = Stage2Dataset(
        train_examples
    )

    dev_dataset = Stage2Dataset(
        dev_examples
    )

    collator = CompletionOnlyCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )

    print("Loading base model...")

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    )

    model.config.use_cache = False

    target_counts = {}

    module_names = [
        name
        for name, _ in model.named_modules()
    ]

    for target in LORA_TARGETS:
        count = sum(
            name.endswith("." + target)
            for name in module_names
        )

        target_counts[target] = count

        if count != 36:
            raise RuntimeError(
                f"Expected 36 {target} modules; "
                f"found {count}"
            )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LORA_TARGETS,
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )

    model.enable_input_require_grads()

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    if trainable_params != 7_372_800:
        raise RuntimeError(
            f"Unexpected trainable parameter "
            f"count: {trainable_params}"
        )

    manifest["lora"][
        "target_module_counts"
    ] = target_counts

    manifest["lora"][
        "trainable_parameters"
    ] = trainable_params

    manifest["lora"][
        "total_parameters"
    ] = total_params

    manifest["lora"][
        "trainable_percentage"
    ] = (
        100.0
        * trainable_params
        / total_params
    )

    write_json(
        output_dir / "run_manifest.json",
        manifest,
    )

    if args.smoke:
        eval_strategy = "steps"
        eval_steps = 1
        save_strategy = "no"
        logging_steps = 1

    else:
        eval_strategy = "epoch"
        eval_steps = None
        save_strategy = "epoch"
        logging_steps = 10

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=(
            PER_DEVICE_TRAIN_BATCH_SIZE
        ),
        per_device_eval_batch_size=(
            PER_DEVICE_EVAL_BATCH_SIZE
        ),
        gradient_accumulation_steps=(
            gradient_accumulation
        ),
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_steps=warmup_steps,
        optim="adamw_torch",
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },
        use_cache=False,
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_total_limit=2,
        load_best_model_at_end=False,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )

    print("\nStarting Trainer...")

    train_result = trainer.train()

    print("\nTraining completed.")

    final_adapter = (
        output_dir / "final_adapter"
    )

    trainer.save_model(
        str(final_adapter)
    )

    tokenizer.save_pretrained(
        final_adapter
    )

    trainer.save_state()

    eval_metrics = trainer.evaluate()

    train_metrics = dict(
        train_result.metrics
    )

    trainer.log_metrics(
        "train",
        train_metrics,
    )

    trainer.save_metrics(
        "train",
        train_metrics,
    )

    trainer.log_metrics(
        "eval",
        eval_metrics,
    )

    trainer.save_metrics(
        "eval",
        eval_metrics,
    )

    max_allocated_gib = (
        torch.cuda.max_memory_allocated()
        / (1024 ** 3)
    )

    max_reserved_gib = (
        torch.cuda.max_memory_reserved()
        / (1024 ** 3)
    )

    summary = {
        "train_metrics":
            train_metrics,
        "eval_metrics":
            eval_metrics,
        "max_gpu_memory_allocated_gib":
            max_allocated_gib,
        "max_gpu_memory_reserved_gib":
            max_reserved_gib,
        "final_adapter":
            str(final_adapter),
    }

    write_json(
        output_dir / "training_summary.json",
        summary,
    )

    print(
        "Max GPU allocated:",
        f"{max_allocated_gib:.3f} GiB",
    )
    print(
        "Max GPU reserved:",
        f"{max_reserved_gib:.3f} GiB",
    )

    print("\nTesting adapter reload...")

    del trainer
    del model

    gc.collect()
    torch.cuda.empty_cache()

    base = (
        AutoModelForCausalLM
        .from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
    )

    reloaded = PeftModel.from_pretrained(
        base,
        final_adapter,
    )

    reloaded.to("cuda:0")
    reloaded.eval()
    reloaded.config.use_cache = True

    verify_example = train_examples[0]

    prompt_length = (
        verify_example["prompt_length"]
    )

    prompt_ids = (
        verify_example["input_ids"]
        [:prompt_length]
    )

    input_tensor = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device="cuda:0",
    )

    with torch.inference_mode():
        generated = reloaded.generate(
            input_ids=input_tensor,
            attention_mask=torch.ones_like(
                input_tensor
            ),
            max_new_tokens=64,
            do_sample=False,
        )

    new_ids = generated[
        0,
        input_tensor.shape[1]:,
    ]

    generated_text = tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
    ).strip()

    if not generated_text:
        raise RuntimeError(
            "Reloaded adapter generated "
            "empty output."
        )

    reload_result = {
        "instance_key":
            verify_example["instance_key"],
        "generated_text":
            generated_text,
        "nonempty":
            True,
    }

    write_json(
        output_dir
        / "adapter_reload_generation.json",
        reload_result,
    )

    print(
        "Reload generation:",
        generated_text,
    )

    print(
        "\nADAPTER_SAVE_RELOAD_OK"
    )

    if args.smoke:
        print(
            "STAGE2_LORA_SMOKE_OK"
        )
    else:
        print(
            "STAGE2_LORA_FULL_TRAIN_OK"
        )


if __name__ == "__main__":
    main()
