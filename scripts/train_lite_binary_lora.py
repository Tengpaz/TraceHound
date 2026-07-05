#!/usr/bin/env python
"""LoRA SFT for AgentDoG-Lite binary JSON-judgment data."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.chat_format import apply_chat_template
from traceguard.lite_binary_eval import read_jsonl, split_sft_rows, write_jsonl


TRAIN_PACKAGES = ("torch", "transformers", "peft", "accelerate")
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_DATA = "data/release/AgentDoG-Lite-TrainningDataset-Binary/messages/train.jsonl"
DEFAULT_OUTPUT = "checkpoints/qwen3_5_0_8b_lite_binary_lora"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--base-model", default=os.getenv("TRACEHOUND_BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--split-dir", help="Where to write train/validation JSONL splits.")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--pretokenize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA module suffixes.",
    )
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run", action="store_true", help="Launch training. Without --run, only write the plan/splits.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow --run without CUDA for tiny debugging.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"missing training data: {data_path}")
    rows = read_jsonl(data_path)
    train_rows, val_rows = split_sft_rows(rows, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_train_samples:
        train_rows = _balanced_limit(train_rows, args.max_train_samples, args.seed + 1)
    if args.max_val_samples:
        val_rows = _balanced_limit(val_rows, args.max_val_samples, args.seed + 2)

    output_dir = Path(args.output_dir)
    split_dir = Path(args.split_dir) if args.split_dir else output_dir / "data_splits"
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "validation.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    plan = build_plan(args, train_path, val_path, train_rows, val_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    missing = missing_packages()
    if missing:
        message = (
            "Missing training packages: "
            + ", ".join(missing)
            + "\nInstall CUDA PyTorch first, then `pip install -e .[train]`."
        )
        if args.run:
            raise SystemExit(message)
        print(message, flush=True)
        return
    if not args.run:
        print("Splits and plan are ready. Add --run on the GPU server to train.", flush=True)
        return
    run_training(args, train_path, val_path)


def build_plan(
    args: argparse.Namespace,
    train_path: Path,
    val_path: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task": "agentdog_lite_binary_lora_sft",
        "base_model": args.base_model,
        "output_dir": args.output_dir,
        "source_data": args.data,
        "splits": {
            "train": str(train_path),
            "validation": str(val_path),
            "train_samples": len(train_rows),
            "validation_samples": len(val_rows),
            "train_labels": _label_counts(train_rows),
            "validation_labels": _label_counts(val_rows),
            "seed": args.seed,
            "val_ratio": args.val_ratio,
        },
        "sequence": {"max_seq_length": args.max_seq_length},
        "trainer": {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "warmup_ratio": args.warmup_ratio,
            "save_total_limit": args.save_total_limit,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dataloader_num_workers": args.dataloader_num_workers,
            "pretokenize": args.pretokenize,
        },
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": _target_modules(args.target_modules),
        },
        "space_policy": "LoRA adapter only; save_total_limit keeps at most one intermediate checkpoint.",
    }


def run_training(args: argparse.Namespace, train_path: Path, val_path: Path) -> None:
    import torch  # type: ignore
    from peft import LoraConfig, get_peft_model  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # type: ignore

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not visible. Use --allow-cpu only for tiny local debugging.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto",
        device_map="auto",
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_target_modules(args.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset = LiteSFTDataset(train_path, tokenizer, args.max_seq_length, pretokenize=args.pretokenize)
    eval_dataset = LiteSFTDataset(val_path, tokenizer, args.max_seq_length, pretokenize=args.pretokenize)
    training_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "save_strategy": "steps",
        "eval_steps": args.eval_steps,
        "report_to": [],
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "save_safetensors": True,
    }
    training_kwargs[_eval_strategy_key(TrainingArguments)] = "steps"
    training_kwargs = _filter_training_args(TrainingArguments, training_kwargs)
    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalLMCollator(tokenizer.pad_token_id or 0),
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir, "eval_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class LiteSFTDataset:
    def __init__(self, path: Path, tokenizer: Any, max_seq_length: int, *, pretokenize: bool = True) -> None:
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.features: list[dict[str, Any]] | None = None
        if pretokenize:
            self.features = []
            for index, row in enumerate(self.rows, start=1):
                self.features.append(self._encode(row))
                if index % 200 == 0 or index == len(self.rows):
                    print(f"[tracehound] tokenized {index}/{len(self.rows)} rows from {path}", flush=True)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.features is not None:
            return self.features[index]
        row = self.rows[index]
        return self._encode(row)

    def _encode(self, row: Mapping[str, Any]) -> dict[str, Any]:
        messages = row["messages"]
        prompt_messages = messages[:-1]
        full_text = apply_chat_template(self.tokenizer, messages, add_generation_prompt=False)
        prompt_text = apply_chat_template(self.tokenizer, prompt_messages, add_generation_prompt=True)
        encoded = self._encode_preserve_target(full_text, prompt_text)
        if encoded is not None:
            return encoded

        full = self.tokenizer(full_text, truncation=True, max_length=self.max_seq_length)
        prompt = self.tokenizer(prompt_text, truncation=True, max_length=self.max_seq_length)
        labels = list(full["input_ids"])
        prefix_len = min(len(prompt["input_ids"]), len(labels))
        labels[:prefix_len] = [-100] * prefix_len
        return {
            "input_ids": list(full["input_ids"]),
            "attention_mask": list(full["attention_mask"]),
            "labels": labels,
        }

    def _encode_preserve_target(self, full_text: str, prompt_text: str) -> dict[str, Any] | None:
        """Left-truncate long prompts while keeping the short JSON target supervised."""
        if not full_text.startswith(prompt_text):
            return None
        answer_text = full_text[len(prompt_text) :]
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        if not answer_ids:
            return None
        if len(answer_ids) >= self.max_seq_length:
            input_ids = list(answer_ids[: self.max_seq_length])
            labels = list(input_ids)
        else:
            prompt_budget = max(self.max_seq_length - len(answer_ids), 0)
            kept_prompt_ids = list(prompt_ids[-prompt_budget:]) if prompt_budget else []
            input_ids = kept_prompt_ids + list(answer_ids)
            labels = [-100] * len(kept_prompt_ids) + list(answer_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


class CausalLMCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[Mapping[str, Any]]) -> dict[str, Any]:
        import torch  # type: ignore

        max_len = max(len(item["input_ids"]) for item in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad_len = max_len - len(item["input_ids"])
            batch["input_ids"].append(list(item["input_ids"]) + [self.pad_token_id] * pad_len)
            batch["attention_mask"].append(list(item["attention_mask"]) + [0] * pad_len)
            batch["labels"].append(list(item["labels"]) + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def missing_packages() -> list[str]:
    return [package for package in TRAIN_PACKAGES if importlib.util.find_spec(package) is None]


def _label_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label"))
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _balanced_limit(rows: Sequence[Mapping[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    import random

    if limit >= len(rows):
        return [dict(row) for row in rows]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("label")), []).append(dict(row))
    rng = random.Random(seed)
    for group_rows in groups.values():
        rng.shuffle(group_rows)
    labels = sorted(groups)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(groups.values()):
        for label in labels:
            if groups[label] and len(selected) < limit:
                selected.append(groups[label].pop())
    rng.shuffle(selected)
    return selected


def _target_modules(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _eval_strategy_key(training_args_cls: Any) -> str:
    import inspect

    params = inspect.signature(training_args_cls.__init__).parameters
    return "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"


def _filter_training_args(training_args_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    import inspect

    params = inspect.signature(training_args_cls.__init__).parameters
    return {key: value for key, value in kwargs.items() if key in params}


if __name__ == "__main__":
    main()
