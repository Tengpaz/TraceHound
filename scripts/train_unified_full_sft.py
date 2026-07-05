#!/usr/bin/env python
"""Full-parameter SFT for unified AgentDoG-Lite four-label outputs."""

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
from traceguard.unified_sft import (
    balanced_limit,
    read_jsonl,
    split_rows,
    token_length_report,
    training_row_to_messages,
    write_jsonl,
)


TRAIN_PACKAGES = ("torch", "transformers", "accelerate")
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
DEFAULT_OUTPUT = "checkpoints/qwen3_5_0_8b_full_sft_unified_notrunc"
LENGTH_THRESHOLDS = (4096, 8192, 12288, 16384)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--base-model", default=os.getenv("TRACEHOUND_BASE_MODEL", DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--split-dir", help="Where to write train/validation JSONL splits.")
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-seq-length", type=int, default=16384)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Reload the best checkpoint before final save. Disabled by default because some Qwen3.5 "
            "Transformers builds emit incompatible checkpoint key prefixes during best-model reload."
        ),
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run", action="store_true", help="Launch training. Without --run, only write plan/splits.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow --run without CUDA for tiny debugging.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"missing training data: {data_path}")
    rows = read_jsonl(data_path)
    train_rows, val_rows = split_rows(rows, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_train_samples:
        train_rows = balanced_limit(train_rows, args.max_train_samples, args.seed + 1)
    if args.max_val_samples:
        val_rows = balanced_limit(val_rows, args.max_val_samples, args.seed + 2)

    output_dir = Path(args.output_dir)
    split_dir = Path(args.split_dir) if args.split_dir else output_dir / "data_splits"
    train_path = split_dir / "train.jsonl"
    val_path = split_dir / "validation.jsonl"
    write_jsonl(train_path, [training_row_to_messages(row) for row in train_rows])
    write_jsonl(val_path, [training_row_to_messages(row) for row in val_rows])

    plan = build_plan(args, train_path, val_path, train_rows, val_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    missing = missing_packages()
    if missing:
        message = "Missing training packages: " + ", ".join(missing)
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
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "task": "agentdog_lite_unified_four_label_full_sft",
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
        "sequence": {
            "max_seq_length": args.max_seq_length,
            "truncation_policy": "no truncation unless sample exceeds max_seq_length; then left-truncate prompt and preserve target",
            "dynamic_padding": True,
        },
        "trainer": {
            "full_parameter_sft": True,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "warmup_ratio": args.warmup_ratio,
            "save_total_limit": args.save_total_limit,
            "load_best_model_at_end": args.load_best_model_at_end,
            "gradient_checkpointing": args.gradient_checkpointing,
            "dataloader_num_workers": args.dataloader_num_workers,
            "bf16": "auto_cuda_bf16",
        },
    }


def run_training(args: argparse.Namespace, train_path: Path, val_path: Path) -> None:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments  # type: ignore

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not visible. Use --allow-cpu only for tiny local debugging.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype or "auto",
    )
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if torch.cuda.is_available():
        model.to("cuda")

    train_dataset = UnifiedSFTDataset(train_path, tokenizer, args.max_seq_length)
    eval_dataset = UnifiedSFTDataset(val_path, tokenizer, args.max_seq_length)
    length_report = {
        "train": token_length_report(train_dataset.length_records, thresholds=LENGTH_THRESHOLDS),
        "validation": token_length_report(eval_dataset.length_records, thresholds=LENGTH_THRESHOLDS),
    }
    Path(args.output_dir, "token_length_report.json").write_text(
        json.dumps(length_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"token_length_report": length_report}, ensure_ascii=False, indent=2), flush=True)

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
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "eval_loss" if args.load_best_model_at_end else None,
        "greater_is_better": False if args.load_best_model_at_end else None,
        "report_to": [],
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "save_safetensors": True,
        "optim": "adamw_torch",
    }
    training_kwargs[_eval_strategy_key(TrainingArguments)] = "steps"
    if not args.load_best_model_at_end:
        training_kwargs.pop("metric_for_best_model", None)
        training_kwargs.pop("greater_is_better", None)
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


class UnifiedSFTDataset:
    def __init__(self, path: Path, tokenizer: Any, max_seq_length: int) -> None:
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.features: list[dict[str, Any]] = []
        self.length_records: list[dict[str, Any]] = []
        for index, row in enumerate(self.rows, start=1):
            feature, record = self._encode(row)
            self.features.append(feature)
            self.length_records.append(record)
            if index % 100 == 0 or index == len(self.rows):
                print(f"[tracehound] tokenized {index}/{len(self.rows)} rows from {path}", flush=True)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.features[index]

    def _encode(self, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        messages = row["messages"]
        prompt_messages = messages[:-1]
        full_text = apply_chat_template(self.tokenizer, messages, add_generation_prompt=False)
        prompt_text = apply_chat_template(self.tokenizer, prompt_messages, add_generation_prompt=True)
        if not full_text.startswith(prompt_text):
            feature = self._encode_fallback(full_text, prompt_text)
            record = {
                "id": row.get("id"),
                "tokens": len(feature["input_ids"]),
                "kept_tokens": len(feature["input_ids"]),
                "truncated": False,
                "fallback": True,
            }
            return feature, record

        answer_text = full_text[len(prompt_text) :]
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        original_tokens = len(prompt_ids) + len(answer_ids)
        severe_warning = False
        if len(answer_ids) >= self.max_seq_length:
            input_ids = list(answer_ids[: self.max_seq_length])
            labels = list(input_ids)
            truncated = True
            severe_warning = True
        elif original_tokens > self.max_seq_length:
            prompt_budget = max(self.max_seq_length - len(answer_ids), 0)
            kept_prompt_ids = list(prompt_ids[-prompt_budget:]) if prompt_budget else []
            input_ids = kept_prompt_ids + list(answer_ids)
            labels = [-100] * len(kept_prompt_ids) + list(answer_ids)
            truncated = True
        else:
            input_ids = list(prompt_ids) + list(answer_ids)
            labels = [-100] * len(prompt_ids) + list(answer_ids)
            truncated = False
        feature = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
        record = {
            "id": row.get("id"),
            "tokens": original_tokens,
            "kept_tokens": len(input_ids),
            "truncated": truncated,
            "truncated_tokens": max(original_tokens - len(input_ids), 0),
        }
        if severe_warning:
            record["severe_warning"] = "assistant_target_exceeded_max_seq_length"
        return feature, record

    def _encode_fallback(self, full_text: str, prompt_text: str) -> dict[str, Any]:
        full = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prompt = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        if len(full) > self.max_seq_length:
            full = full[-self.max_seq_length :]
        labels = list(full)
        prefix_len = min(len(prompt), len(labels))
        labels[:prefix_len] = [-100] * prefix_len
        return {"input_ids": list(full), "attention_mask": [1] * len(full), "labels": labels}


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
