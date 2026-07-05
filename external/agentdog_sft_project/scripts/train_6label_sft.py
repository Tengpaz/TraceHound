#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_sft import (  # noqa: E402
    DataCollatorForCausalSft,
    JsonlSftDataset,
    configure_reproducibility,
    configure_single_process_distributed_env,
    configure_wandb,
    load_training_config,
    normalize_argparse_defaults,
    parse_report_to,
    require_deepspeed_if_requested,
    resolve_save_config,
)


DEFAULT_TRAIN_FILE = "outputs/data/agentdog_6label_sft.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/models/qwen35-0.8b-6label"


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/training_6label_defaults.json")
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = normalize_argparse_defaults(load_training_config(config_args.config))

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--model-path", default="models/Qwen3.5-0.8B")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--num-checkpoints", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--deepspeed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--full-determinism", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report-to", default="tensorboard,wandb")
    parser.add_argument("--wandb-project", default="agentdog-guardrail")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-watch", choices=["false", "gradients", "parameters", "all"], default="false")
    parser.add_argument("--wandb-run-name")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_args)
    args.config = config_args.config
    return args


def main() -> None:
    args = parse_args()
    if args.print_config:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2, sort_keys=True))
        return

    require_deepspeed_if_requested(args.deepspeed)
    configure_single_process_distributed_env(args.deepspeed)
    configure_wandb(args)
    configure_reproducibility(args.seed, args.full_determinism)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    dataset = JsonlSftDataset(args.train_file, tokenizer, args.max_length)
    save_strategy, save_steps = resolve_save_config(len(dataset), args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        data_seed=args.data_seed,
        full_determinism=args.full_determinism,
        logging_steps=args.logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        bf16=True,
        fp16=False,
        report_to=parse_report_to(args.report_to),
        run_name=args.wandb_run_name,
        do_train=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        deepspeed=args.deepspeed,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "resolved_save_strategy": save_strategy,
        "resolved_save_steps": save_steps,
        "num_examples": len(dataset),
        "evaluation": "disabled",
    }
    with open(Path(args.output_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForCausalSft(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
