#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail.taxonomy import canonicalize  # noqa: E402
from scripts.train_sft import (  # noqa: E402
    configure_reproducibility,
    configure_single_process_distributed_env,
    configure_wandb,
    load_training_config,
    normalize_argparse_defaults,
    parse_report_to,
    require_deepspeed_if_requested,
    resolve_save_config,
)


DEFAULT_CONFIG = "configs/grpo_6label_defaults.json"
TARGET_FIELDS = ("risk_source", "failure_mode", "harm_type", "source", "judgment")
REQUIRED_JSON_FIELDS = set(TARGET_FIELDS) | {"rationale"}
SOURCE_LABELS = {"benign", "safe", "unsafe", "false_refusal"}
JUDGMENT_LABELS = {"safe", "unsafe"}


def clean_markdown_escapes(text: str) -> str:
    text = text.replace("\\_", "_")
    text = text.replace("\\.", ".")
    text = text.replace("\\(", "(").replace("\\)", ")")
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.replace("\\<", "<").replace("\\>", ">")
    return text


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = clean_markdown_escapes(text.strip())
    candidates = [stripped]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.append(fence.group(1))
    obj_match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group(0))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def normalize_label(value: Any) -> str | None:
    if value is None:
        return None
    return canonicalize(str(value))


def normalize_source(value: Any) -> str | None:
    label = normalize_label(value)
    return label if label in SOURCE_LABELS else None


def normalize_judgment(value: Any) -> str | None:
    label = normalize_label(value)
    return label if label in JUDGMENT_LABELS else None


def normalize_field(field: str, value: Any) -> str | None:
    if field == "source":
        return normalize_source(value)
    if field == "judgment":
        return normalize_judgment(value)
    return normalize_label(value)


def parse_json_object_arg(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("expected a valid JSON object") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a valid JSON object")
    return parsed


def parse_6label_json(text: str, *, strict_keys: bool) -> tuple[dict[str, str | None], str | None, bool]:
    obj = extract_json_object(text)
    if not obj:
        return {field: None for field in TARGET_FIELDS}, None, False
    if strict_keys and set(obj.keys()) != REQUIRED_JSON_FIELDS:
        return {field: None for field in TARGET_FIELDS}, None, False

    parsed = {field: normalize_field(field, obj.get(field)) for field in TARGET_FIELDS}
    rationale = obj.get("rationale")
    rationale_text = rationale.strip() if isinstance(rationale, str) else None
    valid = rationale_text is not None and all(parsed[field] is not None for field in TARGET_FIELDS)
    return parsed, rationale_text, valid


def encode_chat_prompt(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def iter_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc


def load_grpo_dataset(path: str | Path, tokenizer, limit: int | None = None) -> Dataset:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for line_number, row in iter_jsonl(path):
        if limit is not None and len(rows) >= limit:
            break
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            skipped += 1
            continue
        user_message = messages[0]
        assistant_message = messages[-1]
        if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
            skipped += 1
            continue

        expected, _, valid = parse_6label_json(str(assistant_message.get("content", "")), strict_keys=False)
        if not valid:
            skipped += 1
            continue

        record = {
            "prompt": encode_chat_prompt(tokenizer, str(user_message.get("content", ""))),
            "source_line": line_number,
        }
        for field in TARGET_FIELDS:
            record[f"target_{field}"] = expected[field]
        rows.append(record)

    if not rows:
        raise ValueError(f"No usable GRPO rows loaded from {path}; skipped={skipped}")
    dataset = Dataset.from_list(rows)
    print(json.dumps({"train_file": str(path), "num_examples": len(rows), "skipped": skipped}, indent=2))
    return dataset


class SixLabelReward:
    def __init__(
        self,
        tokenizer,
        reward_weights: dict[str, float],
        rationale_min_tokens: int,
        rationale_max_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.reward_weights = dict(reward_weights)
        self.rationale_min_tokens = rationale_min_tokens
        self.rationale_max_tokens = rationale_max_tokens
        missing = [field for field in TARGET_FIELDS if field not in self.reward_weights]
        if missing:
            raise ValueError(f"Missing reward weights for: {', '.join(missing)}")

    def rationale_token_count(self, rationale: str) -> int:
        return len(self.tokenizer(rationale, add_special_tokens=False)["input_ids"])

    def score_one(self, completion: str, expected: dict[str, str]) -> float:
        parsed, rationale, valid = parse_6label_json(completion, strict_keys=True)
        if not valid or rationale is None:
            return 0.0

        rationale_tokens = self.rationale_token_count(rationale)
        if rationale_tokens < self.rationale_min_tokens or rationale_tokens > self.rationale_max_tokens:
            return 0.0

        reward = 0.0
        for field in TARGET_FIELDS:
            if parsed[field] == expected[field]:
                reward += float(self.reward_weights[field])
        return float(reward)

    def __call__(self, completions, **kwargs) -> list[float]:
        rewards: list[float] = []
        for idx, completion in enumerate(completions):
            content = completion
            if isinstance(completion, list) and completion and isinstance(completion[0], dict):
                content = completion[0].get("content", "")
            expected = {field: kwargs[f"target_{field}"][idx] for field in TARGET_FIELDS}
            rewards.append(self.score_one(str(content), expected))
        return rewards


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG)
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = normalize_argparse_defaults(load_training_config(config_args.config))

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--model-path", default="outputs/models/qwen35-0.8b-6label")
    parser.add_argument("--train-file", default="outputs/data/agentdog_6label_sft.jsonl")
    parser.add_argument("--output-dir", default="outputs/models/qwen35-0.8b-6label-grpo")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--reward-smoke-test", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--num-checkpoints", type=int, default=0)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--use-vllm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--scale-rewards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-type", default="dr_grpo")
    parser.add_argument("--mask-truncated-completions", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--reward-weights", type=parse_json_object_arg)
    parser.add_argument("--rationale-min-tokens", type=int, default=50)
    parser.add_argument("--rationale-max-tokens", type=int, default=300)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_args)
    args.config = config_args.config
    return args


def build_grpo_config(args, dataset_len: int):
    try:
        from trl import GRPOConfig
    except ImportError as exc:
        raise SystemExit(
            "TRL is required for GRPO training. Install it inside the train env first, "
            "for example: conda run -n train python -m pip install 'trl>=0.17.0'"
        ) from exc

    save_strategy, save_steps = resolve_save_config(dataset_len, args)
    candidates = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "full_determinism": args.full_determinism,
        "logging_steps": args.logging_steps,
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": True,
        "fp16": False,
        "report_to": parse_report_to(args.report_to),
        "run_name": args.wandb_run_name,
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "deepspeed": args.deepspeed,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "generation_batch_size": args.generation_batch_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "use_vllm": args.use_vllm,
        "beta": args.beta,
        "epsilon": args.epsilon,
        "scale_rewards": args.scale_rewards,
        "loss_type": args.loss_type,
        "mask_truncated_completions": args.mask_truncated_completions,
    }
    signature = inspect.signature(GRPOConfig.__init__)
    supported = {key for key in signature.parameters if key != "self"}
    filtered = {key: value for key, value in candidates.items() if key in supported and value is not None}
    return GRPOConfig(**filtered), save_strategy, save_steps


def run_reward_smoke_test(args, tokenizer) -> None:
    reward_func = SixLabelReward(
        tokenizer=tokenizer,
        reward_weights=args.reward_weights,
        rationale_min_tokens=args.rationale_min_tokens,
        rationale_max_tokens=args.rationale_max_tokens,
    )
    reward_func.__name__ = "six_label_reward"
    rewards = []
    for _, row in iter_jsonl(args.train_file):
        if args.limit is not None and len(rewards) >= args.limit:
            break
        messages = row.get("messages", [])
        if len(messages) < 2:
            continue
        completion = str(messages[-1].get("content", ""))
        expected, _, valid = parse_6label_json(completion, strict_keys=False)
        if not valid:
            rewards.append(0.0)
            continue
        rewards.append(reward_func.score_one(completion, {field: expected[field] for field in TARGET_FIELDS}))
    print(json.dumps({"num_examples": len(rewards), "rewards": rewards, "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0}, indent=2))


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
    tokenizer.padding_side = "left"

    if args.reward_smoke_test:
        run_reward_smoke_test(args, tokenizer)
        return

    try:
        from trl import GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "TRL is required for GRPO training. Install it inside the train env first, "
            "for example: conda run -n train python -m pip install 'trl>=0.17.0'"
        ) from exc

    train_dataset = load_grpo_dataset(args.train_file, tokenizer, args.limit)
    training_args, save_strategy, save_steps = build_grpo_config(args, len(train_dataset))
    reward_func = SixLabelReward(
        tokenizer=tokenizer,
        reward_weights=args.reward_weights,
        rationale_min_tokens=args.rationale_min_tokens,
        rationale_max_tokens=args.rationale_max_tokens,
    )
    reward_func.__name__ = "six_label_reward"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args) | {
        "resolved_save_strategy": save_strategy,
        "resolved_save_steps": save_steps,
        "num_examples": len(train_dataset),
        "reward_fields": TARGET_FIELDS,
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
