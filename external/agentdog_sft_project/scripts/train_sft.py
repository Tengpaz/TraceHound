#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail.metrics import binary_accuracy, taxonomy_metrics
from guardrail.prompts import (
    binary_target,
    build_binary_prompt,
    build_taxonomy_prompt,
    chat_messages,
)
from guardrail.taxonomy import normalize_taxonomy_value, parse_binary, parse_taxonomy


def _disable_torch_compile():
    def no_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    torch.compile = no_compile


_disable_torch_compile()


class JsonlSftDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_length: int):
        self.rows = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self.rows[idx]["messages"]
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            add_generation_prompt=True,
            tokenize=False,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=False,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        assistant_ids = full_ids[len(prompt_ids) :]

        if len(assistant_ids) >= self.max_length:
            prompt_ids = []
            assistant_ids = assistant_ids[: self.max_length]
        else:
            prompt_budget = self.max_length - len(assistant_ids)
            prompt_ids = prompt_ids[-prompt_budget:]

        input_ids = prompt_ids + assistant_ids
        labels = [-100] * len(prompt_ids) + assistant_ids.copy()
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollatorForCausalSft:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch["input_ids"].append(
                torch.nn.functional.pad(
                    feature["input_ids"],
                    (0, pad_len),
                    value=self.tokenizer.pad_token_id,
                )
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(feature["attention_mask"], (0, pad_len), value=0)
            )
            batch["labels"].append(
                torch.nn.functional.pad(feature["labels"], (0, pad_len), value=-100)
            )
        return {key: torch.stack(value) for key, value in batch.items()}


class AtBenchEvalCallback(TrainerCallback):
    def __init__(
        self,
        enabled: bool,
        task: str,
        benchmark_path: str,
        eval_steps: int,
        eval_limit: int | None,
        tokenizer,
        output_dir: str,
    ):
        self.enabled = enabled
        self.task = task
        self.benchmark_path = benchmark_path
        self.eval_steps = eval_steps
        self.eval_limit = eval_limit
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.data = None
        self.pending_logs: dict[str, float] | None = None

    def load_data(self) -> list[dict]:
        if self.data is None:
            with open(self.benchmark_path, encoding="utf-8") as f:
                self.data = json.load(f)
            if not isinstance(self.data, list):
                raise ValueError(f"Expected a JSON list: {self.benchmark_path}")
        return self.data

    def generation_model(self, model):
        return model.module if hasattr(model, "module") else model

    def model_device(self, model) -> torch.device:
        return next(self.generation_model(model).parameters()).device

    def generate(self, model, prompt: str, max_new_tokens: int) -> str:
        text = self.tokenizer.apply_chat_template(
            chat_messages(prompt),
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model_device(model))
        gen_model = self.generation_model(model)
        with torch.no_grad():
            outputs = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def evaluate_binary(self, model, data: list[dict]) -> tuple[list[dict], dict]:
        records = []
        for example in data[: self.eval_limit]:
            raw = self.generate(model, build_binary_prompt(example), max_new_tokens=8)
            records.append(
                {
                    "conv_id": example.get("conv_id"),
                    "target": binary_target(example),
                    "raw_prediction": raw,
                    "prediction": parse_binary(raw),
                }
            )
        return records, binary_accuracy(records)

    def evaluate_taxonomy(self, model, data: list[dict]) -> tuple[list[dict], dict]:
        unsafe = [x for x in data if int(x["label"]) == 1]
        records = []
        for example in unsafe[: self.eval_limit]:
            raw = self.generate(model, build_taxonomy_prompt(example), max_new_tokens=96)
            records.append(
                {
                    "conv_id": example.get("conv_id"),
                    "target": {
                        "risk_source": normalize_taxonomy_value("risk_source", example["risk_source"]),
                        "failure_mode": normalize_taxonomy_value("failure_mode", example["failure_mode"]),
                        "real_world_harm": normalize_taxonomy_value(
                            "real_world_harm", example["real_world_harm"]
                        ),
                    },
                    "raw_prediction": raw,
                    "prediction": parse_taxonomy(raw),
                }
            )
        return records, taxonomy_metrics(records)

    def write_predictions(self, step: int, records: list[dict], metrics: dict) -> None:
        eval_dir = self.output_dir / "atbench_eval" / f"step-{step}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(eval_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def flatten_metrics(self, metrics: dict, prefix: str = "atbench") -> dict[str, float]:
        flat = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                flat.update(self.flatten_metrics(value, f"{prefix}/{key}"))
            elif isinstance(value, (int, float)):
                flat[f"{prefix}/{key}"] = value
        return flat

    def run_eval(self, model, step: int) -> dict[str, float]:
        data = self.load_data()
        was_training = self.generation_model(model).training
        self.generation_model(model).eval()
        if self.task == "binary":
            records, metrics = self.evaluate_binary(model, data)
            self.write_predictions(step, records, metrics)
        elif self.task == "taxonomy":
            records, metrics = self.evaluate_taxonomy(model, data)
            self.write_predictions(step, records, metrics)
        else:
            binary_records, binary_metrics = self.evaluate_binary(model, data)
            taxonomy_records, taxonomy_metrics_result = self.evaluate_taxonomy(model, data)
            metrics = {"binary": binary_metrics, "taxonomy": taxonomy_metrics_result}
            self.write_predictions(step, binary_records, {"binary": binary_metrics})
            taxonomy_dir = self.output_dir / "atbench_eval" / f"step-{step}" / "taxonomy"
            taxonomy_dir.mkdir(parents=True, exist_ok=True)
            with open(taxonomy_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
                for record in taxonomy_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open(taxonomy_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(taxonomy_metrics_result, f, ensure_ascii=False, indent=2)
        if was_training:
            self.generation_model(model).train()
        return self.flatten_metrics(metrics)

    def on_step_end(self, args, state, control, **kwargs):
        if not self.enabled or self.eval_steps <= 0:
            return
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return
        if not state.is_world_process_zero:
            return
        metrics = self.run_eval(kwargs["model"], state.global_step)
        metrics["step"] = state.global_step
        self.pending_logs = metrics
        control.should_log = True
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self.pending_logs is not None:
            logs.update(self.pending_logs)
            self.pending_logs = None
        return control


def compute_save_steps(num_examples: int, args) -> int:
    if args.max_steps and args.max_steps > 0:
        return max(1, math.floor(args.max_steps / args.num_checkpoints))
    effective_batch = args.per_device_train_batch_size * max(1, args.gradient_accumulation_steps)
    total_steps = math.ceil(num_examples / effective_batch) * args.num_train_epochs
    return max(1, math.floor(total_steps / args.num_checkpoints))


def resolve_save_config(num_examples: int, args) -> tuple[str, int | None]:
    if args.num_checkpoints < 0:
        raise ValueError("--num-checkpoints must be greater than or equal to 0")
    if args.num_checkpoints == 0:
        return "no", None
    return "steps", args.save_steps or compute_save_steps(num_examples, args)


def require_deepspeed_if_requested(deepspeed_config: str | None) -> None:
    if not deepspeed_config:
        return
    try:
        version = importlib.metadata.version("deepspeed")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "DeepSpeed is requested via --deepspeed, but it is not installed in the "
            "current Python environment. Install it in this same environment first, "
            "for example: python -m pip install deepspeed==0.15.4"
        ) from exc
    print(f"Using DeepSpeed {version} with config: {deepspeed_config}")


def configure_single_process_distributed_env(deepspeed_config: str | None) -> None:
    if not deepspeed_config:
        return
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    print("Configured single-process distributed environment for DeepSpeed.")


def load_training_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return config


def normalize_argparse_defaults(config: dict) -> dict:
    defaults = dict(config)
    if isinstance(defaults.get("report_to"), list):
        defaults["report_to"] = ",".join(defaults["report_to"])
    return {key.replace("-", "_"): value for key, value in defaults.items()}


def parse_report_to(value) -> list[str] | str:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items or "none"
    cleaned = value.strip().lower()
    if cleaned in {"none", "null", "off", "disabled"}:
        return "none"
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_wandb(args) -> None:
    report_to = parse_report_to(args.report_to)
    if report_to == "none" or "wandb" not in report_to:
        os.environ.setdefault("WANDB_DISABLED", "true")
        return
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_WATCH", args.wandb_watch)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_run_name:
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)


def configure_reproducibility(seed: int, full_determinism: bool) -> None:
    set_seed(seed)
    if not full_determinism:
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="configs/training_defaults.json")
    config_args, remaining_args = config_parser.parse_known_args()
    config_defaults = normalize_argparse_defaults(load_training_config(config_args.config))

    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--model-path", default="models/Qwen3.5-2B")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", choices=["binary", "taxonomy", "combined"], required=True)
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--benchmark-path", default="data/datasets/atbench500/ATBench500/test.json")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int)
    parser.add_argument("--num-checkpoints", type=int, default=5)
    parser.add_argument("--save-total-limit", type=int, default=5)
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
    parser.add_argument(
        "--eval-atbench-during-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eval-limit", type=int)
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining_args)
    args.config = config_args.config
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

    run_config = vars(args) | {
        "resolved_save_strategy": save_strategy,
        "resolved_save_steps": save_steps,
        "num_examples": len(dataset),
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForCausalSft(tokenizer),
        callbacks=[
            AtBenchEvalCallback(
                enabled=args.eval_atbench_during_training,
                task=args.task,
                benchmark_path=args.benchmark_path,
                eval_steps=args.eval_steps,
                eval_limit=args.eval_limit,
                tokenizer=tokenizer,
                output_dir=args.output_dir,
            )
        ],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
