#!/usr/bin/env python
"""Preference/RL fine-tuning entrypoint for GPU contest environments."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.model_profiles import profile_model_id, resolve_model_profile


TRAIN_PACKAGES = ("torch", "transformers", "datasets", "peft", "trl", "accelerate")


def missing_packages() -> list[str]:
    return [package for package in TRAIN_PACKAGES if importlib.util.find_spec(package) is None]


def dataset_stats(path: Path) -> Dict[str, Any]:
    rows = 0
    complete_pairs = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            item = json.loads(line)
            if item.get("prompt") and item.get("chosen") and item.get("rejected"):
                complete_pairs += 1
    return {"samples": rows, "complete_pairs": complete_pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_preference.jsonl")
    parser.add_argument("--base-model", help="SFT checkpoint path or HF id override.")
    parser.add_argument("--model-profile", default=os.getenv("TRACEHOUND_MODEL_PROFILE", "internlm3-8b-instruct"))
    parser.add_argument("--profile-path", help="Optional model profile JSON path.")
    parser.add_argument("--output-dir", default="checkpoints/preference")
    parser.add_argument("--algorithm", default="dpo", choices=["dpo", "orpo", "grpo"])
    parser.add_argument("--max-samples", type=int, help="Optional smoke-test sample limit.")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-prompt-length", type=int, default=3072)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta.")
    parser.add_argument("--lora-r", type=int, help="Override profile LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, help="Override profile LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, help="Override profile LoRA dropout.")
    parser.add_argument("--target-modules", help="Comma-separated LoRA target modules override.")
    parser.add_argument("--run", action="store_true", help="Actually launch DPO/ORPO where supported.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow --run without visible CUDA for tiny debugging only.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if training dependencies are missing.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(
            f"missing preference data: {data_path}. Run `python scripts/generate_data.py --out data` first."
        )

    profile = resolve_model_profile(args.model_profile, args.profile_path)
    if profile.get("provider") != "huggingface":
        raise SystemExit(f"model profile {profile['name']} is not a Hugging Face local training profile")
    base_model = args.base_model or os.getenv("TRACEHOUND_LOCAL_MODEL_PATH") or profile_model_id(profile)
    plan = build_training_plan(args, profile, base_model, data_path)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
    write_plan(Path(args.output_dir), plan)

    missing = missing_packages()
    if missing:
        message = (
            "Preference-training dependencies are missing. This is expected on the Mac CPU environment.\n"
            f"Missing packages: {', '.join(missing)}\n"
            "On Linux/GPU, install CUDA-matched PyTorch first, then run `pip install -e \".[train]\"`."
        )
        if args.strict or args.run:
            raise SystemExit(message)
        print(message)
        return

    if not args.run:
        print("Dependencies are available. Add --run on the GPU server to launch the selected trainer.")
        return

    if args.algorithm == "grpo":
        raise SystemExit(
            "GRPO needs a contest-specific reward function. The exported RL data and plan are ready; "
            "wire the reward function after the official scoring interface is known."
        )
    run_preference_training(args, profile, base_model, data_path)


def build_training_plan(args: argparse.Namespace, profile: Dict[str, Any], base_model: str, data_path: Path) -> Dict[str, Any]:
    lora = dict(profile.get("lora") or {})
    target_modules = (
        [item.strip() for item in args.target_modules.split(",") if item.strip()]
        if args.target_modules
        else list(lora.get("target_modules") or [])
    )
    return {
        "task": "preference",
        "algorithm": args.algorithm,
        "data": str(data_path),
        "output_dir": args.output_dir,
        "base_model": base_model,
        "model_profile": {
            "name": profile["name"],
            "role": profile.get("role"),
            "recommended_use": profile.get("recommended_use"),
            "smoke": bool(profile.get("smoke", False)),
            "formal_lora": bool(profile.get("formal_lora", False)),
        },
        "prompt_policy": {
            "primary": "tokenizer.apply_chat_template for SFT/local inference",
            "preference_format": "prompt/chosen/rejected JSONL",
        },
        "sequence": {
            "max_seq_length": args.max_seq_length,
            "max_prompt_length": args.max_prompt_length,
            "recommended_max_input_tokens": profile.get("recommended_max_input_tokens"),
            "cost_note": "Preference tuning should not replace runtime compression and rule early exit.",
        },
        "lora": {
            "r": args.lora_r or lora.get("r", 16),
            "alpha": args.lora_alpha or lora.get("alpha", 32),
            "dropout": args.lora_dropout if args.lora_dropout is not None else lora.get("dropout", 0.05),
            "target_modules": target_modules,
        },
        "trainer": {
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "num_train_epochs": args.num_train_epochs,
            "beta": args.beta,
            "max_samples": args.max_samples,
            "run_requested": bool(args.run),
            "grpo_reward_hook": "Define reward_fn(prompt, completion, reference) after contest scoring is known.",
        },
        "stats": dataset_stats(data_path),
    }


def write_plan(output_dir: Path, plan: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preference_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_preference_training(args: argparse.Namespace, profile: Dict[str, Any], base_model: str, data_path: Path) -> None:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from peft import LoraConfig  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments  # type: ignore

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not visible. Use --allow-cpu only for tiny debugging runs.")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=bool(profile.get("trust_remote_code", True)))
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _resolve_torch_dtype(torch, str(profile.get("torch_dtype") or "auto"))
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=bool(profile.get("trust_remote_code", True)),
        device_map=str(profile.get("device_map") or "auto"),
        torch_dtype=dtype,
    )
    rows = list(_read_preference_rows(data_path, args.max_samples))
    dataset = Dataset.from_list(rows)
    lora = build_training_plan(args, profile, base_model, data_path)["lora"]
    peft_config = LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["dropout"]),
        target_modules=list(lora["target_modules"]),
        task_type="CAUSAL_LM",
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )
    trainer = _build_trl_trainer(
        algorithm=args.algorithm,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        training_args=training_args,
        peft_config=peft_config,
        beta=args.beta,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_seq_length,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def _build_trl_trainer(**kwargs: Any) -> Any:
    algorithm = kwargs.pop("algorithm")
    if algorithm == "dpo":
        from trl import DPOTrainer  # type: ignore

        return _instantiate_preference_trainer(DPOTrainer, include_beta=True, **kwargs)
    if algorithm == "orpo":
        from trl import ORPOTrainer  # type: ignore

        return _instantiate_preference_trainer(ORPOTrainer, include_beta=False, **kwargs)
    raise ValueError(f"unsupported trainer algorithm: {algorithm}")


def _instantiate_preference_trainer(trainer_cls: Any, include_beta: bool, **kwargs: Any) -> Any:
    common = {
        "model": kwargs["model"],
        "args": kwargs["training_args"],
        "train_dataset": kwargs["train_dataset"],
        "peft_config": kwargs["peft_config"],
        "max_prompt_length": kwargs["max_prompt_length"],
        "max_length": kwargs["max_length"],
    }
    if include_beta:
        common["beta"] = kwargs["beta"]
    try:
        return trainer_cls(tokenizer=kwargs["tokenizer"], **common)
    except TypeError:
        return trainer_cls(processing_class=kwargs["tokenizer"], **common)


def _read_preference_rows(path: Path, max_samples: int | None) -> Iterable[Dict[str, Any]]:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("prompt") and item.get("chosen") and item.get("rejected"):
                yield {"prompt": item["prompt"], "chosen": item["chosen"], "rejected": item["rejected"]}
                count += 1
                if max_samples and count >= max_samples:
                    break


def _resolve_torch_dtype(torch: Any, dtype_name: str) -> Any:
    normalized = dtype_name.lower()
    if normalized in {"auto", ""}:
        return "auto"
    aliases = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16", "float16": "float16"}
    attr = aliases.get(normalized, normalized)
    return getattr(torch, attr)


if __name__ == "__main__":
    main()
