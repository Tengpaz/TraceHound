#!/usr/bin/env python
"""Evaluate AgentDoG-Lite binary local models on summer-camp ATBench/R-Judge."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.chat_format import apply_chat_template
from traceguard.lite_binary_eval import (
    SUMMER_CAMP_DATASET_ID,
    compute_binary_eval_metrics,
    dataset_to_eval_examples,
    load_eval_rows,
    parse_judgment_output,
    write_jsonl,
)


DEFAULT_MODEL = str(ROOT / "models" / "TraceHound-Base-Qwen3.5-0.8B-Binary")
DEFAULT_DATASET_ROOT = "external/agentdog_official/datasets/summer_camp_teseset"
DEFAULT_OUTPUT_DIR = "reports/tracehound_base_qwen3_5_0_8b_binary_eval"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default=os.getenv("TRACEHOUND_GUARD_MODEL_PATH") or os.getenv("TRACEHOUND_BASE_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--adapter", help="Optional LoRA adapter checkpoint directory.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--datasets", default="atbench,rjudge", help="Comma list: atbench,rjudge")
    parser.add_argument("--limit", type=int, help="Per-dataset smoke-test limit.")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    dataset_root = ensure_dataset_root(Path(args.dataset_root), args.download_dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_bundle = load_model(args)
    summary: dict[str, Any] = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "dataset_root": str(dataset_root),
        "datasets": {},
    }
    for dataset_name in _dataset_names(args.datasets):
        rows = load_eval_rows(dataset_root, dataset_name)
        examples = dataset_to_eval_examples(rows, dataset_name)
        if args.limit:
            examples = examples[: args.limit]
        predictions = evaluate_examples(model_bundle, examples, args)
        metrics = compute_binary_eval_metrics(predictions)
        pred_path = output_dir / f"{dataset_name}_predictions.jsonl"
        write_jsonl(pred_path, predictions)
        summary["datasets"][dataset_name] = {
            "metrics": metrics,
            "predictions": str(pred_path),
        }
        print(f"[tracehound] {dataset_name} metrics")
        print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[tracehound] summary: {summary_path}")


def ensure_dataset_root(root: Path, download: bool) -> Path:
    if (root / "summer_camp_ATBench300.json").exists() and (root / "summer_camp_rjudge.json").exists():
        return root
    if not download:
        raise FileNotFoundError(
            f"missing summer-camp dataset files under {root}; rerun with --download-dataset"
        )
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required for --download-dataset; install `pip install -e .[official]`.") from exc
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=SUMMER_CAMP_DATASET_ID,
        repo_type="dataset",
        local_dir=str(root),
        local_dir_use_symlinks=False,
    )
    return root


def load_model(args: argparse.Namespace) -> dict[str, Any]:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not visible. Use --allow-cpu only for tiny debugging.")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto",
        device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel  # type: ignore

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "torch": torch}


def evaluate_examples(model_bundle: dict[str, Any], examples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch  # type: ignore

    model = model_bundle["model"]
    tokenizer = model_bundle["tokenizer"]
    predictions = []
    do_sample = args.temperature and args.temperature > 0
    for index, example in enumerate(examples, start=1):
        messages = [{"role": "user", "content": example["prompt"]}]
        prompt_text = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_input_tokens,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        input_len = int(encoded["input_ids"].shape[-1])
        generate_kwargs = {
            **encoded,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": bool(do_sample),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = args.temperature
        with torch.no_grad():
            output_ids = model.generate(**generate_kwargs)
        generated_ids = output_ids[0][input_len:]
        raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        prediction, invalid = parse_judgment_output(raw_output)
        predictions.append(
            {
                "id": example["id"],
                "dataset": example["dataset"],
                "gold": example["gold"],
                "prediction": prediction,
                "invalid": invalid,
                "raw_output": raw_output,
                "output_tokens": int(generated_ids.numel()),
                "input_tokens": input_len,
            }
        )
        if index % 25 == 0 or index == len(examples):
            print(f"[tracehound] evaluated {index}/{len(examples)} for {example['dataset']}", flush=True)
    return predictions


def _dataset_names(value: str) -> list[str]:
    mapping = {"atbench": "atbench", "atbench300": "atbench", "rjudge": "rjudge", "r_judge": "rjudge"}
    names = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in mapping:
            raise ValueError(f"unknown dataset: {item}")
        names.append(mapping[key])
    return names or ["atbench", "rjudge"]


if __name__ == "__main__":
    main()
