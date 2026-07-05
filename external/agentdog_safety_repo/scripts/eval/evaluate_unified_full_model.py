#!/usr/bin/env python
"""Evaluate a unified four-label full checkpoint on ATBench/R-Judge.

This is a self-contained companion to `evaluate_unified_lora_model.py`.
It reuses the same prompt, parsing, dataset loading, and metrics, but loads a
full causal-LM checkpoint instead of a PEFT adapter.
"""

from __future__ import annotations

import argparse
from typing import Any

from evaluate_unified_lora_model import dataset_names, evaluate_examples, load_rows, build_example, compute_metrics, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", default="atbench,rjudge")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    from pathlib import Path
    import json

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_full_model(args)

    summary: dict[str, Any] = {
        "model": args.model,
        "dataset_root": args.dataset_root,
        "datasets": {},
    }
    all_predictions: list[dict[str, Any]] = []
    for dataset_name in dataset_names(args.datasets):
        rows = load_rows(Path(args.dataset_root), dataset_name)
        examples = [build_example(row, dataset_name) for row in rows]
        if args.limit:
            examples = examples[: args.limit]
        predictions = evaluate_examples(bundle, examples, args)
        include_taxonomy = dataset_name == "atbench"
        metrics = compute_metrics(predictions, include_taxonomy=include_taxonomy)
        pred_path = output_dir / f"{dataset_name}_predictions.jsonl"
        write_jsonl(pred_path, predictions)
        summary["datasets"][dataset_name] = {
            "metrics": metrics,
            "predictions": str(pred_path),
            "taxonomy_metrics_enabled": include_taxonomy,
        }
        all_predictions.extend(predictions)
        print(f"[tracehound] {dataset_name} metrics")
        print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), flush=True)

    summary["combined_binary"] = compute_metrics(all_predictions, include_taxonomy=False)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[tracehound] summary: {summary_path}", flush=True)


def load_full_model(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("CUDA is not visible. Use --allow-cpu only for tiny debugging.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "torch": torch}


if __name__ == "__main__":
    main()
