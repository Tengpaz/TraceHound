#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail.metrics import binary_accuracy, taxonomy_metrics
from guardrail.prompts import (
    binary_target,
    build_binary_prompt,
    build_taxonomy_prompt,
    chat_messages,
    taxonomy_target,
)
from guardrail.taxonomy import parse_binary, parse_taxonomy, normalize_taxonomy_value


def load_json(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def load_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def generate(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    text = tokenizer.apply_chat_template(
        chat_messages(prompt),
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model_device(model))
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_binary(data, tokenizer, model, limit: int | None) -> tuple[list[dict], dict]:
    records = []
    for example in data[:limit]:
        raw = generate(tokenizer, model, build_binary_prompt(example), max_new_tokens=8)
        records.append(
            {
                "conv_id": example.get("conv_id"),
                "target": binary_target(example),
                "raw_prediction": raw,
                "prediction": parse_binary(raw),
            }
        )
    return records, binary_accuracy(records)


def evaluate_taxonomy(data, tokenizer, model, limit: int | None) -> tuple[list[dict], dict]:
    unsafe = [x for x in data if int(x["label"]) == 1]
    records = []
    for example in unsafe[:limit]:
        raw = generate(tokenizer, model, build_taxonomy_prompt(example), max_new_tokens=96)
        target = {
            "risk_source": normalize_taxonomy_value("risk_source", example["risk_source"]),
            "failure_mode": normalize_taxonomy_value("failure_mode", example["failure_mode"]),
            "real_world_harm": normalize_taxonomy_value("real_world_harm", example["real_world_harm"]),
        }
        records.append(
            {
                "conv_id": example.get("conv_id"),
                "target": target,
                "raw_target": taxonomy_target(example),
                "raw_prediction": raw,
                "prediction": parse_taxonomy(raw),
            }
        )
    return records, taxonomy_metrics(records)


def write_outputs(output_dir: Path, records: list[dict], metrics: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def run_eval(args) -> dict:
    data = load_json(args.benchmark_path)
    tokenizer, model = load_model(args.model_path)
    if args.task == "binary":
        records, metrics = evaluate_binary(data, tokenizer, model, args.limit)
    elif args.task == "taxonomy":
        records, metrics = evaluate_taxonomy(data, tokenizer, model, args.limit)
    else:
        binary_records, binary_metrics = evaluate_binary(data, tokenizer, model, args.limit)
        taxonomy_records, taxonomy_metrics_result = evaluate_taxonomy(data, tokenizer, model, args.limit)
        output_dir = Path(args.output_dir)
        write_outputs(output_dir / "binary", binary_records, binary_metrics)
        write_outputs(output_dir / "taxonomy", taxonomy_records, taxonomy_metrics_result)
        metrics = {"binary": binary_metrics, "taxonomy": taxonomy_metrics_result}
        with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return metrics
    write_outputs(Path(args.output_dir), records, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", choices=["binary", "taxonomy", "combined"], required=True)
    parser.add_argument("--benchmark-path", default="data/datasets/atbench500/ATBench500/test.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
