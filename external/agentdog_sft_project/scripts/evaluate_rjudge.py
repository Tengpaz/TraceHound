#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.binary_safety_eval import (  # noqa: E402
    TARGET_FIELDS,
    binary_metrics,
    encode_prompt_text,
    load_prompt_template,
    loose_parse_prediction,
    pass_at_k_binary_metrics,
    resolve_model_path,
    rollout_record,
    strict_parse_prediction,
)


DEFAULT_INPUT_JSON = "2026_summer_camp_teseset/summer_camp_rjudge.json"
DEFAULT_PROMPT_TEMPLATE = "src/guardrail/6 labels training prompt.md"


def serialize_for_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_rjudge_trajectory(row: dict[str, Any]) -> str:
    profile = row.get("profile") or "You are a helpful assistant."
    tools = json.dumps(row.get("tool_used", []), ensure_ascii=False)
    return (
        "=== Agent Profile ===\n"
        f"{str(profile).strip()}\n"
        f"Available tools: {tools}\n\n"
        "=== Conversation History ===\n"
        f"{serialize_for_prompt(row['contents']).strip()}"
    )


def make_prompt(row: dict[str, Any], template: str) -> str:
    return template.replace("{trajectory}", render_rjudge_trajectory(row))


def load_rows(args) -> list[dict[str, Any]]:
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {args.input_json}")

    rows = []
    required = {"id", "contents", "label"}
    for idx, row in enumerate(data):
        if args.limit is not None and len(rows) >= args.limit:
            break
        missing = required - set(row.keys())
        if missing:
            raise KeyError(f"Row {idx} missing required columns: {sorted(missing)}")
        item = dict(row)
        item["_eval_index"] = idx
        rows.append(item)

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    if args.num_shards > 1:
        rows = [row for row in rows if row["_eval_index"] % args.num_shards == args.shard_index]
    return rows


def expected_from_row(row: dict[str, Any]) -> dict[str, str | None]:
    label = str(row.get("label"))
    if label in {"0", "safe"}:
        judgment = "safe"
    elif label in {"1", "unsafe"}:
        judgment = "unsafe"
    else:
        judgment = None
    return {"judgment": judgment}


def rjudge_rollout_record(raw_output: str, output_tokens: int) -> dict[str, Any]:
    strict_parsed, valid_exact = strict_parse_prediction(raw_output)
    parsed = strict_parsed if valid_exact else loose_parse_prediction(raw_output)
    return {
        "raw_output": raw_output,
        "parsed": parsed,
        "valid_exact_json": valid_exact,
        "contains_think_tag": "<think>" in raw_output or "</think>" in raw_output,
        "output_tokens": output_tokens,
    }


def prediction_record(row: dict[str, Any], input_tokens: int, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    first = rollouts[0] if rollouts else rjudge_rollout_record("", 0)
    return {
        "eval_index": row.get("_eval_index"),
        "id": row["id"],
        "label": row["label"],
        "risk_type": row.get("risk_type"),
        "risk_description": row.get("risk_description"),
        "expected": expected_from_row(row),
        "raw_output": first["raw_output"],
        "parsed": first["parsed"],
        "valid_exact_json": first["valid_exact_json"],
        "contains_think_tag": first["contains_think_tag"],
        "input_tokens": input_tokens,
        "output_tokens": first["output_tokens"],
        "rollouts": rollouts,
    }


def summarize_numbers(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "max": max(values) if values else 0,
        "min": min(values) if values else 0,
        "median": statistics.median(values) if values else 0.0,
        "sum": sum(values),
    }


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    output_tokens = [rollout["output_tokens"] for row in predictions for rollout in row.get("rollouts", [])]
    input_tokens = [row["input_tokens"] for row in predictions]
    strict_valid = sum(1 for row in predictions if row.get("valid_exact_json"))
    parsed_valid = sum(1 for row in predictions if row["parsed"].get("judgment") is not None)
    return {
        "total": total,
        "valid_exact_json": strict_valid,
        "valid_exact_json_rate": strict_valid / total if total else 0.0,
        "valid_parsed_outputs": parsed_valid,
        "valid_parsed_rate": parsed_valid / total if total else 0.0,
        "think_tag_outputs": sum(1 for row in predictions if row["contains_think_tag"]),
        "think_tag_rate": (
            sum(1 for row in predictions if row["contains_think_tag"]) / total if total else 0.0
        ),
        "binary_judgment": binary_metrics(predictions),
        "pass_at_k_binary_judgment": pass_at_k_binary_metrics(predictions),
        "token_cost": {
            "input_tokens": summarize_numbers(input_tokens),
            "output_tokens": summarize_numbers(output_tokens),
        },
    }


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_with_hf(args, model_path: str, tokenizer, rows: list[dict[str, Any]], template: str):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    predictions = []
    do_sample = args.pass_k > 1 or args.temperature > 0
    temperature = args.temperature if do_sample and args.temperature > 0 else 1.0
    for batch_rows in batched(rows, args.batch_size):
        prompt_texts = [encode_prompt_text(tokenizer, make_prompt(row, template)) for row in batch_rows]
        encoded = tokenizer(
            prompt_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        input_tokens = [int(mask.sum().item()) for mask in encoded["attention_mask"]]
        prompt_width = int(encoded["input_ids"].shape[-1])
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=args.top_p,
                num_return_sequences=args.pass_k,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = generated.reshape(len(batch_rows), args.pass_k, generated.shape[-1])
        for row, row_outputs, input_token_count in zip(batch_rows, generated, input_tokens):
            rollouts = []
            for output in row_outputs:
                output_ids = output[prompt_width:]
                raw_output = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                rollouts.append(rjudge_rollout_record(raw_output, int(output_ids.shape[-1])))
            predictions.append(prediction_record(row, input_token_count, rollouts))
        if len(predictions) % 25 == 0 or len(predictions) == len(rows):
            print(f"evaluated {len(predictions)}/{len(rows)}", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate RJudge with the same six-label prompt used by ATBench evaluation."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.pass_k < 1:
        raise ValueError("--pass-k must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = load_prompt_template(args.prompt_template)
    rows = load_rows(args)
    model_path = resolve_model_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    predictions = generate_with_hf(args, model_path, tokenizer, rows, template)

    metrics = compute_metrics(predictions)
    metrics.update(
        {
            "model_path": args.model_path,
            "resolved_model_path": model_path,
            "input_json": args.input_json,
            "prompt_template": args.prompt_template,
            "prompt_version": "six_label_training_prompt",
            "max_new_tokens": args.max_new_tokens,
            "backend": "hf",
            "pass_k": args.pass_k,
            "batch_size": args.batch_size,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "evaluated_fields": ["judgment"],
        }
    )

    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in predictions) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "prompt_template.txt").write_text(template, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
