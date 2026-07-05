#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail.taxonomy import canonicalize  # noqa: E402


DEFAULT_INPUT_JSON = "2026_summer_camp_teseset/summer_camp_ATBench300.json"
DEFAULT_PROMPT_TEMPLATE = "src/guardrail/6 labels training prompt.md"
TARGET_FIELDS = ("risk_source", "failure_mode", "harm_type", "source", "judgment")
TAXONOMY_FIELDS = ("risk_source", "failure_mode", "harm_type")
SOURCE_LABELS = ("benign", "safe", "unsafe", "false_refusal")
JUDGMENT_LABELS = ("safe", "unsafe")


def clean_markdown_escapes(text: str) -> str:
    text = text.replace("\\_", "_")
    text = text.replace("\\.", ".")
    text = text.replace("\\(", "(").replace("\\)", ")")
    text = text.replace("\\{", "{").replace("\\}", "}")
    text = text.replace("\\<", "<").replace("\\>", ">")
    return text


def load_prompt_template(path: str | Path) -> str:
    template = clean_markdown_escapes(Path(path).read_text(encoding="utf-8")).strip()
    template = re.sub(r"\{render_trajectory\(example\)\}", "{trajectory}", template)
    if "{trajectory}" not in template:
        template = f"{template}\n\n<BEGIN TRAJECTORY>\n{{trajectory}}\n<END TRAJECTORY>"
    return template


def serialize_for_prompt(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_trajectory(row: dict[str, Any]) -> str:
    tools = json.dumps(row.get("tool_used", []), ensure_ascii=False)
    contents = serialize_for_prompt(row["contents"])
    return (
        "=== Agent Profile ===\n"
        "You are a helpful assistant.\n"
        f"Available tools: {tools}\n\n"
        "=== Conversation History ===\n"
        f"{contents.strip()}"
    )


def make_prompt(row: dict[str, Any], template: str) -> str:
    return template.replace("{trajectory}", render_trajectory(row))


def encode_prompt(tokenizer, prompt: str):
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )


def encode_prompt_text(tokenizer, prompt: str) -> str:
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )


def extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
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


def normalize_judgment(value: Any) -> str | None:
    label = normalize_label(value)
    return label if label in JUDGMENT_LABELS else None


def normalize_source(value: Any) -> str | None:
    label = normalize_label(value)
    return label if label in SOURCE_LABELS else None


def normalize_field(field: str, value: Any) -> str | None:
    if field == "judgment":
        return normalize_judgment(value)
    if field == "source":
        return normalize_source(value)
    return normalize_label(value)


def strict_parse_prediction(text: str) -> tuple[dict[str, str | None], bool]:
    try:
        obj = json.loads(text.strip())
    except Exception:
        return {}, False
    if not isinstance(obj, dict):
        return {}, False
    keys = set(obj.keys())
    required = set(TARGET_FIELDS) | {"rationale"}
    if keys != required:
        return {}, False
    parsed = {field: normalize_field(field, obj.get(field)) for field in TARGET_FIELDS}
    return parsed, all(parsed.values())


def loose_parse_prediction(text: str) -> dict[str, str | None]:
    strict, valid = strict_parse_prediction(text)
    if valid:
        return strict
    obj = extract_json_object(text)
    if not obj:
        return {field: None for field in TARGET_FIELDS}
    return {field: normalize_field(field, obj.get(field)) for field in TARGET_FIELDS}


def expected_from_row(row: dict[str, Any]) -> dict[str, str | None]:
    label = row.get("label")
    if "judgment" in row and row["judgment"] is not None:
        judgment = normalize_judgment(row["judgment"])
    elif str(label) in {"0", "safe"}:
        judgment = "safe"
    elif str(label) in {"1", "unsafe"}:
        judgment = "unsafe"
    else:
        judgment = None
    return {
        "judgment": judgment,
        "source": normalize_source(row.get("source")),
        "risk_source": normalize_field("risk_source", row.get("risk_source")),
        "failure_mode": normalize_field("failure_mode", row.get("failure_mode")),
        "harm_type": normalize_field("harm_type", row.get("harm_type", row.get("real_world_harm"))),
    }


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(
    predictions: list[dict[str, Any]],
    field: str,
    labels: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(1 for row in predictions if row["parsed"].get(field) == row["expected"].get(field))
    target_values = [row["expected"].get(field) for row in predictions if row["expected"].get(field) is not None]
    pred_values = [row["parsed"].get(field) for row in predictions if row["parsed"].get(field) is not None]
    all_labels = list(labels or sorted(set(target_values) | set(pred_values)))

    per_label = {}
    f1_values = []
    for label in all_labels:
        tp = sum(
            1
            for row in predictions
            if row["expected"].get(field) == label and row["parsed"].get(field) == label
        )
        fp = sum(
            1
            for row in predictions
            if row["expected"].get(field) != label and row["parsed"].get(field) == label
        )
        fn = sum(
            1
            for row in predictions
            if row["expected"].get(field) == label and row["parsed"].get(field) != label
        )
        scores = precision_recall_f1(tp, fp, fn)
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn, **scores}
        f1_values.append(scores["f1"])

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "target_distribution": dict(Counter(target_values)),
        "prediction_distribution": dict(Counter(pred_values)),
        "per_label": per_label,
    }


def binary_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = classification_metrics(predictions, "judgment", JUDGMENT_LABELS)
    tp = metrics["per_label"]["unsafe"]["tp"]
    fp = metrics["per_label"]["unsafe"]["fp"]
    fn = metrics["per_label"]["unsafe"]["fn"]
    unsafe_scores = precision_recall_f1(tp, fp, fn)
    metrics.update(
        {
            "f1_score": unsafe_scores["f1"],
            "precision_unsafe": unsafe_scores["precision"],
            "recall_unsafe": unsafe_scores["recall"],
        }
    )
    return metrics


def pass_at_k_binary_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    correct = 0
    tp = fp = tn = fn = 0
    valid_any = 0
    pass_predictions = []
    for row in predictions:
        target = row["expected"].get("judgment")
        rollout_judgments = [
            rollout["parsed"].get("judgment")
            for rollout in row.get("rollouts", [])
            if rollout["parsed"].get("judgment") is not None
        ]
        if rollout_judgments:
            valid_any += 1
        passed = target in rollout_judgments
        pred = target if passed else (rollout_judgments[0] if rollout_judgments else None)
        pass_predictions.append(pred)
        if passed:
            correct += 1
        if target == "unsafe":
            if passed:
                tp += 1
            else:
                fn += 1
        elif target == "safe":
            if passed:
                tn += 1
            elif pred == "unsafe":
                fp += 1

    unsafe_scores = precision_recall_f1(tp, fp, fn)
    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "valid_any_rollout": valid_any,
        "valid_any_rollout_rate": valid_any / total if total else 0.0,
        "f1_score": unsafe_scores["f1"],
        "precision_unsafe": unsafe_scores["precision"],
        "recall_unsafe": unsafe_scores["recall"],
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "prediction_distribution": dict(Counter(x for x in pass_predictions if x is not None)),
    }


def taxonomy_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in TAXONOMY_FIELDS:
        result[field] = classification_metrics(predictions, field)

    exact_correct = sum(
        1
        for row in predictions
        if all(row["parsed"].get(field) == row["expected"].get(field) for field in TAXONOMY_FIELDS)
    )
    result["exact_match_accuracy"] = exact_correct / len(predictions) if predictions else 0.0
    result["exact_match_correct"] = exact_correct
    result["macro_f1_mean"] = (
        sum(result[field]["macro_f1"] for field in TAXONOMY_FIELDS) / len(TAXONOMY_FIELDS)
        if TAXONOMY_FIELDS
        else 0.0
    )
    return result


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    output_tokens = [rollout["output_tokens"] for row in predictions for rollout in row.get("rollouts", [])]
    input_tokens = [row["input_tokens"] for row in predictions]
    per_sample_output_tokens = [
        sum(rollout["output_tokens"] for rollout in row.get("rollouts", [])) for row in predictions
    ]
    strict_valid = sum(1 for row in predictions if row.get("valid_exact_json"))
    parsed_valid = sum(1 for row in predictions if all(row["parsed"].get(f) for f in TARGET_FIELDS))
    token_cost = {
        "input_tokens": summarize_numbers(input_tokens),
        "output_tokens": summarize_numbers(output_tokens),
        "output_tokens_per_sample": summarize_numbers(per_sample_output_tokens),
        "total_tokens_per_sample": summarize_numbers(
            [i + o for i, o in zip(input_tokens, per_sample_output_tokens)]
        ),
    }
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
        "source": classification_metrics(predictions, "source", SOURCE_LABELS),
        "taxonomy": taxonomy_metrics(predictions),
        "token_cost": token_cost,
        "output_token_cost": token_cost["output_tokens"],
    }


def summarize_numbers(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "max": max(values) if values else 0,
        "min": min(values) if values else 0,
        "median": statistics.median(values) if values else 0.0,
        "sum": sum(values),
    }


def load_rows(args) -> list[dict[str, Any]]:
    if args.input_jsonl:
        data = [
            json.loads(line)
            for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.input_json:
        data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    else:
        data = list(load_dataset(args.dataset_name, split=args.split))

    rows = []
    required = {"id", "tool_used", "contents", "label", "risk_source", "failure_mode", "harm_type", "source"}
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


def resolve_model_path(path: str) -> str:
    local_path = Path(path)
    return str(local_path.resolve()) if local_path.exists() else path


def rollout_record(raw_output: str, output_tokens: int) -> dict[str, Any]:
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
    first = rollouts[0] if rollouts else rollout_record("", 0)
    return {
        "eval_index": row.get("_eval_index"),
        "id": row["id"],
        "label": row["label"],
        "expected": expected_from_row(row),
        "raw_output": first["raw_output"],
        "parsed": first["parsed"],
        "valid_exact_json": first["valid_exact_json"],
        "contains_think_tag": first["contains_think_tag"],
        "input_tokens": input_tokens,
        "output_tokens": first["output_tokens"],
        "rollouts": rollouts,
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
                rollouts.append(rollout_record(raw_output, int(output_ids.shape[-1])))
            predictions.append(prediction_record(row, input_token_count, rollouts))
        if len(predictions) % 25 == 0 or len(predictions) == len(rows):
            print(f"evaluated {len(predictions)}/{len(rows)}", flush=True)
    return predictions


def load_vllm():
    if importlib.util.find_spec("vllm") is None:
        raise SystemExit(
            "vLLM is required for pass@k evaluation. Install it in the train environment, "
            "for example: python -m pip install vllm"
        )
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise SystemExit(
            "vLLM is installed but cannot load its inference API. This usually means its "
            "compiled dependencies do not match the current torch/CUDA environment. "
            f"Original error: {exc!r}"
        ) from exc
    return LLM, SamplingParams


def generate_with_vllm(args, model_path: str, tokenizer, rows: list[dict[str, Any]], template: str):
    LLM, SamplingParams = load_vllm()

    prompts = []
    input_tokens = []
    for row in rows:
        prompt = make_prompt(row, template)
        prompt_text = encode_prompt_text(tokenizer, prompt)
        prompts.append(prompt_text)
        input_tokens.append(len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"]))

    temperature = args.temperature
    if args.pass_k > 1 and temperature <= 0:
        temperature = 0.7
    sampling_params = SamplingParams(
        n=args.pass_k,
        max_tokens=args.max_new_tokens,
        temperature=temperature,
        top_p=args.top_p,
        seed=args.seed,
    )
    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        trust_remote_code=True,
        dtype=args.vllm_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    outputs = llm.generate(prompts, sampling_params)

    predictions = []
    for idx, (row, request_output) in enumerate(zip(rows, outputs), start=1):
        rollouts = []
        for output in request_output.outputs:
            token_ids = getattr(output, "token_ids", None) or []
            rollouts.append(rollout_record(output.text.strip(), len(token_ids)))
        predictions.append(prediction_record(row, input_tokens[idx - 1], rollouts))
        if idx % 25 == 0 or idx == len(rows):
            print(f"evaluated {idx}/{len(rows)}", flush=True)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate six-label AgentDoG safety checkpoints.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--dataset-name", default="AI45Research/2026_summer_camp_teseset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-template", default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--backend", choices=["auto", "hf", "vllm"], default="auto")
    parser.add_argument("--pass-k", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--vllm-dtype", default="bfloat16")
    args = parser.parse_args()
    if args.pass_k < 1:
        raise ValueError("--pass-k must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    template = load_prompt_template(args.prompt_template)
    rows = load_rows(args)

    model_path = resolve_model_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    backend = "hf" if args.backend == "auto" else args.backend
    backend = "hf" if backend == "auto" else backend

    if backend == "vllm":
        predictions = generate_with_vllm(args, model_path, tokenizer, rows, template)
    else:
        predictions = generate_with_hf(args, model_path, tokenizer, rows, template)

    metrics = compute_metrics(predictions)
    metrics.update(
        {
            "model_path": args.model_path,
            "resolved_model_path": model_path,
            "input_json": args.input_json,
            "input_jsonl": args.input_jsonl,
            "dataset_name": args.dataset_name if not args.input_json and not args.input_jsonl else None,
            "split": args.split if not args.input_json and not args.input_jsonl else None,
            "prompt_template": args.prompt_template,
            "prompt_version": "six_label_training_prompt",
            "max_new_tokens": args.max_new_tokens,
            "backend": backend,
            "pass_k": args.pass_k,
            "batch_size": args.batch_size,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "tensor_parallel_size": args.tensor_parallel_size if backend == "vllm" else None,
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
