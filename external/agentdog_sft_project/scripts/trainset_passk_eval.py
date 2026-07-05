#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.binary_safety_eval import (  # noqa: E402
    JUDGMENT_LABELS,
    SOURCE_LABELS,
    encode_prompt_text,
    loose_parse_prediction,
    normalize_judgment,
    normalize_source,
    precision_recall_f1,
    resolve_model_path,
)


DEFAULT_TRAIN_FILE = "outputs/data/agentdog_6label_sft.jsonl"


def load_vllm():
    if importlib.util.find_spec("vllm") is None:
        raise SystemExit(
            "vLLM is required for train-set pass@k evaluation. Install it in the train environment first."
        )
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise SystemExit(
            "vLLM is installed but cannot load its inference API. This usually means its compiled "
            "dependencies do not match the current torch/CUDA environment. "
            f"Original error: {exc!r}"
        ) from exc
    return LLM, SamplingParams


def read_sft_rows(
    path: Path,
    limit: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[dict[str, Any]]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    rows = []
    all_rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Line {line_number} must contain at least two chat messages.")
            target = json.loads(messages[-1]["content"])
            expected_judgment = normalize_judgment(target.get("judgment"))
            expected_source = normalize_source(target.get("source"))
            if expected_judgment not in JUDGMENT_LABELS:
                raise ValueError(f"Line {line_number} has invalid target judgment: {target.get('judgment')!r}")
            if expected_source not in SOURCE_LABELS:
                raise ValueError(f"Line {line_number} has invalid target source: {target.get('source')!r}")
            all_rows.append(
                {
                    "index": len(all_rows),
                    "messages": messages,
                    "prompt_messages": messages[:-1],
                    "expected_judgment": expected_judgment,
                    "expected_source": expected_source,
                }
            )
            if limit is not None and len(all_rows) >= limit:
                break
    for row in all_rows:
        if row["index"] % num_shards == shard_index:
            rows.append(row)
    return rows


def prompt_text(tokenizer, prompt_messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def summarize_numbers(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": sum(values) / len(values) if values else 0.0,
        "max": max(values) if values else 0,
        "min": min(values) if values else 0,
        "median": statistics.median(values) if values else 0.0,
        "sum": sum(values),
    }


def passk_binary_metrics(records: list[dict[str, Any]], field: str, labels: tuple[str, ...]) -> dict[str, Any]:
    total = len(records)
    expected_key = f"expected_{field}"
    parsed_key = f"parsed_{field}"
    success_count_key = f"{field}_success_count"
    passed_key = f"{field}_passed"
    if field == "judgment":
        success_count_key = success_count_key if success_count_key in records[0] else "success_count"
        passed_key = passed_key if passed_key in records[0] else "passed"
    passed = sum(1 for row in records if row.get(passed_key, False))
    per_label = {}
    f1_values = []
    for label in labels:
        tp = sum(1 for row in records if row.get(expected_key) == label and row.get(passed_key, False))
        fn = sum(1 for row in records if row.get(expected_key) == label and not row.get(passed_key, False))
        fp = sum(
            1
            for row in records
            if row.get(expected_key) != label
            and not row.get(passed_key, False)
            and any(rollout[parsed_key] == label for rollout in row["rollouts"])
        )
        scores = precision_recall_f1(tp, fp, fn)
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn, **scores}
        f1_values.append(scores["f1"])
    success_counts = [row.get(success_count_key, 0) for row in records]
    return {
        "pass_at_k_accuracy": passed / total if total else 0.0,
        "pass_at_k_correct": passed,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "target_distribution": dict(Counter(row.get(expected_key) for row in records)),
        "success_count_distribution": dict(Counter(success_counts)),
        "success_count": summarize_numbers(success_counts),
        "valid_rollout_count": sum(
            1 for row in records for rollout in row["rollouts"] if rollout[parsed_key] is not None
        ),
        "per_label": per_label,
    }


def compute_metrics(records: list[dict[str, Any]], pass_k: int) -> dict[str, Any]:
    judgment = passk_binary_metrics(records, "judgment", JUDGMENT_LABELS)
    source = passk_binary_metrics(records, "source", SOURCE_LABELS)
    unsafe_scores = judgment["per_label"]["unsafe"]
    output_tokens = [r["output_tokens"] for row in records for r in row["rollouts"]]
    input_tokens = [row["input_tokens"] for row in records]
    return {
        "total": len(records),
        "pass_k": pass_k,
        "judgment": {
            **judgment,
            "pass_at_k_f1": unsafe_scores["f1"],
            "precision_unsafe": unsafe_scores["precision"],
            "recall_unsafe": unsafe_scores["recall"],
        },
        "source": source,
        "pass_at_k_accuracy": judgment["pass_at_k_accuracy"],
        "pass_at_k_correct": judgment["pass_at_k_correct"],
        "pass_at_k_f1": unsafe_scores["f1"],
        "precision_unsafe": unsafe_scores["precision"],
        "recall_unsafe": unsafe_scores["recall"],
        "target_distribution": judgment["target_distribution"],
        "success_count_distribution": judgment["success_count_distribution"],
        "success_count": judgment["success_count"],
        "valid_rollout_count": judgment["valid_rollout_count"],
        "token_cost": {
            "input_tokens": summarize_numbers(input_tokens),
            "output_tokens": summarize_numbers(output_tokens),
            "output_tokens_per_sample": summarize_numbers(
                [sum(r["output_tokens"] for r in row["rollouts"]) for row in records]
            ),
        },
    }


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def rollout_record(
    text: str,
    expected_judgment: str,
    expected_source: str,
    output_tokens: int,
    rollout_index: int,
) -> dict[str, Any]:
    parsed = loose_parse_prediction(text)
    parsed_judgment = parsed.get("judgment")
    parsed_source = parsed.get("source")
    return {
        "rollout_index": rollout_index,
        "raw_output": text.strip(),
        "parsed_judgment": parsed_judgment,
        "parsed_source": parsed_source,
        "success": parsed_judgment == expected_judgment,
        "judgment_success": parsed_judgment == expected_judgment,
        "source_success": parsed_source == expected_source,
        "output_tokens": output_tokens,
    }


def generate_with_hf(args, model_path: str, tokenizer, rows: list[dict[str, Any]]):
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
    records = []
    do_sample = args.pass_k > 1 or args.temperature > 0
    temperature = args.temperature if do_sample and args.temperature > 0 else 1.0
    for batch_rows in batched(rows, args.batch_size):
        prompt_texts = [prompt_text(tokenizer, row["prompt_messages"]) for row in batch_rows]
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
            for rollout_index, output in enumerate(row_outputs):
                output_ids = output[prompt_width:]
                raw_output = tokenizer.decode(output_ids, skip_special_tokens=True)
                rollouts.append(
                    rollout_record(
                        raw_output,
                        row["expected_judgment"],
                        row["expected_source"],
                        int(output_ids.shape[-1]),
                        rollout_index,
                    )
                )
            judgment_success_count = sum(1 for rollout in rollouts if rollout["judgment_success"])
            source_success_count = sum(1 for rollout in rollouts if rollout["source_success"])
            records.append(
                {
                    "index": row["index"],
                    "expected_judgment": row["expected_judgment"],
                    "expected_source": row["expected_source"],
                    "success_count": judgment_success_count,
                    "passed": judgment_success_count > 0,
                    "judgment_success_count": judgment_success_count,
                    "judgment_passed": judgment_success_count > 0,
                    "source_success_count": source_success_count,
                    "source_passed": source_success_count > 0,
                    "input_tokens": input_token_count,
                    "rollouts": rollouts,
                }
            )
        if len(records) % 25 == 0 or len(records) == len(rows):
            print(f"evaluated {len(records)}/{len(rows)}", flush=True)
    return records


def generate_with_vllm(args, model_path: str, tokenizer, rows: list[dict[str, Any]]):
    prompts = [prompt_text(tokenizer, row["prompt_messages"]) for row in rows]
    input_tokens = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in prompts]

    LLM, SamplingParams = load_vllm()
    sampling_params = SamplingParams(
        n=args.pass_k,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
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

    records = []
    for row, request_output, input_token_count in zip(rows, outputs, input_tokens):
        rollouts = []
        for rollout_index, output in enumerate(request_output.outputs):
            token_ids = getattr(output, "token_ids", None) or []
            rollouts.append(
                rollout_record(
                    output.text,
                    row["expected_judgment"],
                    row["expected_source"],
                    len(token_ids),
                    rollout_index,
                )
            )
        judgment_success_count = sum(1 for rollout in rollouts if rollout["judgment_success"])
        source_success_count = sum(1 for rollout in rollouts if rollout["source_success"])
        records.append(
            {
                "index": row["index"],
                "expected_judgment": row["expected_judgment"],
                "expected_source": row["expected_source"],
                "success_count": judgment_success_count,
                "passed": judgment_success_count > 0,
                "judgment_success_count": judgment_success_count,
                "judgment_passed": judgment_success_count > 0,
                "source_success_count": source_success_count,
                "source_passed": source_success_count > 0,
                "input_tokens": input_token_count,
                "rollouts": rollouts,
            }
        )
        if len(records) % 25 == 0 or len(records) == len(rows):
            print(f"evaluated {len(records)}/{len(rows)}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pass@k judgment accuracy on the 6-label SFT train set.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--pass-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
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
    model_path = resolve_model_path(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rows = read_sft_rows(Path(args.train_file), args.limit, args.num_shards, args.shard_index)

    if args.backend == "vllm":
        records = generate_with_vllm(args, model_path, tokenizer, rows)
    else:
        records = generate_with_hf(args, model_path, tokenizer, rows)

    metrics = compute_metrics(records, args.pass_k)
    metrics.update(
        {
            "model_path": args.model_path,
            "resolved_model_path": model_path,
            "train_file": args.train_file,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "backend": args.backend,
            "batch_size": args.batch_size,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "tensor_parallel_size": args.tensor_parallel_size,
        }
    )
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
