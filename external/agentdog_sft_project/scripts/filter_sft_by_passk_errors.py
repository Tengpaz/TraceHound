#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_FILE = "outputs/data/agentdog_6label_sft.jsonl"
DEFAULT_FIELD_PREDICTIONS = (
    "outputs/eval/qwen35-0.8b-6label-train-pass8-hf/merged/field_passk/field_predictions.jsonl"
)
DEFAULT_OUTPUT = "outputs/data/agentdog_6label_sft_pass8_error_filtered.jsonl"


RULES = {
    "source": 2,
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def infer_pass_k(row: dict[str, Any]) -> int:
    rollouts = row.get("rollouts", [])
    if not rollouts:
        raise ValueError("Cannot infer pass@k because a field prediction row has no rollouts.")
    return len(rollouts)


def success_count(row: dict[str, Any], field: str) -> int:
    key = f"{field}_success_count"
    if key in row:
        return int(row[key])
    expected = row["expected"][field]
    return sum(1 for rollout in row.get("rollouts", []) if rollout.get("parsed", {}).get(field) == expected)


def failure_reasons(row: dict[str, Any], pass_k: int) -> dict[str, int]:
    reasons = {}
    for field, min_errors in RULES.items():
        errors = pass_k - success_count(row, field)
        if errors >= min_errors:
            reasons[field] = errors
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter SFT data by pass@k field error counts.")
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--field-predictions", default=DEFAULT_FIELD_PREDICTIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output")
    args = parser.parse_args()

    train_rows = list(read_jsonl(Path(args.train_file)))
    field_rows = list(read_jsonl(Path(args.field_predictions)))
    if not field_rows:
        raise ValueError("No field prediction rows found.")
    pass_k = infer_pass_k(field_rows[0])

    selected = {}
    reason_counter = Counter()
    for row in field_rows:
        index = int(row["index"])
        if index >= len(train_rows):
            raise IndexError(f"Prediction index {index} is outside train set size {len(train_rows)}")
        target = json.loads(train_rows[index]["messages"][-1]["content"])
        row.setdefault("expected", {})
        row["expected"]["judgment"] = target.get("judgment")
        reasons = failure_reasons(row, pass_k)
        if reasons:
            selected[index] = reasons
            reason_counter.update(reasons.keys())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for index in sorted(selected):
            f.write(json.dumps(train_rows[index], ensure_ascii=False) + "\n")

    manifest_path = Path(args.manifest_output) if args.manifest_output else output_path.with_suffix(".manifest.jsonl")
    with manifest_path.open("w", encoding="utf-8") as f:
        for index in sorted(selected):
            f.write(
                json.dumps(
                    {
                        "index": index,
                        "failure_reasons": selected[index],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "train_file": args.train_file,
        "field_predictions": args.field_predictions,
        "output": str(output_path),
        "manifest_output": str(manifest_path),
        "pass_k": pass_k,
        "total": len(train_rows),
        "selected": len(selected),
        "selected_rate": len(selected) / len(train_rows) if train_rows else 0.0,
        "rules": RULES,
        "reason_counts": dict(reason_counter),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
