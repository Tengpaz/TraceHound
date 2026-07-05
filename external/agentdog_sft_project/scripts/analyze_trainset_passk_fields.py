#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.binary_safety_eval import loose_parse_prediction, normalize_field, normalize_source


DEFAULT_PREDICTIONS = "outputs/eval/qwen35-0.8b-6label-train-pass8-hf/merged/predictions.jsonl"
DEFAULT_TRAIN_FILE = "outputs/data/agentdog_6label_sft.jsonl"
FIELDS = ("source", "harm_type", "risk_source", "failure_mode")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_targets(path: Path) -> dict[int, dict[str, str | None]]:
    targets = {}
    for index, row in enumerate(read_jsonl(path)):
        messages = row["messages"]
        target = json.loads(messages[-1]["content"])
        targets[index] = {
            "source": normalize_source(target.get("source")),
            "harm_type": normalize_field("harm_type", target.get("harm_type")),
            "risk_source": normalize_field("risk_source", target.get("risk_source")),
            "failure_mode": normalize_field("failure_mode", target.get("failure_mode")),
        }
    return targets


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def field_metrics(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for row in records if row[f"{field}_passed"])
    target_labels = sorted(set(row["expected"][field] for row in records if row["expected"][field] is not None))
    prediction_values = [
        rollout["parsed"].get(field)
        for row in records
        for rollout in row["rollouts"]
        if rollout["parsed"].get(field) is not None
    ]
    per_label = {}
    f1_values = []
    for label in target_labels:
        tp = sum(1 for row in records if row["expected"][field] == label and row[f"{field}_passed"])
        fn = sum(1 for row in records if row["expected"][field] == label and not row[f"{field}_passed"])
        fp = sum(
            1
            for row in records
            if row["expected"][field] != label
            and not row[f"{field}_passed"]
            and any(rollout["parsed"].get(field) == label for rollout in row["rollouts"])
        )
        scores = precision_recall_f1(tp, fp, fn)
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn, **scores}
        f1_values.append(scores["f1"])
    success_counts = [row[f"{field}_success_count"] for row in records]
    return {
        "pass_at_k_accuracy": passed / total if total else 0.0,
        "pass_at_k_correct": passed,
        "total": total,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "target_distribution": dict(Counter(row["expected"][field] for row in records)),
        "prediction_distribution": dict(Counter(prediction_values)),
        "success_count_distribution": dict(Counter(success_counts)),
        "success_count_mean": sum(success_counts) / len(success_counts) if success_counts else 0.0,
        "per_label": per_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze field-level pass@k from saved train-set rollouts.")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    targets = load_targets(Path(args.train_file))
    records = []
    for row in read_jsonl(Path(args.predictions)):
        index = int(row["index"])
        expected = targets[index]
        rollouts = []
        for rollout in row.get("rollouts", []):
            parsed = loose_parse_prediction(rollout.get("raw_output", ""))
            rollouts.append(
                {
                    "rollout_index": rollout.get("rollout_index"),
                    "parsed": parsed,
                    "raw_output": rollout.get("raw_output", ""),
                }
            )
        record = {"index": index, "expected": expected, "rollouts": rollouts}
        for field in FIELDS:
            success_count = sum(1 for rollout in rollouts if rollout["parsed"].get(field) == expected[field])
            record[f"{field}_success_count"] = success_count
            record[f"{field}_passed"] = success_count > 0
        records.append(record)

    metrics = {
        "predictions": args.predictions,
        "train_file": args.train_file,
        "total": len(records),
        "pass_k": max((len(row["rollouts"]) for row in records), default=0),
        "fields": {field: field_metrics(records, field) for field in FIELDS},
    }

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.predictions).parent / "field_passk"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "field_predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    (output_dir / "field_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
