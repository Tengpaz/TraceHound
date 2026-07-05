#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from guardrail.prompts import normalize_training_prompt


DATASETS = {
    "binary": "data/datasets/agentdog_raw/AgentDoG-BinarySafety/train.json",
    "taxonomy": "data/datasets/agentdog_raw/AgentDoG-FineGrainedTaxonomy/train.json",
}


def load_json(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(task: str) -> list[dict]:
    rows = []
    tasks = ["binary", "taxonomy"] if task == "combined" else [task]
    for name in tasks:
        for idx, example in enumerate(load_json(DATASETS[name])):
            user, assistant = normalize_training_prompt(example)
            rows.append(
                {
                    "id": f"{name}-{idx:06d}",
                    "task": name,
                    "messages": [
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "taxonomy", "combined"], required=True)
    parser.add_argument("--output-dir", default="outputs/data")
    args = parser.parse_args()

    rows = convert(args.task)
    output_path = Path(args.output_dir) / f"agentdog_{args.task}_sft.jsonl"
    write_jsonl(output_path, rows)
    print(json.dumps({"task": args.task, "examples": len(rows), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
