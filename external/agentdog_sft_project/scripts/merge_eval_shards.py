#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.binary_safety_eval import compute_metrics as compute_atbench_metrics  # noqa: E402
from scripts.trainset_passk_eval import compute_metrics as compute_trainset_metrics  # noqa: E402


def read_predictions(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise ValueError("No predictions found.")
    return rows


def sort_key(row: dict[str, Any]):
    return row.get("eval_index", row.get("index", row.get("id", 0)))


def infer_kind(rows: list[dict[str, Any]]) -> str:
    first = rows[0]
    if "expected_judgment" in first:
        return "trainset"
    if "expected" in first and "parsed" in first:
        return "atbench"
    raise ValueError("Could not infer prediction format. Use --kind.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge eval shard predictions and recompute metrics.")
    parser.add_argument("--input-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kind", choices=["auto", "atbench", "trainset"], default="auto")
    parser.add_argument("--pass-k", type=int, help="Required only for trainset metrics if not inferable.")
    args = parser.parse_args()

    prediction_paths = [Path(path) / "predictions.jsonl" for path in args.input_dirs]
    missing = [str(path) for path in prediction_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prediction files: {missing}")

    rows = sorted(read_predictions(prediction_paths), key=sort_key)
    kind = infer_kind(rows) if args.kind == "auto" else args.kind
    if kind == "trainset":
        pass_k = args.pass_k or max(len(row.get("rollouts", [])) for row in rows)
        metrics = compute_trainset_metrics(rows, pass_k)
    else:
        metrics = compute_atbench_metrics(rows)
    metrics.update(
        {
            "merged_from": args.input_dirs,
            "kind": kind,
            "total": len(rows),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
