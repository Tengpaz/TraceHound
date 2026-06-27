#!/usr/bin/env python
"""Placeholder DPO/ORPO preference entrypoint for contest GPU environments."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict


TRAIN_PACKAGES = ("torch", "transformers", "datasets", "trl", "accelerate")


def missing_packages() -> list[str]:
    return [package for package in TRAIN_PACKAGES if importlib.util.find_spec(package) is None]


def dataset_stats(path: Path) -> Dict[str, Any]:
    rows = 0
    complete_pairs = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            item = json.loads(line)
            if item.get("prompt") and item.get("chosen") and item.get("rejected"):
                complete_pairs += 1
    return {"samples": rows, "complete_pairs": complete_pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_preference.jsonl")
    parser.add_argument("--base-model", default="contest-sft-model")
    parser.add_argument("--output-dir", default="checkpoints/preference")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if training dependencies are missing.")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"missing preference data: {data_path}. Run `python scripts/generate_data.py --out data` first.")

    stats = dataset_stats(data_path)
    print(
        json.dumps(
            {
                "data": str(data_path),
                "base_model": args.base_model,
                "output_dir": args.output_dir,
                "stats": stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    missing = missing_packages()
    if missing:
        message = (
            "Preference optimization is intentionally left for a GPU-capable contest server or optional training environment.\n"
            f"Missing packages: {', '.join(missing)}\n"
            "On Linux/GPU, install the CUDA-matched PyTorch build first, then run `pip install -e \".[train]\"`."
        )
        if args.strict:
            raise SystemExit(message)
        print(message)
        return

    print(
        "Preference-training dependencies are available. This MVP does not launch DPO/ORPO automatically; "
        "wire the official trainer after contest rules are known."
    )


if __name__ == "__main__":
    main()
