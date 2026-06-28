#!/usr/bin/env python
"""Generate synthetic TraceHound data files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.data import built_in_cases, dataset_summary
from traceguard.export import eval_rows, preference_rows, rl_rows, sft_rows, write_jsonl
from traceguard.generation_config import load_generation_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML config, e.g. configs/generation.yaml.")
    parser.add_argument("--out", help="Output directory. Overrides config.")
    parser.add_argument("--scenario", action="append", help="Filter by scenario. May be repeated.")
    parser.add_argument("--label", action="append", choices=["safe", "unsafe"], help="Filter by gold label. May be repeated.")
    parser.add_argument("--limit", type=int, help="Limit number of cases.")
    parser.add_argument("--count", type=int, help="Generate this many eval cases by deterministic variation.")
    parser.add_argument("--include-rl", action="store_true", help="Also write synthetic_rl.jsonl for DPO/GRPO-style training.")
    args = parser.parse_args()

    config = load_generation_config(args.config)
    scenarios = args.scenario if args.scenario is not None else config["scenarios"]
    labels = args.label if args.label is not None else config["labels"]
    limit = args.limit if args.limit is not None else config["limit"]
    count = args.count if args.count is not None else config["count"]
    include_eval = bool(config["include_eval"])
    include_sft = bool(config["include_sft"])
    include_preference = bool(config["include_preference"])
    include_rl = bool(args.include_rl or config["include_rl"])
    if not include_eval and not include_sft and not include_preference and not include_rl:
        raise SystemExit("at least one dataset type must be enabled")

    cases = built_in_cases(scenarios=scenarios, labels=labels, limit=limit, count=count)
    out = Path(args.out or config["out"] or "data")
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    if include_eval:
        counts["synthetic_eval.jsonl"] = write_jsonl(out / "synthetic_eval.jsonl", eval_rows(cases))
    if include_sft:
        counts["synthetic_sft.jsonl"] = write_jsonl(out / "synthetic_sft.jsonl", sft_rows(cases))
    if include_preference:
        counts["synthetic_preference.jsonl"] = write_jsonl(out / "synthetic_preference.jsonl", preference_rows(cases))
    if include_rl:
        counts["synthetic_rl.jsonl"] = write_jsonl(out / "synthetic_rl.jsonl", rl_rows(cases))
    examples = Path("examples")
    examples.mkdir(parents=True, exist_ok=True)
    demo_cases = built_in_cases()
    (examples / "demo_cases.json").write_text(json.dumps(demo_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out), "counts": counts, "summary": dataset_summary(cases)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
