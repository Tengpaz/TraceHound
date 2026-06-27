#!/usr/bin/env python
"""Run quality checks for TraceHound eval JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.quality import quality_check_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    result = quality_check_jsonl(args.path)
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

