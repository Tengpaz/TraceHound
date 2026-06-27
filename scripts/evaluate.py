#!/usr/bin/env python
"""Evaluate TraceHound JSONL data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.config import load_env_file
from traceguard.evaluation import evaluate_dataset
from traceguard.judge import build_remote_judge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to eval JSONL.")
    parser.add_argument("--mode", default="layered", choices=["rules", "compressed", "layered"])
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "api", "hybrid"])
    parser.add_argument("--env-file", default=".env", help="Optional env file for API settings.")
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", help="API key. Prefer TRACEHOUND_API_KEY or .env for local use.")
    parser.add_argument("--model", help="Remote model name.")
    parser.add_argument("--api-path", help="API path. Defaults to /chat/completions.")
    parser.add_argument("--timeout", type=int, help="API timeout in seconds.")
    parser.add_argument("--prompt-mode", default="compressed", choices=["compressed", "full"])
    parser.add_argument("--limit", type=int, help="Evaluate only the first N samples.")
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args()

    load_env_file(args.env_file)
    judge = None
    if args.judge in {"api", "hybrid"}:
        try:
            judge = build_remote_judge(
                judge=args.judge,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                api_path=args.api_path,
                timeout=args.timeout,
                prompt_mode=args.prompt_mode,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    try:
        result = evaluate_dataset(args.path, mode=args.mode, judge=judge, limit=args.limit)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
