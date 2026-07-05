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
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "api", "hybrid", "local", "local-binary"])
    parser.add_argument("--env-file", default=".env", help="Optional env file for API settings.")
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", help="API key. Prefer TRACEHOUND_API_KEY or .env for local use.")
    parser.add_argument("--model", help="Remote model name.")
    parser.add_argument("--api-path", help="API path. Defaults to /chat/completions.")
    parser.add_argument("--timeout", type=int, help="API timeout in seconds.")
    parser.add_argument("--prompt-mode", default="compressed", choices=["compressed", "full"])
    parser.add_argument("--model-profile", help="Local model profile name, for example internlm3-8b-instruct or tracehound-base-qwen3_5-0_8b-binary.")
    parser.add_argument("--model-path", help="Local model path or Hugging Face id override for --judge local/local-binary.")
    parser.add_argument("--profile-path", help="Optional model profile JSON path.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Local model generation budget.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Local model sampling temperature.")
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
    elif args.judge == "local":
        try:
            from traceguard.local_model import build_local_judge

            judge = build_local_judge(
                model_profile=args.model_profile,
                model_path=args.model_path,
                profile_path=args.profile_path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
    elif args.judge == "local-binary":
        try:
            from traceguard.local_model import build_local_binary_judge

            judge = build_local_binary_judge(
                model_profile=args.model_profile,
                model_path=args.model_path,
                profile_path=args.profile_path,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except (RuntimeError, ValueError) as exc:
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
