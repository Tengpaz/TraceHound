#!/usr/bin/env python
"""Run no-training validation with a remote OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.config import api_config_from_env, load_env_file
from traceguard.evaluation import evaluate_dataset
from traceguard.judge import build_remote_judge
from traceguard.model_profiles import resolve_model_profile


def _api_profile_defaults(name: str | None, path: str | None) -> dict:
    if not name:
        return {}
    try:
        profile = resolve_model_profile(name, path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if profile.get("provider") != "openai_compatible":
        raise SystemExit(f"api profile {name} must use provider=openai_compatible")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_eval.jsonl")
    parser.add_argument("--mode", default="layered", choices=["compressed", "layered", "rules"])
    parser.add_argument("--judge", default="hybrid", choices=["api", "hybrid"])
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key")
    parser.add_argument("--model")
    parser.add_argument("--api-path")
    parser.add_argument(
        "--api-profile",
        help=(
            "Optional OpenAI-compatible profile defaults, e.g. intern-s2-preview. "
            "Unset by default so .env/TRACEHOUND_API_BASE and TRACEHOUND_MODEL are used as-is."
        ),
    )
    parser.add_argument("--profile-path", help="Optional model profile JSON path.")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--prompt-mode", default="compressed", choices=["compressed", "full"])
    parser.add_argument("--limit", type=int, help="Start with 1-3 while checking a new API.")
    parser.add_argument("--output", help="Report path. Defaults to reports/api_validation_<timestamp>.json.")
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_profile = args.api_profile if args.api_profile is not None else os.getenv("TRACEHOUND_API_PROFILE")
    profile = _api_profile_defaults(api_profile, args.profile_path)
    try:
        config = api_config_from_env(
            api_base=args.api_base or profile.get("api_base"),
            api_key=args.api_key,
            model=args.model or profile.get("model"),
            api_path=args.api_path or profile.get("api_path"),
            timeout=args.timeout,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    judge = build_remote_judge(
        judge=args.judge,
        api_base=config.api_base,
        api_key=config.api_key,
        model=config.model,
        api_path=config.api_path,
        timeout=config.timeout,
        prompt_mode=args.prompt_mode,
    )
    try:
        result = evaluate_dataset(args.data, mode=args.mode, judge=judge, limit=args.limit)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    result["config"] = {
        "api_base": config.api_base,
        "api_path": config.api_path,
        "model": config.model,
        "timeout": config.timeout,
        "judge": args.judge,
        "mode": args.mode,
        "prompt_mode": args.prompt_mode,
        "api_profile": api_profile or "",
        "limit": args.limit,
        "api_key_present": bool(config.api_key),
    }

    output = Path(args.output) if args.output else Path("reports") / f"api_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nSaved report to {output}")


if __name__ == "__main__":
    main()
