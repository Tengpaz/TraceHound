#!/usr/bin/env python
"""TraceHound runtime guardrail hook for agent systems.

Reads one JSON event from stdin or ``--input`` and prints a decision envelope.
For Claude Code command hooks, pass ``--adapter-json`` to emit only the
platform-native hook decision JSON, or ``--claude-exit-codes`` to block with
exit code 2 and stderr.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.config import load_env_file
from traceguard.guardrail_gateway import evaluate_guardrail_event, read_json_stdin_or_file
from traceguard.judge import build_remote_judge
from traceguard.local_model import build_local_binary_judge, build_local_judge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="JSON event path. Defaults to stdin.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--platform", default="auto", choices=["auto", "generic", "claude-code", "codex", "openclaw"])
    parser.add_argument("--event-type", default="auto", help="auto, pre_tool_use, post_tool_use, or pre_reply.")
    parser.add_argument("--mode", default="layered", choices=["rules", "compressed", "layered"])
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "api", "hybrid", "local", "local-binary"])
    parser.add_argument("--model-profile", help="Local model profile for --judge local/local-binary.")
    parser.add_argument("--model-path", help="Local model path or Hugging Face id override for --judge local/local-binary.")
    parser.add_argument("--server-url", help="Optional TraceHound server base URL, e.g. http://127.0.0.1:8000.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--adapter-json", action="store_true", help="Print only the platform-native adapter JSON.")
    parser.add_argument("--claude-exit-codes", action="store_true", help="Exit 2 on block/ask_confirm for Claude Code hooks.")
    args = parser.parse_args()

    load_env_file(args.env_file)
    payload = read_json_stdin_or_file(args.input)
    if args.server_url:
        result = _post_to_server(args.server_url, payload, args)
    else:
        judge = None
        if args.judge in {"api", "hybrid"}:
            judge = build_remote_judge(judge=args.judge)
        elif args.judge == "local":
            judge = build_local_judge(model_profile=args.model_profile, model_path=args.model_path)
        elif args.judge == "local-binary":
            judge = build_local_binary_judge(model_profile=args.model_profile, model_path=args.model_path)
        result = evaluate_guardrail_event(
            payload,
            platform=args.platform,
            event_type=args.event_type,
            mode=args.mode,
            judge=judge,
        )

    if args.claude_exit_codes:
        _emit_claude_command_result(result)
        return
    if args.adapter_json:
        _emit_adapter_json(result)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def _post_to_server(base_url: str, payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/guardrail/event"
    body = {
        "platform": args.platform,
        "event_type": args.event_type,
        "mode": args.mode,
        "judge": args.judge,
        "event": payload,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"TraceHound guardrail server request failed: {exc}") from exc


def _emit_claude_command_result(result: dict[str, Any]) -> None:
    adapter = result.get("adapter") or {}
    exit_code = int(adapter.get("exit_code", 0))
    if exit_code == 2:
        print(str(adapter.get("stderr") or result.get("reason") or "Blocked by TraceHound."), file=sys.stderr)
        raise SystemExit(2)
    stdout_json = adapter.get("json") or {"decision": "approve", "reason": result.get("reason", "")}
    print(json.dumps(stdout_json, ensure_ascii=False, sort_keys=True))


def _emit_adapter_json(result: dict[str, Any]) -> None:
    adapter = result.get("adapter") or {}
    payload = adapter.get("json") or {}
    if payload:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
