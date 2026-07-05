#!/usr/bin/env python
"""Safely merge TraceHound hooks into an existing Claude Code settings file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.claude_code_hooks import build_tracehound_hook_settings, merge_tracehound_hooks, read_settings, write_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--settings",
        default=".claude/settings.json",
        help="Claude Code settings path to update. Defaults to .claude/settings.json in the current project.",
    )
    parser.add_argument("--global-settings", action="store_true", help="Use ~/.claude/settings.json instead of --settings.")
    parser.add_argument("--tracehound-root", default=str(ROOT), help="TraceHound project root.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", default="layered", choices=["rules", "compressed", "layered"])
    parser.add_argument("--judge", default="heuristic", choices=["heuristic", "api", "hybrid"])
    parser.add_argument("--hook-type", default="command", choices=["command", "http"], help="Claude Code hook handler type.")
    parser.add_argument("--python-command", default="python", help='Python launcher used by Claude Code hook, e.g. "conda run -n tracehound python".')
    parser.add_argument("--timeout", default=30, type=int)
    parser.add_argument("--no-backup", action="store_true", help="Do not create a timestamped backup before writing.")
    parser.add_argument("--dry-run", action="store_true", help="Print merged settings without writing.")
    args = parser.parse_args()

    settings_path = Path("~/.claude/settings.json").expanduser() if args.global_settings else Path(args.settings).expanduser()
    existing = read_settings(settings_path)
    fragment = build_tracehound_hook_settings(
        tracehound_root=args.tracehound_root,
        server_url=args.server_url,
        mode=args.mode,
        judge=args.judge,
        python_command=args.python_command,
        hook_type=args.hook_type,
        timeout=args.timeout,
    )
    merged = merge_tracehound_hooks(existing, fragment)

    if args.dry_run:
        print(json.dumps(merged, ensure_ascii=False, indent=2))
        return

    backup_path = write_settings(settings_path, merged, backup=not args.no_backup)
    print(f"TraceHound hooks installed into {settings_path}")
    if backup_path:
        print(f"Backup written to {backup_path}")
    print("Existing Claude Code settings were preserved; only TraceHound-managed hooks were refreshed.")


if __name__ == "__main__":
    main()
