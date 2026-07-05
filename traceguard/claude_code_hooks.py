"""Utilities for safely installing TraceHound hooks into Claude Code settings."""

from __future__ import annotations

import copy
import json
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

TRACEHOUND_HOOK_MARKERS = ("scripts/guardrail_hook.py", "tracehound-guardrail-hook", "/api/guardrail/claude-code")
DEFAULT_TOOL_MATCHER = "Bash|Read|Write|Edit|MultiEdit"


def build_tracehound_hook_settings(
    *,
    tracehound_root: str | Path,
    server_url: str = "http://127.0.0.1:8000",
    mode: str = "layered",
    judge: str = "heuristic",
    python_command: str = "python",
    hook_type: str = "command",
    timeout: int = 30,
) -> dict[str, Any]:
    """Build the Claude Code settings fragment managed by TraceHound."""

    root = Path(tracehound_root).expanduser().resolve()
    hook_script = root / "scripts" / "guardrail_hook.py"
    if hook_type not in {"command", "http"}:
        raise ValueError("hook_type must be command or http")

    def command(event_type: str) -> str:
        return " ".join(
            [
                python_command,
                shlex.quote(str(hook_script)),
                "--platform claude-code",
                f"--event-type {shlex.quote(event_type)}",
                f"--mode {shlex.quote(mode)}",
                f"--judge {shlex.quote(judge)}",
                f"--server-url {shlex.quote(server_url)}",
                "--adapter-json",
            ]
        )

    def handler(event_type: str) -> dict[str, Any]:
        if hook_type == "http":
            return {
                "type": "http",
                "url": f"{server_url.rstrip('/')}/api/guardrail/claude-code?mode={mode}&judge={judge}",
                "timeout": timeout,
            }
        return {"type": "command", "command": command(event_type), "timeout": timeout}

    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": DEFAULT_TOOL_MATCHER,
                    "hooks": [handler("auto")],
                }
            ],
            "PostToolUse": [
                {
                    "matcher": DEFAULT_TOOL_MATCHER,
                    "hooks": [handler("post_tool_use")],
                }
            ],
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [handler("pre_reply")],
                }
            ],
        }
    }


def merge_tracehound_hooks(existing: dict[str, Any], tracehound_fragment: dict[str, Any]) -> dict[str, Any]:
    """Return settings with TraceHound hooks appended while preserving user settings."""

    merged = copy.deepcopy(existing or {})
    merged = remove_tracehound_hooks(merged)
    existing_hooks = merged.setdefault("hooks", {})
    if not isinstance(existing_hooks, dict):
        raise ValueError("Claude Code settings field `hooks` must be an object")

    new_hooks = tracehound_fragment.get("hooks", {})
    if not isinstance(new_hooks, dict):
        raise ValueError("TraceHound hook fragment field `hooks` must be an object")

    for event_name, new_groups in new_hooks.items():
        if not isinstance(new_groups, list):
            raise ValueError(f"TraceHound hook event `{event_name}` must be a list")
        event_groups = existing_hooks.setdefault(event_name, [])
        if not isinstance(event_groups, list):
            raise ValueError(f"Claude Code settings hooks.{event_name} must be a list")
        for new_group in new_groups:
            event_groups.append(copy.deepcopy(new_group))
    return merged


def remove_tracehound_hooks(settings: dict[str, Any]) -> dict[str, Any]:
    """Remove only previously installed TraceHound hooks, leaving all user hooks intact."""

    cleaned = copy.deepcopy(settings or {})
    hooks = cleaned.get("hooks")
    if not isinstance(hooks, dict):
        return cleaned

    empty_events = []
    for event_name, groups in hooks.items():
        if not isinstance(groups, list):
            continue
        kept_groups = []
        for group in groups:
            if not isinstance(group, dict):
                kept_groups.append(group)
                continue
            group_hooks = group.get("hooks")
            if not isinstance(group_hooks, list):
                kept_groups.append(group)
                continue
            kept_group_hooks = [hook for hook in group_hooks if not is_tracehound_hook(hook)]
            if kept_group_hooks:
                next_group = copy.deepcopy(group)
                next_group["hooks"] = kept_group_hooks
                kept_groups.append(next_group)
            elif len(kept_group_hooks) == len(group_hooks):
                kept_groups.append(group)
        if kept_groups:
            hooks[event_name] = kept_groups
        else:
            empty_events.append(event_name)

    for event_name in empty_events:
        hooks.pop(event_name, None)
    if not hooks:
        cleaned.pop("hooks", None)
    return cleaned


def is_tracehound_hook(hook: Any) -> bool:
    if not isinstance(hook, dict):
        return False
    command = str(hook.get("command") or "")
    url = str(hook.get("url") or "")
    args = hook.get("args")
    args_text = " ".join(str(arg) for arg in args) if isinstance(args, list) else ""
    text = f"{command} {url} {args_text}"
    return any(marker in text for marker in TRACEHOUND_HOOK_MARKERS)


def read_settings(path: str | Path) -> dict[str, Any]:
    settings_path = Path(path).expanduser()
    if not settings_path.exists():
        return {}
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {settings_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Claude Code settings must be a JSON object: {settings_path}")
    return payload


def write_settings(path: str | Path, settings: dict[str, Any], *, backup: bool = True) -> Path | None:
    settings_path = Path(path).expanduser()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if backup and settings_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = settings_path.with_name(f"{settings_path.name}.tracehound-backup-{stamp}")
        backup_path.write_text(settings_path.read_text(encoding="utf-8"), encoding="utf-8")
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return backup_path
