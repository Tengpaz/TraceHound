#!/usr/bin/env python
"""Generic Codex-style guardrail wrapper around a tool dispatcher.

Codex surfaces do not all expose the same public hook interface. This example
shows the integration point TraceHound expects: call ``guard_event`` before a
tool executes and again before the final answer is delivered.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable

TRACEHOUND_URL = "http://127.0.0.1:8000/api/guardrail/event"


def guard_event(event: dict[str, Any], *, event_type: str) -> dict[str, Any]:
    payload = {
        "platform": "codex",
        "event_type": event_type,
        "mode": "layered",
        "judge": "heuristic",
        "event": event,
    }
    request = urllib.request.Request(
        TRACEHOUND_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if not result.get("allow", False):
        raise RuntimeError(f"TraceHound blocked agent action: {result.get('reason')}")
    return result


def guarded_tool_call(
    *,
    session_id: str,
    trajectory: list[dict[str, Any]],
    tool_name: str,
    tool_input: dict[str, Any],
    dispatch: Callable[[str, dict[str, Any]], Any],
) -> Any:
    guard_event(
        {
            "session_id": session_id,
            "trajectory": trajectory,
            "tool_name": tool_name,
            "tool_input": tool_input,
        },
        event_type="pre_tool_use",
    )
    output = dispatch(tool_name, tool_input)
    trajectory.append(
        {
            "step": len(trajectory) + 1,
            "role": "agent",
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": tool_input,
        }
    )
    trajectory.append(
        {
            "step": len(trajectory) + 1,
            "role": "tool",
            "type": "observation",
            "tool_name": tool_name,
            "content": str(output),
        }
    )
    return output


def guarded_final_response(session_id: str, trajectory: list[dict[str, Any]], final_response: str) -> None:
    guard_event(
        {
            "session_id": session_id,
            "trajectory": trajectory,
            "final_response": final_response,
        },
        event_type="pre_reply",
    )
