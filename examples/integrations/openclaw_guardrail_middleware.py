#!/usr/bin/env python
"""OpenClaw-style TraceHound guardrail middleware.

Use this around OpenClaw's tool dispatcher or final-delivery path. The adapter
is intentionally framework-light: pass the current trajectory, candidate tool
call, and final response as dictionaries and let TraceHound normalize them.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class TraceHoundGuardrail:
    endpoint: str = "http://127.0.0.1:8000/api/guardrail/event"
    mode: str = "layered"
    judge: str = "heuristic"
    timeout: int = 30

    def before_tool(self, session_id: str, trajectory: list[dict[str, Any]], tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        return self._check(
            {
                "session_id": session_id,
                "trajectory": trajectory,
                "tool_name": tool_name,
                "tool_input": tool_input,
            },
            event_type="pre_tool_use",
        )

    def after_tool(
        self,
        session_id: str,
        trajectory: list[dict[str, Any]],
        tool_name: str,
        tool_output: Any,
    ) -> dict[str, Any]:
        return self._check(
            {
                "session_id": session_id,
                "trajectory": trajectory,
                "tool_name": tool_name,
                "tool_output": str(tool_output),
            },
            event_type="post_tool_use",
        )

    def before_final_response(self, session_id: str, trajectory: list[dict[str, Any]], final_response: str) -> dict[str, Any]:
        return self._check(
            {
                "session_id": session_id,
                "trajectory": trajectory,
                "final_response": final_response,
            },
            event_type="pre_reply",
        )

    def _check(self, event: dict[str, Any], *, event_type: str) -> dict[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(
                {
                    "platform": "openclaw",
                    "event_type": event_type,
                    "mode": self.mode,
                    "judge": self.judge,
                    "event": event,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        if result.get("adapter", {}).get("action") == "interrupt":
            raise RuntimeError(f"TraceHound interrupted OpenClaw action: {result.get('reason')}")
        return result
