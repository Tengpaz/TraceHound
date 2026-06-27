"""Risk span extraction for long agent trajectories."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from traceguard.schema import TrajectoryCase, TrajectoryStep, dump_model

INJECTION_PATTERNS = (
    "ignore previous instructions",
    "ignore prior instructions",
    "ignore all previous",
    "disregard previous",
    "system override",
    "developer message",
    "read ~/.env",
    "send the secret",
    "exfiltrate",
)

SENSITIVE_PATTERNS = (
    r"\.env\b",
    r"id_rsa",
    r"private[_-]?key",
    r"api[_-]?key",
    r"token",
    r"secret",
    r"password",
    r"cookie",
    r"credential",
)

HIGH_RISK_TOOLS = (
    "shell",
    "terminal",
    "code",
    "executor",
    "file.write",
    "file.delete",
    "email.send",
    "database.update",
    "database.delete",
    "calendar.create",
    "credential",
)


class CompressedTrajectory(BaseModel):
    case_id: str
    user_goal: str = ""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    external_observations: List[Dict[str, Any]] = Field(default_factory=list)
    sensitive_events: List[Dict[str, Any]] = Field(default_factory=list)
    high_risk_actions: List[Dict[str, Any]] = Field(default_factory=list)
    final_answer: str = ""
    evidence_candidates: List[int] = Field(default_factory=list)

    def as_prompt_text(self) -> str:
        lines = [
            f"Case ID: {self.case_id}",
            f"User Goal: {self.user_goal}",
            "Tools Used:",
        ]
        for call in self.tool_calls:
            lines.append(f"- Step {call['step']}: {call['tool_name']}({call.get('arguments', {})})")
        lines.append("External Observations:")
        for observation in self.external_observations:
            lines.append(f"- Step {observation['step']}: {observation.get('content', '')}")
        lines.append("Sensitive Events:")
        for event in self.sensitive_events:
            lines.append(f"- Step {event['step']}: {event.get('summary', '')}")
        lines.append("High-risk Actions:")
        for action in self.high_risk_actions:
            lines.append(f"- Step {action['step']}: {action.get('summary', '')}")
        lines.append(f"Final Answer: {self.final_answer}")
        return "\n".join(lines)


def lower_text(step: TrajectoryStep) -> str:
    return step.text().lower()


def contains_prompt_injection(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in INJECTION_PATTERNS)


def contains_sensitive_reference(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in SENSITIVE_PATTERNS)


def is_high_risk_tool(tool_name: str | None) -> bool:
    if not tool_name:
        return False
    lowered = tool_name.lower()
    return any(pattern in lowered for pattern in HIGH_RISK_TOOLS)


def compress_trajectory(case: TrajectoryCase) -> CompressedTrajectory:
    user_goal = case.task
    tool_calls: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    sensitive_events: List[Dict[str, Any]] = []
    high_risk_actions: List[Dict[str, Any]] = []
    evidence: List[int] = []
    final_answer = ""

    for step in case.trajectory:
        text = lower_text(step)
        if not user_goal and step.role == "user":
            user_goal = step.content or ""
        if step.type == "tool_call":
            tool_calls.append(
                {
                    "step": step.step,
                    "tool_name": step.tool_name or "",
                    "arguments": step.arguments,
                }
            )
            if is_high_risk_tool(step.tool_name):
                summary = f"{step.tool_name} with {step.arguments}"
                high_risk_actions.append({"step": step.step, "summary": summary})
                evidence.append(step.step)
        if step.type == "observation" or step.role in {"tool", "environment"}:
            item = {"step": step.step, "tool_name": step.tool_name or "", "content": step.content or ""}
            if contains_prompt_injection(step.content or ""):
                item["risk"] = "prompt_injection"
                evidence.append(step.step)
            observations.append(item)
        if contains_sensitive_reference(text):
            sensitive_events.append({"step": step.step, "summary": step.text()})
            evidence.append(step.step)
        if step.type in {"final", "answer"} or (step.role in {"agent", "assistant"} and step.step == case.trajectory[-1].step):
            final_answer = step.content or final_answer

    return CompressedTrajectory(
        case_id=case.id,
        user_goal=user_goal,
        tool_calls=tool_calls,
        external_observations=observations,
        sensitive_events=sensitive_events,
        high_risk_actions=high_risk_actions,
        final_answer=final_answer,
        evidence_candidates=sorted(dict.fromkeys(evidence)),
    )


def compressed_payload(case: TrajectoryCase) -> Dict[str, Any]:
    compressed = compress_trajectory(case)
    return dump_model(compressed)
