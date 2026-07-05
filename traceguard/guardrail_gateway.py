"""Protocol adapters for deploying TraceHound as a runtime guardrail.

The core project uses a normalized ``TrajectoryCase``. Real agent runtimes
send very different hook payloads, so this module accepts tolerant JSON
events and turns them into a TraceHound trajectory before producing a portable
decision envelope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from traceguard.guard import TraceGuard
from traceguard.judge import ModelAdapter
from traceguard.pipeline import evaluate_case
from traceguard.schema import GuardDecision, RiskReport, TrajectoryCase, TrajectoryStep, dump_model

PRE_TOOL_EVENTS = {"pre_tool_use", "pretooluse", "before_tool_call", "pre_tool_call"}
POST_TOOL_EVENTS = {"post_tool_use", "posttooluse", "after_tool_observation", "post_tool_call"}
PRE_REPLY_EVENTS = {"pre_reply", "stop", "final_response", "before_final_response"}


def evaluate_guardrail_event(
    payload: Mapping[str, Any],
    *,
    platform: str = "auto",
    event_type: str = "auto",
    mode: str = "layered",
    judge: ModelAdapter | None = None,
) -> dict[str, Any]:
    """Evaluate a hook event from Claude Code, Codex wrappers, OpenClaw, or generic agents."""

    normalized = normalize_guardrail_event(payload, platform=platform, event_type=event_type)
    case: TrajectoryCase = normalized["case"]
    event = normalized["event_type"]
    candidate = normalized.get("candidate_action")

    if event == "pre_tool_use":
        assert isinstance(candidate, TrajectoryStep)
        decision = TraceGuard().before_tool_call(case, candidate)
        report = decision.report
        if judge is not None and mode != "rules" and decision.decision == "allow":
            model_case = case.model_copy(update={"trajectory": case.trajectory + [candidate]})
            report = evaluate_case(model_case, mode=mode, judge=judge)
            decision = GuardDecision(decision=report.decision, report=report, candidate_action=candidate)
    elif event == "post_tool_use":
        assert isinstance(candidate, TrajectoryStep)
        decision = TraceGuard().after_tool_observation(case, candidate)
        report = decision.report
        if judge is not None and mode != "rules" and decision.decision == "allow":
            model_case = case.model_copy(update={"trajectory": case.trajectory + [candidate]})
            report = evaluate_case(model_case, mode=mode, judge=judge)
            decision = GuardDecision(decision=report.decision, report=report, candidate_action=candidate)
    else:
        report = evaluate_case(case, mode=mode, judge=judge)
        decision = GuardDecision(decision=report.decision, report=report, candidate_action=candidate)

    allow = decision.decision in {"allow", "sanitize"}
    return {
        "platform": normalized["platform"],
        "event_type": event,
        "allow": allow,
        "decision": decision.decision,
        "reason": report.reason,
        "confidence": report.confidence,
        "report": dump_model(report),
        "candidate_action": dump_model(candidate) if isinstance(candidate, TrajectoryStep) else None,
        "case": dump_model(case),
        "adapter": platform_response(normalized["platform"], event, decision),
    }


def normalize_guardrail_event(
    payload: Mapping[str, Any],
    *,
    platform: str = "auto",
    event_type: str = "auto",
) -> dict[str, Any]:
    source_platform = _detect_platform(payload, platform)
    event = _detect_event(payload, event_type)
    trajectory = _trajectory_from_payload(payload)
    next_step = max((step.step for step in trajectory), default=0) + 1

    candidate: TrajectoryStep | None = None
    if event == "pre_tool_use":
        candidate = _candidate_tool_step(payload, next_step, source_platform)
    elif event == "post_tool_use":
        candidate = _observation_step(payload, next_step, source_platform)
    elif event == "pre_reply":
        final_text = _first_text(
            payload,
            "final_response",
            "final",
            "response",
            "assistant_response",
            "completion",
            default="",
        )
        if final_text:
            candidate = TrajectoryStep(
                step=next_step,
                role="assistant",
                type="final",
                content=final_text,
                metadata={"platform": source_platform, "event_type": event},
            )
            trajectory.append(candidate)

    case = TrajectoryCase(
        id=str(payload.get("case_id") or payload.get("session_id") or payload.get("sessionId") or f"guardrail-{uuid4().hex[:10]}"),
        task=_task_from_payload(payload, trajectory),
        metadata={
            "platform": source_platform,
            "event_type": event,
            "cwd": payload.get("cwd") or payload.get("project_dir") or payload.get("workspace"),
            "raw_event_keys": sorted(str(key) for key in payload.keys()),
        },
        trajectory=trajectory or [_fallback_user_step(payload)],
    )
    return {"platform": source_platform, "event_type": event, "case": case, "candidate_action": candidate}


def platform_response(platform: str, event_type: str, decision: GuardDecision) -> dict[str, Any]:
    """Return a platform-friendly response without hiding the normalized report."""

    reason = decision.report.reason or "TraceHound guardrail decision."
    if platform == "claude-code":
        if event_type == "pre_tool_use" and decision.decision in {"block", "ask_confirm"}:
            hook_json = {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
            return {
                "exit_code": 2,
                "stderr": reason,
                "json": hook_json,
            }
        if decision.decision in {"block", "ask_confirm"}:
            return {
                "exit_code": 2,
                "stderr": reason,
                "json": {"decision": "block", "reason": reason},
            }
        return {"exit_code": 0, "stdout": "", "json": {}}
    if platform == "openclaw":
        return {
            "action": "continue" if decision.decision in {"allow", "sanitize"} else "interrupt",
            "guard_decision": decision.decision,
            "message": reason,
        }
    if platform == "codex":
        return {
            "allowed": decision.decision in {"allow", "sanitize"},
            "requires_confirmation": decision.decision == "ask_confirm",
            "message": reason,
        }
    return {"allowed": decision.decision in {"allow", "sanitize"}, "message": reason}


def read_json_stdin_or_file(path: str | None = None) -> dict[str, Any]:
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    import sys

    raw = sys.stdin.read()
    return json.loads(raw or "{}")


def _detect_platform(payload: Mapping[str, Any], requested: str) -> str:
    if requested != "auto":
        return requested
    explicit = str(payload.get("platform") or payload.get("runtime") or "").strip().lower()
    if explicit:
        return explicit
    if "hook_event_name" in payload or "hookEventName" in payload or "transcript_path" in payload:
        return "claude-code"
    if "openclaw" in str(payload.get("agent") or payload.get("source") or "").lower():
        return "openclaw"
    if "codex" in str(payload.get("agent") or payload.get("source") or "").lower():
        return "codex"
    return "generic"


def _detect_event(payload: Mapping[str, Any], requested: str) -> str:
    raw = requested if requested != "auto" else str(
        payload.get("event_type")
        or payload.get("hook_event_name")
        or payload.get("hookEventName")
        or payload.get("event")
        or ""
    )
    normalized = raw.replace("-", "_").replace(".", "_").strip().lower()
    if normalized in PRE_TOOL_EVENTS:
        return "pre_tool_use"
    if normalized in POST_TOOL_EVENTS:
        return "post_tool_use"
    if normalized in PRE_REPLY_EVENTS:
        return "pre_reply"
    if payload.get("tool_response") or payload.get("tool_output") or payload.get("observation"):
        return "post_tool_use"
    if payload.get("tool_name") or payload.get("toolName") or payload.get("tool_input"):
        return "pre_tool_use"
    return "pre_reply"


def _trajectory_from_payload(payload: Mapping[str, Any]) -> list[TrajectoryStep]:
    raw_trajectory = payload.get("trajectory") or payload.get("steps") or payload.get("trace")
    if isinstance(raw_trajectory, list):
        return _steps_from_sequence(raw_trajectory)

    messages = payload.get("messages") or payload.get("conversation")
    if isinstance(messages, list):
        return _steps_from_sequence(messages)

    transcript_path = payload.get("transcript_path") or payload.get("transcriptPath")
    if transcript_path:
        parsed = _steps_from_transcript(Path(str(transcript_path)))
        if parsed:
            return parsed

    user_text = _first_text(payload, "user_prompt", "prompt", "task", "instruction", default="")
    if user_text:
        return [TrajectoryStep(step=1, role="user", type="message", content=user_text)]
    return []


def _steps_from_sequence(items: list[Any]) -> list[TrajectoryStep]:
    steps: list[TrajectoryStep] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, Mapping):
            steps.append(TrajectoryStep(step=index, role="unknown", type="message", content=str(item)))
            continue
        if {"step", "role", "type"}.issubset(item):
            try:
                parsed_step = TrajectoryStep.model_validate(item)
                if parsed_step.tool_name:
                    parsed_step.tool_name = _normalize_tool_name(parsed_step.tool_name, {"arguments": parsed_step.arguments or {}})
                steps.append(parsed_step)
                continue
            except Exception:
                pass
        role = str(item.get("role") or item.get("speaker") or item.get("actor") or "assistant")
        tool_name = item.get("tool_name") or item.get("toolName") or item.get("name")
        arguments = item.get("arguments") or item.get("args") or item.get("input") or {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        step_type = str(item.get("type") or ("tool_call" if tool_name else "message"))
        content = _string_content(item.get("content") or item.get("text") or item.get("output") or item.get("message") or "")
        steps.append(
            TrajectoryStep(
                step=int(item.get("step") or index),
                role=role,
                type=step_type,
                content=content,
                tool_name=str(tool_name) if tool_name else None,
                arguments=arguments,
                metadata={"raw": dict(item)},
            )
        )
    return sorted(steps, key=lambda step: step.step)


def _steps_from_transcript(path: Path) -> list[TrajectoryStep]:
    if not path.exists() or not path.is_file():
        return []
    steps: list[TrajectoryStep] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, Mapping):
            steps.extend(_steps_from_sequence([item]))
    for index, step in enumerate(steps, start=1):
        step.step = index
    return steps


def _candidate_tool_step(payload: Mapping[str, Any], step: int, platform: str) -> TrajectoryStep:
    tool_name = str(payload.get("tool_name") or payload.get("toolName") or payload.get("name") or payload.get("tool") or "tool.call")
    tool_name = _normalize_tool_name(tool_name, payload)
    arguments = payload.get("tool_input") or payload.get("toolInput") or payload.get("arguments") or payload.get("input") or {}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return TrajectoryStep(
        step=step,
        role="agent",
        type="tool_call",
        content=_string_content(payload.get("content") or ""),
        tool_name=tool_name,
        arguments=arguments,
        metadata={"platform": platform, "event_payload": _safe_payload_summary(payload)},
    )


def _observation_step(payload: Mapping[str, Any], step: int, platform: str) -> TrajectoryStep:
    tool_name = str(payload.get("tool_name") or payload.get("toolName") or payload.get("name") or payload.get("tool") or "tool.call")
    content = _first_text(payload, "tool_response", "tool_output", "observation", "result", "output", default="")
    return TrajectoryStep(
        step=step,
        role="tool",
        type="observation",
        content=content,
        tool_name=_normalize_tool_name(tool_name, payload),
        metadata={"platform": platform, "event_payload": _safe_payload_summary(payload)},
    )


def _fallback_user_step(payload: Mapping[str, Any]) -> TrajectoryStep:
    return TrajectoryStep(
        step=1,
        role="user",
        type="message",
        content=_first_text(payload, "task", "prompt", "user_prompt", default="Runtime guardrail event"),
    )


def _task_from_payload(payload: Mapping[str, Any], trajectory: list[TrajectoryStep]) -> str:
    explicit = _first_text(payload, "task", "user_prompt", "prompt", "instruction", default="")
    if explicit:
        return explicit
    for step in trajectory:
        if step.role == "user" and step.content:
            return step.content
    return ""


def _normalize_tool_name(tool_name: str, payload: Mapping[str, Any]) -> str:
    lowered = tool_name.lower()
    args = payload.get("tool_input") or payload.get("arguments") or payload.get("input") or {}
    arg_text = json.dumps(args, ensure_ascii=False).lower() if isinstance(args, Mapping) else str(args).lower()
    if lowered in {"read", "glob", "grep"}:
        return "file.read"
    if lowered in {"write", "edit", "multiedit", "notebookedit"}:
        return "file.write"
    if lowered in {"bash", "shell", "terminal"}:
        return "shell.run"
    if "rm -rf" in arg_text or "sudo " in arg_text:
        return "shell.run"
    return lowered


def _first_text(payload: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return _string_content(value)
    return default


def _string_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _safe_payload_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = {}
    for key in ("session_id", "sessionId", "cwd", "tool_name", "toolName", "hook_event_name", "hookEventName"):
        if key in payload:
            summary[key] = payload[key]
    return summary
