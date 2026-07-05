#!/usr/bin/env python
"""Convert official or ad-hoc JSON/JSONL data into TraceHound eval JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.schema import TrajectoryCase, report_from_gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input JSON or JSONL path.")
    parser.add_argument("output", help="Output TraceHound JSONL path.")
    parser.add_argument("--limit", type=int, help="Convert only the first N records.")
    parser.add_argument("--strict", action="store_true", help="Fail on the first unconvertible record.")
    parser.add_argument("--source", default="official", help="Metadata source tag.")
    args = parser.parse_args()

    records = list(iter_records(Path(args.input), limit=args.limit))
    converted: List[Dict[str, Any]] = []
    errors: List[str] = []
    for index, record in enumerate(records, start=1):
        try:
            converted.append(convert_record(record, index=index, source=args.source))
        except (KeyError, TypeError, ValueError) as exc:
            message = f"record {index}: {exc}"
            if args.strict:
                raise SystemExit(message) from exc
            errors.append(message)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {"input": args.input, "output": args.output, "converted": len(converted), "errors": errors}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors and not converted:
        raise SystemExit(1)


def iter_records(path: Path, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                yield json.loads(line)
                if limit is not None:
                    limit -= 1
                    if limit <= 0:
                        break
        return

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        records = raw
    elif isinstance(raw, dict):
        records = raw.get("data") or raw.get("cases") or raw.get("examples") or [raw]
    else:
        raise ValueError("JSON root must be an object or list")
    for index, item in enumerate(records):
        if limit is not None and index >= limit:
            break
        if not isinstance(item, dict):
            raise ValueError(f"record {index + 1} is not an object")
        yield item


def convert_record(raw: Dict[str, Any], *, index: int, source: str) -> Dict[str, Any]:
    case_id = str(raw.get("id") or raw.get("case_id") or raw.get("uid") or f"official_{index:05d}")
    metadata = dict(raw.get("metadata") or {})
    metadata.setdefault("source", source)
    metadata.setdefault("scenario", raw.get("scenario") or raw.get("tool_scenario") or "unknown")
    task = str(raw.get("task") or raw.get("instruction") or raw.get("prompt") or raw.get("question") or "")
    trajectory = _convert_trajectory(raw)
    case_payload = {"id": case_id, "task": task, "metadata": metadata, "trajectory": trajectory}
    TrajectoryCase.model_validate(case_payload)
    output = dict(case_payload)
    gold = _convert_gold(raw)
    if gold is not None:
        output["gold"] = gold
    return output


def _convert_trajectory(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(raw.get("trajectory"), list):
        return [_normalize_step(step, index) for index, step in enumerate(raw["trajectory"], start=1)]
    if isinstance(raw.get("steps"), list):
        return [_normalize_step(step, index) for index, step in enumerate(raw["steps"], start=1)]
    if isinstance(raw.get("events"), list):
        return [_normalize_step(step, index) for index, step in enumerate(raw["events"], start=1)]
    if isinstance(raw.get("messages"), list):
        return [_message_to_step(message, index, total=len(raw["messages"])) for index, message in enumerate(raw["messages"], start=1)]
    raise ValueError("missing trajectory, steps, events, or messages")


def _normalize_step(step: Any, index: int) -> Dict[str, Any]:
    if not isinstance(step, dict):
        raise ValueError(f"step {index} is not an object")
    normalized = {
        "step": int(step.get("step") or step.get("index") or index),
        "role": str(step.get("role") or step.get("actor") or "agent"),
        "type": str(step.get("type") or step.get("event_type") or "message"),
        "content": _string_or_none(step.get("content") or step.get("text") or step.get("observation")),
        "tool_name": step.get("tool_name") or step.get("tool") or step.get("name"),
        "arguments": _dict_or_wrapped(step.get("arguments") or step.get("args") or {}),
        "metadata": step.get("metadata") or {},
    }
    if normalized["type"] == "tool" or normalized["role"] == "tool":
        normalized["type"] = "observation"
        normalized["role"] = "tool"
    return normalized


def _message_to_step(message: Any, index: int, total: int) -> Dict[str, Any]:
    if not isinstance(message, dict):
        raise ValueError(f"message {index} is not an object")
    role = str(message.get("role") or "agent")
    content = _string_or_none(message.get("content") or "") or ""
    step_type = "final" if index == total and role in {"assistant", "agent"} else "message"
    return {
        "step": index,
        "role": role,
        "type": step_type,
        "content": content,
        "tool_name": message.get("tool_name") or message.get("tool"),
        "arguments": _dict_or_wrapped(message.get("arguments") or {}),
        "metadata": message.get("metadata") or {},
    }


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _dict_or_wrapped(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"value": value}


def _convert_gold(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if isinstance(raw.get("gold"), dict):
        return report_from_gold(raw["gold"]).model_dump(mode="json", exclude={"cost"})
    label = raw.get("label")
    if label in {"safe", "unsafe"}:
        if label == "safe":
            return {
                "label": "safe",
                "decision": raw.get("decision") or "allow",
                "risk_source": "none",
                "failure_mode": "none",
                "harm_type": "none",
                "evidence_steps": raw.get("evidence_steps") or [],
                "reason": raw.get("reason") or "",
                "confidence": 1.0,
            }
        return {
            "label": "unsafe",
            "decision": raw.get("decision") or "block",
            "risk_source": raw.get("risk_source") or "inherent_agent_failures",
            "failure_mode": raw.get("failure_mode") or "flawed_planning_or_reasoning",
            "harm_type": raw.get("harm_type") or "functional_and_opportunity_harm",
            "evidence_steps": raw.get("evidence_steps") or [],
            "reason": raw.get("reason") or "",
            "confidence": 1.0,
        }
    return None


if __name__ == "__main__":
    main()
