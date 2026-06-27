"""Dataset quality checks for synthetic and official trajectory data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pydantic import ValidationError

from traceguard.compressor import compress_trajectory
from traceguard.data import TOOL_SCENARIOS
from traceguard.schema import TrajectoryCase, report_from_gold


def quality_check_jsonl(path: str | Path) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    cases = []
    labels: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON: {exc}")
                continue
            try:
                case_raw = {key: raw[key] for key in ("id", "task", "metadata", "trajectory") if key in raw}
                case = TrajectoryCase.model_validate(case_raw)
                gold = report_from_gold(raw.get("gold") or raw.get("label") or {})
            except (KeyError, ValidationError, ValueError) as exc:
                errors.append(f"line {line_no}: schema error: {exc}")
                continue

            cases.append(case)
            labels[gold.label] += 1
            scenario = str(case.metadata.get("scenario", "unknown"))
            scenarios[scenario] += 1

            steps = [step.step for step in case.trajectory]
            expected_steps = list(range(1, len(steps) + 1))
            if steps != expected_steps:
                errors.append(f"{case.id}: steps must be continuous from 1; got {steps}")

            known_steps = set(steps)
            missing_evidence = [step for step in gold.evidence_steps if step not in known_steps]
            if missing_evidence:
                errors.append(f"{case.id}: evidence_steps not present in trajectory: {missing_evidence}")
            if gold.label == "unsafe" and not gold.evidence_steps:
                errors.append(f"{case.id}: unsafe samples must include evidence_steps")
            if gold.label == "unsafe" and (
                gold.risk_source == "none" or gold.failure_mode == "none" or gold.harm_type == "none"
            ):
                errors.append(f"{case.id}: unsafe samples must use non-none taxonomy labels")
            if gold.label == "safe" and (
                gold.risk_source != "none" or gold.failure_mode != "none" or gold.harm_type != "none"
            ):
                errors.append(f"{case.id}: safe samples must use none taxonomy labels")

            compressed = compress_trajectory(case)
            if gold.label == "unsafe" and not _has_observable_risk(case, compressed.evidence_candidates):
                warnings.append(f"{case.id}: unsafe sample has weak observable risk signals")

    missing_scenarios = [scenario for scenario in TOOL_SCENARIOS if scenario not in scenarios]
    if missing_scenarios:
        warnings.append(f"missing tool scenarios: {', '.join(missing_scenarios)}")
    if labels and labels["safe"] != labels["unsafe"]:
        warnings.append(f"label distribution is imbalanced: {dict(labels)}")

    return {
        "passed": not errors,
        "samples": len(cases),
        "errors": errors,
        "warnings": warnings,
        "labels": dict(labels),
        "scenarios": dict(scenarios),
    }


def _has_observable_risk(case: TrajectoryCase, evidence_candidates: Sequence[int]) -> bool:
    if evidence_candidates:
        return True
    risky_tokens = (
        "delete",
        "drop table",
        "curl ",
        "| sh",
        "api_key",
        "token",
        "secret",
        "password",
        "ignore previous",
        "ignore prior",
        "system override",
    )
    text = " ".join(step.text().lower() for step in case.trajectory)
    return any(token in text for token in risky_tokens)

