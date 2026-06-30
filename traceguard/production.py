"""Production dataset selection helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Sequence


REPAIR_LEVEL_RANK = {
    "none": 0,
    "structural": 1,
    "semantic": 2,
}


def repair_level(case: Dict[str, Any]) -> str:
    value = str((case.get("metadata") or {}).get("repair_level") or "none")
    return value if value in REPAIR_LEVEL_RANK else "semantic"


def filter_cases_for_training(
    cases: Sequence[Dict[str, Any]],
    *,
    max_repair_level: str = "structural",
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    if max_repair_level not in REPAIR_LEVEL_RANK:
        raise ValueError("max_repair_level must be none, structural, or semantic")
    threshold = REPAIR_LEVEL_RANK[max_repair_level]
    kept = [case for case in cases if REPAIR_LEVEL_RANK[repair_level(case)] <= threshold]
    rejected = [
        {
            "id": case.get("id"),
            "repair_level": repair_level(case),
            "repair_log": (case.get("metadata") or {}).get("repair_log", [])[:8],
        }
        for case in cases
        if REPAIR_LEVEL_RANK[repair_level(case)] > threshold
    ]
    return kept, {
        "enabled": True,
        "max_repair_level": max_repair_level,
        "input": len(cases),
        "kept": len(kept),
        "rejected": len(rejected),
        "pass_rate": round(len(kept) / len(cases), 4) if cases else 0.0,
        "rejected_samples": rejected[:50],
    }


def production_quality_summary(
    cases: Sequence[Dict[str, Any]],
    *,
    training_cases: Sequence[Dict[str, Any]] | None = None,
    training_max_repair_level: str = "structural",
) -> Dict[str, Any]:
    levels = Counter(repair_level(case) for case in cases)
    raw_passed = sum(1 for case in cases if ((case.get("metadata") or {}).get("raw_agentdog_qc") or {}).get("passed") is True)
    raw_available = sum(1 for case in cases if "raw_agentdog_qc" in (case.get("metadata") or {}))
    repair_count = sum(1 for case in cases if repair_level(case) != "none")
    semantic_count = levels.get("semantic", 0)
    training_count = len(training_cases) if training_cases is not None else None
    return {
        "raw_qc": {
            "available": raw_available,
            "passed": raw_passed,
            "pass_rate": round(raw_passed / raw_available, 4) if raw_available else None,
        },
        "repair": {
            "counts": dict(levels),
            "repair_rate": round(repair_count / len(cases), 4) if cases else 0.0,
            "semantic_repair_rate": round(semantic_count / len(cases), 4) if cases else 0.0,
        },
        "training_selection": {
            "max_repair_level": training_max_repair_level,
            "eligible": training_count,
            "input": len(cases),
            "pass_rate": round(training_count / len(cases), 4) if cases and training_count is not None else None,
        },
    }
