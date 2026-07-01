"""Coverage and split utilities for generated TraceHound datasets."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from traceguard.data import TOOL_SCENARIOS, dataset_summary
from traceguard.taxonomy import FAILURE_MODES, HARM_TYPES, RISK_SOURCES

SPLIT_NAMES = ("train", "eval", "test")


def coverage_matrix(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return AgentDoG-axis coverage counters and pairwise matrices."""

    labels = Counter(_gold(case).get("label", "unknown") for case in cases)
    scenarios = Counter(_scenario(case) for case in cases)
    unsafe = [case for case in cases if _gold(case).get("label") == "unsafe"]
    risk_counts = Counter(_gold(case).get("risk_source", "unknown") for case in unsafe)
    failure_counts = Counter(_gold(case).get("failure_mode", "unknown") for case in unsafe)
    harm_counts = Counter(_gold(case).get("harm_type", "unknown") for case in unsafe)
    triples = Counter(
        _taxonomy_triple_key(
            _gold(case).get("risk_source", "unknown"),
            _gold(case).get("failure_mode", "unknown"),
            _gold(case).get("harm_type", "unknown"),
        )
        for case in unsafe
    )
    return {
        "samples": len(cases),
        "unsafe_samples": len(unsafe),
        "labels": dict(labels),
        "scenarios": _ordered_counts(TOOL_SCENARIOS, scenarios),
        "risk_sources": _ordered_counts(RISK_SOURCES, risk_counts),
        "failure_modes": _ordered_counts(FAILURE_MODES, failure_counts),
        "harm_types": _ordered_counts(HARM_TYPES, harm_counts),
        "coverage": {
            "tool_scenarios": _coverage_stat(TOOL_SCENARIOS, scenarios),
            "risk_sources": _coverage_stat(RISK_SOURCES, risk_counts),
            "failure_modes": _coverage_stat(FAILURE_MODES, failure_counts),
            "harm_types": _coverage_stat(HARM_TYPES, harm_counts),
            "taxonomy_triples": {
                "present": len(triples),
                "total": len(RISK_SOURCES) * len(FAILURE_MODES) * len(HARM_TYPES),
                "ratio": _safe_ratio(len(triples), len(RISK_SOURCES) * len(FAILURE_MODES) * len(HARM_TYPES)),
            },
        },
        "missing": {
            "tool_scenarios": _missing(TOOL_SCENARIOS, scenarios),
            "risk_sources": _missing(RISK_SOURCES, risk_counts),
            "failure_modes": _missing(FAILURE_MODES, failure_counts),
            "harm_types": _missing(HARM_TYPES, harm_counts),
        },
        "matrices": {
            "risk_source_by_failure_mode": _pair_matrix(
                unsafe,
                RISK_SOURCES,
                FAILURE_MODES,
                row_key="risk_source",
                col_key="failure_mode",
            ),
            "risk_source_by_harm_type": _pair_matrix(
                unsafe,
                RISK_SOURCES,
                HARM_TYPES,
                row_key="risk_source",
                col_key="harm_type",
            ),
            "failure_mode_by_harm_type": _pair_matrix(
                unsafe,
                FAILURE_MODES,
                HARM_TYPES,
                row_key="failure_mode",
                col_key="harm_type",
            ),
        },
        "taxonomy_triples": dict(sorted(triples.items())),
    }


def split_cases(
    cases: Sequence[Dict[str, Any]],
    *,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 20260701,
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministically split cases by label and tool scenario strata."""

    ratios = _normalize_ratios(train_ratio, eval_ratio, test_ratio)
    groups: dict[tuple[str, str], list[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        gold = _gold(case)
        groups[(str(gold.get("label", "unknown")), _scenario(case))].append(case)

    splits: Dict[str, List[Dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda case: _stable_hash(seed, str(case.get("id", ""))))
        counts = _split_counts(len(group), ratios)
        cursor = 0
        for split_name in SPLIT_NAMES:
            next_cursor = cursor + counts[split_name]
            splits[split_name].extend(group[cursor:next_cursor])
            cursor = next_cursor

    for split_name in SPLIT_NAMES:
        splits[split_name].sort(key=lambda case: str(case.get("id", "")))
    return splits


def split_summary(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    train_ratio: float = 0.8,
    eval_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 20260701,
) -> Dict[str, Any]:
    total = sum(len(rows) for rows in splits.values())
    return {
        "seed": seed,
        "requested_ratios": {
            "train": train_ratio,
            "eval": eval_ratio,
            "test": test_ratio,
        },
        "total": total,
        "counts": {name: len(splits.get(name, [])) for name in SPLIT_NAMES},
        "actual_ratios": {
            name: _safe_ratio(len(splits.get(name, [])), total)
            for name in SPLIT_NAMES
        },
        "summaries": {
            name: dataset_summary(list(splits.get(name, [])))
            for name in SPLIT_NAMES
        },
    }


def _gold(case: Mapping[str, Any]) -> Mapping[str, Any]:
    gold = case.get("gold")
    return gold if isinstance(gold, Mapping) else {}


def _scenario(case: Mapping[str, Any]) -> str:
    metadata = case.get("metadata")
    if isinstance(metadata, Mapping):
        return str(metadata.get("scenario", "unknown"))
    return "unknown"


def _ordered_counts(labels: Iterable[str], counts: Counter) -> Dict[str, int]:
    result = {label: int(counts.get(label, 0)) for label in labels}
    extras = sorted(str(label) for label in counts if label not in result)
    result.update({label: int(counts.get(label, 0)) for label in extras})
    return result


def _coverage_stat(labels: Sequence[str], counts: Counter) -> Dict[str, Any]:
    present = sum(1 for label in labels if counts.get(label, 0) > 0)
    return {
        "present": present,
        "total": len(labels),
        "ratio": _safe_ratio(present, len(labels)),
    }


def _missing(labels: Sequence[str], counts: Counter) -> List[str]:
    return [label for label in labels if counts.get(label, 0) == 0]


def _pair_matrix(
    cases: Sequence[Mapping[str, Any]],
    rows: Sequence[str],
    cols: Sequence[str],
    *,
    row_key: str,
    col_key: str,
) -> Dict[str, Dict[str, int]]:
    matrix = {row: {col: 0 for col in cols} for row in rows}
    for case in cases:
        gold = _gold(case)
        row = str(gold.get(row_key, "unknown"))
        col = str(gold.get(col_key, "unknown"))
        if row not in matrix:
            matrix[row] = {known_col: 0 for known_col in cols}
        if col not in matrix[row]:
            matrix[row][col] = 0
        matrix[row][col] += 1
    return matrix


def _taxonomy_triple_key(risk_source: str, failure_mode: str, harm_type: str) -> str:
    return f"{risk_source}|{failure_mode}|{harm_type}"


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _normalize_ratios(train_ratio: float, eval_ratio: float, test_ratio: float) -> Dict[str, float]:
    values = {
        "train": max(float(train_ratio), 0.0),
        "eval": max(float(eval_ratio), 0.0),
        "test": max(float(test_ratio), 0.0),
    }
    total = sum(values.values())
    if total <= 0:
        raise ValueError("at least one split ratio must be positive")
    return {name: value / total for name, value in values.items()}


def _split_counts(size: int, ratios: Mapping[str, float]) -> Dict[str, int]:
    if size <= 0:
        return {name: 0 for name in SPLIT_NAMES}
    if size == 1:
        return {"train": 1, "eval": 0, "test": 0}
    if size == 2:
        return {"train": 1, "eval": 0, "test": 1}

    raw = {name: size * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(raw[name]) for name in SPLIT_NAMES}
    remainder = size - sum(counts.values())
    for name in sorted(SPLIT_NAMES, key=lambda split: raw[split] - counts[split], reverse=True):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1

    for name in SPLIT_NAMES:
        if ratios[name] > 0 and counts[name] == 0:
            donor = max(SPLIT_NAMES, key=lambda split: counts[split])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[name] += 1
    return counts


def _stable_hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
