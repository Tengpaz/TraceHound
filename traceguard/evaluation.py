"""Metrics for trajectory safety classification and explanations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from traceguard.judge import ModelAdapter
from traceguard.pipeline import evaluate_case
from traceguard.schema import RiskReport, TrajectoryCase, dump_model, report_from_gold


def load_eval_jsonl(path: str | Path, limit: int | None = None) -> List[Tuple[TrajectoryCase, RiskReport]]:
    rows: List[Tuple[TrajectoryCase, RiskReport]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            case_raw = {key: raw[key] for key in ("id", "task", "metadata", "trajectory") if key in raw}
            if "trajectory" not in case_raw:
                raise ValueError(f"line {line_no}: missing trajectory")
            case = TrajectoryCase.model_validate(case_raw)
            gold_raw = raw.get("gold") or raw.get("label") or {}
            gold = report_from_gold(gold_raw)
            rows.append((case, gold))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def macro_f1(golds: Iterable[str], preds: Iterable[str]) -> float:
    gold_list = list(golds)
    pred_list = list(preds)
    labels = sorted(set(gold_list) | set(pred_list))
    scores: List[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold_list, pred_list) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold_list, pred_list) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold_list, pred_list) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def evidence_hit(gold: RiskReport, pred: RiskReport) -> bool:
    if gold.label == "safe":
        return True
    return bool(set(gold.evidence_steps) & set(pred.evidence_steps))


def evaluate_dataset(
    path: str | Path,
    mode: str = "layered",
    judge: ModelAdapter | None = None,
    limit: int | None = None,
) -> Dict[str, Any]:
    rows = load_eval_jsonl(path, limit=limit)
    predictions = [evaluate_case(case, mode=mode, judge=judge) for case, _ in rows]
    golds = [gold for _, gold in rows]
    total = len(rows)
    correct = sum(1 for gold, pred in zip(golds, predictions) if gold.label == pred.label)
    unsafe_total = sum(1 for gold in golds if gold.label == "unsafe")
    unsafe_correct = sum(1 for gold, pred in zip(golds, predictions) if gold.label == "unsafe" and pred.label == "unsafe")
    safe_pred_total = sum(1 for pred in predictions if pred.label == "safe")
    safe_correct = sum(1 for gold, pred in zip(golds, predictions) if gold.label == "safe" and pred.label == "safe")
    metrics = {
        "samples": total,
        "accuracy": correct / total if total else 0.0,
        "unsafe_recall": unsafe_correct / unsafe_total if unsafe_total else 0.0,
        "safe_precision": safe_correct / safe_pred_total if safe_pred_total else 0.0,
        "macro_f1": macro_f1((g.label for g in golds), (p.label for p in predictions)),
        "risk_source_macro_f1": macro_f1((g.risk_source for g in golds), (p.risk_source for p in predictions)),
        "failure_mode_macro_f1": macro_f1((g.failure_mode for g in golds), (p.failure_mode for p in predictions)),
        "harm_type_macro_f1": macro_f1((g.harm_type for g in golds), (p.harm_type for p in predictions)),
        "evidence_hit_rate": sum(1 for g, p in zip(golds, predictions) if evidence_hit(g, p)) / total if total else 0.0,
        "average_input_tokens": sum(p.cost.input_tokens for p in predictions) / total if total else 0.0,
        "average_output_tokens": sum(p.cost.output_tokens for p in predictions) / total if total else 0.0,
        "average_latency_ms": sum(p.cost.latency_ms for p in predictions) / total if total else 0.0,
        "average_model_calls": sum(p.cost.model_calls for p in predictions) / total if total else 0.0,
        "average_compression_ratio": sum(p.cost.compression_ratio for p in predictions) / total if total else 0.0,
        "average_cost_reduction_ratio": sum(p.cost.cost_reduction_ratio for p in predictions) / total if total else 0.0,
    }
    return {
        "metrics": {key: round(value, 4) if isinstance(value, float) else value for key, value in metrics.items()},
        "predictions": [dump_model(pred) for pred in predictions],
    }
