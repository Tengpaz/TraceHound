"""Evaluation pipeline with rules, compression, judge, and cost accounting."""

from __future__ import annotations

from traceguard.compressor import compressed_payload
from traceguard.cost import build_cost, measure_latency
from traceguard.judge import HeuristicJudge, ModelAdapter
from traceguard.schema import RiskReport, TrajectoryCase, dump_model


def evaluate_case(
    case: TrajectoryCase,
    *,
    mode: str = "layered",
    judge: ModelAdapter | None = None,
) -> RiskReport:
    if mode not in {"rules", "compressed", "layered"}:
        raise ValueError("mode must be one of: rules, compressed, layered")

    active_judge = judge or HeuristicJudge()
    full_payload = dump_model(case)
    compressed = compressed_payload(case)

    with measure_latency() as latency:
        report = active_judge.evaluate(case, mode=mode)

    model_calls = report.cost.model_calls
    early_exit = mode in {"rules", "layered"} and model_calls == 0 and report.confidence >= 0.8
    report.cost = build_cost(
        full_input=full_payload,
        compressed_input=compressed if mode != "rules" else full_payload,
        output=dump_model(report),
        latency_ms=latency["latency_ms"],
        model_calls=model_calls,
        strategy=report.cost.strategy if report.cost.strategy != "none" else mode,
        early_exit=early_exit,
    )
    return report
