#!/usr/bin/env python
"""Run standard TraceHound ablations and write a compact JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.config import load_env_file
from traceguard.evaluation import evaluate_dataset, evaluate_final_answer_only_dataset, load_eval_jsonl
from traceguard.guard import TraceGuard
from traceguard.judge import build_remote_judge
from traceguard.schema import RiskReport, TrajectoryCase, TrajectoryStep, dump_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_eval.jsonl", help="Evaluation JSONL path.")
    parser.add_argument("--output", default="reports/experiments.json", help="JSON report path.")
    parser.add_argument("--no-api", action="store_true", help="Skip API and hybrid API experiments.")
    parser.add_argument("--require-api", action="store_true", help="Fail if API experiments cannot run.")
    parser.add_argument("--api-limit", type=int, default=1, help="Number of rows for API smoke experiments.")
    parser.add_argument("--env-file", default=".env", help="Optional env file for API settings.")
    parser.add_argument("--api-base", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", help="API key. Prefer TRACEHOUND_API_KEY or .env for local use.")
    parser.add_argument("--model", help="Remote model name.")
    parser.add_argument("--api-path", help="API path. Defaults to /chat/completions.")
    parser.add_argument("--timeout", type=int, help="API timeout in seconds.")
    parser.add_argument("--prompt-mode", default="compressed", choices=["compressed", "full"])
    parser.add_argument("--include-predictions", action="store_true", help="Keep per-sample predictions in the JSON report.")
    args = parser.parse_args()

    experiments: List[Dict[str, Any]] = []
    experiments.append(_run_eval("rules", args.data, mode="rules", judge_name="heuristic", include_predictions=args.include_predictions))
    experiments.append(
        _run_eval("compressed", args.data, mode="compressed", judge_name="heuristic", include_predictions=args.include_predictions)
    )
    experiments.append(
        _run_eval("layered", args.data, mode="layered", judge_name="heuristic", include_predictions=args.include_predictions)
    )
    experiments.append(
        _run_eval("no_rules", args.data, mode="compressed", judge_name="heuristic", include_predictions=args.include_predictions)
    )
    experiments.append(
        _run_eval(
            "final_only",
            args.data,
            mode="layered",
            judge_name="heuristic",
            final_only=True,
            include_predictions=args.include_predictions,
        )
    )

    skipped: List[str] = []
    if args.no_api:
        skipped.extend(["hybrid", "api"])
    else:
        load_env_file(args.env_file)
        try:
            hybrid = build_remote_judge(
                judge="hybrid",
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                api_path=args.api_path,
                timeout=args.timeout,
                prompt_mode=args.prompt_mode,
            )
            api = build_remote_judge(
                judge="api",
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                api_path=args.api_path,
                timeout=args.timeout,
                prompt_mode=args.prompt_mode,
            )
            experiments.append(
                _run_eval(
                    "hybrid",
                    args.data,
                    mode="layered",
                    judge_name="hybrid",
                    judge=hybrid,
                    limit=args.api_limit,
                    include_predictions=args.include_predictions,
                )
            )
            experiments.append(
                _run_eval(
                    "api",
                    args.data,
                    mode="compressed",
                    judge_name="api",
                    judge=api,
                    limit=args.api_limit,
                    include_predictions=args.include_predictions,
                )
            )
        except (RuntimeError, ValueError) as exc:
            if args.require_api:
                raise SystemExit(str(exc)) from exc
            skipped.extend(["hybrid", "api"])
            experiments.append({"name": "api_setup", "skipped": True, "error": str(exc)})

    rows = load_eval_jsonl(args.data)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": args.data,
        "experiments": experiments,
        "online_guard": online_guard_metrics(rows),
        "skipped": skipped,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def _run_eval(
    name: str,
    data: str,
    *,
    mode: str,
    judge_name: str,
    judge: Any = None,
    limit: int | None = None,
    final_only: bool = False,
    include_predictions: bool = False,
) -> Dict[str, Any]:
    if final_only:
        result = evaluate_final_answer_only_dataset(data, mode=mode, judge=judge, limit=limit)
    else:
        result = evaluate_dataset(data, mode=mode, judge=judge, limit=limit)
    if not include_predictions:
        result.pop("predictions", None)
    return {
        "name": name,
        "mode": mode,
        "judge": judge_name,
        "limit": limit,
        "final_only": final_only,
        **result,
    }


def online_guard_metrics(rows: List[Tuple[TrajectoryCase, RiskReport]]) -> Dict[str, Any]:
    decisions = [_first_online_decision(case) for case, _ in rows]
    golds = [gold for _, gold in rows]
    total = len(rows)
    unsafe_total = sum(1 for gold in golds if gold.label == "unsafe")
    safe_total = sum(1 for gold in golds if gold.label == "safe")
    interventions = [decision for decision, _ in decisions if decision in {"block", "ask_confirm", "sanitize"}]
    unsafe_interventions = sum(
        1
        for gold, (decision, _) in zip(golds, decisions)
        if gold.label == "unsafe" and decision in {"block", "ask_confirm", "sanitize"}
    )
    safe_false_blocks = sum(1 for gold, (decision, _) in zip(golds, decisions) if gold.label == "safe" and decision == "block")
    decision_steps = [step for _, step in decisions if step is not None]
    return {
        "samples": total,
        "intervention_rate": round(len(interventions) / total, 4) if total else 0.0,
        "unsafe_intervention_recall": round(unsafe_interventions / unsafe_total, 4) if unsafe_total else 0.0,
        "safe_false_block_rate": round(safe_false_blocks / safe_total, 4) if safe_total else 0.0,
        "average_decision_step": round(sum(decision_steps) / len(decision_steps), 4) if decision_steps else 0.0,
        "decisions": {decision: sum(1 for item, _ in decisions if item == decision) for decision in sorted({item for item, _ in decisions})},
    }


def _first_online_decision(case: TrajectoryCase) -> Tuple[str, int | None]:
    guard = TraceGuard()
    for index, step in enumerate(case.trajectory):
        prefix = _prefix_case(case, index)
        if step.type == "tool_call":
            decision = guard.before_tool_call(prefix, step)
            if decision.decision in {"block", "ask_confirm"}:
                return decision.decision, step.step
        if step.type == "observation":
            decision = guard.after_tool_observation(prefix, TrajectoryStep.model_validate(dump_model(step)))
            if decision.decision in {"block", "ask_confirm", "sanitize"}:
                return decision.decision, step.step
    return "allow", None


def _prefix_case(case: TrajectoryCase, end_index: int) -> TrajectoryCase:
    return TrajectoryCase.model_validate(
        {
            "id": case.id,
            "task": case.task,
            "metadata": case.metadata,
            "trajectory": [dump_model(step) for step in case.trajectory[:end_index]],
        }
    )


if __name__ == "__main__":
    main()
