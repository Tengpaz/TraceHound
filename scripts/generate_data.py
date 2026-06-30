#!/usr/bin/env python
"""Generate synthetic TraceHound data files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.data import built_in_cases, dataset_summary
from traceguard.config import load_env_file
from traceguard.export import (
    agentdog15_coarse_sft_rows,
    agentdog15_unified_sft_rows,
    agentdog_binary_sft_rows,
    agentdog_taxonomy_sft_rows,
    eval_rows,
    preference_rows,
    rl_rows,
    sft_rows,
    write_jsonl,
)
from traceguard.generation_config import load_generation_config
from traceguard.judge import build_remote_judge
from traceguard.llm_generation import llm_synthesize_cases
from traceguard.pipeline import evaluate_case
from traceguard.production import filter_cases_for_training, production_quality_summary
from traceguard.quality import filter_cases_by_agentdog_qc
from traceguard.schema import TrajectoryCase, report_from_gold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML config, e.g. configs/generation.yaml.")
    parser.add_argument("--env-file", default=".env", help="Environment file for optional LLM generation/QC.")
    parser.add_argument("--out", help="Output directory. Overrides config.")
    parser.add_argument("--scenario", action="append", help="Filter by scenario. May be repeated.")
    parser.add_argument("--label", action="append", choices=["safe", "unsafe"], help="Filter by gold label. May be repeated.")
    parser.add_argument("--limit", type=int, help="Limit number of cases.")
    parser.add_argument("--count", type=int, help="Generate this many cases with AgentDoG-style taxonomy sampling.")
    parser.add_argument("--generation-backend", choices=["deterministic", "llm"], help="Stage 2 trajectory synthesis backend.")
    parser.add_argument("--agentdog-llm-generate", action="store_true", help="Use LLM-based AgentDoG Stage 2 trajectory synthesis.")
    parser.add_argument("--llm-generation-retries", type=int, help="Retry count for each LLM-generated trajectory.")
    parser.add_argument("--llm-generation-temperature", type=float, help="Temperature for LLM trajectory synthesis.")
    parser.add_argument(
        "--semantic-repair-backend",
        choices=["none", "static", "llm", "llm_then_static"],
        help="Semantic repair backend after deterministic QC failure.",
    )
    parser.add_argument("--semantic-repair-rounds", type=int, help="LLM self-repair rounds for failed trajectories.")
    parser.add_argument("--api-base", help="Override TRACEHOUND_API_BASE for LLM generation/QC.")
    parser.add_argument("--api-key", help="Override TRACEHOUND_API_KEY for LLM generation/QC.")
    parser.add_argument("--model", help="Override TRACEHOUND_MODEL for LLM generation/QC.")
    parser.add_argument("--api-path", help="Override TRACEHOUND_API_PATH for LLM generation/QC.")
    parser.add_argument("--timeout", type=int, help="Override TRACEHOUND_API_TIMEOUT for LLM generation/QC.")
    parser.add_argument("--include-rl", action="store_true", help="Also write synthetic_rl.jsonl for DPO/GRPO-style training.")
    parser.add_argument("--llm-qc", action="store_true", help="Run optional API judge semantic QC and filter mismatches.")
    parser.add_argument("--llm-qc-judge", action="append", help="Optional repeated judge spec, e.g. api or hybrid:model.")
    parser.add_argument(
        "--agentdog-strict-qc",
        action="store_true",
        help="Run AgentDoG-style deterministic QC plus required LLM semantic QC.",
    )
    parser.add_argument(
        "--agentdog-local-qc",
        action="store_true",
        help="Run deterministic AgentDoG-style QC only, overriding qc_policy/llm_qc.",
    )
    parser.add_argument("--write-examples", action="store_true", help="Refresh examples/demo_cases.json.")
    args = parser.parse_args()

    load_env_file(args.env_file)
    config = load_generation_config(args.config)
    scenarios = args.scenario if args.scenario is not None else config["scenarios"]
    labels = args.label if args.label is not None else config["labels"]
    limit = args.limit if args.limit is not None else config["limit"]
    count = args.count if args.count is not None else config["count"]
    generation_backend = args.generation_backend or str(config.get("generation_backend") or "deterministic")
    if args.agentdog_llm_generate:
        generation_backend = "llm"
    if generation_backend not in {"deterministic", "llm"}:
        raise SystemExit("generation_backend must be deterministic or llm")
    include_eval = bool(config["include_eval"])
    include_sft = bool(config["include_sft"])
    include_agentdog_sft = bool(config["include_agentdog_sft"])
    include_agentdog15_official_sft = bool(config.get("include_agentdog15_official_sft", True))
    include_preference = bool(config["include_preference"])
    include_rl = bool(args.include_rl or config["include_rl"])
    qc_policy = str(config.get("qc_policy") or "agentdog_local")
    if args.agentdog_strict_qc:
        qc_policy = "agentdog_strict"
    if args.agentdog_local_qc:
        qc_policy = "agentdog_local"
    if qc_policy not in {"agentdog_local", "agentdog_strict"}:
        raise SystemExit("qc_policy must be agentdog_local or agentdog_strict")
    llm_qc = bool(args.llm_qc or config["llm_qc"] or qc_policy == "agentdog_strict")
    if args.agentdog_local_qc:
        llm_qc = False
    write_examples = bool(args.write_examples or config["write_examples"])
    qc_min_score = float(config["qc_min_score"])
    training_max_repair_level = str(config.get("training_max_repair_level") or "structural")
    semantic_repair_backend = args.semantic_repair_backend or str(config.get("semantic_repair_backend") or "static")
    semantic_repair_rounds = (
        args.semantic_repair_rounds
        if args.semantic_repair_rounds is not None
        else int(config.get("semantic_repair_rounds") or 1)
    )
    write_qc_report = bool(config["write_qc_report"])
    if (
        not include_eval
        and not include_sft
        and not include_agentdog_sft
        and not include_agentdog15_official_sft
        and not include_preference
        and not include_rl
    ):
        raise SystemExit("at least one dataset type must be enabled")

    planner_cases = built_in_cases(scenarios=scenarios, labels=labels, limit=limit, count=count)
    llm_generation_summary: dict[str, Any] = {"enabled": False, "backend": "deterministic"}
    if generation_backend == "llm":
        generation = llm_synthesize_cases(
            planner_cases,
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            api_path=args.api_path,
            timeout=args.timeout,
            max_retries=args.llm_generation_retries
            if args.llm_generation_retries is not None
            else int(config["llm_generation_retries"]),
            temperature=args.llm_generation_temperature
            if args.llm_generation_temperature is not None
            else float(config["llm_generation_temperature"]),
            semantic_repair_backend=semantic_repair_backend,
            semantic_repair_rounds=semantic_repair_rounds,
        )
        cases = generation.cases
        llm_generation_summary = generation.summary
    else:
        cases = planner_cases
    cases, deterministic_qc = filter_cases_by_agentdog_qc(cases, min_quality_score=qc_min_score)
    llm_qc_summary: dict[str, Any] = {"enabled": False}
    if llm_qc:
        judge_specs = args.llm_qc_judge or config.get("llm_qc_judges") or [config["llm_qc_judge"]]
        cases, llm_qc_summary = _apply_llm_qc(
            cases,
            judge_specs=[str(item) for item in judge_specs],
            mode=str(config["llm_qc_mode"]),
            consensus_threshold=float(config["llm_qc_consensus_threshold"]),
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            api_path=args.api_path,
            timeout=args.timeout,
        )
    training_cases, training_filter = filter_cases_for_training(cases, max_repair_level=training_max_repair_level)
    production_summary = production_quality_summary(
        cases,
        training_cases=training_cases,
        training_max_repair_level=training_max_repair_level,
    )
    out = Path(args.out or config["out"] or "data")
    out.mkdir(parents=True, exist_ok=True)
    qc_payload = {
        "policy": qc_policy,
        "generation": llm_generation_summary,
        "deterministic": deterministic_qc,
        "llm": llm_qc_summary,
        "production": production_summary,
        "training_filter": training_filter,
    }
    if write_qc_report:
        (out / "quality_report.json").write_text(
            json.dumps(qc_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rejected_rows = deterministic_qc.get("rejected_samples", []) + llm_qc_summary.get("rejected_samples", [])
        if rejected_rows:
            with (out / "rejected_samples.jsonl").open("w", encoding="utf-8") as handle:
                for item in rejected_rows:
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            rejected_path = out / "rejected_samples.jsonl"
            if rejected_path.exists():
                rejected_path.unlink()
        training_rejected = training_filter.get("rejected_samples") or []
        if training_rejected:
            with (out / "training_rejected_samples.jsonl").open("w", encoding="utf-8") as handle:
                for item in training_rejected:
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        else:
            training_rejected_path = out / "training_rejected_samples.jsonl"
            if training_rejected_path.exists():
                training_rejected_path.unlink()
    if not cases:
        raise SystemExit(
            "AgentDoG QC filtered out all generated cases. "
            f"Diagnostics written to {out / 'quality_report.json'} and {out / 'rejected_samples.jsonl'}"
        )
    counts = {}
    if include_eval:
        counts["synthetic_eval.jsonl"] = write_jsonl(out / "synthetic_eval.jsonl", eval_rows(cases))
    if include_sft:
        counts["synthetic_sft.jsonl"] = write_jsonl(out / "synthetic_sft.jsonl", sft_rows(training_cases))
    if include_agentdog_sft:
        counts["agentdog_binary_sft.jsonl"] = write_jsonl(
            out / "agentdog_binary_sft.jsonl",
            agentdog_binary_sft_rows(training_cases),
        )
        counts["agentdog_taxonomy_sft.jsonl"] = write_jsonl(
            out / "agentdog_taxonomy_sft.jsonl",
            agentdog_taxonomy_sft_rows(training_cases),
        )
    if include_agentdog15_official_sft:
        counts["agentdog15_unified_sft.jsonl"] = write_jsonl(
            out / "agentdog15_unified_sft.jsonl",
            agentdog15_unified_sft_rows(training_cases),
        )
        counts["agentdog15_coarse_sft.jsonl"] = write_jsonl(
            out / "agentdog15_coarse_sft.jsonl",
            agentdog15_coarse_sft_rows(training_cases),
        )
    if include_preference:
        counts["synthetic_preference.jsonl"] = write_jsonl(out / "synthetic_preference.jsonl", preference_rows(training_cases))
    if include_rl:
        counts["synthetic_rl.jsonl"] = write_jsonl(out / "synthetic_rl.jsonl", rl_rows(training_cases))
    if write_examples:
        examples = Path("examples")
        examples.mkdir(parents=True, exist_ok=True)
        demo_cases = built_in_cases()
        (examples / "demo_cases.json").write_text(json.dumps(demo_cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "counts": counts,
                "summary": dataset_summary(cases),
                "training_summary": dataset_summary(training_cases),
                "qc": qc_payload,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _apply_llm_qc(
    cases: list[dict],
    *,
    judge_specs: list[str],
    mode: str,
    consensus_threshold: float,
    api_base: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    api_path: str | None = None,
    timeout: int | None = None,
) -> tuple[list[dict], dict]:
    judges = [
        _build_judge_from_spec(
            spec,
            api_base=api_base,
            api_key=api_key,
            model_override=model,
            api_path=api_path,
            timeout=timeout,
        )
        for spec in judge_specs
    ]
    kept = []
    rejected = []
    for case in cases:
        gold = report_from_gold(case["gold"])
        votes = []
        for judge_meta in judges:
            report = evaluate_case(TrajectoryCase.model_validate(case), mode=mode, judge=judge_meta["judge"])
            votes.append({"spec": judge_meta["spec"], "report": report})
        aggregate = _aggregate_votes(votes)
        consensus = float(aggregate.get("consensus", 0.0))
        aggregate_report = aggregate.get("report")
        if aggregate_report is not None and consensus >= consensus_threshold and _qc_matches(gold, aggregate_report):
            case.setdefault("metadata", {}).setdefault("qc", {})["llm_judge"] = {
                "judges": judge_specs,
                "mode": mode,
                "consensus": consensus,
                "matched": True,
            }
            kept.append(case)
        else:
            rejected.append(
                {
                    "id": case.get("id"),
                    "gold": gold.model_dump(mode="json"),
                    "aggregate": aggregate_report.model_dump(mode="json") if aggregate_report else {},
                    "consensus": consensus,
                    "votes": [
                        {"spec": vote["spec"], "report": vote["report"].model_dump(mode="json")}
                        for vote in votes
                    ],
                }
            )
    return kept, {
        "enabled": True,
        "judges": judge_specs,
        "mode": mode,
        "consensus_threshold": consensus_threshold,
        "generated": len(cases),
        "kept": len(kept),
        "rejected": len(rejected),
        "pass_rate": round(len(kept) / len(cases), 4) if cases else 0.0,
        "rejected_samples": rejected,
    }


def _build_judge_from_spec(
    spec: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    model_override: str | None = None,
    api_path: str | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    parts = spec.split(":", 1)
    judge_name = parts[0]
    model = parts[1] if len(parts) == 2 and parts[1] else None
    if judge_name not in {"api", "hybrid"}:
        raise SystemExit("llm_qc judge specs must start with api or hybrid")
    return {
        "spec": spec,
        "judge": build_remote_judge(
            judge=judge_name,
            api_base=api_base,
            api_key=api_key,
            model=model_override or model,
            api_path=api_path,
            timeout=timeout,
            prompt_mode="compressed",
        ),
    }


def _aggregate_votes(votes: list[dict[str, Any]]) -> dict[str, Any]:
    if not votes:
        return {"report": None, "consensus": 0.0}
    labels = Counter(vote["report"].label for vote in votes)
    label, label_count = labels.most_common(1)[0]
    candidates = [vote["report"] for vote in votes if vote["report"].label == label]
    if label == "safe":
        report = candidates[0]
    else:
        report = _aggregate_unsafe_report(candidates)
    return {"report": report, "consensus": label_count / len(votes)}


def _aggregate_unsafe_report(reports: list[Any]) -> Any:
    base = reports[0].model_copy(deep=True)
    for field in ("risk_source", "failure_mode", "harm_type"):
        values = Counter(getattr(report, field) for report in reports)
        setattr(base, field, values.most_common(1)[0][0])
    evidence = sorted({step for report in reports for step in report.evidence_steps})
    base.evidence_steps = evidence
    base.confidence = round(sum(report.confidence for report in reports) / len(reports), 4)
    return base


def _qc_matches(gold, report) -> bool:
    if gold.label != report.label:
        return False
    if gold.label == "safe":
        return True
    return (
        gold.risk_source == report.risk_source
        and gold.failure_mode == report.failure_mode
        and gold.harm_type == report.harm_type
    )


if __name__ == "__main__":
    main()
