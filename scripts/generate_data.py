#!/usr/bin/env python
"""Generate synthetic TraceHound data files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from threading import Lock
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
    agentdog_unified_sft_rows,
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


_PROGRESS_LOCK = Lock()


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
    parser.add_argument("--llm-generation-concurrency", type=int, help="Concurrent LLM trajectory synthesis requests.")
    parser.add_argument("--llm-qc-concurrency", type=int, help="Concurrent LLM judge requests for semantic QC.")
    parser.add_argument("--checkpoint-dir", help="Checkpoint directory. Defaults to <out>/_checkpoints for LLM generation.")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable incremental checkpoint files.")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint files and skip completed case ids.")
    parser.add_argument("--pause-file", help="Pause scheduling new API calls while this file exists. Defaults to <checkpoint-dir>/PAUSE.")
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
        "--llm-qc-match-policy",
        choices=["exact_taxonomy", "taxonomy_soft_relabel", "label_only", "label_relabel"],
        help=(
            "How LLM QC compares judge output with seed labels. exact_taxonomy requires all unsafe taxonomy "
            "dimensions to match; label_relabel keeps binary-safe labels and relabels unsafe taxonomy from the judge."
        ),
    )
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
    out = Path(args.out or config["out"] or "data")
    checkpoint_dir = _resolve_checkpoint_dir(
        out=out,
        generation_backend=generation_backend,
        cli_checkpoint_dir=args.checkpoint_dir,
        config_checkpoint_dir=config.get("checkpoint_dir"),
        disabled=args.no_checkpoint,
    )
    pause_file = _resolve_pause_file(args.pause_file, config.get("pause_file"), checkpoint_dir)
    resume = bool(args.resume or config.get("resume"))
    llm_generation_concurrency = max(
        1,
        int(
            args.llm_generation_concurrency
            if args.llm_generation_concurrency is not None
            else config.get("llm_generation_concurrency")
            or 1
        ),
    )
    llm_qc_concurrency = max(
        1,
        int(args.llm_qc_concurrency if args.llm_qc_concurrency is not None else config.get("llm_qc_concurrency") or 1),
    )
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
    llm_qc_match_policy = args.llm_qc_match_policy or str(config.get("llm_qc_match_policy") or "exact_taxonomy")
    if llm_qc_match_policy not in {"exact_taxonomy", "taxonomy_soft_relabel", "label_only", "label_relabel"}:
        raise SystemExit("llm_qc_match_policy must be exact_taxonomy, taxonomy_soft_relabel, label_only, or label_relabel")
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
    _emit_cli_progress(
        {
            "phase": "planning",
            "status": "completed",
            "requested": count,
            "planned": len(planner_cases),
            "generation_backend": generation_backend,
            "llm_qc": llm_qc,
        }
    )
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
            progress_callback=_emit_cli_progress,
            concurrency=llm_generation_concurrency,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            pause_file=pause_file,
        )
        cases = generation.cases
        llm_generation_summary = generation.summary
    else:
        cases = planner_cases
    _emit_cli_progress(
        {
            "phase": "deterministic_qc",
            "status": "running",
            "input_cases": len(cases),
            "min_quality_score": qc_min_score,
        }
    )
    cases, deterministic_qc = filter_cases_by_agentdog_qc(cases, min_quality_score=qc_min_score)
    _emit_cli_progress(
        {
            "phase": "deterministic_qc",
            "status": "completed",
            "kept": len(cases),
            "rejected": deterministic_qc.get("rejected", 0),
            "pass_rate": deterministic_qc.get("pass_rate", 0.0),
        }
    )
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
            progress_callback=_emit_cli_progress,
            concurrency=llm_qc_concurrency,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            pause_file=pause_file,
            match_policy=llm_qc_match_policy,
        )
    training_cases, training_filter = filter_cases_for_training(cases, max_repair_level=training_max_repair_level)
    production_summary = production_quality_summary(
        cases,
        training_cases=training_cases,
        training_max_repair_level=training_max_repair_level,
    )
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
        counts["agentdog_unified_sft.jsonl"] = write_jsonl(
            out / "agentdog_unified_sft.jsonl",
            agentdog_unified_sft_rows(training_cases),
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


def _emit_cli_progress(event: dict[str, Any]) -> None:
    payload = {"ts": datetime.now().isoformat(timespec="seconds"), **event}
    with _PROGRESS_LOCK:
        print("TRACEHOUND_PROGRESS " + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _resolve_checkpoint_dir(
    *,
    out: Path,
    generation_backend: str,
    cli_checkpoint_dir: str | None,
    config_checkpoint_dir: Any,
    disabled: bool,
) -> Path | None:
    if disabled:
        return None
    candidate = cli_checkpoint_dir if cli_checkpoint_dir is not None else config_checkpoint_dir
    if candidate in (None, "", False):
        return out / "_checkpoints" if generation_backend == "llm" else None
    return Path(str(candidate))


def _resolve_pause_file(cli_pause_file: str | None, config_pause_file: Any, checkpoint_dir: Path | None) -> Path | None:
    candidate = cli_pause_file if cli_pause_file is not None else config_pause_file
    if candidate not in (None, "", False):
        return Path(str(candidate))
    if checkpoint_dir is not None:
        return checkpoint_dir / "PAUSE"
    return None


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
    progress_callback: Any | None = None,
    concurrency: int = 1,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    pause_file: str | Path | None = None,
    match_policy: str = "exact_taxonomy",
) -> tuple[list[dict], dict]:
    concurrency = max(1, int(concurrency or 1))
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
    total = len(cases)
    kept_by_id: dict[str, dict] = {}
    rejected_by_id: dict[str, dict] = {}
    checkpoint_paths = _prepare_qc_checkpoint(checkpoint_dir, resume=resume)
    if checkpoint_paths is not None and resume:
        kept_by_id.update(
            {
                str(case.get("id")): case
                for case in _load_jsonl(checkpoint_paths["kept"])
                if isinstance(case, dict) and case.get("id")
            }
        )
        rejected_by_id.update(
            {
                str(item.get("id")): item
                for item in _load_jsonl(checkpoint_paths["rejected"])
                if isinstance(item, dict) and item.get("id")
            }
        )
        if progress_callback:
            progress_callback(
                {
                    "phase": "llm_qc",
                    "status": "resumed",
                    "total": total,
                    "completed": len(kept_by_id),
                    "rejected": len(rejected_by_id),
                    "checkpoint_dir": str(checkpoint_paths["dir"]),
                }
            )
    lock = Lock()

    def counts() -> tuple[int, int]:
        with lock:
            return len(kept_by_id), len(rejected_by_id)

    def judge_one(index: int, case: dict) -> tuple[int, str, str, dict | None, dict | None]:
        case_id = str(case.get("id") or f"case_{index}")
        completed, rejected_count = counts()
        if progress_callback:
            progress_callback(
                {
                    "phase": "llm_qc",
                    "status": "running",
                    "current": index,
                    "total": total,
                    "completed": completed,
                    "rejected": rejected_count,
                    "case_id": case_id,
                }
        )
        gold = report_from_gold(case["gold"])
        votes = []
        vote_errors = []
        for judge_meta in judges:
            try:
                report = evaluate_case(TrajectoryCase.model_validate(case), mode=mode, judge=judge_meta["judge"])
            except Exception as exc:  # noqa: BLE001 - LLM QC is a filter; failed judge calls reject one sample.
                vote_errors.append({"spec": judge_meta["spec"], "error": str(exc)[:1000]})
                continue
            votes.append({"spec": judge_meta["spec"], "report": report})
        if vote_errors:
            return (
                index,
                case_id,
                "rejected",
                None,
                {
                    "id": case_id,
                    "gold": gold.model_dump(mode="json"),
                    "aggregate": {},
                    "consensus": 0.0,
                    "error": "llm_qc_judge_error",
                    "vote_errors": vote_errors,
                    "votes": [{"spec": vote["spec"], "report": vote["report"].model_dump(mode="json")} for vote in votes],
                },
            )
        aggregate = _aggregate_votes(votes)
        consensus = float(aggregate.get("consensus", 0.0))
        aggregate_report = aggregate.get("report")
        match = _qc_match_decision(gold, aggregate_report, consensus, consensus_threshold, match_policy)
        if match["accepted"]:
            if match["relabel"] and aggregate_report is not None:
                _relabel_case_gold(case, aggregate_report, match_policy=match_policy)
            case.setdefault("metadata", {}).setdefault("qc", {})["llm_judge"] = {
                "judges": judge_specs,
                "mode": mode,
                "consensus": consensus,
                "matched": True,
                "match_policy": match_policy,
                "match_reason": match["reason"],
                "relabelled": bool(match["relabel"]),
            }
            return index, case_id, "accepted", case, None
        rejected_item = {
            "id": case_id,
            "gold": gold.model_dump(mode="json"),
            "aggregate": aggregate_report.model_dump(mode="json") if aggregate_report else {},
            "consensus": consensus,
            "match_policy": match_policy,
            "match_reason": match["reason"],
            "votes": [{"spec": vote["spec"], "report": vote["report"].model_dump(mode="json")} for vote in votes],
        }
        return index, case_id, "rejected", None, rejected_item

    def record_result(result: tuple[int, str, str, dict | None, dict | None]) -> None:
        index, case_id, status, case, rejected_item = result
        with lock:
            if status == "accepted" and case is not None:
                kept_by_id[case_id] = case
                rejected_by_id.pop(case_id, None)
                completed = len(kept_by_id)
                rejected_count = len(rejected_by_id)
            else:
                rejected_by_id[case_id] = rejected_item or {"id": case_id, "error": "unknown llm qc rejection"}
                completed = len(kept_by_id)
                rejected_count = len(rejected_by_id)
        if checkpoint_paths is not None:
            if status == "accepted" and case is not None:
                _append_jsonl(checkpoint_paths["kept"], case)
            else:
                _append_jsonl(checkpoint_paths["rejected"], rejected_by_id[case_id])
            _write_json(
                checkpoint_paths["state"],
                {
                    "phase": "llm_qc",
                    "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "total": total,
                    "completed": completed,
                    "rejected": rejected_count,
                    "pending": max(total - completed - rejected_count, 0),
                    "concurrency": concurrency,
                },
            )
        if progress_callback:
            progress_callback(
                {
                    "phase": "llm_qc",
                    "status": status,
                    "current": index,
                    "total": total,
                    "completed": completed,
                    "rejected": rejected_count,
                    "case_id": case_id,
                    "consensus": (rejected_item or {}).get("consensus") if rejected_item else None,
                }
            )

    processed = set(kept_by_id) | set(rejected_by_id)
    indexed_cases = [
        (index, case)
        for index, case in enumerate(cases, start=1)
        if str(case.get("id") or f"case_{index}") not in processed
    ]
    if concurrency == 1:
        for index, case in indexed_cases:
            _wait_if_paused(pause_file, progress_callback, phase="llm_qc")
            record_result(judge_one(index, case))
    else:
        case_iter = iter(indexed_cases)
        exhausted = False
        active: dict[Future, tuple[int, dict]] = {}
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while active or not exhausted:
                while not exhausted and len(active) < concurrency:
                    _wait_if_paused(pause_file, progress_callback, phase="llm_qc")
                    try:
                        index, case = next(case_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    active[executor.submit(judge_one, index, case)] = (index, case)
                if not active:
                    continue
                done, _pending = wait(active, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    index, case = active.pop(future, (0, {}))
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001 - isolate one QC worker from the full batch.
                        case_id = str(case.get("id") or f"case_{index}")
                        result = (
                            index,
                            case_id,
                            "rejected",
                            None,
                            {"id": case_id, "error": "llm_qc_worker_error", "detail": str(exc)[:1000]},
                        )
                    record_result(result)

    ordered_ids = [str(case.get("id") or f"case_{index}") for index, case in enumerate(cases, start=1)]
    kept = [kept_by_id[case_id] for case_id in ordered_ids if case_id in kept_by_id]
    rejected = [rejected_by_id[case_id] for case_id in ordered_ids if case_id in rejected_by_id]
    return kept, {
        "enabled": True,
        "judges": judge_specs,
        "mode": mode,
        "consensus_threshold": consensus_threshold,
        "generated": len(cases),
        "kept": len(kept),
        "rejected": len(rejected),
        "pass_rate": round(len(kept) / len(cases), 4) if cases else 0.0,
        "concurrency": concurrency,
        "match_policy": match_policy,
        "checkpoint_dir": str(checkpoint_paths["dir"]) if checkpoint_paths is not None else "",
        "resume": resume,
        "pause_file": str(pause_file) if pause_file else "",
        "rejected_samples": rejected,
    }


def _prepare_qc_checkpoint(checkpoint_dir: str | Path | None, *, resume: bool) -> dict[str, Path] | None:
    if checkpoint_dir is None:
        return None
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "dir": root,
        "kept": root / "llm_qc_kept_cases.jsonl",
        "rejected": root / "llm_qc_rejected_cases.jsonl",
        "state": root / "llm_qc_state.json",
    }
    if not resume:
        for key in ("kept", "rejected", "state"):
            paths[key].unlink(missing_ok=True)
    return paths


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _wait_if_paused(pause_file: str | Path | None, progress_callback: Any | None, *, phase: str) -> None:
    if not pause_file:
        return
    path = Path(pause_file)
    emitted = False
    while path.exists():
        if not emitted and progress_callback:
            progress_callback({"phase": phase, "status": "paused", "pause_file": str(path)})
            emitted = True
        time.sleep(5)
    if emitted and progress_callback:
        progress_callback({"phase": phase, "status": "resumed", "pause_file": str(path)})


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


def _qc_match_decision(
    gold: Any,
    report: Any | None,
    consensus: float,
    consensus_threshold: float,
    match_policy: str,
) -> dict[str, Any]:
    if report is None:
        return {"accepted": False, "relabel": False, "reason": "no_aggregate_report"}
    if consensus < consensus_threshold:
        return {
            "accepted": False,
            "relabel": False,
            "reason": f"low_consensus:{consensus:.3f}<threshold:{consensus_threshold:.3f}",
        }
    if match_policy == "exact_taxonomy":
        accepted = _qc_matches(gold, report)
        return {"accepted": accepted, "relabel": False, "reason": "exact_taxonomy" if accepted else "taxonomy_mismatch"}
    if gold.label != report.label:
        return {
            "accepted": False,
            "relabel": False,
            "reason": f"label_mismatch:{gold.label}!={report.label}",
        }
    if match_policy == "label_only":
        return {"accepted": True, "relabel": False, "reason": "label_match"}
    if match_policy == "label_relabel":
        return {"accepted": True, "relabel": gold.label == "unsafe", "reason": "label_match_relabel_unsafe"}
    if match_policy == "taxonomy_soft_relabel":
        if gold.label == "safe":
            return {"accepted": True, "relabel": False, "reason": "safe_label_match"}
        matched_fields = [
            field
            for field in ("risk_source", "failure_mode", "harm_type")
            if getattr(gold, field) == getattr(report, field)
        ]
        if matched_fields:
            return {
                "accepted": True,
                "relabel": True,
                "reason": "soft_taxonomy_match:" + ",".join(matched_fields),
            }
        return {"accepted": False, "relabel": False, "reason": "soft_taxonomy_no_dimension_match"}
    return {"accepted": False, "relabel": False, "reason": f"unknown_match_policy:{match_policy}"}


def _relabel_case_gold(case: dict, report: Any, *, match_policy: str) -> None:
    previous_gold = case.get("gold", {})
    updated_gold = report.model_dump(mode="json")
    if isinstance(previous_gold, dict) and "cost" in previous_gold:
        updated_gold["cost"] = previous_gold["cost"]
    previous_taxonomy = {
        "label": previous_gold.get("label") if isinstance(previous_gold, dict) else None,
        "risk_source": previous_gold.get("risk_source") if isinstance(previous_gold, dict) else None,
        "failure_mode": previous_gold.get("failure_mode") if isinstance(previous_gold, dict) else None,
        "harm_type": previous_gold.get("harm_type") if isinstance(previous_gold, dict) else None,
    }
    updated_taxonomy = {
        "label": updated_gold.get("label"),
        "risk_source": updated_gold.get("risk_source"),
        "failure_mode": updated_gold.get("failure_mode"),
        "harm_type": updated_gold.get("harm_type"),
    }
    case.setdefault("metadata", {}).setdefault("qc", {})["llm_relabel"] = {
        "match_policy": match_policy,
        "previous": previous_taxonomy,
        "updated": updated_taxonomy,
    }
    case["gold"] = updated_gold


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
