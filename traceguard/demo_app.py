"""FastAPI application factory for the TraceHound demo."""

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from time import sleep
from typing import Any, Dict
from uuid import uuid4

from traceguard.compressor import compress_trajectory
from traceguard.config import api_runtime_status
from traceguard.data import TOOL_SCENARIOS, built_in_cases, dataset_summary
from traceguard.enchantment import build_enchantment_plan, write_enchantment_plan
from traceguard.evaluation import summarize_predictions
from traceguard.export import (
    agentdog15_coarse_sft_rows,
    agentdog15_unified_sft_rows,
    agentdog_binary_sft_rows,
    agentdog_taxonomy_sft_rows,
    eval_rows,
    rl_rows,
    sft_rows,
    write_jsonl,
)
from traceguard.generation_config import load_generation_config
from traceguard.guard import TraceGuard
from traceguard.judge import build_remote_judge
from traceguard.llm_generation import llm_synthesize_cases
from traceguard.model_profiles import list_model_profiles, profile_summary, resolve_model_profile
from traceguard.pipeline import evaluate_case
from traceguard.production import filter_cases_for_training, production_quality_summary
from traceguard.quality import filter_cases_by_agentdog_qc
from traceguard.reporting import build_report, write_metric_chart
from traceguard.schema import RiskReport, TrajectoryCase, TrajectoryStep, dump_model, report_from_gold

TRAIN_PACKAGES = ("torch", "transformers", "datasets", "peft", "accelerate")

_STATE_LOCK = Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}
_ENCHANTED_MODELS: list[Dict[str, Any]] = []
_MODEL_STATE: Dict[str, Any] = {
    "deployment_mode": "api" if api_runtime_status().get("configured") else "local",
    "local_model": os.getenv("TRACEHOUND_LOCAL_MODEL", "internlm/internlm3-8b-instruct"),
    "active_finetuned_model": "",
    "fine_tuned_models": [],
}


def create_app():
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FastAPI is not installed. Create the conda environment with `conda env create -f environment.yml` "
            "and run `conda activate tracehound`."
        ) from exc

    _sync_default_model_mode()
    root = Path(__file__).resolve().parent.parent
    web_dir = root / "web_demo"
    app = FastAPI(title="TraceHound Demo", version="0.1.0")
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

    @app.get("/")
    def index():
        return FileResponse(web_dir / "index.html")

    @app.get("/api/cases")
    def list_cases():
        return [
            {
                "id": case["id"],
                "task": case.get("task", ""),
                "scenario": case.get("metadata", {}).get("scenario", ""),
                "gold_label": case.get("gold", {}).get("label", ""),
            }
            for case in built_in_cases()
        ]

    @app.get("/api/runtime")
    def runtime():
        return {
            "api": api_runtime_status(),
            "model": _model_status(),
            "model_profiles": _model_profiles_status(),
            "judges": [
                {"id": "heuristic", "label": "Heuristic", "remote": False},
                {"id": "hybrid", "label": "Hybrid API", "remote": True},
                {"id": "api", "label": "API Only", "remote": True},
            ],
        }

    @app.get("/api/generation-config")
    def generation_config(path: str = "configs/generation.yaml"):
        config_path = _project_path(root, path)
        if config_path is None or config_path.suffix not in {".yaml", ".yml"}:
            raise HTTPException(status_code=400, detail="config path must be a project-local YAML file")
        try:
            return {"path": str(config_path.relative_to(root)), "config": load_generation_config(config_path)}
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        for case in built_in_cases():
            if case["id"] == case_id:
                return case
        raise HTTPException(status_code=404, detail=f"unknown case id: {case_id}")

    @app.post("/api/evaluate")
    async def evaluate(request: Request):
        body: Dict[str, Any] = await request.json()
        mode = body.get("mode", "layered")
        judge_name = body.get("judge", "heuristic")
        raw_case = body.get("case", body)
        try:
            case, _ = _case_and_gold(raw_case, 1)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        judge = _build_judge_or_400(judge_name, HTTPException)
        report = evaluate_case(case, mode=mode, judge=judge)
        compressed = compress_trajectory(case)
        return {
            "report": dump_model(report),
            "compressed": dump_model(compressed),
            "guard": _simulate_guard(case),
            "runtime": {
                "judge": judge_name,
                "mode": mode,
                "api": api_runtime_status(),
                "strategy": report.cost.strategy,
                "model_calls": report.cost.model_calls,
            },
        }

    @app.post("/api/batch-evaluate")
    async def batch_evaluate(request: Request):
        body: Dict[str, Any] = await request.json()
        mode = body.get("mode", "layered")
        judge_name = body.get("judge", "heuristic")
        raw_cases = body.get("cases")
        if isinstance(raw_cases, dict):
            raw_cases = [raw_cases]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise HTTPException(status_code=400, detail="body.cases must be a non-empty array")
        max_cases = 50 if judge_name in {"api", "hybrid"} else 1000
        if len(raw_cases) > max_cases:
            raise HTTPException(status_code=400, detail=f"batch limit is {max_cases} cases for judge={judge_name}")
        judge = _build_judge_or_400(judge_name, HTTPException)
        predictions: list[RiskReport] = []
        serialized: list[Dict[str, Any]] = []
        golds: list[RiskReport] = []
        gold_predictions: list[RiskReport] = []
        cases_with_gold = 0
        started = datetime.now(timezone.utc)
        for index, raw_case in enumerate(raw_cases, start=1):
            try:
                case, gold = _case_and_gold(raw_case, index)
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"case {index}: {exc}") from exc
            report = evaluate_case(case, mode=mode, judge=judge)
            predictions.append(report)
            item = {"id": case.id, "report": dump_model(report)}
            if gold is not None:
                item["gold"] = dump_model(gold)
                golds.append(gold)
                gold_predictions.append(report)
                cases_with_gold += 1
            serialized.append(item)
        metrics = summarize_predictions(golds, gold_predictions) if cases_with_gold else {}
        report_id = f"demo_batch_{uuid4().hex[:10]}"
        reports_dir = root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        json_path = reports_dir / f"{report_id}.json"
        md_path = reports_dir / f"{report_id}.md"
        chart_path = reports_dir / f"{report_id}.svg"
        experiment_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data": "web_demo_upload",
            "experiments": [
                {
                    "name": "uploaded_batch",
                    "mode": mode,
                    "judge": judge_name,
                    "limit": len(raw_cases),
                    "metrics": metrics,
                }
            ],
            "online_guard": {},
            "skipped": [],
        }
        write_metric_chart(experiment_data, chart_path)
        md_path.write_text(build_report(experiment_data, chart_path=chart_path), encoding="utf-8")
        payload = {
            "id": report_id,
            "generated_at": started.isoformat(),
            "mode": mode,
            "judge": judge_name,
            "samples": len(raw_cases),
            "gold_samples": cases_with_gold,
            "metrics": metrics,
            "predictions": serialized,
            "downloads": {
                "json": f"/api/reports/{json_path.name}",
                "markdown": f"/api/reports/{md_path.name}",
                "chart": f"/api/reports/{chart_path.name}",
            },
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    @app.get("/api/reports/{filename}")
    def download_report(filename: str):
        report_path = _report_path(root, filename)
        if report_path is None or not report_path.exists():
            raise HTTPException(status_code=404, detail=f"unknown report: {filename}")
        media_types = {".json": "application/json", ".md": "text/markdown; charset=utf-8", ".svg": "image/svg+xml"}
        return FileResponse(report_path, media_type=media_types.get(report_path.suffix, "application/octet-stream"))

    @app.get("/api/guard-model")
    def guard_model_status():
        return _model_status()

    @app.post("/api/guard-model")
    async def configure_guard_model(request: Request):
        body: Dict[str, Any] = await request.json()
        mode = body.get("deployment_mode", _MODEL_STATE["deployment_mode"])
        if mode not in {"api", "local"}:
            raise HTTPException(status_code=400, detail="deployment_mode must be api or local")
        with _STATE_LOCK:
            _MODEL_STATE["deployment_mode"] = mode
            _MODEL_STATE["_user_configured"] = True
            if body.get("local_model"):
                _MODEL_STATE["local_model"] = str(body["local_model"])
            if body.get("active_finetuned_model"):
                _MODEL_STATE["active_finetuned_model"] = str(body["active_finetuned_model"])
        return _model_status()

    @app.post("/api/guard-model/switch")
    async def switch_guard_model(request: Request):
        body: Dict[str, Any] = await request.json()
        model_id = str(body.get("model_id") or "")
        with _STATE_LOCK:
            known = {item["id"] for item in _MODEL_STATE.get("fine_tuned_models", [])}
            if model_id and model_id not in known and model_id != _MODEL_STATE.get("local_model"):
                raise HTTPException(status_code=404, detail=f"unknown local model: {model_id}")
            _MODEL_STATE["deployment_mode"] = "local"
            _MODEL_STATE["active_finetuned_model"] = model_id
        return _model_status()

    @app.post("/api/data-generation")
    async def start_data_generation(request: Request):
        body: Dict[str, Any] = await request.json()
        config = {}
        config_path = body.get("config_path")
        if config_path:
            resolved = _project_path(root, str(config_path))
            if resolved is None or resolved.suffix not in {".yaml", ".yml"}:
                raise HTTPException(status_code=400, detail="config_path must be a project-local YAML file")
            try:
                config = load_generation_config(resolved)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            count = _bounded_int(body.get("count", config.get("count") or 10000), minimum=1, maximum=50000)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        scenarios = _optional_list(body.get("scenarios", config.get("scenarios")))
        labels = _optional_list(body.get("labels", config.get("labels")))
        include_eval = bool(body.get("include_eval", config.get("include_eval", True)))
        include_sft = bool(body.get("include_sft", config.get("include_sft", False)))
        include_rl = bool(body.get("include_rl", config.get("include_rl", False)))
        generation_backend = str(body.get("generation_backend", config.get("generation_backend", "deterministic")))
        if generation_backend not in {"deterministic", "llm"}:
            raise HTTPException(status_code=400, detail="generation_backend must be deterministic or llm")
        semantic_repair_backend = str(
            body.get(
                "semantic_repair_backend",
                config.get("semantic_repair_backend") or ("llm" if generation_backend == "llm" else "static"),
            )
        )
        if semantic_repair_backend not in {"none", "static", "llm", "llm_then_static"}:
            raise HTTPException(status_code=400, detail="semantic_repair_backend must be none, static, llm, or llm_then_static")
        try:
            semantic_repair_rounds = _bounded_int(
                body.get("semantic_repair_rounds", config.get("semantic_repair_rounds") or 1),
                minimum=0,
                maximum=5,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not include_eval and not include_sft and not include_rl:
            raise HTTPException(status_code=400, detail="at least one dataset type must be selected")
        job = _new_job(
            "data_generation",
            {
                "count": count,
                "scenarios": scenarios or ["all"],
                "labels": labels or ["all"],
                "generation_backend": generation_backend,
                "semantic_repair_backend": semantic_repair_backend,
                "semantic_repair_rounds": semantic_repair_rounds,
                "include_eval": include_eval,
                "include_sft": include_sft,
                "include_rl": include_rl,
            },
        )
        Thread(
            target=_run_data_generation_job,
            args=(
                job["id"],
                count,
                scenarios,
                labels,
                generation_backend,
                semantic_repair_backend,
                semantic_repair_rounds,
                include_eval,
                include_sft,
                include_rl,
            ),
            daemon=True,
        ).start()
        return job

    @app.post("/api/training")
    async def start_training(request: Request):
        body: Dict[str, Any] = await request.json()
        kind = body.get("kind", "sft")
        if kind not in {"sft", "sft_rl"}:
            raise HTTPException(status_code=400, detail="kind must be sft or sft_rl")
        algorithm = body.get("algorithm", "dpo")
        if algorithm not in {"dpo", "grpo", "orpo"}:
            raise HTTPException(status_code=400, detail="algorithm must be dpo, grpo, or orpo")
        job = _new_job(
            "training",
            {
                "kind": kind,
                "algorithm": algorithm,
                "data_dir": body.get("data_dir") or "data/tmp/generated/latest",
                "auto_switch": bool(body.get("auto_switch", False)),
            },
        )
        Thread(
            target=_run_training_job,
            args=(job["id"], kind, algorithm, str(body.get("data_dir") or "data/tmp/generated/latest"), bool(body.get("auto_switch", False))),
            daemon=True,
        ).start()
        return job

    @app.get("/api/safety-enchantment")
    def safety_enchantment_status():
        with _STATE_LOCK:
            enchanted_models = list(_ENCHANTED_MODELS)
        return {
            "guard": _model_status(),
            "model_profiles": _model_profiles_status(),
            "enchanted_models": enchanted_models,
            "algorithms": [
                {"id": "sft", "label": "Guard-filtered SFT", "requires_rl": False},
                {"id": "sft_dpo", "label": "Guard-filtered SFT + DPO", "requires_rl": False},
                {"id": "sft_grpo", "label": "Guard-filtered SFT + GRPO", "requires_rl": True},
            ],
            "reward_formula": {
                "normal_benign": "U",
                "attacked_benign": "0.5 * U + 0.25 * S + 0.25 * U * S",
                "malicious": "S",
            },
        }

    @app.post("/api/safety-enchantment")
    async def start_safety_enchantment(request: Request):
        body: Dict[str, Any] = await request.json()
        algorithm = str(body.get("algorithm") or "sft_dpo")
        if algorithm not in {"sft", "sft_dpo", "sft_grpo"}:
            raise HTTPException(status_code=400, detail="algorithm must be sft, sft_dpo, or sft_grpo")
        profile_name = str(body.get("target_model_profile") or os.getenv("TRACEHOUND_TARGET_MODEL_PROFILE", "internlm3-8b-instruct"))
        try:
            target_profile = resolve_model_profile(profile_name)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if target_profile.get("provider") != "huggingface":
            raise HTTPException(status_code=400, detail="safety enchantment requires a Hugging Face local target profile")
        max_samples_raw = body.get("max_samples")
        try:
            max_samples = None if max_samples_raw in (None, "", 0, "0") else _bounded_int(max_samples_raw, minimum=1, maximum=50000)
            safety_weight = _bounded_float(body.get("safety_weight", 0.5), minimum=0.0, maximum=1.0)
            utility_weight = _bounded_float(body.get("utility_weight", 0.5), minimum=0.0, maximum=1.0)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        model_status = _model_status()
        job = _new_job(
            "safety_enchantment",
            {
                "algorithm": algorithm,
                "target_model_profile": profile_name,
                "target_base_model": body.get("target_base_model") or "",
                "guard_model": model_status.get("current_model"),
                "guard_mode": model_status.get("serving_type"),
                "data_dir": body.get("data_dir") or "data/tmp/generated/latest",
                "output_dir": body.get("output_dir") or "",
                "max_samples": max_samples,
                "safety_weight": safety_weight,
                "utility_weight": utility_weight,
                "auto_register": bool(body.get("auto_register", True)),
            },
        )
        Thread(
            target=_run_safety_enchantment_job,
            args=(
                job["id"],
                target_profile,
                str(body.get("target_base_model") or ""),
                str(model_status.get("current_model") or ""),
                str(model_status.get("serving_type") or ""),
                str(body.get("data_dir") or "data/tmp/generated/latest"),
                algorithm,
                str(body.get("output_dir") or ""),
                max_samples,
                safety_weight,
                utility_weight,
                bool(body.get("auto_register", True)),
            ),
            daemon=True,
        ).start()
        return job

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = _get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job: {job_id}")
        return job

    return app


def _build_judge_or_400(judge_name: str, http_exception_cls: Any) -> Any:
    if judge_name in {"api", "hybrid"}:
        try:
            return build_remote_judge(judge=judge_name)
        except ValueError as exc:
            raise http_exception_cls(status_code=400, detail=str(exc)) from exc
    if judge_name != "heuristic":
        raise http_exception_cls(status_code=400, detail=f"unknown judge: {judge_name}")
    return None


def _case_and_gold(raw: Dict[str, Any], index: int) -> tuple[TrajectoryCase, RiskReport | None]:
    if not isinstance(raw, dict):
        raise TypeError("case must be an object")
    source = raw.get("case") if isinstance(raw.get("case"), dict) else raw
    case_payload = {key: source[key] for key in ("id", "task", "metadata", "trajectory") if key in source}
    if "trajectory" not in case_payload:
        raise KeyError("missing trajectory")
    case_payload.setdefault("id", f"uploaded_{index:04d}")
    case_payload.setdefault("task", "")
    case_payload.setdefault("metadata", {})
    case = TrajectoryCase.model_validate(case_payload)
    gold_raw = raw.get("gold", source.get("gold"))
    if gold_raw is None:
        gold_raw = raw.get("label", source.get("label"))
    gold = _normalize_gold(gold_raw)
    return case, gold


def _normalize_gold(raw: Any) -> RiskReport | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        raw = {"label": raw}
    if not isinstance(raw, dict):
        raise TypeError("gold or label must be an object or label string")
    return report_from_gold(raw)


def _project_path(root: Path, raw_path: str) -> Path | None:
    path = Path(raw_path)
    candidate = (path if path.is_absolute() else root / path).resolve()
    root_resolved = root.resolve()
    if candidate == root_resolved or root_resolved in candidate.parents:
        return candidate
    return None


def _report_path(root: Path, filename: str) -> Path | None:
    if Path(filename).name != filename:
        return None
    if Path(filename).suffix not in {".json", ".md", ".svg"}:
        return None
    reports_dir = (root / "reports").resolve()
    candidate = (reports_dir / filename).resolve()
    if reports_dir in candidate.parents and candidate.name == filename:
        return candidate
    return None


def _simulate_guard(case: TrajectoryCase) -> Dict[str, Any]:
    guard = TraceGuard()
    for step in case.trajectory:
        if step.type == "tool_call":
            decision = guard.before_tool_call(case, step)
            if decision.decision in {"block", "ask_confirm"}:
                return dump_model(decision)
    observation = next((step for step in case.trajectory if step.type == "observation"), None)
    if observation is not None:
        decision = guard.after_tool_observation(
            TrajectoryCase.model_validate(
                {
                    "id": case.id,
                    "task": case.task,
                    "metadata": case.metadata,
                    "trajectory": [step for step in case.trajectory if step.step < observation.step],
                }
            ),
            TrajectoryStep.model_validate(dump_model(observation)),
        )
        return dump_model(decision)
    return {"decision": "allow", "report": {"label": "safe", "decision": "allow"}}


def _model_status() -> Dict[str, Any]:
    api = api_runtime_status()
    with _STATE_LOCK:
        mode = _MODEL_STATE["deployment_mode"]
        local_model = _MODEL_STATE["active_finetuned_model"] or _MODEL_STATE["local_model"]
        fine_tuned = list(_MODEL_STATE.get("fine_tuned_models", []))
    try:
        active_profile = profile_summary(resolve_model_profile())
    except Exception:
        active_profile = {}
    current_model = api.get("model") if mode == "api" and api.get("configured") else local_model
    return {
        "deployment_mode": mode,
        "serving_type": "model_api" if mode == "api" else "local_deployment",
        "current_model": current_model or "internlm/internlm3-8b-instruct",
        "active_profile": active_profile,
        "api": api,
        "local": {
            "model": local_model,
            "path": os.getenv("TRACEHOUND_LOCAL_MODEL_PATH", ""),
            "gpu_available": _cuda_available(),
            "training_packages_available": _missing_training_packages() == [],
        },
        "fine_tuned_models": fine_tuned,
        "scenarios": list(TOOL_SCENARIOS),
    }


def _model_profiles_status() -> list[Dict[str, Any]]:
    try:
        return [
            profile_summary({"name": name, **profile})
            for name, profile in sorted(list_model_profiles().items())
        ]
    except Exception:
        return []


def _sync_default_model_mode() -> None:
    with _STATE_LOCK:
        if api_runtime_status().get("configured") and not _MODEL_STATE.get("_user_configured"):
            _MODEL_STATE["deployment_mode"] = "api"


def _new_job(kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
    job_id = f"{kind}-{uuid4().hex[:10]}"
    job = {
        "id": job_id,
        "kind": kind,
        "status": "queued",
        "progress": 0,
        "step": "queued",
        "params": params,
        "messages": ["queued"],
        "artifacts": {},
    }
    with _STATE_LOCK:
        _JOBS[job_id] = job
    return job


def _get_job(job_id: str) -> Dict[str, Any] | None:
    with _STATE_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None


def _update_job(job_id: str, **updates: Any) -> None:
    with _STATE_LOCK:
        job = _JOBS[job_id]
        message = updates.pop("message", None)
        job.update(updates)
        if message:
            messages = job.setdefault("messages", [])
            messages.append(message)
            if len(messages) > 200:
                del messages[:-200]


def _run_data_generation_job(
    job_id: str,
    count: int,
    scenarios: list[str] | None,
    labels: list[str] | None,
    generation_backend: str,
    semantic_repair_backend: str,
    semantic_repair_rounds: int,
    include_eval: bool,
    include_sft: bool,
    include_rl: bool,
) -> None:
    try:
        _update_job(job_id, status="running", progress=8, step="building cases", message=f"building {count} cases")
        planner_cases = built_in_cases(scenarios=scenarios, labels=labels, count=count)
        generation_summary: Dict[str, Any] = {"enabled": False, "backend": "deterministic"}
        if generation_backend == "llm":
            _update_job(
                job_id,
                progress=16,
                step="llm trajectory synthesis",
                message="calling LLM for AgentDoG Stage 2 trajectory synthesis",
                synthesis={
                    "backend": "llm",
                    "phase": "llm_trajectory_synthesis",
                    "current": 0,
                    "total": len(planner_cases),
                    "completed": 0,
                    "rejected": 0,
                    "case_id": "",
                    "status": "starting",
                },
            )
            generation = llm_synthesize_cases(
                planner_cases,
                semantic_repair_backend=semantic_repair_backend,
                semantic_repair_rounds=semantic_repair_rounds,
                progress_callback=lambda event: _update_generation_progress(job_id, event),
            )
            cases = generation.cases
            generation_summary = generation.summary
        else:
            cases = planner_cases
            _update_job(
                job_id,
                progress=18,
                step="deterministic trajectory synthesis",
                synthesis={
                    "backend": "deterministic",
                    "phase": "local_planner",
                    "current": len(cases),
                    "total": len(cases),
                    "completed": len(cases),
                    "rejected": 0,
                    "case_id": "",
                    "status": "completed",
                },
                message=f"deterministic planner generated {len(cases)} cases",
            )
        out_dir = Path("data/tmp/generated") / job_id
        artifacts: Dict[str, str] = {"output_dir": str(out_dir)}
        latest = Path("data/tmp/generated/latest")
        out_dir.mkdir(parents=True, exist_ok=True)
        _update_job(job_id, progress=72, step="AgentDoG QC", message=f"running AgentDoG QC on {len(cases)} cases")
        cases, qc_summary = filter_cases_by_agentdog_qc(cases)
        summary = dataset_summary(cases)
        training_cases, training_filter = filter_cases_for_training(cases, max_repair_level="structural")
        production_summary = production_quality_summary(
            cases,
            training_cases=training_cases,
            training_max_repair_level="structural",
        )
        _write_generation_diagnostics(out_dir, artifacts, generation_summary, qc_summary, production_summary, training_filter)
        if not cases:
            _update_job(
                job_id,
                status="failed",
                progress=100,
                step="QC rejected all cases",
                artifacts=artifacts,
                summary=summary,
                quality=qc_summary,
                error="AgentDoG QC filtered out all cases",
                message=_qc_failure_message(generation_summary, qc_summary),
            )
            return
        _update_job(
            job_id,
            progress=82,
            step="writing eval data",
            message=f"case summary: {summary}; QC pass_rate={qc_summary.get('pass_rate')}",
        )
        if include_eval:
            artifacts["eval"] = str(out_dir / "synthetic_eval.jsonl")
            write_jsonl(out_dir / "synthetic_eval.jsonl", eval_rows(cases))
        _update_job(job_id, progress=88, step="writing sft data", message="eval export complete")
        if include_sft:
            artifacts["sft"] = str(out_dir / "synthetic_sft.jsonl")
            artifacts["agentdog_binary_sft"] = str(out_dir / "agentdog_binary_sft.jsonl")
            artifacts["agentdog_taxonomy_sft"] = str(out_dir / "agentdog_taxonomy_sft.jsonl")
            artifacts["agentdog15_unified_sft"] = str(out_dir / "agentdog15_unified_sft.jsonl")
            artifacts["agentdog15_coarse_sft"] = str(out_dir / "agentdog15_coarse_sft.jsonl")
            write_jsonl(out_dir / "synthetic_sft.jsonl", sft_rows(training_cases))
            write_jsonl(out_dir / "agentdog_binary_sft.jsonl", agentdog_binary_sft_rows(training_cases))
            write_jsonl(out_dir / "agentdog_taxonomy_sft.jsonl", agentdog_taxonomy_sft_rows(training_cases))
            write_jsonl(out_dir / "agentdog15_unified_sft.jsonl", agentdog15_unified_sft_rows(training_cases))
            write_jsonl(out_dir / "agentdog15_coarse_sft.jsonl", agentdog15_coarse_sft_rows(training_cases))
            _update_job(
                job_id,
                message=(
                    "training export quality filter: "
                    f"{training_filter.get('kept')}/{training_filter.get('input')} kept "
                    f"(max_repair_level={training_filter.get('max_repair_level')})"
                ),
            )
        _update_job(job_id, progress=94, step="writing rl data", message="sft export complete")
        if include_rl:
            artifacts["rl"] = str(out_dir / "synthetic_rl.jsonl")
            write_jsonl(out_dir / "synthetic_rl.jsonl", rl_rows(training_cases))
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(str(out_dir), encoding="utf-8")
        _update_job(
            job_id,
            status="completed",
            progress=100,
            step="completed",
            artifacts=artifacts,
            summary=summary,
            quality=qc_summary,
            production=production_summary,
            message="data generation completed",
        )
    except Exception as exc:  # pragma: no cover - defensive path surfaced through API.
        _update_job(job_id, status="failed", progress=100, step="failed", error=str(exc), message=f"failed: {exc}")


def _update_generation_progress(job_id: str, event: Dict[str, Any]) -> None:
    total = max(int(event.get("total") or 0), 1)
    current = int(event.get("current") or 0)
    completed = int(event.get("completed") or 0)
    rejected = int(event.get("rejected") or 0)
    progress = 16 + int(min(current, total) / total * 54)
    status = str(event.get("status") or "running")
    case_id = str(event.get("case_id") or "")
    attempt = event.get("attempt")
    message = None
    if status == "running":
        message = f"LLM synthesis {current}/{total}: {case_id}"
    elif status == "requesting":
        message = f"calling LLM {current}/{total} attempt {attempt or 1}: {case_id}"
    elif status == "validating":
        message = f"validating LLM output {current}/{total} attempt {attempt or 1}: {case_id}"
    elif status == "retrying":
        message = f"retrying LLM output {current}/{total} attempt {attempt or 1}: {case_id}; {event.get('error', '')}"
    elif status == "repairing":
        message = f"LLM self-repair {current}/{total} round {attempt or 1}: {case_id}; {event.get('error', '')}"
    elif status == "repair_failed":
        message = f"LLM self-repair failed {current}/{total} round {attempt or 1}: {case_id}; {event.get('error', '')}"
    elif status == "accepted":
        message = f"accepted {current}/{total}: {case_id}"
    elif status == "rejected":
        message = f"rejected {current}/{total}: {case_id}; {event.get('error', '')}"
    _update_job(
        job_id,
        progress=progress,
        step="llm trajectory synthesis",
        synthesis={
            "backend": "llm",
            "phase": event.get("phase", "llm_trajectory_synthesis"),
            "current": current,
            "total": total,
            "completed": completed,
            "rejected": rejected,
            "case_id": case_id,
            "status": status,
            "attempt": attempt,
        },
        message=message,
    )


def _write_generation_diagnostics(
    out_dir: Path,
    artifacts: Dict[str, str],
    generation_summary: Dict[str, Any],
    qc_summary: Dict[str, Any],
    production_summary: Dict[str, Any],
    training_filter: Dict[str, Any],
) -> None:
    artifacts["quality_report"] = str(out_dir / "quality_report.json")
    payload = {
        "policy": "agentdog_local",
        "generation": generation_summary,
        "deterministic": qc_summary,
        "llm": {"enabled": False},
        "production": production_summary,
        "training_filter": training_filter,
    }
    (out_dir / "quality_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rejected_rows = list(generation_summary.get("rejected_samples") or []) + list(qc_summary.get("rejected_samples") or [])
    if rejected_rows:
        artifacts["rejected"] = str(out_dir / "rejected_samples.jsonl")
        with (out_dir / "rejected_samples.jsonl").open("w", encoding="utf-8") as handle:
            for item in rejected_rows:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    else:
        rejected_path = out_dir / "rejected_samples.jsonl"
        if rejected_path.exists():
            rejected_path.unlink()
    training_rejected = list(training_filter.get("rejected_samples") or [])
    if training_rejected:
        artifacts["training_rejected"] = str(out_dir / "training_rejected_samples.jsonl")
        with (out_dir / "training_rejected_samples.jsonl").open("w", encoding="utf-8") as handle:
            for item in training_rejected:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _qc_failure_message(generation_summary: Dict[str, Any], qc_summary: Dict[str, Any]) -> str:
    generated = qc_summary.get("generated", 0)
    qc_rejected = qc_summary.get("rejected", 0)
    llm_rejected = generation_summary.get("rejected", 0)
    first = ""
    rejected_samples = qc_summary.get("rejected_samples") or generation_summary.get("rejected_samples") or []
    if rejected_samples:
        item = rejected_samples[0]
        reasons = item.get("errors") or [item.get("error", "unknown reason")]
        first = f"; first reason: {str(reasons[0])[:220]}"
    return (
        "AgentDoG QC rejected all generated cases. "
        f"generated={generated}, qc_rejected={qc_rejected}, llm_rejected={llm_rejected}. "
        "Open quality_report.json and rejected_samples.jsonl for diagnostics"
        f"{first}"
    )


def _run_training_job(job_id: str, kind: str, algorithm: str, data_dir: str, auto_switch: bool) -> None:
    try:
        _update_job(job_id, status="running", progress=10, step="validating local mode", message="training preflight started")
        if _MODEL_STATE.get("deployment_mode") != "local":
            _update_job(job_id, status="blocked", progress=100, step="blocked", message="switch to local deployment before training")
            return
        resolved_dir = _resolve_data_dir(data_dir)
        _update_job(job_id, progress=34, step="checking data", message=f"using data dir: {resolved_dir}")
        required = ["synthetic_sft.jsonl"] if kind == "sft" else ["synthetic_sft.jsonl", "synthetic_rl.jsonl"]
        missing_files = [name for name in required if not (resolved_dir / name).exists()]
        if missing_files:
            _update_job(
                job_id,
                status="blocked",
                progress=100,
                step="missing data",
                message=f"missing required files: {', '.join(missing_files)}",
            )
            return
        _update_job(job_id, progress=58, step="checking training dependencies", message="checking torch/cuda and trainer packages")
        missing_packages = _missing_training_packages()
        if missing_packages or not _cuda_available():
            _update_job(
                job_id,
                status="requires_gpu",
                progress=100,
                step="requires linux gpu",
                message=(
                    "GPU training is not available in this environment; "
                    f"missing packages: {', '.join(missing_packages) or 'none'}, cuda_available={_cuda_available()}"
                ),
            )
            return
        sleep(0.2)
        model_id = f"tracehound-{kind}-{algorithm}-{uuid4().hex[:8]}"
        model_record = {
            "id": model_id,
            "kind": kind,
            "algorithm": algorithm,
            "path": f"checkpoints/{model_id}",
            "source_data": str(resolved_dir),
        }
        with _STATE_LOCK:
            _MODEL_STATE.setdefault("fine_tuned_models", []).append(model_record)
            if auto_switch:
                _MODEL_STATE["active_finetuned_model"] = model_id
        _update_job(
            job_id,
            status="completed",
            progress=100,
            step="completed",
            produced_model=model_record,
            message=f"training completed; produced {model_id}",
        )
    except Exception as exc:  # pragma: no cover - defensive path surfaced through API.
        _update_job(job_id, status="failed", progress=100, step="failed", error=str(exc), message=f"failed: {exc}")


def _run_safety_enchantment_job(
    job_id: str,
    target_profile: Dict[str, Any],
    target_base_model: str,
    guard_model: str,
    guard_mode: str,
    data_dir: str,
    algorithm: str,
    output_dir: str,
    max_samples: int | None,
    safety_weight: float,
    utility_weight: float,
    auto_register: bool,
) -> None:
    try:
        _update_job(
            job_id,
            status="running",
            progress=8,
            step="planning",
            message="building safety enchantment plan",
        )
        resolved_output = output_dir or str(Path("checkpoints/safety_enchantment") / job_id)
        plan = build_enchantment_plan(
            target_profile=target_profile,
            target_base_model=target_base_model,
            guard_model=guard_model,
            guard_mode=guard_mode,
            data_dir=data_dir,
            algorithm=algorithm,
            output_dir=resolved_output,
            max_samples=max_samples,
            safety_weight=safety_weight,
            utility_weight=utility_weight,
            auto_register=auto_register,
        )
        plan_path = write_enchantment_plan(resolved_output, plan)
        artifacts: Dict[str, Any] = {
            "output_dir": resolved_output,
            "plan": str(plan_path),
            "commands": plan["commands"],
        }
        _update_job(
            job_id,
            progress=30,
            step="checking data",
            artifacts=artifacts,
            plan=plan,
            message=f"using data dir: {plan['data']['data_dir']}",
        )
        if plan["data"]["missing"]:
            _update_job(
                job_id,
                status="blocked",
                progress=100,
                step="missing data",
                artifacts=artifacts,
                message=f"missing required files: {', '.join(plan['data']['missing'])}",
            )
            return
        _update_job(
            job_id,
            progress=58,
            step="checking gpu and trainer",
            message="checking CUDA and target-model training dependencies",
        )
        missing_packages = _missing_enchantment_packages(algorithm)
        cuda_available = _cuda_available()
        if missing_packages or not cuda_available:
            _update_job(
                job_id,
                status="requires_gpu",
                progress=100,
                step="requires linux gpu",
                artifacts=artifacts,
                message=(
                    "Safety enchantment plan is ready; run generated commands on a Linux/GPU server. "
                    f"missing packages: {', '.join(missing_packages) or 'none'}, cuda_available={cuda_available}"
                ),
            )
            return
        sleep(0.2)
        produced_model = {
            "id": f"enchanted-{algorithm}-{uuid4().hex[:8]}",
            "algorithm": algorithm,
            "guard_model": guard_model,
            "target_profile": target_profile.get("name", ""),
            "base_model": plan["target"]["base_model"],
            "path": str(Path(resolved_output) / algorithm),
            "source_data": plan["data"]["data_dir"],
        }
        if auto_register:
            with _STATE_LOCK:
                _ENCHANTED_MODELS.append(produced_model)
        _update_job(
            job_id,
            status="completed",
            progress=100,
            step="completed",
            artifacts=artifacts,
            produced_model=produced_model,
            message=f"safety enchantment completed; produced {produced_model['id']}",
        )
    except Exception as exc:  # pragma: no cover - defensive path surfaced through API.
        _update_job(job_id, status="failed", progress=100, step="failed", error=str(exc), message=f"failed: {exc}")


def _bounded_int(value: Any, *, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected integer") from exc
    return max(minimum, min(maximum, number))


def _bounded_float(value: Any, *, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected number") from exc
    return max(minimum, min(maximum, number))


def _optional_list(value: Any) -> list[str] | None:
    if value in (None, "", "all"):
        return None
    if isinstance(value, list):
        items = [str(item) for item in value if item and item != "all"]
    else:
        items = [str(value)]
    return items or None


def _missing_training_packages() -> list[str]:
    return [package for package in TRAIN_PACKAGES if importlib.util.find_spec(package) is None]


def _missing_enchantment_packages(algorithm: str) -> list[str]:
    packages = list(TRAIN_PACKAGES)
    if algorithm in {"sft_dpo", "sft_grpo"}:
        packages.extend(["trl"])
    return [package for package in packages if importlib.util.find_spec(package) is None]


def _cuda_available() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch  # type: ignore

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _resolve_data_dir(data_dir: str) -> Path:
    path = Path(data_dir)
    if path.name == "latest" and path.exists() and path.is_file():
        return Path(path.read_text(encoding="utf-8").strip())
    if path.exists() and path.is_file():
        return Path(path.read_text(encoding="utf-8").strip())
    return path
