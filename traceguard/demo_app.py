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
from traceguard.evaluation import summarize_predictions
from traceguard.export import eval_rows, rl_rows, sft_rows, write_jsonl
from traceguard.generation_config import load_generation_config
from traceguard.guard import TraceGuard
from traceguard.judge import build_remote_judge
from traceguard.model_profiles import list_model_profiles, profile_summary, resolve_model_profile
from traceguard.pipeline import evaluate_case
from traceguard.reporting import build_report, write_metric_chart
from traceguard.schema import RiskReport, TrajectoryCase, TrajectoryStep, dump_model, report_from_gold

TRAIN_PACKAGES = ("torch", "transformers", "datasets", "peft", "accelerate")

_STATE_LOCK = Lock()
_JOBS: Dict[str, Dict[str, Any]] = {}
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
        if not include_eval and not include_sft and not include_rl:
            raise HTTPException(status_code=400, detail="at least one dataset type must be selected")
        job = _new_job(
            "data_generation",
            {
                "count": count,
                "scenarios": scenarios or ["all"],
                "labels": labels or ["all"],
                "include_eval": include_eval,
                "include_sft": include_sft,
                "include_rl": include_rl,
            },
        )
        Thread(
            target=_run_data_generation_job,
            args=(job["id"], count, scenarios, labels, include_eval, include_sft, include_rl),
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
            job.setdefault("messages", []).append(message)


def _run_data_generation_job(
    job_id: str,
    count: int,
    scenarios: list[str] | None,
    labels: list[str] | None,
    include_eval: bool,
    include_sft: bool,
    include_rl: bool,
) -> None:
    try:
        _update_job(job_id, status="running", progress=8, step="building cases", message=f"building {count} cases")
        cases = built_in_cases(scenarios=scenarios, labels=labels, count=count)
        summary = dataset_summary(cases)
        if not cases:
            raise ValueError("filters produced zero cases")
        out_dir = Path("data/tmp/generated") / job_id
        artifacts: Dict[str, str] = {"output_dir": str(out_dir)}
        latest = Path("data/tmp/generated/latest")
        out_dir.mkdir(parents=True, exist_ok=True)
        _update_job(job_id, progress=28, step="writing eval data", message=f"case summary: {summary}")
        if include_eval:
            artifacts["eval"] = str(out_dir / "synthetic_eval.jsonl")
            write_jsonl(out_dir / "synthetic_eval.jsonl", eval_rows(cases))
        _update_job(job_id, progress=55, step="writing sft data", message="eval export complete")
        if include_sft:
            artifacts["sft"] = str(out_dir / "synthetic_sft.jsonl")
            write_jsonl(out_dir / "synthetic_sft.jsonl", sft_rows(cases))
        _update_job(job_id, progress=78, step="writing rl data", message="sft export complete")
        if include_rl:
            artifacts["rl"] = str(out_dir / "synthetic_rl.jsonl")
            write_jsonl(out_dir / "synthetic_rl.jsonl", rl_rows(cases))
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(str(out_dir), encoding="utf-8")
        _update_job(
            job_id,
            status="completed",
            progress=100,
            step="completed",
            artifacts=artifacts,
            summary=summary,
            message="data generation completed",
        )
    except Exception as exc:  # pragma: no cover - defensive path surfaced through API.
        _update_job(job_id, status="failed", progress=100, step="failed", error=str(exc), message=f"failed: {exc}")


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


def _bounded_int(value: Any, *, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected integer") from exc
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
