"""FastAPI application factory for the TraceHound demo."""

from pathlib import Path
from typing import Any, Dict

from traceguard.compressor import compress_trajectory
from traceguard.data import built_in_cases
from traceguard.guard import TraceGuard
from traceguard.judge import build_remote_judge
from traceguard.pipeline import evaluate_case
from traceguard.schema import TrajectoryCase, TrajectoryStep, dump_model


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
        case_payload = {key: raw_case[key] for key in ("id", "task", "metadata", "trajectory") if key in raw_case}
        case = TrajectoryCase.model_validate(case_payload)
        judge = None
        if judge_name in {"api", "hybrid"}:
            try:
                judge = build_remote_judge(judge=judge_name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        elif judge_name != "heuristic":
            raise HTTPException(status_code=400, detail=f"unknown judge: {judge_name}")
        report = evaluate_case(case, mode=mode, judge=judge)
        compressed = compress_trajectory(case)
        return {
            "report": dump_model(report),
            "compressed": dump_model(compressed),
            "guard": _simulate_guard(case),
        }

    return app


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
