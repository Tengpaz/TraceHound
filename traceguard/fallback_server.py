"""Dependency-free demo server used when FastAPI/uvicorn are unavailable."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, unquote, urlparse

from traceguard.compressor import compress_trajectory
from traceguard.config import api_runtime_status
from traceguard.data import built_in_cases
from traceguard.demo_app import (
    _build_judge_or_400,
    _guardrail_integration_examples,
    _model_profiles_status,
    _model_status,
    _simulate_guard,
)
from traceguard.guardrail_gateway import evaluate_guardrail_event
from traceguard.pipeline import evaluate_case
from traceguard.schema import TrajectoryCase, dump_model


CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}


class DemoRequestHandler(BaseHTTPRequestHandler):
    root = Path(__file__).resolve().parent.parent
    web_dir = root / "web_demo"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._send_file(self.web_dir / "index.html")
            return
        if path.startswith("/static/"):
            relative = path.removeprefix("/static/")
            self._send_file(self.web_dir / relative)
            return
        if path == "/api/cases":
            self._send_json(
                [
                    {
                        "id": case["id"],
                        "task": case.get("task", ""),
                        "scenario": case.get("metadata", {}).get("scenario", ""),
                        "gold_label": case.get("gold", {}).get("label", ""),
                    }
                    for case in built_in_cases()
                ]
            )
            return
        if path == "/api/runtime":
            self._send_json(
                {
                    "api": api_runtime_status(),
                    "judges": [
                        {"id": "heuristic", "label": "Heuristic", "remote": False},
                        {"id": "local-binary", "label": "TraceHound-Base Local", "remote": False},
                        {"id": "local", "label": "Local JSON Report", "remote": False},
                        {"id": "hybrid", "label": "Hybrid API", "remote": True},
                        {"id": "api", "label": "API Only", "remote": True},
                    ],
                    "model": _model_status(),
                    "model_profiles": _model_profiles_status(),
                }
            )
            return
        if path == "/api/guardrail/integrations":
            self._send_json(_guardrail_integration_examples())
            return
        if path.startswith("/api/cases/"):
            case_id = unquote(path.removeprefix("/api/cases/"))
            for case in built_in_cases():
                if case["id"] == case_id:
                    self._send_json(case)
                    return
            self._send_json({"detail": f"unknown case id: {case_id}"}, status=404)
            return
        self._send_json({"detail": "not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if path == "/api/guardrail/event":
            self._handle_guardrail_event(body)
            return
        if path == "/api/guardrail/claude-code":
            query = parse_qs(parsed.query)
            mode = query.get("mode", ["layered"])[0]
            judge_name = query.get("judge", ["heuristic"])[0]
            self._handle_claude_code_event(body, mode=mode, judge_name=judge_name)
            return
        if path != "/api/evaluate":
            self._send_json({"detail": "not found"}, status=404)
            return
        mode = body.get("mode", "layered")
        judge_name = body.get("judge", "heuristic")
        raw_case: Dict[str, Any] = body.get("case", body)
        case_payload = {key: raw_case[key] for key in ("id", "task", "metadata", "trajectory") if key in raw_case}
        case = TrajectoryCase.model_validate(case_payload)
        try:
            judge = _fallback_build_judge(judge_name)
        except FallbackHTTPException as exc:
            self._send_json({"detail": exc.detail}, status=exc.status_code)
            return
        report = evaluate_case(case, mode=mode, judge=judge)
        compressed = compress_trajectory(case)
        self._send_json(
            {
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
        )

    def _handle_guardrail_event(self, body: Dict[str, Any]) -> None:
        platform = str(body.get("platform") or "auto")
        event_type = str(body.get("event_type") or "auto")
        mode = str(body.get("mode") or "layered")
        judge_name = str(body.get("judge") or "heuristic")
        raw_event = body.get("event") if isinstance(body.get("event"), dict) else body
        try:
            judge = _fallback_build_judge(judge_name)
            result = evaluate_guardrail_event(raw_event, platform=platform, event_type=event_type, mode=mode, judge=judge)
        except FallbackHTTPException as exc:
            self._send_json({"detail": exc.detail}, status=exc.status_code)
            return
        except (TypeError, ValueError, KeyError) as exc:
            self._send_json({"detail": str(exc)}, status=400)
            return
        self._send_json(result)

    def _handle_claude_code_event(self, body: Dict[str, Any], *, mode: str, judge_name: str) -> None:
        try:
            judge = _fallback_build_judge(judge_name)
            result = evaluate_guardrail_event(body, platform="claude-code", event_type="auto", mode=mode, judge=judge)
        except FallbackHTTPException as exc:
            self._send_json({"detail": exc.detail}, status=exc.status_code)
            return
        except (TypeError, ValueError, KeyError) as exc:
            self._send_json({"detail": str(exc)}, status=400)
            return
        self._send_json(result.get("adapter", {}).get("json") or {})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"detail": "not found"}, status=404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_fallback_server(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), DemoRequestHandler)
    print(f"TraceHound fallback demo running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


class FallbackHTTPException(Exception):
    def __init__(self, *, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _fallback_build_judge(judge_name: str) -> Any:
    return _build_judge_or_400(judge_name, FallbackHTTPException)
