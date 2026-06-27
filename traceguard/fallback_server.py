"""Dependency-free demo server used when FastAPI/uvicorn are unavailable."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import unquote

from traceguard.compressor import compress_trajectory
from traceguard.data import built_in_cases
from traceguard.demo_app import _simulate_guard
from traceguard.judge import build_remote_judge
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
        if self.path == "/":
            self._send_file(self.web_dir / "index.html")
            return
        if self.path.startswith("/static/"):
            relative = self.path.removeprefix("/static/")
            self._send_file(self.web_dir / relative)
            return
        if self.path == "/api/cases":
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
        if self.path.startswith("/api/cases/"):
            case_id = unquote(self.path.removeprefix("/api/cases/"))
            for case in built_in_cases():
                if case["id"] == case_id:
                    self._send_json(case)
                    return
            self._send_json({"detail": f"unknown case id: {case_id}"}, status=404)
            return
        self._send_json({"detail": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path != "/api/evaluate":
            self._send_json({"detail": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        mode = body.get("mode", "layered")
        judge_name = body.get("judge", "heuristic")
        raw_case: Dict[str, Any] = body.get("case", body)
        case_payload = {key: raw_case[key] for key in ("id", "task", "metadata", "trajectory") if key in raw_case}
        case = TrajectoryCase.model_validate(case_payload)
        try:
            judge = build_remote_judge(judge=judge_name) if judge_name in {"api", "hybrid"} else None
        except ValueError as exc:
            self._send_json({"detail": str(exc)}, status=400)
            return
        report = evaluate_case(case, mode=mode, judge=judge)
        compressed = compress_trajectory(case)
        self._send_json(
            {
                "report": dump_model(report),
                "compressed": dump_model(compressed),
                "guard": _simulate_guard(case),
            }
        )

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
