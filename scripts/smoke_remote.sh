#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SMOKE_DIR="${TRACEHOUND_SMOKE_DIR:-data/tmp/remote_smoke}"
SMOKE_COUNT="${TRACEHOUND_SMOKE_COUNT:-64}"
REPORT_JSON="${TRACEHOUND_SMOKE_REPORT_JSON:-reports/remote_smoke_experiments.json}"
REPORT_MD="${TRACEHOUND_SMOKE_REPORT_MD:-reports/remote_smoke_report.md}"

echo "[tracehound] gpu/server diagnostics"
python scripts/gpu_doctor.py || true
python scripts/list_model_profiles.py || true

echo "[tracehound] generating smoke dataset: ${SMOKE_COUNT} samples"
python scripts/generate_data.py --config configs/generation.yaml --out "$SMOKE_DIR" --count "$SMOKE_COUNT"

echo "[tracehound] checking dataset quality"
python scripts/quality_check.py "$SMOKE_DIR/synthetic_eval.jsonl"

echo "[tracehound] running tests"
pytest

echo "[tracehound] running layered baseline"
python scripts/evaluate.py "$SMOKE_DIR/synthetic_eval.jsonl" --mode layered

echo "[tracehound] running offline ablations"
python scripts/run_experiments.py --data "$SMOKE_DIR/synthetic_eval.jsonl" --output "$REPORT_JSON" --no-api
python scripts/generate_report.py --input "$REPORT_JSON" --output "$REPORT_MD"

echo "[tracehound] smoke complete"
echo "[tracehound] report: $REPORT_MD"
