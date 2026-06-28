#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${TRACEHOUND_ENV_FILE:-.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

HOST="${TRACEHOUND_HOST:-0.0.0.0}"
PORT="${TRACEHOUND_PORT:-8000}"

echo "[tracehound] starting demo on ${HOST}:${PORT}"
python scripts/serve_demo.py --host "$HOST" --port "$PORT"
