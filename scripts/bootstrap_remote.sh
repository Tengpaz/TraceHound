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

ENV_NAME="${TRACEHOUND_ENV_NAME:-tracehound-gpu}"
PYTHON_VERSION="${TRACEHOUND_PYTHON_VERSION:-3.10}"
INSTALL_TRAIN="${TRACEHOUND_INSTALL_TRAIN:-1}"
RUN_SMOKE="${TRACEHOUND_RUN_SMOKE:-1}"
START_DEMO="${TRACEHOUND_START_DEMO:-0}"
HOST="${TRACEHOUND_HOST:-0.0.0.0}"
PORT="${TRACEHOUND_PORT:-8000}"
SKIP_TORCH="${TRACEHOUND_SKIP_TORCH:-0}"
TORCH_INDEX_URL="${TRACEHOUND_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

find_conda() {
  if [[ -n "${TRACEHOUND_CONDA_EXE:-}" ]]; then
    echo "$TRACEHOUND_CONDA_EXE"
    return
  fi
  command -v mamba || command -v conda || command -v micromamba || true
}

CONDA_EXE="$(find_conda)"
if [[ -z "$CONDA_EXE" ]]; then
  echo "No conda-compatible executable found. Install Miniconda, Mambaforge, mamba, or micromamba first." >&2
  exit 1
fi

if "$CONDA_EXE" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[tracehound] conda env exists: $ENV_NAME"
else
  echo "[tracehound] creating conda env: $ENV_NAME"
  "$CONDA_EXE" create -y -n "$ENV_NAME" -c conda-forge "python=$PYTHON_VERSION" pip git
fi

if command -v conda >/dev/null 2>&1; then
  CONDA_RUN=(conda run -n "$ENV_NAME")
else
  CONDA_RUN=("$CONDA_EXE" run -n "$ENV_NAME")
fi

echo "[tracehound] upgrading pip"
"${CONDA_RUN[@]}" python -m pip install --upgrade pip

if [[ "$SKIP_TORCH" == "1" ]]; then
  echo "[tracehound] skipping torch install because TRACEHOUND_SKIP_TORCH=1"
else
  echo "[tracehound] installing CUDA PyTorch wheels from $TORCH_INDEX_URL"
  "${CONDA_RUN[@]}" python -m pip install --index-url "$TORCH_INDEX_URL" torch torchvision torchaudio
fi

EXTRAS=".[dev]"
if [[ "$INSTALL_TRAIN" == "1" ]]; then
  EXTRAS=".[dev,train]"
fi

echo "[tracehound] installing project: $EXTRAS"
"${CONDA_RUN[@]}" python -m pip install -e "$EXTRAS"

echo "[tracehound] environment diagnostic"
"${CONDA_RUN[@]}" python scripts/gpu_doctor.py || true

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "[tracehound] running smoke checks"
  "${CONDA_RUN[@]}" bash scripts/smoke_remote.sh
fi

echo
echo "[tracehound] bootstrap complete"
echo "[tracehound] activate with: conda activate $ENV_NAME"
echo "[tracehound] start demo with: bash scripts/run_remote_demo.sh"

if [[ "$START_DEMO" == "1" ]]; then
  echo "[tracehound] starting demo on ${HOST}:${PORT}"
  exec "${CONDA_RUN[@]}" python scripts/serve_demo.py --host "$HOST" --port "$PORT"
fi
