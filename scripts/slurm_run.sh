#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${TRACEHOUND_ENV_FILE:-.env}"
load_env_defaults() {
  local env_dump
  env_dump="$(mktemp)"
  (
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    env
  ) >"$env_dump"
  while IFS='=' read -r key value; do
    [[ "$key" == TRACEHOUND_* ]] || continue
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done <"$env_dump"
  rm -f "$env_dump"
}

if [[ -f "$ENV_FILE" ]]; then
  load_env_defaults
fi

usage() {
  cat <<'MSG'
Usage:
  bash scripts/slurm_run.sh [--dry-run] '<command>'

Examples:
  bash scripts/slurm_run.sh 'hostname; nvidia-smi; python scripts/gpu_doctor.py --strict'
  TRACEHOUND_SLURM_PARTITION=ai4safe TRACEHOUND_SLURM_GPUS=1 bash scripts/slurm_run.sh 'python scripts/train_sft.py --data data/synthetic_sft.jsonl --model-profile internlm2_5-1_8b-chat --max-samples 32'
  TRACEHOUND_USE_APPTAINER=1 TRACEHOUND_APPTAINER_IMAGE=/path/pytorch.sif bash scripts/slurm_run.sh 'python scripts/gpu_doctor.py --strict'

Key env vars:
  TRACEHOUND_SLURM_PARTITION      default: ai4safe
  TRACEHOUND_SLURM_GPUS           default: 1, set 0 for CPU-only Slurm jobs
  TRACEHOUND_SLURM_NTASKS         default: 1
  TRACEHOUND_SLURM_NTASKS_PER_NODE default: 1
  TRACEHOUND_SLURM_CPUS_PER_TASK  optional, omitted by default
  TRACEHOUND_SLURM_MEM            optional, omitted by default
  TRACEHOUND_SLURM_JOB_NAME       default: TRACEHOUND
  TRACEHOUND_USE_APPTAINER        default: 0
  TRACEHOUND_APPTAINER_IMAGE      required when TRACEHOUND_USE_APPTAINER=1
MSG
}

DRY_RUN="${TRACEHOUND_SLURM_DRY_RUN:-0}"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

COMMAND="${*:-}"
if [[ -z "$COMMAND" ]]; then
  usage >&2
  exit 2
fi

PARTITION="${TRACEHOUND_SLURM_PARTITION:-ai4safe}"
GPUS="${TRACEHOUND_SLURM_GPUS:-1}"
NTASKS="${TRACEHOUND_SLURM_NTASKS:-1}"
NTASKS_PER_NODE="${TRACEHOUND_SLURM_NTASKS_PER_NODE:-1}"
CPUS_PER_TASK="${TRACEHOUND_SLURM_CPUS_PER_TASK:-}"
MEM="${TRACEHOUND_SLURM_MEM:-}"
JOB_NAME="${TRACEHOUND_SLURM_JOB_NAME:-TRACEHOUND}"
MPI="${TRACEHOUND_SLURM_MPI:-}"
TIME_LIMIT="${TRACEHOUND_SLURM_TIME:-}"
EXTRA_SRUN_ARGS="${TRACEHOUND_SLURM_EXTRA_ARGS:-}"
LOG_DIR="${TRACEHOUND_SLURM_LOG_DIR:-log}"
ENV_NAME="${TRACEHOUND_ENV_NAME:-tracehound-gpu}"
CONDA_SH="${TRACEHOUND_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
USE_APPTAINER="${TRACEHOUND_USE_APPTAINER:-0}"
APPTAINER_BIN="${TRACEHOUND_APPTAINER_BIN:-apptainer}"
APPTAINER_IMAGE="${TRACEHOUND_APPTAINER_IMAGE:-}"
APPTAINER_BIND="${TRACEHOUND_APPTAINER_BIND:-$ROOT_DIR:/workspace}"
APPTAINER_WORKDIR="${TRACEHOUND_APPTAINER_WORKDIR:-/workspace}"
APPTAINER_EXTRA_ARGS="${TRACEHOUND_APPTAINER_EXTRA_ARGS:-}"
PYTHONUSERBASE_IN_CONTAINER="${TRACEHOUND_APPTAINER_PYTHONUSERBASE:-$APPTAINER_WORKDIR/pyuser}"
HF_HOME_IN_CONTAINER="${TRACEHOUND_APPTAINER_HF_HOME:-$APPTAINER_WORKDIR/hf_cache}"
TORCH_HOME_IN_CONTAINER="${TRACEHOUND_APPTAINER_TORCH_HOME:-$APPTAINER_WORKDIR/torch_cache}"

mkdir -p "$LOG_DIR"
NOW="$(date +"%Y%m%d_%H%M%S")"
LOG_PATH="$LOG_DIR/${JOB_NAME}-${NOW}.log"

quote() {
  printf "%q" "$1"
}

print_command() {
  printf "%q " "$@"
  printf "\n"
}

SRUN=(srun -p "$PARTITION" -n "$NTASKS" --ntasks-per-node="$NTASKS_PER_NODE" --job-name="$JOB_NAME" --kill-on-bad-exit=1)
if [[ "$GPUS" != "0" ]]; then
  SRUN+=(--gres="gpu:$GPUS")
fi
if [[ -n "$CPUS_PER_TASK" ]]; then
  SRUN+=(--cpus-per-task="$CPUS_PER_TASK")
fi
if [[ -n "$MEM" ]]; then
  SRUN+=(--mem="$MEM")
fi
if [[ -n "$MPI" ]]; then
  SRUN+=(--mpi="$MPI")
fi
if [[ -n "$TIME_LIMIT" ]]; then
  SRUN+=(--time="$TIME_LIMIT")
fi
if [[ -n "$EXTRA_SRUN_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=($EXTRA_SRUN_ARGS)
  SRUN+=("${EXTRA_ARRAY[@]}")
fi

if [[ "$USE_APPTAINER" == "1" ]]; then
  if [[ -z "$APPTAINER_IMAGE" ]]; then
    echo "TRACEHOUND_APPTAINER_IMAGE is required when TRACEHOUND_USE_APPTAINER=1" >&2
    exit 2
  fi
  INNER="cd $(quote "$APPTAINER_WORKDIR") && export PYTHONUSERBASE=$(quote "$PYTHONUSERBASE_IN_CONTAINER") && export PATH=\"\$PYTHONUSERBASE/bin:\$PATH\" && export HF_HOME=$(quote "$HF_HOME_IN_CONTAINER") && export TORCH_HOME=$(quote "$TORCH_HOME_IN_CONTAINER") && $COMMAND"
  APPTAINER=( "$APPTAINER_BIN" exec --nv --bind "$APPTAINER_BIND" )
  if [[ -n "$APPTAINER_EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    APPTAINER_EXTRA_ARRAY=($APPTAINER_EXTRA_ARGS)
    APPTAINER+=("${APPTAINER_EXTRA_ARRAY[@]}")
  fi
  APPTAINER+=( "$APPTAINER_IMAGE" bash -lc "$INNER" )
  FULL=( "${SRUN[@]}" "${APPTAINER[@]}" )
else
  ROOT_QUOTED="$(quote "$ROOT_DIR")"
  CONDA_SH_QUOTED="$(quote "$CONDA_SH")"
  ENV_NAME_QUOTED="$(quote "$ENV_NAME")"
  INNER="cd $ROOT_QUOTED && if [[ -f $CONDA_SH_QUOTED ]]; then source $CONDA_SH_QUOTED && conda activate $ENV_NAME_QUOTED; elif command -v conda >/dev/null 2>&1; then eval \"\$(conda shell.bash hook)\" && conda activate $ENV_NAME_QUOTED; fi; $COMMAND"
  FULL=( "${SRUN[@]}" bash -lc "$INNER" )
fi

echo "[tracehound] slurm log: $LOG_PATH"
if [[ "$DRY_RUN" == "1" ]]; then
  print_command "${FULL[@]}"
  exit 0
fi

"${FULL[@]}" 2>&1 | tee "$LOG_PATH"
