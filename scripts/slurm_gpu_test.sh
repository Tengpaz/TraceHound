#!/usr/bin/env bash
set -euo pipefail

export TRACEHOUND_SLURM_JOB_NAME="${TRACEHOUND_SLURM_JOB_NAME:-TH_GPU_TEST}"
export TRACEHOUND_SLURM_GPUS="${TRACEHOUND_SLURM_GPUS:-1}"
export TRACEHOUND_SLURM_NTASKS="${TRACEHOUND_SLURM_NTASKS:-1}"
export TRACEHOUND_SLURM_NTASKS_PER_NODE="${TRACEHOUND_SLURM_NTASKS_PER_NODE:-1}"

bash scripts/slurm_run.sh "$@" 'hostname; nvidia-smi; python scripts/gpu_doctor.py --strict'
