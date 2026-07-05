#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_NAME="${TRACEHOUND_ENV_NAME:-tracehound-gpu}"
BASE_MODEL="${TRACEHOUND_BASE_MODEL:-Qwen/Qwen3.5-0.8B}"
RUN_ID="${TRACEHOUND_RUN_ID:-qwen3_5_0_8b_lite_binary_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${TRACEHOUND_OUTPUT_DIR:-checkpoints/${RUN_ID}}"
REPORT_DIR="${TRACEHOUND_REPORT_DIR:-reports/${RUN_ID}}"
DATA="${TRACEHOUND_LITE_BINARY_DATA:-data/release/AgentDoG-Lite-TrainningDataset-Binary/messages/train.jsonl}"
DATASET_ROOT="${TRACEHOUND_EVAL_DATASET_ROOT:-external/agentdog_official/datasets/summer_camp_teseset}"
MAX_SEQ_LENGTH="${TRACEHOUND_MAX_SEQ_LENGTH:-4096}"
MAX_INPUT_TOKENS="${TRACEHOUND_MAX_INPUT_TOKENS:-8192}"
MAX_NEW_TOKENS="${TRACEHOUND_MAX_NEW_TOKENS:-32}"
EPOCHS="${TRACEHOUND_EPOCHS:-1}"
GRAD_ACCUM="${TRACEHOUND_GRAD_ACCUM:-16}"
LR="${TRACEHOUND_LR:-2e-4}"
MAX_TRAIN_SAMPLES="${TRACEHOUND_MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${TRACEHOUND_MAX_VAL_SAMPLES:-}"
SAVE_STEPS="${TRACEHOUND_SAVE_STEPS:-100}"
EVAL_STEPS="${TRACEHOUND_EVAL_STEPS:-100}"
DATALOADER_WORKERS="${TRACEHOUND_DATALOADER_WORKERS:-0}"

mkdir -p logs checkpoints reports
LOG_PATH="logs/${RUN_ID}.log"

if command -v conda >/dev/null 2>&1; then
  CONDA_RUN=(conda run -n "$ENV_NAME" --no-capture-output)
else
  CONDA_RUN=()
fi

TRAIN_SAMPLE_ARGS=()
if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
  TRAIN_SAMPLE_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "$MAX_VAL_SAMPLES" ]]; then
  TRAIN_SAMPLE_ARGS+=(--max-val-samples "$MAX_VAL_SAMPLES")
fi

{
  echo "[tracehound] run_id=$RUN_ID"
  echo "[tracehound] base_model=$BASE_MODEL"
  echo "[tracehound] output_dir=$OUTPUT_DIR"
  echo "[tracehound] report_dir=$REPORT_DIR"
  echo "[tracehound] data=$DATA"
  echo "[tracehound] nvidia-smi"
  nvidia-smi || true

  "${CONDA_RUN[@]}" python scripts/train_lite_binary_lora.py \
    --data "$DATA" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --max-seq-length "$MAX_SEQ_LENGTH" \
    --gradient-accumulation-steps "$GRAD_ACCUM" \
    --learning-rate "$LR" \
    --num-train-epochs "$EPOCHS" \
    --save-steps "$SAVE_STEPS" \
    --eval-steps "$EVAL_STEPS" \
    --save-total-limit 1 \
    --dataloader-num-workers "$DATALOADER_WORKERS" \
    "${TRAIN_SAMPLE_ARGS[@]}" \
    --run

  "${CONDA_RUN[@]}" python scripts/evaluate_lite_binary_model.py \
    --base-model "$BASE_MODEL" \
    --adapter "$OUTPUT_DIR" \
    --dataset-root "$DATASET_ROOT" \
    --download-dataset \
    --output-dir "$REPORT_DIR" \
    --datasets atbench,rjudge \
    --max-input-tokens "$MAX_INPUT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS"

  echo "[tracehound] finished: $REPORT_DIR/summary.json"
} 2>&1 | tee "$LOG_PATH"
