#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/taxonomy_lora"
RUN="qwen35_08b_taxonomy_lora_lr8e-6_steps200"
MODEL="/root/autodl-tmp/models/Qwen3.5-0.8B"
DATA="$ROOT/data/agentdog_complete_binary_safe_augmented_unsafe_train.json"
OUT="$ROOT/outputs/$RUN"
LOG="$ROOT/logs/$RUN.log"

mkdir -p "$ROOT/logs" "$OUT"

echo "$(date -Is) starting $RUN"
/root/miniconda3/bin/python "$ROOT/train_qwen_taxonomy_lora.py" \
  --model-path "$MODEL" \
  --data-file "$DATA" \
  --output-dir "$OUT" \
  --learning-rate 8e-6 \
  --max-steps 200 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 8192 \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --save-steps 150 200 \
  --eval-steps 150 200 \
  --save-total-limit 2 \
  --val-ratio 0.2 \
  --seed 42 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  2>&1 | tee "$LOG"

echo "$(date -Is) done $RUN"
