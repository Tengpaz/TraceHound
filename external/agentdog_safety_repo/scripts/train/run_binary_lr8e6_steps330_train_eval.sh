#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/sft_dedup_binary"
RUN="binary_dedup_lr8e-6_steps330"
MODEL_DIR="$ROOT/outputs/models/$RUN"
LOG_DIR="$ROOT/logs"
TRAIN_LOG="$LOG_DIR/$RUN.log"
CKPT="$MODEL_DIR/checkpoint-330"

mkdir -p "$LOG_DIR" "$MODEL_DIR" "$ROOT/outputs/val_eval/$RUN"

echo "$(date -Is) starting training $RUN"
/root/miniconda3/bin/python "$ROOT/train_qwen_binary_sft_maxsteps.py" \
  --model-path /root/autodl-tmp/models/Qwen3.5-0.8B \
  --train-file "$ROOT/data/agentdog_binary_dedup_train.jsonl" \
  --output-dir "$MODEL_DIR" \
  --learning-rate 8e-6 \
  --num-train-epochs 2 \
  --max-steps 330 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --max-length 8192 \
  --warmup-ratio 0.03 \
  --logging-steps 10 \
  --save-steps 25 \
  --save-total-limit 1 \
  --seed 42 \
  2>&1 | tee "$TRAIN_LOG"

if [[ ! -d "$CKPT" ]]; then
  echo "Expected checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "$(date -Is) running clean validation"
/root/miniconda3/bin/python /root/autodl-tmp/sft_basic_lr1e-5/evaluate_binary_val.py \
  --model-path "$CKPT" \
  --validation-file "$ROOT/data/agentdog_binary_dedup_val.jsonl" \
  --output-dir "$ROOT/outputs/val_eval/$RUN/checkpoint-330" \
  --max-new-tokens 32 \
  > "$ROOT/outputs/val_eval/$RUN/checkpoint-330/eval.log" 2>&1

echo "$(date -Is) running summer camp testset eval"
mkdir -p /root/autodl-tmp/eval_qwen35_08b_sft_dedup_lr8e-6_steps330_v4_exact_prompt
/root/miniconda3/bin/python /root/autodl-tmp/binary_safety_eval.py \
  --model-path "$CKPT" \
  --input-jsonl /root/autodl-tmp/eval_qwen35_08b_baseline_v4_no_think/predictions.jsonl \
  --output-dir /root/autodl-tmp/eval_qwen35_08b_sft_dedup_lr8e-6_steps330_v4_exact_prompt \
  > /root/autodl-tmp/eval_qwen35_08b_sft_dedup_lr8e-6_steps330_v4_exact_prompt/run.log 2>&1

echo "$(date -Is) running R-Judge eval"
mkdir -p /root/autodl-tmp/eval_rjudge_sft_dedup_lr8e-6_steps330_original_prompt
/root/miniconda3/bin/python /root/autodl-tmp/rjudge_binary_safety_eval.py \
  --model-path "$CKPT" \
  --input-json /root/autodl-tmp/datasets/summer_camp_rjudge/summer_camp_rjudge.json \
  --output-dir /root/autodl-tmp/eval_rjudge_sft_dedup_lr8e-6_steps330_original_prompt \
  > /root/autodl-tmp/eval_rjudge_sft_dedup_lr8e-6_steps330_original_prompt/run.log 2>&1

echo "$(date -Is) all done $RUN"
