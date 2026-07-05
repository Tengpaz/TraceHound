#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/training_defaults.json}"
EXTRA_ARGS=()

[[ -n "${MODEL_PATH:-}" ]] && EXTRA_ARGS+=(--model-path "${MODEL_PATH}")
[[ -n "${MAX_LENGTH:-}" ]] && EXTRA_ARGS+=(--max-length "${MAX_LENGTH}")
[[ -n "${EPOCHS:-}" ]] && EXTRA_ARGS+=(--num-train-epochs "${EPOCHS}")
[[ -n "${MICRO_BATCH:-}" ]] && EXTRA_ARGS+=(--per-device-train-batch-size "${MICRO_BATCH}")
[[ -n "${GRAD_ACCUM:-}" ]] && EXTRA_ARGS+=(--gradient-accumulation-steps "${GRAD_ACCUM}")
[[ -n "${LR:-}" ]] && EXTRA_ARGS+=(--learning-rate "${LR}")
[[ -n "${LR_SCHEDULER_TYPE:-}" ]] && EXTRA_ARGS+=(--lr-scheduler-type "${LR_SCHEDULER_TYPE}")
[[ -n "${DEEPSPEED_CONFIG:-}" ]] && EXTRA_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
[[ -n "${REPORT_TO:-}" ]] && EXTRA_ARGS+=(--report-to "${REPORT_TO}")
[[ -n "${WANDB_PROJECT:-}" ]] && EXTRA_ARGS+=(--wandb-project "${WANDB_PROJECT}")
[[ -n "${WANDB_ENTITY:-}" ]] && EXTRA_ARGS+=(--wandb-entity "${WANDB_ENTITY}")
[[ -n "${WANDB_MODE:-}" ]] && EXTRA_ARGS+=(--wandb-mode "${WANDB_MODE}")
[[ -n "${WANDB_WATCH:-}" ]] && EXTRA_ARGS+=(--wandb-watch "${WANDB_WATCH}")
[[ -n "${EVAL_STEPS:-}" ]] && EXTRA_ARGS+=(--eval-steps "${EVAL_STEPS}")
[[ -n "${SEED:-}" ]] && EXTRA_ARGS+=(--seed "${SEED}" --data-seed "${SEED}")

python scripts/prepare_sft_data.py --task binary --output-dir outputs/data
python scripts/prepare_sft_data.py --task taxonomy --output-dir outputs/data
python scripts/prepare_sft_data.py --task combined --output-dir outputs/data

python scripts/train_sft.py \
  --config "${CONFIG}" \
  --task binary \
  --train-file outputs/data/agentdog_binary_sft.jsonl \
  --output-dir outputs/models/binary \
  --wandb-run-name "binary-sft" \
  "${EXTRA_ARGS[@]}"

python scripts/train_sft.py \
  --config "${CONFIG}" \
  --task taxonomy \
  --train-file outputs/data/agentdog_taxonomy_sft.jsonl \
  --output-dir outputs/models/taxonomy \
  --wandb-run-name "taxonomy-sft" \
  "${EXTRA_ARGS[@]}"

python scripts/train_sft.py \
  --config "${CONFIG}" \
  --task combined \
  --train-file outputs/data/agentdog_combined_sft.jsonl \
  --output-dir outputs/models/combined \
  --wandb-run-name "combined-sft" \
  "${EXTRA_ARGS[@]}"

python scripts/evaluate_atbench.py --task binary --model-path outputs/models/binary --output-dir outputs/eval/binary/final
python scripts/evaluate_atbench.py --task taxonomy --model-path outputs/models/taxonomy --output-dir outputs/eval/taxonomy/final
python scripts/evaluate_atbench.py --task combined --model-path outputs/models/combined --output-dir outputs/eval/combined/final
