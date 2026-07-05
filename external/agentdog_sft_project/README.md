# Agent Safety Guardrail SFT

This project fine-tunes local Qwen3.5 guardrail models on AgentDoG data and evaluates them on ATBench500. It does not call external model APIs.

## Environment

Activate the training environment before running data preparation, training, or evaluation:

```bash
cd /root/agentdog_sft
source /root/miniconda3/etc/profile.d/conda.sh
conda activate train
```

The current machine has four A800 80GB GPUs available. The local 0.8B checkpoint is stored at:

```text
models/Qwen3.5-0.8B
```

## Prepare Data

```bash
python scripts/prepare_sft_data.py --task binary --output-dir outputs/data
python scripts/prepare_sft_data.py --task taxonomy --output-dir outputs/data
python scripts/prepare_sft_data.py --task combined --output-dir outputs/data
```

## Train

Training defaults are loaded from `configs/training_defaults.json`. Command-line arguments override the JSON values.
The default config may point at a different local checkpoint, so pass `--model-path models/Qwen3.5-0.8B` when training the 0.8B model in this workspace.
`eval_steps` controls how often the current in-memory model is switched to eval mode and evaluated on ATBench500 during training. These metrics are logged through the Trainer logger, so they are written to TensorBoard/W&B when those reporters are enabled.
The default learning-rate schedule is cosine decay with warmup, controlled by `lr_scheduler_type: "cosine"` and `warmup_ratio` in `configs/training_defaults.json`.
By default, training reports to TensorBoard and Weights & Biases. W&B logs training curves and system metrics such as CPU, GPU utilization, and GPU memory. Run `wandb login` once before online logging, or use `--wandb-mode offline`.
During training, ATBench accuracy is logged as `atbench/accuracy`, so W&B will show its step-by-step curve automatically.

```bash
python scripts/train_sft.py \
  --config configs/training_defaults.json \
  --task binary \
  --train-file outputs/data/agentdog_binary_sft.jsonl \
  --output-dir outputs/models/binary \
  --wandb-run-name binary-sft
```

### Four-GPU Qwen3.5-0.8B Training

Use `torchrun` with four local processes. In this environment, use `configs/deepspeed_zero2_torch_adam.json` instead of the default ZeRO-2 config because the default DeepSpeed fused Adam path tries to JIT-compile an extension that is not compatible with the installed CUDA compiler.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_sft.py \
  --config configs/training_defaults.json \
  --model-path models/Qwen3.5-0.8B \
  --task binary \
  --train-file outputs/data/agentdog_binary_sft.jsonl \
  --output-dir outputs/models/qwen35-0.8b-binary \
  --deepspeed configs/deepspeed_zero2_torch_adam.json \
  --report-to tensorboard \
  --wandb-mode disabled
```

For a quick smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_sft.py \
  --config configs/training_defaults.json \
  --model-path models/Qwen3.5-0.8B \
  --task binary \
  --train-file outputs/data/agentdog_binary_sft.jsonl \
  --output-dir outputs/models/qwen35-0.8b-binary-smoke \
  --max-steps 1 \
  --save-steps 1 \
  --logging-steps 1 \
  --no-eval-atbench-during-training \
  --deepspeed configs/deepspeed_zero2_torch_adam.json \
  --report-to none \
  --wandb-mode disabled
```

With the defaults, the effective global train batch size is:

```text
4 GPUs * per_device_train_batch_size 1 * gradient_accumulation_steps 8 = 32
```

Set `num_checkpoints` to `0` to disable intermediate checkpoint saves. In that mode, training only writes the final model at `--output-dir` after `trainer.train()` completes.

### Six-Label SFT

Prepare the six-label SFT data from the augmented AgentDoG JSONL. The output keeps only chat `messages`, using `src/guardrail/6 labels training prompt.md` as the prompt template and the six label fields as the assistant JSON target.

```bash
python scripts/prepare_6label_sft_data.py \
  --input data/agentdog_complete_binary_safe_augmented_unsafe_train.jsonl \
  --output outputs/data/agentdog_6label_sft.jsonl \
  --template "src/guardrail/6 labels training prompt.md"
```

Run four-GPU SFT without ATBench evaluation:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_6label_sft.py \
  --config configs/training_6label_defaults.json
```

For a quick smoke test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_6label_sft.py \
  --config configs/training_6label_defaults.json \
  --output-dir outputs/models/qwen35-0.8b-6label-smoke \
  --max-steps 1 \
  --logging-steps 1 \
  --report-to none \
  --wandb-mode disabled
```

Disable W&B if needed:

```bash
python scripts/train_sft.py ... --report-to tensorboard
```

Use `scripts/run_all.sh` to run data preparation, all three training jobs, and final evaluation. It uses `configs/training_defaults.json` by default. Set `CONFIG=path/to/config.json` to use another config.

To inspect the final effective configuration without loading the model:

```bash
python scripts/train_sft.py \
  --config configs/training_defaults.json \
  --task binary \
  --train-file outputs/data/agentdog_binary_sft.jsonl \
  --output-dir /tmp/print_config \
  --print-config
```

For reproducibility, `seed`, `data_seed`, and `full_determinism` are configured in `configs/training_defaults.json`.

## Evaluate

```bash
python scripts/evaluate_atbench.py \
  --task binary \
  --model-path outputs/models/binary \
  --output-dir outputs/eval/binary/final

python scripts/evaluate_atbench.py \
  --task taxonomy \
  --model-path outputs/models/taxonomy \
  --output-dir outputs/eval/taxonomy/final

python scripts/evaluate_atbench.py \
  --task combined \
  --model-path outputs/models/combined \
  --output-dir outputs/eval/combined/final
```

BinarySafety reports only accuracy. FineGrainedTaxonomy is evaluated only on unsafe ATBench500 samples.

## Six-Label ATBench300 Evaluation

Evaluate a six-label checkpoint on the local summer-camp ATBench300 set:

```bash
python scripts/binary_safety_eval.py \
  --model-path outputs/models/qwen35-0.8b-6label \
  --input-json 2026_summer_camp_teseset/summer_camp_ATBench300.json \
  --output-dir outputs/eval/qwen35-0.8b-6label-atbench300 \
  --max-new-tokens 512
```

For pass@k safe/unsafe evaluation, use the vLLM backend. The pass condition is: at least one rollout predicts the correct `judgment`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/binary_safety_eval.py \
  --model-path outputs/models/qwen35-0.8b-6label \
  --input-json 2026_summer_camp_teseset/summer_camp_ATBench300.json \
  --output-dir outputs/eval/qwen35-0.8b-6label-atbench300-pass5 \
  --backend vllm \
  --pass-k 5 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-new-tokens 512
```

The metrics include safe/unsafe accuracy and F1, source accuracy and macro-F1, three taxonomy dimensions with accuracy and macro-F1, taxonomy exact match, `pass_at_k_binary_judgment`, and token cost summaries.

If vLLM is unavailable, use four independent HuggingFace processes, one per GPU. Each process evaluates one shard and can batch multiple prompts on its GPU:

```bash
MODEL=outputs/models/qwen35-0.8b-6label
BASE=outputs/eval/qwen35-0.8b-6label-atbench300-pass5-hf

for I in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$I python scripts/binary_safety_eval.py \
    --model-path $MODEL \
    --input-json 2026_summer_camp_teseset/summer_camp_ATBench300.json \
    --output-dir ${BASE}/shard-${I} \
    --backend hf \
    --num-shards 4 \
    --shard-index $I \
    --batch-size 4 \
    --pass-k 5 \
    --temperature 0.7 \
    --top-p 0.95 \
    --max-new-tokens 512 &
done
wait

python scripts/merge_eval_shards.py \
  --kind atbench \
  --input-dirs ${BASE}/shard-0 ${BASE}/shard-1 ${BASE}/shard-2 ${BASE}/shard-3 \
  --output-dir ${BASE}/merged
```

To run pass@k on the six-label SFT training set itself, using `judgment` and `source` as metrics:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/trainset_passk_eval.py \
  --model-path outputs/models/qwen35-0.8b-6label \
  --train-file outputs/data/agentdog_6label_sft.jsonl \
  --output-dir outputs/eval/qwen35-0.8b-6label-train-pass5 \
  --pass-k 5 \
  --temperature 0.7 \
  --top-p 0.95 \
  --max-new-tokens 512
```

Each row in `predictions.jsonl` records `judgment_success_count` and `source_success_count`, meaning how many of the `k` rollouts predicted the correct `judgment` and `source` for that sample. The legacy `success_count` field is kept as an alias for `judgment_success_count`.

The same training-set pass@k evaluation can be sharded across four GPUs:

```bash
MODEL=outputs/models/qwen35-0.8b-6label
BASE=outputs/eval/qwen35-0.8b-6label-train-pass5-hf

for I in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$I python scripts/trainset_passk_eval.py \
    --model-path $MODEL \
    --train-file outputs/data/agentdog_6label_sft.jsonl \
    --output-dir ${BASE}/shard-${I} \
    --backend hf \
    --num-shards 4 \
    --shard-index $I \
    --batch-size 4 \
    --pass-k 5 \
    --temperature 0.7 \
    --top-p 0.95 \
    --max-new-tokens 512 &
done
wait

python scripts/merge_eval_shards.py \
  --kind trainset \
  --input-dirs ${BASE}/shard-0 ${BASE}/shard-1 ${BASE}/shard-2 ${BASE}/shard-3 \
  --output-dir ${BASE}/merged
```
