# AgentDoG Safety SFT Repository

This repository collects the scripts, configs, small parameter snapshots, and result summaries used for:

- Binary safety SFT on Qwen3.5-0.8B with `lr=8e-6`, `max_steps=330`.
- Four-dimension taxonomy LoRA training with `lr=8e-6`, `max_steps=200`.
- ATBench and R-Judge evaluation.
- Downstream application checks for email and database guardrails.
- API-based safe-sample augmentation for AgentDoG BinarySafety.

Large model weights and raw datasets are not committed. Put them under `checkpoints/` or `data/` locally.

## Layout

```text
configs/                       Reproducible run configs and path examples
model_params/                  Small model/adapter config snapshots only
scripts/train/                 Binary SFT and taxonomy LoRA training scripts
scripts/eval/                  Unified and binary evaluation scripts
scripts/application/           Email/database Qwen verifier and case generation scripts
scripts/data_generation/       API data-generation and augmentation scripts
prompts/                       Prompt templates for API data generation
reports/                       Metrics, training report, validation history, application results
data/                          Local datasets, ignored by Git
outputs/                       Local generated outputs, ignored by Git
checkpoints/                   Local downloaded checkpoints, ignored by Git
```

## Environment

Install the main dependencies:

```bash
pip install -r requirements.txt
```

The original runs used CUDA, `bfloat16`, and local model paths on the server. Adjust `configs/model_paths.example.json` before reproducing.

## Binary SFT, lr 8e-6, 330 steps

Main files:

- `scripts/train/train_qwen_binary_sft_maxsteps.py`
- `scripts/train/run_binary_lr8e6_steps330_train_eval.sh`
- `configs/binary_lr8e6_steps330.json`
- `model_params/binary_lr8e6_steps330/config.json`

Example:

```bash
bash scripts/train/run_binary_lr8e6_steps330_train_eval.sh
```

The run script assumes the server layout under `/root/autodl-tmp`. Edit the top path variables if your machine differs.

## Taxonomy LoRA

Main files:

- `scripts/train/train_qwen_taxonomy_lora.py`
- `scripts/train/run_taxonomy_lora_200.sh`
- `configs/taxonomy_lora_lr8e6_steps200.json`
- `model_params/taxonomy_lora_lr8e6_steps200/adapter_config.json`

Example:

```bash
bash scripts/train/run_taxonomy_lora_200.sh
```

## Evaluation

Self-contained full-model eval:

```bash
python scripts/eval/evaluate_unified_full_model.py \
  --model /path/to/full/checkpoint \
  --dataset-root /path/to/summer_camp_teseset \
  --output-dir outputs/unified_sft_eval
```

The original `evaluate_unified_sft_model.py` is also kept for traceguard-based environments.

LoRA eval:

```bash
python scripts/eval/evaluate_unified_lora_model.py \
  --base-model /path/to/Qwen3.5-0.8B \
  --adapter /path/to/taxonomy_lora/checkpoint-200 \
  --dataset-root /path/to/summer_camp_teseset \
  --output-dir outputs/taxonomy_lora_eval \
  --datasets atbench,rjudge
```

For R-Judge, the LoRA evaluation script reports binary `safe`/`unsafe` metrics only.

## Application Guardrails

Main files:

- `scripts/application/qwen_binary_verifier_email.py`
- `scripts/application/qwen_binary_verifier_database.py`
- `scripts/application/generate_application_expanded_cases.py`

The application verifier files are drop-in adapters for the original email/database apps. Set:

```bash
export VERIFIER_BACKEND=qwen_binary
export LOCAL_MODEL_PATH=/path/to/binary_lr8e6_steps330/checkpoint-330
```

The 50-case expanded application results are in `reports/application_lr8e6_steps330_expanded50/`.

## API Data Generation

The API augmentation entrypoint is:

```bash
python scripts/data_generation/augment_safe_samples_with_api.py \
  --dataset-path data/datasets/agentdog_raw/AgentDoG-BinarySafety/train.json \
  --prompt-path prompts/security_analyst_json_prompt.txt \
  --output-path outputs/data/agentdog_binary_safe_dedup1000_json_augmented.jsonl \
  --failed-path outputs/data/agentdog_binary_safe_dedup1000_json_augmented_failed.jsonl \
  --max-samples 1000 \
  --concurrency 6 \
  --model openai/gpt-5.5 \
  --api-base https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY
```

Set the API key through the environment, not in Git:

```bash
export OPENROUTER_API_KEY=...
```

The script deduplicates safe BinarySafety rows by default, resumes from existing output rows, writes each successful JSONL row immediately, keeps failed rows in a separate JSONL file, and prints per-row progress.

## Results

See:

- `reports/evaluation_summary.md`
- `reports/binary_lr8e6_steps330/`
- `reports/taxonomy_lora_lr8e6_steps200/`
- `reports/application_lr8e6_steps330_expanded50/`
