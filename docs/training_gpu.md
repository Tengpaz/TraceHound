# Optional GPU Training Guide

TraceHound does not train models in the default Mac environment. The local deliverable is a CPU/API-valid baseline. Use this guide only on a Linux/GPU server after contest rules confirm that fine-tuning is allowed.

## Server Setup

For one-command remote setup, prefer `docs/remote_gpu_deploy.md`:

```bash
cp .env.server.example .env
bash scripts/bootstrap_remote.sh
```

For a manual setup, create or update the GPU conda environment:

```bash
conda env create -f environment.gpu.yml
conda activate tracehound-gpu
```

Install the PyTorch build that matches the server CUDA version. Use the command recommended by the PyTorch website or the contest image documentation, then install optional training packages:

```bash
pip install -e ".[train]"
```

Check that CUDA is visible:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
```

List the prepared Intern profiles:

```bash
python scripts/list_model_profiles.py
```

## Data

Generate synthetic data or convert official data first:

```bash
python scripts/generate_data.py --out data
python scripts/convert_dataset.py data/official/train.jsonl data/tmp/official_train.jsonl
```

The training entrypoints validate files, resolve model profiles, and write plan files by default:

```bash
python scripts/train_sft.py --data data/synthetic_sft.jsonl --model-profile internlm2_5-1_8b-chat --max-samples 32
python scripts/train_preference.py --data data/synthetic_preference.jsonl --model-profile internlm3-8b-instruct --algorithm dpo
```

Add `--run` only on a GPU server to launch the prepared LoRA SFT or DPO/ORPO paths.

```bash
python scripts/train_sft.py \
  --data data/synthetic_sft.jsonl \
  --model-profile internlm3-8b-instruct \
  --output-dir checkpoints/internlm3-sft \
  --run

python scripts/train_preference.py \
  --data data/synthetic_preference.jsonl \
  --base-model checkpoints/internlm3-sft \
  --model-profile internlm3-8b-instruct \
  --algorithm dpo \
  --output-dir checkpoints/internlm3-dpo \
  --run
```

## Recommended Extension Points

- SFT: `scripts/train_sft.py` uses `transformers`, `peft`, and tokenizer `chat_template` with a plain-prompt fallback.
- Preference optimization: `scripts/train_preference.py` prepares DPO/ORPO paths through `trl`; GRPO remains a reward-function hook until the official scoring interface is known.
- Model adapter: `traceguard/local_model.py` provides a local Hugging Face adapter for InternLM and other chat-template models.
- Evaluation: always run `scripts/run_experiments.py` before and after training to compare with the CPU baseline.

## Safety Defaults

- Keep LoRA/checkpoint outputs under `checkpoints/`, which is ignored by git.
- Do not download large model weights on the Mac workstation.
- Do not run full API validation unless quota and contest rules allow it.
- Keep `TRACEHOUND_API_KEY` in `.env`; never commit keys or generated raw service responses containing secrets.
