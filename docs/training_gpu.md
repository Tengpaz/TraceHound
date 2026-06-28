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

## Data

Generate synthetic data or convert official data first:

```bash
python scripts/generate_data.py --out data
python scripts/convert_dataset.py data/official/train.jsonl data/tmp/official_train.jsonl
```

The current training entrypoints validate files and print sample statistics. They intentionally do not launch a trainer until the official model, tokenizer, max length, and submission rules are known:

```bash
python scripts/train_sft.py --data data/synthetic_sft.jsonl --base-model <model-name> --output-dir checkpoints/sft
python scripts/train_preference.py --data data/synthetic_preference.jsonl --base-model checkpoints/sft --output-dir checkpoints/preference
```

## Recommended Extension Points

- SFT: use `transformers`, `datasets`, `peft`, and `accelerate`.
- Preference optimization: use `trl` for DPO or ORPO if the contest permits pairwise preference training.
- Model adapter: add a new `ModelAdapter` in `traceguard/judge.py` if the contest provides local model weights rather than an API.
- Evaluation: always run `scripts/run_experiments.py` before and after training to compare with the CPU baseline.

## Safety Defaults

- Keep LoRA/checkpoint outputs under `checkpoints/`, which is ignored by git.
- Do not download large model weights on the Mac workstation.
- Do not run full API validation unless quota and contest rules allow it.
- Keep `TRACEHOUND_API_KEY` in `.env`; never commit keys or generated raw service responses containing secrets.
