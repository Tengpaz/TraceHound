# Remote GPU Deployment

This guide packages TraceHound for a Linux GPU server while keeping the Mac workstation path lightweight. The default path is conda-based because contest servers often provide a custom CUDA driver, mounted datasets, and restricted Docker permissions.

## One-command Conda Path

On the GPU server:

```bash
git clone git@github.com:Tengpaz/TraceHound.git
cd TraceHound
cp .env.server.example .env
```

Edit `.env` for the server. At minimum, check:

```bash
TRACEHOUND_HOST=0.0.0.0
TRACEHOUND_PORT=8000
TRACEHOUND_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
TRACEHOUND_RUN_SMOKE=1
```

Then run:

```bash
bash scripts/bootstrap_remote.sh
```

The bootstrap script creates `tracehound-gpu`, installs CUDA PyTorch wheels unless disabled, installs TraceHound with dev/train extras, runs `scripts/gpu_doctor.py`, and runs the CPU/API smoke path.

Start the demo:

```bash
conda activate tracehound-gpu
bash scripts/run_remote_demo.sh
```

Open `http://<server-ip>:8000`, or tunnel from your Mac:

```bash
ssh -L 8000:127.0.0.1:8000 <user>@<server-ip>
```

If tunneling, set `TRACEHOUND_HOST=127.0.0.1` in `.env` before starting the demo.

## Docker GPU Path

Use this only when the server has Docker plus NVIDIA Container Toolkit.

```bash
git clone git@github.com:Tengpaz/TraceHound.git
cd TraceHound
cp .env.server.example .env
docker compose -f docker-compose.gpu.yml up -d --build
```

Override the base image if the contest server requires a different CUDA line:

```bash
TRACEHOUND_GPU_BASE_IMAGE=pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime \
docker compose -f docker-compose.gpu.yml up -d --build
```

Persistent paths are mounted from the repo:

- `data/`
- `reports/`
- `checkpoints/`
- `models/`

## Server Validation Commands

```bash
python scripts/gpu_doctor.py --strict
bash scripts/smoke_remote.sh
python scripts/generate_data.py --config configs/generation.yaml --out data
python scripts/run_experiments.py --data data/synthetic_eval.jsonl --no-api
python scripts/generate_report.py --input reports/experiments.json --output reports/experiment_report.md
```

Equivalent Make targets:

```bash
make doctor
make smoke
make demo HOST=0.0.0.0 PORT=8000
```

## Training Preflight

Current training scripts intentionally validate data and dependencies before launching a real trainer. This keeps contest adaptation flexible.

```bash
python scripts/train_sft.py --data data/synthetic_sft.jsonl --base-model <base-model> --output-dir checkpoints/sft --strict
python scripts/train_preference.py --data data/synthetic_preference.jsonl --base-model checkpoints/sft --output-dir checkpoints/preference --strict
```

After the official model interface is known, add the trainer implementation behind these entrypoints instead of changing the evaluation/demo pipeline.

## Common Fixes

- `CUDA is not visible to torch`: check `nvidia-smi`, the NVIDIA driver, and whether `TRACEHOUND_TORCH_INDEX_URL` matches the server. Set `TRACEHOUND_SKIP_TORCH=1` if the contest image already includes a correct PyTorch build.
- API works locally but not on the server: copy only non-secret template values from `.env.server.example`, then add the server API key directly to `.env`; never commit it.
- Port is not reachable: use SSH tunneling or open the firewall/security group for `TRACEHOUND_PORT`.
- Generated reports are missing from git: this is expected. `reports/`, `checkpoints/`, `models/`, and `data/tmp/` are runtime outputs.
