# Remote GPU Deployment Without Docker

TraceHound does not require Docker. The primary deployment path is a plain Linux server plus conda/mamba/micromamba. Docker files are kept only as an optional path for environments that already provide NVIDIA Container Toolkit.

## Recommended Server Flow

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
TRACEHOUND_MODEL_PROFILE=internlm3-8b-instruct
TRACEHOUND_INSTALL_PREFERENCE=0
TRACEHOUND_INSTALL_QLORA=0
TRACEHOUND_INSTALL_OFFICIAL=1
TRACEHOUND_PREINSTALL_NATIVE_DEPS=1
TRACEHOUND_TMPDIR=$HOME/.cache/tracehound/tmp
```

Then run:

```bash
bash scripts/bootstrap_remote.sh
```

The bootstrap script creates `tracehound-gpu`, installs CUDA PyTorch wheels unless disabled, installs TraceHound with dev/train extras, runs `scripts/gpu_doctor.py`, and runs the smoke path.

## Sync From Local Mac

When the GitHub clone is unavailable or you want to mirror the current local workspace to the cluster path, run from the Mac:

```bash
TRACEHOUND_REMOTE_TARGET=ailab-p:/mnt/petrelfs/lichunxiao/TraceHound \
  bash scripts/sync_remote.sh
```

The sync excludes `.git`, `.env` from rsync, caches, generated data, model checkpoints, and reports. If a local `.env` exists, the script copies it separately to the remote path and runs `chmod 600` without printing secrets. To also run bootstrap and the official data smoke:

```bash
TRACEHOUND_REMOTE_BOOTSTRAP=1 TRACEHOUND_REMOTE_OFFICIAL_SMOKE=1 \
  bash scripts/sync_remote.sh
```

If the login node does not expose GPUs, run GPU diagnostics and training through Slurm:

```bash
bash scripts/slurm_gpu_test.sh
TRACEHOUND_SLURM_JOB_NAME=TH_SFT TRACEHOUND_SLURM_GPUS=1 \
  bash scripts/slurm_run.sh \
  'python scripts/train_sft.py --data data/synthetic_sft.jsonl --model-profile internlm2_5-1_8b-chat --max-samples 32'
```

Use `TRACEHOUND_USE_APPTAINER=1` and `TRACEHOUND_APPTAINER_IMAGE=/path/to/image.sif` for Slurm + Apptainer clusters. Full examples are in `docs/slurm_cluster_usage.md`.

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

## If Conda Is Not Installed

Docker is still not needed. Install user-local Miniconda:

```bash
bash scripts/install_miniconda_linux.sh
export PATH="$HOME/miniconda3/bin:$PATH"
bash scripts/bootstrap_remote.sh
```

Or let the bootstrap script do that automatically:

```bash
TRACEHOUND_AUTO_INSTALL_MINICONDA=1 bash scripts/bootstrap_remote.sh
```

This installs into `$HOME/miniconda3` by default. Override with:

```bash
TRACEHOUND_MINICONDA_DIR=/path/to/miniconda bash scripts/install_miniconda_linux.sh
```

## If Git Clone Is Inconvenient

On the Mac, build a clean tarball from committed files only:

```bash
bash scripts/create_deploy_bundle.sh
scp dist/tracehound-<sha>.tar.gz <user>@<server-ip>:~/
```

On the server:

```bash
tar -xzf tracehound-<sha>.tar.gz
cd TraceHound
cp .env.server.example .env
bash scripts/bootstrap_remote.sh
```

The bundle excludes `.git`, `.env`, runtime reports, caches, model checkpoints, and generated temporary data.

## Server Validation Commands

```bash
python scripts/gpu_doctor.py --strict
python scripts/list_model_profiles.py
bash scripts/smoke_remote.sh
python scripts/prepare_agentdog_official.py --download-dataset agentdog10_training --manifest reports/agentdog_official_manifest.json
python scripts/build_agentdog_data.py --source agentdog10 --limit 20 --no-annotate-cot
python scripts/build_agentdog_data.py --source agentdog10 --limit 2 --annotate-cot --cot-backend stub
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

Training scripts validate data and dependencies by default, then write plan files. Add `--run` only after the GPU environment and model choice are verified.

```bash
python scripts/train_sft.py --data data/synthetic_sft.jsonl --model-profile internlm2_5-1_8b-chat --max-samples 32
python scripts/train_sft.py --data data/synthetic_sft.jsonl --model-profile internlm3-8b-instruct --output-dir checkpoints/internlm3-sft --run
python scripts/train_preference.py --data data/synthetic_preference.jsonl --base-model checkpoints/internlm3-sft --model-profile internlm3-8b-instruct --algorithm dpo --run
```

Use `internlm2_5-7b-chat` as the formal fallback profile. Use `intern-s2-preview` through the API judge when local GPU validation is not ready.

## Optional Docker Path

Use this only when the server already has Docker plus NVIDIA Container Toolkit. Do not install Docker just for TraceHound.

```bash
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

## Common Fixes

- `CUDA is not visible to torch`: check `nvidia-smi`, the NVIDIA driver, and whether `TRACEHOUND_TORCH_INDEX_URL` matches the server. Set `TRACEHOUND_SKIP_TORCH=1` if the contest image already includes a correct PyTorch build.
- `fatal: detected dubious ownership in repository at '/tmp'`: the bootstrap uses `TRACEHOUND_TMPDIR=$HOME/.cache/tracehound/tmp` to keep pip builds out of shared `/tmp`. If running commands manually, export `TMPDIR=$HOME/.cache/tracehound/tmp` first.
- `Failed building wheel for sentencepiece` or `No package 'sentencepiece' found`: keep `TRACEHOUND_PREINSTALL_NATIVE_DEPS=1`. The bootstrap preinstalls `sentencepiece`, `protobuf`, `cmake`, and `pkg-config` from conda-forge before pip installs `.[train]`, so pip does not need to compile `sentencepiece`. To recover an already-created environment, run `conda install -y -n tracehound-gpu -c conda-forge sentencepiece protobuf cmake pkg-config`, then rerun `bash scripts/bootstrap_remote.sh`.
- `Failed to build pyarrow`: keep `TRACEHOUND_INSTALL_PREFERENCE=0` for the default deployment. Install `pip install -e ".[preference]"` later only when DPO/ORPO is needed.
- `No matching distribution found for bitsandbytes`: keep `TRACEHOUND_INSTALL_QLORA=0` for the default LoRA/SFT path. Install `pip install -e ".[qlora]"` only if QLoRA is required and the server mirror has a compatible wheel.
- No conda on server: run `bash scripts/install_miniconda_linux.sh`; Docker is not required.
- API works locally but not on the server: copy only non-secret template values from `.env.server.example`, then add the server API key directly to `.env`; never commit it.
- Port is not reachable: use SSH tunneling or open the firewall/security group for `TRACEHOUND_PORT`.
- Generated reports are missing from git: this is expected. `reports/`, `checkpoints/`, `models/`, `dist/`, and `data/tmp/` are runtime outputs.
