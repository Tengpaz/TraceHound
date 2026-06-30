# Slurm GPU Cluster Usage

Use this guide when the login node and GPU nodes are separated and GPUs are available only through Slurm `srun`.

TraceHound supports three GPU execution modes:

- Direct Linux GPU shell: run commands after `conda activate tracehound-gpu`.
- Slurm + conda environment: submit TraceHound commands through `scripts/slurm_run.sh`.
- Slurm + Apptainer: submit the same commands inside a CUDA container with `--nv`.

## 1. Check Partitions

```bash
sinfo
```

For the current cluster, `ai4safe` is the default partition in `.env.server.example`:

```bash
TRACEHOUND_SLURM_PARTITION=ai4safe
```

Override per command when needed:

```bash
TRACEHOUND_SLURM_PARTITION=AI4Good_L1_p bash scripts/slurm_gpu_test.sh
```

## 2. GPU Visibility Smoke Test

```bash
bash scripts/slurm_gpu_test.sh
```

Equivalent explicit form:

```bash
bash scripts/slurm_run.sh 'hostname; nvidia-smi; python scripts/gpu_doctor.py --strict'
```

The default resources are:

```bash
TRACEHOUND_SLURM_PARTITION=ai4safe
TRACEHOUND_SLURM_GPUS=1
TRACEHOUND_SLURM_NTASKS=1
TRACEHOUND_SLURM_NTASKS_PER_NODE=1
TRACEHOUND_SLURM_CPUS_PER_TASK=
TRACEHOUND_SLURM_MEM=
TRACEHOUND_SLURM_JOB_NAME=TRACEHOUND
TRACEHOUND_SLURM_LOG_DIR=log
```

`TRACEHOUND_SLURM_CPUS_PER_TASK` and `TRACEHOUND_SLURM_MEM` are intentionally empty by default so the GPU smoke command matches the cluster tutorial exactly. Add them only for commands that need explicit CPU or memory resources.

Values passed before the command override `.env`, so one-off resource changes are safe:

```bash
TRACEHOUND_SLURM_GPUS=4 TRACEHOUND_SLURM_NTASKS=12 TRACEHOUND_SLURM_NTASKS_PER_NODE=4 \
  bash scripts/slurm_run.sh 'hostname; nvidia-smi'
```

Inspect the generated `srun` command without submitting:

```bash
bash scripts/slurm_run.sh --dry-run 'hostname; nvidia-smi'
```

## 3. CPU/API Jobs On Slurm

Set `TRACEHOUND_SLURM_GPUS=0` to omit `--gres=gpu:*`:

```bash
TRACEHOUND_SLURM_GPUS=0 TRACEHOUND_SLURM_JOB_NAME=TH_EVAL \
  TRACEHOUND_SLURM_CPUS_PER_TASK=1 TRACEHOUND_SLURM_MEM=4G \
  bash scripts/slurm_run.sh \
  'python scripts/evaluate.py data/tmp/remote_smoke/synthetic_eval.jsonl --mode layered'
```

## 4. Data Generation And Evaluation

Small deterministic smoke:

```bash
TRACEHOUND_SLURM_GPUS=0 TRACEHOUND_SLURM_JOB_NAME=TH_DATA \
  bash scripts/slurm_run.sh \
  'python scripts/generate_data.py --count 64 --out data/tmp/slurm_smoke && python scripts/quality_check.py data/tmp/slurm_smoke/synthetic_eval.jsonl'
```

Remote API validation with one sample:

```bash
TRACEHOUND_SLURM_GPUS=0 TRACEHOUND_SLURM_JOB_NAME=TH_API \
  bash scripts/slurm_run.sh \
  'python scripts/validate_api.py --data data/tmp/slurm_smoke/synthetic_eval.jsonl --limit 1 --judge api --mode compressed'
```

Offline experiments:

```bash
TRACEHOUND_SLURM_GPUS=0 TRACEHOUND_SLURM_JOB_NAME=TH_EXP \
  bash scripts/slurm_run.sh \
  'python scripts/run_experiments.py --data data/tmp/slurm_smoke/synthetic_eval.jsonl --output reports/slurm_experiments.json --no-api && python scripts/generate_report.py --input reports/slurm_experiments.json --output reports/slurm_report.md'
```

## 5. LoRA SFT Preflight And Training

Preflight only:

```bash
TRACEHOUND_SLURM_JOB_NAME=TH_SFT_PLAN TRACEHOUND_SLURM_GPUS=1 \
  bash scripts/slurm_run.sh \
  'python scripts/train_sft.py --data data/tmp/slurm_smoke/synthetic_sft.jsonl --model-profile internlm2_5-1_8b-chat --max-samples 32'
```

Actual SFT run:

```bash
TRACEHOUND_SLURM_JOB_NAME=TH_SFT TRACEHOUND_SLURM_GPUS=1 TRACEHOUND_SLURM_MEM=96G \
  bash scripts/slurm_run.sh \
  'python scripts/train_sft.py --data data/tmp/slurm_smoke/synthetic_sft.jsonl --model-profile internlm2_5-1_8b-chat --output-dir checkpoints/internlm-smoke-sft --max-samples 32 --run'
```

Formal candidate:

```bash
TRACEHOUND_SLURM_JOB_NAME=TH_INTERN3_SFT TRACEHOUND_SLURM_GPUS=1 TRACEHOUND_SLURM_MEM=160G \
  bash scripts/slurm_run.sh \
  'python scripts/train_sft.py --data data/synthetic_sft.jsonl --model-profile internlm3-8b-instruct --output-dir checkpoints/internlm3-sft --run'
```

For multi-GPU Slurm jobs:

```bash
TRACEHOUND_SLURM_GPUS=4 \
TRACEHOUND_SLURM_NTASKS=4 \
TRACEHOUND_SLURM_NTASKS_PER_NODE=4 \
bash scripts/slurm_run.sh 'python scripts/gpu_doctor.py --strict'
```

TraceHound training scripts currently use Hugging Face `device_map=auto`; distributed multi-process training should be added only after single-GPU LoRA smoke succeeds on the target model.

## 6. Slurm + Apptainer

Use this when the cluster expects GPU jobs to run in an Apptainer/Singularity CUDA image.

```bash
export TRACEHOUND_USE_APPTAINER=1
export TRACEHOUND_APPTAINER_IMAGE=/mnt/petrelfs/lichunxiao/tracehound/pytorch-2.1.2-cuda11.8.sif
export TRACEHOUND_APPTAINER_BIND=$PWD:/workspace
export TRACEHOUND_APPTAINER_WORKDIR=/workspace
export TRACEHOUND_SLURM_PARTITION=ai4safe
export TRACEHOUND_SLURM_GPUS=1

bash scripts/slurm_run.sh 'hostname; nvidia-smi; python scripts/gpu_doctor.py --strict'
```

The wrapper runs:

```bash
apptainer exec --nv --bind "$TRACEHOUND_APPTAINER_BIND" "$TRACEHOUND_APPTAINER_IMAGE" bash -lc '<command>'
```

Inside the container it also sets:

```bash
PYTHONUSERBASE=/workspace/pyuser
HF_HOME=/workspace/hf_cache
TORCH_HOME=/workspace/torch_cache
PATH=$PYTHONUSERBASE/bin:$PATH
```

If the container does not contain TraceHound dependencies, install them into the user base once:

```bash
TRACEHOUND_USE_APPTAINER=1 bash scripts/slurm_run.sh \
  'python -m pip install --user -e ".[dev,train]"'
```

Then run the same GPU checks or training commands.

## 7. Common Overrides

Use `TRACEHOUND_SLURM_EXTRA_ARGS` for site-specific Slurm flags:

```bash
TRACEHOUND_SLURM_EXTRA_ARGS='--exclusive' bash scripts/slurm_gpu_test.sh
```

Use `TRACEHOUND_SLURM_MPI=pmi2` when submitting MPI-style jobs:

```bash
TRACEHOUND_SLURM_MPI=pmi2 TRACEHOUND_SLURM_GPUS=4 TRACEHOUND_SLURM_NTASKS=12 TRACEHOUND_SLURM_NTASKS_PER_NODE=4 \
  bash scripts/slurm_run.sh 'hostname; nvidia-smi'
```

Logs are written to:

```text
log/<job-name>-<timestamp>.log
```
