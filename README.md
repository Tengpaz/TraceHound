# TraceHound

TraceHound is a lightweight, CPU-first baseline for agent trajectory safety. It follows the AgentDoG trajectory-level diagnosis idea and keeps the first implementation easy to run on macOS while remaining portable to Linux contest servers.

The current MVP focuses on:

- AgentDoG-style `Risk Source / Failure Mode / Real-world Harm` labels.
- Synthetic trajectory data for eval, SFT, and preference formats.
- Rule-based prefiltering, risk span compression, and deterministic heuristic judging.
- Cost statistics and evaluation metrics.
- Dataset quality checks, final-answer-only baseline, and ablation reporting.
- Config-driven 10K-scale synthetic data generation for contest-day retuning.
- API judge token-cost estimates via optional per-1M token pricing env vars.
- InternLM/Intern-S2 model profiles, tokenizer `chat_template` prompt rendering, and optional local HF adapter.
- A FastAPI demo with native HTML/CSS/JS, JSON/JSONL upload, batch evaluation, report downloads, and Guard Model ops.

Training scripts are placeholders by design. They validate inputs and explain optional dependencies, but they do not require GPU packages in the default environment.

## Environment

Use conda to avoid polluting the local Python environment:

```bash
conda env create -f environment.yml
conda activate tracehound
```

The default environment installs the editable project, FastAPI demo dependencies, and pytest. If the environment already exists after pulling changes, update it with:

```bash
conda env update -n tracehound -f environment.yml
```

Optional training dependencies are intentionally separate:

```bash
pip install -e ".[train]"
```

Install training dependencies only on a suitable Linux/GPU server or a contest-provided environment.
Preference-training dependencies are separate because they pull `datasets/pyarrow`, which can be fragile on restricted mirrors:

```bash
pip install -e ".[preference]"
```

`bitsandbytes` is not installed by default because many contest mirrors lag behind current wheels. Enable it only for QLoRA experiments:

```bash
pip install -e ".[qlora]"
```

List prepared local/API model candidates:

```bash
python scripts/list_model_profiles.py
```

## Remote GPU Server

Docker is not required. For a Linux GPU server, use the no-Docker conda bootstrap:

```bash
git clone git@github.com:Tengpaz/TraceHound.git
cd TraceHound
cp .env.server.example .env
bash scripts/bootstrap_remote.sh
```

If the server does not have conda/mamba/micromamba:

```bash
bash scripts/install_miniconda_linux.sh
export PATH="$HOME/miniconda3/bin:$PATH"
bash scripts/bootstrap_remote.sh
```

Then start the demo:

```bash
conda activate tracehound-gpu
bash scripts/run_remote_demo.sh
```

The bootstrap path creates a conda env, optionally installs CUDA PyTorch wheels, installs TraceHound with training extras, runs GPU diagnostics, and runs a smoke test.
It does not install `datasets/trl/pyarrow` unless `TRACEHOUND_INSTALL_PREFERENCE=1` is set, and it does not install `bitsandbytes` unless `TRACEHOUND_INSTALL_QLORA=1` is set.

The server template defaults to the Intern route:

- Local first choice: `internlm/internlm3-8b-instruct`
- Local smoke: `internlm/internlm2_5-1_8b-chat`
- Local fallback: `internlm/internlm2_5-7b-chat`
- API candidate: `intern-s2-preview`

If direct `git clone` is inconvenient, create a clean deploy tarball on your Mac and upload it:

```bash
bash scripts/create_deploy_bundle.sh
scp dist/tracehound-<sha>.tar.gz <user>@<server-ip>:~/
```

Docker GPU deployment remains optional only if the server already has Docker plus NVIDIA Container Toolkit. See `docs/remote_gpu_deploy.md` for CUDA wheel overrides, SSH tunneling, tarball deployment, optional Docker notes, and training preflight commands.

## Quick Start

Generate synthetic data:

```bash
python scripts/generate_data.py --out data
```

The built-in generator currently emits 16 balanced cases across `shell`, `file`, `browser`, `email`, `database`, `code_executor`, `calendar`, and `credential` scenarios. You can filter it:

```bash
python scripts/generate_data.py --out data/tmp --scenario browser --label unsafe --limit 2
```

For contest-day retuning, edit `configs/generation.yaml` and run:

```bash
python scripts/generate_data.py --config configs/generation.yaml
```

The config supports output path, scale, scenario/label filters, and whether to export eval, SFT, preference, and RL-style datasets.

Run quality checks:

```bash
python scripts/quality_check.py data/synthetic_eval.jsonl
```

Run the layered baseline:

```bash
python scripts/evaluate.py data/synthetic_eval.jsonl --mode layered
```

Run offline ablations and generate Markdown plus SVG chart reports:

```bash
python scripts/run_experiments.py --data data/synthetic_eval.jsonl --no-api
python scripts/generate_report.py --input reports/experiments.json --output reports/experiment_report.md
```

Run tests:

```bash
pytest
```

Start the FastAPI demo:

```bash
python scripts/serve_demo.py --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`.

## No-training API Validation

TraceHound can validate with an OpenAI-compatible remote API without local training or GPU dependencies.

Create a local `.env` from the example file:

```bash
cp .env.example .env
```

Fill these values in `.env`:

```bash
TRACEHOUND_API_BASE=https://chat.intern-ai.org.cn/api/v1
TRACEHOUND_API_KEY=your-api-key
TRACEHOUND_MODEL=intern-s2-preview
```

For most OpenAI-compatible services, keep:

```bash
TRACEHOUND_API_PATH=/chat/completions
```

Optional cost estimates use per-1M token prices:

```bash
TRACEHOUND_INPUT_PRICE_PER_1M=2.00
TRACEHOUND_OUTPUT_PRICE_PER_1M=8.00
```

Run a small smoke validation first:

```bash
python scripts/validate_api.py --data data/synthetic_eval.jsonl --limit 1 --judge api
```

Then run the cost-aware hybrid path, which uses rules for high-confidence early exits and calls the API only when needed:

```bash
python scripts/validate_api.py --data data/synthetic_eval.jsonl --judge hybrid --mode layered
```

You can also use the same API judge through the evaluator:

```bash
python scripts/evaluate.py data/synthetic_eval.jsonl --judge api --mode compressed
```

The Web Demo has a `Judge` selector. `hybrid API` and `API only` read credentials from the server-side `.env`; the API key is not sent to the browser.
The page also shows the configured remote model, redacted endpoint origin, selected inference strategy, model call count, latency, and estimated API cost so judges can see when third-party API inference is actually used.

## Intern Local Model Preparation

TraceHound does not hardcode Qwen/Llama/Intern prompt formats. Local inference and SFT render messages with the tokenizer's `chat_template`; if a contest tokenizer lacks one, the code falls back to a model-agnostic `SYSTEM / USER / ASSISTANT` prompt.

On a GPU server, start with the low-cost smoke profile:

```bash
python scripts/evaluate.py data/synthetic_eval.jsonl \
  --judge local \
  --model-profile internlm2_5-1_8b-chat \
  --mode compressed \
  --limit 1
```

Then validate the first-choice base model:

```bash
python scripts/evaluate.py data/synthetic_eval.jsonl \
  --judge local \
  --model-profile internlm3-8b-instruct \
  --mode compressed \
  --limit 1
```

Training commands default to a dry preflight and write plan files. Add `--run` only on the GPU server:

```bash
python scripts/train_sft.py --model-profile internlm2_5-1_8b-chat --max-samples 32
python scripts/train_sft.py --model-profile internlm3-8b-instruct --output-dir checkpoints/internlm3-sft --run
python scripts/train_preference.py --base-model checkpoints/internlm3-sft --model-profile internlm3-8b-instruct --algorithm dpo --run
```

See `docs/intern_model_playbook.md` for InternLM2.5-7B fallback, 20B notes, DPO/ORPO/GRPO planning, and cost-aware validation commands.

## Web Demo Features

The demo has three pages:

- `主页`: project positioning and telemetry summary.
- `Agent轨迹安全评估`: single-case evaluation, JSON/JSONL upload or drag-and-drop, batch evaluation, evidence timeline, online guard simulation, and JSON/Markdown/SVG report download.
- `Guard Model调配`: current serving mode, API/local model state, configurable data generation, live job progress, and local SFT / SFT+RL training preflight hooks.

Batch uploads accept:

- A single internal case object with `trajectory`.
- A JSON array of case objects.
- An object with `cases: [...]`.
- JSONL with one case per line.

## Contest Adaptation

When the official task is released, keep the core pipeline stable:

1. Convert official data into the internal trajectory schema.
2. Run `rules` and `layered` modes as the first baseline.
3. Add a model adapter for the official base model or model service.
4. Export SFT/preference data only if training is allowed and time permits.

The minimal converter accepts JSON/JSONL records with `trajectory`, `steps`, `events`, or `messages`:

```bash
python scripts/convert_dataset.py data/official/sample.jsonl data/tmp/official_eval.jsonl --limit 20
```

Model service adapters should use these environment variables:

- `TRACEHOUND_API_BASE`
- `TRACEHOUND_API_KEY`
- `TRACEHOUND_MODEL`

The default path is fully offline and does not make network requests.

## Project Layout

- `docs/design.md`: Chinese project requirements and design notes.
- `docs/contest_playbook.md`: Contest-day adaptation workflow.
- `docs/remote_gpu_deploy.md`: One-command Linux/GPU deployment notes.
- `docs/training_gpu.md`: Optional Linux/GPU fine-tuning notes.
- `docs/intern_model_playbook.md`: InternLM/Intern-S2 profile, inference, and fine-tuning playbook.
- `configs/generation.yaml`: Default synthetic generation config for quick retuning.
- `configs/model_profiles.json`: Prepared local/API model profiles.
- `.env.server.example`: GPU server environment template.
- `scripts/bootstrap_remote.sh`: Primary no-Docker server bootstrap.
- `scripts/create_deploy_bundle.sh`: Clean tarball packaging for `scp` deployment.
- `Dockerfile.gpu` / `docker-compose.gpu.yml`: Optional NVIDIA Docker deployment, not required.
- `Makefile`: Common setup, smoke, demo, and Docker commands.
- `traceguard/`: Core Python package.
- `scripts/`: Data generation, evaluation, demo server, and training placeholders.
- `web_demo/`: FastAPI-served static assets.
- `tests/`: CPU-only test suite.
