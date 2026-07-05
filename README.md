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
- TraceHound-Base local binary guard support for the full-parameter Qwen3.5-0.8B SFT checkpoint under `models/TraceHound-Base-Qwen3.5-0.8B-Binary`.
- A FastAPI demo with native HTML/CSS/JS, JSON/JSONL upload, batch evaluation, report downloads, Guard Model ops, and safety enchantment for target models.
- Runtime online guardrail endpoints and hook examples for Claude Code, Codex-style wrappers, OpenClaw-style middleware, and generic agent systems.

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

Use the local TraceHound-Base binary checkpoint as a guard model:

```bash
python scripts/evaluate.py data/synthetic_eval.jsonl --judge local-binary --mode layered
python scripts/serve_demo.py --host 127.0.0.1 --port 8000
```

Then send guardrail requests with `"judge": "local-binary"` or choose `TraceHound-Base local` in the Web Demo. The checkpoint is loaded lazily on first local-binary inference.

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
It preinstalls native training dependencies such as `sentencepiece` from conda-forge by default to avoid fragile source builds on restricted server mirrors. If needed, disable that with `TRACEHOUND_PREINSTALL_NATIVE_DEPS=0`.

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

If GPUs are only available through a Slurm queue, use the cluster wrapper instead of running GPU commands on the login node:

```bash
bash scripts/slurm_gpu_test.sh
TRACEHOUND_SLURM_GPUS=0 bash scripts/slurm_run.sh 'python scripts/evaluate.py data/tmp/remote_smoke/synthetic_eval.jsonl --mode layered'
TRACEHOUND_SLURM_JOB_NAME=TH_SFT TRACEHOUND_SLURM_GPUS=1 bash scripts/slurm_run.sh 'python scripts/train_sft.py --data data/synthetic_sft.jsonl --model-profile internlm2_5-1_8b-chat --max-samples 32'
```

For Slurm + Apptainer, set `TRACEHOUND_USE_APPTAINER=1` and `TRACEHOUND_APPTAINER_IMAGE=/path/to/image.sif`. See `docs/slurm_cluster_usage.md`.

## AgentDoG Official Reproduction

TraceHound now separates official reproduction from local synthetic generation.

- Official reproduction uses the public AgentDoG repository, official prompts, official Hugging Face datasets, and the released SFT/RL recipes.
- `scripts/generate_data.py` remains a TraceHound surrogate generator for smoke tests and contest adaptation; it is not labeled as the official AgentDoG DataEngine.
- The public AgentDoG RL runtime release explicitly excludes the original LLM synthesis pipeline and prompt-generation code, so a from-scratch byte-identical data generation pipeline cannot be reconstructed from public assets alone.

Prepare official assets:

```bash
python -m pip install -e ".[official]"
python scripts/prepare_agentdog_official.py \
  --clone-repo \
  --download-dataset agentdog10_training \
  --download-dataset atbench \
  --download-dataset atbench_claw \
  --download-dataset atbench_codex \
  --download-dataset app1_sft
```

See `docs/agentdog_official_reproduction.md` for the official asset manifest and reproduction workflow.

Build official AgentDoG1.0 SFT bundles:

```bash
python scripts/build_agentdog_data.py --source agentdog10 --download-official --limit 20 --no-annotate-cot
```

Build the AgentDoG-Lite binary training dataset for the 2026 summer-camp response contract:

```bash
python scripts/build_agentdog_lite_binary.py \
  --out data/release/AgentDoG-Lite-TrainningDataset-Binary
```

This uses only `AI45Research/AgentDoG1.0-Training-Data/AgentDoG-BinarySafety/train.json` as training source, rewrites the target from bare `safe`/`unsafe` into the AgentDoG-Lite strict JSON judgment format, rewrites the taxonomy labels inside the prompt to the summer-camp snake_case contract, and inspects `AI45Research/2026_summer_camp_teseset` only for schema and label alignment. The summer-camp ATBench300/R-Judge evaluation samples are not copied into training outputs.

Evaluate the local TraceHound-Base binary checkpoint on the held-out summer-camp files:

```bash
python scripts/evaluate_lite_binary_model.py --datasets atbench,rjudge
```

Add AgentDoG-style CoT targets with the API judge/generator, or use the stub backend for a formatting smoke test:

```bash
python scripts/build_agentdog_data.py --source agentdog10 --limit 2 --annotate-cot --cot-backend stub
python scripts/build_agentdog_data.py --source agentdog10 --limit 100 --annotate-cot --cot-backend api --cot-concurrency 2
```

Official data-flow outputs are separated from synthetic outputs:

- `data/sft/official_agentdog10/binary_safety/{all,train,eval,test}.jsonl`
- `data/sft/official_agentdog10/taxonomy_only/{all,train,eval,test}.jsonl`
- `data/sft/official_agentdog10/unified_four_label/{all,train,eval,test}.jsonl`
- `data/sft/official_agentdog10/{coarse,finegrained,unified}_cot/{all,train,eval,test}.jsonl` when CoT annotation is enabled
- `data/sft/official_app1/safety_response_sft/{all,train,eval,test}.jsonl`
- `data/processed/official_atbench/eval_only/all.jsonl`

ATBench remains held-out by default. TraceHound only creates ATBench-derived CoT distillation artifacts when `allow_atbench_cot_distill` is explicitly enabled, and the manifest records the contamination warning.

For a paper-described approximation of the unreleased LLM synthesis engine, use the LLM backend with LLM self-repair:

```bash
python scripts/generate_data.py --config configs/generation_agentdog_llm.yaml --semantic-repair-backend llm
```

This follows the public pipeline shape: taxonomy tuple sampling, scenario/tool planning, LLM trajectory synthesis, deterministic purification, optional LLM self-repair, optional LLM semantic QC, and training-quality filtering. It is still a public-material approximation, not the unreleased official DataEngine source.

## Quick Start

Generate synthetic data:

```bash
python scripts/generate_data.py --out data
```

The generator now supports the full AgentDoG-style synthesis flow: sample a `risk_source / failure_mode / harm_type` tuple, select a tool scenario and tool subset, build a structured execution plan, synthesize the user/tool/agent trajectory with a controlled risk trigger, run QC, then export train/eval formats. The default backend is a deterministic local planner for offline Mac use. To match AgentDoG's LLM Stage 2 trajectory synthesis, enable the LLM backend:

```bash
python scripts/generate_data.py --config configs/generation.yaml --agentdog-llm-generate --count 10
```

For the closest AgentDoG-style data path, combine LLM trajectory synthesis with strict LLM semantic QC:

```bash
python scripts/generate_data.py --config configs/generation.yaml --agentdog-llm-generate --agentdog-strict-qc
```

The same preset is available as:

```bash
python scripts/generate_data.py --config configs/generation_agentdog_llm.yaml
```

You can filter it:

```bash
python scripts/generate_data.py --out data/tmp --scenario browser --label unsafe --limit 2
```

For contest-day retuning, edit `configs/generation.yaml` and run:

```bash
python scripts/generate_data.py --config configs/generation.yaml
```

The config supports output path, scale, `generation_backend: deterministic|llm`, scenario/label filters, train/eval/test split ratios, clean output layout, legacy compatibility copies, and whether to export eval, TraceHound RiskReport SFT, AgentDoG binary SFT, AgentDoG fine-grained taxonomy SFT, preference, and RL-style datasets.
The cleaning/QC path now mirrors AgentDoG's two-layer design as closely as a local pre-contest implementation can: deterministic validators check turn structure, tool invocation legality, step coherence, readability, taxonomy alignment, and unsafe attack success; optional LLM QC adds API judge voting and consensus filtering. Use `qc_policy: agentdog_strict` or `--agentdog-strict-qc` to require the LLM semantic layer; the default `agentdog_local` keeps API calls off for Mac/local generation.
LLM generation also records production quality signals. `raw_agentdog_qc` captures pre-repair quality, `repair_level` is `none`, `structural`, or `semantic`, and `quality_report.json` reports raw pass rate, repair rate, semantic repair rate, and training eligibility. By default, eval exports keep every QC-passing sample, while SFT/preference/RL exports only keep samples up to `training_max_repair_level: structural`; semantic salvage samples are retained for evaluation and diagnostics but not used for training unless you explicitly relax that config.
`examples/demo_cases.json` is not refreshed by default; use `--write-examples` when you intentionally want to update the checked-in demo snapshot.

Long LLM generation jobs support incremental checkpoints, resume, pause, and conservative concurrency:

```bash
python scripts/generate_data.py \
  --config configs/generation_agentdog_llm.yaml \
  --out data/tmp/agentdog_llm_1000 \
  --count 1000 \
  --llm-generation-concurrency 3 \
  --llm-qc-concurrency 2
```

In LLM mode, checkpoints are written by default to `<out>/_checkpoints/` as each case finishes. Key files are `llm_generated_cases.jsonl`, `llm_generation_rejected.jsonl`, `llm_qc_kept_cases.jsonl`, `llm_qc_rejected_cases.jsonl`, and state JSON files for progress monitors. Resume an interrupted run with the same `--out` and `--resume`; completed case ids are skipped:

```bash
python scripts/generate_data.py --config configs/generation_agentdog_llm.yaml --out data/tmp/agentdog_llm_1000 --resume
```

To pause without killing in-flight API calls, create the pause file; remove it to continue:

```bash
touch data/tmp/agentdog_llm_1000/_checkpoints/PAUSE
rm data/tmp/agentdog_llm_1000/_checkpoints/PAUSE
```

Start with concurrency `2-4` unless you know the API rate limits. Higher concurrency is faster but can increase HTTP 400/429 retries and token spend.

Default generated outputs now use a clean dataset bundle layout:

- `cases/all.jsonl`: all QC-passing evaluation cases.
- `cases/train.jsonl`, `cases/eval.jsonl`, `cases/test.jsonl`: deterministic train/eval/test splits over training-eligible cases.
- `train/tracehound_risk_report_sft/{all,train,eval,test}.jsonl`: TraceHound JSON-report SFT task.
- `train/agentdog/binary_safety/{all,train,eval,test}.jsonl`: binary-only AgentDoG safety task.
- `train/agentdog/taxonomy_only/{all,train,eval,test}.jsonl`: fine-grained taxonomy-only AgentDoG task.
- `train/agentdog/unified_four_label/{all,train,eval,test}.jsonl`: unified four-label AgentDoG task.
- `train/agentdog15/{unified,coarse}/{all,train,eval,test}.jsonl`: AgentDoG 1.5 prompt-compatible exports.
- `train/preference/dpo_pairs/{all,train,eval,test}.jsonl`: preference pairs.
- `train/rl/rl_pairs/{all,train,eval,test}.jsonl`: RL-style pairs when `include_rl: true`.
- `metadata/dataset_manifest.json`: file inventory, split counts, summaries, and config metadata.
- `metadata/coverage_matrix.json`: 8 risk sources, 14 failure modes, 10 harm types, tool scenarios, pairwise matrices, and observed taxonomy triples.
- `metadata/quality_report.json`: QC, repair, production filtering, coverage, and split diagnostics.
- `rejected/qc_rejected_samples.jsonl` and `rejected/training_rejected_samples.jsonl` when samples are filtered.

Root-level files such as `synthetic_eval.jsonl`, `agentdog_unified_sft.jsonl`, `quality_report.json`, and `rejected_samples.jsonl` are still written by default as compatibility copies. Set `write_legacy_flat_files: false` in the generation config, or pass `--no-legacy-flat-files`, when you want a cleaner output directory.

AgentDoG training exports are split by objective:

- `agentdog_binary_sft.jsonl`: binary-only supervision. The assistant outputs only `Safe` or `Unsafe`.
- `agentdog_taxonomy_sft.jsonl`: taxonomy-only supervision. The assistant outputs only `Risk Source`, `Failure Mode`, and `Real World Harm`; safe/binary supervision is excluded.
- `agentdog_unified_sft.jsonl`: four-label supervision. The assistant outputs `Safety`, `Risk Source`, `Failure Mode`, and `Real World Harm`, using `None` for the three fine-grained labels when `Safety: Safe`.

The taxonomy-only and unified files use the AgentDoG-style `{"id": "...", "task": "...", "messages": [...]}` shape and embed the full 8/14/10 categorization block in the user prompt. `agentdog15_unified_sft.jsonl` is kept separately as the official AgentDoG 1.5 prompt-compatible export.
`synthetic_sft.jsonl` and `train/tracehound_risk_report_sft/*` now also embed the full AgentDoG 8/14/10 categorization block, but ask the model to return TraceHound's strict JSON `RiskReport` schema with machine-label fields. Preference and RL exports use deterministic hard negatives: unsafe samples keep the unsafe label while corrupting taxonomy or evidence, and safe samples receive plausible false-positive unsafe reports, instead of using a single fixed flipped-label template.

Official AgentDoG1.0 training exports are built separately with:

```bash
python scripts/build_agentdog_data.py --config configs/agentdog_data_flows.yaml --source agentdog10
```

This keeps official SFT base data, TraceHound synthetic data, APP1 safety-response data, and ATBench held-out evaluation data in distinct directories.

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

## Online Guardrail Integration

TraceHound can run as a runtime guardrail service, not just a demo simulator:

```bash
python scripts/serve_demo.py --host 127.0.0.1 --port 8000
```

Agent runtimes can then call:

```text
POST http://127.0.0.1:8000/api/guardrail/event
```

Claude Code can also call the native hook endpoint directly:

```text
POST http://127.0.0.1:8000/api/guardrail/claude-code?mode=layered&judge=heuristic
```

The same logic is available offline:

```bash
python scripts/guardrail_hook.py --platform generic --event-type pre_reply < event.json
```

Claude Code command hooks can use:

```bash
python scripts/guardrail_hook.py \
  --platform claude-code \
  --event-type auto \
  --server-url http://127.0.0.1:8000 \
  --adapter-json
```

To install those hooks into Claude Code without overwriting existing settings:

```bash
python scripts/install_claude_code_hooks.py \
  --settings .claude/settings.json \
  --tracehound-root /Users/a1234/Documents/Code/TraceHound \
  --server-url http://127.0.0.1:8000 \
  --python-command "conda run -n tracehound python"
```

The installer preserves existing settings, creates a timestamped backup, and
refreshes only TraceHound-managed hook entries.

Integration examples:

- `integrations/claude_code/settings.tracehound.example.json`
- `examples/integrations/codex_guardrail_wrapper.py`
- `examples/integrations/openclaw_guardrail_middleware.py`
- `docs/online_guardrail_integrations.md`

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

By default `validate_api.py` uses `.env` / `TRACEHOUND_API_BASE` / `TRACEHOUND_MODEL` exactly as configured. To use a packaged profile such as Intern's OpenAI-compatible endpoint, pass it explicitly:

```bash
python scripts/validate_api.py --data data/synthetic_eval.jsonl --limit 1 --judge api --api-profile intern-s2-preview
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

Guard Model training and safety enchantment are separate paths:

- `scripts/train_sft.py` and `scripts/train_preference.py` fine-tune the Guard Model itself so it judges agent trajectories better.
- `scripts/enchant_safety.py` uses the current Guard Model as filter, judge, and safety reward to fine-tune another target policy/base model.

Plan target-model safety enchantment without launching GPU work:

```bash
python scripts/enchant_safety.py \
  --data-dir data/tmp/generated/latest \
  --target-model-profile internlm3-8b-instruct \
  --algorithm sft_dpo \
  --output-dir checkpoints/safety_enchantment
```

See `docs/intern_model_playbook.md` for InternLM2.5-7B fallback, 20B notes, DPO/ORPO/GRPO planning, and cost-aware validation commands.

## Web Demo Features

The demo has four pages:

- `主页`: project positioning and telemetry summary.
- `Agent轨迹安全评估`: single-case evaluation, JSON/JSONL upload or drag-and-drop, batch evaluation, evidence timeline, online guard simulation, and JSON/Markdown/SVG report download.
- `Guard Model调配`: current serving mode, API/local model state, configurable data generation, live job progress, and Guard Model SFT / SFT+RL training preflight hooks.
- `安全能力附魔`: uses the active Guard Model to produce AgentDoG-style SFT/DPO/GRPO plans for improving a separate target policy/base model.

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
