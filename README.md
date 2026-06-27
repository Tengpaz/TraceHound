# TraceHound

TraceHound is a lightweight, CPU-first baseline for agent trajectory safety. It follows the AgentDoG trajectory-level diagnosis idea and keeps the first implementation easy to run on macOS while remaining portable to Linux contest servers.

The current MVP focuses on:

- AgentDoG-style `Risk Source / Failure Mode / Real-world Harm` labels.
- Synthetic trajectory data for eval, SFT, and preference formats.
- Rule-based prefiltering, risk span compression, and deterministic heuristic judging.
- Cost statistics and evaluation metrics.
- Dataset quality checks, final-answer-only baseline, and ablation reporting.
- A FastAPI demo with native HTML/CSS/JS.

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

## Quick Start

Generate synthetic data:

```bash
python scripts/generate_data.py --out data
```

The built-in generator currently emits 16 balanced cases across `shell`, `file`, `browser`, `email`, `database`, `code_executor`, `calendar`, and `credential` scenarios. You can filter it:

```bash
python scripts/generate_data.py --out data/tmp --scenario browser --label unsafe --limit 2
```

Run quality checks:

```bash
python scripts/quality_check.py data/synthetic_eval.jsonl
```

Run the layered baseline:

```bash
python scripts/evaluate.py data/synthetic_eval.jsonl --mode layered
```

Run offline ablations and generate a Markdown report:

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
TRACEHOUND_API_BASE=https://api.example.com/v1
TRACEHOUND_API_KEY=your-api-key
TRACEHOUND_MODEL=your-model-name
```

For most OpenAI-compatible services, keep:

```bash
TRACEHOUND_API_PATH=/chat/completions
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
The page also shows the configured remote model, redacted endpoint origin, selected inference strategy, model call count, and latency so judges can see when third-party API inference is actually used.

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
- `docs/training_gpu.md`: Optional Linux/GPU fine-tuning notes.
- `traceguard/`: Core Python package.
- `scripts/`: Data generation, evaluation, demo server, and training placeholders.
- `web_demo/`: FastAPI-served static assets.
- `tests/`: CPU-only test suite.
