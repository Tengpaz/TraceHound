# TraceHound Contest Playbook

This playbook is the first-response checklist for a new contest task. The default path is Mac CPU plus optional third-party API validation. GPU training is only moved to a Linux/GPU server when contest rules and time make it worthwhile.

## 10-Step Response Flow

1. Read the official task statement, output schema, scoring rule, and submission format.
2. Copy raw official files under `data/official/`; do not overwrite synthetic data.
3. Convert a small sample into TraceHound schema:

   ```bash
   python scripts/convert_dataset.py data/official/sample.jsonl data/tmp/official_eval.jsonl --limit 20
   ```

4. Run schema and evidence checks:

   ```bash
   python scripts/quality_check.py data/tmp/official_eval.jsonl
   ```

5. Run offline baselines:

   ```bash
   python scripts/evaluate.py data/tmp/official_eval.jsonl --mode rules
   python scripts/evaluate.py data/tmp/official_eval.jsonl --mode layered
   ```

6. Run ablations and generate a report:

   ```bash
   python scripts/run_experiments.py --data data/tmp/official_eval.jsonl --no-api
   python scripts/generate_report.py --input reports/experiments.json --output reports/experiment_report.md
   ```

7. If an API model service is allowed, set `.env` and run a one-row smoke before larger validation:

   ```bash
   python scripts/validate_api.py --data data/tmp/official_eval.jsonl --limit 1 --judge api
   ```

8. Update `traceguard/rules.py`, `traceguard/prompts.py`, or `scripts/convert_dataset.py` only where the official format or scoring requires it.
9. If training is allowed and useful, move to a Linux/GPU server and follow `docs/training_gpu.md`.
10. Before submission, run the exact output command on a clean converted sample and inspect JSON validity, taxonomy labels, evidence steps, and cost fields.

## Environment Split

- Mac CPU: schema, conversion, rules, compression, heuristic eval, API smoke, FastAPI demo, tests.
- Linux CPU: full offline evaluation, report generation, packaging, official submission dry runs.
- Linux GPU: optional SFT, LoRA, DPO/ORPO, model serving, and larger API/model comparisons.

## Adaptation Rules

- Keep the internal schema stable: `id`, `task`, `metadata`, `trajectory`, optional `gold`.
- Keep taxonomy labels close to AgentDoG 8/14/10, with `none` only for safe samples.
- Prefer small API smoke tests before full remote runs.
- Do not add GPU dependencies to the default conda environment.
- Add contest-specific converters instead of changing core schema validation.

