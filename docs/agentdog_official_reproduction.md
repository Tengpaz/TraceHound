# AgentDoG Official Reproduction Mode

TraceHound separates two paths:

- `official_assets`: official AgentDoG repository, prompts, Hugging Face datasets, and SFT/RL recipes.
- `synthetic_surrogate`: TraceHound's local deterministic/LLM generator for pre-contest smoke testing.

The public AgentDoG release does not include the original LLM data synthesis engine or prompt-generation code. The official RL runtime README states that the runtime package intentionally excludes those components. Therefore, a faithful public reproduction should use the released official datasets and recipes rather than treating TraceHound's synthetic generator as the official generator.

## Prepare Official Assets

Install the optional downloader dependency when dataset download is needed:

```bash
python -m pip install -e ".[official]"
```

Clone the official repo and write a manifest:

```bash
python scripts/prepare_agentdog_official.py --clone-repo
```

Download selected official datasets:

```bash
python scripts/prepare_agentdog_official.py --download-dataset atbench --download-dataset app1_sft
```

Download all listed public datasets:

```bash
python scripts/prepare_agentdog_official.py --clone-repo --download-all
```

The manifest is written to:

```text
reports/agentdog_official_manifest.json
```

## Public Official Assets

- Repository: `https://github.com/AI45Lab/AgentDoG`
- ATBench: `AI45Research/ATBench`
- ATBench-Claw: `AI45Research/ATBench-Claw`
- ATBench-Codex: `AI45Research/ATBench-Codex`
- APP1 SFT data: `AI45Research/APP1-Agentic-Safety-SFT-Data`
- RL runtime data: `quantumfr/agentic-lightweight-envs-runtime-20260528`

## What This Means For Generation

For official reproduction, use official datasets and recipes. Do not call TraceHound's `scripts/generate_data.py` output official AgentDoG data. That generator is useful for smoke tests, UI validation, and contest adaptation, but it remains a surrogate because the official LLM synthesis engine is not publicly released.

TraceHound includes the public AgentDoG v1.5 prompt templates under:

```text
prompts/agentdog/v1.5/
```

These prompts are used for official-compatible inference/export paths.

## Paper-Described Synthesis Approximation

When official generated data is unavailable, TraceHound can run a paper-described approximation of the AgentDoG data engine:

1. Sample a target diagnostic tuple from the 8/14/10 taxonomy.
2. Select an agent setting, tool subset, and execution plan.
3. Ask an LLM to synthesize a full tool-mediated trajectory conditioned on that tuple.
4. Run deterministic purification for structure, tool legality, taxonomy alignment, evidence, and attack-success observability.
5. If semantic QC fails, optionally ask the LLM to self-repair the trajectory using the QC report.
6. Run optional LLM judge consensus QC.
7. Export raw QC, repair logs, final QC, and training-selection metadata.

Use:

```bash
python scripts/generate_data.py \
  --config configs/generation_agentdog_llm.yaml \
  --semantic-repair-backend llm
```

The `semantic_repair_backend` options are:

- `none`: reject semantic failures without repair.
- `llm`: LLM self-repair, closest to the public paper description.
- `static`: local deterministic salvage for low-cost smoke tests.
- `llm_then_static`: LLM self-repair with deterministic fallback.

For training-quality control, semantic-repaired samples are excluded from SFT/RL exports by default unless `training_max_repair_level` is relaxed.
