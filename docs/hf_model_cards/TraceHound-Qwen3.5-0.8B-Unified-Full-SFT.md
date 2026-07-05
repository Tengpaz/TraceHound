---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
  - tracehound
  - agentdog
  - agent-safety
  - guardrail
  - full-parameter-sft
  - taxonomy
  - qwen
---

# TraceHound-Qwen3.5-0.8B-Unified-Full-SFT

This repository contains the TraceHound full-parameter unified SFT experiment checkpoint directory.

## Contents

The upload preserves the local experiment directory, including:

- final model files at the repository root
- `checkpoint-200/`
- `checkpoint-225/`
- optimizer, scheduler, RNG state, and trainer state files
- tokenizer files
- train/validation split snapshots
- training plan and evaluation metrics

## Source

Local project path before upload:

```text
checkpoints/qwen3_5_0_8b_full_sft_unified_notrunc_20260705_000140/
```

## Notes

This checkpoint is kept as a reproducibility artifact for the unified/full-SFT experiment. It includes large optimizer states so training can be resumed from intermediate checkpoints.
