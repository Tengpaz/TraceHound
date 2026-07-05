---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
  - tracehound
  - agentdog
  - agent-safety
  - guardrail
  - lora
  - binary-classification
  - qwen
---

# TraceHound-Qwen3.5-0.8B-Lite-Binary-LoRA

This repository contains the TraceHound LoRA checkpoint for the AgentDoG-Lite binary safety task.

## Contents

The upload preserves:

- final LoRA adapter files at the repository root
- `checkpoint-150/`
- optimizer, scheduler, RNG state, and trainer state files
- tokenizer files
- train/validation split snapshots
- training plan and evaluation metrics

## Source

Local project path before upload:

```text
checkpoints/qwen3_5_0_8b_lite_binary_preserve_target_150039/
```

## Notes

This checkpoint is included for LoRA-vs-full-SFT comparison and lightweight reproduction.
