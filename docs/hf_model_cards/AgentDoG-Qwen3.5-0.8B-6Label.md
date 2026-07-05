---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
  - tracehound
  - agentdog
  - agent-safety
  - guardrail
  - six-label
  - reason-generation
  - qwen
---

# AgentDoG-Qwen3.5-0.8B-6Label

This repository contains the imported Qwen3.5-0.8B full-parameter SFT checkpoint for the six-label + reason AgentDoG trajectory safety task.

## Task

Input: an agent trajectory.

Expected JSON fields:

```json
{
  "judgment": "safe|unsafe",
  "source": "Safe|Unsafe|Benign|False_Refusal",
  "risk_source": "...",
  "failure_mode": "...",
  "harm_type": "...",
  "reason": "..."
}
```

## Source

Local project path before upload:

```text
checkpoints/imported/qwen35-0.8b-6label/
```

Original archive:

```text
/Users/a1234/Downloads/qwen35-0.8b-6label.tar.gz
```

## Reported Results

| Task | Dataset | Accuracy | Macro-F1 / F1 | Valid Output |
|---|---|---:|---:|---:|
| Six-label binary field | R-Judge | 0.8582 | 0.8581 | 100% |
| Train-set pass@8 | Train set | 1.0000 | 1.0000 | 100% valid rollout |

## Notes

The TraceHound repository also keeps GRPO training code for the six-label + reason task. Final GRPO result tables should be added after the full run is complete.
