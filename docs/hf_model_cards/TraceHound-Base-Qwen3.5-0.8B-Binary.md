---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
  - tracehound
  - agent-safety
  - guardrail
  - binary-classification
  - qwen
---

# TraceHound-Base-Qwen3.5-0.8B-Binary

This repository contains the TraceHound full-parameter SFT checkpoint for AgentDoG-style binary trajectory safety judging.

## Task

Input: an agent trajectory.

Output contract:

```json
{"judgment":"safe|unsafe"}
```

## Source

Local project path before upload:

```text
models/TraceHound-Base-Qwen3.5-0.8B-Binary/
```

## Reported Results

| Dataset | Accuracy | F1 | Invalid Rate |
|---|---:|---:|---:|
| ATBench / summer-camp | 0.7303 | 0.7791 | 0.0000 |
| R-Judge | 0.8103 | 0.8315 | 0.0000 |

## Notes

This checkpoint is intended for TraceHound demo guardrail inference and binary SFT reproduction. See the GitHub project README for data processing details, evaluation scripts, and related checkpoints.
