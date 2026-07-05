---
license: mit
task_categories:
- text-classification
language:
- en
pretty_name: AgentDoG-Lite-TrainningDataset-Binary
---

# AgentDoG-Lite-TrainningDataset-Binary

Binary SFT training data for the AgentDoG-Lite trajectory safety judgment task.

## Source

- Base training source: `AI45Research/AgentDoG1.0-Training-Data`, `AgentDoG-BinarySafety/train.json`.
- Evaluation references inspected for schema alignment only: `AI45Research/2026_summer_camp_teseset`.
- No `2026_summer_camp_teseset` samples are included in this training dataset.

## Task

Given a full AI-agent trajectory, judge whether the agent's actual behavior is safe or unsafe.
The instruction prompt preserves the AgentDoG1.0 BinarySafety task wording and safe/unsafe definitions.
The 8/14/10 taxonomy aid in the prompt uses the summer-camp snake_case label contract, for example `direct_prompt_injection`, `unconfirmed_or_over_privileged_action`, and `privacy_and_confidentiality_harm`.
The model output contract is:

Expected model output is exactly one strict JSON object:

```json
{
  "judgment": "safe"
}
```

or

```json
{
  "judgment": "unsafe"
}
```

## Files

- `train.json`: instruction/input/output JSON array.
- `train.jsonl`: instruction/input/output JSONL.
- `messages/train.jsonl`: chat-format SFT JSONL.
- `metadata/manifest.json`: source and artifact manifest.
- `metadata/quality_report.json`: output-format and label validation report.
- `metadata/eval_alignment_report.json`: schema comparison against the summer-camp eval references.

## Counts

- Samples: 4000
- Labels: `{"safe": 2000, "unsafe": 2000}`
- Rejected source rows: 0
- Quality pass: `True`

## Evaluation Reference Alignment

- ATBench300 rows inspected: 300
- R-Judge rows inspected: 564
- Both references use the binary label contract `0=safe, 1=unsafe`; the competition response contract is JSON `judgment`.

## Contamination Policy

This dataset is cleaned from AgentDoG1.0 official training data only. The AgentDoG-Lite summer-camp evaluation files and R-Judge-style evaluation files are not used for training rows, few-shot examples, prompt search, or test-set tuning.

