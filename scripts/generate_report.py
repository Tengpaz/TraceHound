#!/usr/bin/env python
"""Generate a Markdown experiment report from run_experiments JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRIC_COLUMNS = [
    "samples",
    "accuracy",
    "unsafe_recall",
    "false_block_rate",
    "macro_f1",
    "evidence_hit_rate",
    "evidence_precision",
    "evidence_recall",
    "average_model_calls",
    "average_latency_ms",
    "average_compression_ratio",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="reports/experiments.json", help="Input experiment JSON.")
    parser.add_argument("--output", default="reports/experiment_report.md", help="Output Markdown path.")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    markdown = build_report(data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(str(output))


def build_report(data: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# TraceHound Experiment Report",
        "",
        f"- Data: `{data.get('data', '')}`",
        f"- Generated at: `{data.get('generated_at', '')}`",
        f"- Skipped: `{', '.join(data.get('skipped', [])) or 'none'}`",
        "",
        "## Ablation Metrics",
        "",
    ]
    lines.extend(_metrics_table(data.get("experiments", [])))
    lines.extend(["", "## Online Guard", ""])
    guard = data.get("online_guard", {})
    for key in ("samples", "intervention_rate", "unsafe_intervention_recall", "safe_false_block_rate", "average_decision_step"):
        lines.append(f"- {key}: `{guard.get(key, '')}`")
    lines.append(f"- decisions: `{json.dumps(guard.get('decisions', {}), ensure_ascii=False, sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `final_only` removes intermediate tool calls and observations, so it is expected to miss risks that occur before the final answer.",
            "- API and hybrid API rows are intentionally small by default to avoid unnecessary quota use.",
            "- GPU training is outside the default Mac validation path; see `docs/training_gpu.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def _metrics_table(experiments: Iterable[Dict[str, Any]]) -> List[str]:
    header = ["name", "judge", "mode", *METRIC_COLUMNS]
    rows = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for experiment in experiments:
        if experiment.get("skipped"):
            continue
        metrics = experiment.get("metrics", {})
        row = [
            experiment.get("name", ""),
            experiment.get("judge", ""),
            experiment.get("mode", ""),
            *[_format_metric(metrics.get(key, "")) for key in METRIC_COLUMNS],
        ]
        rows.append("| " + " | ".join(str(item) for item in row) + " |")
    return rows


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
