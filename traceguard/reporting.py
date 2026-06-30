"""Experiment report rendering with dependency-free SVG charts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


METRIC_COLUMNS = [
    "samples",
    "accuracy",
    "precision",
    "recall",
    "f_score",
    "unsafe_recall",
    "false_block_rate",
    "macro_f1",
    "evidence_hit_rate",
    "evidence_precision",
    "evidence_recall",
    "average_model_calls",
    "average_latency_ms",
    "average_compression_ratio",
    "average_cost_reduction_ratio",
    "total_estimated_cost_usd",
    "average_estimated_cost_usd",
]

CHART_METRICS = [
    ("accuracy", "Accuracy"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f_score", "F-score"),
    ("evidence_hit_rate", "Evidence Hit"),
    ("average_cost_reduction_ratio", "Cost Reduction"),
]


def build_report(data: Dict[str, Any], chart_path: str | Path | None = None) -> str:
    chart_ref = Path(chart_path).name if chart_path else ""
    lines: List[str] = [
        "# TraceHound Experiment Report",
        "",
        f"- Data: `{data.get('data', '')}`",
        f"- Generated at: `{data.get('generated_at', '')}`",
        f"- Skipped: `{', '.join(data.get('skipped', [])) or 'none'}`",
        "",
    ]
    if chart_ref:
        lines.extend(["## Metric Chart", "", f"![Metric chart]({chart_ref})", ""])
    lines.extend(["## Ablation Metrics", ""])
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
            "- `accuracy` is the binary safe/unsafe accuracy.",
            "- `precision`, `recall`, and `f_score` use fine-grained unsafe taxonomy exact match: `risk_source`, `failure_mode`, and `harm_type` must all match.",
            "- `final_only` removes intermediate tool calls and observations, so it is expected to miss risks that occur before the final answer.",
            "- API and hybrid API rows are intentionally small by default to avoid unnecessary quota use.",
            "- `total_estimated_cost_usd` uses `TRACEHOUND_INPUT_PRICE_PER_1M` and `TRACEHOUND_OUTPUT_PRICE_PER_1M` when set.",
            "- GPU training is outside the default Mac validation path; see `docs/training_gpu.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_metric_chart(data: Dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_metric_chart_svg(data), encoding="utf-8")
    return output


def render_metric_chart_svg(data: Dict[str, Any]) -> str:
    experiments = [item for item in data.get("experiments", []) if not item.get("skipped")]
    width = 1060
    row_height = 30
    group_gap = 22
    header_height = 76
    footer_height = 44
    chart_width = 650
    label_x = 34
    bar_x = 290
    metric_label_x = bar_x + chart_width + 18
    height = header_height + max(1, len(experiments)) * (len(CHART_METRICS) * row_height + group_gap) + footer_height
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<defs>",
        '<linearGradient id="bar" x1="0" x2="1" y1="0" y2="0"><stop offset="0" stop-color="#69ffe6"/><stop offset="1" stop-color="#7aa7ff"/></linearGradient>',
        '<filter id="glow"><feGaussianBlur stdDeviation="2.5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        "</defs>",
        '<rect width="100%" height="100%" rx="18" fill="#061018"/>',
        '<rect x="1" y="1" width="1058" height="' + str(height - 2) + '" rx="17" fill="none" stroke="rgba(105,255,230,0.28)"/>',
        '<text x="34" y="38" fill="#eefcff" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="22" font-weight="800">TraceHound Metrics</text>',
        '<text x="34" y="60" fill="#8ba4b5" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="12">binary accuracy / fine-grained taxonomy precision-recall / evidence / cost compression</text>',
    ]
    y = header_height
    if not experiments:
        lines.append(
            '<text x="34" y="108" fill="#8ba4b5" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="14">No experiment rows.</text>'
        )
    for experiment in experiments:
        title = f"{experiment.get('name', '')} · {experiment.get('judge', '')} · {experiment.get('mode', '')}"
        lines.append(
            f'<text x="{label_x}" y="{y + 15}" fill="#eefcff" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="13" font-weight="800">{_escape(title)}</text>'
        )
        metrics = experiment.get("metrics", {})
        metric_y = y + 26
        for key, label in CHART_METRICS:
            value = _bounded_float(metrics.get(key, 0.0))
            bar_width = max(2, chart_width * value)
            lines.append(
                f'<text x="{label_x + 16}" y="{metric_y + 15}" fill="#8ba4b5" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="11">{_escape(label)}</text>'
            )
            lines.append(f'<rect x="{bar_x}" y="{metric_y + 4}" width="{chart_width}" height="14" rx="7" fill="rgba(255,255,255,0.07)"/>')
            lines.append(f'<rect x="{bar_x}" y="{metric_y + 4}" width="{bar_width:.2f}" height="14" rx="7" fill="url(#bar)" filter="url(#glow)"/>')
            lines.append(
                f'<text x="{metric_label_x}" y="{metric_y + 16}" fill="#d6f8ff" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="11">{value:.3f}</text>'
            )
            metric_y += row_height
        y += len(CHART_METRICS) * row_height + group_gap
    lines.append("</svg>")
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
        if abs(value) < 0.0001 and value != 0:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)


def _bounded_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, number))


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)
