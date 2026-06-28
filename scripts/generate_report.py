#!/usr/bin/env python
"""Generate Markdown and SVG experiment reports from run_experiments JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.reporting import build_report, write_metric_chart


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="reports/experiments.json", help="Input experiment JSON.")
    parser.add_argument("--output", default="reports/experiment_report.md", help="Output Markdown path.")
    parser.add_argument("--chart-output", help="Output SVG chart path. Defaults next to Markdown.")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output = Path(args.output)
    chart_output = Path(args.chart_output) if args.chart_output else output.with_name(f"{output.stem}_metrics.svg")
    write_metric_chart(data, chart_output)
    markdown = build_report(data, chart_path=chart_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(json.dumps({"markdown": str(output), "chart": str(chart_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
