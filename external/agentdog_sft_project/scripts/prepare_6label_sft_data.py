#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "data/agentdog_complete_binary_safe_augmented_unsafe_train.jsonl"
DEFAULT_OUTPUT = "outputs/data/agentdog_6label_sft.jsonl"
DEFAULT_TEMPLATE = "src/guardrail/6 labels training prompt.md"
TARGET_FIELDS = (
    "risk_source",
    "failure_mode",
    "harm_type",
    "rationale",
    "source",
    "judgment",
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield line_number, json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc


def clean_markdown_escapes(text: str) -> str:
    text = text.replace("\\_", "_")
    text = text.replace("\\.", ".")
    text = text.replace("\\(", "(").replace("\\)", ")")
    text = text.replace("\\{", "{").replace("\\}", "}")
    return text


def load_prompt_template(path: Path) -> str:
    template = clean_markdown_escapes(path.read_text(encoding="utf-8")).strip()
    template = re.sub(r"\{render_trajectory\(example\)\}", "{trajectory}", template)
    if "{trajectory}" not in template:
        template = f"{template}\n\n<BEGIN TRAJECTORY>\n{{trajectory}}\n<END TRAJECTORY>"
    return template


def normalize_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_prompt(template: str, row: dict[str, Any]) -> str:
    trajectory = row.get("formatted_trajectory")
    if trajectory in (None, ""):
        raise KeyError("formatted_trajectory")
    return template.replace("{trajectory}", normalize_text(trajectory))


def build_target(row: dict[str, Any]) -> str:
    missing = [field for field in TARGET_FIELDS if field not in row or row[field] is None]
    if missing:
        raise KeyError(", ".join(missing))
    target = {field: normalize_text(row[field]) for field in TARGET_FIELDS}
    return json.dumps(target, ensure_ascii=False, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    template = load_prompt_template(Path(args.template))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for line_number, row in read_jsonl(input_path):
            if args.limit is not None and written >= args.limit:
                break
            try:
                prompt = build_prompt(template, row)
                target = build_target(row)
            except KeyError as exc:
                skipped += 1
                print(f"Skipping line {line_number}: missing {exc}")
                continue

            record = {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target},
                ]
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "template": args.template,
                "written": written,
                "skipped": skipped,
                "target_fields": TARGET_FIELDS,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
