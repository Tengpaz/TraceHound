#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def iter_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            yield from json.load(f)
        else:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.json")
    parser.add_argument("--model", default="models/Qwen3.5-0.8B")
    parser.add_argument("--field", default="reason")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    max_tokens = -1
    max_index = -1
    max_text = ""
    count = 0
    missing = 0

    for index, row in enumerate(iter_rows(Path(args.data))):
        if args.field not in row or row[args.field] is None:
            missing += 1
            continue
        text = str(row[args.field])
        token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        count += 1
        if token_count > max_tokens:
            max_tokens = token_count
            max_index = index
            max_text = text

    print(f"data: {args.data}")
    print(f"model/tokenizer: {args.model}")
    print(f"field: {args.field}")
    print(f"rows_with_field: {count}")
    print(f"rows_missing_field: {missing}")
    print(f"max_tokens: {max_tokens}")
    print(f"max_index: {max_index}")
    print("max_reason_preview:")
    print(max_text[:1000])


if __name__ == "__main__":
    main()
