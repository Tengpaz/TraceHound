#!/usr/bin/env python
"""Build AgentDoG fine-grained unsafe source/reason training data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.agentdog_finegrained_source_reason import (  # noqa: E402
    DEFAULT_AGENTDOG_FINEGRAINED_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    build_finegrained_source_reason_dataset,
)
from traceguard.config import load_env_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_AGENTDOG_FINEGRAINED_PATH), help="AgentDoG-FineGrainedTaxonomy train.json.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_ROOT), help="Output dataset directory.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of rows to annotate and merge.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED, help="Deterministic sample seed.")
    parser.add_argument("--backend", choices=["api", "stub"], default="api", help="Annotation backend.")
    parser.add_argument("--concurrency", type=int, default=8, help="Initial annotation concurrency.")
    parser.add_argument("--fallback-concurrency", type=int, default=6, help="Fallback concurrency for transient API failures.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries per annotation row.")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM annotation temperature.")
    parser.add_argument(
        "--source-policy",
        choices=["raw", "sanitized", "minimal"],
        default="raw",
        help="Trajectory form sent to the annotation model; raw auto-falls back on content-filter errors.",
    )
    parser.add_argument("--checkpoint-dir", help="Annotation checkpoint directory.")
    parser.add_argument("--resume", action="store_true", help="Resume from annotation checkpoints.")
    parser.add_argument("--retry-rejected", action="store_true", help="With --resume, retry previously rejected annotation rows.")
    parser.add_argument("--limit", type=int, help="Limit source records for smoke tests.")
    parser.add_argument("--env-file", default=".env", help="Load TRACEHOUND_API_* values from this env file.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress lines.")
    args = parser.parse_args()

    load_env_file(args.env_file)
    summary = build_finegrained_source_reason_dataset(
        input_path=Path(args.input),
        output_root=Path(args.out),
        sample_size=args.sample_size,
        seed=args.seed,
        backend=args.backend,
        concurrency=args.concurrency,
        fallback_concurrency=args.fallback_concurrency,
        max_retries=args.max_retries,
        temperature=args.temperature,
        source_policy=args.source_policy,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        retry_rejected=args.retry_rejected,
        limit=args.limit,
        progress_callback=None if args.quiet else _print_progress,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _print_progress(event: Mapping[str, Any]) -> None:
    status = event.get("status")
    if status == "fallback_concurrency":
        print(
            "source_reason_annotation fallback "
            f"{event.get('from_concurrency')}->{event.get('to_concurrency')} "
            f"retrying={event.get('retrying')}",
            flush=True,
        )
        return
    accepted = event.get("accepted")
    rejected = event.get("rejected")
    total = event.get("total")
    if accepted is None or rejected is None or total is None:
        return
    completed = int(accepted) + int(rejected)
    if completed == int(total) or completed % 25 == 0:
        print(
            f"source_reason_annotation {completed}/{total} accepted={accepted} rejected={rejected}",
            flush=True,
        )


if __name__ == "__main__":
    main()
