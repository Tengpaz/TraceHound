#!/usr/bin/env python
"""Build AgentDoG-Lite binary training data with JSON judgment targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.agentdog_lite_binary import (  # noqa: E402
    DEFAULT_AGENTDOG_BINARY_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SUMMER_CAMP_ROOT,
    build_agentdog_lite_binary_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_AGENTDOG_BINARY_PATH), help="AgentDoG-BinarySafety train.json.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_ROOT), help="Output dataset directory.")
    parser.add_argument(
        "--summer-camp-root",
        default=str(DEFAULT_SUMMER_CAMP_ROOT),
        help="Optional local 2026_summer_camp_teseset directory for alignment reporting.",
    )
    parser.add_argument("--seed", type=int, default=20260704, help="Deterministic shuffle seed.")
    args = parser.parse_args()

    summer_root = Path(args.summer_camp_root) if args.summer_camp_root else None
    summary = build_agentdog_lite_binary_dataset(
        input_path=Path(args.input),
        output_root=Path(args.out),
        summer_camp_root=summer_root,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
