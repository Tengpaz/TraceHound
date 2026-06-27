"""Low-cost token and latency accounting helpers."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

from traceguard.schema import CostStats


def estimate_tokens(value: Any) -> int:
    """Approximate tokens without model-specific tokenizers."""

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    return max(1, (len(text) + 3) // 4)


@contextmanager
def measure_latency() -> Iterator[Dict[str, int]]:
    bucket = {"latency_ms": 0}
    start = time.perf_counter()
    try:
        yield bucket
    finally:
        bucket["latency_ms"] = int((time.perf_counter() - start) * 1000)


def build_cost(
    *,
    full_input: Any,
    compressed_input: Any,
    output: Any,
    latency_ms: int,
    model_calls: int,
    strategy: str,
    early_exit: bool,
) -> CostStats:
    full_tokens = estimate_tokens(full_input)
    input_tokens = estimate_tokens(compressed_input)
    output_tokens = estimate_tokens(output)
    compression_ratio = input_tokens / full_tokens if full_tokens else 1.0
    cost_reduction_ratio = 1 - compression_ratio
    return CostStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_calls=model_calls,
        strategy=strategy,
        early_exit=early_exit,
        compression_ratio=round(compression_ratio, 4),
        cost_reduction_ratio=round(max(0.0, cost_reduction_ratio), 4),
    )

