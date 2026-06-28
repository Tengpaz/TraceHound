"""Low-cost token and latency accounting helpers."""

from __future__ import annotations

import json
import os
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
    input_cost, output_cost, pricing_note = estimate_api_cost(input_tokens, output_tokens, model_calls)
    return CostStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        model_calls=model_calls,
        strategy=strategy,
        early_exit=early_exit,
        compression_ratio=round(compression_ratio, 4),
        cost_reduction_ratio=round(max(0.0, cost_reduction_ratio), 4),
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        estimated_cost_usd=round(input_cost + output_cost, 8),
        pricing_note=pricing_note,
    )


def estimate_api_cost(input_tokens: int, output_tokens: int, model_calls: int) -> tuple[float, float, str]:
    """Estimate API cost using optional per-1M token pricing env vars."""

    if model_calls <= 0:
        return 0.0, 0.0, "offline_or_rule_path"
    input_price = _env_float("TRACEHOUND_INPUT_PRICE_PER_1M", 0.0)
    output_price = _env_float("TRACEHOUND_OUTPUT_PRICE_PER_1M", 0.0)
    if input_price <= 0 and output_price <= 0:
        return 0.0, 0.0, "set TRACEHOUND_INPUT_PRICE_PER_1M and TRACEHOUND_OUTPUT_PRICE_PER_1M for API cost estimates"
    input_cost = input_tokens / 1_000_000 * input_price
    output_cost = output_tokens / 1_000_000 * output_price
    return round(input_cost, 8), round(output_cost, 8), "estimated_from_env_per_1m_token_prices"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
