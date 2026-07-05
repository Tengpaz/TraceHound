#!/usr/bin/env python
"""Build TraceHound datasets from public AgentDoG official releases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.agentdog_data import (
    build_agentdog10_data_flow,
    build_app1_sft_data_flow,
    build_atbench_eval_flow,
    ensure_official_sources,
    load_agentdog_data_flow_config,
)
from traceguard.config import load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agentdog_data_flows.yaml", help="Flat YAML data-flow config.")
    parser.add_argument("--env-file", default=".env", help="Load TRACEHOUND_API_* values from this env file.")
    parser.add_argument("--source", choices=["agentdog10", "app1", "atbench", "all"], help="Override source_mode.")
    parser.add_argument("--official-root", help="Downloaded official dataset root.")
    parser.add_argument("--output-root", help="Output root, usually data.")
    parser.add_argument("--limit", type=int, help="Limit records per source for smoke tests.")
    parser.add_argument("--download-official", action="store_true", help="Download required official datasets first.")
    parser.add_argument("--force-download", action="store_true", help="Replace existing official dataset snapshots.")
    parser.add_argument("--annotate-cot", action="store_true", help="Run LLM/stub CoT annotation.")
    parser.add_argument("--no-annotate-cot", action="store_true", help="Disable CoT annotation even if config enables it.")
    parser.add_argument("--cot-backend", choices=["api", "stub"], help="CoT annotation backend.")
    parser.add_argument("--cot-modes", help="Comma list: coarse,finegrained,unified or all.")
    parser.add_argument("--cot-concurrency", type=int, help="Concurrent CoT requests.")
    parser.add_argument("--cot-max-retries", type=int, help="Retries per CoT annotation sample.")
    parser.add_argument("--cot-temperature", type=float, help="LLM temperature for CoT annotation.")
    parser.add_argument("--checkpoint-dir", help="CoT checkpoint directory.")
    parser.add_argument("--resume", action="store_true", help="Resume CoT annotation checkpoints.")
    parser.add_argument("--pause-file", help="Pause while this file exists.")
    parser.add_argument(
        "--allow-atbench-cot-distill",
        action="store_true",
        help="Allow ATBench-derived CoT distillation artifacts with contamination warning.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    config = load_agentdog_data_flow_config(args.config)
    source_mode = args.source or str(config["source_mode"])
    official_root = Path(args.official_root or config["official_root"])
    output_root = Path(args.output_root or config["output_root"])
    limit = args.limit if args.limit is not None else config.get("limit")
    limit = int(limit) if limit not in (None, "", 0) else None
    annotate_cot = bool(config["annotate_cot"])
    if args.annotate_cot:
        annotate_cot = True
    if args.no_annotate_cot:
        annotate_cot = False
    cot_modes = _cot_modes_from_args(args.cot_modes, config.get("cot_modes") or [])
    cot_backend = args.cot_backend or str(config["cot_backend"])
    cot_concurrency = int(args.cot_concurrency or config["cot_concurrency"] or 1)
    cot_max_retries = int(args.cot_max_retries if args.cot_max_retries is not None else config["cot_max_retries"])
    cot_temperature = float(args.cot_temperature if args.cot_temperature is not None else config["cot_temperature"])
    checkpoint_dir = args.checkpoint_dir or config.get("checkpoint_dir") or None
    resume = bool(args.resume or config.get("resume"))
    pause_file = args.pause_file or config.get("pause_file") or None
    allow_atbench_cot_distill = bool(args.allow_atbench_cot_distill or config["allow_atbench_cot_distill"])
    download_official = bool(args.download_official or config["download_official"])

    summary: dict[str, Any] = {
        "source_mode": source_mode,
        "official_root": str(official_root),
        "output_root": str(output_root),
        "download_actions": [],
        "manifests": {},
    }
    if download_official:
        summary["download_actions"] = ensure_official_sources(source_mode, official_root, force=args.force_download)

    if source_mode in {"agentdog10", "all"}:
        summary["manifests"]["agentdog10"] = build_agentdog10_data_flow(
            official_root=official_root,
            output_root=output_root,
            limit=limit,
            annotate_cot=annotate_cot,
            cot_backend=cot_backend,
            cot_modes=cot_modes,
            cot_concurrency=cot_concurrency,
            cot_max_retries=cot_max_retries,
            cot_temperature=cot_temperature,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            pause_file=pause_file,
            split_train_ratio=float(config["split_train_ratio"]),
            split_eval_ratio=float(config["split_eval_ratio"]),
            split_test_ratio=float(config["split_test_ratio"]),
            split_seed=int(config["split_seed"]),
        )

    if source_mode in {"app1", "all"}:
        summary["manifests"]["app1"] = build_app1_sft_data_flow(
            official_root=official_root,
            output_root=output_root,
            limit=limit,
            split_train_ratio=float(config["split_train_ratio"]),
            split_eval_ratio=float(config["split_eval_ratio"]),
            split_test_ratio=float(config["split_test_ratio"]),
            split_seed=int(config["split_seed"]),
        )

    if source_mode in {"atbench", "all"}:
        summary["manifests"]["atbench"] = build_atbench_eval_flow(
            official_root=official_root,
            output_root=output_root,
            limit=limit,
            allow_cot_distill=allow_atbench_cot_distill,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metadata_dir / "agentdog_data_flow_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _cot_modes_from_args(value: str | None, config_value: Any) -> list[str]:
    if value:
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(config_value, list):
        items = [str(item) for item in config_value]
    elif config_value:
        items = [str(config_value)]
    else:
        items = []
    if not items or "all" in items:
        return ["coarse", "finegrained", "unified"]
    return items


if __name__ == "__main__":
    main()
