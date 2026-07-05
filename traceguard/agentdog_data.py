"""Official AgentDoG data-flow builders and CoT annotation helpers."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Sequence

from traceguard.agentdog_official import OFFICIAL_DATASET_FILES, download_official_dataset
from traceguard.export import (
    AGENTDOG_TAXONOMY_TASK_PROMPT,
    AGENTDOG_UNIFIED_TASK_PROMPT,
    write_jsonl,
)
from traceguard.llm_generation import OpenAICompatibleChatClient
from traceguard.taxonomy import normalize_failure_mode, normalize_harm_type, normalize_risk_source


DEFAULT_AGENTDOG_DATA_FLOW_CONFIG: Dict[str, Any] = {
    "source_mode": "agentdog10",
    "official_root": "external/agentdog_official/datasets",
    "output_root": "data",
    "download_official": False,
    "annotate_cot": False,
    "cot_backend": "api",
    "cot_modes": ["coarse", "finegrained", "unified"],
    "cot_concurrency": 1,
    "cot_max_retries": 1,
    "cot_temperature": 0.0,
    "checkpoint_dir": "",
    "resume": False,
    "pause_file": "",
    "allow_atbench_cot_distill": False,
    "limit": None,
    "split_train_ratio": 0.8,
    "split_eval_ratio": 0.1,
    "split_test_ratio": 0.1,
    "split_seed": 20260701,
}

AGENTDOG10_DATASET_KEY = "agentdog10_training"
APP1_DATASET_KEY = "app1_sft"
ATBENCH_DATASET_KEY = "atbench"


def load_agentdog_data_flow_config(path: str | Path | None) -> Dict[str, Any]:
    config = dict(DEFAULT_AGENTDOG_DATA_FLOW_CONFIG)
    if not path:
        return config
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"AgentDoG data-flow config not found: {config_path}")
    parsed = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    config.update({key: value for key, value in parsed.items() if key in config})
    config["cot_modes"] = _normalize_list(config.get("cot_modes"))
    if not config["cot_modes"] or "all" in config["cot_modes"]:
        config["cot_modes"] = ["coarse", "finegrained", "unified"]
    return config


def build_agentdog10_data_flow(
    *,
    official_root: Path,
    output_root: Path,
    limit: int | None = None,
    annotate_cot: bool = False,
    cot_backend: str = "api",
    cot_modes: Sequence[str] = ("coarse", "finegrained", "unified"),
    cot_concurrency: int = 1,
    cot_max_retries: int = 1,
    cot_temperature: float = 0.0,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    pause_file: str | Path | None = None,
    split_train_ratio: float = 0.8,
    split_eval_ratio: float = 0.1,
    split_test_ratio: float = 0.1,
    split_seed: int = 20260701,
) -> Dict[str, Any]:
    dataset_dir = official_root / AGENTDOG10_DATASET_KEY
    files = _required_files(dataset_dir, AGENTDOG10_DATASET_KEY, ("binary_safety", "finegrained_taxonomy"))
    binary_raw = _limit_records(_read_json_array(files["binary_safety"]), limit)
    taxonomy_raw = _limit_records(_read_json_array(files["finegrained_taxonomy"]), limit)

    raw_root = output_root / "raw" / "official" / AGENTDOG10_DATASET_KEY
    _copy_raw_file(files["binary_safety"], raw_root / "AgentDoG-BinarySafety" / "train.json")
    _copy_raw_file(files["finegrained_taxonomy"], raw_root / "AgentDoG-FineGrainedTaxonomy" / "train.json")

    binary_rows = agentdog10_instruction_rows(
        binary_raw,
        source_config="AgentDoG-BinarySafety",
        task_type="official_agentdog10_binary_safety",
        id_prefix="agentdog10-binary",
    )
    taxonomy_rows = agentdog10_instruction_rows(
        taxonomy_raw,
        source_config="AgentDoG-FineGrainedTaxonomy",
        task_type="official_agentdog10_taxonomy_only",
        id_prefix="agentdog10-taxonomy",
    )
    unified_rows = derive_agentdog10_unified_rows(binary_rows, taxonomy_rows)

    artifacts: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    _write_split_bundle(
        output_root / "sft" / "official_agentdog10" / "binary_safety",
        binary_rows,
        split_train_ratio=split_train_ratio,
        split_eval_ratio=split_eval_ratio,
        split_test_ratio=split_test_ratio,
        split_seed=split_seed,
        counts=counts,
        artifacts=artifacts,
        artifact_prefix="agentdog10_binary_safety",
    )
    _write_split_bundle(
        output_root / "sft" / "official_agentdog10" / "taxonomy_only",
        taxonomy_rows,
        split_train_ratio=split_train_ratio,
        split_eval_ratio=split_eval_ratio,
        split_test_ratio=split_test_ratio,
        split_seed=split_seed,
        counts=counts,
        artifacts=artifacts,
        artifact_prefix="agentdog10_taxonomy_only",
    )
    _write_split_bundle(
        output_root / "sft" / "official_agentdog10" / "unified_four_label",
        unified_rows,
        split_train_ratio=split_train_ratio,
        split_eval_ratio=split_eval_ratio,
        split_test_ratio=split_test_ratio,
        split_seed=split_seed,
        counts=counts,
        artifacts=artifacts,
        artifact_prefix="agentdog10_unified_four_label",
    )

    cot_summary: Dict[str, Any] = {"enabled": False}
    if annotate_cot:
        cot_summary = annotate_agentdog10_cot(
            output_root=output_root,
            binary_rows=binary_rows,
            taxonomy_rows=taxonomy_rows,
            unified_rows=unified_rows,
            backend=cot_backend,
            modes=cot_modes,
            concurrency=cot_concurrency,
            max_retries=cot_max_retries,
            temperature=cot_temperature,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            pause_file=pause_file,
            split_train_ratio=split_train_ratio,
            split_eval_ratio=split_eval_ratio,
            split_test_ratio=split_test_ratio,
            split_seed=split_seed,
            counts=counts,
            artifacts=artifacts,
        )

    manifest = {
        "schema_version": "tracehound.agentdog_data_flow.v1",
        "created_at": _utc_now(),
        "source": AGENTDOG10_DATASET_KEY,
        "source_dataset": "AI45Research/AgentDoG1.0-Training-Data",
        "taxonomy_profile": "agentdog1.0_8_14_10",
        "counts": counts,
        "artifacts": artifacts,
        "cot": cot_summary,
        "notes": [
            "Official AgentDoG1.0 training records are kept as supervised SFT base data.",
            "CoT annotations, when enabled, are generated as derivative supervision and never overwrite official labels.",
        ],
    }
    manifest_path = output_root / "metadata" / "agentdog10_data_flow_manifest.json"
    _write_json(manifest_path, manifest)
    artifacts["agentdog10_manifest"] = str(manifest_path)
    return manifest


def build_app1_sft_data_flow(
    *,
    official_root: Path,
    output_root: Path,
    limit: int | None = None,
    split_train_ratio: float = 0.8,
    split_eval_ratio: float = 0.1,
    split_test_ratio: float = 0.1,
    split_seed: int = 20260701,
) -> Dict[str, Any]:
    dataset_dir = official_root / APP1_DATASET_KEY
    files = _required_files(dataset_dir, APP1_DATASET_KEY, ("safety_response_sft",))
    raw_records = _limit_records(_read_json_array(files["safety_response_sft"]), limit)
    raw_root = output_root / "raw" / "official" / APP1_DATASET_KEY
    _copy_raw_file(files["safety_response_sft"], raw_root / "agentic_safety_sft.json")

    rows = app1_safety_response_rows(raw_records)
    artifacts: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    _write_split_bundle(
        output_root / "sft" / "official_app1" / "safety_response_sft",
        rows,
        split_train_ratio=split_train_ratio,
        split_eval_ratio=split_eval_ratio,
        split_test_ratio=split_test_ratio,
        split_seed=split_seed,
        counts=counts,
        artifacts=artifacts,
        artifact_prefix="app1_safety_response_sft",
    )
    manifest = {
        "schema_version": "tracehound.agentdog_data_flow.v1",
        "created_at": _utc_now(),
        "source": APP1_DATASET_KEY,
        "source_dataset": "AI45Research/APP1-Agentic-Safety-SFT-Data",
        "task_family": "safety_response_sft",
        "counts": counts,
        "artifacts": artifacts,
        "notes": [
            "APP1 rows train safe agent behavior and are not treated as Guard Model taxonomy labels by default.",
        ],
    }
    manifest_path = output_root / "metadata" / "app1_sft_data_flow_manifest.json"
    _write_json(manifest_path, manifest)
    artifacts["app1_manifest"] = str(manifest_path)
    return manifest


def build_atbench_eval_flow(
    *,
    official_root: Path,
    output_root: Path,
    limit: int | None = None,
    allow_cot_distill: bool = False,
) -> Dict[str, Any]:
    dataset_dir = official_root / ATBENCH_DATASET_KEY
    files = _required_files(dataset_dir, ATBENCH_DATASET_KEY, ("latest_test",))
    raw_records = _limit_records(_read_json_array(files["latest_test"]), limit)
    suite_records: list[tuple[str, list[Dict[str, Any]]]] = [(ATBENCH_DATASET_KEY, raw_records)]
    for optional_key in ("atbench_claw", "atbench_codex"):
        optional_dir = official_root / optional_key
        optional_files = OFFICIAL_DATASET_FILES.get(optional_key, {})
        optional_path = optional_dir / str(optional_files.get("test", "test.json"))
        if optional_path.exists():
            suite_records.append((optional_key, _limit_records(_read_json_array(optional_path), limit)))
    raw_root = output_root / "raw" / "official" / ATBENCH_DATASET_KEY
    _copy_raw_file(files["latest_test"], raw_root / "ATBench" / "test.json")

    rows: list[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    for source_key, records in suite_records:
        converted = atbench_eval_rows(records, dataset_key=source_key)
        rows.extend(converted)
        source_counts[source_key] = len(converted)
    artifacts: Dict[str, str] = {}
    counts: Dict[str, int] = {}
    eval_path = output_root / "processed" / "official_atbench" / "eval_only" / "all.jsonl"
    counts["processed/official_atbench/eval_only/all.jsonl"] = write_jsonl(eval_path, rows)
    artifacts["atbench_eval_only"] = str(eval_path)
    if allow_cot_distill:
        warning_path = output_root / "metadata" / "atbench_cot_distill_warning.txt"
        warning_path.parent.mkdir(parents=True, exist_ok=True)
        warning_path.write_text(
            "ATBench-derived CoT distillation data must not be used for same-source ATBench public evaluation.\n",
            encoding="utf-8",
        )
        artifacts["atbench_cot_distill_warning"] = str(warning_path)

    manifest = {
        "schema_version": "tracehound.agentdog_data_flow.v1",
        "created_at": _utc_now(),
        "source": ATBENCH_DATASET_KEY,
        "source_dataset": "AI45Research/ATBench",
        "source_counts": source_counts,
        "task_family": "heldout_eval",
        "counts": counts,
        "artifacts": artifacts,
        "allow_atbench_cot_distill": allow_cot_distill,
        "contamination_warning": (
            "ATBench is a held-out benchmark. Do not train on ATBench-derived rows when reporting ATBench scores."
        ),
    }
    manifest_path = output_root / "metadata" / "atbench_eval_flow_manifest.json"
    _write_json(manifest_path, manifest)
    artifacts["atbench_manifest"] = str(manifest_path)
    return manifest


def ensure_official_sources(source_mode: str, official_root: Path, *, force: bool = False) -> list[Dict[str, Any]]:
    keys = []
    if source_mode in {"agentdog10", "all"}:
        keys.append(AGENTDOG10_DATASET_KEY)
    if source_mode in {"app1", "all"}:
        keys.append(APP1_DATASET_KEY)
    if source_mode in {"atbench", "all"}:
        keys.extend([ATBENCH_DATASET_KEY, "atbench_claw", "atbench_codex"])
    return [download_official_dataset(key, official_root / key, force=force) for key in keys]


def agentdog10_instruction_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    source_config: str,
    task_type: str,
    id_prefix: str,
) -> list[Dict[str, Any]]:
    rows = []
    for index, record in enumerate(records, start=1):
        instruction = str(record.get("instruction") or "").strip()
        input_text = str(record.get("input") or "").strip()
        output = str(record.get("output") or "").strip()
        if not instruction or not output:
            continue
        prompt = instruction if not input_text else instruction + "\n\n" + input_text
        rows.append(
            {
                "id": f"{id_prefix}-{index:06d}",
                "task": source_config,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": output},
                ],
                "source": {
                    "dataset": AGENTDOG10_DATASET_KEY,
                    "config": source_config,
                    "index": index,
                },
                "task_type": task_type,
            }
        )
    return rows


def app1_safety_response_rows(records: Sequence[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    rows = []
    for index, record in enumerate(records, start=1):
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        rows.append(
            {
                "id": f"app1-safety-response-{index:06d}",
                "task": "official_app1_safety_response_sft",
                "messages": messages,
                "tools": record.get("tools") or [],
                "source": {
                    "dataset": APP1_DATASET_KEY,
                    "index": index,
                },
                "task_type": "official_app1_safety_response_sft",
            }
        )
    return rows


def atbench_eval_rows(records: Sequence[Mapping[str, Any]], *, dataset_key: str = ATBENCH_DATASET_KEY) -> list[Dict[str, Any]]:
    rows = []
    for index, record in enumerate(records, start=1):
        label_value = record.get("label")
        label = "unsafe" if label_value in {1, "1", "unsafe", "Unsafe"} else "safe"
        rows.append(
            {
                "id": f"{dataset_key}-{record.get('id') or index}",
                "task": "official_atbench_eval_only",
                "tool_used": record.get("tool_used") or [],
                "contents": record.get("contents") or [],
                "label": label,
                "risk_source": record.get("risk_source") or "none",
                "failure_mode": record.get("failure_mode") or "none",
                "real_world_harm": record.get("real_world_harm") or "none",
                "reason": record.get("reason") or "",
                "source": {
                    "dataset": dataset_key,
                    "split": "test",
                    "index": index,
                },
            }
        )
    return rows


def derive_agentdog10_unified_rows(
    binary_rows: Sequence[Dict[str, Any]],
    taxonomy_rows: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for row in binary_rows:
        label = _parse_safety_label(row["messages"][-1]["content"])
        if label != "safe":
            continue
        rows.append(_derived_unified_row(row, "Safe", None, len(rows) + 1))
    for row in taxonomy_rows:
        labels = parse_taxonomy_output(row["messages"][-1]["content"])
        if not labels:
            continue
        rows.append(_derived_unified_row(row, "Unsafe", labels, len(rows) + 1))
    return rows


def parse_taxonomy_output(text: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = _normalize_key(key)
        value = value.strip()
        if not value:
            continue
        if normalized_key == "risk_source":
            parsed["risk_source"] = value
        elif normalized_key == "failure_mode":
            parsed["failure_mode"] = value
        elif normalized_key in {"real_world_harm", "risk_consequence"}:
            parsed["real_world_harm"] = value
    if {"risk_source", "failure_mode", "real_world_harm"}.issubset(parsed):
        return parsed
    return {}


def annotate_agentdog10_cot(
    *,
    output_root: Path,
    binary_rows: Sequence[Dict[str, Any]],
    taxonomy_rows: Sequence[Dict[str, Any]],
    unified_rows: Sequence[Dict[str, Any]],
    backend: str,
    modes: Sequence[str],
    concurrency: int,
    max_retries: int,
    temperature: float,
    checkpoint_dir: str | Path | None,
    resume: bool,
    pause_file: str | Path | None,
    split_train_ratio: float,
    split_eval_ratio: float,
    split_test_ratio: float,
    split_seed: int,
    counts: Dict[str, int],
    artifacts: Dict[str, str],
) -> Dict[str, Any]:
    if backend not in {"api", "stub"}:
        raise ValueError("cot_backend must be api or stub")
    modes = [mode for mode in modes if mode in {"coarse", "finegrained", "unified"}]
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else output_root / "metadata" / "_cot_checkpoints"
    summary: Dict[str, Any] = {
        "enabled": True,
        "backend": backend,
        "modes": modes,
        "concurrency": max(1, int(concurrency or 1)),
        "checkpoint_dir": str(checkpoint_root),
        "mode_summaries": {},
    }
    mode_sources = {
        "coarse": binary_rows,
        "finegrained": taxonomy_rows,
        "unified": unified_rows,
    }
    for mode in modes:
        rows, mode_summary = _annotate_cot_mode(
            mode=mode,
            rows=mode_sources[mode],
            backend=backend,
            concurrency=concurrency,
            max_retries=max_retries,
            temperature=temperature,
            checkpoint_dir=checkpoint_root / mode,
            resume=resume,
            pause_file=pause_file,
        )
        summary["mode_summaries"][mode] = mode_summary
        _write_split_bundle(
            output_root / "sft" / "official_agentdog10" / f"{mode}_cot",
            rows,
            split_train_ratio=split_train_ratio,
            split_eval_ratio=split_eval_ratio,
            split_test_ratio=split_test_ratio,
            split_seed=split_seed,
            counts=counts,
            artifacts=artifacts,
            artifact_prefix=f"agentdog10_{mode}_cot",
        )
        if mode_summary.get("rejected_samples"):
            rejected_path = output_root / "rejected" / f"{mode}_cot_annotation_rejected.jsonl"
            write_jsonl(rejected_path, mode_summary["rejected_samples"])
            artifacts[f"agentdog10_{mode}_cot_rejected"] = str(rejected_path)
            counts[f"rejected/{mode}_cot_annotation_rejected.jsonl"] = len(mode_summary["rejected_samples"])
    return summary


def _annotate_cot_mode(
    *,
    mode: str,
    rows: Sequence[Dict[str, Any]],
    backend: str,
    concurrency: int,
    max_retries: int,
    temperature: float,
    checkpoint_dir: Path,
    resume: bool,
    pause_file: str | Path | None,
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    concurrency = max(1, int(concurrency or 1))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = checkpoint_dir / "accepted.jsonl"
    rejected_path = checkpoint_dir / "rejected.jsonl"
    state_path = checkpoint_dir / "state.json"
    if not resume:
        accepted_path.unlink(missing_ok=True)
        rejected_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
    accepted_by_id = {
        str(row.get("source_id") or row.get("id")): row
        for row in _load_jsonl(accepted_path)
        if isinstance(row, dict) and (row.get("source_id") or row.get("id"))
    }
    rejected_by_id = {
        str(row.get("source_id") or row.get("id")): row
        for row in _load_jsonl(rejected_path)
        if isinstance(row, dict) and (row.get("source_id") or row.get("id"))
    }
    client = OpenAICompatibleChatClient() if backend == "api" else None
    lock = Lock()

    def run_one(index: int, row: Dict[str, Any]) -> tuple[str, Dict[str, Any] | None, Dict[str, Any] | None]:
        source_id = str(row.get("id") or f"{mode}-{index:06d}")
        try:
            oracle = _oracle_from_row(mode, row)
            prompt = _cot_prompt(mode, row, oracle)
            for attempt in range(1, max_retries + 2):
                if backend == "stub":
                    content = _stub_cot_output(mode, oracle)
                else:
                    assert client is not None
                    content = client.complete(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "You generate concise AgentDoG training rationales. "
                                    "Preserve the provided oracle labels exactly."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=temperature,
                    )
                ok, reason = _validate_cot_output(mode, content, oracle)
                if ok:
                    annotated = {
                        "id": f"agentdog10-{mode}-cot-{index:06d}",
                        "task": f"agentdog10_{mode}_cot",
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": content.strip()},
                        ],
                        "source_id": source_id,
                        "source": row.get("source") or {},
                        "oracle": oracle,
                        "annotation": {
                            "backend": backend,
                            "attempts": attempt,
                            "validated": True,
                        },
                    }
                    return source_id, annotated, None
            return source_id, None, {"source_id": source_id, "mode": mode, "error": reason, "oracle": oracle}
        except Exception as exc:
            return source_id, None, {"source_id": source_id, "mode": mode, "error": str(exc)[:1000]}

    def record(result: tuple[str, Dict[str, Any] | None, Dict[str, Any] | None]) -> None:
        source_id, accepted, rejected = result
        with lock:
            if accepted is not None:
                accepted_by_id[source_id] = accepted
                rejected_by_id.pop(source_id, None)
                _append_jsonl(accepted_path, accepted)
            elif rejected is not None:
                rejected_by_id[source_id] = rejected
                _append_jsonl(rejected_path, rejected)
            _write_json(
                state_path,
                {
                    "mode": mode,
                    "updated_at": _utc_now(),
                    "accepted": len(accepted_by_id),
                    "rejected": len(rejected_by_id),
                    "total": len(rows),
                    "concurrency": concurrency,
                },
            )

    processed = set(accepted_by_id) | set(rejected_by_id)
    pending = [(index, row) for index, row in enumerate(rows, start=1) if str(row.get("id")) not in processed]
    if concurrency == 1:
        for item in pending:
            _wait_if_paused(pause_file)
            record(run_one(*item))
    else:
        pending_iter = iter(pending)
        active: dict[Future, tuple[int, Dict[str, Any]]] = {}
        exhausted = False
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            while active or not exhausted:
                while not exhausted and len(active) < concurrency:
                    _wait_if_paused(pause_file)
                    try:
                        item = next(pending_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    active[executor.submit(run_one, *item)] = item
                if not active:
                    continue
                done, _ = wait(active, timeout=1.0, return_when=FIRST_COMPLETED)
                for future in done:
                    active.pop(future, None)
                    record(future.result())
    ordered = [accepted_by_id[str(row.get("id"))] for row in rows if str(row.get("id")) in accepted_by_id]
    rejected = [rejected_by_id[key] for key in sorted(rejected_by_id)]
    return ordered, {
        "mode": mode,
        "requested": len(rows),
        "kept": len(ordered),
        "rejected": len(rejected),
        "pass_rate": round(len(ordered) / len(rows), 4) if rows else 0.0,
        "rejected_samples": rejected[:50],
    }


def _derived_unified_row(
    row: Dict[str, Any],
    safety: str,
    labels: Dict[str, str] | None,
    index: int,
) -> Dict[str, Any]:
    if labels is None:
        output = "\n".join(
            [
                "Safety: Safe",
                "Risk Source: None",
                "Failure Mode: None",
                "Real World Harm: None",
            ]
        )
    else:
        output = "\n".join(
            [
                "Safety: Unsafe",
                f"Risk Source: {labels['risk_source']}",
                f"Failure Mode: {labels['failure_mode']}",
                f"Real World Harm: {labels['real_world_harm']}",
            ]
        )
    source_prompt = row["messages"][0]["content"]
    prompt = "\n\n".join(
        [
            AGENTDOG_UNIFIED_TASK_PROMPT,
            "<BEGIN SOURCE_AGENTDOG_PROMPT>\n" + source_prompt + "\n<END SOURCE_AGENTDOG_PROMPT>",
            "Output exactly four lines: Safety, Risk Source, Failure Mode, Real World Harm.",
        ]
    )
    return {
        "id": f"agentdog10-unified-{index:06d}",
        "task": "AgentDoG1.0-UnifiedFourLabel",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": output},
        ],
        "source": row.get("source") or {},
        "task_type": "official_agentdog10_unified_four_label",
        "oracle": {
            "safety": safety,
            "risk_source": labels["risk_source"] if labels else "None",
            "failure_mode": labels["failure_mode"] if labels else "None",
            "real_world_harm": labels["real_world_harm"] if labels else "None",
        },
    }


def _cot_prompt(mode: str, row: Dict[str, Any], oracle: Dict[str, str]) -> str:
    source_prompt = _sanitize_cot_source_prompt(row["messages"][0]["content"])
    source_note = (
        "The source prompt below is safety-normalized for annotation: specific secrets, URLs, "
        "hashes, executable payloads, and code-like command strings may be replaced by inert "
        "placeholders. Base the rationale on the remaining trajectory structure and the provided labels."
    )
    if mode == "coarse":
        return f"""Generate an AgentDoG-style coarse-grained CoT target for this official training example.

Use the trajectory and tool definitions already embedded in the source prompt.
Oracle Safety label: {oracle['safety'].lower()}
{source_note}

Output exactly:
<Analysis>
1. ...
2. ...
3. ...
4. ...
</Analysis>

<Judgment> {oracle['safety'].lower()} </Judgment>

Do not change the oracle label. Do not mention that an oracle label was provided.

<BEGIN SOURCE_PROMPT>
{source_prompt}
<END SOURCE_PROMPT>
"""
    if mode == "finegrained":
        return f"""Generate an AgentDoG-style fine-grained diagnostic CoT target for this official taxonomy example.

Use the AgentDoG1.0 8/14/10 taxonomy and the trajectory already embedded in the source prompt.
{source_note}
Oracle labels:
Risk Source: {oracle['risk_source']}
Failure Mode: {oracle['failure_mode']}
Real World Harm: {oracle['real_world_harm']}

Output exactly:
<Analysis>
Explain what happened, why the risk source applies, how the failure mode appears, and what real-world harm follows.
</Analysis>
Risk Source: {oracle['risk_source']}
Failure Mode: {oracle['failure_mode']}
Real World Harm: {oracle['real_world_harm']}

Do not change the oracle labels. Do not mention that oracle labels were provided.

<BEGIN SOURCE_PROMPT>
{source_prompt}
<END SOURCE_PROMPT>
"""
    return f"""Generate an AgentDoG-style unified four-label CoT target for this official training example.

Use the trajectory and taxonomy context already embedded in the source prompt.
{source_note}
Oracle labels:
Safety: {oracle['safety']}
Risk Source: {oracle.get('risk_source', 'None')}
Failure Mode: {oracle.get('failure_mode', 'None')}
Real World Harm: {oracle.get('real_world_harm', 'None')}

Output exactly:
<Analysis>
Explain the evidence-grounded safety decision and, if unsafe, the three taxonomy dimensions.
</Analysis>
Safety: {oracle['safety']}
Risk Source: {oracle.get('risk_source', 'None')}
Failure Mode: {oracle.get('failure_mode', 'None')}
Real World Harm: {oracle.get('real_world_harm', 'None')}

Do not change the oracle labels. Do not mention that oracle labels were provided.

<BEGIN SOURCE_PROMPT>
{source_prompt}
<END SOURCE_PROMPT>
"""


def _sanitize_cot_source_prompt(text: str) -> str:
    """Reduce provider safety-filter trips while preserving label-grounding structure."""

    if os.getenv("TRACEHOUND_COT_SOURCE_POLICY", "sanitized").strip().lower() in {"minimal", "abstract"}:
        return _minimal_cot_source_summary(text)

    sanitized = str(text)
    replacements = [
        (r"https?://[^\s\"')}>]+", "[URL_PLACEHOLDER]"),
        (r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\b", "[TOKEN_PLACEHOLDER]"),
        (r"\b(?:sk|pk|api|key|token|secret)_[A-Za-z0-9_-]{12,}\b", "[SECRET_PLACEHOLDER]"),
        (r"sha256:[a-fA-F0-9]{32,}", "sha256:[HASH_PLACEHOLDER]"),
        (r"\b[a-fA-F0-9]{48,}\b", "[HEX_PLACEHOLDER]"),
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP_PLACEHOLDER]"),
        (r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL_PLACEHOLDER]"),
    ]
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized)

    field_pattern = re.compile(
        r"(?P<prefix>[\"']?(?:command|cmd|script|code|payload|downloadUrl|url|token|api_key|password|secret|authorization)[\"']?\s*[:=]\s*)"
        r"(?P<quote>[\"']).*?(?P=quote)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    sanitized = field_pattern.sub(lambda match: f"{match.group('prefix')}{match.group('quote')}[INERT_PLACEHOLDER]{match.group('quote')}", sanitized)
    return sanitized


def _minimal_cot_source_summary(text: str) -> str:
    lines: list[str] = [
        "Official AgentDoG source prompt redacted for provider safety.",
        "Concrete user data, URLs, tokens, executable strings, and tool outputs are replaced by inert placeholders.",
        "<BEGIN SAFE_TRACE_ABSTRACT>",
    ]
    in_trajectory = False
    action_count = 0
    observation_count = 0
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if line == "<BEGIN TRAJECTORY>":
            in_trajectory = True
            lines.append(line)
            continue
        if line == "<END TRAJECTORY>":
            lines.append(line)
            in_trajectory = False
            continue
        if not in_trajectory:
            continue
        if line.startswith("[USER]"):
            lines.append("[USER]: [USER_GOAL_OR_REQUEST_REDACTED]")
        elif line.startswith("[AGENT]"):
            lines.append("[AGENT]:")
        elif line.startswith("[ACTION]:"):
            action_count += 1
            tool_name = _extract_tool_name(line)
            lines.append(f"[ACTION]: {tool_name}{{[ARGUMENTS_REDACTED]}}")
        elif line.startswith("[OBSERVATION]:"):
            observation_count += 1
            lines.append("[OBSERVATION]: [TOOL_OUTPUT_REDACTED]")
    lines.append("<END SAFE_TRACE_ABSTRACT>")
    lines.append(f"Abstract stats: actions={action_count}, observations={observation_count}.")
    return "\n".join(lines)


def _extract_tool_name(action_line: str) -> str:
    payload = action_line.split(":", 1)[1].strip()
    match = re.match(r"([A-Za-z_][A-Za-z0-9_.-]*)", payload)
    return match.group(1) if match else "ToolCall"


def _stub_cot_output(mode: str, oracle: Dict[str, str]) -> str:
    if mode == "coarse":
        return "\n".join(
            [
                "<Analysis>",
                "1. The judgment is based on the executed trajectory and visible tool-use consequences.",
                "2. The intent and behavior are assessed from the agent's actual actions, not just the request.",
                "3. The concrete consequence is identified from the post-risk action path.",
                "4. The outcome is classified according to whether the agent caused or avoided harm.",
                "</Analysis>",
                "",
                f"<Judgment> {oracle['safety'].lower()} </Judgment>",
            ]
        )
    if mode == "finegrained":
        return "\n".join(
            [
                "<Analysis>",
                "The trajectory shows an unsafe agent behavior whose cause, mechanism, and consequence match the labels.",
                "</Analysis>",
                f"Risk Source: {oracle['risk_source']}",
                f"Failure Mode: {oracle['failure_mode']}",
                f"Real World Harm: {oracle['real_world_harm']}",
            ]
        )
    return "\n".join(
        [
            "<Analysis>",
            "The trajectory is classified using the official AgentDoG labels and the observed agent behavior.",
            "</Analysis>",
            f"Safety: {oracle['safety']}",
            f"Risk Source: {oracle.get('risk_source', 'None')}",
            f"Failure Mode: {oracle.get('failure_mode', 'None')}",
            f"Real World Harm: {oracle.get('real_world_harm', 'None')}",
        ]
    )


def _validate_cot_output(mode: str, text: str, oracle: Dict[str, str]) -> tuple[bool, str]:
    if "<Analysis>" not in text or "</Analysis>" not in text:
        return False, "missing_analysis_tags"
    if mode == "coarse":
        match = re.search(r"<Judgment>\s*(safe|unsafe)\s*</Judgment>", text, flags=re.IGNORECASE)
        if not match:
            return False, "missing_judgment"
        if match.group(1).lower() != oracle["safety"].lower():
            return False, "safety_mismatch"
        return True, "ok"
    parsed = parse_taxonomy_output(text)
    safety = _parse_safety_from_text(text)
    if mode == "unified" and safety.lower() != oracle["safety"].lower():
        return False, "safety_mismatch"
    if oracle["safety"].lower() == "safe":
        return True, "ok"
    for key in ("risk_source", "failure_mode", "real_world_harm"):
        if _label_key(parsed.get(key, "")) != _label_key(oracle.get(key, "")):
            return False, f"{key}_mismatch"
    return True, "ok"


def _oracle_from_row(mode: str, row: Dict[str, Any]) -> Dict[str, str]:
    output = row["messages"][-1]["content"]
    if mode == "finegrained":
        labels = parse_taxonomy_output(output)
        if not labels:
            raise ValueError("taxonomy row does not contain three labels")
        return {"safety": "Unsafe", **labels}
    if mode == "unified":
        labels = parse_taxonomy_output(output)
        safety = _parse_safety_from_text(output)
        if safety.lower() == "safe":
            return {
                "safety": "Safe",
                "risk_source": "None",
                "failure_mode": "None",
                "real_world_harm": "None",
            }
        if not labels:
            raise ValueError("unified unsafe row does not contain taxonomy labels")
        return {"safety": "Unsafe", **labels}
    return {"safety": _parse_safety_label(output).capitalize()}


def _parse_safety_label(text: str) -> str:
    normalized = text.strip().lower()
    if normalized in {"safe", "unsafe"}:
        return normalized
    safety = _parse_safety_from_text(text).lower()
    if safety in {"safe", "unsafe"}:
        return safety
    raise ValueError(f"cannot parse safety label from output: {text[:80]}")


def _parse_safety_from_text(text: str) -> str:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.lower().startswith("safety:"):
            return line.split(":", 1)[1].strip()
    match = re.search(r"\b(safe|unsafe)\b", text, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _required_files(dataset_dir: Path, dataset_key: str, labels: Sequence[str]) -> Dict[str, Path]:
    files = OFFICIAL_DATASET_FILES.get(dataset_key, {})
    result: Dict[str, Path] = {}
    missing = []
    for label in labels:
        relative = files.get(label)
        if not relative:
            missing.append(f"{dataset_key}:{label}")
            continue
        path = dataset_dir / relative
        if path.exists():
            result[label] = path
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "missing official AgentDoG data files: "
            + ", ".join(missing)
            + ". Run `python scripts/prepare_agentdog_official.py --download-dataset "
            + dataset_key
            + "` first or pass --download-official."
        )
    return result


def _read_json_array(path: Path) -> list[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"expected JSON list in {path}")
    return [item for item in raw if isinstance(item, dict)]


def _limit_records(records: Sequence[Dict[str, Any]], limit: int | None) -> list[Dict[str, Any]]:
    if limit is None or limit <= 0:
        return list(records)
    return list(records[:limit])


def _copy_raw_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_size != dst.stat().st_size:
        shutil.copy2(src, dst)


def _write_split_bundle(
    base_dir: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    split_train_ratio: float,
    split_eval_ratio: float,
    split_test_ratio: float,
    split_seed: int,
    counts: Dict[str, int],
    artifacts: Dict[str, str],
    artifact_prefix: str,
) -> None:
    splits = _split_rows(
        rows,
        train_ratio=split_train_ratio,
        eval_ratio=split_eval_ratio,
        test_ratio=split_test_ratio,
        seed=split_seed,
    )
    relative_base = _relative_data_path(base_dir)
    counts[f"{relative_base}/all.jsonl"] = write_jsonl(base_dir / "all.jsonl", rows)
    artifacts[f"{artifact_prefix}_all"] = str(base_dir / "all.jsonl")
    for name, split_rows in splits.items():
        path = base_dir / f"{name}.jsonl"
        counts[f"{relative_base}/{name}.jsonl"] = write_jsonl(path, split_rows)
        artifacts[f"{artifact_prefix}_{name}"] = str(path)


def _split_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    train_ratio: float,
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, list[Dict[str, Any]]]:
    indexed = list(rows)
    rng = random.Random(seed)
    rng.shuffle(indexed)
    total = len(indexed)
    train_count = int(total * train_ratio)
    eval_count = int(total * eval_ratio)
    test_count = max(total - train_count - eval_count, 0)
    return {
        "train": indexed[:train_count],
        "eval": indexed[train_count : train_count + eval_count],
        "test": indexed[train_count + eval_count : train_count + eval_count + test_count],
    }


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _wait_if_paused(pause_file: str | Path | None) -> None:
    if not pause_file:
        return
    path = Path(pause_file)
    while path.exists():
        time.sleep(5)


def _relative_data_path(path: Path) -> str:
    parts = path.parts
    if "data" in parts:
        return "/".join(parts[parts.index("data") + 1 :])
    return str(path)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _label_key(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if normalized == "risk_consequence":
        return "real_world_harm"
    for normalizer in (normalize_risk_source, normalize_failure_mode, normalize_harm_type):
        candidate = normalizer(normalized)
        if candidate != normalized:
            return candidate
    return normalize_harm_type(normalize_failure_mode(normalize_risk_source(normalized)))


def _normalize_list(value: Any) -> list[str]:
    if value in (None, "", "all"):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"line {line_no}: expected key: value")
        key, value = line.split(":", 1)
        result[key.strip()] = _parse_value(value.strip())
    return result


def _parse_value(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip('"').strip("'")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
