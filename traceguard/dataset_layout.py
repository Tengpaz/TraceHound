"""Dataset bundle writer for clean TraceHound generation outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from traceguard.data import dataset_summary
from traceguard.dataset_ops import coverage_matrix, split_cases, split_summary
from traceguard.export import (
    agentdog15_coarse_sft_rows,
    agentdog15_unified_sft_rows,
    agentdog_binary_sft_rows,
    agentdog_taxonomy_sft_rows,
    agentdog_unified_sft_rows,
    eval_rows,
    preference_rows,
    rl_rows,
    sft_rows,
    write_jsonl,
)

RowsFactory = Callable[[Iterable[Dict[str, Any]]], Iterable[Dict[str, Any]]]


def write_dataset_bundle(
    out_dir: Path,
    cases: Sequence[Dict[str, Any]],
    training_cases: Sequence[Dict[str, Any]],
    *,
    include_eval: bool = True,
    include_sft: bool = True,
    include_agentdog_sft: bool = True,
    include_agentdog15_official_sft: bool = True,
    include_preference: bool = True,
    include_rl: bool = False,
    write_clean_layout: bool = True,
    write_legacy_flat_files: bool = True,
    split_train_ratio: float = 0.8,
    split_eval_ratio: float = 0.1,
    split_test_ratio: float = 0.1,
    split_seed: int = 20260701,
    manifest_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Write generated cases using clear directories plus optional legacy copies."""

    out_dir.mkdir(parents=True, exist_ok=True)
    training_splits = split_cases(
        list(training_cases),
        train_ratio=split_train_ratio,
        eval_ratio=split_eval_ratio,
        test_ratio=split_test_ratio,
        seed=split_seed,
    )
    split_info = split_summary(
        training_splits,
        train_ratio=split_train_ratio,
        eval_ratio=split_eval_ratio,
        test_ratio=split_test_ratio,
        seed=split_seed,
    )
    coverage = {
        "all_cases": coverage_matrix(list(cases)),
        "training_cases": coverage_matrix(list(training_cases)),
        "splits": {
            name: coverage_matrix(rows)
            for name, rows in training_splits.items()
        },
    }
    counts: Dict[str, int] = {}
    artifacts: Dict[str, str] = {"output_dir": str(out_dir)}

    if write_clean_layout:
        _write_clean_bundle(
            out_dir,
            cases,
            training_cases,
            training_splits,
            counts=counts,
            artifacts=artifacts,
            include_eval=include_eval,
            include_sft=include_sft,
            include_agentdog_sft=include_agentdog_sft,
            include_agentdog15_official_sft=include_agentdog15_official_sft,
            include_preference=include_preference,
            include_rl=include_rl,
        )

    if write_legacy_flat_files:
        _write_legacy_bundle(
            out_dir,
            cases,
            training_cases,
            counts=counts,
            artifacts=artifacts,
            include_eval=include_eval,
            include_sft=include_sft,
            include_agentdog_sft=include_agentdog_sft,
            include_agentdog15_official_sft=include_agentdog15_official_sft,
            include_preference=include_preference,
            include_rl=include_rl,
        )

    manifest = {
        "schema_version": "tracehound.dataset_bundle.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": dataset_summary(list(cases)),
        "training_summary": dataset_summary(list(training_cases)),
        "splits": split_info,
        "counts": counts,
        "artifacts": artifacts,
        "layout": {
            "clean_layout": bool(write_clean_layout),
            "legacy_flat_files": bool(write_legacy_flat_files),
            "cases_dir": "cases/",
            "training_dir": "train/",
            "metadata_dir": "metadata/",
            "rejected_dir": "rejected/",
        },
    }
    if manifest_extra:
        manifest["extra"] = dict(manifest_extra)

    _write_json(out_dir / "metadata" / "coverage_matrix.json", coverage)
    _write_json(out_dir / "metadata" / "dataset_manifest.json", manifest)
    artifacts["coverage_matrix"] = str(out_dir / "metadata" / "coverage_matrix.json")
    artifacts["dataset_manifest"] = str(out_dir / "metadata" / "dataset_manifest.json")
    counts["metadata/coverage_matrix.json"] = 1
    counts["metadata/dataset_manifest.json"] = 1

    return {
        "counts": counts,
        "artifacts": artifacts,
        "coverage": coverage,
        "splits": split_info,
        "manifest": manifest,
    }


def _write_clean_bundle(
    out_dir: Path,
    cases: Sequence[Dict[str, Any]],
    training_cases: Sequence[Dict[str, Any]],
    training_splits: Mapping[str, Sequence[Dict[str, Any]]],
    *,
    counts: Dict[str, int],
    artifacts: Dict[str, str],
    include_eval: bool,
    include_sft: bool,
    include_agentdog_sft: bool,
    include_agentdog15_official_sft: bool,
    include_preference: bool,
    include_rl: bool,
) -> None:
    if include_eval:
        _write_rows(out_dir, "cases/all.jsonl", eval_rows(cases), counts, artifacts, key="cases_all")
        for split_name, split_cases_ in training_splits.items():
            _write_rows(
                out_dir,
                f"cases/{split_name}.jsonl",
                eval_rows(split_cases_),
                counts,
                artifacts,
                key=f"cases_{split_name}",
            )
    if include_sft:
        _write_task_split(
            out_dir,
            "train/tracehound_risk_report_sft",
            sft_rows,
            training_cases,
            training_splits,
            counts,
            artifacts,
        )
    if include_agentdog_sft:
        _write_task_split(
            out_dir,
            "train/agentdog/binary_safety",
            agentdog_binary_sft_rows,
            training_cases,
            training_splits,
            counts,
            artifacts,
        )
        _write_task_split(
            out_dir,
            "train/agentdog/taxonomy_only",
            agentdog_taxonomy_sft_rows,
            training_cases,
            training_splits,
            counts,
            artifacts,
        )
        _write_task_split(
            out_dir,
            "train/agentdog/unified_four_label",
            agentdog_unified_sft_rows,
            training_cases,
            training_splits,
            counts,
            artifacts,
        )
    if include_agentdog15_official_sft:
        _write_task_split(
            out_dir,
            "train/agentdog15/unified",
            agentdog15_unified_sft_rows,
            training_cases,
            training_splits,
            counts,
            artifacts,
        )
        _write_task_split(
            out_dir,
            "train/agentdog15/coarse",
            agentdog15_coarse_sft_rows,
            training_cases,
            training_splits,
            counts,
            artifacts,
        )
    if include_preference:
        _write_task_split(
            out_dir,
            "train/preference/dpo_pairs",
            preference_rows,
            training_cases,
            training_splits,
            counts,
            artifacts,
        )
    if include_rl:
        _write_task_split(
            out_dir,
            "train/rl/rl_pairs",
            rl_rows,
            training_cases,
            training_splits,
            counts,
            artifacts,
        )


def _write_legacy_bundle(
    out_dir: Path,
    cases: Sequence[Dict[str, Any]],
    training_cases: Sequence[Dict[str, Any]],
    *,
    counts: Dict[str, int],
    artifacts: Dict[str, str],
    include_eval: bool,
    include_sft: bool,
    include_agentdog_sft: bool,
    include_agentdog15_official_sft: bool,
    include_preference: bool,
    include_rl: bool,
) -> None:
    if include_eval:
        _write_rows(out_dir, "synthetic_eval.jsonl", eval_rows(cases), counts, artifacts, key="eval")
    if include_sft:
        _write_rows(out_dir, "synthetic_sft.jsonl", sft_rows(training_cases), counts, artifacts, key="sft")
    if include_agentdog_sft:
        _write_rows(
            out_dir,
            "agentdog_binary_sft.jsonl",
            agentdog_binary_sft_rows(training_cases),
            counts,
            artifacts,
            key="agentdog_binary_sft",
        )
        _write_rows(
            out_dir,
            "agentdog_taxonomy_sft.jsonl",
            agentdog_taxonomy_sft_rows(training_cases),
            counts,
            artifacts,
            key="agentdog_taxonomy_sft",
        )
        _write_rows(
            out_dir,
            "agentdog_unified_sft.jsonl",
            agentdog_unified_sft_rows(training_cases),
            counts,
            artifacts,
            key="agentdog_unified_sft",
        )
    if include_agentdog15_official_sft:
        _write_rows(
            out_dir,
            "agentdog15_unified_sft.jsonl",
            agentdog15_unified_sft_rows(training_cases),
            counts,
            artifacts,
            key="agentdog15_unified_sft",
        )
        _write_rows(
            out_dir,
            "agentdog15_coarse_sft.jsonl",
            agentdog15_coarse_sft_rows(training_cases),
            counts,
            artifacts,
            key="agentdog15_coarse_sft",
        )
    if include_preference:
        _write_rows(
            out_dir,
            "synthetic_preference.jsonl",
            preference_rows(training_cases),
            counts,
            artifacts,
            key="preference",
        )
    if include_rl:
        _write_rows(out_dir, "synthetic_rl.jsonl", rl_rows(training_cases), counts, artifacts, key="rl")


def _write_task_split(
    out_dir: Path,
    base_dir: str,
    rows_factory: RowsFactory,
    all_cases: Sequence[Dict[str, Any]],
    splits: Mapping[str, Sequence[Dict[str, Any]]],
    counts: Dict[str, int],
    artifacts: Dict[str, str],
) -> None:
    artifact_prefix = base_dir.replace("/", "_")
    _write_rows(out_dir, f"{base_dir}/all.jsonl", rows_factory(all_cases), counts, artifacts, key=f"{artifact_prefix}_all")
    for split_name, split_cases_ in splits.items():
        _write_rows(
            out_dir,
            f"{base_dir}/{split_name}.jsonl",
            rows_factory(split_cases_),
            counts,
            artifacts,
            key=f"{artifact_prefix}_{split_name}",
        )


def _write_rows(
    out_dir: Path,
    relative_path: str,
    rows: Iterable[Dict[str, Any]],
    counts: Dict[str, int],
    artifacts: Dict[str, str],
    *,
    key: str,
) -> None:
    path = out_dir / relative_path
    count = write_jsonl(path, rows)
    counts[relative_path] = count
    artifacts[key] = str(path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
