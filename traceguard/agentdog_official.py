"""Official AgentDoG release metadata and prompt helpers.

The public AgentDoG repository releases prompts, model links, datasets, SFT/RL
recipes, and a runtime environment server. It does not release the original
LLM data synthesis engine, so official reproduction in TraceHound means using
the released official assets rather than treating the local synthetic generator
as the official generator.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict
from urllib.parse import quote
from urllib.request import urlopen


OFFICIAL_REPO_URL = "https://github.com/AI45Lab/AgentDoG.git"
OFFICIAL_DATASETS = {
    "agentdog10_training": "AI45Research/AgentDoG1.0-Training-Data",
    "agentdog_lite_summer_camp_test": "AI45Research/2026_summer_camp_teseset",
    "atbench": "AI45Research/ATBench",
    "atbench_claw": "AI45Research/ATBench-Claw",
    "atbench_codex": "AI45Research/ATBench-Codex",
    "app1_sft": "AI45Research/APP1-Agentic-Safety-SFT-Data",
    "rl_runtime": "quantumfr/agentic-lightweight-envs-runtime-20260528",
}
OFFICIAL_DATASET_FILES = {
    "agentdog10_training": {
        "binary_safety": "AgentDoG-BinarySafety/train.json",
        "finegrained_taxonomy": "AgentDoG-FineGrainedTaxonomy/train.json",
        "meta": "meta.json",
    },
    "agentdog_lite_summer_camp_test": {
        "atbench300": "summer_camp_ATBench300.json",
        "rjudge": "summer_camp_rjudge.json",
        "readme": "README.md",
    },
    "app1_sft": {
        "safety_response_sft": "agentic_safety_sft.json",
    },
    "atbench": {
        "latest_test": "ATBench/test.json",
        "original_500_test": "ATBench500/test.json",
        "legacy_test": "test.json",
    },
    "atbench_claw": {
        "test": "test.json",
    },
    "atbench_codex": {
        "test": "test.json",
    },
}
OFFICIAL_PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts" / "agentdog"
PUBLIC_RELEASE_LIMITATIONS = {
    "llm_synthesis_engine_public": False,
    "prompt_generation_code_public": False,
    "official_training_recipes_public": True,
    "official_prompts_public": True,
    "official_datasets_public": True,
}


def load_official_prompt(version: str, name: str) -> str:
    path = OFFICIAL_PROMPT_DIR / version / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"official AgentDoG prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def format_agentdog15_unified_prompt(formatted_trajectory: str) -> str:
    template = load_official_prompt("v1.5", "unified_safety_classification")
    return template.format(formatted_trajectory=formatted_trajectory)


def format_agentdog15_coarse_prompt(formatted_trajectory: str, tool_list_text: str = "") -> str:
    template = load_official_prompt("v1.5", "coarse_grained_moderation")
    return template.format(formatted_trajectory=formatted_trajectory, tool_list_text=tool_list_text)


def official_manifest(repo_dir: Path | None = None, data_root: Path | None = None) -> Dict[str, Any]:
    repo_commit = None
    if repo_dir and (repo_dir / ".git").exists():
        repo_commit = _git_revision(repo_dir)
    data_presence = {}
    if data_root:
        for name, dataset_id in OFFICIAL_DATASETS.items():
            expected = data_root / name
            data_presence[name] = {
                "dataset_id": dataset_id,
                "path": str(expected),
                "present": expected.exists(),
                "expected_files": OFFICIAL_DATASET_FILES.get(name, {}),
                "files_present": {
                    label: (expected / relative).exists()
                    for label, relative in OFFICIAL_DATASET_FILES.get(name, {}).items()
                },
            }
    return {
        "official_repo_url": OFFICIAL_REPO_URL,
        "official_repo_commit": repo_commit,
        "official_datasets": OFFICIAL_DATASETS,
        "official_dataset_files": OFFICIAL_DATASET_FILES,
        "public_release_limitations": PUBLIC_RELEASE_LIMITATIONS,
        "local_prompt_dir": str(OFFICIAL_PROMPT_DIR),
        "data_presence": data_presence,
        "reproduction_mode": "official_assets",
        "synthetic_generator_status": "surrogate_not_official",
    }


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def download_official_dataset(key: str, output_dir: Path, *, force: bool = False) -> Dict[str, Any]:
    """Download one public AgentDoG dataset snapshot using huggingface_hub."""

    if key not in OFFICIAL_DATASETS:
        raise KeyError(f"unknown official AgentDoG dataset key: {key}")
    dataset_id = OFFICIAL_DATASETS[key]
    if output_dir.exists() and force:
        shutil.rmtree(output_dir)
    expected_files = OFFICIAL_DATASET_FILES.get(key, {})
    expected_complete = bool(expected_files) and all((output_dir / relative).exists() for relative in expected_files.values())
    if output_dir.exists() and any(output_dir.iterdir()) and (not expected_files or expected_complete):
        return {
            "kind": "dataset",
            "key": key,
            "dataset_id": dataset_id,
            "status": "present",
            "path": str(output_dir),
            "expected_files": OFFICIAL_DATASET_FILES.get(key, {}),
        }
    if expected_files and os.getenv("TRACEHOUND_HF_SNAPSHOT", "0") != "1":
        return _download_known_dataset_files(key, output_dir)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for official dataset download. "
            "Install it with `python -m pip install -e \".[official]\"`."
        ) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=dataset_id, repo_type="dataset", local_dir=str(output_dir))
    return {
        "kind": "dataset",
        "key": key,
        "dataset_id": dataset_id,
        "status": "downloaded",
        "path": str(output_dir),
        "expected_files": OFFICIAL_DATASET_FILES.get(key, {}),
    }


def _download_known_dataset_files(key: str, output_dir: Path) -> Dict[str, Any]:
    dataset_id = OFFICIAL_DATASETS[key]
    files = OFFICIAL_DATASET_FILES[key]
    base_url = os.getenv("TRACEHOUND_HF_BASE_URL", "https://huggingface.co").rstrip("/")
    downloaded: Dict[str, str] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, relative in files.items():
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"{base_url}/datasets/{dataset_id}/resolve/main/{quote(relative)}"
        tmp = destination.with_suffix(destination.suffix + ".tmp")
        with urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        tmp.replace(destination)
        downloaded[label] = str(destination)
    return {
        "kind": "dataset",
        "key": key,
        "dataset_id": dataset_id,
        "status": "downloaded_known_files",
        "path": str(output_dir),
        "expected_files": files,
        "downloaded_files": downloaded,
        "snapshot_note": "Set TRACEHOUND_HF_SNAPSHOT=1 to use huggingface_hub.snapshot_download instead.",
    }


def _git_revision(repo_dir: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
