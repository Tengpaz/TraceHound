"""Official AgentDoG release metadata and prompt helpers.

The public AgentDoG repository releases prompts, model links, datasets, SFT/RL
recipes, and a runtime environment server. It does not release the original
LLM data synthesis engine, so official reproduction in TraceHound means using
the released official assets rather than treating the local synthetic generator
as the official generator.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict


OFFICIAL_REPO_URL = "https://github.com/AI45Lab/AgentDoG.git"
OFFICIAL_DATASETS = {
    "atbench": "AI45Research/ATBench",
    "atbench_claw": "AI45Research/ATBench-Claw",
    "atbench_codex": "AI45Research/ATBench-Codex",
    "app1_sft": "AI45Research/APP1-Agentic-Safety-SFT-Data",
    "rl_runtime": "quantumfr/agentic-lightweight-envs-runtime-20260528",
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
            }
    return {
        "official_repo_url": OFFICIAL_REPO_URL,
        "official_repo_commit": repo_commit,
        "official_datasets": OFFICIAL_DATASETS,
        "public_release_limitations": PUBLIC_RELEASE_LIMITATIONS,
        "local_prompt_dir": str(OFFICIAL_PROMPT_DIR),
        "data_presence": data_presence,
        "reproduction_mode": "official_assets",
        "synthetic_generator_status": "surrogate_not_official",
    }


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_revision(repo_dir: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
