#!/usr/bin/env python
"""Prepare official AgentDoG release assets for reproduction.

This script uses the public AgentDoG release path: official repository,
official prompts, official Hugging Face datasets, and official SFT/RL recipes.
It does not claim to reproduce the unreleased LLM data synthesis engine.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.agentdog_official import (
    OFFICIAL_DATASETS,
    OFFICIAL_REPO_URL,
    download_official_dataset,
    official_manifest,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="external/agentdog_official", help="Local official asset root.")
    parser.add_argument("--clone-repo", action="store_true", help="Clone or update the official AgentDoG GitHub repository.")
    parser.add_argument(
        "--download-dataset",
        action="append",
        choices=sorted(OFFICIAL_DATASETS),
        help="Download one official dataset by key. Repeatable.",
    )
    parser.add_argument("--download-all", action="store_true", help="Download all listed official datasets.")
    parser.add_argument("--manifest", default="reports/agentdog_official_manifest.json", help="Manifest output path.")
    parser.add_argument("--force", action="store_true", help="Replace existing official repo/dataset directories.")
    args = parser.parse_args()

    root = Path(args.root)
    repo_dir = root / "AgentDoG"
    data_root = root / "datasets"
    root.mkdir(parents=True, exist_ok=True)

    actions: list[dict[str, Any]] = []
    if args.clone_repo:
        actions.append(_prepare_repo(repo_dir, force=args.force))

    dataset_keys = sorted(OFFICIAL_DATASETS) if args.download_all else sorted(set(args.download_dataset or []))
    for key in dataset_keys:
        actions.append(_download_dataset(key, data_root / key, force=args.force))

    manifest = official_manifest(repo_dir=repo_dir, data_root=data_root)
    manifest["actions"] = actions
    write_manifest(Path(args.manifest), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


def _prepare_repo(repo_dir: Path, *, force: bool) -> dict[str, Any]:
    if repo_dir.exists() and force:
        shutil.rmtree(repo_dir)
    if repo_dir.exists():
        if (repo_dir / ".git").exists():
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
            status = "updated"
        else:
            raise SystemExit(f"{repo_dir} exists but is not a git repo; use --force to replace it")
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", OFFICIAL_REPO_URL, str(repo_dir)], check=True)
        status = "cloned"
    return {"kind": "official_repo", "status": status, "path": str(repo_dir)}


def _download_dataset(key: str, output_dir: Path, *, force: bool) -> dict[str, Any]:
    try:
        return download_official_dataset(key, output_dir, force=force)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
