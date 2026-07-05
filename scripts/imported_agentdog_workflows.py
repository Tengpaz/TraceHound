#!/usr/bin/env python
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SFT_ROOT = REPO_ROOT / "external" / "agentdog_sft_project"
SAFETY_REPO_ROOT = REPO_ROOT / "external" / "agentdog_safety_repo"
IMPORTED_6LABEL_MODEL = REPO_ROOT / "checkpoints" / "imported" / "qwen35-0.8b-6label"


@dataclass(frozen=True)
class Workflow:
    name: str
    cwd: Path
    command: tuple[str, ...]
    description: str


WORKFLOWS: dict[str, Workflow] = {
    "prepare-binary-sft": Workflow(
        name="prepare-binary-sft",
        cwd=SFT_ROOT,
        command=(
            sys.executable,
            "scripts/prepare_sft_data.py",
            "--task",
            "binary",
            "--output-dir",
            "outputs/data",
        ),
        description="Build imported AgentDoG binary SFT JSONL data.",
    ),
    "prepare-taxonomy-sft": Workflow(
        name="prepare-taxonomy-sft",
        cwd=SFT_ROOT,
        command=(
            sys.executable,
            "scripts/prepare_sft_data.py",
            "--task",
            "taxonomy",
            "--output-dir",
            "outputs/data",
        ),
        description="Build imported AgentDoG taxonomy SFT JSONL data.",
    ),
    "prepare-6label-sft": Workflow(
        name="prepare-6label-sft",
        cwd=SFT_ROOT,
        command=(
            sys.executable,
            "scripts/prepare_6label_sft_data.py",
            "--input",
            "data/agentdog_complete_binary_safe_augmented_unsafe_train.jsonl",
            "--output",
            "outputs/data/agentdog_6label_sft.jsonl",
            "--template",
            "src/guardrail/6 labels training prompt.md",
        ),
        description="Build six-label + reason SFT data from the augmented AgentDoG file.",
    ),
    "train-6label-sft": Workflow(
        name="train-6label-sft",
        cwd=SFT_ROOT,
        command=(
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/train_6label_sft.py",
            "--config",
            "configs/training_6label_defaults.json",
        ),
        description="Run four-GPU full-parameter six-label SFT.",
    ),
    "train-6label-grpo": Workflow(
        name="train-6label-grpo",
        cwd=SFT_ROOT,
        command=(
            "torchrun",
            "--standalone",
            "--nproc_per_node=4",
            "scripts/train_6label_grpo.py",
            "--config",
            "configs/grpo_6label_defaults.json",
        ),
        description="Run GRPO optimization for the six-label + reason task.",
    ),
    "eval-6label-atbench": Workflow(
        name="eval-6label-atbench",
        cwd=SFT_ROOT,
        command=(
            sys.executable,
            "scripts/binary_safety_eval.py",
            "--model-path",
            str(IMPORTED_6LABEL_MODEL),
            "--input-json",
            "2026_summer_camp_teseset/summer_camp_ATBench300.json",
            "--output-dir",
            str(REPO_ROOT / "outputs" / "imported" / "qwen35-0.8b-6label-atbench300"),
            "--max-new-tokens",
            "512",
        ),
        description="Evaluate the imported six-label checkpoint on summer-camp ATBench300.",
    ),
    "eval-6label-rjudge": Workflow(
        name="eval-6label-rjudge",
        cwd=SFT_ROOT,
        command=(
            sys.executable,
            "scripts/binary_safety_eval.py",
            "--model-path",
            str(IMPORTED_6LABEL_MODEL),
            "--input-json",
            "2026_summer_camp_teseset/summer_camp_rjudge.json",
            "--output-dir",
            str(REPO_ROOT / "outputs" / "imported" / "qwen35-0.8b-6label-rjudge"),
            "--max-new-tokens",
            "512",
        ),
        description="Evaluate the imported six-label checkpoint on summer-camp R-Judge.",
    ),
    "train-binary-lr8e6": Workflow(
        name="train-binary-lr8e6",
        cwd=SAFETY_REPO_ROOT,
        command=("bash", "scripts/train/run_binary_lr8e6_steps330_train_eval.sh"),
        description="Run the imported binary Qwen3.5-0.8B SFT recipe with lr=8e-6 and max_steps=330.",
    ),
    "train-taxonomy-lora": Workflow(
        name="train-taxonomy-lora",
        cwd=SAFETY_REPO_ROOT,
        command=("bash", "scripts/train/run_taxonomy_lora_200.sh"),
        description="Run the imported taxonomy LoRA recipe with lr=8e-6 and max_steps=200.",
    ),
}


def format_command(workflow: Workflow, extra: list[str]) -> str:
    command = [*workflow.command, *extra]
    return " ".join(shlex.quote(part) for part in command)


def list_workflows() -> None:
    print("Imported AgentDoG workflows:\n")
    for workflow in WORKFLOWS.values():
        print(f"- {workflow.name}: {workflow.description}")
        print(f"  cwd: {workflow.cwd.relative_to(REPO_ROOT)}")
        print(f"  cmd: {format_command(workflow, [])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or inspect imported AgentDoG workflows.")
    parser.add_argument("workflow", nargs="?", choices=["list", *WORKFLOWS.keys()], default="list")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments appended after '--'.")
    args = parser.parse_args()

    if args.workflow == "list":
        list_workflows()
        return 0

    workflow = WORKFLOWS[args.workflow]
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    command = [*workflow.command, *extra]
    if not workflow.cwd.exists():
        raise SystemExit(f"Missing workflow directory: {workflow.cwd}")

    print(f"cwd: {workflow.cwd}")
    print(f"cmd: {format_command(workflow, extra)}")
    if args.dry_run:
        return 0
    completed = subprocess.run(command, cwd=workflow.cwd, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
