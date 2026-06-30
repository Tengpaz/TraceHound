"""Safety enchantment planning for policy/base models.

The Guard Model training path teaches a guard to judge trajectories. Safety
enchantment uses the current guard as a filter, judge, or reward model while
fine-tuning another target model to behave more safely.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, Dict

from traceguard.model_profiles import profile_model_id


ENCHANTMENT_ALGORITHMS = {
    "sft": {
        "label": "Guard-filtered SFT",
        "required": ("agentdog_binary_sft.jsonl",),
        "optional": ("synthetic_sft.jsonl",),
    },
    "sft_dpo": {
        "label": "Guard-filtered SFT + DPO",
        "required": ("agentdog_binary_sft.jsonl", "synthetic_preference.jsonl"),
        "optional": ("synthetic_sft.jsonl",),
    },
    "sft_grpo": {
        "label": "Guard-filtered SFT + GRPO",
        "required": ("agentdog_binary_sft.jsonl", "synthetic_rl.jsonl"),
        "optional": ("synthetic_sft.jsonl", "synthetic_preference.jsonl"),
    },
}

DEFAULT_REWARD_FORMULA = {
    "normal_benign": "U",
    "attacked_benign": "0.5 * U + 0.25 * S + 0.25 * U * S",
    "malicious": "S",
    "variables": {
        "U": "task utility or environment success score",
        "S": "current Guard Model safety score or binary safe/unsafe judgment",
    },
}


def resolve_enchantment_data(data_dir: str | Path, algorithm: str) -> Dict[str, Any]:
    if algorithm not in ENCHANTMENT_ALGORITHMS:
        raise ValueError(f"unsupported safety enchantment algorithm: {algorithm}")
    resolved = _resolve_data_dir(data_dir)
    files = {
        "agentdog_binary_sft": resolved / "agentdog_binary_sft.jsonl",
        "tracehound_sft": resolved / "synthetic_sft.jsonl",
        "preference": resolved / "synthetic_preference.jsonl",
        "rl": resolved / "synthetic_rl.jsonl",
        "eval": resolved / "synthetic_eval.jsonl",
        "quality_report": resolved / "quality_report.json",
    }
    required = ENCHANTMENT_ALGORITHMS[algorithm]["required"]
    missing = [name for name in required if not (resolved / name).exists()]
    return {
        "data_dir": str(resolved),
        "algorithm": algorithm,
        "required": list(required),
        "missing": missing,
        "files": {key: str(path) for key, path in files.items() if path.exists()},
        "stats": {key: _jsonl_stats(path) for key, path in files.items() if path.suffix == ".jsonl" and path.exists()},
    }


def build_enchantment_plan(
    *,
    target_profile: Dict[str, Any],
    target_base_model: str,
    guard_model: str,
    guard_mode: str,
    data_dir: str | Path,
    algorithm: str,
    output_dir: str | Path,
    max_samples: int | None = None,
    safety_weight: float = 0.5,
    utility_weight: float = 0.5,
    auto_register: bool = True,
) -> Dict[str, Any]:
    data = resolve_enchantment_data(data_dir, algorithm)
    target_model = target_base_model or profile_model_id(target_profile)
    return {
        "task": "safety_enchantment",
        "algorithm": algorithm,
        "algorithm_label": ENCHANTMENT_ALGORITHMS[algorithm]["label"],
        "guard": {
            "model": guard_model,
            "mode": guard_mode,
            "roles": ["data_filter", "semantic_judge", "safety_reward"],
        },
        "target": {
            "profile": target_profile.get("name", ""),
            "base_model": target_model,
            "provider": target_profile.get("provider", ""),
            "family": target_profile.get("family", ""),
            "recommended_use": target_profile.get("recommended_use", ""),
        },
        "data": data,
        "output_dir": str(output_dir),
        "max_samples": max_samples,
        "auto_register": auto_register,
        "agentdog_reference": {
            "sft": "Train the target policy on guard-filtered safe trajectories.",
            "rl": "Use the guard as safety judge/reward and combine safety with task utility.",
            "reward_formula": DEFAULT_REWARD_FORMULA,
        },
        "reward_weights": {
            "safety": safety_weight,
            "utility": utility_weight,
            "note": "Default UI weights are for planning; GRPO/DPO reward hooks can override them on the GPU server.",
        },
        "commands": _planned_commands(
            data=data,
            target_profile=str(target_profile.get("name", "")),
            target_model=target_model,
            output_dir=str(output_dir),
            algorithm=algorithm,
            max_samples=max_samples,
        ),
    }


def write_enchantment_plan(output_dir: str | Path, plan: Dict[str, Any]) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    plan_path = path / "safety_enchantment_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path


def _planned_commands(
    *,
    data: Dict[str, Any],
    target_profile: str,
    target_model: str,
    output_dir: str,
    algorithm: str,
    max_samples: int | None,
) -> list[str]:
    suffix = f" --max-samples {max_samples}" if max_samples else ""
    commands = [
        (
            "python scripts/train_sft.py "
            f"--data {_q(data['files'].get('agentdog_binary_sft', data['files'].get('tracehound_sft', 'DATA_MISSING')))} "
            f"--model-profile {_q(target_profile)} --base-model {_q(target_model)} "
            f"--output-dir {_q(str(Path(output_dir) / 'sft'))}{suffix} --run"
        )
    ]
    if algorithm == "sft_dpo":
        commands.append(
            "python scripts/train_preference.py "
            f"--data {_q(data['files'].get('preference', 'DATA_MISSING'))} "
            f"--model-profile {_q(target_profile)} --base-model {_q(str(Path(output_dir) / 'sft'))} "
            f"--algorithm dpo --output-dir {_q(str(Path(output_dir) / 'dpo'))}{suffix} --run"
        )
    if algorithm == "sft_grpo":
        commands.append(
            "python scripts/train_preference.py "
            f"--data {_q(data['files'].get('rl', 'DATA_MISSING'))} "
            f"--model-profile {_q(target_profile)} --base-model {_q(str(Path(output_dir) / 'sft'))} "
            f"--algorithm grpo --output-dir {_q(str(Path(output_dir) / 'grpo'))}{suffix} --run"
        )
    return commands


def _resolve_data_dir(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    if path.name == "latest" and path.exists() and path.is_file():
        return Path(path.read_text(encoding="utf-8").strip())
    if path.exists() and path.is_file():
        return Path(path.read_text(encoding="utf-8").strip())
    return path


def _jsonl_stats(path: Path) -> Dict[str, int]:
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return {"samples": rows}


def _q(value: str) -> str:
    return shlex.quote(str(value))
