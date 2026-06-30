#!/usr/bin/env python
"""Plan or launch safety enchantment for a target policy model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.enchantment import build_enchantment_plan, write_enchantment_plan
from traceguard.model_profiles import profile_model_id, resolve_model_profile


TRAIN_PACKAGES = ("torch", "transformers", "peft", "accelerate")
PREFERENCE_PACKAGES = ("datasets", "trl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/tmp/generated/latest")
    parser.add_argument("--algorithm", choices=["sft", "sft_dpo", "sft_grpo"], default="sft_dpo")
    parser.add_argument("--target-model-profile", default=os.getenv("TRACEHOUND_TARGET_MODEL_PROFILE", "internlm3-8b-instruct"))
    parser.add_argument("--target-base-model", help="Target policy HF id or local checkpoint override.")
    parser.add_argument("--guard-model", default=os.getenv("TRACEHOUND_GUARD_MODEL", "tracehound-current-guard"))
    parser.add_argument("--guard-mode", default=os.getenv("TRACEHOUND_GUARD_MODE", "layered"))
    parser.add_argument("--output-dir", default="checkpoints/safety_enchantment")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--safety-weight", type=float, default=0.5)
    parser.add_argument("--utility-weight", type=float, default=0.5)
    parser.add_argument("--run", action="store_true", help="Run planned SFT/DPO commands on a GPU server.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when dependencies or required data are missing.")
    args = parser.parse_args()

    profile = resolve_model_profile(args.target_model_profile)
    if profile.get("provider") != "huggingface":
        raise SystemExit("safety enchantment currently targets Hugging Face local policy models")
    target_model = args.target_base_model or profile_model_id(profile)
    plan = build_enchantment_plan(
        target_profile=profile,
        target_base_model=target_model,
        guard_model=args.guard_model,
        guard_mode=args.guard_mode,
        data_dir=args.data_dir,
        algorithm=args.algorithm,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        safety_weight=args.safety_weight,
        utility_weight=args.utility_weight,
        auto_register=True,
    )
    write_enchantment_plan(args.output_dir, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))

    missing_data = plan["data"]["missing"]
    if missing_data:
        message = f"missing required safety enchantment data files: {', '.join(missing_data)}"
        if args.strict or args.run:
            raise SystemExit(message)
        print(message)
        return

    missing_packages = _missing_packages(args.algorithm)
    if missing_packages:
        message = (
            "Safety enchantment training dependencies are missing. This is expected on the Mac CPU environment.\n"
            f"Missing packages: {', '.join(missing_packages)}\n"
            "On Linux/GPU, install CUDA-matched PyTorch first, then run `pip install -e \".[train,preference]\"`."
        )
        if args.strict or args.run:
            raise SystemExit(message)
        print(message)
        return

    if not args.run:
        print("Plan written. Add --run on the GPU server to launch the planned training commands.")
        return

    if args.algorithm == "sft_grpo":
        raise SystemExit(
            "GRPO plan is written, but the live reward hook should be wired to the official contest environment "
            "before launching a long run."
        )
    for command in plan["commands"]:
        subprocess.run(shlex.split(command), cwd=ROOT, check=True)


def _missing_packages(algorithm: str) -> list[str]:
    packages = list(TRAIN_PACKAGES)
    if algorithm in {"sft_dpo", "sft_grpo"}:
        packages.extend(PREFERENCE_PACKAGES)
    return [package for package in packages if importlib.util.find_spec(package) is None]


if __name__ == "__main__":
    main()
