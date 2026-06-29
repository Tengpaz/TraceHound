#!/usr/bin/env python
"""GPU/server environment diagnostics for TraceHound deployments."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traceguard.model_profiles import profile_summary, resolve_model_profile


TRAIN_PACKAGES = ("torch", "transformers", "peft", "accelerate")
PREFERENCE_PACKAGES = ("datasets", "trl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when CUDA is not visible.")
    args = parser.parse_args()

    report = build_report()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.strict and not report["torch"].get("cuda_available"):
        raise SystemExit("CUDA is not visible to torch. Check NVIDIA driver, CUDA wheel, or container GPU runtime.")


def build_report() -> Dict[str, Any]:
    return {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "nvidia_smi": _nvidia_smi(),
        "packages": {package: _package_version(package) for package in TRAIN_PACKAGES},
        "preference_packages": {package: _package_version(package) for package in PREFERENCE_PACKAGES},
        "torch": _torch_report(),
        "tracehound": {
            "api_configured": bool(os.getenv("TRACEHOUND_API_BASE") and os.getenv("TRACEHOUND_MODEL")),
            "api_key_present": bool(os.getenv("TRACEHOUND_API_KEY")),
            "model_profile": _model_profile_report(),
            "local_model": os.getenv("TRACEHOUND_LOCAL_MODEL", ""),
            "local_model_path": os.getenv("TRACEHOUND_LOCAL_MODEL_PATH", ""),
        },
    }


def _model_profile_report() -> Dict[str, Any]:
    name = os.getenv("TRACEHOUND_MODEL_PROFILE")
    try:
        profile = resolve_model_profile(name)
        return profile_summary(profile)
    except Exception as exc:
        return {"configured": name or "", "error": str(exc)}


def _package_version(package: str) -> str | None:
    if importlib.util.find_spec(package) is None:
        return None
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "installed"


def _torch_report() -> Dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {"installed": False, "cuda_available": False}
    try:
        import torch  # type: ignore

        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / 1024**3, 2),
                        "capability": f"{props.major}.{props.minor}",
                    }
                )
        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": torch.cuda.device_count(),
            "devices": devices,
        }
    except Exception as exc:  # pragma: no cover - environment diagnostic.
        return {"installed": True, "cuda_available": False, "error": str(exc)}


def _nvidia_smi() -> Dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc)}
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    return {"available": bool(rows), "gpus": rows}


if __name__ == "__main__":
    main()
