"""Model profile registry for contest-time base model switching."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = ROOT / "configs" / "model_profiles.json"


def load_model_profiles(path: str | Path | None = None) -> Dict[str, Any]:
    profile_path = Path(path) if path else DEFAULT_PROFILE_PATH
    if not profile_path.is_absolute():
        profile_path = ROOT / profile_path
    if not profile_path.exists():
        raise FileNotFoundError(f"model profile config not found: {profile_path}")
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    profiles = data.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("model profile config must contain a non-empty profiles object")
    return data


def list_model_profiles(path: str | Path | None = None) -> Dict[str, Dict[str, Any]]:
    return dict(load_model_profiles(path).get("profiles", {}))


def default_profile_name(provider: str = "huggingface", path: str | Path | None = None) -> str:
    data = load_model_profiles(path)
    env_name = os.getenv("TRACEHOUND_MODEL_PROFILE")
    if env_name:
        return env_name
    if provider == "openai_compatible":
        return str(data.get("default_api_profile") or "")
    return str(data.get("default_local_profile") or "")


def default_guard_profile_name(path: str | Path | None = None) -> str:
    data = load_model_profiles(path)
    return str(os.getenv("TRACEHOUND_GUARD_MODEL_PROFILE") or data.get("default_guard_profile") or data.get("default_local_profile") or "")


def resolve_model_profile(name: str | None = None, path: str | Path | None = None) -> Dict[str, Any]:
    data = load_model_profiles(path)
    profile_name = name or os.getenv("TRACEHOUND_MODEL_PROFILE") or data.get("default_local_profile")
    profiles = data["profiles"]
    if profile_name not in profiles:
        known = ", ".join(sorted(profiles))
        raise ValueError(f"unknown model profile: {profile_name}. Known profiles: {known}")
    profile = dict(profiles[profile_name])
    profile["name"] = profile_name
    return profile


def profile_model_id(profile: Dict[str, Any]) -> str:
    provider = profile.get("provider")
    if provider == "openai_compatible":
        return str(profile.get("model") or "")
    local_path = profile.get("local_path")
    if local_path:
        path = Path(str(local_path))
        return str(path if path.is_absolute() else ROOT / path)
    return str(profile.get("hf_model_id") or profile.get("model") or "")


def profile_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": profile.get("name", ""),
        "provider": profile.get("provider", ""),
        "family": profile.get("family", ""),
        "role": profile.get("role", ""),
        "task": profile.get("task", ""),
        "model_id": profile_model_id(profile),
        "local_path": str(profile.get("local_path") or ""),
        "recommended_use": profile.get("recommended_use", ""),
        "smoke": bool(profile.get("smoke", False)),
        "formal_lora": bool(profile.get("formal_lora", False)),
        "recommended_max_input_tokens": profile.get("recommended_max_input_tokens"),
    }
