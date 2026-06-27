"""Runtime configuration helpers for optional remote model validation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class ApiConfig:
    api_base: str
    api_key: Optional[str]
    model: str
    api_path: str = "/chat/completions"
    timeout: int = 60


def load_env_file(path: str | Path = ".env") -> Dict[str, str]:
    """Load simple KEY=VALUE entries without adding a dotenv dependency."""

    env_path = Path(path)
    loaded: Dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def api_config_from_env(
    *,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    api_path: Optional[str] = None,
    timeout: Optional[int] = None,
) -> ApiConfig:
    resolved_base = api_base or os.getenv("TRACEHOUND_API_BASE")
    resolved_model = model or os.getenv("TRACEHOUND_MODEL")
    if not resolved_base:
        raise ValueError("TRACEHOUND_API_BASE is required for API validation")
    if not resolved_model:
        raise ValueError("TRACEHOUND_MODEL is required for API validation")
    return ApiConfig(
        api_base=resolved_base,
        api_key=api_key or os.getenv("TRACEHOUND_API_KEY"),
        model=resolved_model,
        api_path=api_path or os.getenv("TRACEHOUND_API_PATH", "/chat/completions"),
        timeout=timeout or int(os.getenv("TRACEHOUND_API_TIMEOUT", "60")),
    )


def redact_api_base(api_base: Optional[str]) -> str:
    if not api_base:
        return ""
    parsed = urlparse(api_base)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return api_base.rstrip("/").split("/")[0]


def api_runtime_status() -> Dict[str, object]:
    api_base = os.getenv("TRACEHOUND_API_BASE", "")
    api_key = os.getenv("TRACEHOUND_API_KEY", "")
    model = os.getenv("TRACEHOUND_MODEL", "")
    api_path = os.getenv("TRACEHOUND_API_PATH", "/chat/completions")
    return {
        "configured": bool(api_base and model),
        "api_base": redact_api_base(api_base),
        "api_path": api_path,
        "model": model,
        "key_present": bool(api_key),
    }
