"""Small dependency-free generation config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


DEFAULT_GENERATION_CONFIG: Dict[str, Any] = {
    "out": "data",
    "count": None,
    "limit": None,
    "scenarios": [],
    "labels": [],
    "generation_backend": "deterministic",
    "include_eval": True,
    "include_sft": True,
    "include_agentdog_sft": True,
    "include_agentdog15_official_sft": True,
    "include_preference": True,
    "include_rl": False,
    "qc_policy": "agentdog_local",
    "llm_qc": False,
    "llm_generation_retries": 2,
    "llm_generation_temperature": 0.2,
    "semantic_repair_backend": "static",
    "semantic_repair_rounds": 1,
    "llm_qc_judge": "api",
    "llm_qc_judges": [],
    "llm_qc_mode": "compressed",
    "llm_qc_consensus_threshold": 0.67,
    "qc_min_score": 0.74,
    "training_max_repair_level": "structural",
    "write_qc_report": True,
    "write_examples": False,
}


def load_generation_config(path: str | Path | None) -> Dict[str, Any]:
    config = dict(DEFAULT_GENERATION_CONFIG)
    if not path:
        return config
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"generation config not found: {config_path}")
    parsed = _parse_simple_yaml(config_path.read_text(encoding="utf-8"))
    config.update({key: value for key, value in parsed.items() if key in config})
    config["scenarios"] = _normalize_filter(config.get("scenarios"))
    config["labels"] = _normalize_filter(config.get("labels"))
    return config


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"line {line_no}: expected key: value")
        key, value = line.split(":", 1)
        result[key.strip()] = _parse_value(value.strip())
    return result


def _parse_value(value: str) -> Any:
    if value == "":
        return None
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_value(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value.strip('"').strip("'")


def _normalize_filter(value: Any) -> list[str]:
    if value in (None, "", "all"):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "", "all")]
    return [str(value)]
