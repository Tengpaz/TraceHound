"""Pydantic schemas for TraceHound inputs and outputs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from traceguard.taxonomy import (
    DECISIONS,
    FAILURE_MODE_LABELS,
    HARM_TYPE_LABELS,
    RISK_SOURCE_LABELS,
    SAFETY_LABELS,
    normalize_failure_mode,
    normalize_harm_type,
    normalize_risk_source,
)


class TraceModel(BaseModel):
    """Base model that keeps unknown official-data fields for later adapters."""

    model_config = ConfigDict(extra="allow")


class CostStats(TraceModel):
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    model_calls: int = 0
    strategy: str = "none"
    early_exit: bool = False
    compression_ratio: float = 1.0
    cost_reduction_ratio: float = 0.0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0
    pricing_note: str = ""

    @field_validator("input_tokens", "output_tokens", "latency_ms", "model_calls")
    @classmethod
    def non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("cost counters must be non-negative")
        return value


class TrajectoryStep(TraceModel):
    step: int
    role: str
    type: str
    content: Optional[str] = None
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("step")
    @classmethod
    def positive_step(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("step must be positive")
        return value

    @field_validator("role", "type")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("role and type must be non-empty")
        return value

    def text(self) -> str:
        parts = [self.content or "", self.tool_name or ""]
        if self.arguments:
            parts.append(str(self.arguments))
        return " ".join(parts)


class RiskReport(TraceModel):
    label: Literal["safe", "unsafe"] = "safe"
    decision: Literal["allow", "block", "ask_confirm", "sanitize"] = "allow"
    risk_source: str = "none"
    failure_mode: str = "none"
    harm_type: str = "none"
    evidence_steps: List[int] = Field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0
    cost: CostStats = Field(default_factory=CostStats)

    @field_validator("risk_source")
    @classmethod
    def valid_risk_source(cls, value: str) -> str:
        value = normalize_risk_source(value)
        if value not in RISK_SOURCE_LABELS:
            raise ValueError(f"unknown risk_source: {value}")
        return value

    @field_validator("failure_mode")
    @classmethod
    def valid_failure_mode(cls, value: str) -> str:
        value = normalize_failure_mode(value)
        if value not in FAILURE_MODE_LABELS:
            raise ValueError(f"unknown failure_mode: {value}")
        return value

    @field_validator("harm_type")
    @classmethod
    def valid_harm_type(cls, value: str) -> str:
        value = normalize_harm_type(value)
        if value not in HARM_TYPE_LABELS:
            raise ValueError(f"unknown harm_type: {value}")
        return value

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("confidence must be in [0, 1]")
        return value

    @field_validator("evidence_steps")
    @classmethod
    def positive_evidence_steps(cls, value: List[int]) -> List[int]:
        if any(step <= 0 for step in value):
            raise ValueError("evidence steps must be positive")
        return sorted(dict.fromkeys(value))

    @model_validator(mode="after")
    def safe_reports_use_none(self) -> "RiskReport":
        if self.label == "safe":
            if self.risk_source != "none" or self.failure_mode != "none" or self.harm_type != "none":
                raise ValueError("safe reports must use none taxonomy labels")
        return self


class TrajectoryCase(TraceModel):
    id: str
    task: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    trajectory: List[TrajectoryStep]
    label: Optional[Dict[str, Any]] = None

    @field_validator("id")
    @classmethod
    def non_empty_id(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("id must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_step_sequence(self) -> "TrajectoryCase":
        steps = [item.step for item in self.trajectory]
        if len(steps) != len(set(steps)):
            raise ValueError("trajectory step numbers must be unique")
        if steps != sorted(steps):
            raise ValueError("trajectory steps must be sorted")
        return self


class GuardDecision(TraceModel):
    decision: Literal["allow", "block", "ask_confirm", "sanitize"]
    report: RiskReport
    candidate_action: Optional[TrajectoryStep] = None


def dump_model(model: BaseModel) -> Dict[str, Any]:
    """Pydantic v2-compatible dumping helper."""

    return model.model_dump(mode="json")


def report_from_gold(raw: Dict[str, Any]) -> RiskReport:
    """Normalize gold records from either `safe` or `label` style fields."""

    data = dict(raw)
    if "label" not in data and "safe" in data:
        data["label"] = "safe" if data["safe"] else "unsafe"
    if "decision" not in data:
        data["decision"] = "allow" if data.get("label") == "safe" else "block"
    if "confidence" not in data:
        data["confidence"] = 1.0
    return RiskReport.model_validate(data)


def validate_taxonomy_label(kind: str, value: str) -> None:
    allowed = {
        "label": SAFETY_LABELS,
        "decision": DECISIONS,
        "risk_source": RISK_SOURCE_LABELS,
        "failure_mode": FAILURE_MODE_LABELS,
        "harm_type": HARM_TYPE_LABELS,
    }[kind]
    if value not in allowed:
        raise ValueError(f"invalid {kind}: {value}")
