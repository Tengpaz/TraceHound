import pytest
from pydantic import ValidationError

from traceguard.schema import RiskReport, TrajectoryCase


def test_schema_parses_minimal_case():
    case = TrajectoryCase.model_validate(
        {
            "id": "case_1",
            "trajectory": [
                {"step": 1, "role": "user", "type": "message", "content": "hello"},
                {"step": 2, "role": "assistant", "type": "final", "content": "safe"},
            ],
        }
    )
    assert case.id == "case_1"


def test_schema_rejects_invalid_step():
    with pytest.raises(ValidationError):
        TrajectoryCase.model_validate(
            {
                "id": "bad",
                "trajectory": [{"step": 0, "role": "user", "type": "message"}],
            }
        )


def test_schema_rejects_invalid_taxonomy_label():
    with pytest.raises(ValidationError):
        RiskReport(risk_source="old_label")


def test_schema_normalizes_legacy_taxonomy_labels():
    report = RiskReport(
        label="unsafe",
        risk_source="inherent_agent_or_llm_failures",
        failure_mode="choosing_malicious_tools",
        harm_type="privacy_confidentiality_harm",
    )

    assert report.risk_source == "inherent_agent_failures"
    assert report.failure_mode == "choosing_malicious_tool"
    assert report.harm_type == "privacy_and_confidentiality_harm"
