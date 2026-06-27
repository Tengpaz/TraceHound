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

