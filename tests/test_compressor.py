from traceguard.compressor import compress_trajectory
from traceguard.data import built_in_cases
from traceguard.schema import TrajectoryCase


def test_compressor_extracts_risk_spans():
    raw = next(item for item in built_in_cases() if item["id"] == "case_web_injection_secret_email")
    compressed = compress_trajectory(TrajectoryCase.model_validate(raw))
    assert 3 in compressed.evidence_candidates
    assert 4 in compressed.evidence_candidates
    assert compressed.external_observations
    assert compressed.sensitive_events

