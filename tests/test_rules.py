from traceguard.data import built_in_cases
from traceguard.rules import RuleEngine
from traceguard.schema import TrajectoryCase


def _case(case_id: str) -> TrajectoryCase:
    raw = next(item for item in built_in_cases() if item["id"] == case_id)
    return TrajectoryCase.model_validate(raw)


def test_rules_hit_indirect_injection_case():
    report = RuleEngine().evaluate(_case("case_web_injection_secret_email"))
    assert report is not None
    assert report.label == "unsafe"
    assert report.risk_source == "indirect_prompt_injection"


def test_rules_do_not_mark_refusal_unsafe():
    report = RuleEngine().evaluate(_case("case_malicious_user_refused"))
    assert report is not None
    assert report.label == "safe"
    assert report.decision == "block"

