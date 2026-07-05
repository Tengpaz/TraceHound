from traceguard.guardrail_gateway import evaluate_guardrail_event
from traceguard.judge import ModelAdapter
from traceguard.local_model import case_to_lite_binary_trajectory, risk_report_from_binary_judgment
from traceguard.schema import RiskReport, TrajectoryCase


def test_binary_judgment_maps_to_risk_report_without_loading_model():
    unsafe = risk_report_from_binary_judgment("unsafe", raw_output='{"judgment":"unsafe"}', model_name="unit")
    assert unsafe.label == "unsafe"
    assert unsafe.decision == "block"
    assert unsafe.binary_judgment == "unsafe"

    safe = risk_report_from_binary_judgment("safe", raw_output='{"judgment":"safe"}', model_name="unit")
    assert safe.label == "safe"
    assert safe.decision == "allow"

    invalid = risk_report_from_binary_judgment(None, invalid=True, raw_output="safe", model_name="unit")
    assert invalid.label == "unsafe"
    assert invalid.decision == "ask_confirm"
    assert invalid.invalid_output is True


def test_case_to_lite_binary_trajectory_formats_tool_calls():
    case = TrajectoryCase.model_validate(
        {
            "id": "case-1",
            "task": "Inspect a file.",
            "metadata": {"platform": "codex", "ignored": "not included"},
            "trajectory": [
                {"step": 1, "role": "user", "type": "message", "content": "Open README."},
                {
                    "step": 2,
                    "role": "agent",
                    "type": "tool_call",
                    "tool_name": "file.read",
                    "arguments": {"path": "README.md"},
                },
                {"step": 3, "role": "tool", "type": "observation", "content": "TraceHound"},
            ],
        }
    )

    text = case_to_lite_binary_trajectory(case)
    assert "=== Task ===" in text
    assert "[ACTION]: file.read" in text
    assert '{"path":"README.md"}' in text
    assert "[ENVIRONMENT]:" in text
    assert "ignored" not in text


def test_guardrail_pre_tool_use_calls_model_after_rule_allow():
    class UnsafeStub(ModelAdapter):
        def evaluate(self, case: TrajectoryCase, mode: str = "layered") -> RiskReport:
            assert case.trajectory[-1].tool_name == "file.read"
            return RiskReport(label="unsafe", decision="block", reason="stub", confidence=0.9)

    result = evaluate_guardrail_event(
        {
            "event_type": "pre_tool_use",
            "session_id": "session-1",
            "task": "Read the docs.",
            "tool_name": "Read",
            "tool_input": {"file_path": "README.md"},
        },
        platform="codex",
        judge=UnsafeStub(),
    )

    assert result["decision"] == "block"
    assert result["reason"] == "stub"
