import json

from traceguard.data import built_in_cases
from traceguard.evaluation import evaluate_dataset, evaluate_final_answer_only_dataset, summarize_predictions
from traceguard.schema import RiskReport


def test_evaluation_outputs_metrics(tmp_path):
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in built_in_cases():
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    result = evaluate_dataset(path, mode="layered")
    assert result["metrics"]["samples"] >= 6
    assert "accuracy" in result["metrics"]
    assert "precision" in result["metrics"]
    assert "recall" in result["metrics"]
    assert "f_score" in result["metrics"]
    assert "evidence_hit_rate" in result["metrics"]
    assert "false_block_rate" in result["metrics"]
    assert "evidence_precision" in result["metrics"]
    assert "evidence_recall" in result["metrics"]


def test_final_answer_only_baseline_misses_intermediate_tool_risk(tmp_path):
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in built_in_cases():
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    layered = evaluate_dataset(path, mode="layered")
    final_only = evaluate_final_answer_only_dataset(path, mode="layered")

    assert final_only["metrics"]["unsafe_recall"] <= layered["metrics"]["unsafe_recall"]


def test_precision_recall_f_score_use_fine_grained_taxonomy_exact_match():
    golds = [
        RiskReport(
            label="unsafe",
            decision="block",
            risk_source="malicious_user_instruction_or_jailbreak",
            failure_mode="unconfirmed_or_over_privileged_action",
            harm_type="privacy_and_confidentiality_harm",
        ),
        RiskReport(
            label="unsafe",
            decision="block",
            risk_source="indirect_prompt_injection",
            failure_mode="failure_to_validate_tool_outputs",
            harm_type="security_and_system_integrity_harm",
        ),
        RiskReport(label="safe", decision="allow"),
    ]
    predictions = [
        RiskReport(
            label="unsafe",
            decision="block",
            risk_source="malicious_user_instruction_or_jailbreak",
            failure_mode="unconfirmed_or_over_privileged_action",
            harm_type="privacy_and_confidentiality_harm",
        ),
        RiskReport(
            label="unsafe",
            decision="block",
            risk_source="indirect_prompt_injection",
            failure_mode="failure_to_validate_tool_outputs",
            harm_type="functional_and_opportunity_harm",
        ),
        RiskReport(
            label="unsafe",
            decision="block",
            risk_source="inherent_agent_failures",
            failure_mode="unconfirmed_or_over_privileged_action",
            harm_type="functional_and_opportunity_harm",
        ),
    ]

    metrics = summarize_predictions(golds, predictions)

    assert metrics["accuracy"] == 0.6667
    assert metrics["precision"] == 0.3333
    assert metrics["recall"] == 0.5
    assert metrics["f_score"] == 0.4
    assert metrics["unsafe_recall"] == 1.0
