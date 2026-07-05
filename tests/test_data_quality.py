import json

from traceguard.data import TOOL_SCENARIOS, built_in_cases, dataset_summary
from traceguard.export import (
    agentdog_binary_sft_rows,
    agentdog_taxonomy_sft_rows,
    agentdog_unified_sft_rows,
    preference_rows,
    sft_rows,
)
from traceguard.quality import filter_cases_by_agentdog_qc, quality_check_jsonl
from traceguard.schema import RiskReport
from traceguard.taxonomy import FAILURE_MODES, HARM_TYPES, RISK_SOURCES


def test_built_in_cases_cover_tool_scenarios_and_labels():
    cases = built_in_cases()
    summary = dataset_summary(cases)
    assert summary["samples"] >= 16
    assert summary["labels"]["safe"] == summary["labels"]["unsafe"]
    for scenario in TOOL_SCENARIOS:
        assert scenario in summary["scenarios"]


def test_scaled_generation_uses_unique_variant_ids():
    cases = built_in_cases(count=100)
    assert len(cases) == 100
    assert len({case["id"] for case in cases}) == 100
    assert cases[-1]["metadata"]["generation_flow"] == "agentdog_three_stage_planner"


def test_agentdog_generation_covers_full_taxonomy_when_scaled():
    cases = built_in_cases(count=2240, labels=["unsafe"])
    risk_sources = {case["gold"]["risk_source"] for case in cases}
    failure_modes = {case["gold"]["failure_mode"] for case in cases}
    harm_types = {case["gold"]["harm_type"] for case in cases}
    assert risk_sources == set(RISK_SOURCES)
    assert failure_modes == set(FAILURE_MODES)
    assert harm_types == set(HARM_TYPES)
    assert all(case["metadata"]["tool_library_size"] >= 10000 for case in cases)
    assert all(case["metadata"]["execution_plan"]["risk_injection_point"] for case in cases)


def test_small_agentdog_generation_diversifies_taxonomy_axes():
    cases = built_in_cases(count=10)
    unsafe = [case for case in cases if case["gold"]["label"] == "unsafe"]
    assert len(unsafe) == 5
    assert len({case["gold"]["risk_source"] for case in unsafe}) > 1
    assert len({case["gold"]["failure_mode"] for case in unsafe}) > 1
    assert len({case["gold"]["harm_type"] for case in unsafe}) > 1


def test_agentdog_sft_exports_binary_and_taxonomy_rows():
    cases = built_in_cases(count=4)
    binary_rows = list(agentdog_binary_sft_rows(cases))
    taxonomy_rows = list(agentdog_taxonomy_sft_rows(cases))
    unified_rows = list(agentdog_unified_sft_rows(cases))
    assert len(binary_rows) == 4
    assert len(taxonomy_rows) == sum(1 for case in cases if case["gold"]["label"] == "unsafe")
    assert len(unified_rows) == 4
    assert binary_rows[0]["task_type"] == "trajectory_level_safety_evaluation"
    assert binary_rows[0]["messages"][-1]["content"] in {"Safe", "Unsafe"}
    assert list(taxonomy_rows[0]) == ["id", "task", "messages"]
    assert taxonomy_rows[0]["id"].startswith("taxonomy-")
    assert taxonomy_rows[0]["task"] == "taxonomy"
    assert [message["role"] for message in taxonomy_rows[0]["messages"]] == ["user", "assistant"]
    assert "Task:\nEvaluate the risk source, failure mode, and real world harm" in taxonomy_rows[0]["messages"][0]["content"]
    assert "<BEGIN CATEGORIZATION>" in taxonomy_rows[0]["messages"][0]["content"]
    assert "<BEGIN TRAJECTORY>" in taxonomy_rows[0]["messages"][0]["content"]
    assert taxonomy_rows[0]["messages"][-1]["content"].startswith("Risk Source:")
    assert "Real World Harm:" in taxonomy_rows[0]["messages"][-1]["content"]
    assert "Safety:" not in taxonomy_rows[0]["messages"][-1]["content"]
    assert list(unified_rows[0]) == ["id", "task", "messages"]
    assert unified_rows[0]["id"].startswith("unified-")
    assert unified_rows[0]["task"] == "unified"
    assert "Safety:" in unified_rows[0]["messages"][-1]["content"]
    assert "Risk Source:" in unified_rows[0]["messages"][-1]["content"]
    assert "Failure Mode:" in unified_rows[0]["messages"][-1]["content"]
    assert "Real World Harm:" in unified_rows[0]["messages"][-1]["content"]
    safe_unified = next(row for row, case in zip(unified_rows, cases) if case["gold"]["label"] == "safe")
    assert safe_unified["messages"][-1]["content"] == "\n".join(
        [
            "Safety: Safe",
            "Risk Source: None",
            "Failure Mode: None",
            "Real World Harm: None",
        ]
    )


def test_tracehound_sft_prompt_is_taxonomy_grounded_json_task():
    case = built_in_cases(count=1)[0]
    row = next(sft_rows([case]))
    prompt = row["messages"][1]["content"]
    target = json.loads(row["messages"][-1]["content"])

    assert row["task_type"] == "tracehound_risk_report_json"
    assert "<BEGIN CATEGORIZATION>" in prompt
    assert "<BEGIN TRAJECTORY_JSON>" in prompt
    assert "Return exactly one strict JSON object" in prompt
    assert '"risk_source": "none | one TraceHound risk_source machine label"' in prompt
    assert "malicious_user_instruction_or_jailbreak" in prompt
    assert "unconfirmed_or_over_privileged_action" in prompt
    assert "privacy_and_confidentiality_harm" in prompt
    assert RiskReport.model_validate(target)


def test_preference_rows_use_deterministic_hard_negatives():
    cases = built_in_cases(count=4)
    rows = list(preference_rows(cases))
    unsafe_row = next(row for row, case in zip(rows, cases) if case["gold"]["label"] == "unsafe")
    safe_row = next(row for row, case in zip(rows, cases) if case["gold"]["label"] == "safe")

    unsafe_chosen = RiskReport.model_validate(json.loads(unsafe_row["chosen"]))
    unsafe_rejected = RiskReport.model_validate(json.loads(unsafe_row["rejected"]))
    assert unsafe_row["negative_type"] == "unsafe_wrong_taxonomy_or_evidence"
    assert unsafe_rejected.label == "unsafe"
    assert (
        unsafe_rejected.risk_source != unsafe_chosen.risk_source
        or unsafe_rejected.failure_mode != unsafe_chosen.failure_mode
        or unsafe_rejected.harm_type != unsafe_chosen.harm_type
        or unsafe_rejected.evidence_steps != unsafe_chosen.evidence_steps
    )
    assert "Incorrect or ungrounded risk report" not in unsafe_rejected.reason

    safe_rejected = RiskReport.model_validate(json.loads(safe_row["rejected"]))
    assert safe_row["negative_type"] == "safe_false_positive"
    assert safe_rejected.label == "unsafe"
    assert safe_rejected.risk_source != "none"
    assert safe_rejected.failure_mode != "none"
    assert safe_rejected.harm_type != "none"
    assert safe_rejected.evidence_steps
    assert "<BEGIN CATEGORIZATION>" in safe_row["prompt"]


def test_quality_check_accepts_generated_cases(tmp_path):
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in built_in_cases():
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    result = quality_check_jsonl(path)
    assert result["passed"] is True
    assert result["errors"] == []
    assert result["warnings"] == []
    assert result["agentdog_qc"]["pass_rate"] == 1.0


def test_agentdog_qc_rejects_invalid_tool_sample():
    case = built_in_cases(count=1, labels=["unsafe"])[0]
    bad = json.loads(json.dumps(case))
    bad["trajectory"][1]["tool_name"] = "unknown_tool.run"

    kept, report = filter_cases_by_agentdog_qc([bad])

    assert kept == []
    assert report["pass_rate"] == 0.0
    assert report["rejected_samples"]
    assert "unknown tool" in " ".join(report["rejected_samples"][0]["errors"])
