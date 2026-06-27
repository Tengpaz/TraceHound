import json

from traceguard.data import TOOL_SCENARIOS, built_in_cases, dataset_summary
from traceguard.quality import quality_check_jsonl


def test_built_in_cases_cover_tool_scenarios_and_labels():
    cases = built_in_cases()
    summary = dataset_summary(cases)
    assert summary["samples"] >= 16
    assert summary["labels"]["safe"] == summary["labels"]["unsafe"]
    for scenario in TOOL_SCENARIOS:
        assert scenario in summary["scenarios"]


def test_quality_check_accepts_generated_cases(tmp_path):
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in built_in_cases():
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    result = quality_check_jsonl(path)
    assert result["passed"] is True
    assert result["errors"] == []
    assert result["warnings"] == []

