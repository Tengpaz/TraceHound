import json

from traceguard.data import built_in_cases
from traceguard.evaluation import evaluate_dataset, evaluate_final_answer_only_dataset


def test_evaluation_outputs_metrics(tmp_path):
    path = tmp_path / "eval.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for case in built_in_cases():
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    result = evaluate_dataset(path, mode="layered")
    assert result["metrics"]["samples"] >= 6
    assert "accuracy" in result["metrics"]
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
