import json

from traceguard.data import built_in_cases
from traceguard.llm_generation import synthesize_case_from_plan
from traceguard.production import filter_cases_for_training, production_quality_summary


class FakeClient:
    def __init__(self, payload):
        self.payload = payload

    def complete(self, messages, *, temperature=0.2):
        return json.dumps(self.payload)


def test_training_filter_excludes_semantic_repaired_cases_by_default():
    planner = built_in_cases(count=1, labels=["unsafe"])[0]
    tool_name = planner["metadata"]["selected_tools"][-1]["name"]
    generated = {
        "id": planner["id"],
        "task": planner["task"],
        "metadata": {"scenario": planner["metadata"]["scenario"]},
        "trajectory": [
            {"step": 1, "role": "user", "type": "message", "content": "Inspect the workspace."},
            {"step": 2, "role": "agent", "type": "tool_call", "tool_name": tool_name, "arguments": {"target": "token=sk-test"}},
            {"step": 3, "role": "tool", "type": "observation", "tool_name": tool_name, "content": "Command completed."},
            {"step": 4, "role": "assistant", "type": "final", "content": "Done."},
        ],
        "gold": planner["gold"],
    }
    case = synthesize_case_from_plan(planner, client=FakeClient(generated), max_retries=0)

    kept, report = filter_cases_for_training([case], max_repair_level="structural")

    assert case["metadata"]["repair_level"] == "semantic"
    assert kept == []
    assert report["rejected"] == 1


def test_production_quality_summary_reports_raw_and_repair_rates():
    native = built_in_cases(count=1, labels=["safe"])[0]
    semantic = built_in_cases(count=1, labels=["unsafe"])[0]
    semantic.setdefault("metadata", {})["repair_level"] = "semantic"
    semantic["metadata"]["raw_agentdog_qc"] = {"passed": False}

    kept, _ = filter_cases_for_training([native, semantic], max_repair_level="structural")
    summary = production_quality_summary([native, semantic], training_cases=kept)

    assert summary["repair"]["counts"]["semantic"] == 1
    assert summary["repair"]["semantic_repair_rate"] == 0.5
    assert summary["raw_qc"]["available"] == 1
    assert summary["training_selection"]["eligible"] == 1
