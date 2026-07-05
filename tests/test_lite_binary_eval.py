import json

from traceguard.lite_binary_eval import (
    build_lite_binary_prompt,
    compute_binary_eval_metrics,
    dataset_to_eval_examples,
    parse_judgment_output,
    split_sft_rows,
)


def test_build_lite_binary_prompt_uses_snake_case_taxonomy_and_json_contract():
    prompt = build_lite_binary_prompt("[USER]: hi\n[AGENT]:\n[ACTION]: Complete{}")
    assert "direct_prompt_injection" in prompt
    assert "unconfirmed_or_over_privileged_action" in prompt
    assert "reputational_and_interpersonal_harm" in prompt
    assert "Direct Prompt Injection" not in prompt
    assert '"judgment": "safe"' in prompt
    assert "Return exactly one strict JSON object" in prompt


def test_parse_judgment_output_accepts_json_and_rejects_old_format():
    assert parse_judgment_output('{"judgment": "safe"}') == ("safe", False)
    assert parse_judgment_output('```json\n{"judgment": "unsafe"}\n```') == ("unsafe", False)
    assert parse_judgment_output("safe") == (None, True)
    assert parse_judgment_output('{"judgment": "maybe"}') == (None, True)


def test_dataset_to_eval_examples_does_not_leak_gold_reason():
    row = {
        "id": 7,
        "tool_used": [{"name": "tool"}],
        "contents": [
            [
                {"role": "user", "content": "do x"},
                {"role": "agent", "thought": "thinking", "action": "Tool{}"},
                {"role": "environment", "content": "{}"},
            ]
        ],
        "label": 1,
        "reason": "gold reason should not be in prompt",
    }
    example = dataset_to_eval_examples([row], "atbench")[0]
    assert example["gold"] == "unsafe"
    assert "gold reason should not be in prompt" not in example["prompt"]
    assert "[THOUGHT]: thinking" in example["prompt"]
    assert "[ACTION]: Tool{}" in example["prompt"]


def test_compute_binary_eval_metrics_counts_invalid_outputs():
    metrics = compute_binary_eval_metrics(
        [
            {"gold": "unsafe", "prediction": "unsafe", "invalid": False, "output_tokens": 4},
            {"gold": "safe", "prediction": "unsafe", "invalid": False, "output_tokens": 8},
            {"gold": "unsafe", "prediction": None, "invalid": True, "output_tokens": 2},
        ]
    )
    assert metrics["samples"] == 3
    assert metrics["accuracy"] == 1 / 3
    assert metrics["invalid_rate"] == 1 / 3
    assert metrics["output_token_cost"]["max"] == 8
    assert metrics["output_token_cost"]["min"] == 2


def test_split_sft_rows_is_stratified():
    rows = []
    for idx in range(20):
        label = "safe" if idx < 10 else "unsafe"
        rows.append(
            {
                "id": str(idx),
                "label": label,
                "messages": [
                    {"role": "user", "content": "u"},
                    {"role": "assistant", "content": json.dumps({"judgment": label})},
                ],
            }
        )
    train, val = split_sft_rows(rows, val_ratio=0.2, seed=1)
    assert len(train) == 16
    assert len(val) == 4
    assert {row["label"] for row in val} == {"safe", "unsafe"}
