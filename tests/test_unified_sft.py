import json

from scripts.train_unified_full_sft import UnifiedSFTDataset
from traceguard.unified_sft import (
    build_eval_examples,
    compute_unified_metrics,
    parse_unified_output,
    training_row_to_messages,
)


def test_training_row_normalizes_alias_labels_to_eval_contract():
    row = {
        "id": "x",
        "label": 1,
        "risk_source": "direct_prompt_injection",
        "failure_mode": "unconfirmed_or_overprivileged_action",
        "harm_type": "reputation_and_interpersonal_harm",
        "original_instruction": "Task:\n<BEGIN TRAJECTORY>\n[USER]: x\n<END TRAJECTORY>",
    }
    converted = training_row_to_messages(row)
    assert converted["label"] == "unsafe"
    assert converted["failure_mode"] == "unconfirmed_or_over_privileged_action"
    assert converted["harm_type"] == "reputational_and_interpersonal_harm"
    assert converted["messages"][-1]["content"] == "\n".join(
        [
            "Safety: unsafe",
            "Risk Source: direct_prompt_injection",
            "Failure Mode: unconfirmed_or_over_privileged_action",
            "Harm Type: reputational_and_interpersonal_harm",
        ]
    )


def test_parse_unified_output_accepts_aliases_and_rejects_bad_format():
    parsed, invalid, reason = parse_unified_output(
        "\n".join(
            [
                "Safety: safe",
                "Risk Source: benign",
                "Failure Mode: unconfirmed_or_overprivileged_action",
                "Harm Type: reputation_and_interpersonal_harm",
            ]
        )
    )
    assert invalid is False
    assert reason is None
    assert parsed["failure_mode"] == "unconfirmed_or_over_privileged_action"
    assert parsed["harm_type"] == "reputational_and_interpersonal_harm"

    parsed, invalid, reason = parse_unified_output("Safety: maybe")
    assert parsed is None
    assert invalid is True
    assert reason.startswith("missing_") or "unknown binary label" in reason


def test_atbench_safe_failure_mode_gold_is_benign():
    row = {
        "id": 1,
        "contents": [[{"role": "user", "content": "hello"}]],
        "label": 0,
        "risk_source": "indirect_prompt_injection",
        "failure_mode": "failure_to_validate_tool_outputs",
        "harm_type": "privacy_and_confidentiality_harm",
    }
    example = build_eval_examples([row], "atbench")[0]
    assert example["gold"]["safety"] == "safe"
    assert example["gold"]["risk_source"] == "indirect_prompt_injection"
    assert example["gold"]["failure_mode"] == "benign"
    assert example["gold"]["harm_type"] == "privacy_and_confidentiality_harm"


def test_unified_metrics_include_four_dimensions():
    predictions = [
        {
            "gold": {
                "safety": "unsafe",
                "risk_source": "direct_prompt_injection",
                "failure_mode": "unauthorized_information_disclosure",
                "harm_type": "privacy_and_confidentiality_harm",
            },
            "prediction": {
                "safety": "unsafe",
                "risk_source": "direct_prompt_injection",
                "failure_mode": "unauthorized_information_disclosure",
                "harm_type": "privacy_and_confidentiality_harm",
            },
            "invalid": False,
            "output_tokens": 16,
        },
        {
            "gold": {
                "safety": "safe",
                "risk_source": "benign",
                "failure_mode": "benign",
                "harm_type": "benign",
            },
            "prediction": {
                "safety": "unsafe",
                "risk_source": "direct_prompt_injection",
                "failure_mode": "unauthorized_information_disclosure",
                "harm_type": "privacy_and_confidentiality_harm",
            },
            "invalid": False,
            "output_tokens": 20,
        },
    ]
    metrics = compute_unified_metrics(predictions, include_taxonomy=True)
    assert metrics["safety"]["accuracy"] == 0.5
    assert metrics["risk_source"]["accuracy"] == 0.5
    assert metrics["failure_mode"]["accuracy"] == 0.5
    assert metrics["harm_type"]["accuracy"] == 0.5
    assert metrics["output_token_cost"]["median"] == 18.0


def test_dataset_left_truncates_prompt_and_preserves_target(tmp_path):
    row = {
        "id": "long",
        "messages": [
            {"role": "user", "content": " ".join(f"p{i}" for i in range(20))},
            {"role": "assistant", "content": "Safety: safe"},
        ],
    }
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    dataset = UnifiedSFTDataset(path, SpaceTokenizer(), max_seq_length=8)
    feature = dataset[0]
    record = dataset.length_records[0]
    assert record["truncated"] is True
    assert len(feature["input_ids"]) == 8
    assert feature["labels"][-2:] == feature["input_ids"][-2:]
    assert all(label == -100 for label in feature["labels"][:-2])


class SpaceTokenizer:
    pad_token = "<pad>"
    eos_token = "</s>"
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = []
        for message in messages:
            if message["role"] == "user":
                rendered.extend(["USER:", message["content"], "ASSISTANT:"])
            elif message["role"] == "assistant":
                rendered.append(message["content"])
        if add_generation_prompt and (not rendered or rendered[-1] != "ASSISTANT:"):
            rendered.append("ASSISTANT:")
        return " ".join(rendered)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}
