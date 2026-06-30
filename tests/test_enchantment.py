import json

from traceguard.enchantment import DEFAULT_REWARD_FORMULA, build_enchantment_plan, resolve_enchantment_data


def test_resolve_enchantment_data_requires_algorithm_files(tmp_path):
    (tmp_path / "agentdog_binary_sft.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "synthetic_rl.jsonl").write_text("{}\n{}\n", encoding="utf-8")

    data = resolve_enchantment_data(tmp_path, "sft_grpo")

    assert data["missing"] == []
    assert data["stats"]["agentdog_binary_sft"]["samples"] == 1
    assert data["stats"]["rl"]["samples"] == 2


def test_build_enchantment_plan_uses_guard_as_reward_and_target_policy(tmp_path):
    (tmp_path / "agentdog_binary_sft.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "synthetic_preference.jsonl").write_text("{}\n", encoding="utf-8")
    profile = {
        "name": "internlm3-8b-instruct",
        "provider": "huggingface",
        "family": "internlm",
        "hf_model_id": "internlm/internlm3-8b-instruct",
        "recommended_use": "formal LoRA",
    }

    plan = build_enchantment_plan(
        target_profile=profile,
        target_base_model="",
        guard_model="tracehound-guard",
        guard_mode="layered",
        data_dir=tmp_path,
        algorithm="sft_dpo",
        output_dir=tmp_path / "out",
        max_samples=8,
    )

    assert plan["task"] == "safety_enchantment"
    assert plan["guard"]["roles"] == ["data_filter", "semantic_judge", "safety_reward"]
    assert plan["target"]["base_model"] == "internlm/internlm3-8b-instruct"
    assert plan["agentdog_reference"]["reward_formula"] == DEFAULT_REWARD_FORMULA
    assert len(plan["commands"]) == 2
    json.dumps(plan)
