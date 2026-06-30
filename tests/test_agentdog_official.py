from pathlib import Path

from traceguard.agentdog_official import (
    OFFICIAL_DATASETS,
    PUBLIC_RELEASE_LIMITATIONS,
    format_agentdog15_unified_prompt,
    official_manifest,
)
from traceguard.data import built_in_cases
from traceguard.export import agentdog15_coarse_sft_rows, agentdog15_unified_sft_rows


def test_official_prompt_template_loads_and_formats():
    prompt = format_agentdog15_unified_prompt("step 1: user asks")

    assert "You are a security analyst reviewing an AI agent trajectory." in prompt
    assert "<BEGIN TRAJECTORY>" in prompt
    assert "step 1: user asks" in prompt


def test_official_exports_use_agentdog15_prompt_source():
    case = built_in_cases(count=1)[0]
    unified = next(agentdog15_unified_sft_rows([case]))
    coarse = next(agentdog15_coarse_sft_rows([case]))

    assert unified["prompt_source"] == "official_agentdog_v1.5"
    assert coarse["prompt_source"] == "official_agentdog_v1.5"
    assert "Safety:" in unified["messages"][1]["content"]
    assert "<Judgment>" in coarse["messages"][1]["content"]


def test_official_manifest_marks_synthetic_generator_as_surrogate(tmp_path):
    manifest = official_manifest(repo_dir=tmp_path / "missing_repo", data_root=tmp_path / "datasets")

    assert manifest["official_datasets"]["app1_sft"] == OFFICIAL_DATASETS["app1_sft"]
    assert PUBLIC_RELEASE_LIMITATIONS["llm_synthesis_engine_public"] is False
    assert manifest["synthetic_generator_status"] == "surrogate_not_official"
    assert all(not item["present"] for item in manifest["data_presence"].values())
