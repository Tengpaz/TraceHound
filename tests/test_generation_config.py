from traceguard.generation_config import load_generation_config


def test_load_generation_config_from_yaml_subset(tmp_path):
    config_path = tmp_path / "generation.yaml"
    config_path.write_text(
        "\n".join(
            [
                "out: data/tmp/generated",
                "count: 10000",
                "limit:",
                "scenarios: [browser, shell]",
                "labels: [unsafe]",
                "generation_backend: llm",
                "include_eval: true",
                "include_sft: false",
                "include_agentdog_sft: true",
                "include_preference: true",
                "include_rl: true",
                "qc_policy: agentdog_strict",
                "llm_qc: true",
                "llm_generation_retries: 3",
                "llm_generation_temperature: 0.15",
                "llm_qc_judge: hybrid",
                "llm_qc_judges: [api, hybrid:intern-s2-preview]",
                "llm_qc_mode: compressed",
                "llm_qc_consensus_threshold: 0.75",
                "qc_min_score: 0.8",
                "write_clean_layout: false",
                "write_legacy_flat_files: false",
                "split_train_ratio: 0.7",
                "split_eval_ratio: 0.2",
                "split_test_ratio: 0.1",
                "split_seed: 42",
                "write_qc_report: false",
                "write_examples: true",
            ]
        ),
        encoding="utf-8",
    )

    config = load_generation_config(config_path)

    assert config["out"] == "data/tmp/generated"
    assert config["count"] == 10000
    assert config["limit"] is None
    assert config["scenarios"] == ["browser", "shell"]
    assert config["labels"] == ["unsafe"]
    assert config["generation_backend"] == "llm"
    assert config["include_eval"] is True
    assert config["include_sft"] is False
    assert config["include_agentdog_sft"] is True
    assert config["include_preference"] is True
    assert config["include_rl"] is True
    assert config["qc_policy"] == "agentdog_strict"
    assert config["llm_qc"] is True
    assert config["llm_generation_retries"] == 3
    assert config["llm_generation_temperature"] == 0.15
    assert config["llm_qc_judge"] == "hybrid"
    assert config["llm_qc_judges"] == ["api", "hybrid:intern-s2-preview"]
    assert config["llm_qc_mode"] == "compressed"
    assert config["llm_qc_consensus_threshold"] == 0.75
    assert config["qc_min_score"] == 0.8
    assert config["write_clean_layout"] is False
    assert config["write_legacy_flat_files"] is False
    assert config["split_train_ratio"] == 0.7
    assert config["split_eval_ratio"] == 0.2
    assert config["split_test_ratio"] == 0.1
    assert config["split_seed"] == 42
    assert config["write_qc_report"] is False
    assert config["write_examples"] is True
