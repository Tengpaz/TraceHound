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
                "include_eval: true",
                "include_sft: false",
                "include_preference: true",
                "include_rl: true",
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
    assert config["include_eval"] is True
    assert config["include_sft"] is False
    assert config["include_preference"] is True
    assert config["include_rl"] is True
