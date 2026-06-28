from traceguard.cost import estimate_api_cost


def test_api_cost_estimate_uses_per_million_token_env(monkeypatch):
    monkeypatch.setenv("TRACEHOUND_INPUT_PRICE_PER_1M", "2")
    monkeypatch.setenv("TRACEHOUND_OUTPUT_PRICE_PER_1M", "8")

    input_cost, output_cost, note = estimate_api_cost(500_000, 250_000, model_calls=1)

    assert input_cost == 1.0
    assert output_cost == 2.0
    assert note == "estimated_from_env_per_1m_token_prices"


def test_api_cost_estimate_is_zero_for_offline_paths(monkeypatch):
    monkeypatch.setenv("TRACEHOUND_INPUT_PRICE_PER_1M", "2")
    monkeypatch.setenv("TRACEHOUND_OUTPUT_PRICE_PER_1M", "8")

    input_cost, output_cost, note = estimate_api_cost(500_000, 250_000, model_calls=0)

    assert input_cost == 0.0
    assert output_cost == 0.0
    assert note == "offline_or_rule_path"
