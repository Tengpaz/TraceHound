from traceguard.config import redact_api_base


def test_redact_api_base_keeps_only_origin():
    assert redact_api_base("http://35.220.164.252:3888/v1/") == "http://35.220.164.252:3888"
    assert redact_api_base("https://api.example.com/v1/chat/completions") == "https://api.example.com"

