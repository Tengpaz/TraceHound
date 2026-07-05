from traceguard.model_profiles import list_model_profiles, profile_model_id, resolve_model_profile


def test_intern_model_profiles_are_available():
    profiles = list_model_profiles()
    assert "internlm3-8b-instruct" in profiles
    assert "internlm2_5-1_8b-chat" in profiles
    assert "internlm2_5-7b-chat" in profiles
    assert "internlm2_5-20b-chat" in profiles
    assert "tracehound-base-qwen3_5-0_8b-binary" in profiles
    assert "intern-s2-preview" in profiles


def test_primary_and_api_profile_ids():
    primary = resolve_model_profile("internlm3-8b-instruct")
    assert primary["provider"] == "huggingface"
    assert profile_model_id(primary) == "internlm/internlm3-8b-instruct"
    assert primary["lora"]["target_modules"]

    api = resolve_model_profile("intern-s2-preview")
    assert api["provider"] == "openai_compatible"
    assert api["api_base"] == "https://chat.intern-ai.org.cn/api/v1"
    assert profile_model_id(api) == "intern-s2-preview"


def test_tracehound_base_binary_profile_uses_local_path_and_contract():
    profile = resolve_model_profile("tracehound-base-qwen3_5-0_8b-binary")
    assert profile["role"] == "binary_guard"
    assert profile["task"] == "agentdog_lite_binary"
    assert profile["output_contract"]["type"] == "binary_judgment"
    assert profile_model_id(profile).endswith("models/TraceHound-Base-Qwen3.5-0.8B-Binary")
