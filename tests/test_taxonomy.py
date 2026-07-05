from traceguard.taxonomy import (
    FAILURE_MODES,
    HARM_TYPES,
    RISK_SOURCES,
    normalize_failure_mode,
    normalize_harm_type,
    normalize_risk_source,
)


def test_agentdog_taxonomy_counts():
    assert len(RISK_SOURCES) == 8
    assert len(FAILURE_MODES) == 14
    assert len(HARM_TYPES) == 10


def test_taxonomy_machine_labels_match_agentdog_lite_testset_style():
    assert set(RISK_SOURCES) == {
        "malicious_user_instruction_or_jailbreak",
        "direct_prompt_injection",
        "indirect_prompt_injection",
        "tool_description_injection",
        "malicious_tool_execution",
        "corrupted_tool_feedback",
        "unreliable_or_misinformation",
        "inherent_agent_failures",
    }
    assert set(FAILURE_MODES) == {
        "flawed_planning_or_reasoning",
        "unconfirmed_or_over_privileged_action",
        "insecure_interaction_or_execution",
        "incorrect_tool_parameters",
        "choosing_malicious_tool",
        "tool_misuse_in_specific_context",
        "failure_to_validate_tool_outputs",
        "procedural_deviation_or_inaction",
        "inefficient_or_wasteful_execution",
        "generation_of_harmful_offensive_content",
        "instruction_for_harmful_illegal_activity",
        "generation_of_malicious_executables",
        "unauthorized_information_disclosure",
        "provide_inaccurate_misleading_or_unverified_information",
    }
    assert set(HARM_TYPES) == {
        "privacy_and_confidentiality_harm",
        "financial_and_economic_harm",
        "security_and_system_integrity_harm",
        "physical_and_health_harm",
        "psychological_and_emotional_harm",
        "reputational_and_interpersonal_harm",
        "info_ecosystem_and_societal_harm",
        "public_service_and_resource_harm",
        "fairness_equity_and_allocative_harm",
        "functional_and_opportunity_harm",
    }


def test_taxonomy_aliases_normalize_legacy_and_spec_draft_labels():
    assert normalize_risk_source("Inherent Agent/LLM Failures") == "inherent_agent_failures"
    assert normalize_risk_source("inherent_agent_or_llm_failures") == "inherent_agent_failures"
    assert normalize_failure_mode("Choosing Malicious Tool") == "choosing_malicious_tool"
    assert normalize_failure_mode("choosing_malicious_tools") == "choosing_malicious_tool"
    assert normalize_failure_mode("unconfirmed_or_overprivileged_action") == "unconfirmed_or_over_privileged_action"
    assert normalize_failure_mode("Insecure Execution or Interaction") == "insecure_interaction_or_execution"
    assert normalize_harm_type("Privacy & Confidentiality Harm") == "privacy_and_confidentiality_harm"
    assert normalize_harm_type("privacy_confidentiality_harm") == "privacy_and_confidentiality_harm"
    assert normalize_harm_type("reputation_and_interpersonal_harm") == "reputational_and_interpersonal_harm"
