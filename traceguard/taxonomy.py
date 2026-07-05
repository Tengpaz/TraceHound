"""AgentDoG-aligned taxonomy labels used by TraceHound."""

from __future__ import annotations

import re

NONE = "none"

RISK_SOURCES = (
    "malicious_user_instruction_or_jailbreak",
    "direct_prompt_injection",
    "indirect_prompt_injection",
    "unreliable_or_misinformation",
    "tool_description_injection",
    "malicious_tool_execution",
    "corrupted_tool_feedback",
    "inherent_agent_failures",
)

FAILURE_MODES = (
    "unconfirmed_or_over_privileged_action",
    "flawed_planning_or_reasoning",
    "incorrect_tool_parameters",
    "choosing_malicious_tool",
    "tool_misuse_in_specific_context",
    "failure_to_validate_tool_outputs",
    "insecure_interaction_or_execution",
    "procedural_deviation_or_inaction",
    "inefficient_or_wasteful_execution",
    "generation_of_harmful_offensive_content",
    "instruction_for_harmful_illegal_activity",
    "generation_of_malicious_executables",
    "unauthorized_information_disclosure",
    "provide_inaccurate_misleading_or_unverified_information",
)

HARM_TYPES = (
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
)

DECISIONS = ("allow", "block", "ask_confirm", "sanitize")
SAFETY_LABELS = ("safe", "unsafe")

RISK_SOURCE_LABELS = (NONE,) + RISK_SOURCES
FAILURE_MODE_LABELS = (NONE,) + FAILURE_MODES
HARM_TYPE_LABELS = (NONE,) + HARM_TYPES

AGENTDOG_ORIGINAL_NAMES = {
    "malicious_user_instruction_or_jailbreak": "Malicious User Instruction or Jailbreak",
    "direct_prompt_injection": "Direct Prompt Injection",
    "indirect_prompt_injection": "Indirect Prompt Injection",
    "unreliable_or_misinformation": "Unreliable or Misinformation",
    "tool_description_injection": "Tool Description Injection",
    "malicious_tool_execution": "Malicious Tool Execution",
    "corrupted_tool_feedback": "Corrupted Tool Feedback",
    "inherent_agent_failures": "Inherent Agent/LLM Failures",
    "unconfirmed_or_over_privileged_action": "Unconfirmed or Over-privileged Action",
    "flawed_planning_or_reasoning": "Flawed Planning or Reasoning",
    "incorrect_tool_parameters": "Incorrect Tool Parameters",
    "choosing_malicious_tool": "Choosing Malicious Tool",
    "tool_misuse_in_specific_context": "Tool Misuse in Specific Context",
    "failure_to_validate_tool_outputs": "Failure to Validate Tool Outputs",
    "insecure_interaction_or_execution": "Insecure Interaction or Execution",
    "procedural_deviation_or_inaction": "Procedural Deviation or Inaction",
    "inefficient_or_wasteful_execution": "Inefficient or Wasteful Execution",
    "generation_of_harmful_offensive_content": "Generation of Harmful/Offensive Content",
    "instruction_for_harmful_illegal_activity": "Instruction for Harmful/Illegal Activity",
    "generation_of_malicious_executables": "Generation of Malicious Executables",
    "unauthorized_information_disclosure": "Unauthorized Information Disclosure",
    "provide_inaccurate_misleading_or_unverified_information": (
        "Provide Inaccurate, Misleading, or Unverified Information"
    ),
    "privacy_and_confidentiality_harm": "Privacy and Confidentiality Harm",
    "financial_and_economic_harm": "Financial and Economic Harm",
    "security_and_system_integrity_harm": "Security and System Integrity Harm",
    "physical_and_health_harm": "Physical and Health Harm",
    "psychological_and_emotional_harm": "Psychological and Emotional Harm",
    "reputational_and_interpersonal_harm": "Reputation and Interpersonal Harm",
    "info_ecosystem_and_societal_harm": "Info-ecosystem and Societal Harm",
    "public_service_and_resource_harm": "Public Service and Resource Harm",
    "fairness_equity_and_allocative_harm": "Fairness, Equity, and Allocative Harm",
    "functional_and_opportunity_harm": "Functional and Opportunity Harm",
}

TAXONOMY_ALIASES = {
    # Historical TraceHound labels.
    "inherent_agent_or_llm_failures": "inherent_agent_failures",
    "choosing_malicious_tools": "choosing_malicious_tool",
    "generation_of_harmful_or_offensive_content": "generation_of_harmful_offensive_content",
    "instruction_for_harmful_or_illegal_activity": "instruction_for_harmful_illegal_activity",
    "privacy_confidentiality_harm": "privacy_and_confidentiality_harm",
    "financial_economic_harm": "financial_and_economic_harm",
    "security_system_integrity_harm": "security_and_system_integrity_harm",
    "physical_health_harm": "physical_and_health_harm",
    "psychological_emotional_harm": "psychological_and_emotional_harm",
    "reputational_interpersonal_harm": "reputational_and_interpersonal_harm",
    "info_ecosystem_societal_harm": "info_ecosystem_and_societal_harm",
    "public_service_resource_harm": "public_service_and_resource_harm",
    "fairness_equity_allocative_harm": "fairness_equity_and_allocative_harm",
    "functional_opportunity_harm": "functional_and_opportunity_harm",
    # Common variants in notes/spec drafts.
    "unconfirmed_or_overprivileged_action": "unconfirmed_or_over_privileged_action",
    "insecure_execution_or_interaction": "insecure_interaction_or_execution",
    "reputation_and_interpersonal_harm": "reputational_and_interpersonal_harm",
    "unreliable_or_mis_information": "unreliable_or_misinformation",
}


def is_valid_risk_source(label: str) -> bool:
    return normalize_risk_source(label) in RISK_SOURCE_LABELS


def is_valid_failure_mode(label: str) -> bool:
    return normalize_failure_mode(label) in FAILURE_MODE_LABELS


def is_valid_harm_type(label: str) -> bool:
    return normalize_harm_type(label) in HARM_TYPE_LABELS


def normalize_risk_source(label: str) -> str:
    return _normalize_label(label)


def normalize_failure_mode(label: str) -> str:
    return _normalize_label(label)


def normalize_harm_type(label: str) -> str:
    return _normalize_label(label)


def _normalize_label(label: str) -> str:
    normalized = _display_key(str(label))
    return TAXONOMY_ALIASES.get(normalized, DISPLAY_NAME_TO_LABEL.get(normalized, normalized))


def _display_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


DISPLAY_NAME_TO_LABEL = {
    _display_key(display): label
    for label, display in AGENTDOG_ORIGINAL_NAMES.items()
}
