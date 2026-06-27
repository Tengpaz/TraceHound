"""AgentDoG-aligned taxonomy labels used by TraceHound."""

from __future__ import annotations

NONE = "none"

RISK_SOURCES = (
    "malicious_user_instruction_or_jailbreak",
    "direct_prompt_injection",
    "indirect_prompt_injection",
    "unreliable_or_misinformation",
    "tool_description_injection",
    "malicious_tool_execution",
    "corrupted_tool_feedback",
    "inherent_agent_or_llm_failures",
)

FAILURE_MODES = (
    "unconfirmed_or_over_privileged_action",
    "flawed_planning_or_reasoning",
    "incorrect_tool_parameters",
    "choosing_malicious_tools",
    "tool_misuse_in_specific_context",
    "failure_to_validate_tool_outputs",
    "insecure_interaction_or_execution",
    "procedural_deviation_or_inaction",
    "inefficient_or_wasteful_execution",
    "generation_of_harmful_or_offensive_content",
    "instruction_for_harmful_or_illegal_activity",
    "generation_of_malicious_executables",
    "unauthorized_information_disclosure",
    "provide_inaccurate_misleading_or_unverified_information",
)

HARM_TYPES = (
    "privacy_confidentiality_harm",
    "financial_economic_harm",
    "security_system_integrity_harm",
    "physical_health_harm",
    "psychological_emotional_harm",
    "reputational_interpersonal_harm",
    "info_ecosystem_societal_harm",
    "public_service_resource_harm",
    "fairness_equity_allocative_harm",
    "functional_opportunity_harm",
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
    "inherent_agent_or_llm_failures": "Inherent Agent or LLM Failures",
    "unconfirmed_or_over_privileged_action": "Unconfirmed or Over-privileged Action",
    "flawed_planning_or_reasoning": "Flawed Planning or Reasoning",
    "incorrect_tool_parameters": "Incorrect Tool Parameters",
    "choosing_malicious_tools": "Choosing Malicious Tools",
    "tool_misuse_in_specific_context": "Tool Misuse in a Specific Context",
    "failure_to_validate_tool_outputs": "Failure to Validate Tool Outputs",
    "insecure_interaction_or_execution": "Insecure Interaction or Execution",
    "procedural_deviation_or_inaction": "Procedural Deviation or Inaction",
    "inefficient_or_wasteful_execution": "Inefficient or Wasteful Execution",
    "generation_of_harmful_or_offensive_content": "Generation of Harmful or Offensive Content",
    "instruction_for_harmful_or_illegal_activity": "Instruction for Harmful or Illegal Activity",
    "generation_of_malicious_executables": "Generation of Malicious Executables",
    "unauthorized_information_disclosure": "Unauthorized Information Disclosure",
    "provide_inaccurate_misleading_or_unverified_information": (
        "Provide Inaccurate, Misleading, or Unverified Information"
    ),
    "privacy_confidentiality_harm": "Privacy & Confidentiality Harm",
    "financial_economic_harm": "Financial & Economic Harm",
    "security_system_integrity_harm": "Security & System Integrity Harm",
    "physical_health_harm": "Physical & Health Harm",
    "psychological_emotional_harm": "Psychological & Emotional Harm",
    "reputational_interpersonal_harm": "Reputational & Interpersonal Harm",
    "info_ecosystem_societal_harm": "Info-ecosystem & Societal Harm",
    "public_service_resource_harm": "Public Service & Resource Harm",
    "fairness_equity_allocative_harm": "Fairness, Equity, and Allocative Harm",
    "functional_opportunity_harm": "Functional & Opportunity Harm",
}


def is_valid_risk_source(label: str) -> bool:
    return label in RISK_SOURCE_LABELS


def is_valid_failure_mode(label: str) -> bool:
    return label in FAILURE_MODE_LABELS


def is_valid_harm_type(label: str) -> bool:
    return label in HARM_TYPE_LABELS

