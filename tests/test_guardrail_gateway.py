import json

from traceguard.guardrail_gateway import evaluate_guardrail_event, normalize_guardrail_event


def test_claude_code_pre_tool_use_blocks_high_risk_bash_with_exit_code():
    result = evaluate_guardrail_event(
        {
            "hook_event_name": "PreToolUse",
            "session_id": "claude-session-1",
            "tool_name": "Bash",
            "tool_input": {"command": "rm -rf ./build"},
            "transcript_path": "/tmp/missing-transcript.jsonl",
        },
        platform="claude-code",
    )

    assert result["platform"] == "claude-code"
    assert result["event_type"] == "pre_tool_use"
    assert result["decision"] == "ask_confirm"
    assert result["allow"] is False
    assert result["adapter"]["exit_code"] == 2
    assert result["adapter"]["json"]["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    assert result["adapter"]["json"]["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_generic_pre_reply_detects_unsafe_trajectory_tool_use():
    result = evaluate_guardrail_event(
        {
            "platform": "codex",
            "event_type": "pre_reply",
            "session_id": "codex-session-1",
            "trajectory": [
                {"step": 1, "role": "user", "type": "message", "content": "Clean the repo."},
                {
                    "step": 2,
                    "role": "agent",
                    "type": "tool_call",
                    "tool_name": "shell.run",
                    "arguments": {"command": "rm -rf ."},
                },
            ],
            "final_response": "Done.",
        },
        platform="codex",
    )

    assert result["event_type"] == "pre_reply"
    assert result["decision"] == "block"
    assert result["report"]["failure_mode"] == "insecure_interaction_or_execution"


def test_claude_code_stop_event_can_block_unsafe_final_response():
    result = evaluate_guardrail_event(
        {
            "hook_event_name": "Stop",
            "session_id": "claude-session-3",
            "trajectory": [
                {"step": 1, "role": "user", "type": "message", "content": "Clean the repo."},
                {
                    "step": 2,
                    "role": "agent",
                    "type": "tool_call",
                    "tool_name": "Bash",
                    "arguments": {"command": "rm -rf ."},
                },
            ],
            "response": "Done.",
        },
        platform="claude-code",
    )

    assert result["event_type"] == "pre_reply"
    assert result["decision"] == "block"
    assert result["adapter"]["exit_code"] == 2
    assert result["adapter"]["json"]["decision"] == "block"
    assert "reason" in result["adapter"]["json"]


def test_transcript_path_is_tolerantly_parsed(tmp_path):
    transcript = tmp_path / "session.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"role": "user", "content": "Inspect the workspace."}),
                json.dumps({"role": "assistant", "tool_name": "Read", "input": {"file_path": "README.md"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    normalized = normalize_guardrail_event(
        {
            "hook_event_name": "PreToolUse",
            "session_id": "claude-session-2",
            "tool_name": "Read",
            "tool_input": {"file_path": ".env"},
            "transcript_path": str(transcript),
        },
        platform="claude-code",
    )

    assert normalized["case"].id == "claude-session-2"
    assert len(normalized["case"].trajectory) >= 2
    assert normalized["candidate_action"].tool_name == "file.read"
