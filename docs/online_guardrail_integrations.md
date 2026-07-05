# Online Guardrail Integrations

TraceHound can run as a trajectory-level guardrail service for real agent
runtimes. The stable integration contract is:

1. The agent sends a JSON event before a tool call, after a tool observation,
   or before final response delivery.
2. TraceHound normalizes the event into `TrajectoryCase`.
3. TraceHound returns `allow`, `block`, `ask_confirm`, or `sanitize` with a
   full `RiskReport`.

Start the server:

```bash
python scripts/serve_demo.py --host 127.0.0.1 --port 8000
```

Main endpoint:

```text
POST http://127.0.0.1:8000/api/guardrail/event
```

Claude Code HTTP hook endpoint:

```text
POST http://127.0.0.1:8000/api/guardrail/claude-code?mode=layered&judge=local-binary
```

Request shape:

```json
{
  "platform": "claude-code | codex | openclaw | generic | auto",
  "event_type": "pre_tool_use | post_tool_use | pre_reply | auto",
  "mode": "rules | compressed | layered",
  "judge": "heuristic | local-binary | local | api | hybrid",
  "event": {
    "session_id": "session-1",
    "trajectory": [],
    "tool_name": "Bash",
    "tool_input": {"command": "rm -rf ./build"}
  }
}
```

Response shape:

```json
{
  "allow": false,
  "decision": "ask_confirm",
  "reason": "Candidate action uses a high-privilege tool and should be confirmed before execution.",
  "report": {},
  "adapter": {}
}
```

## Claude Code

Claude Code exposes lifecycle hooks such as `PreToolUse`, `PostToolUse`, and
`Stop`. TraceHound supports command hooks through:

```bash
tracehound-guardrail-hook \
  --platform claude-code \
  --event-type auto \
  --mode layered \
  --judge local-binary \
  --server-url http://127.0.0.1:8000 \
  --adapter-json
```

Install the hooks with the merge script instead of copying over an existing
settings file:

```bash
python scripts/install_claude_code_hooks.py \
  --settings .claude/settings.json \
  --tracehound-root /Users/a1234/Documents/Code/TraceHound \
  --server-url http://127.0.0.1:8000 \
  --python-command "conda run -n tracehound python"
```

The script preserves existing Claude Code settings, writes a timestamped
backup, removes only old TraceHound-managed hooks, and appends the current
TraceHound hook groups. Use `--global-settings` to update
`~/.claude/settings.json` instead of a project-local `.claude/settings.json`.

The raw fragment in `integrations/claude_code/settings.tracehound.example.json`
is kept for inspection and manual merge only. Do not copy it over an existing
settings file unless you intentionally want to replace that file.

For `PreToolUse`, TraceHound emits Claude Code's native `permissionDecision:
deny` JSON when it decides `block` or `ask_confirm`.
For `Stop`/pre-reply events, TraceHound emits `decision: block` so Claude can
continue instead of delivering an unsafe final response.

If a Claude Code version or deployment prefers exit-code blocking, replace
`--adapter-json` with `--claude-exit-codes`. Do not use both modes together.

Use `Stop`/pre-reply checks for AgentDoG-style online monitoring because they
see the full trajectory and final response draft. Use `PreToolUse` checks for
fast deterministic blocking of risky shell/file/network actions. When
`judge=local-binary`, rules still run first; rule-allowed events are then sent
to the local TraceHound-Base Qwen3.5-0.8B binary checkpoint.

## Codex

Codex surfaces do not all expose the same public hook mechanism. Treat
TraceHound as a guardrail wrapper around whichever tool-dispatch or final-reply
boundary your Codex deployment exposes.

Example wrapper:

```bash
python examples/integrations/codex_guardrail_wrapper.py
```

Minimal pre-reply event:

```bash
python scripts/guardrail_hook.py --platform codex --event-type pre_reply --judge local-binary < event.json
```

If a Codex environment only exposes a final transcript, call TraceHound at
pre-reply time. If it exposes tool dispatch, call TraceHound before high-risk
tools as well.

## OpenClaw

OpenClaw-style agents can integrate TraceHound as middleware around tool
dispatch and final response delivery:

```python
from examples.integrations.openclaw_guardrail_middleware import TraceHoundGuardrail

guard = TraceHoundGuardrail("http://127.0.0.1:8000/api/guardrail/event")
guard.before_tool(session_id, trajectory, tool_name, tool_input)
guard.after_tool(session_id, trajectory, tool_name, tool_output)
guard.before_final_response(session_id, trajectory, final_response)
```

Recommended deployment:

- `pre_tool_use`: rules or layered mode for high-risk tools.
- `post_tool_use`: sanitize prompt-injection-like tool feedback.
- `pre_reply`: compressed or layered mode with API/local guard model for the
  full trajectory decision.

## Offline Hook Mode

Hooks can run without the FastAPI server:

```bash
python scripts/guardrail_hook.py --platform generic --event-type pre_reply --judge local-binary < event.json
```

Offline mode uses the local heuristic judge by default. Add `--judge local-binary`
to use `models/TraceHound-Base-Qwen3.5-0.8B-Binary`, or add `--judge api` /
`--judge hybrid` to use the configured OpenAI-compatible endpoint.

For platform-native output without the full TraceHound debug envelope:

```bash
python scripts/guardrail_hook.py --platform claude-code --event-type auto --adapter-json < claude-event.json
```
