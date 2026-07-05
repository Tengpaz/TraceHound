import json

from traceguard.claude_code_hooks import (
    build_tracehound_hook_settings,
    is_tracehound_hook,
    merge_tracehound_hooks,
    read_settings,
    write_settings,
)


def test_merge_tracehound_hooks_preserves_existing_settings_and_hooks(tmp_path):
    existing = {
        "permissions": {"allow": ["Bash(git status)"]},
        "env": {"FOO": "bar"},
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [{"type": "command", "command": "/usr/local/bin/user-audit", "timeout": 5}],
                }
            ],
            "SessionStart": [{"hooks": [{"type": "command", "command": "echo hello"}]}],
        },
    }
    fragment = build_tracehound_hook_settings(tracehound_root=tmp_path, server_url="http://127.0.0.1:8000")

    merged = merge_tracehound_hooks(existing, fragment)

    assert merged["permissions"] == existing["permissions"]
    assert merged["env"] == existing["env"]
    assert merged["hooks"]["SessionStart"] == existing["hooks"]["SessionStart"]
    assert merged["hooks"]["PreToolUse"][0]["hooks"][0]["command"] == "/usr/local/bin/user-audit"
    tracehound_hooks = [
        hook
        for group in merged["hooks"]["PreToolUse"]
        for hook in group.get("hooks", [])
        if is_tracehound_hook(hook)
    ]
    assert len(tracehound_hooks) == 1


def test_merge_tracehound_hooks_is_idempotent_and_refreshes_old_tracehound_hook(tmp_path):
    first_fragment = build_tracehound_hook_settings(tracehound_root=tmp_path / "old", server_url="http://127.0.0.1:8000")
    second_fragment = build_tracehound_hook_settings(tracehound_root=tmp_path / "new", server_url="http://127.0.0.1:9000")
    first = merge_tracehound_hooks({}, first_fragment)
    second = merge_tracehound_hooks(first, second_fragment)
    third = merge_tracehound_hooks(second, second_fragment)

    for settings in (second, third):
        all_tracehound = [
            hook
            for groups in settings["hooks"].values()
            for group in groups
            for hook in group.get("hooks", [])
            if is_tracehound_hook(hook)
        ]
        assert len(all_tracehound) == 3
        assert all("127.0.0.1:9000" in hook["command"] for hook in all_tracehound)
        assert all("/old/" not in hook["command"] for hook in all_tracehound)


def test_http_hook_refreshes_previous_command_hooks(tmp_path):
    command_fragment = build_tracehound_hook_settings(tracehound_root=tmp_path, hook_type="command")
    http_fragment = build_tracehound_hook_settings(
        tracehound_root=tmp_path,
        server_url="http://127.0.0.1:8000",
        hook_type="http",
    )

    merged = merge_tracehound_hooks(merge_tracehound_hooks({}, command_fragment), http_fragment)
    all_tracehound = [
        hook
        for groups in merged["hooks"].values()
        for group in groups
        for hook in group.get("hooks", [])
        if is_tracehound_hook(hook)
    ]

    assert len(all_tracehound) == 3
    assert all(hook["type"] == "http" for hook in all_tracehound)
    assert all(hook["url"].endswith("/api/guardrail/claude-code?mode=layered&judge=heuristic") for hook in all_tracehound)


def test_read_write_settings_creates_backup(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"hooks": {"SessionStart": []}}), encoding="utf-8")

    backup_path = write_settings(settings_path, {"hooks": {"Stop": []}}, backup=True)

    assert backup_path is not None
    assert backup_path.exists()
    assert read_settings(settings_path) == {"hooks": {"Stop": []}}
    assert json.loads(backup_path.read_text(encoding="utf-8")) == {"hooks": {"SessionStart": []}}
