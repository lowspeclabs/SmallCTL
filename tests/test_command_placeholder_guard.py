from __future__ import annotations

from types import SimpleNamespace

import pytest

from smallctl.graph.tool_loop_guards import _detect_command_placeholder
from smallctl.graph.state import PendingToolCall


def _pending(tool_name: str, command: str) -> PendingToolCall:
    return PendingToolCall(
        tool_name=tool_name,
        args={"command": command},
        raw_arguments='{"command": "' + command + '"}',
    )


@pytest.mark.parametrize(
    "tool_name",
    ["shell_exec", "ssh_exec"],
)
def test_detects_angle_bracket_placeholder_in_command(tool_name: str) -> None:
    pending = _pending(tool_name, "docker run <original_docker_run_command>")
    harness = SimpleNamespace()

    result = _detect_command_placeholder(harness, pending)

    assert result is not None
    message, details = result
    assert "placeholder" in message.lower()
    assert details["placeholder"] == "<original_docker_run_command>"
    assert details["reason"] == "command_contains_placeholder"


def test_ignores_commands_without_placeholders() -> None:
    pending = _pending("ssh_exec", "docker ps")
    harness = SimpleNamespace()

    assert _detect_command_placeholder(harness, pending) is None


def test_ignores_redirections_and_process_substitution() -> None:
    """Legitimate shell syntax like `< /dev/null` or `<(cmd)` must not be flagged."""
    pending = _pending("shell_exec", "cat < /dev/null && echo <( date )")
    harness = SimpleNamespace()

    assert _detect_command_placeholder(harness, pending) is None


def test_ignores_non_shell_tools() -> None:
    pending = PendingToolCall(
        tool_name="file_write",
        args={"path": "<placeholder>"},
        raw_arguments='{"path": "<placeholder>"}',
    )
    harness = SimpleNamespace()

    assert _detect_command_placeholder(harness, pending) is None
