from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from smallctl.harness.tool_result_rendering import (
    _format_recovery_hint,
    build_tool_result_message,
)
from smallctl.models.tool_result import ToolEnvelope


class _FakeArtifactStore:
    def compact_tool_message(self, artifact: Any, result: Any, **kwargs: Any) -> str:  # noqa: ARG002
        return str(result.error or result.output or "")


class _FakeHarness:
    def __init__(self) -> None:
        self.state = MagicMock()
        self.state.run_brief.original_task = "test task"
        self.state.step_count = 1
        self.artifact_store = _FakeArtifactStore()
        self.context_policy = MagicMock()
        self.context_policy.tool_result_inline_token_limit = 250

    def _current_user_task(self) -> str:
        return "test task"

    def _runlog(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        pass


class _FakeService:
    def __init__(self) -> None:
        self.harness = _FakeHarness()


def test_format_recovery_hint_empty_when_no_metadata() -> None:
    assert _format_recovery_hint({}) == ""


def test_format_recovery_hint_with_next_required_action_dict() -> None:
    metadata = {
        "next_required_action": {
            "tool_name": "ssh_exec",
            "required_arguments": {"command": "echo ok"},
            "notes": ["Run validator first."],
        }
    }
    hint = _format_recovery_hint(metadata)
    assert "Recovery hint:" in hint
    assert "Next required action:" in hint
    assert '"tool_name": "ssh_exec"' in hint
    assert "Run validator first." in hint


def test_format_recovery_hint_with_next_required_action_str() -> None:
    metadata = {"next_required_action": "fresh clone or clean reset"}
    hint = _format_recovery_hint(metadata)
    assert "Next required action: fresh clone or clean reset" in hint


def test_format_recovery_hint_with_next_required_tool() -> None:
    metadata = {
        "next_required_tool": {
            "tool_name": "file_read",
            "required_arguments": {"path": "/etc/apt/sources.list"},
        }
    }
    hint = _format_recovery_hint(metadata)
    assert "Recovery hint:" in hint
    assert "Next required tool:" in hint
    assert '"tool_name": "file_read"' in hint


def test_format_recovery_hint_with_both() -> None:
    metadata = {
        "next_required_action": {"tool_name": "ssh_exec"},
        "next_required_tool": {"tool_name": "file_read"},
    }
    hint = _format_recovery_hint(metadata)
    assert "Next required action:" in hint
    assert "Next required tool:" in hint


@pytest.mark.asyncio
async def test_build_tool_result_message_appends_recovery_hint_on_failure() -> None:
    service = _FakeService()
    result = ToolEnvelope(
        success=False,
        error="Blocked by guard.",
        metadata={
            "next_required_action": {
                "tool_name": "ssh_exec",
                "required_arguments": {"command": "validate"},
            }
        },
    )

    message = await build_tool_result_message(
        service,
        tool_name="shell_exec",
        result=result,
        artifact=None,
        tool_call_id="call_123",
    )

    content = str(message.content or "")
    assert message.role == "tool"
    assert "Blocked by guard." in content
    assert "Recovery hint:" in content
    assert "validate" in content


@pytest.mark.asyncio
async def test_build_tool_result_message_skips_recovery_hint_on_success() -> None:
    service = _FakeService()
    result = ToolEnvelope(
        success=True,
        output="ok",
        metadata={
            "next_required_action": {
                "tool_name": "ssh_exec",
                "required_arguments": {"command": "validate"},
            }
        },
    )

    message = await build_tool_result_message(
        service,
        tool_name="shell_exec",
        result=result,
        artifact=None,
        tool_call_id="call_123",
    )

    content = str(message.content or "")
    assert message.role == "tool"
    assert "Recovery hint:" not in content


@pytest.mark.asyncio
async def test_build_tool_result_message_appends_dry_run_hint_on_success() -> None:
    service = _FakeService()
    result = ToolEnvelope(
        success=True,
        output={"stdout": "Dry run only. Re-run with --execute to apply.", "stderr": "", "exit_code": 0},
        metadata={"dry_run_hint": "This command performed a dry run only. Re-run with --execute."},
    )

    message = await build_tool_result_message(
        service,
        tool_name="shell_exec",
        result=result,
        artifact=None,
        tool_call_id="call_123",
    )

    content = str(message.content or "")
    assert message.role == "tool"
    assert "Hint:" in content
    assert "dry run" in content.lower()
