from __future__ import annotations

from types import SimpleNamespace
import asyncio

from smallctl.state import LoopState
from smallctl.harness.tool_dispatch import dispatch_tool_call
from smallctl.tools.dispatcher_tool_guards import (
    _guard_remote_file_tool_request,
    _guard_remote_shell_tool_request,
)


def _make_state(*, task_mode: str = "", cwd: str = "/tmp") -> LoopState:
    state = LoopState(cwd=cwd)
    state.task_mode = task_mode
    return state


def test_remote_guard_blocks_local_file_read_when_remote_execute() -> None:
    """When task_mode is remote_execute, local file_read on remote-looking paths should be blocked."""
    state = _make_state(task_mode="remote_execute")
    result = _guard_remote_file_tool_request(
        "file_read",
        {"path": "/root/.ssh/authorized_keys"},
        state=state,
        ssh_available=True,
    )
    assert result is not None
    assert result.success is False
    assert "remote" in result.error.lower()


def test_remote_guard_allows_local_file_read_when_local_execute() -> None:
    """When task_mode is local_execute, local file_read should be allowed even for /root paths."""
    state = _make_state(task_mode="local_execute")
    result = _guard_remote_file_tool_request(
        "file_read",
        {"path": "/root/.ssh/authorized_keys"},
        state=state,
        ssh_available=True,
    )
    assert result is None


def test_remote_guard_allows_dir_list_when_local_execute() -> None:
    """When task_mode is local_execute, dir_list should be allowed even for /root paths."""
    state = _make_state(task_mode="local_execute")
    result = _guard_remote_file_tool_request(
        "dir_list",
        {"path": "/root/.ssh"},
        state=state,
        ssh_available=True,
    )
    assert result is None


def test_remote_guard_allows_shell_exec_when_local_execute() -> None:
    """When task_mode is local_execute, shell_exec should be allowed even for remote-looking commands."""
    state = _make_state(task_mode="local_execute")
    result = _guard_remote_shell_tool_request(
        "ls -la /root/.ssh/",
        state=state,
        ssh_available=True,
    )
    assert result is None


def test_remote_guard_blocks_shell_exec_when_remote_execute() -> None:
    """When task_mode is remote_execute, shell_exec with remote-looking commands should be blocked."""
    state = _make_state(task_mode="remote_execute")
    # Need remote scope active for the guard to fire
    state.scratchpad["_last_task_handoff"] = {"task_mode": "remote_execute"}
    result = _guard_remote_shell_tool_request(
        "systemctl status nginx",
        state=state,
        ssh_available=True,
    )
    assert result is not None
    assert result.success is False
    assert "remote" in result.error.lower()


def test_remote_guard_allows_file_read_for_chat_mode() -> None:
    """Chat mode should not block local file tools."""
    state = _make_state(task_mode="chat")
    result = _guard_remote_file_tool_request(
        "file_read",
        {"path": "/root/.ssh/authorized_keys"},
        state=state,
        ssh_available=True,
    )
    assert result is None


def test_remote_guard_allows_file_read_for_analysis_mode() -> None:
    """Analysis mode should not block local file tools."""
    state = _make_state(task_mode="analysis")
    result = _guard_remote_file_tool_request(
        "file_read",
        {"path": "/root/.ssh/authorized_keys"},
        state=state,
        ssh_available=True,
    )
    assert result is None


def test_remote_guard_allows_file_read_for_plan_only_mode() -> None:
    """Plan-only mode should not block local file tools."""
    state = _make_state(task_mode="plan_only")
    result = _guard_remote_file_tool_request(
        "file_read",
        {"path": "/root/.ssh/authorized_keys"},
        state=state,
        ssh_available=True,
    )
    assert result is None


def test_remote_guard_allows_file_read_for_debug_inspect_mode() -> None:
    """Debug-inspect mode should not block local file tools."""
    state = _make_state(task_mode="debug_inspect")
    result = _guard_remote_file_tool_request(
        "file_read",
        {"path": "/root/.ssh/authorized_keys"},
        state=state,
        ssh_available=True,
    )
    assert result is None


def test_direct_ssh_exec_rejects_local_workspace_command() -> None:
    state = _make_state(task_mode="local_execute", cwd="/workspace")
    state.run_brief.original_task = (
        "read /workspace/temp/proxmox-manager/AGENTS.md and run the local proxmox cli against its API on 192.168.1.74"
    )

    class _Registry:
        def names(self) -> set[str]:
            return {"ssh_exec", "shell_exec"}

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        _runlog=lambda *_args, **_kwargs: None,
    )
    result = asyncio.run(
        dispatch_tool_call(
            harness,
            "ssh_exec",
            {"host": "192.168.1.74", "command": "/workspace/temp/proxmox-manager/proxmox doctor"},
        )
    )

    assert result.success is False
    assert result.metadata["reason"] == "local_command_requires_shell_exec"
    assert result.metadata["required_tool"] == "shell_exec"
