from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path
from typing import Any

import pytest

from smallctl.state import LoopState
from smallctl.tools import fs_mutations, network_ssh_helpers, ssh_files, shell
from smallctl.tools.base import ToolSpec, build_tool_schema
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.registry import ToolRegistry
from smallctl.tools import network_interactive_sessions
from smallctl.tools.shell_foreground import (
    _is_safe_compile_lint_command,
    shell_exec_foreground,
)
from smallctl.tools.ssh_parsing import normalize_ssh_arguments, normalize_ssh_target


class _DenyingApprovalHarness:
    def __init__(self) -> None:
        self.event_handler = object()
        self.approval_calls: list[dict[str, Any]] = []

    async def request_shell_approval(self, **kwargs: Any) -> bool:
        self.approval_calls.append(kwargs)
        return False


class _ApprovingApprovalHarness(_DenyingApprovalHarness):
    async def request_shell_approval(self, **kwargs: Any) -> bool:
        self.approval_calls.append(kwargs)
        return True


# --- H9: shell_exec(job_id=...) polling reaches the handler -------------------


def _make_shell_exec_dispatcher() -> ToolDispatcher:
    state = LoopState(cwd="/tmp")
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="shell_exec",
            description="run shell command",
            schema=build_tool_schema(
                properties={
                    "command": {"type": "string"},
                    "job_id": {"type": "string"},
                    "background": {"type": "boolean"},
                    "timeout_sec": {"type": "integer"},
                },
                required=[],
            ),
            handler=lambda **kwargs: shell.shell_exec(state=state, harness=None, **kwargs),
        )
    )
    return ToolDispatcher(registry, phase="execute")


async def _dispatch_shell_exec(arguments: dict[str, Any]) -> Any:
    dispatcher = _make_shell_exec_dispatcher()
    return await dispatcher.dispatch("shell_exec", arguments)


def test_h9_shell_exec_job_id_poll_reaches_handler() -> None:
    result = asyncio.run(_dispatch_shell_exec({"job_id": "job_123"}))

    assert result.success is False
    assert result.metadata.get("validation_error") != "empty_command"
    assert "Unknown job id" in str(result.error)


def test_h9_shell_exec_empty_command_and_job_id_still_rejected() -> None:
    result = asyncio.run(_dispatch_shell_exec({"command": "  ", "job_id": ""}))

    assert result.success is False
    assert result.metadata.get("validation_error") == "empty_command"


# --- H12: compile/lint auto-approval bypass -----------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "ruff check && rm -rf ~",
        "mypy src; rm -rf /tmp/i",
        "python3 -m py_compile x.py && git reset --hard HEAD~5",
        "cd / && ruff check",
        "ruff check --fix src",
        "ruff check src > /tmp/out.txt",
        "ruff check $(whoami)",
        "ruff check `whoami`",
        "ruff check src &",
        "ruff check | sh",
        "ruff check src && ",
        "ruff format src",
        "python3 -m py_compile",
    ],
)
def test_h12_unsafe_or_mixed_commands_fail_closed(command: str) -> None:
    assert _is_safe_compile_lint_command(command, cwd="/tmp") is False


@pytest.mark.parametrize(
    "command",
    [
        "ruff check src",
        "mypy src && ruff check",
        "mypy src",
        "flake8 src",
        "shellcheck deploy.sh",
    ],
)
def test_h12_pure_lint_commands_auto_approve(command: str) -> None:
    assert _is_safe_compile_lint_command(command, cwd="/tmp") is True


def test_h12_workspace_bounded_cd_allowed(tmp_path: Path) -> None:
    subdir = tmp_path / "pkg"
    subdir.mkdir()
    command = f"cd {subdir.name} && ruff check"
    assert _is_safe_compile_lint_command(command, cwd=str(tmp_path)) is True


def test_h12_cd_outside_workspace_fails_closed(tmp_path: Path) -> None:
    assert _is_safe_compile_lint_command("cd .. && ruff check", cwd=str(tmp_path / "sub")) is False
    assert _is_safe_compile_lint_command("cd / && ruff check", cwd=str(tmp_path)) is False


def test_h12_mixed_lint_and_mutation_reaches_approval_fn() -> None:
    harness = _DenyingApprovalHarness()
    state = LoopState(cwd="/tmp")

    async def _create_process(**kwargs: Any) -> Any:
        raise AssertionError("process must not be spawned when approval is denied")

    result = asyncio.run(
        shell_exec_foreground(
            "ruff check && touch /tmp/x",
            state=state,
            timeout_sec=30,
            harness=harness,
            create_process=_create_process,
        )
    )

    assert len(harness.approval_calls) == 1
    assert result["success"] is False
    assert result.get("status") == "denied"
    assert result["metadata"].get("approval_denied") is True


# --- H13: SSH argument injection ----------------------------------------------


def test_h13_dash_prefixed_host_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_ssh_target(host="-oProxyCommand=touch /tmp/pwned")
    with pytest.raises(ValueError):
        normalize_ssh_arguments({"host": "-oProxyCommand=touch /tmp/pwned", "command": "id"})


def test_h13_dash_prefixed_user_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_ssh_target(host="example.test", user="-oProxyCommand=touch /tmp/pwned")
    with pytest.raises(ValueError):
        normalize_ssh_arguments({"host": "example.test", "user": "-v", "command": "id"})


def test_h13_dash_prefixed_identity_file_rejected() -> None:
    with pytest.raises(ValueError):
        normalize_ssh_arguments(
            {"host": "example.test", "identity_file": "-oProxyCommand=touch /tmp/pwned", "command": "id"}
        )


def test_h13_build_ssh_command_raises_for_dash_host() -> None:
    with pytest.raises(ValueError):
        network_ssh_helpers.build_ssh_command(
            host="-oProxyCommand=touch /tmp/pwned",
            command="id",
            user=None,
            port=22,
            identity_file=None,
            password=None,
        )


def test_h13_built_argv_contains_double_dash_before_target() -> None:
    cmd, _env, password_file_path = network_ssh_helpers.build_ssh_command(
        host="example.test",
        command="whoami",
        user="root",
        port=22,
        identity_file=None,
        password=None,
    )
    assert password_file_path is None
    tokens = shlex.split(cmd)
    assert "--" in tokens
    separator_index = tokens.index("--")
    assert tokens[separator_index + 1] == "root@example.test"
    assert tokens[separator_index + 2] == "whoami"


def test_h13_ssh_file_tool_surfaces_invalid_ssh_target() -> None:
    result = asyncio.run(
        ssh_files.ssh_file_read(
            path="/etc/hostname",
            host="-oProxyCommand=touch /tmp/pwned",
            state=LoopState(cwd="/tmp"),
        )
    )
    assert result["success"] is False
    assert result["metadata"].get("reason") == "invalid_ssh_target"


# --- H14: approval gate honored by remote-mutation and file tools --------------


def test_h14_ssh_file_write_denied_by_approval_spawns_no_ssh(monkeypatch) -> None:
    spawned: list[dict[str, Any]] = []

    async def _fake_run_ssh_command(**kwargs: Any) -> dict[str, Any]:
        spawned.append(kwargs)
        raise AssertionError("ssh subprocess must not be spawned when approval is denied")

    monkeypatch.setattr(ssh_files.network, "run_ssh_command", _fake_run_ssh_command)
    harness = _DenyingApprovalHarness()

    result = asyncio.run(
        ssh_files.ssh_file_write(
            target="root@192.0.2.10",
            path="/etc/cron.d/pwn",
            content="* * * * * root id\n",
            state=LoopState(cwd="/tmp"),
            harness=harness,
        )
    )

    assert result["success"] is False
    assert result.get("status") == "denied"
    assert result["metadata"].get("approval_denied") is True
    assert spawned == []
    assert len(harness.approval_calls) == 1


def test_h14_ssh_file_read_stays_ungated(monkeypatch) -> None:
    payload = {"ok": True, "path": "/etc/hostname", "content": "host\n", "sha256": "x"}

    async def _fake_run_ssh_command(**kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "output": {"stdout": json.dumps(payload), "stderr": "", "exit_code": 0},
            "error": None,
            "metadata": {},
        }

    monkeypatch.setattr(ssh_files.network, "run_ssh_command", _fake_run_ssh_command)
    harness = _DenyingApprovalHarness()

    result = asyncio.run(
        ssh_files.ssh_file_read(
            target="root@192.0.2.10",
            path="/etc/hostname",
            state=LoopState(cwd="/tmp"),
            harness=harness,
        )
    )

    assert result["success"] is True
    assert harness.approval_calls == []


def test_h14_ssh_file_patch_dry_run_stays_ungated(monkeypatch) -> None:
    payload = {
        "ok": True,
        "path": "/etc/app.conf",
        "dry_run": True,
        "matched_region_previews": [],
        "actual_occurrences": 1,
        "expected_occurrences": 1,
    }

    async def _fake_run_ssh_command(**kwargs: Any) -> dict[str, Any]:
        return {
            "success": True,
            "output": {"stdout": json.dumps(payload), "stderr": "", "exit_code": 0},
            "error": None,
            "metadata": {},
        }

    monkeypatch.setattr(ssh_files.network, "run_ssh_command", _fake_run_ssh_command)
    harness = _DenyingApprovalHarness()

    result = asyncio.run(
        ssh_files.ssh_file_patch(
            target="root@192.0.2.10",
            path="/etc/app.conf",
            target_text="old",
            replacement_text="new",
            dry_run=True,
            state=LoopState(cwd="/tmp"),
            harness=harness,
        )
    )

    assert result["success"] is True
    assert harness.approval_calls == []


def test_h14_ssh_session_start_denied_by_approval_spawns_no_process(monkeypatch) -> None:
    spawned: list[dict[str, Any]] = []

    async def _fake_create_process(**kwargs: Any) -> Any:
        spawned.append(kwargs)
        raise AssertionError("ssh subprocess must not be spawned when approval is denied")

    monkeypatch.setattr(
        network_interactive_sessions,
        "_build_ssh_command",
        lambda **kwargs: ("ssh -- root@192.0.2.10 bash", None, None),
    )
    from smallctl.tools import shell as shell_module

    monkeypatch.setattr(shell_module, "create_process", _fake_create_process)
    harness = _DenyingApprovalHarness()

    result = asyncio.run(
        network_interactive_sessions.ssh_session_start(
            host="192.0.2.10",
            user="root",
            command="bash",
            state=LoopState(cwd="/tmp"),
            harness=harness,
        )
    )

    assert result["success"] is False
    assert result.get("status") == "denied"
    assert result["metadata"].get("approval_denied") is True
    assert spawned == []
    assert len(harness.approval_calls) == 1
    assert network_interactive_sessions._SSH_INTERACTIVE_SESSIONS == {}


def test_h14_file_delete_denied_by_approval_keeps_file(tmp_path: Path) -> None:
    target = tmp_path / "keep.txt"
    target.write_text("precious", encoding="utf-8")
    harness = _DenyingApprovalHarness()

    result = asyncio.run(
        fs_mutations.file_delete(
            path="keep.txt",
            cwd=str(tmp_path),
            state=LoopState(cwd=str(tmp_path)),
            harness=harness,
        )
    )

    assert result["success"] is False
    assert result.get("status") == "denied"
    assert result["metadata"].get("approval_denied") is True
    assert result["metadata"].get("requires_approval") is True
    assert target.exists()
    assert len(harness.approval_calls) == 1


def test_h14_file_delete_without_approval_channel_preserves_flag(tmp_path: Path) -> None:
    target = tmp_path / "gone.txt"
    target.write_text("trash", encoding="utf-8")

    result = asyncio.run(
        fs_mutations.file_delete(
            path="gone.txt",
            cwd=str(tmp_path),
            state=LoopState(cwd=str(tmp_path)),
        )
    )

    assert result["success"] is True
    assert result["metadata"].get("requires_approval") is False
    assert not target.exists()
