from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smallctl.graph.model_stream_resolution_support import (
    _chunk_error_failure_message,
    _chunk_error_failure_type,
)
from smallctl.state import LoopState
from smallctl.tools import network_interactive_sessions
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.register import build_registry
from smallctl.tools.shell_foreground import _is_safe_compile_lint_command
from smallctl.tools.shell_support_delete_guards import (
    _shell_workspace_destructive_delete_guard,
)


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


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None


class _FakeStream:
    async def read(self, _size: int = -1) -> bytes:
        await asyncio.sleep(0)
        return b""


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stdin = _FakeStdin()
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        await asyncio.sleep(0)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _install_fake_session(
    session_id: str, *, harness: Any
) -> tuple[dict[str, Any], _FakeProcess]:
    proc = _FakeProcess()
    session: dict[str, Any] = {
        "proc": proc,
        "stdout": [],
        "stderr": [],
        "host": "192.0.2.10",
        "user": "root",
        "command": "bash",
        "started_at": time.time(),
        "timeout_sec": 900,
        "harness": harness,
        "password_file_path": None,
        "tasks": [],
        "unchanged_read_count": 0,
    }
    network_interactive_sessions._SSH_INTERACTIVE_SESSIONS[session_id] = session
    return session, proc


def _remove_session(session_id: str) -> None:
    network_interactive_sessions._SSH_INTERACTIVE_SESSIONS.pop(session_id, None)


# --- H14: ssh_session_send routes through risk policy + approval --------------


def test_h14_ssh_session_send_denied_by_approval_writes_no_bytes() -> None:
    harness = _DenyingApprovalHarness()
    session_id = "sshint-h14-deny"
    _install_fake_session(session_id, harness=harness)
    try:
        result = asyncio.run(
            network_interactive_sessions.ssh_session_send(
                session_id=session_id,
                input="rm -rf /",
                wait_sec=0,
                state=LoopState(cwd="/tmp"),
                harness=harness,
            )
        )
        session = network_interactive_sessions._SSH_INTERACTIVE_SESSIONS[session_id]
        proc = session["proc"]
        assert result["success"] is False
        assert result.get("status") == "denied"
        assert result["metadata"].get("approval_denied") is True
        assert len(harness.approval_calls) == 1
        assert proc.stdin.writes == []
    finally:
        _remove_session(session_id)


def test_h14_ssh_session_send_approved_writes_bytes() -> None:
    harness = _ApprovingApprovalHarness()
    session_id = "sshint-h14-allow"
    _install_fake_session(session_id, harness=harness)
    try:
        result = asyncio.run(
            network_interactive_sessions.ssh_session_send(
                session_id=session_id,
                input="y",
                wait_sec=0,
                state=LoopState(cwd="/tmp"),
                harness=harness,
            )
        )
        session = network_interactive_sessions._SSH_INTERACTIVE_SESSIONS[session_id]
        proc = session["proc"]
        assert result["success"] is True
        assert len(harness.approval_calls) == 1
        assert proc.stdin.writes == [b"y\n"]
    finally:
        _remove_session(session_id)


# --- M18: interactive SSH honors configured StrictHostKeyChecking -------------


def _start_interactive_session(monkeypatch: pytest.MonkeyPatch, mode: str) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def _fake_build_ssh_command(**kwargs: Any) -> tuple[str, None, None]:
        captured.update(kwargs)
        return ("ssh -- root@192.0.2.10 bash", None, None)

    async def _fake_create_process(**kwargs: Any) -> _FakeProcess:
        return _FakeProcess()

    from smallctl.tools import shell as shell_module

    monkeypatch.setattr(
        network_interactive_sessions, "_build_ssh_command", _fake_build_ssh_command
    )
    monkeypatch.setattr(shell_module, "create_process", _fake_create_process)
    harness = SimpleNamespace(
        config=SimpleNamespace(ssh_strict_host_key_checking=mode),
        event_handler=None,
    )
    result = asyncio.run(
        network_interactive_sessions.ssh_session_start(
            host="192.0.2.10",
            user="root",
            command="bash",
            state=LoopState(cwd="/tmp"),
            harness=harness,
        )
    )
    result["_captured_build_kwargs"] = captured
    return result


def _cleanup_started_sessions() -> None:
    for session_id, session in list(
        network_interactive_sessions._SSH_INTERACTIVE_SESSIONS.items()
    ):
        tasks = session.get("tasks") if isinstance(session, dict) else None
        if tasks:
            asyncio.run(asyncio.sleep(0))
        network_interactive_sessions._SSH_INTERACTIVE_SESSIONS.pop(session_id, None)


@pytest.mark.parametrize("mode", ["yes", "no", "accept-new"])
def test_m18_interactive_session_honors_configured_host_key_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    try:
        result = _start_interactive_session(monkeypatch, mode)
        assert result["success"] is True
        captured = result["_captured_build_kwargs"]
        assert captured.get("strict_host_key_checking") == mode
        assert result["metadata"].get("ssh_strict_host_key_checking") == mode
    finally:
        _cleanup_started_sessions()


def test_m18_interactive_session_invalid_mode_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned: list[Any] = []

    async def _fake_create_process(**kwargs: Any) -> Any:
        spawned.append(kwargs)
        raise AssertionError("process must not spawn for invalid ssh config")

    from smallctl.tools import shell as shell_module

    monkeypatch.setattr(shell_module, "create_process", _fake_create_process)
    harness = SimpleNamespace(
        config=SimpleNamespace(ssh_strict_host_key_checking="bogus"),
        event_handler=None,
    )
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
    assert result["metadata"].get("reason") == "invalid_ssh_config"
    assert result["metadata"].get("config_key") == "ssh_strict_host_key_checking"
    assert spawned == []
    assert network_interactive_sessions._SSH_INTERACTIVE_SESSIONS == {}


# --- H12: mutating lint invocations are not auto-approved ---------------------


@pytest.mark.parametrize(
    "command",
    [
        "mypy --install-types --non-interactive",
        "mypy --install-types src",
        "mypy --html-report /tmp/mypy-report src",
        "mypy --junit-xml=/tmp/mypy.xml src",
        "flake8 --output-file=/tmp/flake8.txt src",
        "flake8 --output-file /tmp/flake8.txt src",
    ],
)
def test_h12_mutating_lint_flag_sets_not_auto_approved(command: str) -> None:
    assert _is_safe_compile_lint_command(command, cwd="/tmp") is False


@pytest.mark.parametrize(
    "command",
    [
        "mypy src/",
        "mypy --strict src/",
        "flake8 src",
        "shellcheck deploy.sh",
        "ruff check src",
    ],
)
def test_h12_non_mutating_lint_invocations_stay_auto_approved(command: str) -> None:
    assert _is_safe_compile_lint_command(command, cwd="/tmp") is True


def test_h12_py_compile_dropped_from_auto_approve() -> None:
    # python3 -m py_compile writes __pycache__ bytecode by default and offers
    # no flag to suppress the write, so it is no longer auto-approved.
    assert _is_safe_compile_lint_command("python3 -m py_compile x.py", cwd="/tmp") is False


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
        "ruff format src",
    ],
)
def test_h12_original_bypass_payloads_stay_rejected(command: str) -> None:
    assert _is_safe_compile_lint_command(command, cwd="/tmp") is False


# --- L29: workspace containment runs before disposable allowances --------------


def test_l29_out_of_workspace_disposable_delete_blocked(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))

    result = _shell_workspace_destructive_delete_guard(
        state, "rm -rf /var/tmp/outside-ws/__pycache__"
    )

    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "workspace_destructive_delete_blocked"
    assert result["metadata"]["blocked_targets"][0]["reasons"] == ["outside_workspace"]


def test_l29_out_of_workspace_explicit_delete_blocked(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "delete /var/tmp/outside-ws/target"

    result = _shell_workspace_destructive_delete_guard(
        state, "rm -rf /var/tmp/outside-ws/target"
    )

    assert result is not None
    assert result["success"] is False
    assert result["metadata"]["blocked_targets"][0]["reasons"] == ["outside_workspace"]


def test_l29_in_workspace_disposable_delete_still_allowed(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    cache_dir = tmp_path / "__pycache__"
    cache_dir.mkdir()

    assert (
        _shell_workspace_destructive_delete_guard(state, f"rm -rf {cache_dir}") is None
    )


# --- M2: low-risk but mutating tools reject unknown args -----------------------


def _make_real_dispatcher(state: LoopState) -> ToolDispatcher:
    provider = SimpleNamespace(state=state, log=logging.getLogger("test.gap"))
    registry = build_registry(provider)
    return ToolDispatcher(registry, state=state, phase="execute")


def test_m2_phase_contract_update_unknown_arg_rejected(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _make_real_dispatcher(state)

    result = asyncio.run(
        dispatcher.dispatch(
            "phase_contract_update",
            {
                "contract": {"version": 1, "active_phase": "build", "phases": {}},
                "bogus_key": True,
            },
        )
    )

    assert result.success is False
    assert "bogus_key" in str(result.error)
    assert result.metadata["validation_error"] == "schema_validation"
    kinds = [issue["kind"] for issue in result.metadata["validation_issues"]]
    assert "additional_property" in kinds


def test_m2_plan_export_unknown_arg_rejected_without_write(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _make_real_dispatcher(state)
    export_path = tmp_path / "plan.md"

    result = asyncio.run(
        dispatcher.dispatch(
            "plan_export",
            {"path": str(export_path), "bogus_key": True},
        )
    )

    assert result.success is False
    assert "bogus_key" in str(result.error)
    assert not export_path.exists()


def test_m2_plan_step_update_unknown_arg_rejected(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _make_real_dispatcher(state)

    result = asyncio.run(
        dispatcher.dispatch(
            "plan_step_update",
            {"step_id": "s1", "status": "done", "bogus_key": True},
        )
    )

    assert result.success is False
    assert "bogus_key" in str(result.error)


def test_m2_read_only_low_risk_tool_unknown_arg_warns_visibly(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _make_real_dispatcher(state)

    result = asyncio.run(dispatcher.dispatch("loop_status", {"bogus_key": True}))

    assert result.success is True
    assert "bogus_key" in str(result.output)
    assert "Warning" in str(result.output)
    assert result.metadata["ignored_arguments"] == ["bogus_key"]


# --- M14: bare HTTP 403 is not mislabeled as content policy --------------------


def test_m14_structured_content_policy_403_keeps_content_policy_wording() -> None:
    details = {
        "type": "content_policy_violation",
        "reason": "provider_content_policy_block",
        "provider_profile": "openrouter",
        "provider_error": "request blocked",
        "status_code": 403,
    }
    message = _chunk_error_failure_message(details)
    assert "content policy" in message.lower()
    assert _chunk_error_failure_type(details) == "content_policy_violation"


def test_m14_403_with_content_policy_reason_uses_content_policy_wording() -> None:
    details = {"status_code": 403, "reason": "content_policy_violation"}
    message = _chunk_error_failure_message(details)
    assert "content policy" in message.lower()
    assert _chunk_error_failure_type(details) == "content_policy_violation"


def test_m14_403_with_provider_message_naming_content_policy() -> None:
    details = {
        "status_code": 403,
        "message": "This request was blocked by the provider content policy",
    }
    message = _chunk_error_failure_message(details)
    assert "content policy" in message.lower()
    assert _chunk_error_failure_type(details) == "content_policy_violation"


def test_m14_bare_403_generic_message_uses_neutral_wording() -> None:
    details = {"status_code": 403, "message": "Forbidden: key lacks entitlement"}
    message = _chunk_error_failure_message(details)
    assert "content policy" not in message.lower()
    assert "Forbidden: key lacks entitlement" in message
    assert "403" in message
    assert _chunk_error_failure_type(details) == "provider"


def test_m14_bare_403_without_message_uses_neutral_wording() -> None:
    message = _chunk_error_failure_message({"status_code": 403})
    assert "content policy" not in message.lower()
    assert "403" in message
    assert _chunk_error_failure_type({"status_code": 403}) == "provider"
