from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.graph.tool_execution_nodes import dispatch_tools
from smallctl.models.events import UIEvent, UIEventType
from smallctl.models.tool_result import ToolEnvelope
from smallctl.tools import shell
from smallctl.tools import shell_foreground
from smallctl.tools import shell_support
from smallctl.tools.ui_streaming import BufferedUIEventEmitter
from smallctl.state import LoopState


def test_shell_env_includes_python_bin_without_name_error() -> None:
    env = shell._shell_env()

    assert isinstance(env, dict)
    assert str(Path(sys.executable).absolute().parent) in env["PATH"].split(os.pathsep)


def test_buffered_ui_event_emitter_batches_chunks_until_flush() -> None:
    emitted: list[object] = []

    async def _emit(_handler: object, event: object) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        _emit=_emit,
        event_handler=object(),
    )
    emitter = BufferedUIEventEmitter(
        harness=harness,
        event_type=UIEventType.SHELL_STREAM,
        flush_interval_sec=60.0,
    )

    async def _run() -> None:
        await emitter.emit_text("hello ")
        await emitter.emit_text("world")
        assert emitted == []
        await emitter.flush()

    asyncio.run(_run())

    assert len(emitted) == 1
    assert emitted[0].event_type == UIEventType.SHELL_STREAM
    assert emitted[0].content == "hello world"


def test_shell_exec_foreground_cancellation_kills_active_process_during_streaming(monkeypatch) -> None:
    class _BlockingStream:
        def __init__(self, first_chunk: bytes = b"") -> None:
            self._first_chunk = first_chunk
            self._sent_first = False
            self.first_read = asyncio.Event()
            self.release = asyncio.Event()

        async def read(self, _chunk_size: int) -> bytes:
            if not self._sent_first:
                self._sent_first = True
                self.first_read.set()
                return self._first_chunk
            await self.release.wait()
            return b""

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = _BlockingStream(b"stream chunk\n")
            self.stderr = _BlockingStream(b"")
            self.returncode: int | None = None
            self.kill_calls = 0
            self.wait_calls = 0
            self._wait_released = asyncio.Event()

        def kill(self) -> None:
            self.kill_calls += 1
            self.returncode = -9
            self.stdout.release.set()
            self.stderr.release.set()
            self._wait_released.set()

        async def wait(self) -> int:
            self.wait_calls += 1
            await self._wait_released.wait()
            return int(self.returncode or -9)

    monkeypatch.setattr(
        shell_foreground,
        "evaluate_risk_policy",
        lambda *args, **kwargs: SimpleNamespace(
            allowed=True,
            requires_approval=False,
            proof_bundle={},
        ),
    )

    async def _no_sudo_guard(**kwargs):
        return None

    monkeypatch.setattr(shell_foreground, "ensure_sudo_credentials", _no_sudo_guard)

    async def _run() -> None:
        proc = _FakeProcess()
        harness = SimpleNamespace(event_handler=None, _active_processes=set())
        state = LoopState(cwd="/tmp")

        async def _create_process(**kwargs):
            harness._active_processes.add(proc)
            return proc

        task = asyncio.create_task(
            shell_foreground.shell_exec_foreground(
                "tail -f build.log",
                state=state,
                timeout_sec=30,
                harness=harness,
                create_process=_create_process,
            )
        )
        await proc.stdout.first_read.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert proc.kill_calls == 1
        assert proc.wait_calls >= 1
        assert proc not in harness._active_processes

    asyncio.run(_run())


def test_foreground_command_guard_detects_service_and_follow_commands() -> None:
    assert shell_support._likely_long_running_foreground_reason("caddy run /etc/caddy/Caddyfile")
    assert shell_support._likely_long_running_foreground_reason("npm run dev")
    assert shell_support._likely_long_running_foreground_reason("cd /srv/app && npm run dev")
    assert shell_support._likely_long_running_foreground_reason("journalctl -f -u caddy")

    assert shell_support._likely_long_running_foreground_reason("caddy start") is None
    assert shell_support._likely_long_running_foreground_reason("caddy run --help") is None
    assert shell_support._likely_long_running_foreground_reason("systemctl restart caddy") is None
    assert shell_support._likely_long_running_foreground_reason("timeout 20s caddy run /etc/caddy/Caddyfile") is None
    assert shell_support._likely_long_running_foreground_reason("nohup uvicorn app:app >/tmp/app.log 2>&1 &") is None


def test_shell_exec_blocks_likely_foreground_service_without_launching() -> None:
    state = LoopState(cwd="/tmp")

    result = asyncio.run(shell.shell_exec(command="npm run dev", state=state, harness=None))

    assert result["success"] is False
    assert result["metadata"]["reason"] == "long_running_foreground_command"
    assert "background=True" in result["error"]


def test_shell_exec_rejects_foreground_command_with_job_id_before_polling() -> None:
    state = LoopState(cwd="/tmp")

    result = asyncio.run(
        shell.shell_exec(
            command="python3 ./temp/cron_matcher.py",
            job_id="verify_cron_001",
            background=False,
            state=state,
            harness=None,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "invalid_shell_exec_job_id_with_foreground_command"
    assert "omit `job_id`" in result["error"]
    assert "Unknown job id" not in result["error"]


def test_dispatch_cancellation_records_synthetic_tool_result() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 3

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"ssh_exec"}

    events: list[object] = []
    runlog: list[tuple[tuple[object, ...], dict[str, object]]] = []

    async def _emit(_handler: object, event: object) -> None:
        events.append(event)

    async def _dispatch(_tool_name: str, _args: dict[str, object]) -> object:
        raise asyncio.CancelledError

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        _dispatch_tool_call=_dispatch,
        _active_dispatch_task=None,
        _cancel_source="ui_stop_button",
        _runlog=lambda *args, **kwargs: runlog.append((args, kwargs)),
        _emit=_emit,
        log=SimpleNamespace(log=lambda *args, **kwargs: None),
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="ssh_exec",
                args={"host": "192.168.1.89", "command": "caddy run /etc/caddy/Caddyfile"},
                tool_call_id="call-1",
                source="recovered",
            )
        ],
    )

    asyncio.run(dispatch_tools(graph_state, SimpleNamespace(harness=harness, event_handler=object())))

    assert graph_state.final_result == {"status": "cancelled", "reason": "cancel_requested"}
    assert graph_state.last_tool_results
    result = graph_state.last_tool_results[0].result
    assert result.status == "cancelled"
    assert result.metadata["reason"] == "tool_dispatch_cancelled"
    assert result.metadata["cancellation_source"] == "ui_stop_button"
    assert "thread-1:3:call-1:ssh_exec" in state.tool_execution_records
    assert any(entry[0][0] == "tool_dispatch_cancelled" for entry in runlog)


def test_dispatch_emits_tool_call_event_before_result() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 1
    state.run_brief.original_task = "run a command"

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"shell_exec"}

    events: list[UIEvent] = []

    async def _emit(_handler: object, event: UIEvent) -> None:
        events.append(event)

    async def _dispatch(_tool_name: str, _args: dict[str, object]) -> ToolEnvelope:
        return ToolEnvelope(success=True, output={"stdout": "ok", "stderr": "", "exit_code": 0})

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        _dispatch_tool_call=_dispatch,
        _active_dispatch_task=None,
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        log=SimpleNamespace(log=lambda *args, **kwargs: None),
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="loop",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="shell_exec",
                args={"command": "echo hi"},
                tool_call_id="call-1",
            )
        ],
    )

    asyncio.run(dispatch_tools(graph_state, SimpleNamespace(harness=harness, event_handler=object())))

    tool_call_events = [e for e in events if e.event_type == UIEventType.TOOL_CALL]
    tool_result_events = [e for e in events if e.event_type == UIEventType.TOOL_RESULT]

    assert len(tool_call_events) == 1
    assert tool_call_events[0].content == "shell_exec"
    assert tool_call_events[0].data.get("tool_call_id") == "call-1"
    assert tool_call_events[0].data.get("display_text") == 'shell_exec({"command": "echo hi"})'
    assert len(tool_result_events) == 1
    assert not any(
        e.event_type == UIEventType.SYSTEM and "Invoking" in str(e.content)
        for e in events
    )


def test_shell_exec_feeds_configured_sudo_password_inline() -> None:
    class _CapturingStdin:
        def __init__(self) -> None:
            self.writes: list[bytes] = []
            self.closed = False

        def write(self, data: bytes) -> None:
            self.writes.append(data)

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _PromptStream:
        def __init__(self, prompt: bytes = b"[sudo] password for user: ") -> None:
            self._prompt = prompt
            self._sent = False

        async def read(self, _size: int) -> bytes:
            if not self._sent:
                self._sent = True
                return self._prompt
            return b""

    class _FakeProcessWithStdin:
        def __init__(self) -> None:
            self.stdout = _PromptStream()
            self.stderr = _PromptStream(b"")
            self.stdin = _CapturingStdin()
            self.returncode: int | None = 0
            self._waited = False

        async def wait(self) -> int:
            self._waited = True
            return 0

        def kill(self) -> None:
            self.returncode = 0

    async def _create_process(**kwargs):
        nonlocal _proc
        _proc = _FakeProcessWithStdin()
        return _proc

    harness = SimpleNamespace(
        event_handler=None,
        _active_processes=set(),
        get_sudo_password=lambda command: "secret-sudo-password",
    )
    state = LoopState(cwd="/tmp")
    _proc: _FakeProcessWithStdin | None = None

    async def _run() -> dict[str, Any]:
        return await shell_foreground.shell_exec_foreground(
            "sudo systemctl restart nginx",
            state=state,
            timeout_sec=30,
            harness=harness,
            create_process=_create_process,
        )

    result = asyncio.run(_run())

    assert result["success"] is True
    assert _proc is not None
    assert b"secret-sudo-password\n" in _proc.stdin.writes
    assert _proc.stdin.closed is True
    assert harness._active_processes == set()


def test_shell_exec_returns_needs_human_when_no_sudo_password_configured(monkeypatch) -> None:
    class _PromptStream:
        def __init__(self, prompt: bytes = b"[sudo] password for user: ") -> None:
            self._prompt = prompt
            self._sent = False

        async def read(self, _size: int) -> bytes:
            if not self._sent:
                self._sent = True
                return self._prompt
            return b""

    class _FakeProcessWithStdin:
        def __init__(self) -> None:
            self.stdout = _PromptStream()
            self.stderr = _PromptStream(b"")
            self.stdin = None
            self.returncode: int | None = 1
            self._waited = False

        async def wait(self) -> int:
            self._waited = True
            return 1

        def kill(self) -> None:
            self.returncode = 1

    async def _create_process(**kwargs):
        return _FakeProcessWithStdin()

    async def _no_sudo_guard(**kwargs):
        return None

    monkeypatch.setattr(shell_foreground, "ensure_sudo_credentials", _no_sudo_guard)

    harness = SimpleNamespace(
        event_handler=None,
        _active_processes=set(),
        get_sudo_password=lambda command: None,
    )
    state = LoopState(cwd="/tmp")

    async def _run() -> dict[str, Any]:
        return await shell_foreground.shell_exec_foreground(
            "sudo systemctl restart nginx",
            state=state,
            timeout_sec=30,
            harness=harness,
            create_process=_create_process,
        )

    result = asyncio.run(_run())

    assert result["success"] is False
    assert result.get("status") == "needs_human"
    assert "sudo_password_required" in str(result["metadata"].get("reason", ""))


def test_ssh_keygen_known_hosts_removal_reaches_approval_in_diagnosis_remediation() -> None:
    from smallctl.risk_policy import evaluate_risk_policy

    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    decision = evaluate_risk_policy(
        state,
        tool_name="shell_exec",
        tool_risk="high",
        phase="execute",
        action="ssh-keygen -R 192.168.1.161 -f ~/.ssh/known_hosts",
        expected_effect="Remove stale known_hosts entry.",
        rollback="Re-add the key if needed.",
        verification="Retry SSH connection.",
        approval_available=True,
    )

    assert decision.allowed is True
    assert decision.requires_approval is True
    assert decision.approval_kind == "shell"


def test_non_ssh_keygen_mutating_shell_still_blocked_without_claim() -> None:
    from smallctl.risk_policy import evaluate_risk_policy

    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_classification"] = "diagnosis_remediation"

    decision = evaluate_risk_policy(
        state,
        tool_name="shell_exec",
        tool_risk="high",
        phase="execute",
        action="rm -f /tmp/example",
        expected_effect="Remove file.",
        rollback="Restore from backup.",
        verification="Check file is gone.",
        approval_available=True,
    )

    assert decision.allowed is False
    assert "supported claim" in decision.reason.lower()


def test_shell_exec_blocks_pipe_to_shell_on_failed_preflight(monkeypatch) -> None:
    """curl/wget | sh must be blocked when the URL HEAD preflight returns 404."""

    async def _failing_pipe_preflight(command: str) -> tuple[bool, str]:
        if "curl" in command.lower() and "| bash" in command.lower():
            return True, "URL returned 404 on HEAD preflight."
        return False, ""

    monkeypatch.setattr(shell, "_preflight_pipe_to_shell_command", _failing_pipe_preflight)

    state = LoopState(cwd="/tmp")
    result = asyncio.run(
        shell.shell_exec(command="curl -fsSL https://example.com/install.sh | bash", state=state, harness=None)
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "pipe_to_shell_preflight_failed"
    assert "404" in result["error"]


def test_shell_exec_allows_pipe_to_shell_after_clean_preflight(monkeypatch) -> None:
    """curl/wget | sh is allowed when the URL HEAD preflight succeeds."""

    async def _clean_pipe_preflight(command: str) -> tuple[bool, str]:
        return False, ""

    monkeypatch.setattr(shell, "_preflight_pipe_to_shell_command", _clean_pipe_preflight)

    state = LoopState(cwd="/tmp")
    result = asyncio.run(
        shell.shell_exec(command="curl -fsSL https://example.com/install.sh | bash", state=state, harness=None)
    )

    # Should not be blocked by the pipe-to-shell preflight guard.
    assert result["metadata"].get("reason") != "pipe_to_shell_preflight_failed"
