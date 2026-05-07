from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.graph.tool_execution_nodes import dispatch_tools
from smallctl.models.events import UIEventType
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
