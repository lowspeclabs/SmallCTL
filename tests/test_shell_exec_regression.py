from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from smallctl.models.events import UIEventType
from smallctl.tools import shell
from smallctl.tools import shell_foreground
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
