from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.tools import shell
from smallctl.tools.network_interactive_sessions import (
    _SSH_INTERACTIVE_SESSIONS,
    ssh_session_read,
    ssh_session_start,
    ssh_session_send,
    ssh_session_send_and_read,
)


class _ChunkStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _WritableStdin:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None


class _InteractiveProcess:
    def __init__(self, stdout: str) -> None:
        self.stdout = _ChunkStream([stdout.encode("utf-8"), b""])
        self.stderr = _ChunkStream([b""])
        self.stdin = _WritableStdin()
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    async def wait(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15


async def _run_send_test(
    proc: _InteractiveProcess,
    input_text: str,
    send_newline: bool,
    monkeypatch,
) -> tuple[dict, list[bytes]]:
    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(shell, "create_process", _fake_create_process)

    _SSH_INTERACTIVE_SESSIONS.clear()
    try:
        start = await ssh_session_start(
            host="192.0.2.10",
            user="root",
            command="./installer",
        )
        assert start["success"] is True
        session_id = start["output"]["session_id"]

        result = await ssh_session_send(
            session_id=session_id,
            input=input_text,
            send_newline=send_newline,
            wait_sec=0,
        )
        return result, proc.stdin.writes
    finally:
        _SSH_INTERACTIVE_SESSIONS.clear()


def test_ssh_session_send_with_send_newline_true_appends_newline(monkeypatch) -> None:
    proc = _InteractiveProcess("Continue? (y/N)")
    result, writes = asyncio.run(
        _run_send_test(proc, "y", send_newline=True, monkeypatch=monkeypatch)
    )
    assert result["success"] is True
    assert writes == [b"y\n"]


def test_ssh_session_send_with_send_newline_false_sends_raw_input(monkeypatch) -> None:
    proc = _InteractiveProcess("Password:")
    result, writes = asyncio.run(
        _run_send_test(proc, "secret", send_newline=False, monkeypatch=monkeypatch)
    )
    assert result["success"] is True
    assert writes == [b"secret"]


def test_ssh_session_send_and_read_returns_updated_output(monkeypatch) -> None:
    prompts = [
        "Continue? (y/N)",
        "Installing packages...\nDone.",
    ]

    class _MultiPromptProcess:
        def __init__(self, outputs: list[str]) -> None:
            self._outputs = outputs
            self._idx = 0
            self.stdout = _ChunkStream([outputs[0].encode("utf-8"), b""])
            self.stderr = _ChunkStream([b""])
            self.stdin = _WritableStdin()
            self.returncode: int | None = None

        async def wait(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    async def _fake_create_process(**_kwargs):
        return _MultiPromptProcess(prompts)

    monkeypatch.setattr(shell, "create_process", _fake_create_process)

    async def _run() -> None:
        _SSH_INTERACTIVE_SESSIONS.clear()
        try:
            start = await ssh_session_start(
                host="192.0.2.10",
                user="root",
                command="./installer",
            )
            assert start["success"] is True
            session_id = start["output"]["session_id"]

            result = await ssh_session_send_and_read(
                session_id=session_id,
                input="y",
                send_newline=True,
                wait_sec=0.01,
            )
            assert result["success"] is True
        finally:
            _SSH_INTERACTIVE_SESSIONS.clear()

    asyncio.run(_run())


def test_ssh_session_send_preserves_backward_compatibility_without_send_newline(monkeypatch) -> None:
    proc = _InteractiveProcess("Continue? (y/N)")

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(shell, "create_process", _fake_create_process)

    async def _run() -> None:
        _SSH_INTERACTIVE_SESSIONS.clear()
        try:
            start = await ssh_session_start(
                host="192.0.2.10",
                user="root",
                command="./installer",
            )
            assert start["success"] is True
            session_id = start["output"]["session_id"]

            result = await ssh_session_send(
                session_id=session_id,
                input="y",
                wait_sec=0,
            )
            assert result["success"] is True
            assert proc.stdin.writes == [b"y\n"]
        finally:
            _SSH_INTERACTIVE_SESSIONS.clear()

    asyncio.run(_run())


def test_ssh_session_send_detects_unchanged_prompt_after_send(monkeypatch) -> None:
    proc = _InteractiveProcess("Continue? (y/N)")

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(shell, "create_process", _fake_create_process)

    async def _run() -> None:
        _SSH_INTERACTIVE_SESSIONS.clear()
        try:
            start = await ssh_session_start(
                host="192.0.2.10",
                user="root",
                command="./installer",
            )
            assert start["success"] is True
            session_id = start["output"]["session_id"]

            result = await ssh_session_send(
                session_id=session_id,
                input="y",
                send_newline=True,
                wait_sec=0,
            )
            assert result["success"] is True
            assert "_note" in result["output"]
            assert "unchanged" in result["output"]["_note"].lower()
        finally:
            _SSH_INTERACTIVE_SESSIONS.clear()

    asyncio.run(_run())


def test_ssh_session_read_detects_repeated_unchanged_prompt(monkeypatch) -> None:
    proc = _InteractiveProcess("Continue? (y/N)")

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(shell, "create_process", _fake_create_process)

    async def _run() -> None:
        _SSH_INTERACTIVE_SESSIONS.clear()
        try:
            start = await ssh_session_start(
                host="192.0.2.10",
                user="root",
                command="./installer",
            )
            assert start["success"] is True
            session_id = start["output"]["session_id"]

            first = await ssh_session_read(session_id=session_id, wait_sec=0)
            second = await ssh_session_read(session_id=session_id, wait_sec=0)

            assert first["success"] is True
            assert second["success"] is True
            assert second["metadata"]["interactive_output_unchanged"] is True
            assert second["metadata"]["unchanged_read_count"] >= 1
            assert "unchanged" in second["output"]["_note"].lower()
        finally:
            _SSH_INTERACTIVE_SESSIONS.clear()

    asyncio.run(_run())


def test_ssh_session_send_and_read_fails_gracefully_for_unknown_session() -> None:
    result = asyncio.run(
        ssh_session_send_and_read(
            session_id="nonexistent-session",
            input="y",
        )
    )
    assert result["success"] is False
    assert "Unknown" in result["error"]
