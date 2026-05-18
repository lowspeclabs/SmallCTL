from __future__ import annotations

import asyncio

from smallctl.state import LoopState
from smallctl.tools import network, shell


def _state() -> LoopState:
    state = LoopState()
    state.current_phase = "execute"
    return state


class _ChunkStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _LoopingProcess:
    def __init__(self, stdout: str) -> None:
        self.stdout = _ChunkStream([stdout.encode("utf-8"), b""])
        self.stderr = _ChunkStream([b""])
        self.stdin = None
        self.returncode: int | None = None
        self.killed = False
        self.terminated = False
        self._done = asyncio.Event()

    async def wait(self) -> int | None:
        await self._done.wait()
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._done.set()

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15
        self._done.set()


class _DoneProcess:
    def __init__(self, stdout: str = "ok\n") -> None:
        self.stdout = _ChunkStream([stdout.encode("utf-8"), b""])
        self.stderr = _ChunkStream([b""])
        self.stdin = None
        self.returncode = 0

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        return None

    def terminate(self) -> None:
        return None


def test_shell_exec_blocks_yes_pipe_to_interactive_installer() -> None:
    result = asyncio.run(
        shell.shell_exec(
            command="yes | /opt/fogproject/bin/installfog.sh",
            state=_state(),
            timeout_sec=300,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "unsafe_yes_pipe_interactive_installer"
    assert "--autoaccept" in result["error"]
    assert "printf" in result["error"]


def test_ssh_exec_blocks_yes_pipe_to_interactive_installer_before_launch() -> None:
    result = asyncio.run(
        network.run_ssh_command(
            host="192.0.2.10",
            user="root",
            command="yes | /opt/fogproject/bin/installfog.sh",
            timeout_sec=300,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "unsafe_yes_pipe_interactive_installer"
    assert result["metadata"]["host"] == "192.0.2.10"


def test_ssh_exec_blocks_single_echo_pipe_to_interactive_installer_before_launch() -> None:
    result = asyncio.run(
        network.run_ssh_command(
            host="192.0.2.10",
            user="root",
            command='echo "Y" | /opt/fogproject/bin/installfog.sh',
            timeout_sec=300,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "unsafe_single_answer_pipe_interactive_installer"
    assert "single-answer" in result["error"]
    assert "printf" in result["error"]


def test_ssh_exec_auto_extends_default_timeout_for_installer_like_command(monkeypatch) -> None:
    async def _fake_create_process(**_kwargs):
        return _DoneProcess()

    monkeypatch.setattr(network, "create_process", _fake_create_process)

    result = asyncio.run(
        network.run_ssh_command(
            host="192.0.2.10",
            user="root",
            command="/opt/fogproject/bin/installfog.sh -Y",
            timeout_sec=60,
        )
    )

    assert result["success"] is True
    extension = result["metadata"]["timeout_sec_auto_extended"]
    assert extension["from"] == 60
    assert extension["to"] == 600


def test_ssh_exec_requires_remote_installer_preflight_before_installfog() -> None:
    state = _state()
    result = asyncio.run(
        network.run_ssh_command(
            host="192.0.2.10",
            user="root",
            command="cd /opt/fogproject/bin && ./installfog.sh -Y",
            timeout_sec=600,
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "remote_installer_preflight_required"
    assert "git status --short" in result["metadata"]["next_required_action"]
    assert "test -x /opt/fogproject/bin/installfog.sh" in result["metadata"]["next_required_action"]


def test_ssh_exec_allows_clean_remote_installer_preflight(monkeypatch) -> None:
    state = _state()
    state.scratchpad["_remote_installer_preflight"] = {
        "192.0.2.10|root|/opt/fogproject/bin|/opt/fogproject/bin/installfog.sh": {
            "status": "clean",
            "created_at_step": state.step_count,
        }
    }

    async def _fake_create_process(**_kwargs):
        return _DoneProcess()

    monkeypatch.setattr(network, "create_process", _fake_create_process)

    result = asyncio.run(
        network.run_ssh_command(
            host="192.0.2.10",
            user="root",
            command="cd /opt/fogproject/bin && ./installfog.sh -Y",
            timeout_sec=600,
            state=state,
        )
    )

    assert result["success"] is True


def test_ssh_exec_failed_remote_installer_preflight_recommends_clean_reset() -> None:
    state = _state()
    state.scratchpad["_remote_installer_preflight"] = {
        "192.0.2.10|root|/opt/fogproject/bin|/opt/fogproject/bin/installfog.sh": {
            "status": "missing_critical_files",
            "created_at_step": state.step_count,
            "checks": ["git status --short", "test -x /opt/fogproject/bin/installfog.sh"],
        }
    }

    result = asyncio.run(
        network.run_ssh_command(
            host="192.0.2.10",
            user="root",
            command="cd /opt/fogproject/bin && ./installfog.sh -Y",
            timeout_sec=600,
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "remote_installer_preflight_failed"
    assert "fresh clone or clean reset" in result["error"]


def test_shell_exec_stops_repeated_invalid_input_loop(monkeypatch) -> None:
    loop_output = "\n".join(
        [
            "Choice: [2]   Invalid input, please try again.",
            "Choice: [2]   Invalid input, please try again.",
            "Choice: [2]   Invalid input, please try again.",
        ]
    )
    proc = _LoopingProcess(loop_output)

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(shell, "_create_process", _fake_create_process)

    result = asyncio.run(
        shell.shell_exec(
            command="./installer",
            state=_state(),
            timeout_sec=300,
        )
    )

    assert result["success"] is False
    assert proc.killed is True
    assert result["metadata"]["reason"] == "interactive_invalid_input_loop"
    assert result["metadata"]["invalid_input_count"] >= 3
    assert "Invalid input" in result["metadata"]["output"]["stdout"]


def test_ssh_exec_stops_repeated_invalid_input_loop(monkeypatch) -> None:
    loop_output = "\n".join(
        [
            "Choice: [2]   Invalid input, please try again.",
            "Choice: [2]   Invalid input, please try again.",
            "Choice: [2]   Invalid input, please try again.",
        ]
    )
    proc = _LoopingProcess(loop_output)

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(network, "create_process", _fake_create_process)

    result = asyncio.run(
        network.run_ssh_command(
            host="192.0.2.10",
            user="root",
            command="./installer",
            timeout_sec=300,
        )
    )

    assert result["success"] is False
    assert proc.terminated is True
    assert result["metadata"]["reason"] == "interactive_invalid_input_loop"
    assert result["metadata"]["ssh_error_class"] == "interactive_invalid_input_loop"
    assert "Invalid input" in result["metadata"]["output"]["stdout"]
