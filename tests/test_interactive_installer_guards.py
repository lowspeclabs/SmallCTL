from __future__ import annotations

import asyncio

from smallctl.state import LoopState
from smallctl.tools import network, shell
from smallctl.tools.shell_support import _looks_like_remote_installer_mutation


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


class _ProbeProcess:
    def __init__(self, stdout: str) -> None:
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


def test_remote_installer_mutation_detector_allows_read_only_installfog_verifiers() -> None:
    assert not _looks_like_remote_installer_mutation("pwd && test -x /root/fogproject/bin/installfog.sh")
    assert not _looks_like_remote_installer_mutation("ls -la /root/fogproject/bin/installfog.sh")
    assert not _looks_like_remote_installer_mutation("stat /root/fogproject/bin/installfog.sh")


def test_remote_installer_mutation_detector_detects_actual_installfog_execution() -> None:
    assert _looks_like_remote_installer_mutation("/root/fogproject/bin/installfog.sh")
    assert _looks_like_remote_installer_mutation("cd /root/fogproject/bin && ./installfog.sh -Y")
    assert _looks_like_remote_installer_mutation("bash /root/fogproject/bin/installfog.sh")


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


def test_ssh_exec_requires_remote_installer_preflight_before_installfog(monkeypatch) -> None:
    state = _state()

    probe_stdout = (
        "__PREFLIGHT_PWD__\n"
        "/root\n"
        "__PREFLIGHT_GIT_TOPLEVEL__\n"
        "/opt/fogproject/bin\n"
        "__PREFLIGHT_GIT_STATUS__\n"
        "\n"
        "__PREFLIGHT_SCRIPT__\n"
        "EXECUTABLE\n"
        "__PREFLIGHT_HELP__\n"
        "Usage: installfog.sh [-h?dEUuHSCKYXTFA] [--autoaccept]\n"
        "__PREFLIGHT_PRESEED__\n"
        "\n"
        "__PREFLIGHT_DONE__\n"
    )

    async def _fake_create_process(**_kwargs):
        return _ProbeProcess(probe_stdout)

    monkeypatch.setattr(network, "create_process", _fake_create_process)
    monkeypatch.setattr(shell, "create_process", _fake_create_process)

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
    assert result["metadata"]["preflight_probes"]["script_exists"] is True
    assert result["metadata"]["preflight_probes"]["script_executable"] is True
    assert "--autoaccept" in result["error"]


def test_ssh_exec_allows_clean_remote_installer_preflight(monkeypatch) -> None:
    state = _state()
    state.scratchpad["_remote_installer_preflight"] = {
        "192.0.2.10|root|/opt/fogproject/bin": {
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
        "192.0.2.10|root|/opt/fogproject/bin": {
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


def test_ssh_exec_timeout_with_prompt_suggests_interactive_session(monkeypatch) -> None:
    proc = _LoopingProcess("Should the installer try to disable the local firewall for you now? (y/N)")

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(network, "create_process", _fake_create_process)

    result = asyncio.run(
        network.run_ssh_command(
            host="192.0.2.10",
            user="root",
            command="./prompt-tool",
            timeout_sec=1,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["failure_kind"] == "interactive_prompt_wait"
    assert result["metadata"]["detected_prompt"]["type"] == "yes_no"
    assert "ssh_session_start" in result["metadata"]["suggested_tools"]


def test_ssh_session_start_returns_prompt_state_and_accepts_input(monkeypatch) -> None:
    proc = _InteractiveProcess("Should the installer try to disable the local firewall for you now? (y/N)")

    async def _fake_create_process(**_kwargs):
        return proc

    monkeypatch.setattr(shell, "create_process", _fake_create_process)

    start = asyncio.run(
        network.ssh_session_start(
            host="192.0.2.10",
            user="root",
            command="./installer",
        )
    )

    assert start["success"] is True
    session_id = start["output"]["session_id"]
    assert start["output"]["status"] == "waiting_for_input"
    assert start["output"]["detected_prompt"]["type"] == "yes_no"

    sent = asyncio.run(network.ssh_session_send(session_id=session_id, input="n\n", wait_sec=0))
    assert sent["success"] is True
    assert proc.stdin.writes == [b"n\n"]

    closed = asyncio.run(network.ssh_session_close(session_id=session_id))
    assert closed["success"] is True
    assert proc.terminated is True


def test_shell_exec_requires_local_installer_preflight_before_install_script(monkeypatch) -> None:
    state = _state()

    probe_stdout = (
        "__PREFLIGHT_PWD__\n"
        "/root\n"
        "__PREFLIGHT_GIT_TOPLEVEL__\n"
        "/opt/fogproject/bin\n"
        "__PREFLIGHT_GIT_STATUS__\n"
        "\n"
        "__PREFLIGHT_SCRIPT__\n"
        "EXECUTABLE\n"
        "__PREFLIGHT_HELP__\n"
        "Usage: installfog.sh [-h?dEUuHSCKYXTFA] [--autoaccept]\n"
        "__PREFLIGHT_PRESEED__\n"
        "\n"
        "__PREFLIGHT_DONE__\n"
    )

    async def _fake_create_process(**_kwargs):
        return _ProbeProcess(probe_stdout)

    monkeypatch.setattr(shell, "_create_process", _fake_create_process)

    result = asyncio.run(
        shell.shell_exec(
            command="cd /opt/fogproject/bin && ./installfog.sh -Y",
            state=state,
            timeout_sec=600,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "remote_installer_preflight_required"
    assert result["metadata"]["preflight_probes"]["script_exists"] is True
    assert result["metadata"]["preflight_probes"]["script_executable"] is True
    assert "--autoaccept" in result["error"]
    # Non-interactive flags detected, so interactive session tools should NOT be exposed
    assert "_expose_interactive_session_tools" not in state.scratchpad


def test_shell_exec_local_preflight_exposes_interactive_tools_when_no_flags_found(monkeypatch) -> None:
    state = _state()

    probe_stdout = (
        "__PREFLIGHT_PWD__\n"
        "/root\n"
        "__PREFLIGHT_GIT_TOPLEVEL__\n"
        "/opt/fogproject/bin\n"
        "__PREFLIGHT_GIT_STATUS__\n"
        "\n"
        "__PREFLIGHT_SCRIPT__\n"
        "EXECUTABLE\n"
        "__PREFLIGHT_HELP__\n"
        "Usage: installfog.sh [-h?dEUuHSCKYXTFA]\n"
        "__PREFLIGHT_DONE__\n"
    )

    async def _fake_create_process(**_kwargs):
        return _ProbeProcess(probe_stdout)

    monkeypatch.setattr(shell, "_create_process", _fake_create_process)

    result = asyncio.run(
        shell.shell_exec(
            command="cd /opt/fogproject/bin && ./installfog.sh -Y",
            state=state,
            timeout_sec=600,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["preflight_probes"]["is_interactive"] is True
    assert state.scratchpad.get("_expose_interactive_session_tools") is True


def test_shell_exec_allows_clean_local_installer_preflight(monkeypatch) -> None:
    state = _state()
    state.scratchpad["_remote_installer_preflight"] = {
        "localhost|stephen|/opt/fogproject/bin": {
            "status": "clean",
            "created_at_step": state.step_count,
        }
    }

    async def _fake_create_process(**_kwargs):
        return _DoneProcess()

    monkeypatch.setattr(shell, "_create_process", _fake_create_process)

    result = asyncio.run(
        shell.shell_exec(
            command="cd /opt/fogproject/bin && ./installfog.sh -Y",
            state=state,
            timeout_sec=600,
        )
    )

    assert result["success"] is True


def test_shell_exec_failed_local_installer_preflight_recommends_clean_reset() -> None:
    state = _state()
    state.scratchpad["_remote_installer_preflight"] = {
        "localhost|stephen|/opt/fogproject/bin": {
            "status": "missing_critical_files",
            "created_at_step": state.step_count,
            "checks": ["git status --short", "test -x /opt/fogproject/bin/installfog.sh"],
        }
    }

    result = asyncio.run(
        shell.shell_exec(
            command="cd /opt/fogproject/bin && ./installfog.sh -Y",
            state=state,
            timeout_sec=600,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "remote_installer_preflight_failed"
    assert "fresh clone or clean reset" in result["error"]
