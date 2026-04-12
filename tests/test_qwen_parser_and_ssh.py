from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from smallctl.client.client import OpenAICompatClient
from smallctl.graph.tool_call_parser import parse_tool_calls
from smallctl.state import LoopState
from smallctl.tools.dispatcher import normalize_tool_request
from smallctl.tools import network


class _FakeStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeProc:
    def __init__(self, *, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.stdout = _FakeStream([stdout, b""])
        self.stderr = _FakeStream([stderr, b""])
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        return None


class _Registry:
    @staticmethod
    def names() -> set[str]:
        return {"task_complete"}


def test_qwen_distilled_wrappers_preserve_task_complete_and_plan_reasoning() -> None:
    model_name = "qwen3.5-4b-claude-4.6-opus-reasoning-distilled"
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "Field reasoning.\n",
                            "content": (
                                "<analysis>Inspect the gathered evidence.</analysis>\n"
                                "<plan>1. Call task_complete with the final answer.</plan>\n"
                                "<execution><tool_call>"
                                '{"name":"task_complete","arguments":{"message":"Task complete."}}'
                                "</tool_call>"
                            ),
                        }
                    }
                ]
            },
        }
    ]

    stream = OpenAICompatClient.collect_stream(chunks, reasoning_mode="auto")

    assert "Field reasoning." in stream.thinking_text
    assert "Inspect the gathered evidence." in stream.thinking_text
    assert "Call task_complete with the final answer." in stream.thinking_text
    assert "<analysis>" not in stream.assistant_text
    assert "<plan>" not in stream.assistant_text
    assert "<execution>" not in stream.assistant_text
    assert "<tool_call>" in stream.assistant_text

    harness = SimpleNamespace(
        registry=_Registry(),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model=model_name),
        _runlog=lambda *args, **kwargs: None,
    )
    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="execute"),
        deps=SimpleNamespace(harness=harness),
        model_name=model_name,
    )

    assert len(parse_result.pending_tool_calls) == 1
    assert parse_result.pending_tool_calls[0].tool_name == "task_complete"
    assert parse_result.pending_tool_calls[0].args["message"] == "Task complete."
    assert parse_result.final_assistant_text == ""


def test_ssh_exec_distinguishes_remote_exit_from_transport_failure() -> None:
    state = LoopState(cwd="/tmp")

    with patch.object(
        network,
        "create_process",
        AsyncMock(return_value=_FakeProc(returncode=1)),
    ):
        result = asyncio.run(
            network.ssh_exec(
                host="192.168.1.63",
                user="root",
                password="secret",
                command="which guacamole || which apache-guacamole",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is False
    assert result["error"] == "Remote SSH command exited with code 1"
    assert result["metadata"]["failure_kind"] == "remote_command"
    assert result["metadata"]["ssh_transport_succeeded"] is True
    assert result["metadata"]["output"]["exit_code"] == 1
    assert not any("username/password" in hint.lower() for hint in result["metadata"]["hints"])


def test_ssh_exec_retries_when_accept_new_is_rejected() -> None:
    state = LoopState(cwd="/tmp")
    create_process = AsyncMock(
        side_effect=[
            _FakeProc(
                returncode=255,
                stderr=b"command-line line 0: keyword StrictHostKeyChecking extra arguments at end of line\n",
            ),
            _FakeProc(
                returncode=0,
                stdout=b"ii  guacamole 1.5.0\n",
            ),
        ]
    )

    with patch.object(network, "create_process", create_process):
        result = asyncio.run(
            network.ssh_exec(
                host="192.168.1.63",
                user="root",
                password="secret",
                command="dpkg -l | grep guacamole",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is True
    assert result["output"]["stdout"] == "ii  guacamole 1.5.0\n"
    assert result["metadata"]["ssh_option_retry"] == "strict_host_key_checking_no"
    assert result["metadata"]["ssh_option_retry_reason"] == "accept_new_incompatible"
    assert create_process.await_count == 2
    first_command = create_process.await_args_list[0].kwargs["command"]
    second_command = create_process.await_args_list[1].kwargs["command"]
    assert "StrictHostKeyChecking=accept-new" in first_command
    assert "StrictHostKeyChecking=no" in second_command


def test_ssh_exec_recovers_missing_user_from_task_context() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = (
        'ssh root@192.168.1.63 with username root password "@S02v1735" '
        "is apache guacamole installed?"
    )

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.63",
            "command": "which guac && guac --version || echo 'Guacamole not installed'",
            "password": "@S02v1735",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["user"] == "root"
    assert metadata["recovered_ssh_user"] == "root"


def test_ssh_exec_recovers_connection_probe_command_from_task_context() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = 'ssh into root@192.168.1.63 password "@S02v1735"'

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.63",
            "password": "@S02v1735",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["user"] == "root"
    assert args["command"] == "whoami"
    assert metadata["recovered_ssh_user"] == "root"
    assert metadata["recovered_ssh_command"] == "whoami"


def test_normalize_tool_request_repairs_small_model_tool_aliases() -> None:
    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "use_shell_exec",
        {"command": "pwd"},
        phase="execute",
        state=LoopState(cwd="."),
    )

    assert intercepted is None
    assert tool_name == "shell_exec"
    assert args == {"command": "pwd"}
    assert metadata["repaired_tool_alias_from"] == "use_shell_exec"
    assert metadata["repaired_tool_alias_to"] == "shell_exec"


def test_remote_task_guard_blocks_local_shell_exec_for_remote_context() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = (
        'ssh root@192.168.1.63 with username root password "@S02v1735" '
        "is apache guacamole installed?"
    )

    def _get(name: str):
        if name == "ssh_exec":
            return SimpleNamespace(phase_allowed=lambda phase: True)
        return None

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=_get),
        "shell_exec",
        {"command": "docker ps"},
        phase="execute",
        state=state,
    )

    assert tool_name == "shell_exec"
    assert args == {"command": "docker ps"}
    assert metadata == {}
    assert intercepted is not None
    assert intercepted.success is False
    assert intercepted.error == "This is a remote task. Use `ssh_exec`, not local `shell_exec`."
    assert intercepted.metadata["reason"] == "remote_task_requires_ssh_exec"


def test_normalize_ssh_arguments_prefers_target_user_at_host() -> None:
    normalized = network.normalize_ssh_arguments(
        {
            "target": "root@192.168.1.63",
            "command": "whoami",
            "password": "@S02v1735",
        }
    )

    assert normalized["host"] == "192.168.1.63"
    assert normalized["user"] == "root"
    assert "target" not in normalized


def test_normalize_ssh_arguments_supports_username_alias() -> None:
    normalized = network.normalize_ssh_arguments(
        {
            "host": "192.168.1.63",
            "username": "root",
            "command": "whoami",
        }
    )

    assert normalized["host"] == "192.168.1.63"
    assert normalized["user"] == "root"


def test_normalize_ssh_arguments_requires_host_or_target() -> None:
    try:
        network.normalize_ssh_arguments({"command": "whoami"})
    except ValueError as exc:
        assert str(exc) == "SSH target requires either `target` or `host`."
    else:
        raise AssertionError("expected ValueError for missing SSH target")


def test_parse_ssh_exec_args_from_shell_command_recovers_connection_probe() -> None:
    normalized = network.parse_ssh_exec_args_from_shell_command("ssh root@192.168.1.63")

    assert normalized == {
        "host": "192.168.1.63",
        "user": "root",
        "command": "whoami",
    }
