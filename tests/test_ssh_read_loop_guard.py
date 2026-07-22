from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.state import PendingToolCall
from smallctl.graph.tool_loop_guards import _ssh_exec_repeats_remote_read_target
from smallctl.graph.tool_loop_guards_support import _semantic_tool_call_fingerprint, _tool_call_fingerprint
from smallctl.state import LoopState


def _attempt(command: str) -> dict[str, str]:
    args = {"command": command, "host": "192.168.1.110"}
    return {
        "tool_name": "ssh_exec",
        "fingerprint": _tool_call_fingerprint("ssh_exec", args),
        "semantic_fingerprint": _semantic_tool_call_fingerprint("ssh_exec", args),
    }


def test_ssh_exec_read_loop_detects_same_remote_file_with_varied_commands() -> None:
    state = LoopState()
    state.scratchpad["_tool_attempt_history"] = [
        _attempt("cat /root/docker-medium-challenge/compose.yaml"),
        _attempt("cd /root/docker-medium-challenge && grep -n 'DB_HOST' compose.yaml"),
    ]
    harness = SimpleNamespace(state=state)
    pending = PendingToolCall(
        tool_name="ssh_exec",
        args={
            "command": "cd /root/docker-medium-challenge && grep -n 'DB_HOST|postgres-db' compose.yaml",
            "host": "192.168.1.110",
        },
    )

    assert _ssh_exec_repeats_remote_read_target(harness, pending) is True


def test_ssh_exec_read_loop_allows_a_new_remote_file() -> None:
    state = LoopState()
    state.scratchpad["_tool_attempt_history"] = [
        _attempt("cat /root/docker-medium-challenge/compose.yaml"),
        _attempt("cd /root/docker-medium-challenge && grep -n 'DB_HOST' compose.yaml"),
    ]
    harness = SimpleNamespace(state=state)
    pending = PendingToolCall(
        tool_name="ssh_exec",
        args={"command": "cat /root/docker-medium-challenge/web/default.conf", "host": "192.168.1.110"},
    )

    assert _ssh_exec_repeats_remote_read_target(harness, pending) is False
