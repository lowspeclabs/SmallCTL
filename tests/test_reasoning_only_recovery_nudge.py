from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.model_stream_loop import _build_reasoning_only_nudge


def _tool(name: str) -> dict[str, object]:
    return {"function": {"name": name}}


def test_reasoning_only_repair_nudge_names_ssh_tools_when_exposed() -> None:
    harness = SimpleNamespace(state=SimpleNamespace(recent_errors=[]))

    nudge = _build_reasoning_only_nudge(
        [_tool("ssh_exec"), _tool("ssh_file_read"), _tool("task_complete")],
        phase="repair",
        harness=harness,
    )

    assert "remote target" in nudge
    assert "ssh_exec/ssh_file_read" in nudge
    assert "file_patch/file_write/ast_patch" not in nudge
    assert "shell_exec" not in nudge


def test_reasoning_only_general_nudge_names_ssh_tools_when_exposed() -> None:
    harness = SimpleNamespace(state=SimpleNamespace(recent_errors=[]))

    nudge = _build_reasoning_only_nudge(
        [_tool("ssh_exec"), _tool("ssh_file_write"), _tool("task_complete")],
        harness=harness,
    )

    assert "remote action" in nudge
    assert "ssh_exec/ssh_file_write" in nudge
    assert "file/read tools" not in nudge
