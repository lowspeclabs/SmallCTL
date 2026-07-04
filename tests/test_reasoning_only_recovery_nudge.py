from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.model_stream_loop import _build_reasoning_only_nudge


def _tool(name: str) -> dict[str, object]:
    return {"function": {"name": name}}


def _harness(model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        client=SimpleNamespace(model=model_name),
        state=SimpleNamespace(
            scratchpad={"_model_name": model_name},
            recent_errors=[],
        ),
    )


def test_reasoning_only_repair_nudge_names_ssh_tools_when_exposed() -> None:
    nudge = _build_reasoning_only_nudge(
        [_tool("ssh_exec"), _tool("ssh_file_read"), _tool("task_complete")],
        phase="repair",
        harness=_harness("qwen2.5-coder-7b-instruct"),
    )

    assert "remote target" in nudge
    assert "ssh_exec/ssh_file_read" in nudge
    assert "file_patch/file_write/ast_patch" not in nudge
    assert "shell_exec" not in nudge


def test_reasoning_only_general_nudge_names_ssh_tools_when_exposed() -> None:
    nudge = _build_reasoning_only_nudge(
        [_tool("ssh_exec"), _tool("ssh_file_write"), _tool("task_complete")],
        harness=_harness("qwen2.5-coder-7b-instruct"),
    )

    assert "remote action" in nudge
    assert "ssh_exec/ssh_file_write" in nudge
    assert "file/read tools" not in nudge


def test_gemma_4_reasoning_only_nudge_tells_model_to_close_channel() -> None:
    nudge = _build_reasoning_only_nudge(
        [_tool("file_read"), _tool("file_write"), _tool("task_complete")],
        harness=_harness("Gemma 4 12b"),
    )

    assert "End the current reasoning block now" in nudge
    assert "<|channel>thought" in nudge
    assert "<channel|>" in nudge
    assert "JSON object" in nudge
    # Larger Gemma-4 variants now also receive a concrete JSON example.
    assert '"name":"file_read"' in nudge
