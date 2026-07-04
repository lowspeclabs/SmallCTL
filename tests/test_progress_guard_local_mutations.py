from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.progress_guard import (
    _artifact_read_is_continuation_page,
    _check_completion_confabulation,
    _tool_history_entry_is_mutation,
    _turn_has_actionable_progress,
)
from smallctl.state import LoopState


def test_turn_has_actionable_progress_counts_file_patch() -> None:
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="file_patch",
                args={"path": "./temp/pong.py"},
                result=SimpleNamespace(
                    success=True,
                    metadata={"path": "./temp/pong.py", "changed": True},
                ),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_turn_has_actionable_progress_counts_file_write() -> None:
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(state=state)

    graph_state = SimpleNamespace(
        last_tool_results=[
            SimpleNamespace(
                tool_name="file_write",
                args={"path": "./temp/pong.py"},
                result=SimpleNamespace(
                    success=True,
                    metadata={"path": "./temp/pong.py", "changed": True},
                ),
            )
        ],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_artifact_read_continuation_page_counts_as_progress() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_artifact_read_coverage"] = {
        "pong_py": {
            "ranges": [{"start_line": 1, "end_line": 150}],
        }
    }
    harness = SimpleNamespace(state=state)

    record = SimpleNamespace(
        tool_name="artifact_read",
        args={"artifact_id": "pong_py", "start_line": 150, "end_line": 275},
        result=SimpleNamespace(success=True, metadata={}),
    )

    assert _artifact_read_is_continuation_page(harness, record) is True

    graph_state = SimpleNamespace(
        last_tool_results=[record],
        last_assistant_text="",
    )

    assert _turn_has_actionable_progress(harness, graph_state) is True


def test_tool_history_entry_is_mutation_counts_file_write() -> None:
    assert _tool_history_entry_is_mutation(
        'file_write|{"path": "./temp/pong.py"}|success'
    ) is True


def test_tool_history_entry_is_mutation_counts_shell_mkdir() -> None:
    assert _tool_history_entry_is_mutation(
        'shell_exec|{"command": "mkdir -p ./temp"}|success'
    ) is True


def test_tool_history_entry_is_mutation_ignores_shell_read() -> None:
    assert _tool_history_entry_is_mutation(
        'shell_exec|{"command": "ls -la ./temp"}|success'
    ) is False


def test_tool_history_entry_is_mutation_ignores_failed_shell() -> None:
    assert _tool_history_entry_is_mutation(
        'shell_exec|{"command": "rm -rf ./temp"}|error:exit 1'
    ) is False


def test_check_completion_confabulation_skips_after_shell_mkdir() -> None:
    state = LoopState(cwd="/tmp")
    state.tool_history.append(
        'shell_exec|{"command": "mkdir -p ./temp"}|success'
    )
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)

    graph_state = SimpleNamespace(
        last_assistant_text="I have already created the directory.",
        last_thinking_text="",
    )

    assert _check_completion_confabulation(harness, graph_state) is None
    assert state.scratchpad.get("_confabulation_nudged") is None
