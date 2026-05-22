from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.progress_guard import (
    _artifact_read_is_continuation_page,
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
