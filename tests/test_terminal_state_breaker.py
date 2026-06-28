from __future__ import annotations

from pathlib import Path

import pytest

from smallctl.challenge_progress import (
    terminal_readiness_state,
    record_code_change,
    record_verifier_result,
    _pending_file_mutation_intent,
)
from smallctl.graph.model_call_nodes import _conclusion_signature, _apply_terminal_conclusion_tracking
from smallctl.graph.state import GraphRunState
from smallctl.graph.task_completion_outcomes import maybe_auto_complete_terminal_readiness
from smallctl.state import LoopState


async def _noop_emit(*_args, **_kwargs):
    pass


def test_terminal_readiness_with_written_and_verified_file() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    record_verifier_result(
        state,
        tool_name="shell_exec",
        command="python3 ./temp/example.py",
        verifier_kind="run_target",
        verdict="pass",
        exit_code=0,
    )
    # Create the file so it exists
    Path("./temp/example.py").parent.mkdir(parents=True, exist_ok=True)
    Path("./temp/example.py").write_text("print('ok')", encoding="utf-8")
    readiness = terminal_readiness_state(state)
    assert readiness is not None
    assert readiness["ready"] is True
    assert readiness["verified"] is True
    assert "./temp/example.py" in readiness["existing_paths"]


def test_terminal_readiness_blocks_without_verification() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    Path("./temp/example.py").parent.mkdir(parents=True, exist_ok=True)
    Path("./temp/example.py").write_text("print('ok')", encoding="utf-8")
    readiness = terminal_readiness_state(state)
    assert readiness is None


def test_terminal_readiness_blocks_with_failing_verifier() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    record_verifier_result(
        state,
        tool_name="shell_exec",
        command="python3 ./temp/example.py",
        verifier_kind="run_target",
        verdict="fail",
        exit_code=1,
    )
    Path("./temp/example.py").parent.mkdir(parents=True, exist_ok=True)
    Path("./temp/example.py").write_text("print('ok')", encoding="utf-8")
    readiness = terminal_readiness_state(state)
    assert readiness is None


def test_terminal_readiness_blocks_with_open_write_session() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    record_verifier_result(
        state,
        tool_name="shell_exec",
        command="python3 ./temp/example.py",
        verifier_kind="run_target",
        verdict="pass",
        exit_code=0,
    )
    state.write_session = type("WS", (), {"status": "open"})()
    Path("./temp/example.py").parent.mkdir(parents=True, exist_ok=True)
    Path("./temp/example.py").write_text("print('ok')", encoding="utf-8")
    readiness = terminal_readiness_state(state)
    assert readiness is None


def test_write_session_deadline_blocks_after_three_nonterminal_steps() -> None:
    from smallctl.challenge_progress import _write_session_deadline_block
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    record_verifier_result(
        state,
        tool_name="shell_exec",
        command="python3 ./temp/example.py",
        verifier_kind="run_target",
        verdict="pass",
        exit_code=0,
    )
    blocked = None
    for _ in range(4):
        blocked = _write_session_deadline_block(state, tool_name="file_read")
    assert blocked is not None
    assert blocked.metadata["reason"] == "write_deadline_terminal_guard"


def test_conclusion_signature_detects_marker_phrases() -> None:
    assert _conclusion_signature("I have enough evidence to proceed.") == "enough evidence"
    assert _conclusion_signature("Sufficient evidence has been gathered.") == "sufficient evidence"
    assert _conclusion_signature("All evidence gathered, now writing report.") == "all evidence gathered"
    assert _conclusion_signature("Now write report and finalize.") == "now write report"
    assert _conclusion_signature("Prepare report for submission.") == "prepare report"
    assert _conclusion_signature("Create artifact from findings.") == "create artifact"
    assert _conclusion_signature("Final answer ready.") == "final answer ready"
    assert _conclusion_signature("Call task_complete now.") == "call task_complete"
    assert _conclusion_signature("Complete task and exit.") == "complete task"
    assert _conclusion_signature("No markers here.") == ""


def test_conclusion_signature_strips_code_blocks_and_json() -> None:
    text = "```python\nprint('ok')\n```\nI have enough evidence to finish."
    assert _conclusion_signature(text) == "enough evidence"
    text = '{"tool": "file_read"}\nFinal answer ready.'
    assert _conclusion_signature(text) == "final answer ready"


def test_apply_terminal_conclusion_tracking_injects_nudge_on_repeat() -> None:
    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    harness = type(
        "Harness",
        (),
        {
            "state": state,
            "_runlog": lambda self, event, message, **kwargs: None,
        },
    )()
    graph_state = type("GS", (), {"pending_tool_calls": []})()

    # First occurrence: no nudge, signature stored
    injected = _apply_terminal_conclusion_tracking(harness, graph_state, "I have enough evidence.")
    assert injected is False
    assert state.scratchpad["_terminal_conclusion_signatures"] == ["enough evidence"]

    # Second occurrence with no tool dispatch: nudge injected
    injected = _apply_terminal_conclusion_tracking(harness, graph_state, "I have enough evidence.")
    assert injected is True
    assert state.scratchpad["_terminal_conclusion_signatures"] == []
    messages = state.recent_messages
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert "terminal_conclusion_repetition" in messages[0].metadata.get("recovery_kind", "")


def test_apply_terminal_conclusion_tracking_resets_on_new_signature() -> None:
    state = LoopState()
    harness = type("Harness", (), {"state": state, "_runlog": lambda *a, **k: None})()
    graph_state = type("GS", (), {"pending_tool_calls": []})()

    _apply_terminal_conclusion_tracking(harness, graph_state, "I have enough evidence.")
    assert state.scratchpad["_terminal_conclusion_signatures"] == ["enough evidence"]

    _apply_terminal_conclusion_tracking(harness, graph_state, "Now write report.")
    assert state.scratchpad["_terminal_conclusion_signatures"] == ["now write report"]


def test_apply_terminal_conclusion_tracking_no_nudge_when_tool_calls_present() -> None:
    state = LoopState()
    harness = type("Harness", (), {"state": state, "_runlog": lambda *a, **k: None})()
    graph_state = type("GS", (), {"pending_tool_calls": ["some_tool"]})()

    _apply_terminal_conclusion_tracking(harness, graph_state, "I have enough evidence.")
    _apply_terminal_conclusion_tracking(harness, graph_state, "I have enough evidence.")
    # No nudge because pending_tool_calls is non-empty
    assert len(state.recent_messages) == 0
    assert state.scratchpad["_terminal_conclusion_signatures"] == ["enough evidence"]


def test_apply_terminal_conclusion_tracking_uses_readiness_nudge_when_verified() -> None:
    from smallctl.challenge_progress import record_code_change, record_verifier_result

    state = LoopState()
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    record_verifier_result(
        state,
        tool_name="shell_exec",
        command="python3 ./temp/example.py",
        verifier_kind="run_target",
        verdict="pass",
        exit_code=0,
    )
    Path("./temp/example.py").parent.mkdir(parents=True, exist_ok=True)
    Path("./temp/example.py").write_text("print('ok')", encoding="utf-8")

    harness = type("Harness", (), {"state": state, "_runlog": lambda *a, **k: None})()
    graph_state = type("GS", (), {"pending_tool_calls": []})()

    _apply_terminal_conclusion_tracking(harness, graph_state, "I have enough evidence.")
    _apply_terminal_conclusion_tracking(harness, graph_state, "I have enough evidence.")

    assert len(state.recent_messages) == 1
    assert "Call task_complete now" in state.recent_messages[0].content


def test_pending_file_mutation_intent_from_active_intent() -> None:
    state = LoopState()
    state.active_intent = "requested_file_patch"
    assert _pending_file_mutation_intent(state) is True

    state.active_intent = "requested_write_file"
    assert _pending_file_mutation_intent(state) is True

    state.active_intent = "general_task"
    assert _pending_file_mutation_intent(state) is False

    state.active_intent = ""
    assert _pending_file_mutation_intent(state) is False


def test_pending_file_mutation_intent_from_task_text() -> None:
    state = LoopState()
    state.run_brief.original_task = "Patch the existing file at `./temp/example.py`."
    assert _pending_file_mutation_intent(state) is True

    state.run_brief.original_task = "Create a new file at `./temp/example.py`."
    assert _pending_file_mutation_intent(state) is False

    state.run_brief.original_task = ""
    state.working_memory.current_goal = "patch file.html without overwriting"
    assert _pending_file_mutation_intent(state) is True


@pytest.mark.anyio
async def test_auto_complete_terminal_readiness_finalizes_when_ready(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    target = tmp_path / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('ok')", encoding="utf-8")
    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    record_verifier_result(
        state,
        tool_name="shell_exec",
        command="python3 ./temp/example.py",
        verifier_kind="run_target",
        verdict="pass",
        exit_code=0,
    )

    harness = type(
        "Harness",
        (),
        {
            "state": state,
            "_runlog": lambda *args, **kwargs: None,
            "_emit": _noop_emit,
        },
    )()
    graph_state = GraphRunState(loop_state=state, thread_id="test", run_mode="loop")
    graph_state.last_assistant_text = "Done."

    completed = await maybe_auto_complete_terminal_readiness(graph_state, harness, None)
    assert completed is True
    assert graph_state.final_result is not None
    assert graph_state.final_result["status"] == "completed"
    assert state.scratchpad["_task_complete"] is True


@pytest.mark.anyio
async def test_auto_complete_terminal_readiness_skips_when_not_ready(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    target = tmp_path / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('ok')", encoding="utf-8")
    record_code_change(state, tool_name="file_write", path="./temp/example.py")

    harness = type(
        "Harness",
        (),
        {
            "state": state,
            "_runlog": lambda *args, **kwargs: None,
            "_emit": _noop_emit,
        },
    )()
    graph_state = GraphRunState(loop_state=state, thread_id="test", run_mode="loop")

    completed = await maybe_auto_complete_terminal_readiness(graph_state, harness, None)
    assert completed is False
    assert graph_state.final_result is None


@pytest.mark.anyio
async def test_auto_complete_terminal_readiness_skips_with_open_write_session(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "Build a self-contained Python script at `./temp/example.py`."
    target = tmp_path / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('ok')", encoding="utf-8")
    record_code_change(state, tool_name="file_write", path="./temp/example.py")
    record_verifier_result(
        state,
        tool_name="shell_exec",
        command="python3 ./temp/example.py",
        verifier_kind="run_target",
        verdict="pass",
        exit_code=0,
    )
    state.write_session = type("WS", (), {"status": "open"})()

    harness = type(
        "Harness",
        (),
        {
            "state": state,
            "_runlog": lambda *args, **kwargs: None,
            "_emit": _noop_emit,
        },
    )()
    graph_state = GraphRunState(loop_state=state, thread_id="test", run_mode="loop")

    completed = await maybe_auto_complete_terminal_readiness(graph_state, harness, None)
    assert completed is False
    assert graph_state.final_result is None
