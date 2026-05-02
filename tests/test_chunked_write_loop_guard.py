from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.lifecycle_nodes import resume_loop_run
from smallctl.graph.routing import LoopRoute
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_execution_nodes import dispatch_tools
from smallctl.graph.tool_outcomes import apply_tool_outcomes
from smallctl.graph.tool_outcome_resolution import maybe_apply_terminal_tool_outcome
from smallctl.graph.tool_execution_recovery import handle_failed_file_write_outcome
from smallctl.models.tool_result import ToolEnvelope
from smallctl.prompts import build_system_prompt
from smallctl.state import LoopState
from smallctl.tools import control, fs
from smallctl.tools.fs_loop_guard import (
    build_loop_guard_status,
    clear_loop_guard_outline_requirement,
    clear_loop_guard_verification_requirement,
)
from smallctl.write_session_fsm import new_write_session


def _make_state(tmp_path: Path) -> LoopState:
    state = LoopState(cwd=str(tmp_path))
    state.active_tool_profiles = ["core"]
    return state


def _attach_write_session(state: LoopState, target: Path, *, session_id: str = "ws-guard") -> None:
    state.write_session = new_write_session(
        session_id=session_id,
        target_path=str(target),
        intent="replace_file",
        next_section="imports",
    )


def test_file_patch_blocks_identical_repeat_when_replacement_keeps_target(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "task_queue.py"
    target.write_text("class Task:\n    priority: int\n", encoding="utf-8")

    first = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="class Task:\n",
            replacement_text="class Task:\n    task_id: str\n",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert first["success"] is True
    assert target.read_text(encoding="utf-8").count("task_id: str") == 1

    second = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="class Task:\n",
            replacement_text="class Task:\n    task_id: str\n",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert second["success"] is False
    assert second["metadata"]["error_kind"] == "repeat_sensitive_patch_already_applied"
    assert target.read_text(encoding="utf-8").count("task_id: str") == 1


def test_completed_write_session_rejects_late_file_write_without_mutating_stage(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "guarded_complete.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-done__guarded_complete__stage.py"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("print('promoted')\n", encoding="utf-8")

    _attach_write_session(state, target, session_id="ws-done")
    assert state.write_session is not None
    state.write_session.status = "complete"
    state.write_session.write_staging_path = str(stage)
    state.write_session.write_sections_completed = ["complete_file"]

    result = asyncio.run(
        fs.file_write(
            path=str(target),
            content="print('late mutation')\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-done",
            section_name="complete_file",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "write_session_already_terminal"
    assert result["metadata"]["write_session_status"] == "complete"
    assert stage.read_text(encoding="utf-8") == "print('promoted')\n"


def _seed_outline_mode(state: LoopState, target: Path, *, pending_read: bool = False) -> str:
    resolved = str(target.resolve())
    state.scratchpad["_chunk_write_loop_guard"] = {
        "version": 1,
        "events": [],
        "paths": {
            resolved: {
                "recent_writes": [],
                "section_checkpoints": ["imports"],
                "escalation_level": 2,
                "pending_read_before_write": pending_read,
                "blocked_attempts": 2,
                "writes_since_last_read": 0,
                "last_read_at": 0.0,
                "last_score": 5,
                "last_section_name": "helpers",
                "last_next_section_name": "main_logic",
                "last_error_kind": "chunked_write_loop_guard",
                "session_id": "ws-guard",
                "outline_required": True,
            }
        },
    }
    return resolved


def test_chunked_write_loop_guard_requires_read_after_checkpoint_revisit(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "guarded.py"
    _attach_write_session(state, target)

    first = asyncio.run(
        fs.file_write(
            path=str(target),
            content="import os\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert first["success"] is True

    blocked = asyncio.run(
        fs.file_write(
            path=str(target),
            content="import os\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert blocked["success"] is False
    assert blocked["metadata"]["error_kind"] == "chunked_write_checkpoint_revisit"
    assert blocked["metadata"]["loop_guard_schedule_read"] is True

    status = asyncio.run(control.loop_status(state))
    active_paths = status["output"]["loop_guard"]["active_paths"]
    assert active_paths
    assert active_paths[0]["pending_read_before_write"] is True
    assert active_paths[0]["last_section_name"] == "imports"

    read_back = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))
    assert read_back["success"] is True
    assert read_back["metadata"]["read_from_staging"] is True

    status_after_read = asyncio.run(control.loop_status(state))
    assert status_after_read["output"]["loop_guard"]["active_paths"][0]["pending_read_before_write"] is False

    second = asyncio.run(
        fs.file_write(
            path=str(target),
            content="class Task:\n    pass\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="types",
            next_section_name="",
        )
    )
    assert second["success"] is True


def test_partial_file_read_does_not_clear_loop_guard_verification_requirement(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "guarded_partial.py"
    _attach_write_session(state, target)

    first = asyncio.run(
        fs.file_write(
            path=str(target),
            content="line 1\nline 2\nline 3\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert first["success"] is True

    blocked = asyncio.run(
        fs.file_write(
            path=str(target),
            content="line 1\nline 2\nline 3\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert blocked["success"] is False

    partial_read = asyncio.run(
        fs.file_read(
            path=str(target),
            cwd=str(tmp_path),
            start_line=1,
            end_line=1,
            state=state,
        )
    )
    assert partial_read["success"] is True
    assert partial_read["metadata"]["complete_file"] is False

    status_after_partial = asyncio.run(control.loop_status(state))
    assert status_after_partial["output"]["loop_guard"]["active_paths"][0]["pending_read_before_write"] is True

    full_read = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))
    assert full_read["success"] is True
    assert full_read["metadata"]["complete_file"] is True

    status_after_full = asyncio.run(control.loop_status(state))
    assert status_after_full["output"]["loop_guard"]["active_paths"][0]["pending_read_before_write"] is False


def test_truncated_file_read_does_not_clear_loop_guard_verification_requirement(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "guarded_truncated.py"
    _attach_write_session(state, target)
    large_chunk = ("x" * 70000) + "\n" + ("y" * 70000) + "\n"

    first = asyncio.run(
        fs.file_write(
            path=str(target),
            content=large_chunk,
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert first["success"] is True

    blocked = asyncio.run(
        fs.file_write(
            path=str(target),
            content=large_chunk,
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert blocked["success"] is False

    truncated_read = asyncio.run(
        fs.file_read(
            path=str(target),
            cwd=str(tmp_path),
            max_bytes=64,
            state=state,
        )
    )
    assert truncated_read["success"] is True
    assert truncated_read["metadata"]["complete_file"] is False
    assert truncated_read["metadata"]["truncated"] is True

    status_after_truncated = asyncio.run(control.loop_status(state))
    assert status_after_truncated["output"]["loop_guard"]["active_paths"][0]["pending_read_before_write"] is True


def test_chunked_write_loop_guard_stagnation_scoring_when_checkpoint_gate_disabled(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.scratchpad["_chunk_write_loop_guard_config"] = {
        "enabled": True,
        "checkpoint_gate": False,
        "diff_gate": False,
        "stagnation_threshold": 3,
        "level2_threshold": 5,
    }
    target = tmp_path / "stagnant.py"
    _attach_write_session(state, target)

    first = asyncio.run(
        fs.file_write(
            path=str(target),
            content="import sys\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert first["success"] is True

    blocked = asyncio.run(
        fs.file_write(
            path=str(target),
            content="import sys\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert blocked["success"] is False
    assert blocked["metadata"]["error_kind"] == "chunked_write_loop_guard"
    assert blocked["metadata"]["loop_guard_score"] >= 5
    assert blocked["metadata"]["loop_guard_escalation_level"] == 2

    status = asyncio.run(control.loop_status(state))
    recent_events = status["output"]["loop_guard"]["recent_events"]
    assert recent_events
    assert recent_events[-1]["event"] == "loop_guard_triggered"


def test_chunked_write_loop_guard_detects_near_identical_projected_content(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.scratchpad["_chunk_write_loop_guard_config"] = {
        "enabled": True,
        "checkpoint_gate": False,
        "diff_gate": False,
        "similarity_threshold": 0.9,
        "stagnation_threshold": 3,
        "level2_threshold": 5,
    }
    target = tmp_path / "near_identical.py"
    _attach_write_session(state, target)
    base = "\n".join(f"line {idx}" for idx in range(1, 41)) + "\n"
    variant = base.replace("line 40", "line 40  # tiny tweak")

    first = asyncio.run(
        fs.file_write(
            path=str(target),
            content=base,
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert first["success"] is True

    blocked = asyncio.run(
        fs.file_write(
            path=str(target),
            content=variant,
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert blocked["success"] is False
    assert blocked["metadata"]["error_kind"] == "chunked_write_loop_guard"
    assert blocked["metadata"]["loop_guard_signals"]["hash_stagnation"] is False
    assert blocked["metadata"]["loop_guard_signals"]["similarity_stagnation"] is True
    assert blocked["metadata"]["loop_guard_signals"]["similarity_ratio"] >= 0.9
    assert blocked["metadata"]["loop_guard_score"] >= 3


def test_failed_chunked_write_loop_guard_schedules_file_read_recovery(tmp_path: Path) -> None:
    state = _make_state(tmp_path)

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
    )
    deps = SimpleNamespace(event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-loop-guard",
        run_mode="execute",
    )
    record = ToolExecutionRecord(
        operation_id="op-loop-guard",
        tool_name="file_write",
        args={"path": str(tmp_path / "guarded.py")},
        tool_call_id="tool-loop-guard",
        result=ToolEnvelope(
            success=False,
            error="LoopGuard blocked the write.",
            metadata={
                "path": str(tmp_path / "guarded.py"),
                "error_kind": "chunked_write_loop_guard",
                "loop_guard_schedule_read": True,
                "loop_guard_score": 4,
                "loop_guard_escalation_level": 1,
                "section_name": "helpers",
            },
        ),
    )

    asyncio.run(
        handle_failed_file_write_outcome(
            graph_state=graph_state,
            harness=harness,
            deps=deps,
            record=record,
        )
    )

    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_read"
    assert pending.args == {"path": str(tmp_path / "guarded.py")}
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["recovery_kind"] == "chunked_write_loop_guard"


def test_failed_chunked_file_append_loop_guard_schedules_file_read_recovery(tmp_path: Path) -> None:
    state = _make_state(tmp_path)

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
    )
    deps = SimpleNamespace(event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-loop-guard-append",
        run_mode="execute",
    )
    record = ToolExecutionRecord(
        operation_id="op-loop-guard-append",
        tool_name="file_append",
        args={"path": str(tmp_path / "guarded.py")},
        tool_call_id="tool-loop-guard-append",
        result=ToolEnvelope(
            success=False,
            error="LoopGuard blocked the append.",
            metadata={
                "path": str(tmp_path / "guarded.py"),
                "error_kind": "chunked_write_loop_guard",
                "loop_guard_schedule_read": True,
                "loop_guard_score": 4,
                "loop_guard_escalation_level": 1,
                "section_name": "helpers",
            },
        ),
    )

    asyncio.run(
        handle_failed_file_write_outcome(
            graph_state=graph_state,
            harness=harness,
            deps=deps,
            record=record,
        )
    )

    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_read"
    assert pending.args == {"path": str(tmp_path / "guarded.py")}
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["recovery_kind"] == "chunked_write_loop_guard"


def test_chunked_write_loop_guard_recovery_read_can_be_scheduled_again_after_verification_read(tmp_path: Path) -> None:
    state = _make_state(tmp_path)

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
    )
    deps = SimpleNamespace(event_handler=None)
    record = ToolExecutionRecord(
        operation_id="op-loop-guard-repeat",
        tool_name="file_write",
        args={"path": str(tmp_path / "guarded.py")},
        tool_call_id="tool-loop-guard-repeat",
        result=ToolEnvelope(
            success=False,
            error="LoopGuard blocked the write.",
            metadata={
                "path": str(tmp_path / "guarded.py"),
                "error_kind": "chunked_write_loop_guard",
                "loop_guard_schedule_read": True,
                "loop_guard_score": 4,
                "loop_guard_escalation_level": 1,
                "section_name": "helpers",
            },
        ),
    )

    graph_state_first = GraphRunState(
        loop_state=state,
        thread_id="thread-loop-guard-first",
        run_mode="execute",
    )
    asyncio.run(
        handle_failed_file_write_outcome(
            graph_state=graph_state_first,
            harness=harness,
            deps=deps,
            record=record,
        )
    )
    assert graph_state_first.pending_tool_calls
    assert state.scratchpad.get("_chunk_write_loop_guard_read_scheduled")

    clear_loop_guard_verification_requirement(
        state,
        path=str(tmp_path / "guarded.py"),
        cwd=str(tmp_path),
    )
    assert state.scratchpad.get("_chunk_write_loop_guard_read_scheduled") is None

    graph_state_second = GraphRunState(
        loop_state=state,
        thread_id="thread-loop-guard-second",
        run_mode="execute",
    )
    asyncio.run(
        handle_failed_file_write_outcome(
            graph_state=graph_state_second,
            harness=harness,
            deps=deps,
            record=record,
        )
    )
    assert graph_state_second.pending_tool_calls
    assert graph_state_second.pending_tool_calls[0].tool_name == "file_read"


def test_chunked_write_loop_guard_hard_abort_after_outline_resume_retry(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    state.scratchpad["_chunk_write_loop_guard_config"] = {
        "enabled": True,
        "checkpoint_gate": False,
        "diff_gate": False,
        "stagnation_threshold": 3,
        "level2_threshold": 5,
    }
    target = tmp_path / "guarded.py"
    _attach_write_session(state, target)

    first = asyncio.run(
        fs.file_write(
            path=str(target),
            content="import sys\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert first["success"] is True

    level2 = asyncio.run(
        fs.file_write(
            path=str(target),
            content="import sys\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert level2["success"] is False
    assert level2["metadata"]["loop_guard_escalation_level"] == 2
    assert level2["metadata"]["loop_guard_outline_required"] is True

    read_back = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))
    assert read_back["success"] is True

    clear_loop_guard_outline_requirement(
        state,
        path=str(target),
        cwd=str(tmp_path),
    )

    aborted = asyncio.run(
        fs.file_write(
            path=str(target),
            content="import sys\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-guard",
            section_name="imports",
            next_section_name="types",
        )
    )
    assert aborted["success"] is False
    assert aborted["metadata"]["loop_guard_hard_abort"] is True
    assert aborted["metadata"]["error_kind"] == "chunked_write_loop_guard_hard_abort"
    assert "Model stuck in write loop" in aborted["error"]

    status = asyncio.run(control.loop_status(state))
    active_paths = status["output"]["loop_guard"]["active_paths"]
    assert active_paths[0]["escalation_level"] == 4
    assert active_paths[0]["outline_required"] is False


def test_loop_guard_outline_mode_prompt_requires_ask_human(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "guarded.py"
    _seed_outline_mode(state, target, pending_read=False)

    prompt = build_system_prompt(
        state,
        "execute",
        available_tool_names=["ask_human", "file_read", "file_write", "task_complete"],
    )

    assert "LOOPGUARD OUTLINE MODE" in prompt
    assert "ask_human(question='...')" in prompt
    assert "Do not call `file_write`" in prompt
    assert "`continue`" in prompt


def test_dispatch_tools_blocks_file_write_during_loop_guard_outline_mode(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "guarded.py"
    _seed_outline_mode(state, target, pending_read=False)
    dispatched: list[tuple[str, dict[str, object]]] = []

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="unexpected dispatch")

    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: {"file_write"}, get=lambda _name: None),
        log=logging.getLogger("test.loop_guard_outline.dispatch"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-outline-dispatch",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
            tool_name="file_write",
            args={"path": str(target), "write_session_id": "ws-guard", "content": "print('x')\n"},
        )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == []
    assert graph_state.pending_tool_calls == []
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["recovery_kind"] == "chunked_write_loop_guard_outline"


def test_loop_guard_outline_interrupt_resume_continue_clears_outline_mode(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "guarded.py"
    resolved = _seed_outline_mode(state, target, pending_read=False)
    emitted: list[object] = []

    async def _emit(_handler, event) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _log_conversation_state=lambda *args, **kwargs: None,
        _is_continue_like_followup=lambda value: str(value or "").strip().lower() == "continue",
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(loop_state=state, thread_id="thread-outline", run_mode="execute")
    record = ToolExecutionRecord(
        operation_id="op-outline",
        tool_name="ask_human",
        args={"question": "Outline ready"},
        tool_call_id="tool-outline",
        result=ToolEnvelope(
            success=True,
            output={"status": "human_input_required", "question": f"Outline for {resolved}"},
        ),
    )

    terminal = asyncio.run(
        maybe_apply_terminal_tool_outcome(
            graph_state,
            deps,
            record,
            chat_mode=False,
        )
    )

    assert terminal is True
    assert state.pending_interrupt["kind"] == "chunked_write_loop_guard_outline"
    assert state.pending_interrupt["path"] == resolved
    assert graph_state.final_result["status"] == "needs_human"

    asyncio.run(resume_loop_run(graph_state, deps, human_input="continue"))

    status = build_loop_guard_status(state)
    assert status["active_paths"][0]["outline_required"] is False
    assert state.pending_interrupt is None
    assert state.recent_messages[-1].metadata["recovery_kind"] == "chunked_write_loop_guard_outline_resume"


def test_apply_tool_outcomes_finalizes_on_chunked_write_loop_guard_hard_abort(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    emitted: list[object] = []

    async def _emit(_handler, event) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        state=state,
        log=logging.getLogger("test.loop_guard_outline.abort"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda error, **kwargs: {"status": "failed", "reason": error, "error": {"message": error, **kwargs}},
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-hard-abort",
        run_mode="execute",
        last_tool_results=[
            ToolExecutionRecord(
                operation_id="op-hard-abort",
                tool_name="file_write",
                args={"path": str(tmp_path / "guarded.py")},
                tool_call_id="tool-hard-abort",
                result=ToolEnvelope(
                    success=False,
                    error="Model stuck in write loop on section 'helpers' for path 'guarded.py' after 3 attempts.",
                    metadata={
                        "path": str(tmp_path / "guarded.py"),
                        "section_name": "helpers",
                        "loop_guard_hard_abort": True,
                        "loop_guard_postmortem": (
                            f"Model stuck in write loop on section 'helpers' for path "
                            f"'{tmp_path / 'guarded.py'}' after 3 attempts."
                        ),
                        "loop_guard_attempts": 3,
                        "loop_guard_trigger_kind": "stagnation",
                        "loop_guard_score": 5,
                    },
                ),
            )
        ],
    )

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == LoopRoute.FINALIZE
    assert graph_state.final_result["status"] == "failed"
    assert "Model stuck in write loop" in graph_state.final_result["reason"]
    assert graph_state.error is not None
    assert emitted


def test_apply_tool_outcomes_finalizes_on_chunked_file_append_loop_guard_hard_abort(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    emitted: list[object] = []

    async def _emit(_handler, event) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        state=state,
        log=logging.getLogger("test.loop_guard_outline.append_abort"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda error, **kwargs: {"status": "failed", "reason": error, "error": {"message": error, **kwargs}},
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-hard-abort-append",
        run_mode="execute",
        last_tool_results=[
            ToolExecutionRecord(
                operation_id="op-hard-abort-append",
                tool_name="file_append",
                args={"path": str(tmp_path / "guarded.py")},
                tool_call_id="tool-hard-abort-append",
                result=ToolEnvelope(
                    success=False,
                    error="Model stuck in write loop on section 'helpers' for path 'guarded.py' after 3 attempts.",
                    metadata={
                        "path": str(tmp_path / "guarded.py"),
                        "section_name": "helpers",
                        "loop_guard_hard_abort": True,
                        "loop_guard_postmortem": (
                            f"Model stuck in write loop on section 'helpers' for path "
                            f"'{tmp_path / 'guarded.py'}' after 3 attempts."
                        ),
                        "loop_guard_attempts": 3,
                        "loop_guard_trigger_kind": "stagnation",
                        "loop_guard_score": 5,
                    },
                ),
            )
        ],
    )

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == LoopRoute.FINALIZE
    assert graph_state.final_result["status"] == "failed"
    assert "Model stuck in write loop" in graph_state.final_result["reason"]
    assert graph_state.error is not None
    assert emitted
