from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.autocontinue import _DURABLE_AUTOCONTINUE_KEY
from smallctl.graph.nodes import interpret_model_output
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_call_parser_support import _detect_patch_existing_stage_read_contract_violation
from smallctl.graph.tool_outcomes import apply_tool_outcomes
from smallctl.graph.write_recovery import build_synthetic_write_args, recover_write_intent
from smallctl.graph.write_recovery_parsing import _attach_session_metadata
from smallctl.graph.write_session_patch_recovery import _maybe_schedule_patch_existing_stage_read_recovery
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState, WriteSession
from smallctl.tools.fs_sessions import infer_write_session_intent
from smallctl.write_session_fsm import archive_interrupted_write_session, new_write_session


def _make_state() -> LoopState:
    state = LoopState(cwd=str(Path.cwd()))
    state.active_tool_profiles = ["core"]
    state.artifacts = {}
    return state


def _make_harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )


def test_serialize_pending_tool_call_preserves_raw_arguments() -> None:
    from smallctl.graph.state import _serialize_pending_tool_call, _coerce_pending_tool_call

    pending = PendingToolCall(
        tool_name="file_read",
        args={"path": "./foo.py"},
        raw_arguments='{"path": "./foo.py"}',
        source="system",
    )
    serialized = _serialize_pending_tool_call(pending)
    assert serialized["raw_arguments"] == '{"path": "./foo.py"}'

    restored = _coerce_pending_tool_call(serialized)
    assert restored is not None
    assert restored.raw_arguments == '{"path": "./foo.py"}'


@pytest.mark.asyncio
async def test_system_sourced_pending_calls_bypass_interpret_model_output() -> None:
    state = _make_state()
    harness = _make_harness(state)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args={"path": "./foo.py"},
            source="system",
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = await interpret_model_output(graph_state, deps)
    assert route.value == "dispatch_tools"


def test_missing_required_args_not_overwritten_by_patch_contract_check() -> None:
    state = _make_state()
    harness = _make_harness(state)

    pending = PendingToolCall(
        tool_name="file_write",
        args={"path": "./foo.py"},
        source="model",
    )

    missing = _detect_patch_existing_stage_read_contract_violation(harness, pending)
    assert missing is None


def test_patch_existing_session_gets_replace_file_for_write_first_task(tmp_path: Path) -> None:
    target = tmp_path / "job_retry_engine.py"
    target.write_text("old content\n", encoding="utf-8")

    state = _make_state()
    state.cwd = str(tmp_path)
    state.run_brief.original_task = "Create a python script that implements a job retry engine"
    state.scratchpad["_force_chunk_mode_targets"] = [str(target)]

    intent = infer_write_session_intent(str(target), str(tmp_path))
    assert intent == "patch_existing"

    from smallctl.graph.tool_write_session_policy import _ensure_chunk_write_session

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        registry=None,
        client=SimpleNamespace(model="small-model"),
        config=SimpleNamespace(
            staged_execution_enabled=True,
        ),
    )

    session = _ensure_chunk_write_session(harness, str(target))
    assert session is not None
    assert session.write_session_intent == "replace_file"


def test_patch_existing_session_keeps_patch_existing_for_non_write_task(tmp_path: Path) -> None:
    target = tmp_path / "job_retry_engine.py"
    target.write_text("old content\n", encoding="utf-8")

    state = _make_state()
    state.cwd = str(tmp_path)
    state.run_brief.original_task = "Fix the bug in the existing job retry engine"
    state.scratchpad["_force_chunk_mode_targets"] = [str(target)]

    from smallctl.graph.tool_write_session_policy import _ensure_chunk_write_session

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        registry=None,
        client=SimpleNamespace(model="small-model"),
        config=SimpleNamespace(
            staged_execution_enabled=True,
        ),
    )

    session = _ensure_chunk_write_session(harness, str(target))
    assert session is not None
    assert session.write_session_intent == "patch_existing"


def test_chunk_mode_prearm_migrates_interrupted_stage_for_same_target(tmp_path: Path) -> None:
    target = tmp_path / "temp" / "task_queue.py"
    target.parent.mkdir(parents=True, exist_ok=True)

    state = _make_state()
    state.cwd = str(tmp_path)
    state.run_brief.original_task = "save the file"
    state.scratchpad["_force_chunk_mode_targets"] = [str(target)]

    old_session = new_write_session(
        session_id="ws_old",
        target_path=str(target),
        intent="replace_file",
        next_section="helpers",
    )
    old_stage = tmp_path / ".smallctl" / "write_sessions" / "ws_old__task_queue__stage.py"
    old_stage.parent.mkdir(parents=True, exist_ok=True)
    old_stage.write_text(
        "import heapq\n\nclass TaskQueue:\n    pass\n",
        encoding="utf-8",
    )
    old_session.write_staging_path = str(old_stage)
    old_session.write_sections_completed = ["imports", "types_interfaces"]
    old_session.write_section_ranges = {
        "imports": {"start": 0, "end": 13},
        "types_interfaces": {"start": 14, "end": 39},
    }
    state.write_session = old_session
    archive_interrupted_write_session(state, reason="task_switch_abandoned")
    state.write_session = None

    from smallctl.graph.tool_write_session_policy import _ensure_chunk_write_session

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        registry=None,
        client=SimpleNamespace(model="small-model"),
        config=SimpleNamespace(staged_execution_enabled=True),
    )

    session = _ensure_chunk_write_session(harness, str(target))

    assert session is not None
    assert session.write_session_id != "ws_old"
    assert session.write_staging_path
    assert Path(session.write_staging_path).read_text(encoding="utf-8") == old_stage.read_text(encoding="utf-8")
    assert session.write_sections_completed == ["imports", "types_interfaces"]
    assert session.write_section_ranges == old_session.write_section_ranges
    assert session.write_next_section == "helpers"
    assert any(event == "write_session_stage_migrated" for event, _message, _data in runlog_events)


def test_recovered_intent_injects_overwrite_for_patch_existing_first_write() -> None:
    state = _make_state()
    session = WriteSession(
        write_session_id="ws-test",
        write_target_path="./foo.py",
        write_session_intent="patch_existing",
        write_staging_path="",
        write_target_existed_at_start=True,
        status="open",
    )
    state.write_session = session

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        registry=None,
        config=SimpleNamespace(
            enable_write_intent_recovery=True,
            write_recovery_allow_raw_text_targets=True,
            enable_assistant_code_write_recovery=True,
            write_recovery_min_confidence="high",
        ),
    )

    intent = recover_write_intent(
        harness=harness,
        pending=PendingToolCall(
            tool_name="file_write",
            args={"path": "./foo.py", "content": "print(1)"},
        ),
    )
    assert intent is not None
    assert intent.path == "./foo.py"
    _attach_session_metadata(intent, harness=harness)
    assert intent.replace_strategy == "overwrite"
    assert "patch_existing_first_write_overwrite" in intent.evidence

    args = build_synthetic_write_args(intent)
    assert args["replace_strategy"] == "overwrite"


def test_recovered_intent_does_not_override_explicit_replace_strategy() -> None:
    state = _make_state()
    session = WriteSession(
        write_session_id="ws-test",
        write_target_path="./foo.py",
        write_session_intent="patch_existing",
        write_staging_path="",
        write_target_existed_at_start=True,
        status="open",
    )
    state.write_session = session

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        registry=None,
        config=SimpleNamespace(
            enable_write_intent_recovery=True,
            write_recovery_allow_raw_text_targets=True,
            enable_assistant_code_write_recovery=True,
            write_recovery_min_confidence="high",
        ),
    )

    intent = recover_write_intent(
        harness=harness,
        pending=PendingToolCall(
            tool_name="file_write",
            args={"path": "./foo.py", "content": "print(1)", "replace_strategy": "append"},
        ),
    )
    assert intent is not None
    assert intent.replace_strategy == "append"
    _attach_session_metadata(intent, harness=harness)
    assert intent.replace_strategy == "append"


@pytest.mark.asyncio
async def test_apply_tool_outcomes_schedules_file_read_and_routes_to_dispatch(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "logwatch.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-1-stage.py"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("print('staged')\n", encoding="utf-8")

    session = WriteSession(
        write_session_id="ws-1",
        write_target_path=str(target),
        write_session_intent="patch_existing",
        write_staging_path=str(stage),
        write_target_existed_at_start=True,
        status="open",
        write_sections_completed=[],
        write_section_ranges={},
        write_current_section="",
        write_next_section="",
        write_failed_local_patches=0,
        write_pending_finalize=False,
        write_last_verifier={},
    )
    state.write_session = session

    harness = _make_harness(state)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_write",
            args={
                "path": "./logwatch.py",
                "content": "replacement body",
                "write_session_id": "ws-1",
                "section_name": "imports",
            },
            tool_call_id="tool-1",
            result=ToolEnvelope(
                success=False,
                error="Patch-existing write sessions need an explicit first-chunk choice.",
                metadata={
                    "path": str(target),
                    "staging_path": str(stage),
                    "write_session_id": "ws-1",
                    "write_session_intent": "patch_existing",
                    "replace_strategy": "auto",
                    "staged_only": True,
                    "error_kind": "patch_existing_requires_explicit_replace_strategy",
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = await apply_tool_outcomes(graph_state, deps)

    assert route.value == "next_step"
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_read"
    assert pending.args == {"path": str(target)}
    assert pending.source == "system"
    assert state.scratchpad.get("_patch_existing_stage_read_contract") is not None


def test_replace_strategy_schema_excludes_auto() -> None:
    from smallctl.tools.register_filesystem import register_filesystem_tools

    registrations: list[dict] = []

    def make_registration(**kwargs):
        return kwargs

    def inject_cwd(handler):
        return handler

    def inject_state_and_cwd(handler):
        return handler

    register_filesystem_tools(
        register=lambda items: registrations.extend(items),
        make_registration=make_registration,
        inject_cwd=inject_cwd,
        inject_state_and_cwd=inject_state_and_cwd,
        core_profile="core",
        mutate_profile="mutate",
        support_profile="support",
    )

    file_write_reg = next(r for r in registrations if r["name"] == "file_write")
    schema = file_write_reg["schema"]
    replace_strategy = schema["properties"]["replace_strategy"]
    assert "enum" in replace_strategy
    assert set(replace_strategy["enum"]) == {"append", "overwrite"}
    assert "auto" not in replace_strategy["enum"]


def test_stale_file_patch_after_recent_success_nudges_verify_not_retry() -> None:
    from smallctl.graph.write_session_patch_recovery import _maybe_schedule_file_patch_read_recovery

    state = _make_state()
    state.tool_execution_records = {
        "op-prior": {
            "operation_id": "op-prior",
            "tool_name": "file_patch",
            "args": {
                "path": "./temp/pong.py",
                "target_text": "    curses.endwin()",
                "replacement_text": "    # removed to avoid double end",
            },
            "result": {
                "success": True,
                "output": "Patched 1 occurrence.",
                "metadata": {"path": "./temp/pong.py", "changed": True},
            },
        }
    }
    harness = _make_harness(state)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-stale-patch",
        run_mode="loop",
    )
    record = ToolExecutionRecord(
        operation_id="op-retry",
        tool_name="file_patch",
        args={
            "path": "./temp/pong.py",
            "target_text": "    curses.endwin()",
            "replacement_text": "    # removed to avoid double end",
        },
        tool_call_id="tc-retry",
        result=ToolEnvelope(
            success=False,
            error="Target text not found.",
            metadata={
                "path": "./temp/pong.py",
                "error_kind": "patch_target_not_found",
            },
        ),
    )

    scheduled = _maybe_schedule_file_patch_read_recovery(graph_state, harness, record)

    assert scheduled is True
    assert len(graph_state.pending_tool_calls) == 1
    assert graph_state.pending_tool_calls[0].tool_name == "file_read"
    assert state.recent_messages
    recovery_message = state.recent_messages[-1]
    assert recovery_message.metadata["recovery_kind"] == "file_patch_already_applied_verify"
    assert "already applied" in recovery_message.content.lower()
    assert "task_complete" in recovery_message.content
    assert "before asking for another patch" not in recovery_message.content.lower()


def test_already_applied_file_patch_autocontinues_with_traceback_focused_read(tmp_path: Path) -> None:
    from smallctl.graph.write_session_patch_recovery import _maybe_schedule_file_patch_read_recovery

    state = _make_state()
    target = tmp_path / "pong.py"
    state.cwd = str(tmp_path)
    state.tool_execution_records = {
        "op-prior": {
            "operation_id": "op-prior",
            "tool_name": "file_patch",
            "args": {
                "path": str(target),
                "target_text": "self.stdscr.addch(curses.color_pair(1), y, 1, '|')",
                "replacement_text": "self.stdscr.addch(COLOR_LEFT_PADDLE, y, 1, '|')",
            },
            "result": {
                "success": True,
                "output": "Patched 1 occurrence.",
                "metadata": {"path": str(target), "changed": True},
            },
        }
    }
    harness = _make_harness(state)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-already-applied",
        run_mode="loop",
    )
    record = ToolExecutionRecord(
        operation_id="op-retry",
        tool_name="file_patch",
        args={
            "path": str(target),
            "target_text": "self.stdscr.addch(curses.color_pair(1), y, 1, '|')",
            "replacement_text": "self.stdscr.addch(COLOR_LEFT_PADDLE, y, 1, '|')",
        },
        tool_call_id="tc-retry",
        result=ToolEnvelope(
            success=False,
            error="This exact patch already landed.",
            metadata={
                "path": str(target),
                "requested_path": str(target),
                "error_kind": "repeat_sensitive_patch_already_applied",
                "next_required_tool": {
                    "tool_name": "file_read",
                    "required_arguments": {
                        "path": str(target),
                        "start_line": 305,
                        "end_line": 320,
                    },
                },
            },
        ),
    )

    scheduled = _maybe_schedule_file_patch_read_recovery(graph_state, harness, record)

    assert scheduled is True
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_read"
    assert pending.args == {"path": str(target), "start_line": 305, "end_line": 320}
    assert state.scratchpad["_repeat_guard_one_shot_fingerprints"]
    durable_queue = state.scratchpad.get(_DURABLE_AUTOCONTINUE_KEY)
    assert isinstance(durable_queue, list)
    assert durable_queue[-1]["args"] == pending.args
    recovery_message = state.recent_messages[-1]
    assert recovery_message.metadata["recovery_kind"] == "file_patch_already_applied_verify"
    assert "already applied" in recovery_message.content.lower()
