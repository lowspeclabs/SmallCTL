from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from pathlib import Path
import pytest

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.write_recovery import (
    recover_write_intent,
    _maybe_prepend_existing_content,
    build_synthetic_write_args,
)
from smallctl.graph.tool_outcomes import (
    _maybe_schedule_write_recovery_readback,
)
from smallctl.graph.state import GraphRunState, ToolExecutionRecord, PendingToolCall
from smallctl.graph.tool_execution_nodes import dispatch_tools
from smallctl.models.tool_result import ToolEnvelope
from smallctl.write_session_fsm import new_write_session
from smallctl.state import LoopState
from smallctl.tools.dispatcher import ToolDispatcher

async def _fake_invoke(**kwargs):
    return {"success": True}

class _FakeHarness:
    def __init__(self, session=None, cwd=None):
        self.state = SimpleNamespace(
            write_session=session,
            cwd=cwd or "/tmp/workspace",
            scratchpad={},
            recent_messages=[],
            touch=lambda: None,
            current_phase="explore",
            append_message=lambda msg: self.state.recent_messages.append(msg)
        )
        self.config = SimpleNamespace(
            enable_write_intent_recovery=True,
            write_recovery_allow_raw_text_targets=True,
            enable_assistant_code_write_recovery=True,
            write_recovery_min_confidence="high",
            enforce_write_recovery_readback=True
        )
        self.log = SimpleNamespace(
            warning=lambda *args: None,
            exception=lambda *args: None,
            info=lambda *args: None
        )
        self.registry = SimpleNamespace(
            get=lambda name: SimpleNamespace(
                tool_name=name,
                category="core",
                risk="low",
                tier="warm",
                phase_allowed=lambda p: True,
                schema={"type": "object", "properties": {}, "required": []},
                invoke=_fake_invoke
            )
        )
    
    def _runlog(self, *args, **kwargs):
        pass


def test_fix_d_append_semantics_merge(tmp_path):
    # Setup: existing file
    target_file = tmp_path / "app.py"
    target_file.write_text("import os\n", encoding="utf-8")
    
    harness = _FakeHarness(cwd=str(tmp_path))
    
    # Simulate a recovered intent from a 'file_append' tool call
    partial_tool_calls = [
        {
            "id": "tc_1",
            "type": "function",
            "function": {
                "name": "file_append",
                "arguments": json.dumps({"path": "app.py"})
            }
        }
    ]
    
    intent = recover_write_intent(
        harness=harness,
        pending=None,
        assistant_text="[file_write] {\"path\": \"app.py\", \"content\": \"import sys\\n\"}",
        partial_tool_calls=partial_tool_calls
    )
    
    assert intent._is_append is True
    assert intent.path == "app.py"
    assert intent.content == "import sys\n"
    
    # Apply prepend logic
    _maybe_prepend_existing_content(intent, harness=harness)
    
    assert intent.content == "import os\nimport sys\n"

def test_fix_a_readback_gate_scheduling():
    harness = _FakeHarness()
    graph_state = GraphRunState(
        loop_state=LoopState(thread_id="test_thread"),
        thread_id="test_thread",
        run_mode="loop"
    )
    
    record = ToolExecutionRecord(
        operation_id="op_1",
        tool_name="file_write",
        args={"path": "main.py", "content": "print('hello')"},
        tool_call_id="write_recovery_ws123", # recovered prefix
        result=ToolEnvelope(success=True, output="File written.")
    )
    
    # Initially no pending calls
    assert len(graph_state.pending_tool_calls) == 0
    
    _maybe_schedule_write_recovery_readback(graph_state, harness, record)
    
    # Should have scheduled a file_read
    assert len(graph_state.pending_tool_calls) == 1
    assert graph_state.pending_tool_calls[0].tool_name == "file_read"
    assert graph_state.pending_tool_calls[0].args["path"] == "main.py"
    assert harness.state.scratchpad["_write_recovery_readback_scheduled"] == "write_recovery_ws123|main.py|write_recovery_readback"

def test_fix_b_repair_phase_task_complete_allowed():
    import asyncio
    harness = _FakeHarness()
    harness.state.current_phase = "repair"
    dispatcher = ToolDispatcher(
        registry=harness.registry,
        state=harness.state,
        phase=harness.state.current_phase
    )
    
    # 'task_complete' is now allowed in repair phase
    envelope = asyncio.run(dispatcher.dispatch("task_complete", {}))

    assert envelope.success is True

def test_fix_c_exit_code_in_compaction():
    # surgical test of the logic we added in tool_results.py and __init__.py
    exit_code = 1
    compact = "[artifact A1]"
    status_tag = "EXIT_CODE=0" if exit_code == 0 else f"EXIT_CODE={exit_code} (FAILED)"
    compact_with_code = f"{status_tag}\n{compact}"
    
    assert "EXIT_CODE=1 (FAILED)" in compact_with_code
    
    exit_code = 0
    status_tag = "EXIT_CODE=0" if exit_code == 0 else f"EXIT_CODE={exit_code} (FAILED)"
    assert "EXIT_CODE=0" in status_tag


def test_declined_write_recovery_empty_payload_is_blocked_before_dispatch(tmp_path: Path):
    state = LoopState(cwd=str(tmp_path))
    state.write_session = new_write_session(
        session_id="ws_3a106b",
        target_path="./temp/patch_dependency_sim.py",
        intent="replace_file",
        suggested_sections=["imports", "types_interfaces"],
        next_section="imports",
    )
    state.scratchpad["_last_write_recovery"] = {
        "status": "declined",
        "path": "./temp/patch_dependency_sim.py",
        "write_session_id": "ws_3a106b",
        "content_chars": 0,
    }
    dispatched: list[tuple[str, dict[str, object]]] = []
    emitted: list[object] = []
    runlog_events: list[tuple[str, dict[str, object]]] = []

    async def _emit(_handler, event) -> None:
        emitted.append(event)

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="unexpected dispatch")

    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(schema_validation_max_repair_attempts=2),
        registry=SimpleNamespace(names=lambda: {"file_write"}, get=lambda _name: None),
        log=logging.getLogger("test.write_recovery.empty_payload_dispatch_guard"),
        _runlog=lambda event, _message, **kwargs: runlog_events.append((event, kwargs)),
        _emit=_emit,
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
        _failure=lambda reason, error_type, details=None: {
            "status": "failed",
            "reason": reason,
            "error": {"type": error_type, "details": details or {}},
        },
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-empty-write-dispatch-guard",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_write",
                args={
                    "write_session_id": "ws_3a106b",
                    "path": "./temp/patch_dependency_sim.py",
                    "section_name": "imports",
                },
                tool_call_id="GHkveuh5utkC5f7CZikgU1kajtErwA0O",
                raw_arguments=json.dumps(
                    {
                        "write_session_id": "ws_3a106b",
                        "path": "./temp/patch_dependency_sim.py",
                        "section_name": "imports",
                    },
                    ensure_ascii=True,
                ),
            )
        ],
    )

    asyncio.run(dispatch_tools(graph_state, GraphRuntimeDeps(harness=harness, event_handler=None)))

    assert dispatched == []
    assert graph_state.pending_tool_calls == []
    assert len(graph_state.last_tool_results) == 1
    assert graph_state.last_tool_results[0].result.status == "blocked"
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["recovery_kind"] == "schema_validation"
    assert state.recent_messages[-1].metadata["required_fields"] == ["content"]
    assert state.scratchpad["_last_write_session_schema_failure"]["required_fields"] == ["content"]
    assert any(event == "tool_call_validation_pre_dispatch_block" for event, _ in runlog_events)

