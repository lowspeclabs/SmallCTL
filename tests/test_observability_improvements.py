from __future__ import annotations

import logging
import pytest
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

from smallctl.state import LoopState
from smallctl.models.conversation import ConversationMessage
from smallctl.context.policy import ContextPolicy
from smallctl.context.assembler import PromptAssembler
from smallctl.context.retrieval import LexicalRetriever
from smallctl.context.summarizer import ContextSummarizer
from smallctl.harness.runtime_facade import _run_teardown, _autosave_chat_session_state
from smallctl.harness.tool_dispatch import dispatch_tool_call
from smallctl.graph.model_stream import process_model_stream


class DummyHarness:
    def __init__(self, state: LoopState):
        self.state = state
        self.conversation_id = "test-session-123"
        self.log = logging.getLogger("test.dummy_harness")
        self._pending_task_shutdown_reason = ""
        self.registry = SimpleNamespace(
            names=lambda: ["shell_exec", "file_read"],
            get=lambda name: None
        )
        self.dispatcher = SimpleNamespace(
            dispatch=self._fake_dispatch
        )
        self.thinking_start_tag = "<think>"
        self.thinking_end_tag = "</think>"
        self.event_handler = None
        self.approvals = SimpleNamespace(
            reject_pending_shell_approvals=lambda: None,
            reject_pending_sudo_password_prompts=lambda: None
        )
        self._active_processes = set()

    def _runlog(self, *args, **kwargs):
        pass

    async def _fake_dispatch(self, tool_name: str, args: dict[str, Any]):
        from smallctl.models.tool_result import ToolEnvelope
        return ToolEnvelope(success=True, output="success")

    def _current_user_task(self):
        return "do a test task"


def test_context_metrics_increment():
    state = LoopState(cwd="/dummy")
    state.scratchpad = {}

    # Test PromptAssembler
    assembler = PromptAssembler()
    # Call build_messages and catch the expected exception (since frame_compiler is not mocked)
    # The counter increment happens at the very beginning of the method.
    try:
        assembler.build_messages(state=state, system_prompt="system")
    except Exception:
        pass
    assert state.scratchpad["_context_metrics"]["assemble_calls"] == 1

    # Test LexicalRetriever
    retriever = LexicalRetriever()
    retriever._record_retrieval_call(state)
    assert state.scratchpad["_context_metrics"]["retrieval_calls"] == 1

    # Test ContextSummarizer
    summarizer = ContextSummarizer()
    summarizer._record_compaction_call(state)
    assert state.scratchpad["_context_metrics"]["compaction_calls"] == 1


def test_idle_session_teardown_and_autosave_skip(caplog):
    state = LoopState(cwd="/dummy")
    state.thread_id = "thread-123"
    state.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.step_count = 0
    state.task_received_at = ""
    state.scratchpad = {
        "_fama_config": {"enabled": False},
        "_context_metrics": {"assemble_calls": 0, "retrieval_calls": 0, "compaction_calls": 0}
    }

    harness = DummyHarness(state)

    # Verify autosave exits early without calling background persistence scheduling
    autosave_called = False
    def fake_schedule(*args, **kwargs):
        nonlocal autosave_called
        autosave_called = True

    import smallctl.harness.runtime_facade
    orig_schedule = getattr(smallctl.harness.runtime_facade, "_schedule_background_persistence", None)
    try:
        smallctl.harness.runtime_facade._schedule_background_persistence = fake_schedule
        _autosave_chat_session_state(harness)
        assert not autosave_called, "Should skip autosaving when session is idle"
    finally:
        if orig_schedule:
            smallctl.harness.runtime_facade._schedule_background_persistence = orig_schedule

    # Run teardown under caplog to inspect log output
    with caplog.at_level(logging.WARNING):
        import asyncio
        asyncio.run(_run_teardown(harness))

    # Check for warnings in logs
    assert "session_idle" in caplog.text
    assert "context_pipeline_idle" in caplog.text


@pytest.mark.asyncio
async def test_session_audit_and_counters():
    state = LoopState(cwd="/dummy")
    state.thread_id = "thread-123"
    state.created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.step_count = 1
    state.task_received_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    state.scratchpad = {}

    harness = DummyHarness(state)

    # 1. Dispatch a tool call and verify counter increment
    envelope = await dispatch_tool_call(harness, "file_read", {"path": "test.txt"})
    assert state.scratchpad["_tools_dispatched"] == 1

    # 2. Mock model call processing and verify model calls counter
    graph_state = SimpleNamespace(
        pending_tool_calls=[],
        last_tool_results=[],
        last_assistant_text="",
        last_thinking_text="",
        last_usage={},
        final_result=None
    )
    deps = SimpleNamespace(harness=harness)

    async def fake_stream_loop(*args, **kwargs):
        return {"chunks": [], "salvage_partial_stream": False, "last_chunk_error_details": None,
                "stream_ended_without_done": False, "stream_ended_without_done_details": None,
                "trigger_early_4b_fallback": False, "stream_completed_cleanly": True, "first_token_time": 0.0}

    async def fake_resolve(*args, **kwargs):
        from smallctl.graph.model_stream_fallback import StreamProcessingResult
        return StreamProcessingResult(chunks=[])

    import smallctl.graph.model_stream
    orig_loop = smallctl.graph.model_stream.run_model_stream_loop
    orig_resolve = smallctl.graph.model_stream.resolve_model_stream_result
    try:
        smallctl.graph.model_stream.run_model_stream_loop = fake_stream_loop
        smallctl.graph.model_stream.resolve_model_stream_result = fake_resolve

        await process_model_stream(graph_state, deps, messages=[], tools=[])
        assert state.scratchpad["_model_calls"] == 1
    finally:
        smallctl.graph.model_stream.run_model_stream_loop = orig_loop
        smallctl.graph.model_stream.resolve_model_stream_result = orig_resolve



def test_fama_tool_exposure_logs_hidden_reasons():
    from smallctl.fama.tool_policy import fama_hidden_tool_reasons_for_exposure

    state = SimpleNamespace(
        current_phase="execute",
        scratchpad={"_fama_config": {"enabled": True}},
        stagnation_counters={},
        repair_cycle_id="",
    )
    state.scratchpad["_expose_interactive_session_tools"] = False
    schemas = [{"function": {"name": "ssh_session_start"}}]

    reasons = fama_hidden_tool_reasons_for_exposure(
        schemas,
        state=state,
        mode="loop",
        config=SimpleNamespace(fama_enabled=True),
    )

    assert reasons["ssh_session_start"] == ["interactive_ssh_tools_not_exposed"]


def test_schema_repair_decision_logs_parse_failure_and_repair_decision():
    from smallctl.graph.node_support import schema_validation_repair_decision
    from smallctl.graph.state import PendingToolCall

    state = SimpleNamespace(step_count=3, scratchpad={})
    events = []
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(schema_validation_retry_budget=2),
        _runlog=lambda event, message, **data: events.append((event, message, data)),
    )
    pending = PendingToolCall(
        tool_name="file_patch",
        args={},
        tool_call_id="call-1",
        raw_arguments="{bad",
        parser_metadata={
            "arguments_parse_error": {"kind": "malformed_json_arguments", "message": "bad json"}
        },
    )

    decision = schema_validation_repair_decision(
        harness,
        pending,
        "Tool call missing required fields",
        {"required_fields": ["replacement_text"]},
        target_path="src/app.py",
    )

    assert decision.status == "repair"
    event_names = [event for event, _message, _data in events]
    assert "tool_call_parse_failure" in event_names
    assert "schema_validation_repair_decision" in event_names
    repair_event = next(data for event, _message, data in events if event == "schema_validation_repair_decision")
    assert repair_event["status"] == "repair"
    assert repair_event["target_path"] == "src/app.py"
    assert repair_event["required_fields"] == ["replacement_text"]


def test_recovery_guidance_renders_fresh_tool_call_repair_hint():
    from smallctl.context.frame_recovery_rendering import fresh_tool_call_repair_hint_lines, render_recovery_guidance

    state = LoopState(cwd="/tmp")
    state.step_count = 3
    state.scratchpad["_last_tool_call_repair_hint"] = {
        "tool_name": "file_read",
        "step_count": 2,
        "repair_kinds": ["null_optional_to_omit"],
    }

    assert fresh_tool_call_repair_hint_lines(state) == [
        "Latest tool-call repair: file_read args repaired via null_optional_to_omit; send that shape directly next time."
    ]
    assert render_recovery_guidance(state) == fresh_tool_call_repair_hint_lines(state)


def test_write_recovery_readback_logs_recovery_decision():
    from smallctl.graph.state import GraphRunState, ToolExecutionRecord
    from smallctl.graph.write_session_recovery import _maybe_schedule_write_recovery_readback
    from smallctl.models.tool_result import ToolEnvelope

    state = LoopState(cwd="/tmp")
    state.scratchpad = {}
    graph_state = GraphRunState(loop_state=state, thread_id="t", run_mode="loop")
    events = []
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(enforce_write_recovery_readback=True),
        _runlog=lambda event, message, **data: events.append((event, message, data)),
    )
    record = ToolExecutionRecord(
        operation_id="op-1",
        tool_name="file_write",
        args={"path": "app.py"},
        tool_call_id="write_recovery_call-1",
        result=ToolEnvelope(success=True),
    )

    assert _maybe_schedule_write_recovery_readback(graph_state, harness, record) is True
    decision = next(data for event, _message, data in events if event == "recovery_decision")
    assert decision["recovery_kind"] == "write_recovery_readback"
    assert decision["tool_name"] == "file_read"
    assert decision["path"] == "app.py"


def test_task_completion_remote_verifier_logs_recovery_decision():
    from smallctl.graph.state import GraphRunState, ToolExecutionRecord
    from smallctl.graph.task_completion_outcomes import _maybe_schedule_task_complete_remote_mutation_verifier
    from smallctl.models.tool_result import ToolEnvelope

    state = LoopState(cwd="/tmp")
    state.scratchpad = {}
    graph_state = GraphRunState(loop_state=state, thread_id="t", run_mode="loop")
    events = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: events.append((event, message, data)),
    )
    record = ToolExecutionRecord(
        operation_id="op-1",
        tool_name="task_complete",
        args={"message": "done"},
        tool_call_id="call-1",
        result=ToolEnvelope(
            success=False,
            error="needs verification",
            metadata={
                "reason": "remote_mutation_requires_verification",
                "next_required_action": {
                    "tool_names": ["ssh_file_read"],
                    "required_arguments": {
                        "host": "192.0.2.1",
                        "user": "root",
                        "path": "/etc/app.conf",
                    },
                },
            },
        ),
    )

    assert _maybe_schedule_task_complete_remote_mutation_verifier(graph_state, harness, record) is True
    decision = next(data for event, _message, data in events if event == "recovery_decision")
    assert decision["recovery_kind"] == "task_complete_remote_mutation_verifier_autocontinue"
    assert decision["tool_name"] == "ssh_file_read"
    assert decision["path"] == "/etc/app.conf"


def test_task_completion_post_change_verifier_logs_recovery_decision():
    from smallctl.graph.state import GraphRunState, ToolExecutionRecord
    from smallctl.graph.task_completion_outcomes import (
        _maybe_schedule_task_complete_post_change_verifier,
    )
    from smallctl.models.tool_result import ToolEnvelope

    state = LoopState(cwd="/tmp")
    state.scratchpad = {}
    graph_state = GraphRunState(loop_state=state, thread_id="t", run_mode="loop")
    events = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: events.append((event, message, data)),
    )
    record = ToolExecutionRecord(
        operation_id="op-1",
        tool_name="task_complete",
        args={"message": "done"},
        tool_call_id="call-1",
        result=ToolEnvelope(
            success=False,
            error="Cannot complete the task until the latest file change is verified.",
            metadata={
                "reason": "post_change_verification_required",
                "next_required_action": {
                    "tool_name": "shell_exec",
                    "required_arguments": {
                        "command": "python3 -m py_compile app.py",
                    },
                },
            },
        ),
    )

    assert _maybe_schedule_task_complete_post_change_verifier(graph_state, harness, record) is True
    assert len(graph_state.pending_tool_calls) == 1
    assert graph_state.pending_tool_calls[0].tool_name == "shell_exec"
    assert graph_state.pending_tool_calls[0].args["command"] == "python3 -m py_compile app.py"
    decision = next(data for event, _message, data in events if event == "recovery_decision")
    assert decision["recovery_kind"] == "task_complete_post_change_verifier_autocontinue"
    assert decision["tool_name"] == "shell_exec"
    assert decision["command"] == "python3 -m py_compile app.py"


def test_finalize_writes_run_summary(tmp_path):
    from smallctl.harness.core_facade import _finalize
    from smallctl.logging_utils import RunLogger

    state = LoopState(cwd="/tmp")
    state.scratchpad = {}
    state.step_count = 2
    state.inactive_steps = 0
    state.token_usage = 0
    logger = RunLogger(tmp_path / "run")
    events = []
    harness = SimpleNamespace(
        state=state,
        run_logger=logger,
        checkpoint_on_exit=False,
        _pending_task_shutdown_reason="",
        _finalize_task_scope=lambda **kwargs: {
            "task_id": "task-0001",
            "summary_path": str(tmp_path / "run" / "tasks" / "task-0001" / "task_summary.json"),
            "status": kwargs.get("status"),
        },
        _runlog=lambda event, message, **data: events.append((event, message, data)),
        _record_terminal_experience=lambda result: None,
        _rewrite_active_plan_export=lambda: None,
        _persist_checkpoint=lambda result: None,
        _schedule_background_persistence=None,
    )

    result = _finalize(harness, {"status": "completed", "message": "done"})

    assert result["status"] == "completed"
    run_summary = tmp_path / "run" / "run_summary.json"
    assert run_summary.exists()
    payload = __import__("json").loads(run_summary.read_text(encoding="utf-8"))
    assert payload["summary_kind"] == "run"
    assert payload["latest_task_id"] == "task-0001"


@pytest.mark.asyncio
async def test_shell_approval_timeout_emits_event():
    from smallctl.harness.approvals import ApprovalService

    events = []

    async def _emit(handler, event):
        del handler, event

    harness = SimpleNamespace(
        allow_interactive_shell_approval=True,
        shell_approval_session_default=False,
        event_handler="handler",
        _emit=_emit,
        _runlog=lambda event, message, **data: events.append((event, message, data)),
    )
    service = ApprovalService(harness)

    result = await service.request_shell_approval(command="ls", cwd="/tmp", timeout_sec=1)

    assert result is False
    timeout_events = [e for e in events if e[0] == "approval_timeout"]
    assert len(timeout_events) == 1
    data = timeout_events[0][2]
    assert data["command"] == "ls"
    assert data["timeout_sec"] == 1


@pytest.mark.asyncio
async def test_shell_approval_uses_human_timeout_not_command_timeout():
    from smallctl.harness.approvals import ApprovalService

    events = []
    harness = SimpleNamespace(
        allow_interactive_shell_approval=True,
        shell_approval_session_default=False,
        config=SimpleNamespace(needs_human_timeout_sec=45),
        event_handler=lambda _event: None,
        _emit=AsyncMock(side_effect=lambda _handler, event: events.append(event)),
        _runlog=Mock(),
    )
    service = ApprovalService(harness)

    pending = asyncio.create_task(
        service.request_shell_approval(command="sleep 1", cwd="/tmp", timeout_sec=1)
    )
    await asyncio.sleep(0)
    event = events[0]
    assert event.data["timeout_sec"] == 45
    service.resolve_shell_approval(event.data["approval_id"], True)
    assert await pending is True


@pytest.mark.asyncio
async def test_sudo_password_timeout_emits_event():
    from smallctl.harness.approvals import ApprovalService

    events = []

    async def _emit(handler, event):
        del handler, event

    harness = SimpleNamespace(
        allow_interactive_shell_approval=True,
        shell_approval_session_default=False,
        event_handler="handler",
        _emit=_emit,
        _runlog=lambda event, message, **data: events.append((event, message, data)),
    )
    service = ApprovalService(harness)

    result = await service.request_sudo_password(command="ls", prompt_text="password:", timeout_sec=1)

    assert result is None
    timeout_events = [e for e in events if e[0] == "sudo_password_timeout"]
    assert len(timeout_events) == 1
    data = timeout_events[0][2]
    assert data["command"] == "ls"
    assert data["timeout_sec"] == 1


@pytest.mark.asyncio
async def test_ssh_tool_block_logs_classification():
    from smallctl.harness.tool_dispatch import dispatch_tool_call

    state = LoopState(cwd="/tmp")
    state.task_mode = "local_execute"
    state.scratchpad = {}
    events = []
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: ["ssh_exec"], get=lambda name: None),
        dispatcher=SimpleNamespace(dispatch=lambda tool, args: None),
        allow_interactive_shell_approval=False,
        shell_approval_session_default=False,
        config=SimpleNamespace(
            graph_dispatch_tools_timeout_sec=None,
            fama_enabled=False,
            interactive_shell_approval=False,
        ),
        _runlog=lambda event, message, **data: events.append((event, message, data)),
    )

    result = await dispatch_tool_call(harness, "ssh_exec", {"host": "remote", "command": "ls"})
    assert result.success is False
    blocked = [e for e in events if e[0] == "tool_dispatch_blocked"]
    assert len(blocked) == 1
    data = blocked[0][2]
    assert data["tool_name"] == "ssh_exec"
    assert data["tool_category"] == "ssh"
    assert data["task_mode"] == "local_execute"
    assert data["block_reason"] == "local_coding_ssh_block"


def test_retrieval_history_records_kind_and_logs():
    from smallctl.context.retrieval import LexicalRetriever
    from smallctl.state import ArtifactSnippet

    state = LoopState(cwd="/tmp")
    state.step_count = 5
    state.scratchpad = {}
    events = []
    state._runlog = lambda event, message, **data: events.append((event, message, data))

    retriever = LexicalRetriever()
    snippets = [
        ArtifactSnippet(artifact_id="A1", text="body1"),
        ArtifactSnippet(artifact_id="A2", text="body2"),
    ]
    retriever._record_retrieved_id_history(
        state, "_retrieved_artifact_history", [s.artifact_id for s in snippets], kind="artifact"
    )

    history = state.scratchpad["_retrieved_artifact_history"]
    assert len(history) == 2
    assert history[0]["id"] == "A1"
    assert history[0]["kind"] == "artifact"
    assert history[0]["retrieved_at_step"] == 5
    logged = [e for e in events if e[0] == "retrieval_history_recorded"]
    assert len(logged) == 1
    assert logged[0][2]["kind"] == "artifact"
    assert logged[0][2]["count"] == 2


def test_verifier_decision_logs_verdict():
    from smallctl.harness.tool_result_artifact_updates import _apply_verifier_and_evidence_updates
    from smallctl.models.tool_result import ToolEnvelope

    state = LoopState(cwd="/tmp")
    state.scratchpad = {}
    events = []
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(subtask_ledger_enabled=False),
        _runlog=lambda event, message, **data: events.append((event, message, data)),
    )
    service = SimpleNamespace(harness=harness)

    result = ToolEnvelope(
        success=True,
        output={"exit_code": 0, "stdout": "ok", "stderr": ""},
    )
    _apply_verifier_and_evidence_updates(
        service,
        tool_name="shell_exec",
        result=result,
        artifact=None,
        arguments={"command": "python3 -m py_compile app.py"},
    )

    decision = [e for e in events if e[0] == "verifier_decision"]
    assert len(decision) == 1
    data = decision[0][2]
    assert data["tool_name"] == "shell_exec"
    assert data["verdict"] == "pass"
    assert data["command"] == "python3 -m py_compile app.py"
