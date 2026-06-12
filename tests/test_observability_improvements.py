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
