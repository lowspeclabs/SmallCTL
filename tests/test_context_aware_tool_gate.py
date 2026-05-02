from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.interpret_nodes import interpret_chat_output, interpret_model_output
from smallctl.graph.routing import LoopRoute
from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.graph.tool_execution_nodes import dispatch_tools
from smallctl.harness.tool_visibility import resolve_turn_tool_exposure
from smallctl.models.conversation import ConversationMessage
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState


def _tool_schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {},
        },
    }


def _make_registry(*tool_names: str):
    names = list(tool_names)
    return SimpleNamespace(
        names=lambda: set(names),
        get=lambda _name: SimpleNamespace(schema={}),
        export_openai_tools=lambda **kwargs: [_tool_schema(name) for name in names],
    )


async def _emit(*args, **kwargs) -> None:
    del args, kwargs


def test_interpret_blocks_registered_but_hidden_model_tool_and_keeps_allowed_loop_tool(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.run_brief.original_task = "Investigate the current workspace"
    state.working_memory.current_goal = state.run_brief.original_task

    harness = SimpleNamespace(
        state=state,
        registry=_make_registry("file_read", "plan_export"),
        summarizer=None,
        summarizer_client=None,
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _extract_planning_request=lambda _task: None,
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-loop-hidden-tool",
        run_mode="loop",
        pending_tool_calls=[
            PendingToolCall(tool_name="plan_export", args={"format": "markdown"}),
            PendingToolCall(tool_name="file_read", args={"path": "README.md"}),
        ],
    )

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.DISPATCH_TOOLS
    assert [pending.tool_name for pending in graph_state.pending_tool_calls] == ["file_read"]
    assert state.recent_messages
    recovery_message = state.recent_messages[-1]
    assert recovery_message.metadata["recovery_kind"] == "tool_not_exposed_this_turn"
    assert "plan_export" in recovery_message.content
    assert "no active plan" in recovery_message.content
    assert "file_read" in recovery_message.content


def test_hidden_chat_write_tool_schedules_minimal_retry_exposure(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.run_brief.original_task = "Inspect the current file and explain what should change"
    state.working_memory.current_goal = state.run_brief.original_task

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        registry=SimpleNamespace(
            names=lambda: {"file_read", "file_patch"},
            get=lambda _name: None,
            export_openai_tools=lambda **kwargs: [_tool_schema("file_read"), _tool_schema("file_patch")],
        ),
        summarizer=None,
        summarizer_client=None,
        _current_user_task=lambda: "inspect the current file and explain what should change",
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _extract_planning_request=lambda _task: None,
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-chat-hidden-write-retry",
        run_mode="chat",
        pending_tool_calls=[
            PendingToolCall(tool_name="file_patch", args={"path": "src/app.py"}, source="model"),
        ],
    )

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert state.recent_messages
    recovery_message = state.recent_messages[-1]
    assert recovery_message.metadata["recovery_kind"] == "tool_not_exposed_this_turn"
    assert recovery_message.metadata["retry_tool_name"] == "file_patch"
    assert "Retry on the next turn with `file_patch` immediately." in recovery_message.content
    assert "file_patch" in resolve_turn_tool_exposure(harness, "chat")["names"]


def test_dispatch_hard_blocks_hidden_model_tool_in_chat_mode(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.run_brief.original_task = "Inspect this log and explain what failed"
    state.working_memory.current_goal = state.run_brief.original_task

    dispatched: list[tuple[str, dict[str, object]]] = []
    runlog_events: list[str] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="unexpected dispatch")

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        registry=SimpleNamespace(
            names=lambda: {"file_read", "shell_exec"},
            get=lambda _name: None,
            export_openai_tools=lambda **kwargs: [_tool_schema("file_read"), _tool_schema("shell_exec")],
        ),
        _current_user_task=lambda: "inspect this log and explain what failed",
        _runlog=lambda event, *args, **kwargs: runlog_events.append(event),
        _emit=_emit,
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
        log=logging.getLogger("test.context_aware_tool_gate.chat"),
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-chat-hidden-tool",
        run_mode="chat",
        pending_tool_calls=[
            PendingToolCall(tool_name="shell_exec", args={"command": "pwd"}, source="model"),
        ],
    )

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == []
    assert len(graph_state.last_tool_results) == 1
    result = graph_state.last_tool_results[0].result
    assert result.success is False
    assert result.metadata["reason"] == "tool_not_exposed_this_turn"
    assert result.metadata["tool_name"] == "shell_exec"
    assert result.metadata["run_mode"] == "chat"
    assert result.metadata["allowed_tools"] == ["file_read"]
    assert "tool_blocked_not_exposed" in runlog_events


def test_dispatch_hidden_file_write_schedules_retry_exposure_for_next_chat_turn(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.run_brief.original_task = "inspect this log and explain what failed"
    state.working_memory.current_goal = state.run_brief.original_task

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        return ToolEnvelope(success=True, output={"tool": tool_name, "args": args})

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        registry=SimpleNamespace(
            names=lambda: {"file_read", "file_write"},
            get=lambda _name: None,
            export_openai_tools=lambda **kwargs: [_tool_schema("file_read"), _tool_schema("file_write")],
        ),
        _current_user_task=lambda: "inspect this log and explain what failed",
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
        log=logging.getLogger("test.context_aware_tool_gate.chat_retry"),
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-chat-hidden-file-write-retry",
        run_mode="chat",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_write",
                args={"path": "report.md", "content": "# Report\n"},
                source="model",
            ),
        ],
    )

    asyncio.run(dispatch_tools(graph_state, deps))

    assert len(graph_state.last_tool_results) == 1
    result = graph_state.last_tool_results[0].result
    assert result.success is False
    assert result.metadata["reason"] == "tool_not_exposed_this_turn"
    assert result.metadata["retry_scheduled"] is True
    assert state.recent_messages
    recovery_message = state.recent_messages[-1]
    assert recovery_message.metadata["retry_tool_name"] == "file_write"
    assert "Retry on the next turn with `file_write` immediately." in recovery_message.content
    assert "file_write" in resolve_turn_tool_exposure(harness, "chat")["names"]


def test_hidden_loop_web_fetch_points_to_artifact_read_and_schedules_retry(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core", "network"]
    state.run_brief.original_task = "ssh into the remote host and fix nginx"
    state.working_memory.current_goal = (
        "Continue remote task over SSH on root@192.168.1.63. "
        "User follow-up: debug nginx and do a websearch first"
    )
    state.artifacts["A0007"] = ArtifactRecord(
        artifact_id="A0007",
        kind="web_fetch",
        source="https://example.com/nginx-fix",
        created_at="2026-05-01T00:00:00+00:00",
        size_bytes=256,
        summary="Fetched nginx fix article",
        tool_name="web_fetch",
    )
    state.scratchpad["_last_task_handoff"] = {
        "effective_task": state.working_memory.current_goal,
        "current_goal": state.working_memory.current_goal,
        "recent_research_artifact_ids": ["A0007"],
    }

    def _registry_get(name: str):
        if name == "web_fetch":
            return SimpleNamespace(schema={}, openai_schema=lambda: _tool_schema("web_fetch"))
        return None

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        registry=SimpleNamespace(
            names=lambda: {"file_read", "web_fetch"},
            get=_registry_get,
            export_openai_tools=lambda **kwargs: [_tool_schema("file_read")],
        ),
        summarizer=None,
        summarizer_client=None,
        _current_user_task=lambda: "ssh into the remote host and fix nginx",
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _extract_planning_request=lambda _task: None,
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-loop-hidden-web-fetch",
        run_mode="loop",
        pending_tool_calls=[
            PendingToolCall(tool_name="web_fetch", args={"result_id": "r1"}, source="model"),
        ],
    )

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert state.recent_messages
    recovery_message = state.recent_messages[-1]
    assert recovery_message.metadata["recovery_kind"] == "tool_not_exposed_this_turn"
    assert recovery_message.metadata["retry_tool_name"] == "web_fetch"
    assert "artifact_read(artifact_id='A0007')" in recovery_message.content
    assert "Retry on the next turn with `web_fetch` immediately." in recovery_message.content
    assert "web_fetch" in resolve_turn_tool_exposure(harness, "loop")["names"]


def test_dispatch_allows_system_recovery_call_to_bypass_hidden_tool_gate_in_planning_mode(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.run_brief.original_task = "Draft a plan"
    state.working_memory.current_goal = state.run_brief.original_task

    dispatched: list[tuple[str, dict[str, object]]] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output={"status": "ok"})

    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(
            names=lambda: {"file_read", "plan_export"},
            get=lambda _name: None,
            export_openai_tools=lambda **kwargs: [_tool_schema("file_read"), _tool_schema("plan_export")],
        ),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
        log=logging.getLogger("test.context_aware_tool_gate.planning"),
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-planning-system-bypass",
        run_mode="planning",
        pending_tool_calls=[
            PendingToolCall(tool_name="plan_export", args={"format": "markdown"}, source="system"),
        ],
    )

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == [("plan_export", {"format": "markdown"})]
    assert graph_state.last_tool_results
    assert graph_state.last_tool_results[0].result.success is True


def test_interpret_chat_ignores_hidden_task_complete_when_chat_turn_is_terminal_only(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.run_brief.original_task = "hello"
    state.working_memory.current_goal = "hello"
    state.scratchpad["_chat_tools_suppressed_reason"] = "non_lookup_chat_terminal_only"

    runlog_events: list[str] = []
    harness = SimpleNamespace(
        state=state,
        registry=_make_registry("task_complete", "task_fail"),
        _chat_mode_tools=lambda: [],
        _runlog=lambda event, *args, **kwargs: runlog_events.append(event),
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-chat-hello",
        run_mode="chat",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="task_complete",
                args={"message": "Hello!"},
                source="model",
            ),
        ],
        last_assistant_text="Hello! It's nice to meet you.",
    )

    route = asyncio.run(interpret_chat_output(graph_state, deps))

    assert route == LoopRoute.FINALIZE
    assert graph_state.pending_tool_calls == []
    assert graph_state.final_result is not None
    assert graph_state.final_result["status"] == "chat_completed"
    assert graph_state.final_result["assistant"] == "Hello! It's nice to meet you."
    assert "chat_hidden_terminal_tool_ignored" in runlog_events


def test_interpret_model_output_does_not_auto_finalize_non_chat_execution_with_tool_evidence(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.task_mode = "remote_execute"
    state.active_tool_profiles = ["core"]
    state.run_brief.original_task = "ssh into 192.168.1.63 and use pubkey auth"
    state.working_memory.current_goal = state.run_brief.original_task
    state.working_memory.known_facts = ["ssh authentication failed using the prior credential"]
    state.recent_messages = [
        ConversationMessage(
            role="tool",
            content="root@192.168.1.63: Permission denied (publickey,password).",
        )
    ]
    state.recent_errors.append("ssh_exec: root@192.168.1.63: Permission denied (publickey,password).")

    runlog_events: list[str] = []
    harness = SimpleNamespace(
        state=state,
        registry=_make_registry("shell_exec", "task_complete"),
        summarizer=None,
        summarizer_client=None,
        _runlog=lambda event, *args, **kwargs: runlog_events.append(event),
        _record_experience=lambda *args, **kwargs: None,
        _emit=_emit,
        _extract_planning_request=lambda _task: None,
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-loop-no-auto-finalize",
        run_mode="loop",
        pending_tool_calls=[],
        last_assistant_text=(
            "The prior SSH password appears invalid. I'll try a different approach next by "
            "checking local key material and preparing a pubkey-auth flow."
        ),
        last_thinking_text="I should execute shell commands to gather SSH key state next.",
    )

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert graph_state.final_result is None
    assert "auto_finalize" not in runlog_events
