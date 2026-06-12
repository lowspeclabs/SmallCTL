from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.context.retrieval import LexicalRetriever
from smallctl.graph.state import GraphRunState
from smallctl.graph.tool_outcomes import _maybe_request_apt_deb822_validator
from smallctl.models.events import UIEvent, UIEventType
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ExperienceMemory, LoopState


def test_remote_ssh_recovery_prefers_actionable_ssh_memories_over_generic_completion() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 7
    state.current_phase = "repair"
    state.task_mode = "remote_execute"
    state.active_intent = "requested_ssh_exec"
    state.last_failure_class = "remote_interactive_stall"
    state.run_brief.original_task = "ssh to the remote host and recover the apt install after tty and noninteractive retries stalled"
    state.working_memory.current_goal = "recover the remote apt installer"
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-complete",
            intent="general_task",
            tool_name="task_complete",
            outcome="success",
            notes="Remote task completed successfully.",
            confidence=0.9,
        ),
        ExperienceMemory(
            memory_id="mem-ssh",
            intent="requested_ssh_exec",
            tool_name="ssh_session_start",
            outcome="success",
            notes="Open an interactive SSH session first when apt needs a tty and noninteractive retries stall.",
            environment_tags=["tty", "noninteractive", "apt"],
            confidence=0.9,
        ),
    ]

    ranked = LexicalRetriever()._rank_experiences(state=state)

    assert ranked
    assert ranked[0][1].tool_name == "ssh_session_start"


def test_apt_deb822_preflight_creates_approval_interrupt_instead_of_autocontinue() -> None:
    events: list[UIEvent] = []
    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.active_tool_profiles = ["network"]

    async def _emit(handler, event):
        if handler is not None:
            handler(event)

    harness = SimpleNamespace(
        state=state,
        _emit=_emit,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
    )
    deps = SimpleNamespace(harness=harness, event_handler=lambda event: events.append(event))
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="loop")
    record = SimpleNamespace(
        tool_name="ssh_exec",
        tool_call_id="call-1",
        operation_id="op-1",
        result=ToolEnvelope(
            success=False,
            error="APT blocked pending deb822 validation.",
            metadata={"reason": "apt_deb822_preflight_required", "host": "192.168.1.162", "user": "root"},
        ),
    )

    created = asyncio.run(_maybe_request_apt_deb822_validator(graph_state, deps, record))

    assert created is True
    assert graph_state.pending_tool_calls == []
    assert state.pending_interrupt["kind"] == "apt_deb822_validator_approval"
    assert state.pending_interrupt["tool_name"] == "ssh_exec"
    assert "debian.sources" in state.pending_interrupt["question"]
    assert any(event.event_type == UIEventType.ALERT for event in events)
    assert any(event == "apt_deb822_validator_approval_requested" for event, _message, _data in runlog_events)
