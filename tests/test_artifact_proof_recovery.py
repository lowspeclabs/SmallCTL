from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from smallctl.graph.chat_progress import build_repeated_tool_loop_interrupt_payload
from smallctl.graph.state import PendingToolCall
from smallctl.graph.tool_execution_recovery import handle_repeated_tool_loop
from smallctl.state import ArtifactRecord, LoopState


def _make_harness(model_name: str = "qwen2.5-coder-7b-instruct") -> SimpleNamespace:
    state = LoopState(cwd=".")
    state.current_phase = "explore"
    state.active_tool_profiles = ["core"]
    state.scratchpad["_model_name"] = model_name
    state.run_brief.original_task = "Summarize the artifact output in a table"
    state.working_memory.current_goal = "Summarize the artifact output in a table"
    state.recent_errors.append(
        "artifact_print: Artifact E-A0001 not found in state. "
        "No artifacts exist in the current session state. Re-execute the original tool call."
    )
    return SimpleNamespace(
        state=state,
        client=SimpleNamespace(model=model_name),
        log=logging.getLogger("test.artifact_proof_recovery"),
        _runlog=lambda *args, **kwargs: None,
    )


def test_repeated_artifact_loop_interrupt_prefers_missing_evidence_guidance() -> None:
    harness = _make_harness()
    graph_state = SimpleNamespace(thread_id="thread-1")
    pending = PendingToolCall(tool_name="artifact_print", args={"artifact_id": "E-A0001"})

    payload = build_repeated_tool_loop_interrupt_payload(
        harness=harness,
        graph_state=graph_state,
        pending=pending,
        repeat_error="Guard tripped: repeated tool call loop (artifact_print repeated with identical arguments)",
    )

    assert "The artifact you requested, `E-A0001`, does not exist in this session." in payload["guidance"]
    assert "unavailable in the current session state" in payload["guidance"]
    assert "Do not claim or infer" in payload["guidance"]
    assert "Produce the requested table or summary now" not in payload["guidance"]


def test_repeated_missing_artifact_loop_injects_non_synthesis_nudge() -> None:
    harness = _make_harness()
    pending = PendingToolCall(tool_name="artifact_print", args={"artifact_id": "E-A0001"})
    graph_state = SimpleNamespace(
        pending_tool_calls=[pending],
        last_tool_results=["stale"],
        thread_id="thread-1",
    )
    deps = SimpleNamespace(event_handler=None)

    result = asyncio.run(
        handle_repeated_tool_loop(
            harness=harness,
            graph_state=graph_state,
            deps=deps,
            pending=pending,
            repeat_error="Guard tripped: repeated tool call loop (artifact_print repeated with identical arguments)",
        )
    )

    assert result is None
    assert graph_state.pending_tool_calls == []
    assert graph_state.last_tool_results == []
    assert harness.state.recent_messages
    recovery_message = harness.state.recent_messages[-1]
    assert recovery_message.role == "system"
    assert recovery_message.metadata["recovery_kind"] == "artifact_missing_evidence"
    assert "The artifact you requested, `E-A0001`, does not exist in this session." in recovery_message.content
    assert "cannot verify the claim from the current session" in recovery_message.content


def test_missing_artifact_guidance_skips_small_model_specific_sentence_for_9b() -> None:
    harness = _make_harness(model_name="wrench-9b")
    graph_state = SimpleNamespace(thread_id="thread-1")
    pending = PendingToolCall(tool_name="artifact_print", args={"artifact_id": "E-A0001"})

    payload = build_repeated_tool_loop_interrupt_payload(
        harness=harness,
        graph_state=graph_state,
        pending=pending,
        repeat_error="Guard tripped: repeated tool call loop (artifact_print repeated with identical arguments)",
    )

    assert "The artifact you requested" not in payload["guidance"]
    assert "unavailable in the current session state" in payload["guidance"]


def test_repeated_web_artifact_loop_interrupt_prefers_synthesis_guidance() -> None:
    harness = _make_harness()
    harness.state.run_brief.original_task = (
        "Do a websearch on turbo quant, what it is, how it works, then present a detailed summary of your findings"
    )
    harness.state.working_memory.current_goal = harness.state.run_brief.original_task
    harness.state.artifacts["A0002"] = ArtifactRecord(
        artifact_id="A0002",
        kind="web_fetch",
        source="https://research.google/blog/turboquant",
        created_at="2026-04-27T00:00:00+00:00",
        size_bytes=2048,
        summary="TurboQuant article excerpt",
        tool_name="web_fetch",
    )
    graph_state = SimpleNamespace(thread_id="thread-1")
    pending = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0002", "start_line": 1, "end_line": 50})

    payload = build_repeated_tool_loop_interrupt_payload(
        harness=harness,
        graph_state=graph_state,
        pending=pending,
        repeat_error="Guard tripped: repeated tool call loop (artifact_read repeated with identical arguments)",
    )

    assert "You already have enough evidence from artifact A0002" in payload["guidance"]
    assert "task_complete(message='...')" in payload["guidance"]


def test_repeated_web_artifact_loop_injects_summary_exit_nudge() -> None:
    harness = _make_harness()
    harness.state.run_brief.original_task = (
        "Do a websearch on turbo quant, what it is, how it works, then present a detailed summary of your findings"
    )
    harness.state.working_memory.current_goal = harness.state.run_brief.original_task
    harness.state.artifacts["A0002"] = ArtifactRecord(
        artifact_id="A0002",
        kind="web_fetch",
        source="https://research.google/blog/turboquant",
        created_at="2026-04-27T00:00:00+00:00",
        size_bytes=2048,
        summary="TurboQuant article excerpt",
        tool_name="web_fetch",
    )
    pending = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0002", "start_line": 1, "end_line": 50})
    graph_state = SimpleNamespace(
        pending_tool_calls=[pending],
        last_tool_results=["stale"],
        thread_id="thread-1",
    )
    deps = SimpleNamespace(event_handler=None)

    result = asyncio.run(
        handle_repeated_tool_loop(
            harness=harness,
            graph_state=graph_state,
            deps=deps,
            pending=pending,
            repeat_error="Guard tripped: repeated tool call loop (artifact_read repeated with identical arguments)",
        )
    )

    assert result is None
    assert graph_state.pending_tool_calls == []
    assert graph_state.last_tool_results == []
    recovery_message = harness.state.recent_messages[-1]
    assert recovery_message.role == "system"
    assert recovery_message.metadata["recovery_kind"] == "artifact_summary_exit"
    assert "task_complete(message='...')" in recovery_message.content
