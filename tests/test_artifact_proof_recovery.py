from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

from smallctl.context import format_reused_artifact_message
from smallctl.graph.chat_progress import build_repeated_tool_loop_interrupt_payload
from smallctl.graph.state import PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_execution_recovery import handle_repeated_tool_loop
from smallctl.models.tool_result import ToolEnvelope
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


def test_repeated_successful_shell_exec_is_suppressed_and_steers_to_completion() -> None:
    harness = _make_harness()
    harness.state.run_brief.original_task = "Create temp/restart_backoff.py and verify tests pass"
    pending = PendingToolCall(
        tool_name="shell_exec",
        args={"command": "python3 -m pytest temp/restart_backoff.py -v"},
    )
    graph_state = SimpleNamespace(
        pending_tool_calls=[pending],
        last_tool_results=[
            ToolExecutionRecord(
                operation_id="op:shell_exec",
                tool_name="shell_exec",
                args=dict(pending.args),
                tool_call_id=None,
                result=ToolEnvelope(
                    success=True,
                    output={"stdout": "5 passed", "stderr": "", "exit_code": 0},
                    metadata={"verdict": "pass"},
                ),
            )
        ],
        thread_id="thread-1",
    )
    deps = SimpleNamespace(event_handler=None)

    result = asyncio.run(
        handle_repeated_tool_loop(
            harness=harness,
            graph_state=graph_state,
            deps=deps,
            pending=pending,
            repeat_error="Guard tripped: repeated tool call loop (shell_exec repeated with identical arguments after prior nudge)",
        )
    )

    assert result is None
    assert graph_state.pending_tool_calls == []
    assert graph_state.last_tool_results == []
    assert not hasattr(graph_state, "final_result")
    recovery_message = harness.state.recent_messages[-1]
    assert recovery_message.metadata["recovery_kind"] == "shell_exec_already_succeeded"
    assert "already succeeded" in recovery_message.content
    assert "task_complete" in recovery_message.content


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


def test_reused_small_complete_artifact_inlines_content(tmp_path: Path) -> None:
    content_path = tmp_path / "A0027.txt"
    content_path.write_text("SHELL=/bin/sh\n24 * * * * root run-parts /etc/cron.hourly\n", encoding="utf-8")
    artifact = ArtifactRecord(
        artifact_id="A0027",
        kind="ssh_file_read",
        source="/etc/crontab",
        created_at="2026-05-12T00:00:00+00:00",
        size_bytes=62,
        summary="crontab full file (2 lines)",
        tool_name="ssh_file_read",
        content_path=str(content_path),
        metadata={"complete_file": True, "path": "/etc/crontab", "truncated": False},
    )

    message = format_reused_artifact_message(artifact, tool_name="ssh_file_read")

    assert "Full cached content is visible below" in message
    assert "24 * * * * root run-parts /etc/cron.hourly" in message
    assert "do not call `artifact_read`" in message


def test_repeated_artifact_summary_exit_stays_blocked_after_first_nudge(tmp_path: Path) -> None:
    harness = _make_harness()
    harness.state.run_brief.original_task = "read the cron jobs and summarize them in plain English"
    harness.state.working_memory.current_goal = harness.state.run_brief.original_task
    content_path = tmp_path / "A0027.txt"
    content_path.write_text(
        "SHELL=/bin/sh\n24 * * * * root run-parts /etc/cron.hourly\n",
        encoding="utf-8",
    )
    harness.state.artifacts["A0027"] = ArtifactRecord(
        artifact_id="A0027",
        kind="ssh_file_read",
        source="/etc/crontab",
        created_at="2026-05-12T00:00:00+00:00",
        size_bytes=62,
        summary="crontab full file (2 lines)",
        tool_name="ssh_file_read",
        content_path=str(content_path),
        metadata={"complete_file": True, "path": "/etc/crontab", "truncated": False},
    )
    pending = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0027"})
    graph_state = SimpleNamespace(
        pending_tool_calls=[pending],
        last_tool_results=["stale"],
        thread_id="thread-1",
        interrupt_payload=None,
    )
    deps = SimpleNamespace(event_handler=None)
    repeat_error = "Guard tripped: repeated tool call loop (artifact_read repeated with identical arguments)"

    first = asyncio.run(
        handle_repeated_tool_loop(
            harness=harness,
            graph_state=graph_state,
            deps=deps,
            pending=pending,
            repeat_error=repeat_error,
        )
    )
    second = asyncio.run(
        handle_repeated_tool_loop(
            harness=harness,
            graph_state=graph_state,
            deps=deps,
            pending=pending,
            repeat_error=repeat_error,
        )
    )

    assert first is None
    assert second is None
    assert graph_state.interrupt_payload is None
    assert graph_state.pending_tool_calls == []
    assert harness.state.retrieval_cache[0] == "A0027"
    summary_nudges = [
        message for message in harness.state.recent_messages
        if message.metadata.get("recovery_kind") == "artifact_summary_exit"
    ]
    assert len(summary_nudges) == 1
    assert "Full content of A0027 is pinned" in summary_nudges[0].content
    assert "24 * * * * root run-parts /etc/cron.hourly" in summary_nudges[0].content
