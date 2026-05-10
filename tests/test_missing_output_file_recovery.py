from __future__ import annotations

from types import SimpleNamespace

from smallctl.context.messages import next_step_hint
from smallctl.graph.chat_progress import build_file_read_recovery_message
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_outcomes import _maybe_emit_missing_requested_output_file_nudge
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState


def test_missing_requested_output_file_read_recovery_points_to_file_write(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = (
        "SSH to root@192.168.1.89, list all mounted filesystems with size, used, available, "
        "and mountpoint, identify any filesystem above 80% usage, and print a concise summary "
        "to stdout even if you also save details to ./temp/filesystems.txt."
    )
    state.working_memory.current_goal = state.run_brief.original_task
    state.tool_history = [
        'ssh_exec|{"command": "df -h", "host": "192.168.1.89", "user": "root"}|success',
        'file_read|{"path": "./temp/filesystems.txt"}|error:File does not exist',
    ]
    harness = SimpleNamespace(state=state)

    message = build_file_read_recovery_message(
        harness,
        PendingToolCall(tool_name="file_read", args={"path": "./temp/filesystems.txt"}),
    )

    assert "requested output file" in message
    assert "file_write(path='./temp/filesystems.txt'" in message
    assert "prior successful tool output" in message
    assert "Do not call `task_complete` until the `file_write` succeeds" in message


def test_first_missing_requested_output_file_read_schedules_write_retry(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = (
        "SSH to root@192.168.1.89, list all listening TCP and UDP ports with owning "
        "processes, print a summarized exposure report to stdout, and save the full "
        "listing to ./temp/listening_ports.txt."
    )
    state.working_memory.current_goal = state.run_brief.original_task
    state.tool_history = [
        'ssh_exec|{"command": "ss -tulnp", "host": "192.168.1.89", "user": "root"}|success',
    ]
    runlog_events: list[str] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, *args, **kwargs: runlog_events.append(event),
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-missing-output", run_mode="chat")
    record = ToolExecutionRecord(
        operation_id="op-file-read",
        tool_name="file_read",
        args={"path": "./temp/listening_ports.txt"},
        tool_call_id="call-file-read",
        result=ToolEnvelope(
            success=False,
            error=f"File does not exist: {tmp_path / 'temp' / 'listening_ports.txt'}",
            metadata={
                "path": str(tmp_path / "temp" / "listening_ports.txt"),
                "read_result": "missing",
                "requested_path": "./temp/listening_ports.txt",
            },
        ),
    )

    assert _maybe_emit_missing_requested_output_file_nudge(graph_state, harness, record) is True

    recovery_message = state.recent_messages[-1]
    assert recovery_message.metadata["recovery_kind"] == "missing_requested_output_file"
    assert recovery_message.metadata["retry_tool_name"] == "file_write"
    assert recovery_message.metadata["retry_scheduled"] is True
    assert "requested output file" in recovery_message.content
    assert "file_write(path='./temp/listening_ports.txt'" in recovery_message.content
    assert state.scratchpad["_retry_tool_exposures"][0]["tool_name"] == "file_write"
    assert "missing_requested_output_file_nudge" in runlog_events


def test_missing_requested_output_file_nudge_only_emits_once(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "Run df -h and save details to ./temp/filesystems.txt."
    state.working_memory.current_goal = state.run_brief.original_task
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    graph_state = GraphRunState(loop_state=state, thread_id="thread-missing-output-once", run_mode="chat")
    record = ToolExecutionRecord(
        operation_id="op-file-read",
        tool_name="file_read",
        args={"path": "./temp/filesystems.txt"},
        tool_call_id="call-file-read",
        result=ToolEnvelope(
            success=False,
            error="File does not exist: /tmp/work/temp/filesystems.txt",
            metadata={"read_result": "missing"},
        ),
    )

    assert _maybe_emit_missing_requested_output_file_nudge(graph_state, harness, record) is True
    assert _maybe_emit_missing_requested_output_file_nudge(graph_state, harness, record) is False
    assert len(state.recent_messages) == 1


def test_summary_exit_hint_is_suppressed_when_request_requires_saved_output() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0001",
        kind="dir_list",
        source="/workspace/temp",
        created_at="2026-05-10T16:58:00+00:00",
        size_bytes=100,
        summary="temp listing",
        tool_name="dir_list",
    )
    result = ToolEnvelope(success=True, output=[], metadata={})

    hint = next_step_hint(
        artifact,
        result=result,
        request_text=(
            "List filesystems, print a concise summary, and save details to "
            "./temp/filesystems.txt."
        ),
    )

    assert "Synthesize the answer now" not in hint
