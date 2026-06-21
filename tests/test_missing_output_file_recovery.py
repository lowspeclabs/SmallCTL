from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.context.messages import next_step_hint
from smallctl.graph.chat_progress import build_file_read_recovery_message
from smallctl.graph.interpret_nodes import _working_memory_signals_completion
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_outcomes import (
    _maybe_clear_missing_input_after_local_alias_read,
    _maybe_clear_missing_input_after_remote_readback,
    _maybe_emit_missing_requested_output_file_nudge,
)
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState
from smallctl.tools.control import task_complete
from smallctl.tools.fs_listing import file_read


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


def test_remote_missing_requested_output_file_read_schedules_ssh_readback(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.task_mode = "remote_execute"
    state.run_brief.original_task = (
        "SSH to root@192.168.1.89, gather incident triage, print a concise summary, "
        "and save evidence to ./temp/incident_triage.txt."
    )
    state.working_memory.current_goal = state.run_brief.original_task
    state.tool_history = [
        'ssh_exec|{"command": "cat >> /home/stephen/Scripts/Harness-Redo/temp/incident_triage.txt", "host": "192.168.1.89", "user": "root"}|success',
    ]
    runlog_events: list[str] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, *args, **kwargs: runlog_events.append(event),
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-remote-missing-output", run_mode="chat")
    record = ToolExecutionRecord(
        operation_id="op-local-file-read-remote-output",
        tool_name="file_read",
        args={"path": "./temp/incident_triage.txt"},
        tool_call_id="call-local-file-read-remote-output",
        result=ToolEnvelope(
            success=False,
            error=f"File does not exist: {tmp_path / 'temp' / 'incident_triage.txt'}",
            metadata={
                "path": str(tmp_path / "temp" / "incident_triage.txt"),
                "read_result": "missing",
                "requested_path": "./temp/incident_triage.txt",
            },
        ),
    )

    assert _maybe_emit_missing_requested_output_file_nudge(graph_state, harness, record) is True

    recovery_message = state.recent_messages[-1]
    assert recovery_message.metadata["recovery_kind"] == "missing_remote_requested_output_file"
    assert recovery_message.metadata["retry_tool_name"] == "ssh_file_read"
    assert recovery_message.metadata["retry_scheduled"] is True
    assert "local workspace" in recovery_message.content
    assert "ssh_file_read" in recovery_message.content
    assert "_unresolved_missing_input_file" not in state.scratchpad
    assert state.scratchpad["_retry_tool_exposures"][0]["tool_name"] == "ssh_file_read"
    assert "missing_remote_requested_output_file_nudge" in runlog_events


def test_successful_remote_readback_clears_stale_local_missing_input_blocker(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.task_mode = "remote_execute"
    state.run_brief.original_task = (
        "SSH to root@192.168.1.89, gather incident triage, print a concise summary, "
        "and save evidence to ./temp/incident_triage.txt."
    )
    state.working_memory.current_goal = state.run_brief.original_task
    state.scratchpad["_unresolved_missing_input_file"] = {
        "path": "home/stephen/Scripts/Harness-Redo/temp/incident_triage.txt"
    }
    state.scratchpad["_missing_input_file_nudged"] = (
        "missing-input-file:home/stephen/Scripts/Harness-Redo/temp/incident_triage.txt"
    )
    runlog_events: list[str] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, *args, **kwargs: runlog_events.append(event),
    )
    record = ToolExecutionRecord(
        operation_id="op-ssh-file-read-output",
        tool_name="ssh_file_read",
        args={"path": "/home/stephen/Scripts/Harness-Redo/temp/incident_triage.txt"},
        tool_call_id="call-ssh-file-read-output",
        result=ToolEnvelope(
            success=True,
            output={"content": "incident triage"},
            metadata={
                "path": "/home/stephen/Scripts/Harness-Redo/temp/incident_triage.txt",
                "host": "192.168.1.89",
                "complete_file": True,
            },
        ),
    )

    assert _maybe_clear_missing_input_after_remote_readback(harness, record) is True

    assert "_unresolved_missing_input_file" not in state.scratchpad
    assert "_missing_input_file_nudged" not in state.scratchpad
    assert "missing_required_input_file_blocker_cleared_by_ssh_file_read" in runlog_events


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


def test_missing_required_input_file_recovery_suggests_nearby_match(tmp_path) -> None:
    (tmp_path / "foginstall.txt").write_text("install notes", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = (
        'read fogxinstall.txt and follow the instructions to install fog server on root@192.168.1.89'
    )
    state.working_memory.current_goal = state.run_brief.original_task
    harness = SimpleNamespace(state=state)

    message = build_file_read_recovery_message(
        harness,
        PendingToolCall(tool_name="file_read", args={"path": "fogxinstall.txt"}),
    )

    assert "required input file" in message
    assert "does not exist" in message
    assert "`foginstall.txt`" in message
    assert "Do not claim the task is complete" in message


def test_missing_required_input_file_nudge_records_completion_blocker(tmp_path) -> None:
    (tmp_path / "foginstall.txt").write_text("install notes", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "read fogxinstall.txt and install FOG"
    state.working_memory.current_goal = state.run_brief.original_task
    runlog_events: list[str] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, *args, **kwargs: runlog_events.append(event),
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-missing-input", run_mode="chat")
    record = ToolExecutionRecord(
        operation_id="op-file-read-missing-input",
        tool_name="file_read",
        args={"path": "fogxinstall.txt"},
        tool_call_id="call-file-read-missing-input",
        result=ToolEnvelope(
            success=False,
            error=f"File does not exist: {tmp_path / 'fogxinstall.txt'}",
            metadata={
                "path": str(tmp_path / "fogxinstall.txt"),
                "read_result": "missing",
                "requested_path": "fogxinstall.txt",
            },
        ),
    )

    assert _maybe_emit_missing_requested_output_file_nudge(graph_state, harness, record) is True

    recovery_message = state.recent_messages[-1]
    assert recovery_message.metadata["recovery_kind"] == "missing_required_input_file"
    assert state.scratchpad["_unresolved_missing_input_file"]["path"] == "fogxinstall.txt"
    assert "missing_required_input_file_nudge" in runlog_events


def test_missing_required_input_file_nudge_records_high_confidence_alias(tmp_path) -> None:
    (tmp_path / "foginstall.txt").write_text("install notes", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "read fogxinstall.txt and install FOG"
    state.working_memory.current_goal = state.run_brief.original_task
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    graph_state = GraphRunState(loop_state=state, thread_id="thread-missing-input-alias", run_mode="chat")

    missing = asyncio.run(file_read("fogxinstall.txt", cwd=str(tmp_path), state=state))
    assert missing["success"] is False
    assert missing["metadata"]["suggested_path"] == "foginstall.txt"
    assert missing["metadata"]["suggestion_confidence"] == "high"

    record = ToolExecutionRecord(
        operation_id="op-file-read-missing-input",
        tool_name="file_read",
        args={"path": "fogxinstall.txt"},
        tool_call_id="call-file-read-missing-input",
        result=ToolEnvelope(
            success=False,
            error=missing["error"],
            metadata=missing["metadata"],
        ),
    )

    assert _maybe_emit_missing_requested_output_file_nudge(graph_state, harness, record) is True
    assert state.scratchpad["_unresolved_missing_input_file"]["suggested_path"] == "foginstall.txt"


def test_successful_local_alias_read_clears_missing_required_input_blocker(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.scratchpad["_unresolved_missing_input_file"] = {
        "path": "fogxinstall.txt",
        "suggested_path": "foginstall.txt",
        "suggestion_confidence": "high",
    }
    state.scratchpad["_missing_input_file_nudged"] = "missing-input-file:fogxinstall.txt"
    runlog_events: list[str] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, *args, **kwargs: runlog_events.append(event),
    )
    record = ToolExecutionRecord(
        operation_id="op-file-read-alias",
        tool_name="file_read",
        args={"path": "foginstall.txt"},
        tool_call_id="call-file-read-alias",
        result=ToolEnvelope(
            success=True,
            output="install notes",
            metadata={"path": str(tmp_path / "foginstall.txt"), "complete_file": True},
        ),
    )

    assert _maybe_clear_missing_input_after_local_alias_read(harness, record) is True

    assert "_unresolved_missing_input_file" not in state.scratchpad
    assert "_missing_input_file_nudged" not in state.scratchpad
    assert state.scratchpad["_required_input_aliases"]["fogxinstall.txt"]["resolved_path"] == "foginstall.txt"
    assert "missing_required_input_file_blocker_cleared_by_local_alias_read" in runlog_events


def test_task_complete_blocks_unresolved_missing_required_input(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.scratchpad["_unresolved_missing_input_file"] = {"path": "fogxinstall.txt"}
    harness = SimpleNamespace(state=state)

    result = asyncio.run(task_complete("done", state, harness))

    assert result["success"] is False
    assert result["metadata"]["reason"] == "missing_required_input_file"
    assert "fogxinstall.txt" in result["error"]


def test_task_complete_blocks_phase_begin_with_zero_mutations(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "read ./temp/tetris-spec.md; if phase 3 is implemented, begin phase 4"
    state.working_memory.current_goal = state.run_brief.original_task
    state.challenge_progress.task_category = "coding"
    state.challenge_progress.code_change_count = 0
    harness = SimpleNamespace(state=state)

    result = asyncio.run(task_complete("Phase 3 is implemented; Phase 4 is active.", state, harness))

    assert result["success"] is False
    assert result["metadata"]["reason"] == "mutation_expected_but_no_code_changes"
    assert "zero code changes" in result["error"]


def test_task_complete_allows_reported_environment_blocker_for_run_report_task(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    task = "read, then run ./temp/vikunja-9b.py, then propose fixes/improvements to the script"
    state.run_brief.original_task = task
    state.working_memory.current_goal = task
    state.last_verifier_verdict = {
        "verdict": "fail",
        "failure_mode": "environment",
        "command": "python3 ./temp/vikunja-9b.py info",
        "key_stderr": "Network error: [Errno 111] Connection refused",
    }
    harness = SimpleNamespace(state=state)

    result = asyncio.run(
        task_complete(
            "The run is blocked by an environment issue: connection refused because no Vikunja server is running.",
            state,
            harness,
        )
    )

    assert result["success"] is True
    assert result["output"]["status"] == "complete"


def test_task_complete_blocks_environment_failure_for_mutation_task(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    task = "improve the script input validation"
    state.run_brief.original_task = task
    state.working_memory.current_goal = task
    state.last_verifier_verdict = {
        "verdict": "fail",
        "failure_mode": "environment",
        "command": "python3 ./temp/vikunja-9b.py info",
        "key_stderr": "Network error: [Errno 111] Connection refused",
    }
    harness = SimpleNamespace(state=state)

    result = asyncio.run(
        task_complete(
            "The run is blocked by an environment issue: connection refused because no Vikunja server is running.",
            state,
            harness,
        )
    )

    assert result["success"] is False
    assert "latest verifier verdict is still failing" in result["error"]


def test_task_complete_allows_reported_docker_daemon_failure_for_report_task(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    task = "ssh root@192.168.1.89 and report back the docker containers installed on that host"
    state.run_brief.original_task = task
    state.working_memory.current_goal = task
    state.last_verifier_verdict = {
        "verdict": "fail",
        "failure_mode": "logic",
        "command": "docker ps -a",
        "key_stderr": "Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?",
    }
    harness = SimpleNamespace(state=state)

    result = asyncio.run(
        task_complete(
            "Unable to retrieve Docker containers because the Docker daemon is not running on the remote host 192.168.1.89.",
            state,
            harness,
        )
    )

    assert result["success"] is True
    assert result["output"]["status"] == "complete"


def test_force_finalize_completion_signal_ignores_missing_required_input(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.scratchpad["_task_complete"] = True
    state.scratchpad["_unresolved_missing_input_file"] = {"path": "fogxinstall.txt"}
    harness = SimpleNamespace(state=state)

    assert _working_memory_signals_completion(harness) is False
