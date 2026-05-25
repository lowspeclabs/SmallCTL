from __future__ import annotations

import json
from pathlib import Path

from aho.log_review import LogReview, review_logs


def test_review_logs_reads_task_summary(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    (log_dir / "task_summary.json").write_text(
        json.dumps({
            "stall_classification": "timeout_after_progress",
            "challenge_progress": {"verified_after_last_change": True},
            "error": {"type": "backend_stream_failure"},
        }),
        encoding="utf-8",
    )
    (log_dir / "harness.jsonl").write_text(
        json.dumps({"event": "task_complete", "timestamp": "2026-01-01T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    review = review_logs(log_dir)
    assert isinstance(review, LogReview)
    assert review.stall_classification == "timeout_after_progress"
    assert review.verified is True
    assert review.backend is True
    assert review.task_status == "completed"


def test_review_logs_counts_ask_human_from_both_sources(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    (log_dir / "tools.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "dispatch_start", "timestamp": "t1", "data": {"tool_name": "ask_human", "arguments": {"question": "q1"}, "operation_id": "op1"}}),
            json.dumps({"event": "dispatch_complete", "timestamp": "t2", "data": {"tool_name": "ask_human", "arguments": {"question": "q1"}, "operation_id": "op1", "success": True}}),
            json.dumps({"event": "dispatch_complete", "timestamp": "t3", "data": {"tool_name": "ask_human", "arguments": {"question": "q2"}, "operation_id": "op2", "success": True}}),
        ]) + "\n",
        encoding="utf-8",
    )
    review = review_logs(log_dir)
    assert len(review.ask_human_calls) == 2
    questions = {ah["question"] for ah in review.ask_human_calls}
    assert questions == {"q1", "q2"}


def test_review_logs_counts_tool_failures(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    (log_dir / "tools.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "dispatch_complete", "timestamp": "t1", "data": {"tool_name": "ssh_exec", "success": False, "error": "timeout"}}),
            json.dumps({"event": "dispatch_complete", "timestamp": "t2", "data": {"tool_name": "file_write", "success": True, "result": {"success": False, "error": "disk full"}}}),
        ]) + "\n",
        encoding="utf-8",
    )
    review = review_logs(log_dir)
    assert len(review.tool_failures) == 2
    assert review.tool_failures[0]["tool_name"] == "ssh_exec"
    assert review.tool_failures[1]["tool_name"] == "file_write"


def test_review_logs_detects_guard_events(tmp_path: Path) -> None:
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    (log_dir / "harness.jsonl").write_text(
        "\n".join([
            json.dumps({"event": "tool_loop_guard", "timestamp": "t1", "data": {"reason": "repeated loop"}}),
            json.dumps({"event": "fama_tool_call_blocked", "timestamp": "t2", "data": {"active_mitigation": "fama"}}),
            json.dumps({"event": "context_limit", "timestamp": "t3", "data": {"tokens": 100}}),
            json.dumps({"event": "stream_chunk_error_exhausted", "timestamp": "t4", "data": {"recovery_kind": "stall"}}),
        ]) + "\n",
        encoding="utf-8",
    )
    review = review_logs(log_dir)
    assert len(review.guard_events) == 4
    events = {g["event"] for g in review.guard_events}
    assert events == {"tool_loop_guard", "fama_tool_call_blocked", "context_limit", "stream_chunk_error_exhausted"}
