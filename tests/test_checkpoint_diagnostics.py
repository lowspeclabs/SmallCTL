"""Phase 5: checkpoint diagnostics without storage migration."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from smallctl.graph.checkpoint import FileCheckpointSaver


CHECKPOINT_FILE_NAME = ".smallctl-langgraph-checkpoints.json"


def _empty_checkpoint(path: Path) -> FileCheckpointSaver:
    return FileCheckpointSaver(path)


def _write_valid_checkpoint(
    saver: FileCheckpointSaver,
    *,
    thread_id: str = "thread-1",
    checkpoint_id: str = "chk-1",
    task_id: str = "task-1",
    task_path: str = "prepare_prompt",
    channel: str = "loop_state",
    value: Any = {"secret": "value"},
    timestamp: str | None = None,
) -> None:
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
        }
    }
    checkpoint = {
        "id": checkpoint_id,
        "channel_values": {channel: value},
        "channel_versions": {channel: 1},
        "ts": "2024-01-01T00:00:00+00:00",
    }
    metadata: dict[str, Any] = {
        "source": "loop",
        "step": 1,
        "parents": {},
        "run_id": "run-1",
    }
    if timestamp is not None:
        metadata["timestamp"] = timestamp
    saver.put(config, checkpoint, metadata, {channel: 1})
    saver.put_writes(config, [(channel, value)], task_id, task_path)


def test_empty_checkpoint_file_returns_empty_summary(tmp_path: Path) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    saver = _empty_checkpoint(path)
    assert saver.checkpoint_history_summary() == []
    assert saver.checkpoint_history_summary(thread_id="missing") == []


def test_malformed_records_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    saver = FileCheckpointSaver(path)
    _write_valid_checkpoint(saver, task_id="task-1", channel="loop_state")
    # Inject malformed entries directly into the in-memory writes table to
    # verify the summary method is defensive. These will not be flushed to disk.
    saver.writes[("thread-1", "", "chk-1")][("bad-task", 0)] = (
        None,  # task_id is not a string
        "channel-x",
        saver.serde.dumps_typed({"value": "x"}),
        "path",
    )
    saver.writes[("thread-1", "", "chk-1")][("bad-task-2", 1)] = (
        "bad-task-2",
        "channel-y",
        None,  # value is not a typed tuple
        "path",
    )
    summary = saver.checkpoint_history_summary()
    assert len(summary) == 1
    assert summary[0]["task_id"] == "task-1"
    assert summary[0]["channel"] == "loop_state"


def test_normal_write_history_returns_metadata(tmp_path: Path) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    saver = FileCheckpointSaver(path)
    _write_valid_checkpoint(
        saver,
        thread_id="thread-1",
        checkpoint_id="chk-1",
        task_id="task-1",
        task_path="prepare_prompt",
        channel="loop_state",
        timestamp="2024-01-01T12:00:00+00:00",
    )
    _write_valid_checkpoint(
        saver,
        thread_id="thread-1",
        checkpoint_id="chk-2",
        task_id="task-2",
        task_path="dispatch_tools",
        channel="pending_tool_calls",
        timestamp="2024-01-01T12:00:01+00:00",
    )
    summary = saver.checkpoint_history_summary(thread_id="thread-1")
    assert len(summary) == 2
    for entry in summary:
        assert "checkpoint_id" in entry
        assert "checkpoint_ns" in entry
        assert "task_id" in entry
        assert "channel" in entry
        assert "task_path" in entry
        assert "timestamp" in entry
    assert summary[0]["checkpoint_id"] == "chk-1"
    assert summary[0]["task_id"] == "task-1"
    assert summary[0]["task_path"] == "prepare_prompt"
    assert summary[0]["channel"] == "loop_state"
    assert summary[0]["timestamp"] == "2024-01-01T12:00:00+00:00"
    assert summary[1]["checkpoint_id"] == "chk-2"
    assert summary[1]["task_id"] == "task-2"
    assert summary[1]["task_path"] == "dispatch_tools"
    assert summary[1]["channel"] == "pending_tool_calls"
    assert summary[1]["timestamp"] == "2024-01-01T12:00:01+00:00"


def test_write_history_is_sorted_by_checkpoint_timestamp(tmp_path: Path) -> None:
    saver = FileCheckpointSaver(tmp_path / CHECKPOINT_FILE_NAME)
    _write_valid_checkpoint(
        saver, checkpoint_id="later", task_id="later", timestamp="2024-01-01T12:00:02+00:00"
    )
    _write_valid_checkpoint(
        saver, checkpoint_id="earlier", task_id="earlier", timestamp="2024-01-01T12:00:01+00:00"
    )

    assert [entry["checkpoint_id"] for entry in saver.checkpoint_history_summary()] == [
        "earlier",
        "later",
    ]


def test_independent_savers_preserve_each_others_writes(tmp_path: Path) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    first = FileCheckpointSaver(path)
    second = FileCheckpointSaver(path)

    _write_valid_checkpoint(first, checkpoint_id="first", task_id="first")
    _write_valid_checkpoint(second, checkpoint_id="second", task_id="second")

    restored = FileCheckpointSaver(path)
    assert {entry["checkpoint_id"] for entry in restored.checkpoint_history_summary(thread_id="thread-1")} == {
        "first",
        "second",
    }


def test_summary_filtered_by_checkpoint_id(tmp_path: Path) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    saver = FileCheckpointSaver(path)
    _write_valid_checkpoint(saver, checkpoint_id="chk-1", task_id="task-1", channel="loop_state")
    _write_valid_checkpoint(saver, checkpoint_id="chk-2", task_id="task-2", channel="pending_tool_calls")
    summary = saver.checkpoint_history_summary(thread_id="thread-1", checkpoint_id="chk-2")
    assert len(summary) == 1
    assert summary[0]["checkpoint_id"] == "chk-2"


def test_summary_without_thread_id_uses_latest_thread(tmp_path: Path) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    saver = FileCheckpointSaver(path)
    _write_valid_checkpoint(saver, thread_id="thread-1", checkpoint_id="chk-1", task_id="task-1")
    _write_valid_checkpoint(saver, thread_id="thread-2", checkpoint_id="chk-2", task_id="task-2")
    summary = saver.checkpoint_history_summary()
    assert len(summary) == 1
    assert summary[0]["task_id"] == "task-2"


def test_summary_does_not_return_decoded_values(tmp_path: Path) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    saver = FileCheckpointSaver(path)
    _write_valid_checkpoint(
        saver,
        value={"api_key": "super-secret", "command": "rm -rf /"},
    )
    summary = saver.checkpoint_history_summary()
    assert len(summary) == 1
    text = json.dumps(summary, default=str)
    assert "super-secret" not in text
    assert "rm -rf" not in text
    assert "value" not in summary[0]


def test_checkpoint_browser_history_prints_summary(tmp_path: Path) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    saver = FileCheckpointSaver(path)
    _write_valid_checkpoint(
        saver,
        thread_id="thread-1",
        checkpoint_id="chk-1",
        task_id="task-1",
        task_path="prepare_prompt",
        channel="loop_state",
        timestamp="2024-01-01T12:00:00+00:00",
    )
    result = subprocess.run(
        [sys.executable, "Agent-Tools/checkpoint_browser.py", str(tmp_path), "--history"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert len(data) == 1
    assert data[0]["checkpoint_id"] == "chk-1"
    assert data[0]["task_id"] == "task-1"
    assert data[0]["task_path"] == "prepare_prompt"
    assert data[0]["channel"] == "loop_state"
    assert data[0]["timestamp"] == "2024-01-01T12:00:00+00:00"
    assert "value" not in data[0]


def test_checkpoint_browser_history_handles_legacy_file(tmp_path: Path) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    path.write_text(json.dumps({"data": {"state": {"current_phase": "explore"}}}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "Agent-Tools/checkpoint_browser.py", str(tmp_path), "--history"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data == []


def test_checkpoint_browser_default_behavior_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / CHECKPOINT_FILE_NAME
    path.write_text(json.dumps({"data": {"state": {"current_phase": "explore"}}}), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "Agent-Tools/checkpoint_browser.py", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert "1 checkpoint(s)" in result.stdout


def test_checkpoint_browser_history_no_checkpoint_file(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "Agent-Tools/checkpoint_browser.py", str(tmp_path), "--history"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
