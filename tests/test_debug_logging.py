from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from smallctl.logging_utils import (
    RunLogger,
    SUBSYSTEM_CHANNELS,
    create_run_logger,
    synthetic_trace_id,
)


def test_run_logger_includes_schema_version_and_level(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run")
    logger.log("harness", "test_event", "test message", level="debug")

    row = json.loads((logger.run_dir / "harness.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["event_schema_version"] == 1
    assert row["level"] == "debug"
    assert row["event"] == "test_event"


def test_run_logger_writes_run_header(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run")
    header_path = logger.run_dir / "run_header.json"
    assert header_path.exists()
    header = json.loads(header_path.read_text(encoding="utf-8"))
    assert header["event_schema_version"] == 1
    assert "harness" in header["channels"]


def test_run_logger_subsystem_filter_drops_debug_events(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run", debug_subsystems=["tools"])
    # harness debug event should be dropped because only "tools" is enabled
    logger.log("harness", "harness_debug", "hidden", level="debug", subsystem="graph")
    logger.log("tools", "tools_debug", "visible", level="debug", subsystem="tools")
    logger.log("harness", "harness_info", "visible", level="info", subsystem="graph")

    harness_lines = (logger.run_dir / "harness.jsonl").read_text(encoding="utf-8").splitlines()
    harness_events = [json.loads(line)["event"] for line in harness_lines if line.strip()]
    assert "harness_debug" not in harness_events
    assert "harness_info" in harness_events

    tools_lines = (logger.run_dir / "tools.jsonl").read_text(encoding="utf-8").splitlines()
    tools_events = [json.loads(line)["event"] for line in tools_lines if line.strip()]
    assert "tools_debug" in tools_events


def test_run_logger_token_sampling_drops_tokens(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run", debug_tokens=False)
    logger.set_call_count(1)
    for i in range(150):
        logger.log("model_output", "model_token", "token", token=f"tok{i}")

    lines = (logger.run_dir / "model_output.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    token_events = [e for e in events if e["event"] == "model_token"]
    # first 100 (indices 0-99) + every 20th starting at 100 (100, 120, 140) = 103
    assert len(token_events) == 103


def test_run_logger_token_sampling_disabled_includes_all_tokens(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run", debug_tokens=True)
    logger.set_call_count(1)
    for i in range(10):
        logger.log("model_output", "model_token", "token", token=f"tok{i}")

    lines = (logger.run_dir / "model_output.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    token_events = [e for e in events if e["event"] == "model_token"]
    assert len(token_events) == 10


def test_run_logger_token_sampling_does_not_write_fragmented_text_stream(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run", debug_tokens=False)
    logger.set_call_count(1)
    for i in range(150):
        logger.log("model_output", "model_token", "token", token=f"tok{i} ")

    text = (logger.run_dir / "model_output.log").read_text(encoding="utf-8")
    # Sampled tokens must not be concatenated into a misleading continuous
    # stream; the text log should contain no token fragments at all.
    assert "tok0" not in text
    assert "tok120" not in text


def test_run_logger_token_sampling_writes_complete_model_output_events(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run", debug_tokens=False)
    logger.log("model_output", "model_output", "assistant output complete", assistant_text="hello world")

    text = (logger.run_dir / "model_output.log").read_text(encoding="utf-8")
    assert "assistant output complete" in text
    assert "hello world" in text


def test_run_logger_size_cap_rotates(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run", log_max_mb=0)
    logger.log("harness", "big_event", "x" * 1000)
    # Force the cap check by logging again
    logger.log("harness", "trigger", "y")
    # Rotation may have occurred; just ensure the logger still works
    assert (logger.run_dir / "harness.jsonl").exists()


def test_synthetic_trace_id_format() -> None:
    state = SimpleNamespace(
        thread_id="sess-1",
        step_count=5,
        scratchpad={"_active_task_id": "task-0003"},
    )
    trace_id = synthetic_trace_id(state, suffix="ctx")
    assert trace_id == "sess-1:task-0003:step-5:ctx"


def test_synthetic_trace_id_fallback() -> None:
    trace_id = synthetic_trace_id(None, suffix="ui")
    assert trace_id == "run:task:step-0:ui"


def test_run_logger_sets_trace_task_step_call(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run")
    logger.set_trace_id("sess:task:step-1:call-2")
    logger.set_task_id("task")
    logger.set_step_count(1)
    logger.set_call_count(2)
    logger.log("harness", "turn_start", "turn started")

    row = json.loads((logger.run_dir / "harness.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert row["trace_id"] == "sess:task:step-1:call-2"
    assert row["task_id"] == "task"
    assert row["step_count"] == 1
    assert row["call_count"] == 2


def test_debug_signal_escalate_writes_marker(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run")
    signal_path = tmp_path / "debug-signal"
    signal_path.write_text("escalate:3", encoding="utf-8")
    result = logger.handle_debug_signal(signal_path)
    assert result == {"command": "escalate", "turns": 3}
    rows = [json.loads(line) for line in (logger.run_dir / "harness.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(r["event"] == "debug_escalation" for r in rows)


def test_debug_signal_snapshot_writes_file(tmp_path: Path) -> None:
    logger = RunLogger(tmp_path / "run")
    logger.log("harness", "first", "event")
    signal_path = tmp_path / "debug-signal"
    signal_path.write_text("snapshot", encoding="utf-8")
    result = logger.handle_debug_signal(signal_path)
    assert result is not None
    assert result["command"] == "snapshot"
    snapshot_path = Path(result["snapshot_path"])
    assert snapshot_path.exists()


@pytest.mark.parametrize("subsystem,channels", list(SUBSYSTEM_CHANNELS.items()))
def test_subsystem_channel_ownership(subsystem: str, channels: tuple[str, ...]) -> None:
    assert all(isinstance(ch, str) for ch in channels)
    assert len(set(channels)) == len(channels)


def test_create_run_logger_passes_options(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    logger = create_run_logger(
        "logs",
        debug_subsystems=["graph"],
        log_max_mb=50,
        debug_tokens=True,
    )
    assert logger.debug_subsystems == {"graph"}
    assert logger.log_max_mb == 50
    assert logger.debug_tokens is True
    assert (logger.run_dir / "run_header.json").exists()
