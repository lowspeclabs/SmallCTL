from __future__ import annotations

import json

from types import SimpleNamespace

from smallctl.graph.model_stream_loop import _next_model_call_trace_id
from smallctl.logging_utils import RunLogger


def test_run_logger_trace_id_is_written_to_all_channels(tmp_path) -> None:
    logger = RunLogger(tmp_path / "run")
    logger.set_trace_id("thread-1:7")

    logger.log("harness", "turn_start", "turn started")
    logger.log("tools", "dispatch_start", "tool started")
    logger.log("chat", "chunk", "chat chunk")
    logger.log("model_output", "model_token", "assistant token", token="ok")

    for channel in ("harness", "tools", "chat", "model_output"):
        row = json.loads((logger.run_dir / f"{channel}.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert row["trace_id"] == "thread-1:7"


def test_run_logger_preserves_passwords_but_redacts_tokens_in_jsonl_output(tmp_path) -> None:
    logger = RunLogger(tmp_path / "run")

    logger.log(
        "harness",
        "transport_debug",
        "logging sensitive boundary",
        password="hunter2",
        payload={"token": "abc123", "nested": {"ssh_password": "secret"}},
    )

    row = json.loads((logger.run_dir / "harness.jsonl").read_text(encoding="utf-8").splitlines()[0])
    data = row["data"]
    assert data["password"] == "hunter2"
    assert data["payload"]["token"] != "abc123"
    assert data["payload"]["nested"]["ssh_password"] == "secret"


def test_model_call_trace_id_changes_when_step_count_resets() -> None:
    state = SimpleNamespace(
        thread_id="a89aa278",
        step_count=0,
        scratchpad={"_active_task_id": "task-0001"},
    )
    harness = SimpleNamespace(state=state, conversation_id="")

    first = _next_model_call_trace_id(harness)
    state.step_count = 3
    second = _next_model_call_trace_id(harness)
    state.step_count = 0
    state.scratchpad["_active_task_id"] = "task-0002"
    third = _next_model_call_trace_id(harness)

    assert first == "a89aa278:task-0001:step-0:call-1"
    assert second == "a89aa278:task-0001:step-3:call-2"
    assert third == "a89aa278:task-0002:step-0:call-3"
    assert len({first, second, third}) == 3
