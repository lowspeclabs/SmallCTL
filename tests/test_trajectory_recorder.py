from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from smallctl.harness.trajectory_recorder import TrajectoryRecorder


def _make_harness(**kwargs: Any) -> Any:
    scratchpad = dict(kwargs.pop("scratchpad", {}))
    run_brief = SimpleNamespace(
        original_task=kwargs.pop("task", "test task"),
        effective_task=kwargs.pop("task", "test task"),
    )
    state = SimpleNamespace(
        thread_id="t-123",
        run_brief=run_brief,
        scratchpad=scratchpad,
    )
    return SimpleNamespace(state=state, conversation_id="t-123", **kwargs)


def test_record_tool_plan_trajectory_writes_jsonl(tmp_path) -> None:
    recorder = TrajectoryRecorder(base_dir=tmp_path)
    harness = _make_harness(
        scratchpad={
            "_tool_plan": {
                "steps": [
                    {"id": "E1", "tool": "file_read", "args": {"path": "a.py"}},
                ]
            },
            "_tool_plan_observations_text": "read a.py -> ok",
            "_tool_plan_refine_verdict": "revise",
            "_recovery_metrics": {
                "tool_plan_invocations": 1,
                "tool_plan_total_tokens": 400,
                "tool_plan_refine_verdict": "revise",
            },
        }
    )
    result = {"status": "completed", "latency_metrics": {"tool_execution_duration_sec": 3.5}}
    out_path = recorder.record_tool_plan_trajectory(harness, result)
    assert out_path is not None
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["runtime"] == "tool_plan"
    assert payload["success"] is True
    assert payload["refine_verdict"] == "revise"
    assert payload["metrics"]["tool_plan_total_tokens"] == 400
    assert payload["tool_plan"][0]["tool"] == "file_read"


def test_record_skips_when_no_state() -> None:
    recorder = TrajectoryRecorder(base_dir=Path("/tmp"))
    harness = SimpleNamespace()
    assert recorder.record_tool_plan_trajectory(harness, {}) is None
