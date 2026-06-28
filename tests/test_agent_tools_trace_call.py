from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Agent-Tools"))

from trace_call import _render_text, _resolve_trace_id, _run_matches_trace


def test_resolve_trace_id_expands_step_call_suffix_with_colon(tmp_path: Path) -> None:
    run_dir = tmp_path / "abc-20260619"
    run_dir.mkdir()
    (run_dir / "harness.jsonl").write_text(
        json.dumps({"trace_id": "abc:task-0002:step-1:call-1", "event": "x"}) + "\n",
        encoding="utf-8",
    )

    assert _resolve_trace_id(run_dir, "step-7:call-7", False) == "abc:task-0002:step-7:call-7"


def test_run_matches_trace_detects_mismatched_prefix(tmp_path: Path) -> None:
    run_dir = tmp_path / "def-20260619"
    run_dir.mkdir()

    assert _run_matches_trace(run_dir, "abc:task-0001:step-1:call-1") is False
    assert _run_matches_trace(run_dir, "def:task-0001:step-1:call-1") is True
    assert _run_matches_trace(run_dir, "step-1:call-1") is True


def test_render_text_compact_collapses_ui_event_records(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-20260628"
    run_dir.mkdir()
    grouped = {
        "harness": [
            {"timestamp": "2026-06-28T12:00:00.000000+00:00", "event": "ui_event", "message": "", "data": {"event_type": "status_update"}},
            {"timestamp": "2026-06-28T12:00:00.100000+00:00", "event": "ui_event", "message": "", "data": {"event_type": "status_update"}},
            {"timestamp": "2026-06-28T12:00:00.200000+00:00", "event": "ui_event", "message": "", "data": {"event_type": "bubble_added"}},
            {"timestamp": "2026-06-28T12:00:01.000000+00:00", "event": "dispatch_start", "message": "start", "data": {}},
        ],
        "model_output": [],
        "chat": [],
        "tools": [],
    }

    compact_output = _render_text(run_dir, "abc:task-0001:step-1:call-1", grouped, compact=True)
    assert "3 ui_event records collapsed" in compact_output
    assert "status_update=2" in compact_output
    assert "bubble_added=1" in compact_output
    assert "first=2026-06-28T12:00:00.000000+00:00" in compact_output
    assert "last=2026-06-28T12:00:00.200000+00:00" in compact_output

    non_compact_output = _render_text(run_dir, "abc:task-0001:step-1:call-1", grouped, compact=False)
    assert non_compact_output.count("ui_event  ") == 3
    assert "collapsed" not in non_compact_output
