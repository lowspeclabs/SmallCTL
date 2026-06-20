from __future__ import annotations

import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Agent-Tools"))

from trace_call import _resolve_trace_id, _run_matches_trace


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
