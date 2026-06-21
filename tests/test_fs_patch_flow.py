from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import pytest

from smallctl.state import LoopState
from smallctl.tools.fs_patch_flow import handle_file_patch
from smallctl.write_session_fsm import new_write_session


def _make_state(tmp_path: Path) -> LoopState:
    state = LoopState(cwd=str(tmp_path))
    state.active_tool_profiles = {"core"}
    return state


def test_handle_file_patch_dry_run_previews_without_writing(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "counter.py"
    original = "count = 0\n"
    target.write_text(original, encoding="utf-8")

    result = asyncio.run(
        handle_file_patch(
            path=str(target),
            target_text="count = 0",
            replacement_text="count = 1",
            cwd=str(tmp_path),
            state=state,
            dry_run=True,
        )
    )

    assert result["success"] is True
    assert "Dry run" in result["output"]
    assert target.read_text(encoding="utf-8") == original
    assert result["metadata"].get("dry_run") is True
    assert result["metadata"].get("occurrence_count") == 1


def test_handle_file_patch_repair_cycle_read_gate_blocks_without_read(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "queue.py"
    target.write_text("class Queue:\n    pass\n", encoding="utf-8")
    state.repair_cycle_id = "repair-cycle-1"
    state.tool_execution_records = {
        "record-1": {
            "tool_name": "file_patch",
            "args": {"path": str(target)},
            "result": {"success": False, "metadata": {"requested_path": str(target)}},
        }
    }

    result = asyncio.run(
        handle_file_patch(
            path=str(target),
            target_text="class Queue:\n    pass\n",
            replacement_text="class Queue:\n    items = []\n",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "repair_cycle_read_required"
    assert result["metadata"]["next_required_tool"]["tool_name"] == "file_read"


def test_handle_file_patch_repair_cycle_gate_emits_no_stdout(tmp_path: Path) -> None:
    """Regression test for the stray DEBUG print in fs_patch_flow.py."""
    state = _make_state(tmp_path)
    target = tmp_path / "queue.py"
    target.write_text("class Queue:\n    pass\n", encoding="utf-8")
    state.repair_cycle_id = "repair-cycle-stdout"
    state.tool_execution_records = {
        "record-1": {
            "tool_name": "file_patch",
            "args": {"path": str(target)},
            "result": {"success": False, "metadata": {"requested_path": str(target)}},
        }
    }

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        asyncio.run(
            handle_file_patch(
                path=str(target),
                target_text="class Queue:\n    pass\n",
                replacement_text="class Queue:\n    items = []\n",
                cwd=str(tmp_path),
                state=state,
            )
        )
    finally:
        sys.stdout = old_stdout

    assert captured.getvalue() == ""


def test_handle_file_patch_staged_only_patches_staging_copy(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "session.py"
    target.write_text("VERSION = '0.1'\n", encoding="utf-8")
    session = new_write_session(
        session_id="ws-patch",
        target_path=str(target),
        intent="patch_existing",
    )
    state.write_session = session

    result = asyncio.run(
        handle_file_patch(
            path=str(target),
            target_text="VERSION = '0.1'",
            replacement_text="VERSION = '0.2'",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
        )
    )

    assert result["success"] is True
    assert "Staged copy" in result["output"]
    assert result["metadata"].get("staged_only") is True
    staging_path = Path(result["metadata"]["staging_path"])
    assert staging_path.read_text(encoding="utf-8") == "VERSION = '0.2'\n"
    assert target.read_text(encoding="utf-8") == "VERSION = '0.1'\n"
    assert result["metadata"].get("write_session_final_chunk") is True
    assert "patch" in result["metadata"].get("write_sections_completed", [])


def test_handle_file_patch_target_not_found_recovery(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "missing.py"

    result = asyncio.run(
        handle_file_patch(
            path=str(target),
            target_text="def old():\n    pass\n",
            replacement_text="def new():\n    pass\n",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "patch_target_not_found"
    assert result["metadata"]["actual_occurrences"] == 0
    assert "not found" in result["error"].lower()


def test_handle_file_patch_occurrence_mismatch_is_reported(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "repeat.py"
    target.write_text("flag = 1\nflag = 1\n", encoding="utf-8")

    result = asyncio.run(
        handle_file_patch(
            path=str(target),
            target_text="flag = 1",
            replacement_text="flag = 2",
            cwd=str(tmp_path),
            state=state,
            expected_occurrences=1,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "patch_occurrence_mismatch"
    assert result["metadata"]["actual_occurrences"] == 2
