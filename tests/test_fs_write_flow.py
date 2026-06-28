from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from smallctl.state import LoopState
from smallctl.tools.fs_write_flow import handle_file_write_session
from smallctl.write_session_fsm import new_write_session


def _make_state(tmp_path: Path) -> LoopState:
    state = LoopState(cwd=str(tmp_path))
    state.active_tool_profiles = {"core"}
    return state


def test_handle_file_write_session_section_progression(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="ws-progress",
        target_path=str(target),
        intent="replace_file",
    )
    state.write_session = session

    first = handle_file_write_session(
        path=str(target),
        content="import os\n",
        cwd=str(tmp_path),
        encoding="utf-8",
        state=state,
        session_id=None,
        write_session_id=session.write_session_id,
        section_name="imports",
        section_id=None,
        section_role=None,
        next_section_name="helpers",
        replace_strategy="append",
        expected_followup_verifier=None,
        session=session,
    )

    assert first["success"] is True
    assert first["metadata"]["section_name"] == "imports"
    assert first["metadata"]["write_next_section"] == "helpers"
    assert "imports" in session.write_sections_completed

    second = handle_file_write_session(
        path=str(target),
        content="def helper():\n    return 1\n",
        cwd=str(tmp_path),
        encoding="utf-8",
        state=state,
        session_id=None,
        write_session_id=session.write_session_id,
        section_name="helpers",
        section_id=None,
        section_role=None,
        next_section_name=None,
        replace_strategy="append",
        expected_followup_verifier=None,
        session=session,
    )

    assert second["success"] is True
    assert second["metadata"]["section_name"] == "helpers"
    assert second["metadata"]["write_next_section"] == ""
    assert "imports" in session.write_sections_completed
    assert "helpers" in session.write_sections_completed
    staged = Path(session.write_staging_path).read_text(encoding="utf-8")
    assert "import os" in staged
    assert "def helper" in staged


def test_handle_file_write_session_overwrite_allows_full_file_section(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="ws-overwrite",
        target_path=str(target),
        intent="replace_file",
    )
    session.write_sections_completed = ["imports"]
    state.write_session = session

    result = handle_file_write_session(
        path=str(target),
        content="print('hello')\n",
        cwd=str(tmp_path),
        encoding="utf-8",
        state=state,
        session_id=None,
        write_session_id=session.write_session_id,
        section_name="full_file",
        section_id=None,
        section_role=None,
        next_section_name=None,
        replace_strategy="overwrite",
        expected_followup_verifier=None,
        session=session,
    )

    assert result["success"] is True
    assert result["metadata"]["section_name"] == "full_file"
    assert result["metadata"]["replace_strategy"] == "overwrite"
    assert session.write_sections_completed == ["full_file"]
    staged = Path(session.write_staging_path).read_text(encoding="utf-8")
    assert staged == "print('hello')\n"


def test_handle_file_write_session_overwrite_blocks_new_section_after_progress(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="ws-block",
        target_path=str(target),
        intent="replace_file",
    )
    session.write_sections_completed = ["imports"]
    state.write_session = session

    result = handle_file_write_session(
        path=str(target),
        content="def helper():\n    pass\n",
        cwd=str(tmp_path),
        encoding="utf-8",
        state=state,
        session_id=None,
        write_session_id=session.write_session_id,
        section_name="helpers",
        section_id=None,
        section_role=None,
        next_section_name=None,
        replace_strategy="overwrite",
        expected_followup_verifier=None,
        session=session,
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "chunked_write_overwrite_new_section_after_progress"


def test_handle_file_write_session_first_overwrite_allowed(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="ws-first-overwrite",
        target_path=str(target),
        intent="replace_file",
    )
    state.write_session = session

    result = handle_file_write_session(
        path=str(target),
        content="x = 1\n",
        cwd=str(tmp_path),
        encoding="utf-8",
        state=state,
        session_id=None,
        write_session_id=session.write_session_id,
        section_name="full_file",
        section_id=None,
        section_role=None,
        next_section_name=None,
        replace_strategy="overwrite",
        expected_followup_verifier=None,
        session=session,
    )

    assert result["success"] is True
    assert result["metadata"]["replace_strategy"] == "overwrite"


def test_handle_file_write_session_rejects_terminal_session(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="ws-terminal",
        target_path=str(target),
        intent="replace_file",
    )
    session.status = "complete"
    state.write_session = session

    result = handle_file_write_session(
        path=str(target),
        content="x = 1\n",
        cwd=str(tmp_path),
        encoding="utf-8",
        state=state,
        session_id=None,
        write_session_id=session.write_session_id,
        section_name="body",
        section_id=None,
        section_role=None,
        next_section_name=None,
        replace_strategy="append",
        expected_followup_verifier=None,
        session=session,
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "write_session_already_terminal"


def test_handle_file_write_session_html_without_scripts_is_final_chunk(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "report.html"
    session = new_write_session(
        session_id="ws-html-report",
        target_path=str(target),
        intent="replace_file",
    )
    state.write_session = session

    html = (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "    <meta charset=\"UTF-8\">\n"
        "    <title>Report</title>\n"
        "    <style>body { font-family: sans-serif; }</style>\n"
        "</head>\n"
        "<body>\n"
        "    <h1>Report</h1>\n"
        "    <p>This is a complete HTML report with no JavaScript.</p>\n"
        "</body>\n"
        "</html>\n"
    )

    result = handle_file_write_session(
        path=str(target),
        content=html,
        cwd=str(tmp_path),
        encoding="utf-8",
        state=state,
        session_id=None,
        write_session_id=session.write_session_id,
        section_name="header",
        section_id=None,
        section_role=None,
        next_section_name=None,
        replace_strategy="auto",
        expected_followup_verifier=None,
        session=session,
    )

    assert result["success"] is True
    assert result["metadata"]["write_session_final_chunk"] is True
    assert result["metadata"]["write_next_section"] == ""
