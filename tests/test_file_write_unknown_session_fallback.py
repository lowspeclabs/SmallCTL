from __future__ import annotations

import asyncio

from smallctl.state import LoopState
from smallctl.tools.fs import file_write
from smallctl.tools.fs_write_flow import handle_file_write_session


def test_file_write_with_unknown_write_session_id_falls_back_to_direct_write(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "rogue-grid-defense.html"
    content = "<!DOCTYPE html>\n<html><head><title>Test</title></head><body></body></html>\n"

    result = asyncio.run(
        file_write(
            str(target),
            content,
            cwd=str(tmp_path),
            state=state,
            write_session_id="rogue-grid-s1",
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == content
    events = state.scratchpad.get("_write_session_events", [])
    assert any(e["event"] == "unknown_write_session_id_fallback_to_direct_write" for e in events)


def test_file_write_with_matching_write_session_id_uses_session(tmp_path) -> None:
    from smallctl.write_session_fsm import new_write_session

    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "rogue-grid-defense.html"
    session = new_write_session(
        session_id="rogue-grid-s1",
        target_path=str(target),
        intent="replace_file",
    )
    state.write_session = session

    content = "<!DOCTYPE html>\n<html><head><title>Test</title></head><body></body></html>\n"
    result = asyncio.run(
        file_write(
            str(target),
            content,
            cwd=str(tmp_path),
            state=state,
            write_session_id="rogue-grid-s1",
            section_name="full_file",
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True


def test_missing_active_write_session_recovery_hint_does_not_reuse_phantom_session(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "rogue-grid-defense.html"
    content = "<!DOCTYPE html>\n<html><head><title>Test</title></head><body></body></html>\n"

    result = handle_file_write_session(
        path=str(target),
        content=content,
        cwd=str(tmp_path),
        encoding="utf-8",
        state=state,
        session_id=None,
        write_session_id="rogue-grid-s1",
        section_name=None,
        section_id=None,
        section_role=None,
        next_section_name=None,
        replace_strategy="overwrite",
        expected_followup_verifier=None,
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "missing_active_write_session"
    next_required = result["metadata"]["next_required_tool"]
    assert next_required["tool_name"] == "file_write"
    assert "write_session_id" not in next_required["required_fields"]
    assert "write_session_id" not in next_required["required_arguments"]
    assert "path" in next_required["required_arguments"]
    notes = " ".join(next_required.get("notes", []))
    assert "Omit `write_session_id`" in notes
    assert "harness can recover" not in notes.lower()


def test_unknown_write_session_id_with_active_same_target_session_is_rejected(tmp_path) -> None:
    """
    If a write session is active for the same target path but the model supplies a
    fabricated write_session_id, the unknown-ID fallback must not silently overwrite
    the session-owned target. The bare-write-to-session-owned-path guard must reject
    the write and tell the model to use the active session ID.
    """
    from smallctl.write_session_fsm import new_write_session

    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "rogue-grid-defense.html"
    active_session = new_write_session(
        session_id="active-session",
        target_path=str(target),
        intent="replace_file",
    )
    state.write_session = active_session

    content = "<!DOCTYPE html>\n<html><head><title>Test</title></head><body></body></html>\n"
    result = asyncio.run(
        file_write(
            str(target),
            content,
            cwd=str(tmp_path),
            state=state,
            write_session_id="rogue-grid-s1",
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "bare_write_to_session_owned_path"
    assert result["metadata"]["write_session_id"] == "active-session"
    if target.exists():
        assert target.read_text(encoding="utf-8") != content
    events = state.scratchpad.get("_write_session_events", [])
    assert any(e["event"] == "unknown_write_session_id_fallback_to_direct_write" for e in events)
    assert any(e["event"] == "bare_write_intercepted" for e in events)
