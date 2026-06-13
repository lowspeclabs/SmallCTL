from __future__ import annotations

import asyncio
from pathlib import Path

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


def test_unknown_write_session_id_with_active_same_target_session_continues_session(tmp_path) -> None:
    """
    If a write session is active for the same target path but the model supplies a
    fabricated write_session_id, path-based implicit resolution must continue the
    active session instead of falling back to a direct write or rejecting the call.
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

    assert result["success"] is True
    assert result["metadata"]["write_session_id"] == "active-session"
    assert result["metadata"]["section_name"] == "full_file"
    staging_path = result["metadata"]["staging_path"]
    assert Path(staging_path).read_text(encoding="utf-8") == content
    # Target should not be promoted until finalize.
    if target.exists():
        assert target.read_text(encoding="utf-8") != content
    events = state.scratchpad.get("_write_session_events", [])
    assert any(
        e["event"] == "implicit_session_continued"
        and e["details"].get("reason") == "path_match_overrides_unknown_id"
        for e in events
    )


def test_bare_file_write_continues_active_same_target_session(tmp_path) -> None:
    from smallctl.write_session_fsm import new_write_session

    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="active-session",
        target_path=str(target),
        intent="replace_file",
    )
    session.write_next_section = "helpers"
    state.write_session = session

    content = "def helper():\n    return 1\n"
    result = asyncio.run(
        file_write(
            str(target),
            content,
            cwd=str(tmp_path),
            state=state,
            section_name="helpers",
        )
    )

    assert result["success"] is True
    assert result["metadata"]["write_session_id"] == "active-session"
    assert result["metadata"]["section_name"] == "helpers"
    events = state.scratchpad.get("_write_session_events", [])
    assert any(e["event"] == "implicit_session_resolved" for e in events)


def test_bare_file_write_infers_next_section(tmp_path) -> None:
    from smallctl.write_session_fsm import new_write_session

    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="active-session",
        target_path=str(target),
        intent="replace_file",
    )
    session.write_next_section = "main_logic"
    state.write_session = session

    content = "def main():\n    pass\n"
    result = asyncio.run(
        file_write(
            str(target),
            content,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["section_name"] == "main_logic"


def test_bare_file_write_overwrite_with_complete_html_becomes_full_file(tmp_path) -> None:
    from smallctl.write_session_fsm import new_write_session

    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "index.html"
    session = new_write_session(
        session_id="active-session",
        target_path=str(target),
        intent="replace_file",
    )
    state.write_session = session

    content = (
        "<!DOCTYPE html>\n<html>\n<head><title>T</title></head>\n"
        "<body><script>console.log(1)</script></body>\n</html>\n"
    )
    result = asyncio.run(
        file_write(
            str(target),
            content,
            cwd=str(tmp_path),
            state=state,
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True
    assert result["metadata"]["section_name"] == "full_file"
    assert result["metadata"]["write_session_final_chunk"] is True


def test_ambiguous_bare_write_after_sections_asks_for_section_name(tmp_path) -> None:
    from smallctl.write_session_fsm import new_write_session

    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="active-session",
        target_path=str(target),
        intent="replace_file",
    )
    session.write_sections_completed = ["imports"]
    state.write_session = session

    content = "def helper():\n    pass\n"
    result = asyncio.run(
        file_write(
            str(target),
            content,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "implicit_session_section_ambiguous"
    next_required = result["metadata"]["next_required_tool"]
    assert next_required["tool_name"] == "file_write"
    assert "section_name" in next_required["required_fields"]
    assert "write_session_id" not in next_required["required_fields"]
    assert next_required["required_arguments"] == {"path": str(target)}


def test_implicit_session_creation_from_section_name(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "app.py"

    result = asyncio.run(
        file_write(
            str(target),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="main_logic",
        )
    )

    assert result["success"] is True
    assert state.write_session is not None
    assert state.write_session.write_target_path == str(target)
    assert result["metadata"]["section_name"] == "imports"
    assert result["metadata"]["write_session_id"] == state.write_session.write_session_id
    events = state.scratchpad.get("_write_session_events", [])
    assert any(e["event"] == "implicit_session_created" for e in events)


def test_implicit_session_creation_continues_on_second_bare_section(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "app.py"

    asyncio.run(
        file_write(
            str(target),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="main_logic",
        )
    )
    first_session_id = state.write_session.write_session_id

    result = asyncio.run(
        file_write(
            str(target),
            "def main():\n    pass\n",
            cwd=str(tmp_path),
            state=state,
            section_name="main_logic",
        )
    )

    assert result["success"] is True
    assert state.write_session.write_session_id == first_session_id
    assert "main_logic" in state.write_session.write_sections_completed


def test_small_complete_html_one_shot_does_not_create_session(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "index.html"
    content = "<!DOCTYPE html>\n<html><head><title>T</title></head><body></body></html>\n"

    result = asyncio.run(
        file_write(
            str(target),
            content,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert state.write_session is None
    assert target.read_text(encoding="utf-8") == content


def test_patch_over_rewrite_guard_still_blocks_accidental_full_rewrite(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "existing.py"
    target.write_text("original line 1\noriginal line 2\n", encoding="utf-8")

    result = asyncio.run(
        file_write(
            str(target),
            "completely different content\nthat shares no lines\n",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "patch_over_rewrite_guard"


def test_two_files_can_have_concurrent_implicit_sessions(tmp_path) -> None:
    """Two distinct paths can have separate active write sessions in one task."""
    state = LoopState(cwd=str(tmp_path))
    target_a = tmp_path / "app.py"
    target_b = tmp_path / "utils.py"

    result_a = asyncio.run(
        file_write(
            str(target_a),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="main_logic",
        )
    )
    assert result_a["success"] is True
    session_a_id = result_a["metadata"]["write_session_id"]

    result_b = asyncio.run(
        file_write(
            str(target_b),
            "import typing\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="helpers",
        )
    )
    assert result_b["success"] is True
    session_b_id = result_b["metadata"]["write_session_id"]

    assert session_a_id != session_b_id
    assert len(state.active_write_sessions_by_path) == 2
    assert state.write_session.write_session_id == session_b_id

    # Continue the first file without needing a session ID.
    result_a2 = asyncio.run(
        file_write(
            str(target_a),
            "def main():\n    pass\n",
            cwd=str(tmp_path),
            state=state,
            section_name="main_logic",
        )
    )
    assert result_a2["success"] is True
    assert result_a2["metadata"]["write_session_id"] == session_a_id
    assert "main_logic" in state.active_write_sessions_by_path[str(target_a)].write_sections_completed


def test_switching_back_resumes_correct_staged_content(tmp_path) -> None:
    """Switching between files preserves each staged file's own content and ranges."""
    state = LoopState(cwd=str(tmp_path))
    target_a = tmp_path / "app.py"
    target_b = tmp_path / "utils.py"

    asyncio.run(
        file_write(
            str(target_a),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="main_logic",
        )
    )
    asyncio.run(
        file_write(
            str(target_b),
            "import typing\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="helpers",
        )
    )

    # Write second section for each file.
    asyncio.run(
        file_write(
            str(target_a),
            "def main():\n    pass\n",
            cwd=str(tmp_path),
            state=state,
            section_name="main_logic",
        )
    )
    asyncio.run(
        file_write(
            str(target_b),
            "def helper():\n    return 1\n",
            cwd=str(tmp_path),
            state=state,
            section_name="helpers",
        )
    )

    session_a = state.active_write_sessions_by_path[str(target_a)]
    session_b = state.active_write_sessions_by_path[str(target_b)]
    assert Path(session_a.write_staging_path).read_text(encoding="utf-8") == "import os\ndef main():\n    pass\n"
    assert Path(session_b.write_staging_path).read_text(encoding="utf-8") == "import typing\ndef helper():\n    return 1\n"


def test_task_switch_archives_all_incomplete_sessions(tmp_path) -> None:
    """Task boundary reset archives every incomplete active session."""
    state = LoopState(cwd=str(tmp_path))
    target_a = tmp_path / "app.py"
    target_b = tmp_path / "utils.py"

    asyncio.run(
        file_write(
            str(target_a),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="main_logic",
        )
    )
    asyncio.run(
        file_write(
            str(target_b),
            "import typing\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="helpers",
        )
    )

    from smallctl.write_session_fsm import archive_interrupted_write_session

    archive_interrupted_write_session(state, reason="task_switch_test")

    archived = state.scratchpad.get("_archived_write_sessions", [])
    assert len(archived) == 2
    archived_paths = {item["write_session_id"] for item in archived}

    assert state.active_write_sessions_by_path == {}
    # Interrupted archive preserves the legacy alias for compatibility callers.
    assert state.write_session is not None
    assert state.write_session.write_session_id in archived_paths


def test_task_boundary_reset_clears_all_incomplete_sessions(tmp_path) -> None:
    """A full task boundary reset leaves no active sessions behind."""
    state = LoopState(cwd=str(tmp_path))
    target_a = tmp_path / "app.py"
    target_b = tmp_path / "utils.py"

    asyncio.run(
        file_write(
            str(target_a),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="main_logic",
        )
    )
    asyncio.run(
        file_write(
            str(target_b),
            "import typing\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="helpers",
        )
    )

    from smallctl.write_session_fsm import archive_interrupted_write_session

    archive_interrupted_write_session(state, reason="task_switch_test")
    # Clear the compatibility alias that archive_interrupted_write_session keeps.
    state.write_session = None

    assert state.active_write_sessions_by_path == {}
    assert state.write_session is None
    archived = state.scratchpad.get("_archived_write_sessions", [])
    assert len(archived) == 2
    archived_paths = {item["write_target_path"] for item in archived}
    assert str(target_a) in archived_paths
    assert str(target_b) in archived_paths


def test_legacy_single_session_checkpoint_migrates_to_path_map(tmp_path) -> None:
    """A checkpoint with only state.write_session is usable through the path map."""
    from smallctl.write_session_fsm import new_write_session

    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "legacy.py"
    session = new_write_session(
        session_id="legacy-s1",
        target_path=str(target),
        intent="replace_file",
    )
    session.write_next_section = "main"
    state.write_session = session

    result = asyncio.run(
        file_write(
            str(target),
            "print('hello')\n",
            cwd=str(tmp_path),
            state=state,
            section_name="main",
        )
    )

    assert result["success"] is True
    assert result["metadata"]["write_session_id"] == "legacy-s1"
    assert state.active_write_sessions_by_path[str(target)].write_session_id == "legacy-s1"
