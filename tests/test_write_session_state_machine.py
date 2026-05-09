from __future__ import annotations

from smallctl.state import LoopState
from smallctl.tools.control import loop_status
from smallctl.write_session_fsm import (
    archive_interrupted_write_session,
    archive_terminal_write_session,
    new_write_session,
    recent_write_session_events,
    record_write_session_event,
    transition_write_session,
)


def test_write_session_allows_legal_transition_open_to_local_repair() -> None:
    session = new_write_session(
        session_id="ws_1",
        target_path="a.py",
        intent="replace_file",
    )
    changed = transition_write_session(
        session,
        next_status="local_repair",
        next_mode="local_repair",
        pending_finalize=True,
    )
    assert changed is True
    assert session.status == "local_repair"
    assert session.write_session_mode == "local_repair"
    assert session.write_pending_finalize is True


def test_write_session_blocks_illegal_transition_complete_to_open() -> None:
    session = new_write_session(
        session_id="ws_2",
        target_path="b.py",
        intent="replace_file",
    )
    transition_write_session(session, next_status="complete")
    changed = transition_write_session(session, next_status="open")
    assert changed is False
    assert session.status == "complete"


def test_write_session_event_log_is_capped_and_exposed() -> None:
    state = LoopState(cwd="/tmp")
    session = new_write_session(
        session_id="ws_3",
        target_path="c.py",
        intent="replace_file",
    )
    for idx in range(45):
        record_write_session_event(
            state,
            event="tick",
            session=session,
            details={"idx": idx},
        )
    events = recent_write_session_events(state, limit=50)
    assert len(events) == 40
    assert events[-1]["details"]["idx"] == 44


def test_loop_status_includes_write_session_events() -> None:
    state = LoopState(cwd="/tmp")
    state.write_session = new_write_session(
        session_id="ws_4",
        target_path="d.py",
        intent="replace_file",
    )
    record_write_session_event(state, event="session_opened", session=state.write_session)
    payload = __import__("asyncio").run(loop_status(state))
    assert payload["success"] is True
    assert payload["output"]["write_session_events"]


def test_archive_terminal_write_session_clears_active_handle() -> None:
    state = LoopState(cwd="/tmp")
    state.write_session = new_write_session(
        session_id="ws_done",
        target_path="done.py",
        intent="replace_file",
    )
    transition_write_session(state.write_session, next_status="complete")

    archived = archive_terminal_write_session(state, reason="continue_like_task")

    assert archived is not None
    assert archived["write_session_id"] == "ws_done"
    assert state.write_session is None
    assert state.scratchpad["_archived_write_sessions"][-1]["reason"] == "continue_like_task"
    assert recent_write_session_events(state, limit=1)[0]["event"] == "terminal_write_session_cleared_on_continue"


def test_archive_interrupted_write_session_preserves_stage_metadata() -> None:
    state = LoopState(cwd="/tmp")
    state.write_session = new_write_session(
        session_id="ws_open",
        target_path="task_queue.py",
        intent="replace_file",
        next_section="helpers",
    )
    state.write_session.write_staging_path = "/tmp/.smallctl/write_sessions/ws_open__task_queue__stage.py"
    state.write_session.write_sections_completed = ["imports"]
    state.write_session.write_section_ranges = {"imports": {"start": 0, "end": 12}}

    archived = archive_interrupted_write_session(state, reason="task_switch_abandoned")

    assert archived is not None
    assert archived["write_session_id"] == "ws_open"
    assert archived["write_staging_path"].endswith("ws_open__task_queue__stage.py")
    assert archived["write_sections_completed"] == ["imports"]
    assert archived["write_section_ranges"] == {"imports": {"start": 0, "end": 12}}
    assert archived["write_next_section"] == "helpers"
    assert state.write_session is not None
    assert recent_write_session_events(state, limit=1)[0]["event"] == "interrupted_write_session_archived"
