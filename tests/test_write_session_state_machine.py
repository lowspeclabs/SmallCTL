from __future__ import annotations

from smallctl.state import LoopState
from smallctl.tools.control import loop_status
from smallctl.write_session_fsm import (
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
