from __future__ import annotations

from types import SimpleNamespace

from smallctl.harness import Harness
from smallctl.state import LoopState
from smallctl.tools.memory import append_session_notepad_entry, log_note


def test_log_note_appends_and_dedupes() -> None:
    async def _run() -> None:
        state = LoopState(cwd="/tmp")
        first = await log_note(state=state, content="CWD is /tmp", tag="env")
        second = await log_note(state=state, content="CWD is /tmp", tag="env")

        assert first["success"] is True
        assert second["success"] is True
        assert second["metadata"]["duplicate"] is True
        payload = state.scratchpad["_session_notepad"]
        assert payload["entries"] == ["[env] CWD is /tmp"]

    import asyncio

    asyncio.run(_run())


def test_task_boundary_reset_preserves_session_notepad() -> None:
    state = LoopState(cwd="/tmp")
    append_session_notepad_entry(state, content="CWD is /tmp", tag="env")
    state.working_memory.known_facts = ["file_read: README exists"]
    state.recent_messages = []

    dummy_harness = SimpleNamespace(
        state=state,
        _initial_phase="explore",
        _configured_planning_mode=False,
        _runlog=lambda *args, **kwargs: None,
    )

    Harness._reset_task_boundary_state(
        dummy_harness,
        reason="run_task",
        new_task="next task",
        previous_task="old task",
    )

    payload = state.scratchpad.get("_session_notepad")
    assert isinstance(payload, dict)
    assert payload.get("entries") == ["[env] CWD is /tmp"]
    assert state.working_memory.known_facts == []
