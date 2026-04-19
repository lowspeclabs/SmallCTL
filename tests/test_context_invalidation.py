from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smallctl.harness.memory import MemoryService
from smallctl.state import LoopState, MemoryEntry, WriteSession, memory_entry_is_stale
from smallctl.tools.fs_sessions import _record_file_change


class _RunLogHarness(SimpleNamespace):
    def __init__(self, *, state: LoopState) -> None:
        self.state = state
        self.provider_profile = "lmstudio"
        self.context_policy = SimpleNamespace(memory_staleness_step_limit=10, soft_prompt_token_limit=2048, max_prompt_tokens=4096)
        self._events: list[tuple[str, dict[str, object]]] = []
        self._current_user_task = lambda: state.run_brief.original_task
        self._looks_like_shell_request = lambda _task: False
        self._compact_oversized_tool_messages = lambda **kwargs: False

    def _runlog(self, event: str, _message: str, **data: object) -> None:
        self._events.append((event, data))


def test_file_change_invalidation_marks_path_fact_stale_and_queues_hint() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.step_count = 8
    state.working_memory.known_facts = ["src/app.py currently passes verifier"]
    state.working_memory.known_fact_meta = [
        MemoryEntry(
            content="src/app.py currently passes verifier",
            created_at_step=4,
            created_phase="verify",
            freshness="current",
            confidence=0.9,
        )
    ]

    event = state.invalidate_context(
        reason="file_changed",
        paths=["src/app.py"],
        details={"state_change": "File changed: src/app.py"},
    )

    assert event["reason"] == "file_changed"
    assert event["invalidated_fact_count"] == 1
    assert state.scratchpad["_invalidated_facts_queue"] == ["src/app.py currently passes verifier"]
    assert memory_entry_is_stale(
        state.working_memory.known_fact_meta[0],
        current_step=state.step_count,
        current_phase=state.current_phase,
    )


def test_fs_session_file_change_records_context_invalidation_event(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    changed = tmp_path / "src" / "app.py"
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("print('ok')\n", encoding="utf-8")

    _record_file_change(state, changed)

    invalidations = state.scratchpad.get("_context_invalidations", [])
    assert invalidations
    assert invalidations[-1]["reason"] == "file_changed"
    assert changed.as_posix().lower() in invalidations[-1]["paths"]


def test_memory_service_emits_phase_environment_and_write_target_invalidations() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.run_brief.original_task = "update src/new.py"
    state.working_memory.current_goal = "update src/new.py"
    state.working_memory.known_facts = ["execute phase still relies on explore assumptions"]
    state.working_memory.known_fact_meta = [
        MemoryEntry(
            content="execute phase still relies on explore assumptions",
            created_at_step=1,
            created_phase="explore",
            freshness="current",
            confidence=0.8,
        )
    ]
    state.scratchpad["_last_contract_phase_seen"] = "explore"
    state.scratchpad["_last_environment_fingerprint"] = "explore|analysis|/old|"
    state.scratchpad["_last_write_session_target"] = "src/old.py"
    state.write_session = WriteSession(
        write_session_id="ws-1",
        write_target_path="src/new.py",
        write_session_mode="chunked_author",
        write_session_intent="patch_existing",
        status="open",
    )

    harness = _RunLogHarness(state=state)
    MemoryService(harness).update_working_memory(recent_messages_limit=8)

    events = [event for event, _ in harness._events if event == "context_invalidated"]
    assert events
    context_events = [data for event, data in harness._events if event == "context_invalidated"]
    reasons = [str(data.get("reason") or "") for data in context_events]
    assert "phase_advanced" in reasons
    assert "environment_changed" in reasons
    assert "write_session_target_changed" in reasons
    phase_event = next(data for data in context_events if str(data.get("reason") or "") == "phase_advanced")
    assert phase_event["invalidated_fact_count"] >= 1
    assert phase_event["invalidated_facts"]
    assert "invalidated_memory_ids" in phase_event
