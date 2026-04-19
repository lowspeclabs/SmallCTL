from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smallctl.context.frame_compiler import PromptStateFrameCompiler
from smallctl.harness.memory import MemoryService
from smallctl.state import (
    ContextBrief,
    EpisodicSummary,
    LoopState,
    MemoryEntry,
    TurnBundle,
    WriteSession,
    memory_entry_is_stale,
)
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


def test_frame_compiler_prunes_file_invalidated_bundles_briefs_and_summaries() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.turn_bundles = [
        TurnBundle(
            bundle_id="TB-src",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(1, 2),
            phase="execute",
            summary_lines=["Updated src/app.py"],
            files_touched=["src/app.py"],
        ),
        TurnBundle(
            bundle_id="TB-docs",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(3, 4),
            phase="execute",
            summary_lines=["Updated docs/readme.md"],
            files_touched=["docs/readme.md"],
        ),
    ]
    state.context_briefs = [
        ContextBrief(
            brief_id="B-src",
            created_at="2026-04-19T00:00:00+00:00",
            tier="warm",
            step_range=(1, 2),
            task_goal="Patch src/app.py",
            current_phase="execute",
            key_discoveries=["src/app.py patched"],
            tools_tried=["file_patch"],
            blockers=[],
            files_touched=["src/app.py"],
            artifact_ids=["A-src"],
            next_action_hint="Run verifier",
            staleness_step=2,
        ),
        ContextBrief(
            brief_id="B-docs",
            created_at="2026-04-19T00:00:00+00:00",
            tier="warm",
            step_range=(3, 4),
            task_goal="Update docs",
            current_phase="execute",
            key_discoveries=["docs/readme.md updated"],
            tools_tried=["file_write"],
            blockers=[],
            files_touched=["docs/readme.md"],
            artifact_ids=["A-docs"],
            next_action_hint="Finalize docs",
            staleness_step=4,
        ),
    ]
    summaries = [
        EpisodicSummary(
            summary_id="S-src",
            created_at="2026-04-19T00:00:00+00:00",
            files_touched=["src/app.py"],
            notes=["Patched src/app.py"],
        ),
        EpisodicSummary(
            summary_id="S-docs",
            created_at="2026-04-19T00:00:00+00:00",
            files_touched=["docs/readme.md"],
            notes=["Patched docs/readme.md"],
        ),
    ]
    state.invalidate_context(
        reason="file_changed",
        paths=["src/app.py"],
        details={"state_change": "File changed: src/app.py"},
    )

    frame = PromptStateFrameCompiler().compile(state=state, retrieved_summaries=summaries)

    assert [bundle.bundle_id for bundle in frame.evidence_packet.turn_bundles] == ["TB-docs"]
    assert [brief.brief_id for brief in frame.evidence_packet.context_briefs] == ["B-docs"]
    assert [summary.summary_id for summary in frame.evidence_packet.summaries] == ["S-docs"]
    dropped = {(drop.lane, drop.reason): set(drop.dropped_ids) for drop in frame.drop_log}
    assert dropped[("turn_bundles", "context_invalidated")] == {"TB-src"}
    assert dropped[("context_briefs", "context_invalidated")] == {"B-src"}
    assert dropped[("episodic_summaries", "context_invalidated")] == {"S-src"}


def test_frame_compiler_prunes_verifier_invalidated_optimistic_items() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.last_failure_class = "test"
    state.turn_bundles = [
        TurnBundle(
            bundle_id="TB-optimistic",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(1, 2),
            phase="repair",
            summary_lines=["Verified fix and all tests pass"],
            files_touched=["src/app.py"],
        ),
        TurnBundle(
            bundle_id="TB-neutral",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(3, 4),
            phase="repair",
            summary_lines=["Collect failure traces from verifier"],
            files_touched=["src/app.py"],
        ),
    ]
    state.context_briefs = [
        ContextBrief(
            brief_id="B-optimistic",
            created_at="2026-04-19T00:00:00+00:00",
            tier="warm",
            step_range=(1, 2),
            task_goal="Repair tests",
            current_phase="repair",
            key_discoveries=["All tests pass after patch"],
            tools_tried=["shell_exec"],
            blockers=[],
            files_touched=["src/app.py"],
            artifact_ids=["A1"],
            next_action_hint="Complete task",
            staleness_step=2,
            new_facts=["verified success on pytest"],
        ),
        ContextBrief(
            brief_id="B-neutral",
            created_at="2026-04-19T00:00:00+00:00",
            tier="warm",
            step_range=(3, 4),
            task_goal="Repair tests",
            current_phase="repair",
            key_discoveries=["Verifier still failing on assertions"],
            tools_tried=["shell_exec"],
            blockers=["failing tests"],
            files_touched=["src/app.py"],
            artifact_ids=["A2"],
            next_action_hint="Collect failing test names",
            staleness_step=4,
        ),
    ]
    summaries = [
        EpisodicSummary(
            summary_id="S-optimistic",
            created_at="2026-04-19T00:00:00+00:00",
            notes=["Successfully fixed all tests"],
            failed_approaches=["test"],
        ),
        EpisodicSummary(
            summary_id="S-neutral",
            created_at="2026-04-19T00:00:00+00:00",
            notes=["Still investigating failure output"],
            failed_approaches=["timeout"],
        ),
    ]
    state.invalidate_context(
        reason="verifier_failed",
        details={"state_change": "Verifier failure invalidated optimistic context"},
    )

    frame = PromptStateFrameCompiler().compile(state=state, retrieved_summaries=summaries)

    assert [bundle.bundle_id for bundle in frame.evidence_packet.turn_bundles] == ["TB-neutral"]
    assert [brief.brief_id for brief in frame.evidence_packet.context_briefs] == ["B-neutral"]
    assert [summary.summary_id for summary in frame.evidence_packet.summaries] == ["S-neutral"]
