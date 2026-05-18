from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from smallctl.context.frame_compiler import PromptStateFrameCompiler
from smallctl.harness.memory import MemoryService
from smallctl.state import (
    ArtifactRecord,
    ArtifactSnippet,
    ContextBrief,
    EvidenceRecord,
    EpisodicSummary,
    ExperienceMemory,
    FailureEvent,
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
    assert "invalidated_turn_bundle_ids" in phase_event
    assert "invalidated_brief_ids" in phase_event
    assert "invalidated_summary_ids" in phase_event
    assert "invalidated_artifact_ids" in phase_event
    assert "invalidated_observation_ids" in phase_event


def test_invalidate_context_persists_experience_staleness_metadata() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-src",
            phase="repair",
            intent="requested_file_patch",
            tool_name="file_patch",
            outcome="success",
            confidence=0.85,
            notes="Successfully patched src/app.py",
        )
    ]

    event = state.invalidate_context(
        reason="file_changed",
        paths=["src/app.py"],
        details={"state_change": "File changed: src/app.py"},
    )

    assert event["invalidated_memory_count"] == 1
    stale_index = state.scratchpad.get("_experience_staleness")
    assert isinstance(stale_index, dict)
    assert "mem-src" in stale_index
    marker = stale_index["mem-src"]
    assert marker["stale"] is True
    assert marker["reason"] == "file_changed"
    assert "file_changed" in marker["reasons"]
    assert "src/app.py" in marker["paths"]


def test_invalidate_context_persists_lane_staleness_metadata() -> None:
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
        )
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
        )
    ]
    state.episodic_summaries = [
        EpisodicSummary(
            summary_id="S-src",
            created_at="2026-04-19T00:00:00+00:00",
            files_touched=["src/app.py"],
            notes=["Patched src/app.py"],
        )
    ]
    state.artifacts = {
        "A-src": ArtifactRecord(
            artifact_id="A-src",
            kind="file_read",
            source="src/app.py",
            created_at="2026-04-19T00:00:00+00:00",
            size_bytes=100,
            summary="src/app.py snapshot",
            metadata={"path": "src/app.py"},
        )
    }
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-src",
            statement="Read src/app.py",
            phase="execute",
            tool_name="file_read",
            metadata={"path": "src/app.py"},
        )
    ]

    event = state.invalidate_context(
        reason="file_changed",
        paths=["src/app.py"],
        details={"state_change": "File changed: src/app.py"},
    )

    assert event["invalidated_turn_bundle_ids"] == ["TB-src"]
    assert event["invalidated_brief_ids"] == ["B-src"]
    assert event["invalidated_summary_ids"] == ["S-src"]
    assert event["invalidated_artifact_ids"] == ["A-src"]
    assert event["invalidated_observation_ids"] == ["E-src"]
    assert "TB-src" in state.scratchpad["_turn_bundle_staleness"]
    assert "B-src" in state.scratchpad["_context_brief_staleness"]
    assert "S-src" in state.scratchpad["_summary_staleness"]
    assert "A-src" in state.scratchpad["_artifact_staleness"]
    assert "E-src" in state.scratchpad["_observation_staleness"]


def test_prune_context_staleness_indexes_removes_orphaned_entries() -> None:
    state = LoopState(cwd="/tmp")
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-live",
            phase="execute",
            intent="requested_file_read",
            tool_name="file_read",
            outcome="success",
            notes="Read docs/readme.md",
        )
    ]
    state.turn_bundles = [
        TurnBundle(
            bundle_id="TB-live",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(1, 2),
            phase="execute",
            summary_lines=["Read docs/readme.md"],
            files_touched=["docs/readme.md"],
        )
    ]
    state.context_briefs = [
        ContextBrief(
            brief_id="B-live",
            created_at="2026-04-19T00:00:00+00:00",
            tier="warm",
            step_range=(1, 2),
            task_goal="Read docs/readme.md",
            current_phase="execute",
            key_discoveries=["docs/readme.md read"],
            tools_tried=["file_read"],
            blockers=[],
            files_touched=["docs/readme.md"],
            artifact_ids=["A-live"],
            next_action_hint="Summarize",
            staleness_step=2,
        )
    ]
    state.episodic_summaries = [
        EpisodicSummary(
            summary_id="S-live",
            created_at="2026-04-19T00:00:00+00:00",
            notes=["read docs"],
        )
    ]
    state.artifacts = {
        "A-live": ArtifactRecord(
            artifact_id="A-live",
            kind="file_read",
            source="docs/readme.md",
            created_at="2026-04-19T00:00:00+00:00",
            size_bytes=10,
            summary="docs",
        )
    }
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-live",
            statement="Read docs/readme.md",
            phase="execute",
            tool_name="file_read",
        )
    ]
    state.scratchpad["_experience_staleness"] = {
        "mem-live": {"stale": True},
        "mem-orphan": {"stale": True},
    }
    state.scratchpad["_turn_bundle_staleness"] = {
        "TB-live": {"stale": True},
        "TB-orphan": {"stale": True},
    }
    state.scratchpad["_context_brief_staleness"] = {
        "B-live": {"stale": True},
        "B-orphan": {"stale": True},
    }
    state.scratchpad["_summary_staleness"] = {
        "S-live": {"stale": True},
        "S-orphan": {"stale": True},
    }
    state.scratchpad["_artifact_staleness"] = {
        "A-live": {"stale": True},
        "A-orphan": {"stale": True},
    }
    state.scratchpad["_observation_staleness"] = {
        "E-live": {"stale": True},
        "E-orphan": {"stale": True},
    }

    state.prune_context_staleness_indexes()

    assert set(state.scratchpad["_experience_staleness"].keys()) == {"mem-live"}
    assert set(state.scratchpad["_turn_bundle_staleness"].keys()) == {"TB-live"}
    assert set(state.scratchpad["_context_brief_staleness"].keys()) == {"B-live"}
    assert set(state.scratchpad["_summary_staleness"].keys()) == {"S-live"}
    assert set(state.scratchpad["_artifact_staleness"].keys()) == {"A-live"}
    assert set(state.scratchpad["_observation_staleness"].keys()) == {"E-live"}


def test_update_working_memory_prunes_orphaned_staleness_indexes() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.run_brief.original_task = "inspect docs"
    state.working_memory.current_goal = "inspect docs"
    state.scratchpad["_experience_staleness"] = {"mem-orphan": {"stale": True}}
    state.scratchpad["_turn_bundle_staleness"] = {"TB-orphan": {"stale": True}}
    state.scratchpad["_context_brief_staleness"] = {"B-orphan": {"stale": True}}
    state.scratchpad["_summary_staleness"] = {"S-orphan": {"stale": True}}
    state.scratchpad["_artifact_staleness"] = {"A-orphan": {"stale": True}}
    state.scratchpad["_observation_staleness"] = {"E-orphan": {"stale": True}}

    harness = _RunLogHarness(state=state)
    MemoryService(harness).update_working_memory(recent_messages_limit=8)

    assert "_experience_staleness" not in state.scratchpad
    assert "_turn_bundle_staleness" not in state.scratchpad
    assert "_context_brief_staleness" not in state.scratchpad
    assert "_summary_staleness" not in state.scratchpad
    assert "_artifact_staleness" not in state.scratchpad
    assert "_observation_staleness" not in state.scratchpad


def test_reinforce_experience_success_clears_staleness_marker() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-src",
            phase="repair",
            intent="requested_file_patch",
            tool_name="file_patch",
            outcome="success",
            confidence=0.7,
            notes="Successfully patched src/app.py",
        )
    ]
    state.scratchpad["_experience_staleness"] = {
        "mem-src": {
            "stale": True,
            "reason": "file_changed",
            "reasons": ["file_changed"],
            "paths": ["src/app.py"],
            "updated_at": "2026-04-19T00:00:00+00:00",
            "phase": "repair",
        }
    }

    state.reinforce_experience("mem-src", success=True)

    assert "_experience_staleness" not in state.scratchpad


def test_upsert_success_experience_clears_staleness_marker() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.scratchpad["_experience_staleness"] = {
        "mem-src": {
            "stale": True,
            "reason": "file_changed",
            "reasons": ["file_changed"],
            "paths": ["src/app.py"],
            "updated_at": "2026-04-19T00:00:00+00:00",
            "phase": "repair",
        }
    }

    state.upsert_experience(
        ExperienceMemory(
            memory_id="mem-src",
            phase="repair",
            intent="requested_file_patch",
            tool_name="file_patch",
            outcome="success",
            confidence=0.8,
            notes="Successfully patched src/app.py after rerun",
        )
    )

    assert "_experience_staleness" not in state.scratchpad


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


def test_frame_compiler_keeps_newer_password_summary_after_old_verifier_failure() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.invalidate_context(
        reason="verifier_failed",
        details={
            "command": "curl -s https://fogproject.org/scripts/fog-install.sh | bash",
            "target": "192.168.1.89 :: curl -s https://fogproject.org/scripts/fog-install.sh | bash",
            "failure_mode": "logic",
            "state_change": "Verifier failure invalidated optimistic context",
        },
    )

    summaries = [
        EpisodicSummary(
            summary_id="task-0003-summary",
            created_at="2026-05-14T19:45:10+00:00",
            notes=[
                'Task task-0003 failed: ssh root@192.168.1.89 with password "Temp@Pass" and install fog pxe server.'
            ],
            failed_approaches=["Guard tripped: repeated tool call loop"],
        )
    ]

    frame = PromptStateFrameCompiler().compile(state=state, retrieved_summaries=summaries)

    assert [summary.summary_id for summary in frame.evidence_packet.summaries] == ["task-0003-summary"]
    assert not [
        drop
        for drop in frame.drop_log
        if drop.lane == "episodic_summaries" and "task-0003-summary" in drop.dropped_ids
    ]


def test_password_text_is_not_treated_as_optimistic_context() -> None:
    state = LoopState(cwd="/tmp")
    state.working_memory.known_facts = ['SSH password is "Temp@Pass" for the remote host']
    state.working_memory.known_fact_meta = [
        MemoryEntry(
            content='SSH password is "Temp@Pass" for the remote host',
            freshness="current",
            confidence=0.9,
        )
    ]

    event = state.invalidate_context(
        reason="verifier_failed",
        details={"state_change": "Verifier failure invalidated optimistic context"},
    )

    assert event["invalidated_fact_count"] == 0
    assert state.working_memory.known_fact_meta[0].freshness == "current"


def test_frame_compiler_treats_fama_failure_like_verifier_failure() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
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
        ),
        EpisodicSummary(
            summary_id="S-neutral",
            created_at="2026-04-19T00:00:00+00:00",
            notes=["Still investigating failure output"],
        ),
    ]
    state.scratchpad["_context_invalidations"] = [
        {
            "reason": "fama_failure_detected",
            "paths": [],
            "fama_signal": "early_stop",
            "step": state.step_count,
        }
    ]

    frame = PromptStateFrameCompiler().compile(state=state, retrieved_summaries=summaries)

    assert [bundle.bundle_id for bundle in frame.evidence_packet.turn_bundles] == ["TB-neutral"]
    assert [brief.brief_id for brief in frame.evidence_packet.context_briefs] == ["B-neutral"]
    assert [summary.summary_id for summary in frame.evidence_packet.summaries] == ["S-neutral"]


def test_frame_compiler_prunes_invalidated_experience_memories() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    experiences = [
        ExperienceMemory(
            memory_id="mem-src",
            phase="repair",
            intent="requested_file_patch",
            tool_name="file_patch",
            outcome="success",
            notes="Successfully patched src/app.py",
        ),
        ExperienceMemory(
            memory_id="mem-docs",
            phase="repair",
            intent="requested_file_write",
            tool_name="file_write",
            outcome="success",
            notes="Updated docs/readme.md",
        ),
    ]
    state.invalidate_context(
        reason="file_changed",
        paths=["src/app.py"],
        details={"state_change": "File changed: src/app.py"},
    )

    frame = PromptStateFrameCompiler().compile(state=state, retrieved_experiences=experiences)

    assert [memory.memory_id for memory in frame.experience_packet.memories] == ["mem-docs"]
    drop = next(
        item
        for item in frame.drop_log
        if item.lane == "experience_memories" and item.reason == "context_invalidated"
    )
    assert set(drop.dropped_ids) == {"mem-src"}


def test_frame_compiler_prunes_durably_stale_experiences_without_recent_events() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    experiences = [
        ExperienceMemory(
            memory_id="mem-stale",
            phase="repair",
            intent="requested_file_patch",
            tool_name="file_patch",
            outcome="success",
            notes="Successfully patched src/app.py",
        ),
        ExperienceMemory(
            memory_id="mem-fresh",
            phase="repair",
            intent="requested_file_patch",
            tool_name="file_patch",
            outcome="success",
            notes="Patched docs/readme.md",
        ),
    ]
    state.scratchpad["_experience_staleness"] = {
        "mem-stale": {
            "stale": True,
            "reason": "file_changed",
            "reasons": ["file_changed"],
            "paths": ["src/app.py"],
            "updated_at": "2026-04-19T00:00:00+00:00",
            "phase": "repair",
        }
    }
    state.scratchpad["_context_invalidations"] = []

    frame = PromptStateFrameCompiler().compile(state=state, retrieved_experiences=experiences)

    assert [memory.memory_id for memory in frame.experience_packet.memories] == ["mem-fresh"]
    drop = next(
        item
        for item in frame.drop_log
        if item.lane == "experience_memories" and item.reason == "context_invalidated"
    )
    assert set(drop.dropped_ids) == {"mem-stale"}


def test_frame_compiler_prunes_durably_stale_non_experience_lanes_without_recent_events() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.turn_bundles = [
        TurnBundle(
            bundle_id="TB-stale",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(1, 2),
            phase="execute",
            summary_lines=["Patched src/app.py"],
            files_touched=["src/app.py"],
        ),
        TurnBundle(
            bundle_id="TB-fresh",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(3, 4),
            phase="execute",
            summary_lines=["Patched docs/readme.md"],
            files_touched=["docs/readme.md"],
        ),
    ]
    state.context_briefs = [
        ContextBrief(
            brief_id="B-stale",
            created_at="2026-04-19T00:00:00+00:00",
            tier="warm",
            step_range=(1, 2),
            task_goal="Patch src/app.py",
            current_phase="execute",
            key_discoveries=["src/app.py patched"],
            tools_tried=["file_patch"],
            blockers=[],
            files_touched=["src/app.py"],
            artifact_ids=["A-stale"],
            next_action_hint="Run verifier",
            staleness_step=2,
        ),
        ContextBrief(
            brief_id="B-fresh",
            created_at="2026-04-19T00:00:00+00:00",
            tier="warm",
            step_range=(3, 4),
            task_goal="Patch docs/readme.md",
            current_phase="execute",
            key_discoveries=["docs/readme.md patched"],
            tools_tried=["file_write"],
            blockers=[],
            files_touched=["docs/readme.md"],
            artifact_ids=["A-fresh"],
            next_action_hint="Finalize docs",
            staleness_step=4,
        ),
    ]
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-stale",
            statement="Read src/app.py",
            phase="execute",
            tool_name="file_read",
            metadata={"path": "src/app.py"},
        ),
        EvidenceRecord(
            evidence_id="E-fresh",
            statement="Read docs/readme.md",
            phase="execute",
            tool_name="file_read",
            metadata={"path": "docs/readme.md"},
        ),
    ]
    state.artifacts = {
        "A-stale": ArtifactRecord(
            artifact_id="A-stale",
            kind="file_read",
            source="src/app.py",
            created_at="2026-04-19T00:00:00+00:00",
            size_bytes=100,
            summary="src/app.py snapshot",
            metadata={"path": "src/app.py"},
        ),
        "A-fresh": ArtifactRecord(
            artifact_id="A-fresh",
            kind="file_read",
            source="docs/readme.md",
            created_at="2026-04-19T00:00:00+00:00",
            size_bytes=80,
            summary="docs/readme.md snapshot",
            metadata={"path": "docs/readme.md"},
        ),
    }
    summaries = [
        EpisodicSummary(
            summary_id="S-stale",
            created_at="2026-04-19T00:00:00+00:00",
            files_touched=["src/app.py"],
            notes=["Patched src/app.py"],
        ),
        EpisodicSummary(
            summary_id="S-fresh",
            created_at="2026-04-19T00:00:00+00:00",
            files_touched=["docs/readme.md"],
            notes=["Patched docs/readme.md"],
        ),
    ]
    retrieved = [
        ArtifactSnippet(artifact_id="A-stale", text="src artifact"),
        ArtifactSnippet(artifact_id="A-fresh", text="docs artifact"),
    ]
    state.scratchpad["_turn_bundle_staleness"] = {"TB-stale": {"stale": True}}
    state.scratchpad["_context_brief_staleness"] = {"B-stale": {"stale": True}}
    state.scratchpad["_summary_staleness"] = {"S-stale": {"stale": True}}
    state.scratchpad["_artifact_staleness"] = {"A-stale": {"stale": True}}
    state.scratchpad["_observation_staleness"] = {"E-stale": {"stale": True}}
    state.scratchpad["_context_invalidations"] = []

    frame = PromptStateFrameCompiler().compile(
        state=state,
        retrieved_summaries=summaries,
        retrieved_artifacts=retrieved,
    )

    assert [bundle.bundle_id for bundle in frame.evidence_packet.turn_bundles] == ["TB-fresh"]
    assert [brief.brief_id for brief in frame.evidence_packet.context_briefs] == ["B-fresh"]
    assert [summary.summary_id for summary in frame.evidence_packet.summaries] == ["S-fresh"]
    assert [snippet.artifact_id for snippet in frame.artifact_packet.snippets] == ["A-fresh"]
    assert [packet.observation_id for packet in frame.evidence_packet.observations] == ["E-fresh"]


def test_frame_compiler_prunes_file_invalidated_artifact_snippets() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.artifacts = {
        "A-src": ArtifactRecord(
            artifact_id="A-src",
            kind="file_read",
            source="src/app.py",
            created_at="2026-04-19T00:00:00+00:00",
            size_bytes=100,
            summary="src/app.py snapshot",
            metadata={"path": "src/app.py"},
        ),
        "A-docs": ArtifactRecord(
            artifact_id="A-docs",
            kind="file_read",
            source="docs/readme.md",
            created_at="2026-04-19T00:00:00+00:00",
            size_bytes=80,
            summary="docs/readme.md snapshot",
            metadata={"path": "docs/readme.md"},
        ),
    }
    retrieved = [
        ArtifactSnippet(artifact_id="A-src", text="src artifact"),
        ArtifactSnippet(artifact_id="A-docs", text="docs artifact"),
    ]
    state.invalidate_context(
        reason="file_changed",
        paths=["src/app.py"],
        details={"state_change": "File changed: src/app.py"},
    )

    frame = PromptStateFrameCompiler().compile(state=state, retrieved_artifacts=retrieved)

    assert [snippet.artifact_id for snippet in frame.artifact_packet.snippets] == ["A-docs"]
    drop = next(
        item
        for item in frame.drop_log
        if item.lane == "artifact_snippets" and item.reason == "context_invalidated"
    )
    assert set(drop.dropped_ids) == {"A-src"}


def test_frame_compiler_prunes_stale_observations_after_invalidation() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-src",
            statement="Read src/app.py",
            phase="execute",
            tool_name="file_read",
            metadata={"path": "src/app.py"},
        ),
        EvidenceRecord(
            evidence_id="E-docs",
            statement="Read docs/readme.md",
            phase="execute",
            tool_name="file_read",
            metadata={"path": "docs/readme.md"},
        ),
    ]
    state.invalidate_context(
        reason="file_changed",
        paths=["src/app.py"],
        details={"state_change": "File changed: src/app.py"},
    )

    frame = PromptStateFrameCompiler().compile(state=state)

    assert [packet.observation_id for packet in frame.evidence_packet.observations] == ["E-docs"]
    drop = next(
        item
        for item in frame.drop_log
        if item.lane == "normalized_observations" and item.reason == "context_invalidated"
    )
    assert drop.dropped_count == 1
    assert set(drop.dropped_ids) == {"E-src"}


def test_phase_advanced_preserves_recent_evidence() -> None:
    """Evidence created in the current or immediately preceding step must survive
    phase-advanced invalidation so the model can reason about the successful tool
    call that triggered the transition."""
    state = LoopState(cwd="/tmp")
    state.step_count = 4
    state.current_phase = "execute"

    # Old evidence from repair phase (step 1) – should be invalidated
    state.reasoning_graph.evidence_records.append(
        EvidenceRecord(
            evidence_id="E-OLD",
            statement="old repair observation",
            phase="repair",
            tool_name="ssh_exec",
            created_at_step=1,
        )
    )

    # Recent evidence from repair phase (step 3) – the successful ssh_exec that
    # caused the repair -> execute transition. Must NOT be invalidated.
    state.reasoning_graph.evidence_records.append(
        EvidenceRecord(
            evidence_id="E-RECOVERY",
            statement="ssh_exec succeeded",
            phase="repair",
            tool_name="ssh_exec",
            created_at_step=3,
        )
    )

    state.invalidate_context(
        reason="phase_advanced",
        details={"from_phase": "repair", "to_phase": "execute"},
    )

    stale_ids = set(state.scratchpad.get("_observation_staleness", {}).keys())
    assert "E-OLD" in stale_ids
    assert "E-RECOVERY" not in stale_ids

    frame = PromptStateFrameCompiler().compile(state=state)
    observation_ids = {p.observation_id for p in frame.evidence_packet.observations}
    assert "E-RECOVERY" in observation_ids
    assert "E-OLD" not in observation_ids


def test_phase_advanced_preserves_recent_known_facts_and_read_artifacts() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 10
    state.current_phase = "execute"
    state.working_memory.known_fact_meta = [
        MemoryEntry(
            content="ssh_file_read: Caddyfile full file (7 lines)",
            created_at_step=9,
            created_phase="repair",
            freshness="current",
        )
    ]
    state.working_memory.known_facts = [entry.content for entry in state.working_memory.known_fact_meta]
    state.artifacts["A-caddyfile"] = ArtifactRecord(
        artifact_id="A-caddyfile",
        kind="ssh_file_read",
        source="/etc/caddy/Caddyfile",
        created_at="2026-05-03T21:34:18+00:00",
        size_bytes=134,
        summary="Caddyfile full file",
        tool_name="ssh_file_read",
        metadata={"phase": "repair", "created_at_step": 9, "path": "/etc/caddy/Caddyfile"},
    )

    event = state.invalidate_context(
        reason="phase_advanced",
        details={"from_phase": "repair", "to_phase": "execute"},
    )

    assert event["invalidated_fact_count"] == 0
    assert "A-caddyfile" not in state.scratchpad.get("_artifact_staleness", {})


def test_environment_changed_preserves_recent_evidence() -> None:
    """Same grace applies for environment_changed invalidation."""
    state = LoopState(cwd="/tmp")
    state.step_count = 2
    state.current_phase = "execute"

    state.reasoning_graph.evidence_records.append(
        EvidenceRecord(
            evidence_id="E-FRESH",
            statement="http_get 200 OK",
            phase="repair",
            tool_name="http_get",
            created_at_step=1,
        )
    )

    state.invalidate_context(
        reason="environment_changed",
        details={"state_change": "Execution environment changed"},
    )

    stale_ids = set(state.scratchpad.get("_observation_staleness", {}).keys())
    assert "E-FRESH" not in stale_ids


def test_path_scoped_verifier_failure_preserves_unrelated_validated_remote_facts() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "execute"
    state.step_count = 18
    state.working_memory.known_fact_meta = [
        MemoryEntry(
            content="ssh_exec: ssh_exec SUCCESS: journalctl -xeu caddy.service",
            created_at_step=14,
            created_phase="execute",
            freshness="current",
        ),
        MemoryEntry(
            content="ssh_file_read: Caddyfile full file (7 lines) /etc/caddy/Caddyfile",
            created_at_step=4,
            created_phase="repair",
            freshness="current",
        ),
    ]
    state.working_memory.known_facts = [entry.content for entry in state.working_memory.known_fact_meta]
    state.artifacts["A-caddyfile"] = ArtifactRecord(
        artifact_id="A-caddyfile",
        kind="ssh_file_read",
        source="/etc/caddy/Caddyfile",
        created_at="2026-05-03T21:34:18+00:00",
        size_bytes=134,
        summary="ssh_file_read SUCCESS: /etc/caddy/Caddyfile",
        tool_name="ssh_file_read",
        metadata={
            "path": "/etc/caddy/Caddyfile",
            "verifier_verdict": "pass",
            "verifier_command": "ssh_file_read /etc/caddy/Caddyfile",
        },
    )

    event = state.invalidate_context(
        reason="verifier_failed",
        details={
            "command": "grep -i error /var/log/caddy/*.log",
            "target": "192.168.1.89 :: grep -i error /var/log/caddy/*.log",
            "failure_mode": "path",
        },
    )

    assert event["paths"] == ["/var/log/caddy/*.log"]
    assert event["invalidated_fact_count"] == 0
    assert "A-caddyfile" not in state.scratchpad.get("_artifact_staleness", {})


def test_phase_advanced_execute_to_repair_preserves_failure_evidence_and_artifacts() -> None:
    """Regression: after a failed installer, execute->repair must keep diagnostic context."""
    state = LoopState(cwd="/tmp")
    state.step_count = 12
    state.current_phase = "repair"

    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-fog-error",
            statement="CRITICAL: ERROR: Site 001-fog does not exist!",
            phase="execute",
            tool_name="ssh_exec",
            created_at_step=8,
            negative=True,
            metadata={"command": "bash /opt/fog/install.sh", "exit_code": 1, "stderr": "Site does not exist"},
        ),
        EvidenceRecord(
            evidence_id="E-php-missing",
            statement="Module php does not exist!",
            phase="execute",
            tool_name="ssh_exec",
            created_at_step=9,
            negative=True,
            metadata={"command": "apachectl configtest"},
        ),
        EvidenceRecord(
            evidence_id="E-db-missing",
            statement="db.cfg MISSING",
            phase="execute",
            tool_name="ssh_file_read",
            created_at_step=9,
            negative=True,
            metadata={"path": "/opt/fog/db.cfg"},
        ),
        EvidenceRecord(
            evidence_id="E-success-claim",
            statement="Installation completed successfully",
            phase="execute",
            tool_name="ssh_exec",
            created_at_step=10,
            negative=False,
            metadata={"command": "bash /opt/fog/install.sh"},
        ),
    ]

    state.artifacts = {
        "A-installer-output": ArtifactRecord(
            artifact_id="A-installer-output",
            kind="ssh_exec",
            source="bash /opt/fog/install.sh",
            created_at="2026-05-14T20:00:00+00:00",
            size_bytes=256,
            summary="Installer output containing failure strings: Site 001-fog does not exist",
            tool_name="ssh_exec",
            metadata={"phase": "execute", "created_at_step": 8, "command": "bash /opt/fog/install.sh", "exit_code": 1},
        ),
        "A-success-artifact": ArtifactRecord(
            artifact_id="A-success-artifact",
            kind="file_write",
            source="/opt/fog/config.php",
            created_at="2026-05-14T20:00:00+00:00",
            size_bytes=128,
            summary="Install succeeded",
            tool_name="file_write",
            metadata={"phase": "execute", "created_at_step": 10, "path": "/opt/fog/config.php", "verifier_verdict": "pass"},
        ),
    }

    event = state.invalidate_context(
        reason="phase_advanced",
        details={"from_phase": "execute", "to_phase": "repair"},
    )

    stale_obs = set(state.scratchpad.get("_observation_staleness", {}).keys())
    stale_art = set(state.scratchpad.get("_artifact_staleness", {}).keys())

    assert "E-fog-error" not in stale_obs
    assert "E-php-missing" not in stale_obs
    assert "E-db-missing" not in stale_obs
    assert "E-success-claim" in stale_obs

    assert "A-installer-output" not in stale_art
    assert "A-success-artifact" in stale_art

    assert event["invalidated_observation_count"] == 1
    assert event["invalidated_artifact_count"] == 1


def test_verifier_failed_preserves_negative_evidence_and_stores_repair_capsule() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 7
    state.current_phase = "repair"
    state.last_failure_class = "logic"
    state.working_memory.failures = ["Tried reinstalling FOG"]
    state.working_memory.next_actions = ["Check PHP module availability"]
    state.failure_events = [
        FailureEvent(
            event_id="F-1",
            timestamp=0.0,
            failure_class="logic",
            severity="recoverable",
            source="verifier",
            message="FOG installer failed",
            suggested_next_action="Inspect apache modules",
        )
    ]
    state.last_verifier_verdict = {"verdict": "fail", "exit_code": 1, "command": "bash /opt/fog/install.sh"}

    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-negative",
            statement="FOG installer returned error",
            phase="execute",
            tool_name="ssh_exec",
            negative=True,
            metadata={"command": "bash /opt/fog/install.sh"},
        ),
        EvidenceRecord(
            evidence_id="E-optimistic",
            statement="All tests pass after patch",
            phase="execute",
            tool_name="ssh_exec",
            negative=False,
            metadata={"command": "bash /opt/fog/install.sh", "verifier_verdict": "pass"},
        ),
    ]

    event = state.invalidate_context(
        reason="verifier_failed",
        details={
            "command": "bash /opt/fog/install.sh",
            "target": "192.168.1.89 :: bash /opt/fog/install.sh",
            "failure_mode": "logic",
        },
    )

    stale_obs = set(state.scratchpad.get("_observation_staleness", {}).keys())
    assert "E-negative" not in stale_obs
    assert "E-optimistic" in stale_obs

    capsule = state.scratchpad.get("_repair_continuity_capsule")
    assert isinstance(capsule, dict)
    assert capsule["command"] == "bash /opt/fog/install.sh"
    assert capsule["failure_mode"] == "logic"
    assert capsule["verdict"] == "fail"
    assert capsule["exit_code"] == 1
    assert capsule["last_attempted_fix"] == "Tried reinstalling FOG"
    assert capsule["next_suggested_action"] == "Check PHP module availability"
    assert capsule["suggested_next_action"] == "Inspect apache modules"

    frame = PromptStateFrameCompiler().compile(state=state)
    wm_text = frame.spine.working_memory_text
    assert "Repair continuity:" in wm_text
    assert "Failed command: bash /opt/fog/install.sh" in wm_text
    assert "Suspected cause: logic" in wm_text
    assert "Last attempted fix: Tried reinstalling FOG" in wm_text
    assert "Next suggested action: Check PHP module availability" in wm_text


def test_repair_continuity_capsule_expires_after_five_turns() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 12
    state.scratchpad["_repair_continuity_capsule"] = {
        "created_at_step": 6,
        "command": "bash /opt/fog/install.sh",
        "failure_mode": "logic",
    }

    frame = PromptStateFrameCompiler().compile(state=state)
    wm_text = frame.spine.working_memory_text
    assert "Repair continuity:" not in wm_text

    state.step_count = 11
    frame = PromptStateFrameCompiler().compile(state=state)
    wm_text = frame.spine.working_memory_text
    assert "Repair continuity:" in wm_text


def test_guard_trip_preserved_context_survives_stale_lane_filters() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 15
    state.current_phase = "execute"
    state.scratchpad["_guard_trip_preserved_summary_ids"] = ["task-0003-summary"]
    state.scratchpad["_guard_trip_preserved_artifact_ids"] = ["A0010"]
    state.scratchpad["_guard_trip_preserved_observation_ids"] = ["E-A0010"]
    state.scratchpad["_guard_trip_recovery_capsule"] = {
        "created_at_step": 15,
        "summary_id": "task-0003-summary",
        "failed_tool": "web_fetch",
        "goal": "Install FOG after web research",
        "preserved_artifact_ids": ["A0010"],
    }
    state.scratchpad["_summary_staleness"] = {
        "task-0003-summary": {"stale": True, "reason": "context_invalidated"}
    }
    state.scratchpad["_artifact_staleness"] = {
        "A0010": {"stale": True, "reason": "phase_advanced"},
        "A-web": {"stale": True, "reason": "phase_advanced"},
    }
    state.scratchpad["_observation_staleness"] = {
        "E-A0010": {"stale": True, "reason": "phase_advanced"},
        "E-web": {"stale": True, "reason": "phase_advanced"},
    }

    summary = EpisodicSummary(
        summary_id="task-0003-summary",
        created_at="2026-05-14T19:45:10+00:00",
        notes=["Task task-0003 failed after PHP dependencies were installed"],
        failed_approaches=["Guard tripped: repeated tool call loop"],
        artifact_ids=["A0010"],
    )
    state.artifacts = {
        "A0010": ArtifactRecord(
            artifact_id="A0010",
            kind="ssh_exec",
            source="apt-get install -y php php-cli",
            created_at="2026-05-14T19:45:01+00:00",
            size_bytes=200,
            summary="ssh_exec SUCCESS: apt-get install -y php php-cli",
            tool_name="ssh_exec",
            metadata={"success": True, "exit_code": 0, "evidence_id": "E-A0010"},
        ),
        "A-web": ArtifactRecord(
            artifact_id="A-web",
            kind="web_fetch",
            source="web_fetch",
            created_at="2026-05-14T19:45:02+00:00",
            size_bytes=100,
            summary="Web fetch budget exhausted",
            tool_name="web_fetch",
            metadata={"evidence_id": "E-web"},
        ),
    }
    state.reasoning_graph.evidence_records = [
        EvidenceRecord(
            evidence_id="E-A0010",
            statement="ssh_exec SUCCESS: apt-get install -y php php-cli",
            phase="execute",
            tool_name="ssh_exec",
            metadata={"artifact_id": "A0010"},
        ),
        EvidenceRecord(
            evidence_id="E-web",
            statement="web_fetch failed: budget exhausted",
            phase="execute",
            tool_name="web_fetch",
            metadata={"artifact_id": "A-web"},
        ),
    ]

    frame = PromptStateFrameCompiler().compile(
        state=state,
        retrieved_summaries=[summary],
        retrieved_artifacts=[
            ArtifactSnippet(artifact_id="A0010", text="PHP dependencies installed"),
            ArtifactSnippet(artifact_id="A-web", text="web fetch failed"),
        ],
    )

    assert [item.summary_id for item in frame.evidence_packet.summaries] == ["task-0003-summary"]
    assert [item.artifact_id for item in frame.artifact_packet.snippets] == ["A0010"]
    assert [item.observation_id for item in frame.evidence_packet.observations] == ["E-A0010"]
    assert "Guard trip recovery:" in frame.spine.working_memory_text
    assert "Do not retry web_fetch with the same arguments" in frame.spine.working_memory_text

    dropped = {(item.lane, tuple(item.dropped_ids)) for item in frame.drop_log}
    assert ("artifact_snippets", ("A-web",)) in dropped
    assert ("normalized_observations", ("E-web",)) in dropped
