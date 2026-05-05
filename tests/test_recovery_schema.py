from __future__ import annotations

from smallctl.recovery_schema import FailureEvent, ReflectionMemory, Subtask, SubtaskLedger
from smallctl.state import LoopState


def test_loop_state_round_trips_recovery_state() -> None:
    state = LoopState(step_count=4)
    state.failure_events.append(
        FailureEvent(
            event_id="F1",
            timestamp=1.5,
            failure_class="wrong_path",
            severity="warning",
            source="fama",
            message="Path failed.",
            evidence=["No such file or directory"],
            fama_kind="remote_local_confusion",
            tool_name="file_read",
            operation_id="op-1",
            suggested_next_action="Verify the path.",
        )
    )
    state.reflexion_memory.append(
        ReflectionMemory(
            reflection_id="R1",
            timestamp=2.5,
            task_id="task-1",
            failure_class="wrong_path",
            subtask_id="S1",
            lesson="The path was unverified.",
            avoid="Do not guess paths.",
            next_safe_action="Run dir_list.",
            evidence_summary="file_read failed",
        )
    )
    state.subtask_ledger = SubtaskLedger(
        task_id="task-1",
        subtasks=[
            Subtask(
                subtask_id="S1",
                title="Inspect target",
                goal="Find the correct file",
                status="active",
                evidence=["read repo tree"],
                failure_classes=["wrong_path"],
            )
        ],
        active_subtask_id="S1",
    )

    restored = LoopState.from_dict(state.to_dict())

    assert restored.failure_events[0].failure_class == "wrong_path"
    assert restored.failure_events[0].suggested_next_action == "Verify the path."
    assert restored.reflexion_memory[0].next_safe_action == "Run dir_list."
    assert restored.subtask_ledger is not None
    assert restored.subtask_ledger.active() is not None
    assert restored.subtask_ledger.active().subtask_id == "S1"


def test_loop_state_accepts_old_payload_without_recovery_fields() -> None:
    restored = LoopState.from_dict({"schema_version": 2, "current_phase": "explore"})

    assert restored.failure_events == []
    assert restored.reflexion_memory == []
    assert restored.subtask_ledger is None


def test_recovery_coercion_drops_invalid_entries() -> None:
    restored = LoopState.from_dict(
        {
            "failure_events": [{"event_id": "", "failure_class": "wrong_path"}],
            "reflexion_memory": [{"reflection_id": "R1"}],
            "subtask_ledger": {
                "task_id": "task-1",
                "active_subtask_id": "missing",
                "subtasks": [{"subtask_id": "S1", "title": "One", "goal": ""}],
            },
        }
    )

    assert restored.failure_events == []
    assert restored.reflexion_memory == []
    assert restored.subtask_ledger is not None
    assert restored.subtask_ledger.active_subtask_id is None
    assert [task.subtask_id for task in restored.subtask_ledger.subtasks] == ["S1"]
