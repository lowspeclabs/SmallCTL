from __future__ import annotations

from types import SimpleNamespace

from smallctl.harness.reflexion_service import ReflexionService
from smallctl.recovery_schema import FailureEvent, Subtask, SubtaskLedger
from smallctl.state import LoopState


def _harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(
            reflexion_enabled=True,
            reflexion_max_items=5,
            reflexion_min_failure_severity="warning",
        ),
        _runlog=lambda *args, **kwargs: None,
    )


def _failure(failure_class: str = "wrong_path") -> FailureEvent:
    return FailureEvent(
        event_id="F1",
        timestamp=1.0,
        failure_class=failure_class,
        severity="warning",
        source="fama",
        message=f"{failure_class}: failed",
        evidence=["No such file or directory"],
        subtask_id="S1",
        suggested_next_action="Verify the path.",
    )


def test_reflexion_service_creates_template_reflection() -> None:
    state = LoopState()
    service = ReflexionService(_harness(state))

    reflection = service.maybe_create_reflection(_failure("wrong_path"))

    assert reflection is not None
    assert reflection.failure_class == "wrong_path"
    assert "path" in reflection.lesson.lower()
    assert state.reflexion_memory == [reflection]
    assert state.scratchpad["_recovery_metrics"]["reflections_created"] == 1


def test_reflexion_service_dedupes_and_reinforces() -> None:
    state = LoopState()
    service = ReflexionService(_harness(state))

    first = service.maybe_create_reflection(_failure("empty_write"))
    second = service.maybe_create_reflection(_failure("empty_write"))

    assert first is second
    assert len(state.reflexion_memory) == 1
    assert state.reflexion_memory[0].score > 1.0
    assert state.scratchpad["_recovery_metrics"]["reflections_created"] == 1
    assert state.scratchpad["_recovery_metrics"]["reflections_reinforced"] == 1


def test_reflexion_service_selects_active_subtask_reflections() -> None:
    state = LoopState()
    state.subtask_ledger = SubtaskLedger(
        task_id="task-1",
        subtasks=[Subtask(subtask_id="S1", title="One", goal="one", status="active")],
        active_subtask_id="S1",
    )
    service = ReflexionService(_harness(state))
    service.maybe_create_reflection(_failure("wrong_path"))
    other = _failure("verifier_failed")
    other.event_id = "F2"
    other.subtask_id = "S2"
    service.maybe_create_reflection(other)

    selected = service.select_for_prompt(
        task_text="task",
        active_subtask=state.subtask_ledger.active(),
        limit=3,
    )

    assert [item.subtask_id for item in selected] == ["S1"]


def test_reflexion_service_ignores_below_min_severity() -> None:
    state = LoopState()
    harness = _harness(state)
    harness.config.reflexion_min_failure_severity = "recoverable"
    service = ReflexionService(harness)

    assert service.maybe_create_reflection(_failure("wrong_path")) is None
    assert state.reflexion_memory == []


def test_reflexion_service_has_tool_plan_invalid_template() -> None:
    state = LoopState()
    service = ReflexionService(_harness(state))

    reflection = service.maybe_create_reflection(_failure("tool_plan_invalid"))

    assert reflection is not None
    assert "toolplan" in reflection.lesson.lower()


def test_reflexion_service_has_tool_plan_unsafe_template() -> None:
    state = LoopState()
    service = ReflexionService(_harness(state))

    reflection = service.maybe_create_reflection(_failure("tool_plan_unsafe"))

    assert reflection is not None
    assert "safety policy" in reflection.lesson.lower()
