from __future__ import annotations

from smallctl.context.assembler import PromptAssembler
from smallctl.recovery_schema import FailureEvent, ReflectionMemory, Subtask, SubtaskLedger
from smallctl.state import LoopState


def _state() -> LoopState:
    state = LoopState()
    state.run_brief.original_task = "Fix src/app.py and rerun verifier"
    state.scratchpad["_recovery_config"] = {
        "reflexion_enabled": True,
        "reflexion_inject_top_k": 3,
        "subtask_inject_completed_limit": 1,
    }
    state.subtask_ledger = SubtaskLedger(
        task_id="task-1",
        subtasks=[
            Subtask(
                subtask_id="S1",
                title="Repair verifier failure",
                goal="Fix failing test",
                status="active",
                attempts=2,
                blockers=["verifier_failed: pytest failed"],
                next_action="Patch one failing assertion.",
            )
        ],
        active_subtask_id="S1",
    )
    state.failure_events.append(
        FailureEvent(
            event_id="F1",
            timestamp=1.0,
            failure_class="verifier_failed",
            severity="warning",
            source="verifier",
            message="verifier_failed: pytest tests/test_app.py",
            evidence=["pytest failed"],
            subtask_id="S1",
            suggested_next_action="Read the failing output and patch one narrow cause.",
        )
    )
    state.reflexion_memory.append(
        ReflectionMemory(
            reflection_id="R1",
            timestamp=2.0,
            task_id="task-1",
            failure_class="verifier_failed",
            subtask_id="S1",
            lesson="The task is not complete because verification failed.",
            avoid="Do not call task_complete while the verifier is failing.",
            next_safe_action="Rerun the smallest check.",
            evidence_summary="pytest failed",
        )
    )
    return state


def test_prompt_includes_recovery_guidance_as_working_memory() -> None:
    state = _state()
    assembly = PromptAssembler().build_messages(state=state, system_prompt="System.")
    content = assembly.messages[0]["content"]

    assert "Recovery guidance:" in content
    assert "Active subtask S1 [active]: Repair verifier failure" in content
    assert "Latest failure: verifier_failed" in content
    assert "Do not call task_complete while the verifier is failing" in content
    assert assembly.section_tokens["recovery_guidance"] > 0
    assert state.reflexion_memory[0].used_count == 1
    assert state.scratchpad["_recovery_metrics"]["reflections_injected"] == 1


def test_prompt_omits_recovery_guidance_when_disabled() -> None:
    state = _state()
    state.scratchpad["_recovery_config"]["reflexion_enabled"] = False
    assembly = PromptAssembler().build_messages(state=state, system_prompt="System.")

    assert "Recovery guidance:" not in assembly.messages[0]["content"]
    assert assembly.section_tokens["recovery_guidance"] == 0
