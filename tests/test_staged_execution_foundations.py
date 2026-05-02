from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from smallctl.context.step_sandbox import build_step_sandbox_prompt
from smallctl.graph.plan_execution import PlanExecutionEngine
from smallctl.models.conversation import ConversationMessage
from smallctl.state import (
    ExecutionPlan,
    LoopState,
    PlanStep,
    StepEvidenceArtifact,
    StepOutputSpec,
    StepVerifierSpec,
)
from smallctl.tools import control
from smallctl.tools.base import ToolSpec
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.registry import ToolRegistry


def test_staged_plan_fields_round_trip_through_loop_state() -> None:
    state = LoopState()
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="ship staged runtime",
        steps=[
            PlanStep(
                step_id="S1",
                title="Edit schema",
                task="Add durable staged fields",
                tool_allowlist=["file_read"],
                prompt_token_budget=123,
                acceptance=["fields persist"],
                verifiers=[StepVerifierSpec(kind="file_exists", args={"path": "src/x.py"})],
                outputs_expected=[StepOutputSpec(kind="file", ref="src/x.py")],
                max_retries=5,
                retry_count=2,
                failure_reasons=["first failure"],
            )
        ],
    )
    state.plan_execution_mode = True
    state.active_step_id = "S1"
    state.active_step_run_id = "run-1"
    state.step_evidence["S0"] = StepEvidenceArtifact(step_id="S0", summary="dependency done")

    restored = LoopState.from_dict(state.to_dict())
    step = restored.active_plan.find_step("S1")

    assert restored.plan_execution_mode is True
    assert restored.active_step_id == "S1"
    assert restored.active_step_run_id == "run-1"
    assert restored.step_evidence["S0"].summary == "dependency done"
    assert step.task == "Add durable staged fields"
    assert step.tool_allowlist == ["file_read"]
    assert step.prompt_token_budget == 123
    assert step.verifiers[0].kind == "file_exists"
    assert step.outputs_expected[0].kind == "file"
    assert step.max_retries == 5
    assert step.retry_count == 2
    assert step.failure_reasons == ["first failure"]


def test_dispatcher_rejects_tool_outside_active_step_allowlist() -> None:
    async def handler() -> dict:
        return {"success": True, "output": "ran", "error": None, "metadata": {}}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="shell_exec",
            description="run",
            schema={"type": "object", "properties": {}, "required": []},
            handler=handler,
        )
    )
    state = LoopState(
        active_plan=ExecutionPlan(
            plan_id="plan-1",
            goal="goal",
            steps=[PlanStep(step_id="S1", title="Only read", tool_allowlist=["file_read"])],
        ),
        plan_execution_mode=True,
        active_step_id="S1",
        active_step_run_id="run-1",
    )

    result = asyncio.run(ToolDispatcher(registry, state=state).dispatch("shell_exec", {}))

    assert result.success is False
    assert result.metadata["reason"] == "tool_not_allowed_for_step"
    assert result.metadata["step_id"] == "S1"


def test_step_complete_does_not_set_global_task_complete() -> None:
    state = LoopState(
        plan_execution_mode=True,
        active_step_id="S1",
        active_step_run_id="run-1",
    )

    result = asyncio.run(control.step_complete("done", state, SimpleNamespace()))

    assert result["success"] is True
    assert state.scratchpad["_step_complete_requested"] is True
    assert state.scratchpad["_step_complete_message"] == "done"
    assert "_task_complete" not in state.scratchpad


def test_plan_execution_engine_dependency_order_and_retry_blocking(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        steps=[
            PlanStep(step_id="S1", title="first"),
            PlanStep(step_id="S2", title="second", depends_on=["S1"], max_retries=1),
        ],
    )
    engine = PlanExecutionEngine(state)

    assert engine.validate_plan(plan).valid
    assert engine.ready_step_ids(plan) == ["S1"]
    engine.activate_step(plan, "S1")
    engine.complete_step(plan, "S1", StepEvidenceArtifact(step_id="S1", summary="done"))
    assert engine.ready_step_ids(plan) == ["S2"]
    engine.activate_step(plan, "S2")
    engine.fail_step(plan, "S2", "nope")

    assert plan.find_step("S2").status == "blocked"
    assert state.pending_interrupt["kind"] == "staged_step_blocked"


def test_step_prompt_excludes_non_dependency_steps_and_global_recent_messages() -> None:
    state = LoopState()
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal text",
        steps=[
            PlanStep(step_id="S1", title="Dependency"),
            PlanStep(step_id="S2", title="Active", depends_on=["S1"], task="active task"),
            PlanStep(step_id="S3", title="FORBIDDEN non dependency"),
        ],
    )
    state.active_step_id = "S2"
    state.active_step_run_id = "run-2"
    state.step_evidence["S1"] = StepEvidenceArtifact(step_id="S1", summary="dependency evidence")
    state.append_message(ConversationMessage(role="user", content="FORBIDDEN global recent message"))
    harness = SimpleNamespace(state=state, config=SimpleNamespace(staged_step_prompt_tokens=0))

    prompt = build_step_sandbox_prompt(harness, state.active_plan.find_step("S2"))
    rendered = "\n".join(str(message.get("content", "")) for message in prompt)

    assert "goal text" in rendered
    assert "active task" in rendered
    assert "dependency evidence" in rendered
    assert "FORBIDDEN non dependency" not in rendered
    assert "FORBIDDEN global recent message" not in rendered
