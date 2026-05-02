from __future__ import annotations

from typing import Any

from langgraph.graph import END

from .nodes import (
    apply_tool_outcomes,
    dispatch_tools,
    interpret_model_output,
    model_call,
    persist_tool_results,
    prepare_staged_prompt,
    select_staged_tools,
)
from .plan_execution import PlanExecutionEngine
from .plan_verification import StepCompletionGate, compact_step_evidence
from .runtime import LoopGraphRuntime
from .runtime_base import (
    LoopGraphPayload,
    RuntimeGraphSpec,
    apply_outcomes_node,
    interpret_node,
    load_runtime_state,
    model_call_node,
    prepare_prompt_node,
    route_if_final_else,
    route_if_final_else_pending_else,
    route_if_interrupt_else_final_else_pending_else,
    serialize_runtime_state,
)


class StagedExecutionRuntime(LoopGraphRuntime):
    _run_mode = "staged_execution"
    _run_execution_message = "executing staged execution runtime"
    _empty_result_message = "Staged execution graph ended without a terminal result."

    GRAPH_SPEC = RuntimeGraphSpec(
        node_map={
            "initialize_run": "_initialize_run_node",
            "activate_or_finalize_step": "_activate_or_finalize_step_node",
            "prepare_prompt": "_prepare_staged_prompt_node",
            "model_call": "_staged_model_call_node",
            "interpret_model_output": "_interpret_model_output_node",
            "dispatch_tools": "_dispatch_tools_node",
            "persist_tool_results": "_persist_tool_results_node",
            "apply_tool_outcomes": "_apply_staged_tool_outcomes_node",
            "verify_step_completion": "_verify_step_completion_node",
            "interrupt_for_human": "_interrupt_for_human_node",
        },
        edge_map={
            "initialize_run": (
                "_route_after_initialize",
                {"activate_or_finalize_step": "activate_or_finalize_step", END: END},
            ),
            "activate_or_finalize_step": (
                "_route_after_activate",
                {"prepare_prompt": "prepare_prompt", "interrupt_for_human": "interrupt_for_human", END: END},
            ),
            "prepare_prompt": ("_route_after_prepare_prompt", {"model_call": "model_call", END: END}),
            "model_call": (
                "_route_after_model_call",
                {"interpret_model_output": "interpret_model_output", END: END},
            ),
            "interpret_model_output": (
                "_route_after_interpret",
                {"dispatch_tools": "dispatch_tools", "activate_or_finalize_step": "activate_or_finalize_step", END: END},
            ),
            "dispatch_tools": ("_route_after_dispatch", {"persist_tool_results": "persist_tool_results", END: END}),
            "apply_tool_outcomes": (
                "_route_after_staged_apply",
                {
                    "dispatch_tools": "dispatch_tools",
                    "verify_step_completion": "verify_step_completion",
                    "prepare_prompt": "prepare_prompt",
                    "interrupt_for_human": "interrupt_for_human",
                    END: END,
                },
            ),
            "verify_step_completion": (
                "_route_after_verify",
                {
                    "activate_or_finalize_step": "activate_or_finalize_step",
                    "prepare_prompt": "prepare_prompt",
                    "interrupt_for_human": "interrupt_for_human",
                    END: END,
                },
            ),
        },
        static_edges=[("persist_tool_results", "apply_tool_outcomes"), ("interrupt_for_human", "activate_or_finalize_step")],
    )

    def _before_run(self, harness: Any) -> None:
        harness.state.plan_execution_mode = True

    async def resume(self, human_input: str) -> dict[str, object]:
        self._apply_blocked_step_resume_choice(human_input)
        return await self._resume_langgraph(human_input)

    async def _initialize_run_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        harness = self.deps.harness
        if harness._cancel_requested:
            graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
            return serialize_runtime_state(graph_state)
        if not harness.state.thread_id:
            harness.state.thread_id = harness.conversation_id
        harness.state.pending_interrupt = None
        harness.state.plan_execution_mode = True
        task = str(payload.get("input_task", "") or "")
        if task and not harness.state.run_brief.original_task:
            harness.state.run_brief.original_task = task
        return serialize_runtime_state(graph_state)

    async def _activate_or_finalize_step_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        harness = self.deps.harness
        if graph_state.final_result is not None:
            return serialize_runtime_state(graph_state)
        plan = harness.state.active_plan or harness.state.draft_plan
        if plan is None:
            graph_state.final_result = harness._failure("No approved plan is available for staged execution.", error_type="staged_execution")
            return serialize_runtime_state(graph_state)

        engine = PlanExecutionEngine(harness.state)
        validation = engine.validate_plan(plan)
        if not validation.valid:
            harness.log.info(
                "staged_plan_validated plan_id=%s valid=%s errors=%s",
                plan.plan_id,
                False,
                validation.errors,
            )
            graph_state.final_result = harness._failure(
                "Staged plan validation failed.",
                error_type="staged_plan_validation",
                details={"errors": validation.errors},
            )
            return serialize_runtime_state(graph_state)
        harness.log.info(
            "staged_plan_validated plan_id=%s valid=%s step_count=%d",
            plan.plan_id,
            True,
            len(list(plan.iter_steps())),
        )

        if engine.is_plan_complete(plan):
            plan.status = "completed"
            harness.state.plan_execution_mode = False
            harness.state.active_step_id = ""
            harness.state.active_step_run_id = ""
            harness.log.info(
                "staged_plan_completed plan_id=%s steps_completed=%d",
                plan.plan_id,
                len(list(plan.iter_steps())),
            )
            graph_state.final_result = {
                "status": "complete",
                "message": f"Completed staged plan {plan.plan_id}.",
                "plan_id": plan.plan_id,
                "step_evidence": harness.state.step_evidence,
            }
            return serialize_runtime_state(graph_state)

        active = engine.get_next_step(plan)
        if active is None:
            graph_state.final_result = harness._failure(
                "No runnable staged plan step is available.",
                error_type="staged_execution",
                details={"plan_id": plan.plan_id},
            )
            return serialize_runtime_state(graph_state)
        if active.status != "in_progress" or harness.state.active_step_id != active.step_id:
            active = engine.activate_step(plan, active.step_id)
            harness.log.info(
                "staged_step_activated plan_id=%s step_id=%s step_run_id=%s attempt=%d",
                plan.plan_id,
                active.step_id,
                harness.state.active_step_run_id,
                int(active.retry_count or 0) + 1,
            )
        return serialize_runtime_state(graph_state)

    async def _prepare_staged_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await prepare_prompt_node(self, payload, prepare_staged_prompt)

    async def _staged_model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await model_call_node(self, payload, select_tools_fn=select_staged_tools, model_call_fn=model_call)

    async def _apply_staged_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await apply_outcomes_node(
            self,
            payload,
            apply_tool_outcomes,
            clear_final_result_on_interrupt=True,
        )

    async def _verify_step_completion_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        harness = self.deps.harness
        plan = harness.state.active_plan or harness.state.draft_plan
        step = plan.find_step(harness.state.active_step_id) if plan is not None and harness.state.active_step_id else None
        if plan is None or step is None:
            graph_state.final_result = harness._failure("No active step to verify.", error_type="staged_verification")
            return serialize_runtime_state(graph_state)
        engine = PlanExecutionEngine(harness.state)
        if harness.state.scratchpad.get("_step_failed_requested"):
            reason = str(harness.state.scratchpad.get("_step_failed_message") or "Step failed.")
            engine.fail_step(plan, step.step_id, reason)
            harness.log.info(
                "staged_step_failed plan_id=%s step_id=%s step_run_id=%s reason=%s",
                plan.plan_id,
                step.step_id,
                harness.state.active_step_run_id,
                reason,
            )
            if harness.state.pending_interrupt:
                graph_state.interrupt_payload = harness.state.pending_interrupt
            return serialize_runtime_state(graph_state)

        result = await StepCompletionGate().verify_step(harness, step)
        harness.log.info(
            "staged_step_verified plan_id=%s step_id=%s step_run_id=%s passed=%s",
            plan.plan_id,
            step.step_id,
            result.step_run_id,
            result.passed,
        )
        if result.passed:
            evidence = compact_step_evidence(harness, step, result)
            harness.log.info(
                "staged_step_evidence_created plan_id=%s step_id=%s step_run_id=%s attempt=%d artifact_ids=%d",
                plan.plan_id,
                step.step_id,
                evidence.step_run_id,
                evidence.attempt,
                len(evidence.artifact_ids),
            )
            engine.complete_step(plan, step.step_id, evidence)
        else:
            reason = "Step verification failed: " + ", ".join(result.failed_criteria)
            engine.fail_step(plan, step.step_id, reason)
            harness.log.info(
                "staged_step_failed plan_id=%s step_id=%s step_run_id=%s reason=%s",
                plan.plan_id,
                step.step_id,
                result.step_run_id,
                reason,
            )
            if harness.state.pending_interrupt:
                graph_state.interrupt_payload = harness.state.pending_interrupt
        return serialize_runtime_state(graph_state)

    @staticmethod
    def _route_after_initialize(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "activate_or_finalize_step")

    @staticmethod
    def _route_after_activate(payload: LoopGraphPayload) -> str:
        if payload.get("interrupt_payload") is not None:
            return "interrupt_for_human"
        return route_if_final_else(payload, "prepare_prompt")

    @staticmethod
    def _route_after_interpret(payload: LoopGraphPayload) -> str:
        return route_if_final_else_pending_else(
            payload,
            pending_step="dispatch_tools",
            fallback_step="activate_or_finalize_step",
        )

    @staticmethod
    def _route_after_staged_apply(payload: LoopGraphPayload) -> str:
        if payload.get("interrupt_payload") is not None:
            return "interrupt_for_human"
        if payload.get("final_result") is not None:
            return END
        scratchpad = payload.get("loop_state", {}).get("scratchpad", {})
        if isinstance(scratchpad, dict) and (
            scratchpad.get("_step_complete_requested") or scratchpad.get("_step_failed_requested")
        ):
            return "verify_step_completion"
        if payload.get("pending_tool_calls"):
            return "dispatch_tools"
        return "prepare_prompt"

    @staticmethod
    def _route_after_verify(payload: LoopGraphPayload) -> str:
        return route_if_interrupt_else_final_else_pending_else(
            payload,
            interrupt_step="interrupt_for_human",
            pending_step="dispatch_tools",
            fallback_step="activate_or_finalize_step",
        )

    def _apply_blocked_step_resume_choice(self, human_input: str) -> None:
        pending = self.deps.harness.state.pending_interrupt or {}
        if not isinstance(pending, dict) or pending.get("kind") != "staged_step_blocked":
            return
        choice = human_input.strip().lower()
        plan = self.deps.harness.state.active_plan or self.deps.harness.state.draft_plan
        step_id = str(pending.get("step_id") or "")
        step = plan.find_step(step_id) if plan is not None and step_id else None
        if step is None:
            return
        if "skip" in choice:
            step.status = "skipped"
        elif "retry" in choice:
            step.status = "pending"
        self.deps.harness.state.pending_interrupt = None
        self.deps.harness.state.touch()
