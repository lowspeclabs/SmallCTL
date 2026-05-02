from __future__ import annotations

from typing import Any

from langgraph.graph import END

from .nodes import apply_planning_tool_outcomes, model_call, prepare_planning_prompt, select_planning_tools
from .runtime import LoopGraphRuntime
from .runtime_base import (
    LoopGraphPayload,
    RuntimeGraphSpec,
    apply_outcomes_node,
    interpret_node,
    load_runtime_state,
    model_call_node,
    prepare_prompt_node,
    route_if_final_else_pending_else,
    route_if_interrupt_else_final_else,
    serialize_runtime_state,
)
from .nodes import interpret_planning_output, resume_planning_run
from .state import GraphRunState


class PlanningGraphRuntime(LoopGraphRuntime):
    _run_mode = "planning"
    _run_execution_message = "executing planning runtime"
    _empty_result_message = "Planning graph ended without a terminal result."

    GRAPH_SPEC = RuntimeGraphSpec(
        node_map={
            "initialize_run": "_initialize_run_node",
            "prepare_step": "_prepare_step_node",
            "prepare_prompt": "_prepare_planning_prompt_node",
            "model_call": "_planning_model_call_node",
            "interpret_planning_output": "_interpret_planning_output_node",
            "dispatch_tools": "_dispatch_tools_node",
            "persist_tool_results": "_persist_tool_results_node",
            "apply_tool_outcomes": "_apply_planning_tool_outcomes_node",
            "interrupt_for_human": "_interrupt_for_human_node",
        },
        edge_map={
            "initialize_run": ("_route_after_initialize", {"prepare_step": "prepare_step", END: END}),
            "prepare_step": (
                "_route_after_prepare_step",
                {"prepare_prompt": "prepare_prompt", "dispatch_tools": "dispatch_tools", END: END},
            ),
            "prepare_prompt": ("_route_after_prepare_prompt", {"model_call": "model_call", END: END}),
            "model_call": (
                "_route_after_model_call",
                {"interpret_model_output": "interpret_planning_output", END: END},
            ),
            "interpret_planning_output": (
                "_route_after_planning_interpret",
                {"dispatch_tools": "dispatch_tools", "prepare_step": "prepare_step", END: END},
            ),
            "dispatch_tools": ("_route_after_dispatch", {"persist_tool_results": "persist_tool_results", END: END}),
            "apply_tool_outcomes": (
                "_route_after_planning_apply",
                {"prepare_step": "prepare_step", "interrupt_for_human": "interrupt_for_human", END: END},
            ),
        },
        static_edges=[("persist_tool_results", "apply_tool_outcomes"), ("interrupt_for_human", "prepare_step")],
    )

    def _before_run(self, harness: Any) -> None:
        harness.state.planning_mode_enabled = True

    async def resume(self, human_input: str) -> dict[str, object]:
        harness = self.deps.harness
        pending = harness.state.pending_interrupt or {}
        if isinstance(pending, dict) and pending.get("kind") == "plan_execute_approval":
            normalized = human_input.strip().lower()
            if normalized in {"yes", "y", "approve", "approved", "execute", "go ahead", "run it"}:
                graph_state = GraphRunState(
                    loop_state=harness.state,
                    thread_id=harness.state.thread_id or harness.conversation_id,
                    run_mode=self._run_mode,
                )
                await resume_planning_run(graph_state, self.deps, human_input=human_input)
                plan = harness.state.active_plan or harness.state.draft_plan
                if plan is not None:
                    harness.state.planning_mode_enabled = False
                    if bool(getattr(getattr(harness, "config", None), "staged_execution_enabled", False)):
                        from .runtime_staged import StagedExecutionRuntime

                        harness.state.plan_execution_mode = True
                        return await StagedExecutionRuntime.from_harness(
                            harness,
                            event_handler=self.deps.event_handler,
                        ).run(plan.goal or harness.state.run_brief.original_task or human_input)
                    return await LoopGraphRuntime.from_harness(
                        harness,
                        event_handler=self.deps.event_handler,
                    ).run(plan.goal or harness.state.run_brief.original_task or human_input)
        return await self._resume_langgraph(human_input)

    async def _prepare_planning_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await prepare_prompt_node(self, payload, prepare_planning_prompt)

    async def _planning_model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await model_call_node(
            self,
            payload,
            select_tools_fn=select_planning_tools,
            model_call_fn=model_call,
        )

    async def _interpret_planning_output_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await interpret_node(self, payload, interpret_planning_output)

    async def _apply_planning_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await apply_outcomes_node(
            self,
            payload,
            apply_planning_tool_outcomes,
            clear_final_result_on_interrupt=True,
        )

    @staticmethod
    def _route_after_planning_interpret(payload: LoopGraphPayload) -> str:
        return route_if_final_else_pending_else(payload, pending_step="dispatch_tools", fallback_step="prepare_step")

    @staticmethod
    def _route_after_planning_apply(payload: LoopGraphPayload) -> str:
        return route_if_interrupt_else_final_else(
            payload,
            interrupt_step="interrupt_for_human",
            fallback_step="prepare_step",
        )
