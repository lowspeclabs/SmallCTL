from __future__ import annotations

from typing import Any

from langgraph.graph import END
from langgraph.types import Command, interrupt

from .deps import GraphRuntimeDeps
from .nodes import (
    apply_planning_tool_outcomes,
    apply_tool_outcomes,
    dispatch_tools,
    initialize_loop_run,
    initialize_planning_run,
    interpret_planning_output,
    interpret_model_output,
    model_call,
    persist_tool_results,
    prepare_planning_prompt,
    prepare_indexer_prompt,
    prepare_loop_step,
    prepare_prompt,
    resume_loop_run,
    resume_planning_run,
    select_indexer_tools,
    select_loop_tools,
    select_planning_tools,
)
from .state import GraphRunState
from .runtime_base import (
    CompiledGraphRuntimeBase,
    checkpoint_config,
    LoopGraphPayload,
    RuntimeGraphSpec,
    apply_outcomes_node,
    coerce_graph_values_payload,
    interpret_node,
    load_runtime_state,
    model_call_node,
    prepare_prompt_node,
    route_if_final_else,
    route_if_final_else_pending_else,
    route_if_interrupt_else_final_else,
    route_if_interrupt_else_final_else_pending_else,
    serialize_runtime_state,
)


LANGGRAPH_RECURSION_LIMIT = 512


class LoopGraphRuntime(CompiledGraphRuntimeBase):
    _run_mode = "loop"
    _run_execution_message = "executing loop runtime"
    _empty_result_message = "Graph loop ended without a terminal result."

    GRAPH_SPEC = RuntimeGraphSpec(
        node_map={
            "initialize_run": "_initialize_run_node",
            "prepare_step": "_prepare_step_node",
            "prepare_prompt": "_prepare_prompt_node",
            "model_call": "_model_call_node",
            "interpret_model_output": "_interpret_model_output_node",
            "dispatch_tools": "_dispatch_tools_node",
            "persist_tool_results": "_persist_tool_results_node",
            "apply_tool_outcomes": "_apply_tool_outcomes_node",
            "interrupt_for_human": "_interrupt_for_human_node",
        },
        edge_map={
            "initialize_run": (
                "_route_after_initialize",
                {"prepare_step": "prepare_step", END: END},
            ),
            "prepare_step": (
                "_route_after_prepare_step",
                {
                    "prepare_prompt": "prepare_prompt",
                    "dispatch_tools": "dispatch_tools",
                    END: END,
                },
            ),
            "prepare_prompt": (
                "_route_after_prepare_prompt",
                {"model_call": "model_call", END: END},
            ),
            "model_call": (
                "_route_after_model_call",
                {"interpret_model_output": "interpret_model_output", END: END},
            ),
            "interpret_model_output": (
                "_route_after_interpret",
                {"dispatch_tools": "dispatch_tools", "prepare_step": "prepare_step", END: END},
            ),
            "dispatch_tools": (
                "_route_after_dispatch",
                {"persist_tool_results": "persist_tool_results", END: END},
            ),
            "apply_tool_outcomes": (
                "_route_after_apply",
                {
                    "dispatch_tools": "dispatch_tools",
                    "prepare_step": "prepare_step",
                    "interrupt_for_human": "interrupt_for_human",
                    END: END,
                },
            ),
        },
        static_edges=[
            ("persist_tool_results", "apply_tool_outcomes"),
            ("interrupt_for_human", "prepare_step"),
        ],
    )

    async def resume(self, human_input: str) -> dict[str, object]:
        return await self._resume_langgraph(human_input)

    async def _resume_langgraph(self, human_input: str) -> dict[str, object]:
        harness = self.deps.harness
        if not harness.has_pending_interrupt():
            return harness._finalize(
                harness._failure(
                    "No pending interrupt to resume.",
                    error_type="interrupt",
                )
        )
        command = Command(resume=human_input)
        return await self._execute_langgraph(command)

    async def _initialize_run_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        await initialize_loop_run(graph_state, self.deps, task=str(payload.get("input_task", "")))
        return serialize_runtime_state(graph_state)

    async def _prepare_step_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        await prepare_loop_step(graph_state, self.deps)
        return serialize_runtime_state(graph_state)

    async def _prepare_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await prepare_prompt_node(self, payload, prepare_prompt)

    async def _model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await model_call_node(
            self,
            payload,
            select_tools_fn=select_loop_tools,
            model_call_fn=model_call,
        )

    async def _interpret_model_output_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await interpret_node(self, payload, interpret_model_output)

    async def _dispatch_tools_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        await dispatch_tools(graph_state, self.deps)
        return serialize_runtime_state(graph_state)

    async def _persist_tool_results_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        await persist_tool_results(graph_state, self.deps)
        return serialize_runtime_state(graph_state)

    async def _apply_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await apply_outcomes_node(
            self,
            payload,
            apply_tool_outcomes,
            clear_final_result_on_interrupt=True,
        )

    async def _interrupt_for_human_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        payload_value = graph_state.interrupt_payload or graph_state.loop_state.pending_interrupt or {}
        human_input = interrupt(payload_value)
        await resume_loop_run(graph_state, self.deps, human_input=str(human_input))
        graph_state.interrupt_payload = None
        return serialize_runtime_state(graph_state)

    @staticmethod
    def _route_after_initialize(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "prepare_step")

    @staticmethod
    def _route_after_prepare_step(payload: LoopGraphPayload) -> str:
        return route_if_final_else_pending_else(
            payload,
            pending_step="dispatch_tools",
            fallback_step="prepare_prompt",
        )

    @staticmethod
    def _route_after_prepare_prompt(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "model_call")

    @staticmethod
    def _route_after_model_call(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "interpret_model_output")

    @staticmethod
    def _route_after_interpret(payload: LoopGraphPayload) -> str:
        return route_if_final_else_pending_else(
            payload,
            pending_step="dispatch_tools",
            fallback_step="prepare_step",
        )

    @staticmethod
    def _route_after_dispatch(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "persist_tool_results")

    @staticmethod
    def _route_after_apply(payload: LoopGraphPayload) -> str:
        return route_if_interrupt_else_final_else_pending_else(
            payload,
            interrupt_step="interrupt_for_human",
            pending_step="dispatch_tools",
            fallback_step="prepare_step",
        )
from .runtime_chat import ChatGraphRuntime
from .runtime_specialized import IndexerGraphRuntime, PlanningGraphRuntime
from .runtime_auto import AutoGraphRuntime
