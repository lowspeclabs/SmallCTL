from __future__ import annotations

from typing import Any

from langgraph.graph import END

from .nodes import (
    apply_chat_tool_outcomes,
    interpret_chat_output,
    model_call,
    prepare_chat_prompt,
    select_chat_tools,
)
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
)
from .runtime import LoopGraphRuntime


class ChatGraphRuntime(LoopGraphRuntime):
    _run_mode = "chat"
    _run_execution_message = "executing chat runtime"
    _empty_result_message = "Chat graph ended without a terminal result."

    GRAPH_SPEC = RuntimeGraphSpec(
        node_map={
            "initialize_run": "_initialize_run_node",
            "prepare_step": "_prepare_step_node",
            "prepare_chat_prompt": "_prepare_chat_prompt_node",
            "model_call": "_chat_model_call_node",
            "interpret_chat_output": "_interpret_chat_output_node",
            "dispatch_tools": "_dispatch_tools_node",
            "persist_tool_results": "_persist_tool_results_node",
            "apply_chat_tool_outcomes": "_apply_chat_tool_outcomes_node",
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
                    "prepare_prompt": "prepare_chat_prompt",
                    "dispatch_tools": "dispatch_tools",
                    END: END,
                },
            ),
            "prepare_chat_prompt": (
                "_route_after_prepare_prompt",
                {"model_call": "model_call", END: END},
            ),
            "model_call": (
                "_route_after_model_call",
                {"interpret_model_output": "interpret_chat_output", END: END},
            ),
            "interpret_chat_output": (
                "_route_after_interpret",
                {"dispatch_tools": "dispatch_tools", "prepare_step": "prepare_step", END: END},
            ),
            "dispatch_tools": (
                "_route_after_dispatch",
                {"persist_tool_results": "persist_tool_results", END: END},
            ),
            "apply_chat_tool_outcomes": (
                "_route_after_chat_apply",
                {
                    "dispatch_tools": "dispatch_tools",
                    "prepare_step": "prepare_step",
                    "interrupt_for_human": "interrupt_for_human",
                    END: END,
                },
            ),
        },
        static_edges=[
            ("persist_tool_results", "apply_chat_tool_outcomes"),
        ],
    )

    async def _prepare_chat_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await prepare_prompt_node(self, payload, prepare_chat_prompt)

    async def _chat_model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await model_call_node(
            self,
            payload,
            select_tools_fn=select_chat_tools,
            model_call_fn=model_call,
        )

    async def _interpret_chat_output_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await interpret_node(self, payload, interpret_chat_output)

    async def _apply_chat_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await apply_outcomes_node(self, payload, apply_chat_tool_outcomes)

    @staticmethod
    def _route_after_chat_apply(payload: LoopGraphPayload) -> str:
        return route_if_interrupt_else_final_else_pending_else(
            payload,
            interrupt_step="interrupt_for_human",
            pending_step="dispatch_tools",
            fallback_step="prepare_step",
        )
