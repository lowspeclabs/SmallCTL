from __future__ import annotations

from langgraph.graph import END

from .nodes import apply_chat_tool_outcomes, model_call, prepare_indexer_prompt, select_indexer_tools
from .runtime import LoopGraphRuntime
from .runtime_base import (
    LoopGraphPayload,
    RuntimeGraphSpec,
    apply_outcomes_node,
    checkpoint_config,
    coerce_graph_values_payload,
    interpret_node,
    load_runtime_state,
    model_call_node,
    prepare_prompt_node,
    route_if_final_else,
    serialize_runtime_state,
)
from .nodes import interpret_chat_output


class IndexerGraphRuntime(LoopGraphRuntime):
    """A rigid code indexing runtime optimized for SLM traversal and extraction."""

    _run_mode = "indexer"
    _run_execution_message = "executing indexer runtime"

    GRAPH_SPEC = RuntimeGraphSpec(
        node_map={
            "initialize_run": "_initialize_run_node",
            "prepare_step": "_prepare_step_node",
            "prepare_indexer_prompt": "_prepare_indexer_prompt_node",
            "model_call": "_indexer_model_call_node",
            "interpret_indexer_output": "_interpret_indexer_output_node",
            "dispatch_tools": "_dispatch_tools_node",
            "persist_tool_results": "_persist_tool_results_node",
            "apply_indexer_tool_outcomes": "_apply_indexer_tool_outcomes_node",
        },
        edge_map={
            "initialize_run": ("_route_after_initialize", {"prepare_step": "prepare_step", END: END}),
            "prepare_step": (
                "_route_after_prepare_step",
                {"prepare_prompt": "prepare_indexer_prompt", "dispatch_tools": "dispatch_tools", END: END},
            ),
            "prepare_indexer_prompt": ("_route_after_prepare_prompt", {"model_call": "model_call", END: END}),
            "model_call": (
                "_route_after_model_call",
                {"interpret_model_output": "interpret_indexer_output", END: END},
            ),
            "interpret_indexer_output": (
                "_route_after_interpret",
                {"dispatch_tools": "dispatch_tools", END: END},
            ),
            "dispatch_tools": ("_route_after_dispatch", {"persist_tool_results": "persist_tool_results", END: END}),
            "apply_indexer_tool_outcomes": (
                "_route_after_indexer_apply",
                {"prepare_step": "prepare_step", END: END},
            ),
        },
        static_edges=[("persist_tool_results", "apply_indexer_tool_outcomes")],
    )

    async def _execute_langgraph(self, payload: LoopGraphPayload) -> dict[str, object]:
        harness = self.deps.harness
        compiled = self._build_compiled_graph()
        config = checkpoint_config(harness, recursion_limit=512)
        values = await compiled.ainvoke(payload, config)
        graph_state = load_runtime_state(self, coerce_graph_values_payload(values))
        harness.state = graph_state.loop_state
        result = graph_state.final_result or harness._failure(
            "Indexer graph ended without a terminal result.",
            error_type="runtime",
        )
        return harness._finalize(result)

    async def _prepare_indexer_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await prepare_prompt_node(self, payload, prepare_indexer_prompt)

    async def _indexer_model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        return await model_call_node(
            self,
            payload,
            select_tools_fn=select_indexer_tools,
            model_call_fn=model_call,
        )

    async def _interpret_indexer_output_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        if graph_state.pending_tool_calls:
            return serialize_runtime_state(graph_state)
        graph_state.final_result = {
            "status": "stopped",
            "reason": "no_indexer_tool_calls",
            "assistant": graph_state.last_assistant_text,
        }
        return serialize_runtime_state(graph_state)

    async def _apply_indexer_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = load_runtime_state(self, payload)
        for record in graph_state.last_tool_results:
            if record.tool_name == "index_finalize" and record.result.success:
                graph_state.final_result = {
                    "status": "completed",
                    "index_manifest": record.result.output,
                }
                break
        return await apply_outcomes_node(self, serialize_runtime_state(graph_state), apply_chat_tool_outcomes)

    @staticmethod
    def _route_after_indexer_apply(payload: LoopGraphPayload) -> str:
        return route_if_final_else(payload, "prepare_step")
