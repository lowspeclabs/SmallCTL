from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from .checkpoint import create_graph_checkpointer
from .deps import GraphRuntimeDeps
from .nodes import (
    apply_chat_tool_outcomes,
    apply_planning_tool_outcomes,
    apply_tool_outcomes,
    dispatch_tools,
    initialize_loop_run,
    initialize_planning_run,
    interpret_chat_output,
    interpret_planning_output,
    interpret_model_output,
    model_call,
    persist_tool_results,
    prepare_chat_prompt,
    prepare_planning_prompt,
    prepare_indexer_prompt,
    prepare_loop_step,
    prepare_prompt,
    resume_loop_run,
    resume_planning_run,
    select_chat_tools,
    select_indexer_tools,
    select_loop_tools,
    select_planning_tools,
)
from .routing import LoopRoute
from .state import GraphRunState, inflate_graph_state, serialize_graph_state
from ..state import json_safe_value


class LoopGraphPayload(TypedDict, total=False):
    loop_state: dict[str, Any]
    thread_id: str
    run_mode: str
    pending_tool_calls: list[dict[str, Any]]
    last_assistant_text: str
    last_thinking_text: str
    last_usage: dict[str, Any]
    last_tool_results: list[dict[str, Any]]
    final_result: dict[str, Any] | None
    interrupt_payload: dict[str, Any] | None
    error: dict[str, Any] | None
    input_task: str


LANGGRAPH_RECURSION_LIMIT = 512


class LoopGraphRuntime:
    def __init__(self, deps: GraphRuntimeDeps) -> None:
        self.deps = deps

    @classmethod
    def from_harness(
        cls,
        harness: object,
        *,
        event_handler: object = None,
    ) -> "LoopGraphRuntime":
        return cls(
            GraphRuntimeDeps(
                harness=harness,
                event_handler=event_handler,
            ),
        )

    async def run(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        harness._runlog(
            "runtime_execution",
            "executing loop runtime",
            execution_path=self._execution_path(),
        )
        return await self._run_langgraph(task)

    async def resume(self, human_input: str) -> dict[str, object]:
        return await self._resume_langgraph(human_input)

    async def _run_langgraph(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        if not harness.state.thread_id:
            harness.state.thread_id = harness.conversation_id
        payload = self._serialize_state(
            GraphRunState(
                loop_state=harness.state,
                thread_id=harness.state.thread_id,
                run_mode="loop",
            ),
            input_task=task,
        )
        return await self._execute_langgraph(payload)

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

    async def _execute_langgraph(self, payload: LoopGraphPayload | Command) -> dict[str, object]:
        harness = self.deps.harness
        compiled = self._build_compiled_loop_graph()
        config = self._checkpoint_config()
        interrupt_payload: dict[str, Any] | None = None
        async for chunk in compiled.astream(payload, config):
            if not isinstance(chunk, dict):
                continue
            interrupt_chunk = chunk.get("__interrupt__")
            interrupt_payload = _coerce_interrupt_payload(interrupt_chunk)
            if interrupt_payload is not None:
                break
        snapshot = compiled.get_state(config)
        values = _coerce_graph_values_payload(getattr(snapshot, "values", None))
        if values:
            graph_state = self._load_state(values)
            harness.state = graph_state.loop_state
        else:
            graph_state = GraphRunState(
                loop_state=harness.state,
                thread_id=harness.state.thread_id or harness.conversation_id,
                run_mode="loop",
            )
        if interrupt_payload is not None:
            result = {
                "status": "needs_human",
                "message": {
                    "status": "human_input_required",
                    "question": interrupt_payload.get("question", ""),
                },
                "interrupt": interrupt_payload,
            }
            return harness._finalize(result)
        result = graph_state.final_result or harness._failure(
            "Graph loop ended without a terminal result.",
            error_type="runtime",
        )
        return harness._finalize(result)

    def _build_compiled_loop_graph(self):
        builder = StateGraph(LoopGraphPayload)
        builder.add_node("initialize_run", self._initialize_run_node)
        builder.add_node("prepare_step", self._prepare_step_node)
        builder.add_node("prepare_prompt", self._prepare_prompt_node)
        builder.add_node("model_call", self._model_call_node)
        builder.add_node("interpret_model_output", self._interpret_model_output_node)
        builder.add_node("dispatch_tools", self._dispatch_tools_node)
        builder.add_node("persist_tool_results", self._persist_tool_results_node)
        builder.add_node("apply_tool_outcomes", self._apply_tool_outcomes_node)
        builder.add_node("interrupt_for_human", self._interrupt_for_human_node)
        builder.add_edge(START, "initialize_run")
        builder.add_conditional_edges(
            "initialize_run",
            self._route_after_initialize,
            {"prepare_step": "prepare_step", END: END},
        )
        builder.add_conditional_edges(
            "prepare_step",
            self._route_after_prepare_step,
            {
                "prepare_prompt": "prepare_prompt",
                "dispatch_tools": "dispatch_tools",
                END: END,
            },
        )
        builder.add_conditional_edges(
            "prepare_prompt",
            self._route_after_prepare_prompt,
            {"model_call": "model_call", END: END},
        )
        builder.add_conditional_edges(
            "model_call",
            self._route_after_model_call,
            {"interpret_model_output": "interpret_model_output", END: END},
        )
        builder.add_conditional_edges(
            "interpret_model_output",
            self._route_after_interpret,
            {"dispatch_tools": "dispatch_tools", "prepare_step": "prepare_step", END: END},
        )
        builder.add_conditional_edges(
            "dispatch_tools",
            self._route_after_dispatch,
            {"persist_tool_results": "persist_tool_results", END: END},
        )
        builder.add_edge("persist_tool_results", "apply_tool_outcomes")
        builder.add_conditional_edges(
            "apply_tool_outcomes",
            self._route_after_apply,
            {"prepare_step": "prepare_step", "interrupt_for_human": "interrupt_for_human", END: END},
        )
        builder.add_edge("interrupt_for_human", "prepare_step")
        return builder.compile(checkpointer=self._get_checkpointer())

    def _get_checkpointer(self):
        harness = self.deps.harness
        saver = getattr(harness, "_graph_checkpointer", None)
        if saver is None:
            saver = create_graph_checkpointer(
                backend=getattr(harness, "graph_checkpointer", "memory"),
                path=getattr(harness, "graph_checkpoint_path", None),
            )
            setattr(harness, "_graph_checkpointer", saver)
        return saver

    def _checkpoint_config(self, *, thread_id: str | None = None) -> dict[str, Any]:
        harness = self.deps.harness
        resolved_thread_id = thread_id or harness.state.thread_id or harness.conversation_id
        return {
            "configurable": {
                "thread_id": resolved_thread_id,
                "checkpoint_ns": "",
            },
            "recursion_limit": LANGGRAPH_RECURSION_LIMIT,
        }

    def _load_state(self, payload: dict[str, Any]) -> GraphRunState:
        graph_state = inflate_graph_state(payload)
        self.deps.harness.state = graph_state.loop_state
        return graph_state

    @staticmethod
    def _serialize_state(
        graph_state: GraphRunState,
        *,
        input_task: str | None = None,
    ) -> LoopGraphPayload:
        payload: LoopGraphPayload = serialize_graph_state(graph_state)
        if input_task is not None:
            payload["input_task"] = input_task
        return payload

    def _execution_path(self) -> str:
        return "compiled"

    def restore(self, *, thread_id: str | None = None) -> bool:
        harness = self.deps.harness
        saver = self._get_checkpointer()
        candidate_thread_ids: list[str] = []
        if thread_id:
            candidate_thread_ids.append(thread_id)
        elif harness.state.thread_id:
            candidate_thread_ids.append(harness.state.thread_id)
        if hasattr(saver, "latest_thread_id"):
            latest_thread_id = saver.latest_thread_id()
            if latest_thread_id and latest_thread_id not in candidate_thread_ids:
                candidate_thread_ids.append(latest_thread_id)
        for candidate_thread_id in candidate_thread_ids:
            checkpoint_tuple = saver.get_tuple(
                self._checkpoint_config(thread_id=candidate_thread_id)
            )
            values = (
                _coerce_graph_values_payload(checkpoint_tuple.checkpoint.get("channel_values"))
                if checkpoint_tuple
                else {}
            )
            if not values:
                continue
            graph_state = self._load_state(values)
            harness.state = graph_state.loop_state
            if not harness.state.thread_id:
                harness.state.thread_id = candidate_thread_id
            return True
        return False

    async def _initialize_run_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await initialize_loop_run(graph_state, self.deps, task=str(payload.get("input_task", "")))
        return self._serialize_state(graph_state)

    async def _prepare_step_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await prepare_loop_step(graph_state, self.deps)
        return self._serialize_state(graph_state)

    async def _prepare_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        messages = await prepare_prompt(graph_state, self.deps)
        next_payload = self._serialize_state(graph_state)
        if messages is not None:
            graph_state.loop_state.scratchpad["_compiled_prompt_messages"] = messages
            next_payload = self._serialize_state(graph_state)
        return next_payload

    async def _model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        messages = graph_state.loop_state.scratchpad.pop("_compiled_prompt_messages", [])
        tools = select_loop_tools(graph_state, self.deps)
        await model_call(graph_state, self.deps, messages=messages, tools=tools)
        return self._serialize_state(graph_state)

    async def _interpret_model_output_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await interpret_model_output(graph_state, self.deps)
        return self._serialize_state(graph_state)

    async def _dispatch_tools_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await dispatch_tools(graph_state, self.deps)
        return self._serialize_state(graph_state)

    async def _persist_tool_results_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await persist_tool_results(graph_state, self.deps)
        return self._serialize_state(graph_state)

    async def _apply_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await apply_tool_outcomes(graph_state, self.deps)
        if graph_state.interrupt_payload is not None:
            graph_state.final_result = None
        return self._serialize_state(graph_state)

    async def _interrupt_for_human_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        payload_value = graph_state.interrupt_payload or graph_state.loop_state.pending_interrupt or {}
        human_input = interrupt(payload_value)
        await resume_loop_run(graph_state, self.deps, human_input=str(human_input))
        graph_state.interrupt_payload = None
        return self._serialize_state(graph_state)

    @staticmethod
    def _route_after_initialize(payload: LoopGraphPayload) -> str:
        return END if payload.get("final_result") is not None else "prepare_step"

    @staticmethod
    def _route_after_prepare_step(payload: LoopGraphPayload) -> str:
        if payload.get("final_result") is not None:
            return END
        if payload.get("pending_tool_calls"):
            return "dispatch_tools"
        return "prepare_prompt"

    @staticmethod
    def _route_after_prepare_prompt(payload: LoopGraphPayload) -> str:
        return END if payload.get("final_result") is not None else "model_call"

    @staticmethod
    def _route_after_model_call(payload: LoopGraphPayload) -> str:
        return END if payload.get("final_result") is not None else "interpret_model_output"

    @staticmethod
    def _route_after_interpret(payload: LoopGraphPayload) -> str:
        if payload.get("final_result") is not None:
            return END
        if payload.get("pending_tool_calls"):
            return "dispatch_tools"
        return "prepare_step"

    @staticmethod
    def _route_after_dispatch(payload: LoopGraphPayload) -> str:
        return END if payload.get("final_result") is not None else "persist_tool_results"

    @staticmethod
    def _route_after_apply(payload: LoopGraphPayload) -> str:
        if payload.get("interrupt_payload") is not None:
            return "interrupt_for_human"
        if payload.get("final_result") is not None:
            return END
        return "prepare_step"


class ChatGraphRuntime(LoopGraphRuntime):

    async def run(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        harness._runlog(
            "runtime_execution",
            "executing chat runtime",
            execution_path=self._execution_path(),
        )
        if not harness.state.thread_id:
            harness.state.thread_id = harness.conversation_id
            
        payload = self._serialize_state(
            GraphRunState(
                loop_state=harness.state,
                thread_id=harness.state.thread_id,
                run_mode="chat",
            ),
            input_task=task,
        )
        return await self._execute_langgraph(payload)

    async def _execute_langgraph(self, payload: LoopGraphPayload | Command) -> dict[str, Any]:
        harness = self.deps.harness
        compiled = self._build_compiled_chat_graph()
        config = self._checkpoint_config()
        interrupt_payload: dict[str, Any] | None = None
        async for chunk in compiled.astream(payload, config):
            if not isinstance(chunk, dict):
                continue
            interrupt_chunk = chunk.get("__interrupt__")
            interrupt_payload = _coerce_interrupt_payload(interrupt_chunk)
            if interrupt_payload is not None:
                break
        
        snapshot = compiled.get_state(config)
        values = _coerce_graph_values_payload(getattr(snapshot, "values", None))
        if values:
            graph_state = self._load_state(values)
            harness.state = graph_state.loop_state
        else:
            graph_state = GraphRunState(
                loop_state=harness.state,
                thread_id=harness.state.thread_id or harness.conversation_id,
                run_mode="chat",
            )
        
        if interrupt_payload is not None:
            result = {
                "status": "needs_human",
                "message": {
                    "status": "human_input_required",
                    "question": interrupt_payload.get("question", ""),
                },
                "interrupt": interrupt_payload,
            }
            return harness._finalize(result)
            
        result = graph_state.final_result or harness._failure(
            "Chat graph ended without a terminal result.",
            error_type="runtime",
        )
        return harness._finalize(result)

    def _build_compiled_chat_graph(self):
        builder = StateGraph(LoopGraphPayload)
        builder.add_node("initialize_run", self._initialize_run_node)
        builder.add_node("prepare_step", self._prepare_step_node)
        builder.add_node("prepare_chat_prompt", self._prepare_chat_prompt_node)
        builder.add_node("model_call", self._chat_model_call_node)
        builder.add_node("interpret_chat_output", self._interpret_chat_output_node)
        builder.add_node("dispatch_tools", self._dispatch_tools_node)
        builder.add_node("persist_tool_results", self._persist_tool_results_node)
        builder.add_node("apply_chat_tool_outcomes", self._apply_chat_tool_outcomes_node)
        builder.add_node("interrupt_for_human", self._interrupt_for_human_node)
        builder.add_edge(START, "initialize_run")
        builder.add_conditional_edges(
            "initialize_run",
            self._route_after_initialize,
            {"prepare_step": "prepare_step", END: END},
        )
        builder.add_conditional_edges(
            "prepare_step",
            self._route_after_prepare_step,
            {
                "prepare_prompt": "prepare_chat_prompt",
                "dispatch_tools": "dispatch_tools",
                END: END,
            },
        )
        builder.add_conditional_edges(
            "prepare_chat_prompt",
            self._route_after_prepare_prompt,
            {"model_call": "model_call", END: END},
        )
        builder.add_conditional_edges(
            "model_call",
            self._route_after_model_call,
            {"interpret_model_output": "interpret_chat_output", END: END},
        )
        builder.add_conditional_edges(
            "interpret_chat_output",
            self._route_after_interpret,
            {"dispatch_tools": "dispatch_tools", END: END},
        )
        builder.add_conditional_edges(
            "dispatch_tools",
            self._route_after_dispatch,
            {"persist_tool_results": "persist_tool_results", END: END},
        )
        builder.add_edge("persist_tool_results", "apply_chat_tool_outcomes")
        builder.add_conditional_edges(
            "apply_chat_tool_outcomes",
            self._route_after_chat_apply,
            {
                "prepare_step": "prepare_step",
                "interrupt_for_human": "interrupt_for_human",
                END: END
            },
        )
        return builder.compile(checkpointer=self._get_checkpointer())

    async def _prepare_chat_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        messages = await prepare_chat_prompt(graph_state, self.deps)
        next_payload = self._serialize_state(graph_state)
        if messages is not None:
            graph_state.loop_state.scratchpad["_compiled_prompt_messages"] = messages
            next_payload = self._serialize_state(graph_state)
        return next_payload

    async def _chat_model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        messages = graph_state.loop_state.scratchpad.pop("_compiled_prompt_messages", [])
        tools = select_chat_tools(graph_state, self.deps)
        await model_call(graph_state, self.deps, messages=messages, tools=tools)
        return self._serialize_state(graph_state)

    async def _interpret_chat_output_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await interpret_chat_output(graph_state, self.deps)
        return self._serialize_state(graph_state)

    async def _apply_chat_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await apply_chat_tool_outcomes(graph_state, self.deps)
        return self._serialize_state(graph_state)

    @staticmethod
    def _route_after_chat_apply(payload: LoopGraphPayload) -> str:
        if payload.get("interrupt_payload") is not None:
            return "interrupt_for_human"
        return END if payload.get("final_result") is not None else "prepare_step"


class PlanningGraphRuntime(LoopGraphRuntime):

    async def run(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        harness.state.planning_mode_enabled = True
        harness._runlog(
            "runtime_execution",
            "executing planning runtime",
            execution_path=self._execution_path(),
        )
        if not harness.state.thread_id:
            harness.state.thread_id = harness.conversation_id
        payload = self._serialize_state(
            GraphRunState(
                loop_state=harness.state,
                thread_id=harness.state.thread_id,
                run_mode="planning",
            ),
            input_task=task,
        )
        return await self._execute_planning_langgraph(payload)

    async def resume(self, human_input: str) -> dict[str, object]:
        harness = self.deps.harness
        pending = harness.state.pending_interrupt or {}
        if isinstance(pending, dict) and pending.get("kind") == "plan_execute_approval":
            normalized = human_input.strip().lower()
            if normalized in {"yes", "y", "approve", "approved", "execute", "go ahead", "run it"}:
                graph_state = GraphRunState(
                    loop_state=harness.state,
                    thread_id=harness.state.thread_id or harness.conversation_id,
                    run_mode="planning",
                )
                await resume_planning_run(graph_state, self.deps, human_input=human_input)
                plan = harness.state.active_plan or harness.state.draft_plan
                if plan is not None:
                    harness.state.planning_mode_enabled = False
                    return await LoopGraphRuntime.from_harness(
                        harness,
                        event_handler=self.deps.event_handler,
                    ).run(plan.goal or harness.state.run_brief.original_task or human_input)
        return await self._resume_langgraph(human_input)

    async def _execute_planning_langgraph(self, payload: LoopGraphPayload | Command) -> dict[str, object]:
        harness = self.deps.harness
        compiled = self._build_compiled_planning_graph()
        config = self._checkpoint_config()
        interrupt_payload: dict[str, Any] | None = None
        async for chunk in compiled.astream(payload, config):
            if not isinstance(chunk, dict):
                continue
            interrupt_chunk = chunk.get("__interrupt__")
            interrupt_payload = _coerce_interrupt_payload(interrupt_chunk)
            if interrupt_payload is not None:
                break
        snapshot = compiled.get_state(config)
        values = _coerce_graph_values_payload(getattr(snapshot, "values", None))
        if values:
            graph_state = self._load_state(values)
            harness.state = graph_state.loop_state
        else:
            graph_state = GraphRunState(
                loop_state=harness.state,
                thread_id=harness.state.thread_id or harness.conversation_id,
                run_mode="planning",
            )
        if interrupt_payload is not None:
            result = {
                "status": "needs_human",
                "message": {
                    "status": "human_input_required",
                    "question": interrupt_payload.get("question", ""),
                },
                "interrupt": interrupt_payload,
            }
            return harness._finalize(result)
        result = graph_state.final_result or harness._failure(
            "Planning graph ended without a terminal result.",
            error_type="runtime",
        )
        return harness._finalize(result)

    def _build_compiled_planning_graph(self):
        builder = StateGraph(LoopGraphPayload)
        builder.add_node("initialize_run", self._initialize_run_node)
        builder.add_node("prepare_step", self._prepare_step_node)
        builder.add_node("prepare_prompt", self._prepare_planning_prompt_node)
        builder.add_node("model_call", self._planning_model_call_node)
        builder.add_node("interpret_model_output", self._interpret_planning_output_node)
        builder.add_node("dispatch_tools", self._dispatch_tools_node)
        builder.add_node("persist_tool_results", self._persist_tool_results_node)
        builder.add_node("apply_tool_outcomes", self._apply_planning_tool_outcomes_node)
        builder.add_node("interrupt_for_human", self._interrupt_for_human_node)
        builder.add_edge(START, "initialize_run")
        builder.add_conditional_edges(
            "initialize_run",
            self._route_after_initialize,
            {"prepare_step": "prepare_step", END: END},
        )
        builder.add_conditional_edges(
            "prepare_step",
            self._route_after_prepare_step,
            {
                "prepare_prompt": "prepare_prompt",
                "dispatch_tools": "dispatch_tools",
                END: END,
            },
        )
        builder.add_conditional_edges(
            "prepare_prompt",
            self._route_after_prepare_prompt,
            {"model_call": "model_call", END: END},
        )
        builder.add_conditional_edges(
            "model_call",
            self._route_after_model_call,
            {"interpret_model_output": "interpret_model_output", END: END},
        )
        builder.add_conditional_edges(
            "interpret_model_output",
            self._route_after_planning_interpret,
            {"dispatch_tools": "dispatch_tools", "prepare_step": "prepare_step", END: END},
        )
        builder.add_conditional_edges(
            "dispatch_tools",
            self._route_after_dispatch,
            {"persist_tool_results": "persist_tool_results", END: END},
        )
        builder.add_edge("persist_tool_results", "apply_tool_outcomes")
        builder.add_conditional_edges(
            "apply_tool_outcomes",
            self._route_after_planning_apply,
            {"prepare_step": "prepare_step", "interrupt_for_human": "interrupt_for_human", END: END},
        )
        builder.add_edge("interrupt_for_human", "prepare_step")
        return builder.compile(checkpointer=self._get_checkpointer())

    async def _prepare_planning_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        messages = await prepare_planning_prompt(graph_state, self.deps)
        next_payload = self._serialize_state(graph_state)
        if messages is not None:
            graph_state.loop_state.scratchpad["_compiled_prompt_messages"] = messages
            next_payload = self._serialize_state(graph_state)
        return next_payload

    async def _planning_model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        messages = graph_state.loop_state.scratchpad.pop("_compiled_prompt_messages", [])
        tools = select_planning_tools(graph_state, self.deps)
        await model_call(graph_state, self.deps, messages=messages, tools=tools)
        return self._serialize_state(graph_state)

    async def _interpret_planning_output_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await interpret_planning_output(graph_state, self.deps)
        return self._serialize_state(graph_state)

    async def _apply_planning_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        await apply_planning_tool_outcomes(graph_state, self.deps)
        if graph_state.interrupt_payload is not None:
            graph_state.final_result = None
        return self._serialize_state(graph_state)

    @staticmethod
    def _route_after_planning_interpret(payload: LoopGraphPayload) -> str:
        if payload.get("final_result") is not None:
            return END
        if payload.get("pending_tool_calls"):
            return "dispatch_tools"
        return "prepare_step"

    @staticmethod
    def _route_after_planning_apply(payload: LoopGraphPayload) -> str:
        if payload.get("interrupt_payload") is not None:
            return "interrupt_for_human"
        if payload.get("final_result") is not None:
            return END
        return "prepare_step"


class IndexerGraphRuntime(LoopGraphRuntime):
    """A rigid code indexing runtime optimized for SLM traversal and extraction."""

    async def run(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        harness._runlog(
            "runtime_execution",
            "executing indexer runtime",
            execution_path=self._execution_path(),
        )
        return await self._run_langgraph(task)

    async def _run_langgraph(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        if not harness.state.thread_id:
            harness.state.thread_id = harness.conversation_id
        payload = self._serialize_state(
            GraphRunState(
                loop_state=harness.state,
                thread_id=harness.state.thread_id,
                run_mode="indexer",
            ),
            input_task=task,
        )
        return await self._execute_langgraph(payload)

    async def _execute_langgraph(self, payload: LoopGraphPayload) -> dict[str, object]:
        harness = self.deps.harness
        compiled = self._build_compiled_indexer_graph()
        config = self._checkpoint_config()
        values = await compiled.ainvoke(payload, config)
        graph_state = self._load_state(_coerce_graph_values_payload(values))
        harness.state = graph_state.loop_state
        result = graph_state.final_result or harness._failure(
            "Indexer graph ended without a terminal result.",
            error_type="runtime",
        )
        return harness._finalize(result)

    def _build_compiled_indexer_graph(self):
        builder = StateGraph(LoopGraphPayload)
        builder.add_node("initialize_run", self._initialize_run_node)
        builder.add_node("prepare_step", self._prepare_step_node)
        builder.add_node("prepare_indexer_prompt", self._prepare_indexer_prompt_node)
        builder.add_node("model_call", self._indexer_model_call_node)
        builder.add_node("interpret_indexer_output", self._interpret_indexer_output_node)
        builder.add_node("dispatch_tools", self._dispatch_tools_node)
        builder.add_node("persist_tool_results", self._persist_tool_results_node)
        builder.add_node("apply_indexer_tool_outcomes", self._apply_indexer_tool_outcomes_node)
        
        builder.add_edge(START, "initialize_run")
        builder.add_conditional_edges(
            "initialize_run",
            self._route_after_initialize,
            {"prepare_step": "prepare_step", END: END},
        )
        builder.add_conditional_edges(
            "prepare_step",
            self._route_after_prepare_step,
            {
                "prepare_prompt": "prepare_indexer_prompt",
                "dispatch_tools": "dispatch_tools",
                END: END,
            },
        )
        builder.add_conditional_edges(
            "prepare_indexer_prompt",
            self._route_after_prepare_prompt,
            {"model_call": "model_call", END: END},
        )
        builder.add_conditional_edges(
            "model_call",
            self._route_after_model_call,
            {"interpret_model_output": "interpret_indexer_output", END: END},
        )
        builder.add_conditional_edges(
            "interpret_indexer_output",
            self._route_after_interpret,
            {"dispatch_tools": "dispatch_tools", END: END},
        )
        builder.add_conditional_edges(
            "dispatch_tools",
            self._route_after_dispatch,
            {"persist_tool_results": "persist_tool_results", END: END},
        )
        builder.add_edge("persist_tool_results", "apply_indexer_tool_outcomes")
        builder.add_conditional_edges(
            "apply_indexer_tool_outcomes",
            self._route_after_indexer_apply,
            {"prepare_step": "prepare_step", END: END},
        )
        return builder.compile(checkpointer=self._get_checkpointer())

    async def _prepare_indexer_prompt_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        messages = await prepare_indexer_prompt(graph_state, self.deps)
        next_payload = self._serialize_state(graph_state)
        if messages is not None:
            graph_state.loop_state.scratchpad["_compiled_prompt_messages"] = messages
            next_payload = self._serialize_state(graph_state)
        return next_payload

    async def _indexer_model_call_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        messages = graph_state.loop_state.scratchpad.pop("_compiled_prompt_messages", [])
        tools = select_indexer_tools(graph_state, self.deps)
        await model_call(graph_state, self.deps, messages=messages, tools=tools)
        return self._serialize_state(graph_state)

    async def _interpret_indexer_output_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        # Indexer-specific interpretation: strictly follow tool calls
        if graph_state.pending_tool_calls:
             return self._serialize_state(graph_state)
        
        # If no tools, it might be a completion. Since indexers don't chat, 
        # we assume it's stuck or done if it didn't call finalize.
        graph_state.final_result = {
            "status": "stopped",
            "reason": "no_indexer_tool_calls",
            "assistant": graph_state.last_assistant_text,
        }
        return self._serialize_state(graph_state)

    async def _apply_indexer_tool_outcomes_node(self, payload: LoopGraphPayload) -> LoopGraphPayload:
        graph_state = self._load_state(payload)
        # Check if index_finalize was called
        for record in graph_state.last_tool_results:
            if record.tool_name == "index_finalize" and record.result.success:
                graph_state.final_result = {
                    "status": "completed",
                    "index_manifest": record.result.output,
                }
                break
        
        await apply_chat_tool_outcomes(graph_state, self.deps)
        return self._serialize_state(graph_state)

    @staticmethod
    def _route_after_indexer_apply(payload: LoopGraphPayload) -> str:
        return END if payload.get("final_result") is not None else "prepare_step"


class AutoGraphRuntime:
    def __init__(self, deps: GraphRuntimeDeps) -> None:
        self.deps = deps

    @classmethod
    def from_harness(
        cls,
        harness: object,
        *,
        event_handler: object = None,
    ) -> "AutoGraphRuntime":
        return cls(
            GraphRuntimeDeps(
                harness=harness,
                event_handler=event_handler,
            ),
        )

    async def run(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        if harness.has_pending_interrupt():
            pending = harness.get_pending_interrupt() or {}
            interrupt_kind = str(pending.get("kind") or "ask_human")
            harness._runlog(
                "runtime_route",
                "routing task to interrupt resume",
                interrupt_kind=interrupt_kind,
                execution_path=self._execution_path(),
            )
            if interrupt_kind == "plan_execute_approval":
                return await PlanningGraphRuntime.from_harness(
                    harness,
                    event_handler=self.deps.event_handler,
                ).resume(task)
            return await LoopGraphRuntime.from_harness(
                harness,
                event_handler=self.deps.event_handler,
            ).resume(task)
        mode = await harness.decide_run_mode(task)
        harness._runlog(
            "runtime_route",
            "routing task to runtime",
            mode=mode,
            execution_path=self._execution_path(),
        )
        if mode == "planning":
            return await PlanningGraphRuntime.from_harness(
                harness,
                event_handler=self.deps.event_handler,
            ).run(task)
        if mode == "chat":
            return await ChatGraphRuntime.from_harness(
                harness,
                event_handler=self.deps.event_handler,
            ).run(task)
        return await LoopGraphRuntime.from_harness(
            harness,
            event_handler=self.deps.event_handler,
        ).run(task)

    def _execution_path(self) -> str:
        return "compiled"


def _coerce_graph_values_payload(value: Any) -> dict[str, Any]:
    normalized = json_safe_value(value or {})
    return normalized if isinstance(normalized, dict) else {}


def _coerce_interrupt_payload(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    if isinstance(value, dict):
        normalized = json_safe_value(value)
        return normalized if isinstance(normalized, dict) else None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        first = value[0]
        payload = getattr(first, "value", first)
    else:
        payload = value
    normalized = json_safe_value(payload)
    if isinstance(normalized, dict):
        return normalized
    return {"kind": "ask_human", "question": str(payload or "")}
