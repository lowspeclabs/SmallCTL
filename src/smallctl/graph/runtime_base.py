from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .runtime_payloads import (
    LoopGraphPayload,
    build_runtime_payload,
    checkpoint_config,
    coerce_graph_values_payload,
    coerce_interrupt_payload,
    execute_streaming_graph,
    get_runtime_checkpointer,
    load_runtime_state,
    restore_runtime_state,
    route_if_final_else,
    route_if_final_else_pending_else,
    route_if_interrupt_else_final_else,
    route_if_interrupt_else_final_else_pending_else,
    serialize_runtime_state,
)
from .state import GraphRunState


@dataclass(frozen=True)
class RuntimeGraphSpec:
    """Configuration for building a runtime StateGraph."""

    node_map: dict[str, str]
    edge_map: dict[str, tuple[str, dict[str, str]]]
    static_edges: list[tuple[str, str]]
    entry_point: tuple[str, str] = (START, "initialize_run")


class RuntimeGraphBuilder:
    """Builds a compiled StateGraph from a runtime instance and spec."""

    def __init__(self, runtime: Any, spec: RuntimeGraphSpec) -> None:
        self.runtime = runtime
        self.spec = spec

    def build(self) -> Any:
        builder = StateGraph(LoopGraphPayload)

        for node_name, method_name in self.spec.node_map.items():
            fn = getattr(self.runtime, method_name)
            builder.add_node(node_name, fn)

        entry_source, entry_target = self.spec.entry_point
        builder.add_edge(entry_source, entry_target)

        for source, (router_name, targets) in self.spec.edge_map.items():
            router = getattr(self.runtime, router_name)
            # LangGraph routing via Command(goto=...) is stable here, while
            # chained add_conditional_edges() transitions can stall on this version.
            route_node_name = f"route__{source}"
            builder.add_node(route_node_name, self._make_route_node(router, targets))
            builder.add_edge(source, route_node_name)

        for source, target in self.spec.static_edges:
            builder.add_edge(source, target)

        return builder.compile(checkpointer=get_runtime_checkpointer(self.runtime.deps.harness))

    @staticmethod
    def _make_route_node(
        router: Callable[[LoopGraphPayload], str],
        targets: dict[str, str],
    ) -> Callable[[LoopGraphPayload], Awaitable[Command]]:
        async def route_node(payload: LoopGraphPayload) -> Command:
            next_step = router(payload)
            return Command(goto=targets[next_step])

        return route_node


class CompiledGraphRuntimeBase:
    _run_mode = "loop"
    _run_execution_message = "executing loop runtime"
    _empty_result_message = "Graph loop ended without a terminal result."
    _recursion_limit = 512

    def __init__(self, deps: Any) -> None:
        self.deps = deps

    @classmethod
    def from_harness(
        cls,
        harness: object,
        *,
        event_handler: object = None,
    ):
        from .deps import GraphRuntimeDeps

        return cls(
            GraphRuntimeDeps(
                harness=harness,
                event_handler=event_handler,
            ),
        )

    async def run(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        self._before_run(harness)
        harness._runlog(
            "runtime_execution",
            self._run_execution_message,
            execution_path=self._execution_path(),
        )
        return await self._run_langgraph(task)

    def _before_run(self, harness: Any) -> None:
        del harness

    async def _run_langgraph(self, task: str) -> dict[str, object]:
        harness = self.deps.harness
        payload = build_runtime_payload(harness, run_mode=self._run_mode, input_task=task)
        return await self._execute_langgraph(payload)

    async def _execute_langgraph(self, payload: LoopGraphPayload | Command) -> dict[str, object]:
        return await execute_streaming_graph(
            self,
            payload,
            build_graph=self._build_compiled_graph,
            empty_result_message=self._empty_result_message,
            recursion_limit=self._recursion_limit,
        )

    def _build_compiled_graph(self):
        return RuntimeGraphBuilder(self, self.GRAPH_SPEC).build()

    def _execution_path(self) -> str:
        return "compiled"

    def restore(self, *, thread_id: str | None = None) -> bool:
        return restore_runtime_state(self, thread_id=thread_id, recursion_limit=self._recursion_limit)


async def prepare_prompt_node(
    runtime: Any,
    payload: LoopGraphPayload,
    prepare_fn: Callable[[Any, Any], Awaitable[list[dict[str, Any]] | None]],
) -> LoopGraphPayload:
    graph_state = load_runtime_state(runtime, payload)
    messages = await prepare_fn(graph_state, runtime.deps)
    next_payload = serialize_runtime_state(graph_state)
    if messages is not None:
        graph_state.loop_state.scratchpad["_compiled_prompt_messages"] = messages
        next_payload = serialize_runtime_state(graph_state)
    return next_payload


async def model_call_node(
    runtime: Any,
    payload: LoopGraphPayload,
    *,
    select_tools_fn: Callable[[Any, Any], list[dict[str, Any]]],
    model_call_fn: Callable[[Any, Any], Awaitable[None]],
) -> LoopGraphPayload:
    graph_state = load_runtime_state(runtime, payload)
    messages = graph_state.loop_state.scratchpad.pop("_compiled_prompt_messages", [])
    tools = select_tools_fn(graph_state, runtime.deps)
    await model_call_fn(graph_state, runtime.deps, messages=messages, tools=tools)
    return serialize_runtime_state(graph_state)


async def interpret_node(
    runtime: Any,
    payload: LoopGraphPayload,
    interpret_fn: Callable[[Any, Any], Awaitable[None]],
) -> LoopGraphPayload:
    graph_state = load_runtime_state(runtime, payload)
    await interpret_fn(graph_state, runtime.deps)
    return serialize_runtime_state(graph_state)


async def apply_outcomes_node(
    runtime: Any,
    payload: LoopGraphPayload,
    apply_fn: Callable[[Any, Any], Awaitable[None]],
    *,
    clear_final_result_on_interrupt: bool = False,
) -> LoopGraphPayload:
    graph_state = load_runtime_state(runtime, payload)
    await apply_fn(graph_state, runtime.deps)
    if clear_final_result_on_interrupt and graph_state.interrupt_payload is not None:
        graph_state.final_result = None
    return serialize_runtime_state(graph_state)
