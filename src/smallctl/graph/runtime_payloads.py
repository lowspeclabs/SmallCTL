from __future__ import annotations

import asyncio
from typing import Any

from langgraph.types import Command
from typing_extensions import TypedDict

from .checkpoint import create_graph_checkpointer
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
    latency_metrics: dict[str, Any]
    input_task: str


def coerce_graph_values_payload(value: Any) -> dict[str, Any]:
    normalized = json_safe_value(value or {})
    return normalized if isinstance(normalized, dict) else {}


def coerce_interrupt_payload(value: Any) -> dict[str, Any] | None:
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


def route_if_final_else(payload: LoopGraphPayload, next_step: str) -> str:
    from langgraph.graph import END

    return END if payload.get("final_result") is not None else next_step


def route_if_final_else_pending_else(
    payload: LoopGraphPayload,
    *,
    pending_step: str,
    fallback_step: str,
) -> str:
    from langgraph.graph import END

    if payload.get("final_result") is not None:
        return END
    if payload.get("pending_tool_calls"):
        return pending_step
    return fallback_step


def route_if_interrupt_else_final_else_pending_else(
    payload: LoopGraphPayload,
    *,
    interrupt_step: str,
    pending_step: str,
    fallback_step: str,
) -> str:
    from langgraph.graph import END

    if payload.get("interrupt_payload") is not None:
        return interrupt_step
    if payload.get("final_result") is not None:
        return END
    if payload.get("pending_tool_calls"):
        return pending_step
    return fallback_step


def route_if_interrupt_else_final_else(
    payload: LoopGraphPayload,
    *,
    interrupt_step: str,
    fallback_step: str,
) -> str:
    from langgraph.graph import END

    if payload.get("interrupt_payload") is not None:
        return interrupt_step
    if payload.get("final_result") is not None:
        return END
    return fallback_step


def get_runtime_checkpointer(harness: Any):
    saver = getattr(harness, "_graph_checkpointer", None)
    if saver is None:
        saver = create_graph_checkpointer(
            backend=getattr(harness, "graph_checkpointer", "memory"),
            path=getattr(harness, "graph_checkpoint_path", None),
        )
        setattr(harness, "_graph_checkpointer", saver)
    return saver


def checkpoint_config(harness: Any, *, recursion_limit: int, thread_id: str | None = None) -> dict[str, Any]:
    resolved_thread_id = thread_id or harness.state.thread_id or harness.conversation_id
    return {
        "configurable": {
            "thread_id": resolved_thread_id,
            "checkpoint_ns": "",
        },
        "recursion_limit": recursion_limit,
    }


def serialize_runtime_state(graph_state: GraphRunState, *, input_task: str | None = None) -> LoopGraphPayload:
    payload: LoopGraphPayload = serialize_graph_state(graph_state)
    if input_task is not None:
        payload["input_task"] = input_task
    return payload


def build_runtime_payload(
    harness: Any,
    *,
    run_mode: str,
    input_task: str | None = None,
) -> LoopGraphPayload:
    if not harness.state.thread_id:
        harness.state.thread_id = harness.conversation_id
    return serialize_runtime_state(
        GraphRunState(
            loop_state=harness.state,
            thread_id=harness.state.thread_id,
            run_mode=run_mode,
        ),
        input_task=input_task,
    )


def load_runtime_state(runtime: Any, payload: dict[str, Any]) -> GraphRunState:
    graph_state = inflate_graph_state(payload)
    runtime.deps.harness.state = graph_state.loop_state
    return graph_state


async def execute_streaming_graph(
    runtime: Any,
    payload: LoopGraphPayload | Command,
    *,
    build_graph,
    empty_result_message: str,
    recursion_limit: int,
) -> dict[str, Any]:
    harness = runtime.deps.harness
    compiled = build_graph()
    config = checkpoint_config(harness, recursion_limit=recursion_limit)
    interrupt_payload: dict[str, Any] | None = None
    try:
        async for chunk in compiled.astream(payload, config):
            if not isinstance(chunk, dict):
                continue
            interrupt_chunk = chunk.get("__interrupt__")
            interrupt_payload = coerce_interrupt_payload(interrupt_chunk)
            if interrupt_payload is not None:
                break
    except asyncio.CancelledError:
        harness._finalize(
            {
                "status": "cancelled",
                "reason": str(getattr(harness, "_pending_task_shutdown_reason", "") or "cancelled"),
            }
        )
        raise
    snapshot = compiled.get_state(config)
    values = coerce_graph_values_payload(getattr(snapshot, "values", None))
    if values:
        graph_state = load_runtime_state(runtime, values)
    else:
        graph_state = GraphRunState(
            loop_state=harness.state,
            thread_id=harness.state.thread_id or harness.conversation_id,
            run_mode=getattr(runtime, "_run_mode", "loop"),
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
        metrics = graph_state.latency_metrics
        if isinstance(metrics, dict) and metrics:
            result["latency_metrics"] = metrics
        return harness._finalize(result)
    result = graph_state.final_result or harness._failure(
        empty_result_message,
        error_type="runtime",
    )
    metrics = graph_state.latency_metrics
    if isinstance(metrics, dict) and metrics:
        result = dict(result)
        result["latency_metrics"] = metrics
    return harness._finalize(result)


def restore_runtime_state(runtime: Any, *, thread_id: str | None = None, recursion_limit: int) -> bool:
    harness = runtime.deps.harness
    saver = get_runtime_checkpointer(harness)
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
            checkpoint_config(harness, recursion_limit=recursion_limit, thread_id=candidate_thread_id)
        )
        values = (
            coerce_graph_values_payload(checkpoint_tuple.checkpoint.get("channel_values"))
            if checkpoint_tuple
            else {}
        )
        if not values:
            continue
        graph_state = load_runtime_state(runtime, values)
        harness.state = graph_state.loop_state
        if not harness.state.thread_id:
            harness.state.thread_id = candidate_thread_id
        return True
    return False
