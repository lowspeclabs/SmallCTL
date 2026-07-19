from __future__ import annotations

import asyncio
import contextlib
import inspect
from datetime import datetime, timezone
import time
from typing import Any

from langgraph.errors import GraphBubbleUp, NodeTimeoutError
from langgraph.types import Command
from typing_extensions import TypedDict

from .checkpoint import create_graph_checkpointer
from .state import GraphRunState, inflate_graph_state, serialize_graph_state
from ..logging_utils import runlog as _runlog
from ..models.events import UIEvent, UIEventType
from ..redaction import redact_sensitive_text
from ..state import json_safe_value


DEFAULT_GRAPH_IDLE_WATCHDOG_SEC = 300.0


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


def serialize_runtime_state(graph_state: GraphRunState, *, input_task: str | None = None, artifact_store: Any = None) -> LoopGraphPayload:
    payload: LoopGraphPayload = serialize_graph_state(graph_state, artifact_store=artifact_store)
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
        artifact_store=getattr(harness, "artifact_store", None),
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
    _touch_graph_activity(harness)
    watchdog_task = _start_graph_idle_watchdog(runtime)
    from .runtime_base import begin_graph_retry_attempts, end_graph_retry_attempts

    retry_attempts_token = begin_graph_retry_attempts()
    stream_events_enabled = _config_value(harness, "langgraph_stream_events_enabled", False)
    try:
        if stream_events_enabled:
            async for chunk in compiled.astream(
                payload, config, stream_mode=["updates", "tasks", "checkpoints"]
            ):
                _touch_graph_activity(harness)
                if not isinstance(chunk, tuple) or len(chunk) != 2:
                    continue
                mode, lg_payload = chunk
                event = _normalize_langgraph_stream_event(mode, lg_payload, harness=harness)
                if event is not None:
                    await _forward_langgraph_event(runtime, event)
                if isinstance(lg_payload, dict) and _contains_interrupt(lg_payload):
                    interrupt_payload = coerce_interrupt_payload(
                        _extract_interrupt_from_updates_payload(lg_payload)
                    )
                    if interrupt_payload is not None:
                        break
        else:
            async for chunk in compiled.astream(payload, config):
                _touch_graph_activity(harness)
                if not isinstance(chunk, dict):
                    continue
                interrupt_chunk = chunk.get("__interrupt__")
                interrupt_payload = coerce_interrupt_payload(interrupt_chunk)
                if interrupt_payload is not None:
                    break
    except (NodeTimeoutError, TimeoutError) as exc:
        if isinstance(exc, NodeTimeoutError):
            from .runtime_base import GraphNodeTimeoutError, graph_node_timeout_sec

            timeout_sec = graph_node_timeout_sec(harness, exc.node)
            exc = GraphNodeTimeoutError(exc.node, timeout_sec or 0.0)
        node_name = str(getattr(exc, "node_name", "") or "")
        timeout_sec = getattr(exc, "timeout_sec", None)
        elapsed = None
        node_started = getattr(harness, "_active_graph_node_started_monotonic", None)
        if isinstance(node_started, (int, float)):
            elapsed = round(time.monotonic() - float(node_started), 3)
        _runlog(
            harness,
            "runtime_timeout" if not node_name else f"{node_name}_timeout",
            "graph node timed out",
            node=node_name,
            timeout_sec=timeout_sec,
            elapsed_sec=elapsed,
        )
        return harness._finalize(
            harness._failure(
                str(exc),
                error_type="graph_node_timeout" if node_name else "runtime_timeout",
                details={
                    "node": node_name,
                    "timeout_sec": timeout_sec,
                },
            )
        )
    except asyncio.CancelledError:
        harness._finalize(
            {
                "status": "cancelled",
                "reason": str(getattr(harness, "_pending_task_shutdown_reason", "") or "cancelled"),
            }
        )
        raise
    except GraphBubbleUp:
        # LangGraph bubble-up signals (e.g. GraphInterrupt) must propagate
        # unchanged; they are control flow, not unexpected errors.
        raise
    except Exception as exc:
        # Final outer boundary: unexpected graph exceptions (node bugs, router
        # surprises, LangGraph GraphRecursionError, ...) must still finalize
        # exactly once so the summary/checkpoint/terminal path runs.
        error_summary = _sanitize_error_summary(exc)
        _runlog(
            harness,
            "runtime_graph_error",
            "unexpected graph exception",
            exception_type=type(exc).__name__,
            error=error_summary,
        )
        return harness._finalize(
            harness._failure(
                error_summary,
                error_type="runtime_graph_error",
                details={
                    "exception_type": type(exc).__name__,
                },
            )
        )
    finally:
        if watchdog_task is not None:
            watchdog_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog_task
        end_graph_retry_attempts(retry_attempts_token)
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


def _config_value(harness: Any, name: str, default: Any) -> Any:
    config = getattr(harness, "config", None)
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return getattr(harness, name, default)


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _graph_idle_watchdog_sec(harness: Any) -> float | None:
    return _positive_float(
        _config_value(harness, "graph_idle_watchdog_sec", DEFAULT_GRAPH_IDLE_WATCHDOG_SEC)
    )


def _touch_graph_activity(harness: Any) -> None:
    setattr(harness, "_last_graph_activity_monotonic", time.monotonic())


def _start_graph_idle_watchdog(runtime: Any) -> asyncio.Task[None] | None:
    interval = _graph_idle_watchdog_sec(runtime.deps.harness)
    if interval is None:
        return None
    return asyncio.create_task(_graph_idle_watchdog(runtime, interval_sec=interval))


async def _graph_idle_watchdog(runtime: Any, *, interval_sec: float) -> None:
    harness = runtime.deps.harness
    while True:
        await asyncio.sleep(interval_sec)
        last_activity = getattr(harness, "_last_graph_activity_monotonic", None)
        if not isinstance(last_activity, (int, float)):
            continue
        idle_sec = time.monotonic() - float(last_activity)
        if idle_sec < interval_sec:
            continue
        node_name = str(getattr(harness, "_active_graph_node_name", "") or "")
        node_started = getattr(harness, "_active_graph_node_started_monotonic", None)
        node_elapsed_sec = (
            round(time.monotonic() - float(node_started), 3)
            if isinstance(node_started, (int, float))
            else None
        )
        _touch_graph_activity(harness)
        _runlog(
            harness,
            "harness_idle_watchdog",
            "active graph runtime emitted no activity within watchdog window",
            idle_sec=round(idle_sec, 3),
            watchdog_sec=interval_sec,
            active_node=node_name,
            active_node_elapsed_sec=node_elapsed_sec,
        )


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


def _extract_interrupt_from_updates_payload(payload: Any) -> Any:
    """Return the interrupt value from an updates-mode payload, even if nested."""
    if not isinstance(payload, dict):
        return None
    if "__interrupt__" in payload:
        return payload["__interrupt__"]
    for value in payload.values():
        if isinstance(value, dict) and "__interrupt__" in value:
            return value["__interrupt__"]
    return None


def _contains_interrupt(payload: Any) -> bool:
    """Return True if the payload contains an interrupt channel value."""
    return _extract_interrupt_from_updates_payload(payload) is not None


def _extract_node_from_update_payload(payload: Any) -> str:
    """Return the node name from an updates-mode payload."""
    if isinstance(payload, dict):
        keys = [key for key in payload.keys() if key != "__interrupt__"]
        if keys:
            return str(keys[0])
    return ""


def _sanitize_error_summary(error: Any) -> str:
    """Return a short, sanitized error summary with no state or secrets."""
    if isinstance(error, BaseException):
        name = type(error).__name__
        message = str(error).strip()
        if not message:
            return name
        redacted = redact_sensitive_text(message)
        return f"{name}: {redacted[:200]}"
    if isinstance(error, dict):
        error_type = str(error.get("type") or error.get("error_type") or "error")
        message = str(error.get("message") or error.get("error") or "").strip()
        if not message:
            return error_type
        redacted = redact_sensitive_text(message)
        return f"{error_type}: {redacted[:200]}"
    text = str(error).strip()
    if not text:
        return "error"
    redacted = redact_sensitive_text(text)
    return redacted[:200]


def _normalize_langgraph_stream_event(
    mode: str, payload: Any, *, harness: Any | None = None
) -> UIEvent | None:
    """Normalize a LangGraph multi-mode stream event into a UIEvent.

    The returned event contains only metadata: category, node name, task/checkpoint
    ID, and status. Raw state, prompts, model output, tool args, and tool output are
    deliberately excluded.
    """
    del harness
    timestamp = datetime.now(timezone.utc).isoformat()
    if mode == "updates":
        node_name = _extract_node_from_update_payload(payload)
        return UIEvent(
            event_type=UIEventType.SYSTEM,
            content=f"Graph update: {node_name or 'unknown'}",
            data={
                "graph_event": True,
                "category": "update",
                "node": node_name,
                "task_id": None,
                "status": "success",
            },
            timestamp=timestamp,
        )
    if mode == "tasks":
        if not isinstance(payload, dict):
            return None
        task_id = str(payload.get("id") or "")
        task_name = str(payload.get("name") or "")
        has_input = "input" in payload
        has_result = "result" in payload or "error" in payload or "interrupts" in payload
        if not has_input and not has_result:
            return None
        interrupts = payload.get("interrupts")
        if interrupts:
            return UIEvent(
                event_type=UIEventType.SYSTEM,
                content=f"Graph interrupt: {task_name or 'unknown'}",
                data={
                    "graph_event": True,
                    "category": "interrupt",
                    "node": task_name,
                    "task_id": task_id,
                    "status": "needs_human",
                },
                timestamp=timestamp,
            )
        error = payload.get("error")
        if error is not None:
            return UIEvent(
                event_type=UIEventType.SYSTEM,
                content=f"Graph task error: {task_name or 'unknown'}",
                data={
                    "graph_event": True,
                    "category": "task_finished",
                    "node": task_name,
                    "task_id": task_id,
                    "status": "error",
                    "error_summary": _sanitize_error_summary(error),
                },
                timestamp=timestamp,
            )
        category = "task_started" if has_input else "task_finished"
        return UIEvent(
            event_type=UIEventType.SYSTEM,
            content=f"Graph task {category.replace('_', ' ')}: {task_name or 'unknown'}",
            data={
                "graph_event": True,
                "category": category,
                "node": task_name,
                "task_id": task_id,
                "status": "success",
            },
            timestamp=timestamp,
        )
    if mode == "checkpoints":
        if not isinstance(payload, dict):
            return None
        config = payload.get("config") or {}
        if not isinstance(config, dict):
            config = {}
        configurable = config.get("configurable") or {}
        if not isinstance(configurable, dict):
            configurable = {}
        checkpoint_id = str(configurable.get("checkpoint_id") or "")
        checkpoint_ns = str(
            configurable.get("checkpoint_ns")
            or (payload.get("metadata") or {}).get("checkpoint_ns")
            or ""
        )
        tasks = payload.get("tasks") or []
        node_name = ""
        if isinstance(tasks, list) and tasks:
            last_task = tasks[-1]
            if isinstance(last_task, dict):
                node_name = str(last_task.get("name") or "")
        if not node_name:
            next_nodes = payload.get("next") or []
            if isinstance(next_nodes, (list, tuple)) and next_nodes:
                node_name = str(next_nodes[0])
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        return UIEvent(
            event_type=UIEventType.SYSTEM,
            content=f"Graph checkpoint: {node_name or 'unknown'}",
            data={
                "graph_event": True,
                "category": "checkpoint",
                "node": node_name,
                "task_id": checkpoint_id,
                "status": "checkpoint",
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_source": str(metadata.get("source") or ""),
            },
            timestamp=timestamp,
        )
    return None


async def _forward_langgraph_event(runtime: Any, event: UIEvent) -> None:
    """Forward normalized event metadata to the runlog and optional UI handler."""
    harness = getattr(runtime.deps, "harness", None)
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        data = event.data if isinstance(event.data, dict) else {}
        runlog("langgraph_stream_event", "LangGraph stream event", **data)
    handler = getattr(runtime.deps, "event_handler", None)
    if handler is None:
        return
    if inspect.iscoroutinefunction(handler):
        await handler(event)
    else:
        handler(event)
