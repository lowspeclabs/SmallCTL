from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Any

from ..logging_utils import log_kv
from ..models.tool_result import ToolEnvelope
from ..state import json_safe_value
from ..tools.dispatcher import normalize_tool_request
from .node_support import HALLUCINATION_MAP, ToolNotFoundError
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord, build_operation_id
from .tool_execution_support import _store_tool_execution_record


async def dispatch_tool_dag(
    graph_state: GraphRunState,
    deps: Any,
    batches: list[list[PendingToolCall]],
    *,
    max_parallel: int = 4,
    timeout_sec: int = 30,
    preserve_result_order: bool = True,
) -> list[ToolExecutionRecord]:
    """Dispatch read-only tool plan batches concurrently.

    Each batch is executed with ``asyncio.gather``.  Within a batch, individual
    calls are capped by *timeout_sec*.  Partial failures are tolerated: one
    timeout does not cancel the rest of the batch.
    """
    harness = deps.harness
    all_records: list[ToolExecutionRecord] = []

    for batch_index, batch in enumerate(batches):
        semaphore = asyncio.Semaphore(max_parallel)

        async def _run_one(pending: PendingToolCall) -> ToolExecutionRecord:
            async with semaphore:
                return await _dispatch_single_tool(
                    graph_state,
                    harness,
                    pending,
                    timeout_sec=timeout_sec,
                    batch_index=batch_index,
                    batch_size=len(batch),
                )

        coros = [_run_one(p) for p in batch]
        results = await asyncio.gather(*coros, return_exceptions=True)

        for pending, result in zip(batch, results):
            if isinstance(result, ToolExecutionRecord):
                all_records.append(result)
            else:
                # Exception fallback
                exc = result if isinstance(result, BaseException) else Exception(str(result))
                error_msg = f"DAG dispatch error for `{pending.tool_name}`: {exc}"
                log_kv(harness.log, logging.WARNING, "tool_dag_dispatch_error", tool_name=pending.tool_name, error=str(exc))
                envelope = ToolEnvelope(
                    success=False,
                    error=error_msg,
                    metadata={
                        "tool_name": pending.tool_name,
                        "reason": "dag_dispatch_exception",
                        "exception_type": type(exc).__name__,
                        "dag_batch_index": batch_index,
                        "dag_batch_size": len(batch),
                    },
                )
                operation_id = build_operation_id(
                    thread_id=graph_state.thread_id,
                    step_count=harness.state.step_count,
                    tool_call_id=pending.tool_call_id,
                    tool_name=pending.tool_name,
                )
                _store_tool_execution_record(
                    harness,
                    operation_id=operation_id,
                    thread_id=graph_state.thread_id,
                    step_count=harness.state.step_count,
                    pending=pending,
                    result=envelope,
                )
                all_records.append(
                    ToolExecutionRecord(
                        operation_id=operation_id,
                        tool_name=pending.tool_name,
                        args=pending.args,
                        tool_call_id=pending.tool_call_id,
                        result=envelope,
                    )
                )

    return all_records


async def _dispatch_single_tool(
    graph_state: GraphRunState,
    harness: Any,
    pending: PendingToolCall,
    *,
    timeout_sec: int,
    batch_index: int = 0,
    batch_size: int = 1,
) -> ToolExecutionRecord:
    """Dispatch a single read-only tool call with timeout."""
    started = perf_counter()

    def _record(envelope: ToolEnvelope) -> ToolExecutionRecord:
        elapsed_ms = max(0.0, (perf_counter() - started) * 1000.0)
        metadata = dict(getattr(envelope, "metadata", {}) or {})
        metadata["dag_latency_ms"] = round(elapsed_ms, 3)
        metadata["dag_batch_index"] = batch_index
        metadata["dag_batch_size"] = batch_size
        envelope.metadata = metadata
        _store_tool_execution_record(
            harness,
            operation_id=operation_id,
            thread_id=graph_state.thread_id,
            step_count=harness.state.step_count,
            pending=pending,
            result=envelope,
        )
        return ToolExecutionRecord(
            operation_id=operation_id,
            tool_name=pending.tool_name,
            args=pending.args,
            tool_call_id=pending.tool_call_id,
            result=envelope,
        )

    registry = getattr(harness, "registry", None)
    if registry is not None:
        normalized_tool_name, normalized_args, intercepted_result, _ = normalize_tool_request(
            registry,
            pending.tool_name,
            pending.args,
            phase=getattr(getattr(harness, "dispatcher", None), "phase", None),
            state=harness.state,
        )
    else:
        normalized_tool_name, normalized_args, intercepted_result = pending.tool_name, pending.args, None

    pending.tool_name = normalized_tool_name
    pending.args = normalized_args

    operation_id = build_operation_id(
        thread_id=graph_state.thread_id,
        step_count=harness.state.step_count,
        tool_call_id=pending.tool_call_id,
        tool_name=pending.tool_name,
    )

    if intercepted_result is not None:
        return _record(intercepted_result)

    # Registry check
    names_fn = getattr(registry, "names", None) if registry is not None else None
    if callable(names_fn) and pending.tool_name not in names_fn():
        if pending.tool_name in HALLUCINATION_MAP:
            mapped_tool = HALLUCINATION_MAP[pending.tool_name]
            raw_id = (
                pending.args.get("path")
                or pending.args.get("artifact_id")
                or pending.args.get("pattern")
                or "A000X"
            )
            artifact_id = str(raw_id).split("/")[-1]
            if not artifact_id.startswith("A") and "A" in artifact_id:
                idx = artifact_id.find("A")
                artifact_id = artifact_id[idx:]
            hint = (
                f"Tool '{pending.tool_name}' is unavailable. "
                f"Use '{mapped_tool}(artifact_id=\"{artifact_id}\")' instead."
            )
            envelope = ToolEnvelope(
                success=True,
                output=hint,
                metadata={"interceptor_hit": True, "hallucinated_tool": pending.tool_name},
            )
        else:
            envelope = ToolEnvelope(
                success=False,
                error=f"Unknown tool: {pending.tool_name}",
                metadata={"tool_name": pending.tool_name},
            )
        return _record(envelope)

    dispatch_fn = getattr(harness, "_dispatch_tool_call", None)
    if not callable(dispatch_fn):
        envelope = ToolEnvelope(
            success=False,
            error="Tool dispatcher is unavailable in the current harness context.",
            metadata={"tool_name": pending.tool_name, "reason": "dispatcher_unavailable"},
        )
        return _record(envelope)

    try:
        result = await asyncio.wait_for(
            dispatch_fn(pending.tool_name, pending.args),
            timeout=timeout_sec,
        )
    except asyncio.TimeoutError:
        envelope = ToolEnvelope(
            success=False,
            error=f"Tool `{pending.tool_name}` timed out after {timeout_sec}s in DAG batch.",
            metadata={
                "tool_name": pending.tool_name,
                "reason": "dag_timeout",
                "timeout_sec": timeout_sec,
            },
        )
        return _record(envelope)
    except Exception as exc:
        envelope = ToolEnvelope(
            success=False,
            error=f"Tool `{pending.tool_name}` raised {type(exc).__name__}: {exc}",
            metadata={
                "tool_name": pending.tool_name,
                "reason": "dag_exception",
                "exception_type": type(exc).__name__,
            },
        )
        return _record(envelope)

    return _record(result)
