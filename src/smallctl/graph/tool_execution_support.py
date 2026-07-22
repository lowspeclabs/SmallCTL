from __future__ import annotations

from typing import Any

from ..durable_tool_results import compact_tool_result_for_durable_state
from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from ..state import json_safe_value


def _tool_envelope_from_dict(payload: dict[str, Any]) -> ToolEnvelope:
    metadata = json_safe_value(payload.get("metadata") or {})
    if not isinstance(metadata, dict):
        metadata = {}
    return ToolEnvelope(
        success=bool(payload.get("success")),
        status=payload.get("status"),
        output=json_safe_value(payload.get("output")),
        error=None if payload.get("error") is None else str(payload.get("error")),
        metadata=metadata,
    )


def _conversation_message_from_dict(payload: dict[str, Any]) -> ConversationMessage:
    normalized = json_safe_value(payload)
    if not isinstance(normalized, dict):
        normalized = {}
    role = str(normalized.get("role", "tool"))
    content = normalized.get("content")
    if content is not None:
        content = str(content)
    name = normalized.get("name")
    if name is not None:
        name = str(name)
    tool_call_id = normalized.get("tool_call_id")
    if tool_call_id is not None:
        tool_call_id = str(tool_call_id)
    tool_calls = normalized.get("tool_calls")
    metadata = normalized.get("metadata")
    return ConversationMessage(
        role=role,
        content=content,
        name=name,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls if isinstance(tool_calls, list) else [],
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _get_tool_execution_record(harness: Any, operation_id: str) -> dict[str, Any]:
    records = getattr(harness.state, "tool_execution_records", None)
    if not isinstance(records, dict):
        harness.state.tool_execution_records = {}
        return {}
    record = records.get(operation_id)
    return dict(record) if isinstance(record, dict) else {}


def _sanitize_tool_args_for_durable_state(harness: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of tool args for durable state."""
    if not isinstance(args, dict):
        return {}
    return dict(args)


def _store_tool_execution_record(
    harness: Any,
    *,
    operation_id: str,
    thread_id: str,
    step_count: int,
    pending: Any,
    result: Any,
) -> None:
    existing = _get_tool_execution_record(harness, operation_id)
    step_id = str(getattr(harness.state, "active_step_id", "") or "")
    step_run_id = str(getattr(harness.state, "active_step_run_id", "") or "")
    plan = getattr(harness.state, "active_plan", None) or getattr(harness.state, "draft_plan", None)
    plan_id = str(getattr(plan, "plan_id", "") or "")
    step_attempt = 0
    if step_id and plan is not None and hasattr(plan, "find_step"):
        step = plan.find_step(step_id)
        if step is not None:
            step_attempt = int(getattr(step, "retry_count", 0) or 0) + 1
    durable_args = _sanitize_tool_args_for_durable_state(harness, pending.args)
    existing.update(
        {
            "operation_id": operation_id,
            "thread_id": thread_id,
            "step_count": step_count,
            "tool_name": pending.tool_name,
            "tool_call_id": pending.tool_call_id,
            "source": str(getattr(pending, "source", "model") or "model"),
            "args": durable_args,
            "result": compact_tool_result_for_durable_state(
                result.to_dict(),
                tool_name=str(pending.tool_name or ""),
            ),
            "plan_id": plan_id,
            "step_id": step_id,
            "step_run_id": step_run_id,
            "step_attempt": step_attempt,
            "evidence_context": {
                "operation_id": operation_id,
                "thread_id": thread_id,
                "step_count": step_count,
                "phase": str(getattr(harness.state, "current_phase", "") or ""),
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
                "source": str(getattr(pending, "source", "model") or "model"),
                "args": durable_args,
                "replayed": bool(result.metadata.get("cache_hit")) if isinstance(result.metadata, dict) else False,
                "artifact_id": str(result.metadata.get("artifact_id", "") or "").strip() if isinstance(result.metadata, dict) else "",
                "plan_id": plan_id,
                "step_id": step_id,
                "step_run_id": step_run_id,
                "step_attempt": step_attempt,
            },
        }
    )
    harness.state.tool_execution_records[operation_id] = existing


def _has_matching_tool_message(harness: Any, message: ConversationMessage) -> bool:
    for existing in reversed(harness.state.recent_messages):
        if existing.role != "tool":
            continue
        if existing.name != message.name:
            continue
        if existing.tool_call_id != message.tool_call_id:
            continue
        if existing.content != message.content:
            continue
        if existing.metadata != message.metadata:
            continue
        return True
    return False
