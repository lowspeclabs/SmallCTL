from __future__ import annotations

import re
from typing import Any

from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import json_safe_value
from ..tools.fs_loop_guard import outline_mode_violation
from ..models.tool_result import ToolEnvelope
from .state import PendingToolCall, ToolExecutionRecord, build_operation_id

_FILE_WRITE_TOOLS = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
}

_EXPLICIT_FILE_EDIT_TASK_RE = re.compile(
    r"\b(?:write|create|edit|modify|patch|replace|append|update|delete|remove|fix|save)\b.*\b(?:file|path|script|config|document|report|summary|notes|artifact)\b"
    r"|"
    r"\b(?:file|path|script|config|document|report|summary|notes|artifact)\b.*\b(?:write|create|edit|modify|patch|replace|append|update|delete|remove|fix|save)\b"
    r"|"
    r"\b(?:create|write|save)\b.{0,80}?\bat\s+[`\"']?/?[^\s`\"']+"
    r"|"
    r"\b(?:ssh_file_write|file_write|file_patch|ast_patch)\b",
    re.IGNORECASE | re.DOTALL,
)

_TASK_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])(?:\./)?/?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+")


def _task_text(state: Any) -> str:
    run_brief = getattr(state, "run_brief", None)
    task = str(getattr(run_brief, "original_task", "") or "").strip()
    if not task:
        wm = getattr(state, "working_memory", None)
        task = str(getattr(wm, "current_goal", "") or "").strip()
    return task


def _normalize_task_path(path: str) -> str:
    normalized = str(path or "").strip().strip("`\"'")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/.,;:")


def _task_mentions_target_path(state: Any, target: str) -> bool:
    normalized_target = _normalize_task_path(target)
    if not normalized_target:
        return False
    for candidate in _TASK_PATH_RE.findall(_task_text(state)):
        if _normalize_task_path(candidate) == normalized_target:
            return True
    return False


def _task_explicitly_requests_file_edit(state: Any) -> bool:
    task = _task_text(state)
    return bool(task and _EXPLICIT_FILE_EDIT_TASK_RE.search(task))


def _long_running_remote_timeout_write_guard(state: Any, pending: PendingToolCall) -> str | None:
    if pending.tool_name not in _FILE_WRITE_TOOLS:
        return None
    scratchpad = getattr(state, "scratchpad", None)
    timeout_context = scratchpad.get("_last_long_running_remote_command_timeout") if isinstance(scratchpad, dict) else None
    if not isinstance(timeout_context, dict):
        return None
    target = str(pending.args.get("path") or pending.args.get("target_path") or "").strip()
    if _task_explicitly_requests_file_edit(state) or _task_mentions_target_path(state, target):
        return None
    command = str(timeout_context.get("command") or "").strip()
    host = str(timeout_context.get("host") or "").strip()
    remote = f" on {host}" if host else ""
    target_text = f" Target attempted: {target}." if target else ""
    return (
        "Blocked file-write repair after a long-running remote SSH command timed out. "
        f"The prior command{remote} was still making installer/service progress: `{command}`. "
        "Continue the remote command with a larger `timeout_sec`, or run it detached with output redirected to a log and verify that log/service state."
        + target_text
    )


def _synthetic_blocked_records(
    graph_state: Any,
    reason: str,
) -> list[ToolExecutionRecord]:
    records: list[ToolExecutionRecord] = []
    for pending in graph_state.pending_tool_calls:
        result = ToolEnvelope.make_error(
            pending.tool_name,
            reason,
            status="blocked",
            reason="pre_dispatch_guard_blocked",
            tool_call_id=pending.tool_call_id,
            args=json_safe_value(pending.args),
        )
        records.append(
            ToolExecutionRecord(
                operation_id=build_operation_id(
                    thread_id=graph_state.thread_id,
                    step_count=getattr(graph_state.loop_state, "step_count", 0),
                    tool_call_id=pending.tool_call_id,
                    tool_name=pending.tool_name,
                ),
                tool_name=pending.tool_name,
                args=pending.args,
                tool_call_id=pending.tool_call_id,
                result=result,
            )
        )
    return records


async def _block_long_running_remote_timeout_write(
    *,
    graph_state: Any,
    deps: Any,
    pending: PendingToolCall,
) -> bool:
    harness = deps.harness
    remote_timeout_write_guard = _long_running_remote_timeout_write_guard(harness.state, pending)
    if remote_timeout_write_guard is None:
        return False
    harness.state.recent_errors.append(remote_timeout_write_guard)
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=remote_timeout_write_guard,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "long_running_remote_command",
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
            },
        )
    )
    harness._runlog(
        "long_running_remote_timeout_write_guard",
        "blocked file-write repair after remote installer timeout",
        step=harness.state.step_count,
        tool_name=pending.tool_name,
        arguments=json_safe_value(pending.args),
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content=remote_timeout_write_guard,
            data={
                "repair_kind": "long_running_remote_command",
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
            },
        ),
    )
    graph_state.last_tool_results = _synthetic_blocked_records(
        graph_state, reason=remote_timeout_write_guard
    )
    graph_state.pending_tool_calls = []
    return True


async def _block_outline_mode_violation(
    *,
    graph_state: Any,
    deps: Any,
    pending: PendingToolCall,
) -> bool:
    harness = deps.harness
    outline_violation = outline_mode_violation(
        harness.state,
        tool_name=pending.tool_name,
        args=pending.args,
    )
    if outline_violation is None:
        return False
    message = str(outline_violation.get("message") or "").strip()
    harness.state.recent_errors.append(message)
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "chunked_write_loop_guard_outline",
                "target_path": outline_violation.get("path"),
                "write_session_id": outline_violation.get("write_session_id"),
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
            },
        )
    )
    harness._runlog(
        "chunked_write_loop_guard_outline_violation",
        "blocked tool while loop guard outline mode was active",
        tool_name=pending.tool_name,
        tool_call_id=pending.tool_call_id,
        target_path=outline_violation.get("path"),
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content=message,
            data={
                "repair_kind": "chunked_write_loop_guard_outline",
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
                "target_path": outline_violation.get("path"),
            },
        ),
    )
    graph_state.last_tool_results = _synthetic_blocked_records(graph_state, reason=message)
    graph_state.pending_tool_calls = []
    return True
