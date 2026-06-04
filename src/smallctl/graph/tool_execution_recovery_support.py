from __future__ import annotations

import json
from typing import Any

from ..models.conversation import ConversationMessage
from ..state import json_safe_value
from .state import PendingToolCall
from .tool_call_parser import _extract_artifact_id_from_args
from .tool_execution_recovery_constants import (
    CHUNK_WRITE_LOOP_GUARD_TOOLS,
    TERMINAL_WRITE_SESSION_REPAIR_KEY,
    WRITE_TOOLS,
)
from .tool_execution_recovery_helpers import current_verifier_already_passed, shell_exec_success_record_for_pending
from .tool_loop_guards import _tool_call_fingerprint


def _artifact_read_loop_exceeded_limit(
    harness: Any,
    pending: PendingToolCall,
    limit: int = 5,
) -> bool:
    """
    Returns True if there have been `limit` or more artifact_read calls
    on the same artifact without an intervening write operation.
    """
    if pending.tool_name != "artifact_read":
        return False
    artifact_id = _extract_artifact_id_from_args(pending.args)
    if not artifact_id:
        return False

    from .tool_loop_guards import _tool_attempt_history

    history = _tool_attempt_history(harness)
    read_count = 0
    for item in reversed(history):
        tool_name = str(item.get("tool_name", ""))
        if tool_name in WRITE_TOOLS:
            break
        if tool_name == "artifact_read":
            fingerprint = str(item.get("fingerprint", ""))
            try:
                payload = json.loads(fingerprint)
            except Exception:
                continue
            args = payload.get("args", {}) if isinstance(payload, dict) else {}
            if str(args.get("artifact_id", "")).strip() == artifact_id:
                read_count += 1
                if read_count >= limit:
                    return True
    return False


def _maybe_emit_terminal_write_session_reuse_nudge(harness: Any, record: Any) -> bool:
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if str(metadata.get("error_kind") or "").strip() != "write_session_already_terminal":
        return False
    write_session_id = str(metadata.get("write_session_id") or record.args.get("write_session_id") or "").strip()
    if not write_session_id:
        return False
    path = str(record.args.get("path") or metadata.get("target_path") or metadata.get("path") or "").strip()
    signature = "|".join([record.tool_name, write_session_id, path])
    seen = harness.state.scratchpad.setdefault(TERMINAL_WRITE_SESSION_REPAIR_KEY, [])
    if not isinstance(seen, list):
        seen = []
    if signature in seen:
        return False
    seen.append(signature)
    harness.state.scratchpad[TERMINAL_WRITE_SESSION_REPAIR_KEY] = seen[-20:]
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Write Session `{write_session_id}` is terminal and cannot be reused. "
                f"Do not reuse write_session_id={write_session_id}; omit `write_session_id` "
                "for direct overwrite or start a fresh session. If this is a narrow repair, "
                "prefer `file_patch`/`ast_patch` without the stale write_session_id."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "terminal_write_session_reuse",
                "write_session_id": write_session_id,
                "target_path": path,
                "tool_name": record.tool_name,
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "terminal_write_session_reuse_nudge",
            "nudged model away from reusing terminal write_session_id",
            tool_name=record.tool_name,
            write_session_id=write_session_id,
            target_path=path,
        )
    return True


def _suppress_repeated_successful_shell_exec(
    *,
    harness: Any,
    graph_state: Any,
    pending: PendingToolCall,
    repeat_error: str,
) -> bool:
    if pending.tool_name != "shell_exec":
        return False
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    if isinstance(scratchpad, dict) and isinstance(scratchpad.get("_last_verifier_stale_after_mutation"), dict):
        return False
    if shell_exec_success_record_for_pending(graph_state, pending) is None and not current_verifier_already_passed(harness, pending):
        return False

    command = str(pending.args.get("command") or "").strip()
    signature = _tool_call_fingerprint(pending.tool_name, pending.args)
    if isinstance(scratchpad, dict) and scratchpad.get("_shell_exec_success_reuse_nudged") != signature:
        scratchpad["_shell_exec_success_reuse_nudged"] = signature
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    "The exact `shell_exec` command already succeeded and no later mutation invalidated it. "
                    f"Do not rerun `{command}`. Use that verifier evidence and call `task_complete(message=...)` now; "
                    "if the task is not actually satisfied, explain the remaining blocker instead."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "shell_exec_already_succeeded",
                    "guard": "repeated_tool_loop",
                    "tool_name": pending.tool_name,
                    "command": command,
                },
            )
        )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "shell_exec_repeated_success_suppressed",
            "suppressed repeated successful shell_exec and nudged terminal completion",
            step=harness.state.step_count,
            tool_name=pending.tool_name,
            arguments=json_safe_value(pending.args),
            guard_error=repeat_error,
        )
    graph_state.pending_tool_calls = []
    graph_state.last_tool_results = []
    return True


def _maybe_schedule_chunked_write_loop_guard_read(
    graph_state: Any,
    harness: Any,
    record: Any,
) -> bool:
    if record.tool_name not in CHUNK_WRITE_LOOP_GUARD_TOOLS or record.result.success:
        return False
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if not bool(metadata.get("loop_guard_schedule_read")):
        return False

    path = str(metadata.get("path") or record.args.get("path") or "").strip()
    if not path:
        return False

    signature = "|".join(
        [
            path,
            str(metadata.get("error_kind") or ""),
            str(metadata.get("loop_guard_escalation_level") or ""),
            str(metadata.get("loop_guard_score") or ""),
            str(metadata.get("section_name") or ""),
        ]
    )
    if harness.state.scratchpad.get("_chunk_write_loop_guard_read_scheduled") == signature:
        return False
    harness.state.scratchpad["_chunk_write_loop_guard_read_scheduled"] = signature

    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args={"path": path},
            raw_arguments=json.dumps({"path": path}, ensure_ascii=True, sort_keys=True),
            source="system",
        )
    ]
    from .tool_call_parser import allow_repeated_tool_call_once

    allow_repeated_tool_call_once(harness, "file_read", {"path": path})

    escalation_level = int(metadata.get("loop_guard_escalation_level", 0) or 0)
    base_message = (
        f"LoopGuard blocked a repeated chunk write for `{path}`. "
        "Reading the current staged content before another write."
    )
    if escalation_level >= 2:
        base_message = (
            f"LoopGuard blocked a repeated chunk write for `{path}` again. "
            "Reading the current staged content now; then send a 3-bullet outline with `ask_human(...)` before the next `file_write`."
        )
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=base_message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "chunked_write_loop_guard",
                "target_path": path,
                "escalation_level": escalation_level,
            },
        )
    )
    harness._runlog(
        "chunked_write_loop_guard_read_scheduled",
        "scheduled mandatory file_read after chunked write loop guard block",
        path=path,
        error_kind=str(metadata.get("error_kind") or ""),
        escalation_level=escalation_level,
        stagnation_score=int(metadata.get("loop_guard_score", 0) or 0),
    )
    return True
