from __future__ import annotations

from typing import Any

from ..harness.tool_visibility import (
    hidden_tool_reason,
    recent_hidden_tool_recovery_artifact_id,
    resolve_turn_tool_exposure,
)
from .state import GraphRunState, PendingToolCall


_HIDDEN_TOOL_REASON_LABELS = {
    "missing_index": "missing index",
    "no_artifacts": "no artifacts yet",
    "no_active_plan": "no active plan",
    "no_background_jobs": "no background jobs",
    "write_session_not_finalizable": "write session not finalizable",
}
_CHAT_SYNTHETIC_TERMINAL_TOOLS = {"task_complete", "task_fail"}


def _format_allowed_tool_summary(names: list[str], *, limit: int = 8) -> str:
    visible_names = [str(name).strip() for name in names if str(name).strip()]
    if not visible_names:
        return "No tools are available on this turn."
    shown = visible_names[:limit]
    summary = ", ".join(shown)
    if len(visible_names) > limit:
        summary = f"{summary}, ..."
    return f"Available now: {summary}"


def _validation_handoff_hint_for_blocked_tool(pending: PendingToolCall, *, mode: str) -> str:
    tool_name = str(pending.tool_name or "").strip()
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode != "planning" or tool_name not in {"run", "shell_exec"}:
        return ""
    command = str(
        pending.args.get("command")
        or pending.args.get("cmd")
        or pending.args.get("args")
        or pending.args.get("code")
        or ""
    ).strip()
    if command:
        return (
            "Planning mode cannot execute shell commands. For phase verification, call "
            f"`request_validation_execution(command={command!r})` instead; after approval the loop runtime will run it via `shell_exec`. "
            "Do not promote the phase from static file reads alone."
        )
    return (
        "Planning mode cannot execute shell commands and there is no tool named `run`. "
        "For phase verification, identify the exact verifier/test command and call `request_validation_execution(command=...)`; "
        "after approval the loop runtime will run it via `shell_exec`. Do not promote the phase from static file reads alone."
    )


def _build_hidden_tool_block_message(
    blocked_calls: list[PendingToolCall],
    *,
    allowed_names: list[str],
    harness: Any,
    mode: str,
) -> str:
    blocked_bits: list[str] = []
    recovery_hints: list[str] = []
    for pending in blocked_calls:
        label = f"`{pending.tool_name}`"
        reason_code = hidden_tool_reason(
            pending.tool_name,
            state=harness.state,
            mode=mode,
        )
        reason_text = _HIDDEN_TOOL_REASON_LABELS.get(str(reason_code or "").strip())
        if reason_text:
            label = f"{label} ({reason_text})"
        blocked_bits.append(label)
        artifact_id = recent_hidden_tool_recovery_artifact_id(
            harness.state,
            tool_name=pending.tool_name,
        )
        if artifact_id:
            recovery_hints.append(f"Use `artifact_read(artifact_id='{artifact_id}')` for the full fetched body.")
        validation_hint = _validation_handoff_hint_for_blocked_tool(pending, mode=mode)
        if validation_hint:
            recovery_hints.append(validation_hint)
    blocked_summary = ", ".join(blocked_bits)
    message = (
        f"Registered but unavailable on this turn: {blocked_summary}. "
        f"{_format_allowed_tool_summary(allowed_names)}"
    )
    if recovery_hints:
        message = f"{message} {' '.join(recovery_hints)}"
    return message


def _rerouteable_hidden_tool_call(
    blocked_calls: list[PendingToolCall],
    *,
    hidden_reasons: dict[str, str | None],
    mode: str,
) -> PendingToolCall | None:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in {"chat", "planning", "loop"}:
        return None
    for pending in blocked_calls:
        if hidden_reasons.get(pending.tool_name):
            continue
        if normalized_mode == "loop":
            if pending.tool_name == "web_fetch":
                return pending
            continue
        if pending.tool_name in {
            "shell_exec",
            "ssh_exec",
            "file_patch",
            "ast_patch",
            "file_write",
            "file_append",
            "finalize_write_session",
            "web_fetch",
        }:
            return pending
    return None


def _hidden_tool_retry_message(pending: PendingToolCall) -> str:
    tool_name = str(pending.tool_name or "").strip()
    if not tool_name:
        return ""
    details: list[str] = [
        f"Retry on the next turn with `{tool_name}` immediately.",
        "Do not restart the analysis or re-read the same evidence unless the patch/tool arguments truly need new context.",
    ]
    path = str(pending.args.get("path") or pending.args.get("target_path") or "").strip()
    if path:
        details.append(f"Target path: `{path}`.")
    command = str(pending.args.get("command") or "").strip()
    if command:
        details.append(f"Reuse this command: `{command}`.")
    return " ".join(details)


def _strip_hidden_chat_terminal_completion_calls(
    graph_state: GraphRunState,
    harness: Any,
) -> bool:
    if graph_state.run_mode != "chat" or not graph_state.pending_tool_calls:
        return False

    assistant_text = str(graph_state.last_assistant_text or "").strip()
    if not assistant_text:
        return False

    suppressed_reason = str(
        getattr(harness.state, "scratchpad", {}).get("_chat_tools_suppressed_reason") or ""
    ).strip()
    if suppressed_reason not in {"non_lookup_chat_terminal_only", "smalltalk_no_tools"}:
        return False

    model_calls = [
        pending
        for pending in graph_state.pending_tool_calls
        if str(getattr(pending, "source", "model") or "model").strip().lower() == "model"
    ]
    if len(model_calls) != len(graph_state.pending_tool_calls):
        return False
    if any(pending.tool_name not in _CHAT_SYNTHETIC_TERMINAL_TOOLS for pending in model_calls):
        return False

    tool_exposure = resolve_turn_tool_exposure(harness, graph_state.run_mode)
    allowed_tool_names = {
        str(name).strip()
        for name in tool_exposure.get("names", [])
        if str(name).strip()
    }
    if any(pending.tool_name in allowed_tool_names for pending in model_calls):
        return False

    blocked_tools = [pending.tool_name for pending in model_calls]
    graph_state.pending_tool_calls = []
    harness._runlog(
        "chat_hidden_terminal_tool_ignored",
        "ignored hidden chat terminal tool and finalized from assistant prose",
        blocked_tools=blocked_tools,
        suppressed_reason=suppressed_reason,
    )
    return True
