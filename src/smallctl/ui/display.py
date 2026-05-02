"""
smallctl/ui/display.py
----------------------
Display formatting and rendering helpers for the SmallctlApp UI.

This module extracts display/formatting logic from the main app to keep
the orchestration code clean and pipeline-like.
"""

from __future__ import annotations

from difflib import SequenceMatcher
import re
from typing import Any

from ..models.events import (
    UIEvent,
    UIEventType,
    UIStatusSnapshot,
    compute_activity_for_event,
)


_DUPLICATE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "completed",
    "complete",
    "for",
    "found",
    "from",
    "identified",
    "in",
    "including",
    "into",
    "is",
    "of",
    "on",
    "or",
    "successfully",
    "the",
    "to",
    "with",
}


def format_restore_status(status: dict[str, Any]) -> str:
    """Format the restore status dict into a user-friendly message."""
    if status.get("status") == "not_found":
        thread_id = str(status.get("thread_id") or "").strip()
        if thread_id:
            return f"No persisted graph state found for thread {thread_id}."
        return "No persisted graph state found."

    thread_id = str(status.get("thread_id") or "").strip() or "unknown"
    interrupt = status.get("interrupt")
    if isinstance(interrupt, dict):
        question = str(interrupt.get("question") or "").strip()
        if question:
            return (
                f"Restored graph state for thread {thread_id}. "
                f"Submit a reply to continue: {question}"
            )
    return f"Restored graph state for thread {thread_id}."


def should_render_run_log_row(row: dict[str, Any]) -> bool:
    """
    Determine if a run log row should be rendered in the UI.
    Keeps model/tool/chat protocol logs out of the main transcript.
    """
    channel = str(row.get("channel") or "")
    event = str(row.get("event") or "")

    if channel in {"tools", "chat", "model_output"}:
        return False
    if channel != "harness":
        return False
    if event in {
        "chunk",
        "model_token",
        "model_output",
        "model_thinking",
        "harness_tool_dispatch",
        "harness_tool_result",
        "tool_replay_hit",
    }:
        return False
    return True


def should_render_event(event: UIEvent, *, show_system_messages: bool, show_tool_calls: bool) -> bool:
    """Determine if an event should be rendered based on user preferences."""
    if event.event_type == UIEventType.STATUS:
        return False
    if event.event_type in {UIEventType.SYSTEM, UIEventType.METRICS} and not show_system_messages:
        return False
    if event.event_type in {UIEventType.TOOL_CALL, UIEventType.TOOL_RESULT} and not show_tool_calls:
        return False
    return True


def format_run_log_row(row: dict[str, Any]) -> str:
    """Format a run log row for display."""
    msg = row.get("message") or ""
    if len(msg) > 1024:
        msg = msg[:1024] + "... [truncated]"
    return f"[{row.get('channel')}] {row.get('event')}: {msg}"


def compute_activity_for_event(
    event: UIEvent,
    *,
    active_task_done: bool | None = None,
) -> str | None:
    """
    Compute the activity status text for a given UI event.
    Returns None if no activity update is needed.
    """
    # Check for explicit status_activity in data
    if "status_activity" in event.data:
        return str(event.data.get("status_activity") or "").strip()

    if event.event_type == UIEventType.TOOL_CALL:
        tool_name = str(event.content or event.data.get("tool_name") or "").strip()
        return f"running {tool_name}..." if tool_name else "running tool..."

    if event.event_type == UIEventType.TOOL_RESULT:
        if active_task_done is False:
            return "thinking..."
        return ""

    if event.event_type in {UIEventType.ASSISTANT, UIEventType.THINKING}:
        return "responding..."

    if event.event_type == UIEventType.ERROR:
        return ""

    return None


def format_tool_call_for_display(tool_name: str, args: dict[str, Any]) -> str:
    """Format a tool call for display in the UI."""
    display_text = f"**{tool_name}**"
    if args:
        # Format args as compact key=value pairs
        arg_parts = []
        for k, v in args.items():
            if isinstance(v, str):
                # Truncate long strings
                if len(v) > 50:
                    v = v[:47] + "..."
                arg_parts.append(f"{k}={v!r}")
            else:
                arg_parts.append(f"{k}={v}")
        if arg_parts:
            display_text += f"({', '.join(arg_parts)})"
    return display_text


def format_tool_result_for_display(
    result: Any,
    *,
    max_length: int = 200,
) -> str:
    """Format a tool result for display in the UI."""
    if result is None:
        return "(no result)"

    if isinstance(result, dict):
        # Check for summary or message fields
        text = result.get("summary") or result.get("message") or result.get("output")
        if text:
            text = str(text)
        else:
            # Compact JSON representation
            import json
            text = json.dumps(result, default=str)
    else:
        text = str(result)

    if len(text) > max_length:
        text = text[: max_length - 3] + "..."

    return text


def should_promote_tool_args_to_assistant(tool_name: str, args: dict[str, Any]) -> str | None:
    """
    Check if tool arguments should be promoted to assistant text.
    Returns the text to promote, or None if no promotion should happen.
    """
    if tool_name == "task_complete":
        promote_text = str(args.get("message") or "").strip()
        if promote_text:
            return promote_text
    elif tool_name == "ask_human":
        promote_text = str(args.get("question") or "").strip()
        if promote_text:
            return promote_text
    return None


def check_duplicate_promotion(
    promote_text: str,
    active_assistant_text: str,
) -> bool:
    """
    Check if the promotion text would duplicate existing assistant text.
    Returns True if promotion should be skipped as a duplicate.
    """
    if not active_assistant_text:
        return False

    promote_norm = _normalize_duplicate_text(promote_text)
    active_norm = _normalize_duplicate_text(active_assistant_text)
    if not promote_norm or not active_norm:
        return False

    if promote_norm in active_norm or active_norm in promote_norm:
        return True

    similarity = SequenceMatcher(None, promote_norm, active_norm).ratio()
    if similarity >= 0.72:
        return True

    promote_tokens = set(_salient_duplicate_tokens(promote_text))
    active_tokens = set(_salient_duplicate_tokens(active_assistant_text))
    if not promote_tokens or not active_tokens:
        return False

    shared = promote_tokens & active_tokens
    overlap = len(shared) / min(len(promote_tokens), len(active_tokens))
    return len(shared) >= 4 and overlap >= 0.7


def _normalize_duplicate_text(text: str) -> str:
    return " ".join(_raw_duplicate_tokens(text))


def _salient_duplicate_tokens(text: str) -> list[str]:
    salient: list[str] = []
    for token in _raw_duplicate_tokens(text):
        if token in _DUPLICATE_STOPWORDS:
            continue
        if _is_standalone_ip(token):
            continue
        salient.append(token)
    return salient


def _raw_duplicate_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.findall(r"[a-z0-9./:+_-]+", str(text or "").lower()):
        normalized = token.strip(".,;!?()[]{}")
        if normalized:
            tokens.append(normalized)
    return tokens


def _is_standalone_ip(token: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", token))


class StatusState:
    """
    Immutable state object for status bar values.
    Encapsulates all status information to simplify refresh logic.
    """

    def __init__(
        self,
        *,
        model: str = "n/a",
        phase: str = "explore",
        step: int | str = 0,
        mode: str = "execution",
        plan: str = "",
        active_step: str = "",
        activity: str = "",
        contract_flow_ui: bool = False,
        contract_phase: str = "",
        acceptance_progress: str = "",
        latest_verdict: str = "",
        token_usage: int = 0,
        token_total: int = 0,
        token_limit: int = 0,
        api_errors: int = 0,
    ) -> None:
        self.model = model
        self.phase = phase
        self.step = step
        self.mode = mode
        self.plan = plan
        self.active_step = active_step
        self.activity = activity
        self.contract_flow_ui = contract_flow_ui
        self.contract_phase = contract_phase
        self.acceptance_progress = acceptance_progress
        self.latest_verdict = latest_verdict
        self.token_usage = token_usage
        self.token_total = token_total
        self.token_limit = token_limit
        self.api_errors = api_errors

    @classmethod
    def from_harness(
        cls,
        harness: Any,
        harness_kwargs: dict[str, Any],
        *,
        activity: str = "",
        api_errors: int = 0,
        active_task: Any = None,
    ) -> "StatusState":
        snapshot_activity = activity
        if not snapshot_activity and active_task is not None and not active_task.done():
            snapshot_activity = "thinking..."
        snapshot = UIStatusSnapshot.from_harness(
            harness,
            harness_kwargs,
            activity=snapshot_activity,
            api_errors=api_errors,
        )
        return cls.from_snapshot(snapshot)

    @classmethod
    def from_snapshot(cls, snapshot: UIStatusSnapshot | dict[str, Any]) -> "StatusState":
        if isinstance(snapshot, UIStatusSnapshot):
            payload = snapshot.to_dict()
        else:
            payload = dict(snapshot)
        return cls(
            model=str(payload.get("model", "n/a") or "n/a"),
            phase=str(payload.get("phase", "explore") or "explore"),
            step=payload.get("step", 0),
            mode=str(payload.get("mode", "execution") or "execution"),
            plan=str(payload.get("plan", "") or ""),
            active_step=str(payload.get("active_step", "") or ""),
            activity=str(payload.get("activity", "") or ""),
            contract_flow_ui=bool(payload.get("contract_flow_ui", False)),
            contract_phase=str(payload.get("contract_phase", "") or ""),
            acceptance_progress=str(payload.get("acceptance_progress", "") or ""),
            latest_verdict=str(payload.get("latest_verdict", "") or ""),
            token_usage=max(0, int(payload.get("token_usage", 0) or 0)),
            token_total=max(0, int(payload.get("token_total", 0) or 0)),
            token_limit=max(0, int(payload.get("token_limit", 0) or 0)),
            api_errors=max(0, int(payload.get("api_errors", 0) or 0)),
        )
