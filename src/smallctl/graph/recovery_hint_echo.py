from __future__ import annotations

import re
from typing import Any


def _normalize_recovery_hint_command(command: str) -> str:
    """Normalize a command string for comparison (strip whitespace, collapse spaces)."""
    return re.sub(r"\s+", " ", str(command or "").strip())


def record_recovery_hint(state: Any, *, hint_type: str, command: str = "", metadata: dict[str, Any] | None = None) -> None:
    """Record a recovery hint that was issued so we can detect model echoing later."""
    if state is None:
        return
    hints = getattr(state, "recovery_hints_issued", None)
    if not isinstance(hints, list):
        hints = []
        state.recovery_hints_issued = hints
    normalized = _normalize_recovery_hint_command(command)
    if not normalized:
        return
    step_count = int(getattr(state, "step_count", 0) or 0)
    hints.append({
        "hint_type": hint_type,
        "command": normalized,
        "step_count": step_count,
        "executed_step": None,
        "metadata": dict(metadata or {}),
    })
    # Keep only last 20 hints
    if len(hints) > 20:
        hints[:] = hints[-20:]
    if hasattr(state, "touch"):
        state.touch()


def mark_recovery_hint_executed(state: Any, *, tool_name: str, args: dict[str, Any]) -> None:
    """Mark any matching recovery hint as executed so we know it was actually run."""
    if state is None:
        return
    hints = getattr(state, "recovery_hints_issued", None)
    if not isinstance(hints, list):
        return
    command = ""
    if tool_name in {"shell_exec", "ssh_exec", "bash_exec"}:
        command = str(args.get("command") or "").strip()
    elif tool_name == "file_read":
        command = f"cat {args.get('path', '')}"
    elif tool_name == "dir_list":
        command = f"ls {args.get('path', '')}"
    normalized = _normalize_recovery_hint_command(command)
    if not normalized:
        return
    current_step = int(getattr(state, "step_count", 0) or 0)
    for hint in hints:
        hint_cmd = str(hint.get("command") or "").strip()
        if not hint_cmd:
            continue
        if normalized == hint_cmd:
            hint["executed_step"] = current_step
        elif normalized in hint_cmd or hint_cmd in normalized:
            min_len = min(len(normalized), len(hint_cmd))
            if min_len > 0 and max(len(normalized), len(hint_cmd)) / min_len <= 1.5:
                hint["executed_step"] = current_step


def is_recovery_hint_echo(state: Any, *, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Check if a proposed tool call matches a recently executed recovery hint.

    Returns the matching hint record only if the hint was actually executed
    (not just suggested) and the match is within the last 5 turns.
    """
    if state is None:
        return None
    hints = getattr(state, "recovery_hints_issued", None)
    if not isinstance(hints, list):
        return None
    current_step = int(getattr(state, "step_count", 0) or 0)
    # Only check hints from last 5 turns that were actually executed
    recent_hints = [
        h for h in hints
        if current_step - int(h.get("step_count", 0) or 0) <= 5
        and h.get("executed_step") is not None
    ]
    if not recent_hints:
        return None
    command = ""
    if tool_name in {"shell_exec", "ssh_exec", "bash_exec"}:
        command = str(args.get("command") or "").strip()
    elif tool_name == "file_read":
        command = f"cat {args.get('path', '')}"
    elif tool_name == "dir_list":
        command = f"ls {args.get('path', '')}"
    normalized = _normalize_recovery_hint_command(command)
    if not normalized:
        return None
    for hint in recent_hints:
        hint_cmd = str(hint.get("command") or "").strip()
        if not hint_cmd:
            continue
        # Check if the normalized commands match
        if normalized == hint_cmd:
            return hint
        # Also check if one is a substring of the other (partial match for compound commands)
        if normalized in hint_cmd or hint_cmd in normalized:
            # Only match if they're substantially similar (not just a small substring)
            min_len = min(len(normalized), len(hint_cmd))
            if min_len > 0 and max(len(normalized), len(hint_cmd)) / min_len <= 1.5:
                return hint
    return None


def intercept_recovery_hint_echo(state: Any, pending_tool_calls: list[Any]) -> tuple[list[Any], list[dict[str, Any]]]:
    """Filter out tool calls that echo recently issued recovery hints.

    Returns (filtered_calls, skipped_hints) where skipped_hints describes
    what was intercepted for potential UI display.
    """
    if state is None or not pending_tool_calls:
        return pending_tool_calls, []
    filtered: list[Any] = []
    skipped: list[dict[str, Any]] = []
    for pending in pending_tool_calls:
        args = dict(getattr(pending, "args", {}) or {})
        tool_name = str(getattr(pending, "tool_name", "") or "").strip()
        matched_hint = is_recovery_hint_echo(state, tool_name=tool_name, args=args)
        if matched_hint is not None:
            skipped.append({
                "tool_name": tool_name,
                "args": args,
                "matched_hint": matched_hint,
            })
            continue
        filtered.append(pending)
    return filtered, skipped
