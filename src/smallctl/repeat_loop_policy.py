from __future__ import annotations

FILE_READ_REPEAT_MIN = 7


def repeated_action_limit(tool_name: str, default_limit: int) -> int:
    """Return the coarse repeat-action threshold for a tool."""
    if str(tool_name or "").strip() == "file_read":
        return max(default_limit, FILE_READ_REPEAT_MIN)
    return default_limit


def strict_identical_limit(tool_name: str, default_limit: int) -> int:
    """Return the live strict identical-call threshold for a tool."""
    if str(tool_name or "").strip() == "file_read":
        return max(default_limit, FILE_READ_REPEAT_MIN)
    return default_limit


def strict_window_limit(tool_name: str, default_limit: int) -> int:
    """Return the live strict repeat-window threshold for a tool."""
    if str(tool_name or "").strip() == "file_read":
        return max(default_limit, FILE_READ_REPEAT_MIN)
    return default_limit
