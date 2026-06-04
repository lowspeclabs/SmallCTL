from __future__ import annotations

from typing import Any

from .tool_result_verification_removal import _removal_task_text

_AUDIT_TASK_KEYWORDS = frozenset([
    "audit", "investigate", "review", "assess", "check", "report on",
    "inspect", "examine", "analyze", "verify compliance", "document",
])


def _is_audit_task(state: Any) -> bool:
    """Return True when the original task is an audit/investigation."""
    task_text = _removal_task_text(state).lower()
    if not task_text:
        return False
    return any(kw in task_text for kw in _AUDIT_TASK_KEYWORDS)
