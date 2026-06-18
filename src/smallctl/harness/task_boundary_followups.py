from __future__ import annotations

import re
from typing import Any

from .task_boundary_constants import _FOLLOWUP_FILLERS


_CONTINUE_VARIANTS = {
    "continue",
    "cntinue",
    "conitnue",
    "continune",
    "cotinue",
    "keep going",
    "resume",
    "proceed",
    "go on",
    "carry on",
}

_SSH_CREDENTIAL_FOLLOWUP_RE = re.compile(
    r"\b(?:ssh\s+)?(?:password|passphrase|credential|credentials)\b\s*(?:is|are|=|:)",
    re.IGNORECASE,
)


def is_continue_like_followup(task: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(task or "").strip().lower()).strip()
    if not normalized:
        return False
    tokens = [token for token in normalized.split() if token not in _FOLLOWUP_FILLERS]
    collapsed = " ".join(tokens)
    return collapsed in _CONTINUE_VARIANTS


def has_plan_execution_approval_context(state: Any) -> bool:
    pending_interrupt = getattr(state, "pending_interrupt", None)
    if isinstance(pending_interrupt, dict) and pending_interrupt.get("kind") == "plan_execute_approval":
        return True
    planner_interrupt = getattr(state, "planner_interrupt", None)
    if str(getattr(planner_interrupt, "kind", "") or "").strip() == "plan_execute_approval":
        return True
    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    status = str(getattr(plan, "status", "") or "").strip().lower()
    return status == "awaiting_approval"


def is_ssh_credential_followup(state: Any, task: str) -> bool:
    text = str(task or "").strip()
    if not text or not _SSH_CREDENTIAL_FOLLOWUP_RE.search(text):
        return False
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    recovery_state = scratchpad.get("_ssh_auth_recovery_state")
    if isinstance(recovery_state, dict) and recovery_state:
        return True
    recent_errors = getattr(state, "recent_errors", None)
    if isinstance(recent_errors, list):
        return any("permission denied" in str(error or "").lower() for error in recent_errors[-4:])
    return False
