from __future__ import annotations

import re
from typing import Any

_SSH_USERNAME_TASK_PATTERNS = (
    re.compile(r"\busername\s+(?:is\s+)?(?P<user>[A-Za-z0-9._-]+)\b", re.IGNORECASE),
    re.compile(r"\buser\s+(?:is\s+)?(?P<user>[A-Za-z0-9._-]+)\b", re.IGNORECASE),
)
_SSH_PASSWORD_TASK_PATTERNS = (
    re.compile(r'\bpassword\s*(?:(?:is\s+)|[=:]\s*)?"(?P<password>[^"\r\n]+)"', re.IGNORECASE),
    re.compile(r"\bpassword\s*(?:(?:is\s+)|[=:]\s*)?'(?P<password>[^'\r\n]+)'", re.IGNORECASE),
    re.compile(r"\bpassword\s*(?:(?:is\s+)|[=:]\s*)?(?P<password>[^\s,;]+)", re.IGNORECASE),
)
_SSH_PASSWORD_INVALID_TOKENS = {
    "authentication",
    "auth",
    "enabled",
    "required",
    "prompt",
    "prompted",
    "[REDACTED]",
}


def infer_ssh_user_from_state_context(host: str, *, state: Any | None = None) -> str:
    target_host = str(host or "").strip().lower()
    if not target_host or state is None:
        return ""

    for text in ssh_task_context_texts(state):
        if not text:
            continue
        embedded_match = re.search(
            rf"\b(?P<user>[A-Za-z0-9._-]+)@{re.escape(target_host)}\b",
            text,
            re.IGNORECASE,
        )
        if embedded_match is not None:
            return str(embedded_match.group("user") or "").strip()

        lowered = text.lower()
        if target_host not in lowered:
            continue
        for pattern in _SSH_USERNAME_TASK_PATTERNS:
            match = pattern.search(text)
            if match is not None:
                return str(match.group("user") or "").strip()
    return ""


def infer_ssh_password_from_state_context(
    host: str,
    *,
    user: str | None = None,
    state: Any | None = None,
) -> str:
    target_host = str(host or "").strip().lower()
    target_user = str(user or "").strip().lower()
    if not target_host or state is None:
        return ""

    for text in ssh_task_context_texts(state):
        if not text_mentions_ssh_target(text, host=target_host, user=target_user):
            continue
        for pattern in _SSH_PASSWORD_TASK_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            candidate = str(match.group("password") or "").strip()
            if looks_like_ssh_password(candidate):
                return candidate
    return ""


def text_mentions_ssh_target(text: str, *, host: str, user: str | None = None) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    if host and host in lowered:
        return True
    normalized_user = str(user or "").strip().lower()
    return bool(host and normalized_user and f"{normalized_user}@{host}" in lowered)


def looks_like_ssh_password(candidate: str) -> bool:
    stripped = str(candidate or "").strip()
    if not stripped:
        return False
    return stripped.lower() not in _SSH_PASSWORD_INVALID_TOKENS


def ssh_task_context_texts(state: Any) -> list[str]:
    texts: list[str] = []

    run_brief = getattr(state, "run_brief", None)
    original_task = str(getattr(run_brief, "original_task", "") or "").strip()
    if original_task:
        texts.append(original_task)

    current_goal = str(getattr(getattr(state, "working_memory", None), "current_goal", "") or "").strip()
    if current_goal:
        texts.append(current_goal)

    for message in getattr(state, "recent_messages", []) or []:
        if str(getattr(message, "role", "") or "").strip().lower() != "user":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            texts.append(content)

    return texts
