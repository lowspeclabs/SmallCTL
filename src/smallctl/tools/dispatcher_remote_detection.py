from __future__ import annotations

from typing import Any

from .dispatcher_ssh_context import ssh_task_context_texts as _ssh_task_context_texts

_SSH_TASK_TARGET_RE = None
_AT_HOST_TARGET_RE = None
_IPV4_RE = None
_REMOTE_TASK_HINT_RE = None
_LOCAL_CLARIFICATION_RE = None


def _ensure_regexes():
    global _SSH_TASK_TARGET_RE, _AT_HOST_TARGET_RE, _IPV4_RE, _REMOTE_TASK_HINT_RE, _LOCAL_CLARIFICATION_RE
    if _SSH_TASK_TARGET_RE is None:
        import re
        _SSH_TASK_TARGET_RE = re.compile(
            r"\b(?:ssh|scp|sftp)\s+(?:[A-Za-z0-9._-]+@)?(?P<host>[A-Za-z0-9._-]+)\b",
            re.IGNORECASE,
        )
        _AT_HOST_TARGET_RE = re.compile(r"\b[A-Za-z0-9._-]+@(?P<host>[A-Za-z0-9._-]+)\b", re.IGNORECASE)
        _IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        _REMOTE_TASK_HINT_RE = re.compile(r"\b(?:remote|ssh|username|password|server|host)\b", re.IGNORECASE)
        _LOCAL_CLARIFICATION_RE = re.compile(
            r"\b(?:local(?:ly)?\s+(?:first|now|here|on\s+this\s+host|on\s+this\s+machine)|"
            r"on\s+this\s+host\s+(?:first|now)|"
            r"start\s+(?:with|on)\s+(?:the\s+)?local|"
            r"do\s+this\s+locally|"
            r"what\s+about\s+(?:the\s+)?local)\b",
            re.IGNORECASE,
        )


def task_clearly_targets_remote_ssh_host(state: Any | None) -> bool:
    if state is None:
        return False
    _ensure_regexes()
    # If the task classifier has already decided this is a local execution task
    # (e.g., using a local CLI/script against a remote API), do not let the
    # dispatcher override that decision and force ssh_exec.
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    if task_mode == "local_execute":
        return False
    for text in _ssh_task_context_texts(state):
        if not text:
            continue
        # If the user explicitly asked for local operations, do not classify as remote
        if _LOCAL_CLARIFICATION_RE.search(text) is not None:
            return False
        if _SSH_TASK_TARGET_RE.search(text) is not None:
            return True
        if _AT_HOST_TARGET_RE.search(text) is not None:
            return True
        if _IPV4_RE.search(text) is not None and _REMOTE_TASK_HINT_RE.search(text) is not None:
            return True
        lowered = text.lower()
        if "connect to " in lowered and (
            _IPV4_RE.search(text) is not None or _SSH_TASK_TARGET_RE.search(text) is not None
        ):
            return True
    return False


def task_requests_ssh_connection_probe(state: Any | None) -> bool:
    if state is None:
        return False
    for text in _ssh_task_context_texts(state):
        lowered = str(text or "").strip().lower()
        if not lowered:
            continue
        if any(
            marker in lowered
            for marker in (
                "ssh into ",
                "ssh to ",
                "ssh in to ",
                "log into ",
                "login to ",
                "connect to ",
            )
        ) and "?" not in lowered:
            return True
    return False
