from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

MEMORY_NAMESPACES = {
    "chat",
    "coding",
    "local_shell",
    "ssh_remote",
    "debugging",
    "planning",
    "incidents",
}

_LOCAL_SHELL_TOOL_NAMES = {"shell_exec", "bash_exec"}
_REMOTE_TOOL_NAMES = {"ssh_exec"}
_CODING_TOOL_NAMES = {
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
    "file_delete",
    "write_file",
}
_READ_TOOL_NAMES = {
    "file_read",
    "read_file",
    "artifact_read",
    "artifact_grep",
    "dir_list",
    "grep",
}
_PLANNING_TOOL_NAMES = {
    "ask_human",
    "loop_status",
    "plan_request_execution",
}
_TERMINAL_TOOL_NAMES = {"task_complete", "task_fail"}
_DEBUG_HINTS = (
    "debug",
    "debugging",
    "inspect",
    "trace",
    "stack trace",
    "error",
    "failed",
    "failure",
    "broken",
    "bug",
    "logs",
    "log ",
    "test failure",
)
_INCIDENT_HINTS = (
    "service down",
    "failed deploy",
    "restart",
    "recover",
    "restore",
    "outage",
    "broken host",
    "unreachable host",
    "daemon",
    "package remediation",
    "remediation",
)
_REMOTE_HINTS = ("remote", "ssh", "host", "server", "unreachable")
_CHAT_HINTS = ("chat_completed", "friendly chat", "greeting", "conversational")
_INTENT_NAMESPACE_OVERRIDES = {
    "plan_execution": "planning",
    "inspect_repo": "coding",
    "author_write": "coding",
    "write_file": "coding",
    "mutate_repo": "coding",
    "requested_file_write": "coding",
    "requested_file_append": "coding",
    "requested_file_patch": "coding",
    "requested_ast_patch": "coding",
    "requested_file_delete": "coding",
    "requested_write_file": "coding",
    "requested_shell_exec": "local_shell",
    "requested_bash_exec": "local_shell",
    "requested_ssh_exec": "ssh_remote",
}


@dataclass(frozen=True)
class NamespaceRouting:
    preferred: frozenset[str] = field(default_factory=frozenset)
    allowed: frozenset[str] = field(default_factory=frozenset)
    blocked: frozenset[str] = field(default_factory=frozenset)
    preferred_bonus: float = 9.0
    allowed_bonus: float = 0.5
    neutral_penalty: float = -3.0


def normalize_memory_namespace(value: str) -> str:
    namespace = str(value or "").strip().lower()
    if namespace not in MEMORY_NAMESPACES:
        return ""
    return namespace


def infer_memory_namespace(
    *,
    task_mode: str = "",
    tool_name: str = "",
    intent: str = "",
    intent_tags: Iterable[str] | None = None,
    environment_tags: Iterable[str] | None = None,
    entity_tags: Iterable[str] | None = None,
    notes: str = "",
    original_task: str = "",
) -> str:
    normalized_task_mode = str(task_mode or "").strip().lower()
    normalized_tool_name = str(tool_name or "").strip().lower()
    normalized_intent = str(intent or "").strip().lower()
    text = " ".join(
        part
        for part in (
            normalized_intent,
            notes,
            original_task,
            " ".join(_normalize_tags(intent_tags)),
            " ".join(_normalize_tags(environment_tags)),
            " ".join(_normalize_tags(entity_tags)),
        )
        if str(part or "").strip()
    ).lower()

    task_namespace = _default_namespace_for_task_mode(normalized_task_mode, text=text)
    if normalized_tool_name in _TERMINAL_TOOL_NAMES:
        if _looks_like_incident(text=text, task_mode=normalized_task_mode):
            return "incidents"
        return task_namespace or _namespace_from_intent(normalized_intent, text=text) or ""

    tool_namespace = _namespace_from_tool_name(
        normalized_tool_name,
        task_mode=normalized_task_mode,
        text=text,
    )
    if tool_namespace:
        return tool_namespace

    if task_namespace == "ssh_remote" and _looks_like_incident(text=text, task_mode=normalized_task_mode):
        return "incidents"
    if task_namespace:
        return task_namespace

    intent_namespace = _namespace_from_intent(normalized_intent, text=text)
    if intent_namespace == "ssh_remote" and _looks_like_incident(text=text, task_mode=normalized_task_mode):
        return "incidents"
    if intent_namespace:
        return intent_namespace

    if _looks_like_incident(text=text, task_mode=normalized_task_mode):
        return "incidents"
    if _looks_like_chat_memory(text):
        return "chat"
    return ""


def namespace_preferences_for_task_mode(
    task_mode: str,
    *,
    available_namespaces: set[str] | None = None,
) -> NamespaceRouting:
    normalized_task_mode = str(task_mode or "").strip().lower()
    available = {
        namespace
        for namespace in (available_namespaces or set())
        if normalize_memory_namespace(namespace)
    }

    if normalized_task_mode == "chat":
        return NamespaceRouting(
            preferred=frozenset({"chat"}),
            allowed=frozenset({"planning"}),
            blocked=frozenset({"local_shell", "ssh_remote", "incidents"}),
        )
    if normalized_task_mode == "analysis":
        return NamespaceRouting(
            preferred=frozenset({"coding", "debugging"}),
            allowed=frozenset({"planning"}),
        )
    if normalized_task_mode == "plan_only":
        blocked = {"local_shell", "ssh_remote", "incidents"} if "planning" in available else set()
        return NamespaceRouting(
            preferred=frozenset({"planning"}),
            allowed=frozenset({"debugging"}),
            blocked=frozenset(blocked),
        )
    if normalized_task_mode == "local_execute":
        return NamespaceRouting(
            preferred=frozenset({"local_shell"}),
            allowed=frozenset({"coding", "debugging"}),
        )
    if normalized_task_mode == "remote_execute":
        return NamespaceRouting(
            preferred=frozenset({"ssh_remote"}),
            allowed=frozenset({"debugging", "incidents"}),
        )
    if normalized_task_mode == "debug_inspect":
        return NamespaceRouting(
            preferred=frozenset({"debugging"}),
            allowed=frozenset({"coding", "planning", "incidents"}),
        )
    return NamespaceRouting(neutral_penalty=-1.0)


def namespace_bucket(namespace: str, routing: NamespaceRouting) -> str:
    normalized = normalize_memory_namespace(namespace)
    if normalized and normalized in routing.blocked:
        return "blocked"
    if normalized and normalized in routing.preferred:
        return "preferred"
    if normalized and normalized in routing.allowed:
        return "allowed"
    return "neutral"


def _normalize_tags(values: Iterable[str] | None) -> list[str]:
    return [str(value or "").strip().lower() for value in (values or []) if str(value or "").strip()]


def _default_namespace_for_task_mode(task_mode: str, *, text: str) -> str:
    if task_mode == "chat":
        return "chat"
    if task_mode == "analysis":
        return "debugging" if _looks_like_debugging(text) else "coding"
    if task_mode == "plan_only":
        return "planning"
    if task_mode == "local_execute":
        return "local_shell"
    if task_mode == "remote_execute":
        return "incidents" if _looks_like_incident(text=text, task_mode=task_mode) else "ssh_remote"
    if task_mode == "debug_inspect":
        return "debugging"
    return ""


def _namespace_from_tool_name(tool_name: str, *, task_mode: str, text: str) -> str:
    if tool_name in _REMOTE_TOOL_NAMES:
        return "ssh_remote"
    if tool_name in _LOCAL_SHELL_TOOL_NAMES:
        return "local_shell"
    if tool_name in _CODING_TOOL_NAMES:
        return "coding"
    if tool_name in _READ_TOOL_NAMES:
        return "debugging" if task_mode == "debug_inspect" or _looks_like_debugging(text) else "coding"
    if tool_name in _PLANNING_TOOL_NAMES:
        return "planning"
    return ""


def _namespace_from_intent(intent: str, *, text: str) -> str:
    direct = _INTENT_NAMESPACE_OVERRIDES.get(intent)
    if direct:
        return direct
    if "plan" in intent:
        return "planning"
    if "debug" in intent or "inspect" in intent:
        return "debugging"
    if "chat" in intent or (intent == "general_task" and _looks_like_chat_memory(text)):
        return "chat"
    return ""


def _looks_like_debugging(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _DEBUG_HINTS)


def _looks_like_incident(*, text: str, task_mode: str) -> bool:
    lowered = str(text or "").lower()
    if not any(marker in lowered for marker in _INCIDENT_HINTS):
        return False
    if task_mode == "remote_execute":
        return True
    return any(marker in lowered for marker in _REMOTE_HINTS)


def _looks_like_chat_memory(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _CHAT_HINTS)
