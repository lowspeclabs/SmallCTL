from __future__ import annotations

import json
import re
from typing import Any

from ..experience_tags import PHASE_TAG_PREFIX
from ..state import clip_string_list, clip_text_value
from .task_classifier import (
    classify_runtime_intent,
    is_smalltalk,
    looks_like_author_write_request,
    looks_like_write_file_request,
    looks_like_write_patch_request,
)
from .task_classifier_constants import LOCAL_SHELL_OVERRIDE_RE as _LOCAL_SHELL_OVERRIDE_RE
from .task_classifier_constants import READONLY_SUGGESTION_MARKERS
from .task_classifier_support import task_has_local_command_target, task_is_local_ssh_file_target, task_is_local_system_target

_MEMORY_MARKERS = (
    "save this in memory",
    "save memory",
    "remember this",
    "store this in memory",
    "store this",
    "note this",
    "pin this",
    "persist this",
    "keep this in memory",
    "write this down",
)

_MEMORY_PREFIXES = (
    "save this in memory",
    "store this in memory",
    "keep this in memory",
    "remember this",
    "note this",
    "pin this",
    "persist this",
    "write this down",
    "save memory",
    "store this",
    "remember",
    "note",
    "pin",
    "persist",
    "keep",
    "write down",
    "save",
    "store",
)

_REMOTE_TASK_HINT_RE = re.compile(r"\b(?:remote|ssh|server|host|username|password)\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_USER_AT_HOST_RE = re.compile(r"\b[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\b")
_REMOTE_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|never|don't|dont)\b.*\b(?:connect\s+to|ssh\s+to|use)\b.*\b(?:remote|ssh|server|host)\b"
    r"|"
    r"\b(?:remote|ssh|server|host)\b.*\b(?:do\s+not|never|don't|dont)\b.*\b(?:connect\s+to|ssh\s+to|use)\b",
    re.IGNORECASE,
)
_TRACEBACK_HEADER_RE = re.compile(r"Traceback\s*\(most recent call last\):", re.IGNORECASE)
_TERMINAL_PROMPT_RE = re.compile(r"\S+@[\w.-]+[:~]?[^\$]*\$?\s*")


_READONLY_INTENT_TOOLS = {
    "artifact_grep",
    "artifact_list",
    "artifact_read",
    "dir_list",
    "file_read",
    "git_diff",
    "git_status",
    "grep",
    "long_context_lookup",
    "loop_status",
    "read_file",
    "search",
    "ssh_file_read",
    "summarize_report",
    "web_fetch",
    "web_search",
}

_EXECUTION_INTENT_TOOLS = {
    "shell_exec",
    "ssh_exec",
    "ssh_session_start",
    "ssh_session_send",
    "ssh_session_send_and_read",
}

_MUTATION_INTENT_TOOLS = {
    "ast_patch",
    "file_append",
    "file_patch",
    "file_write",
    "memory_update",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "ssh_file_write",
}

_WEAK_RUNTIME_INTENTS = {
    "",
    "chat_only",
    "content_lookup",
    "general_task",
    "inspect_repo",
    "readonly_lookup",
}


def _phase_tag(state: Any) -> str:
    phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    return f"{PHASE_TAG_PREFIX}{phase}" if phase else ""


def _promoted_intent_for_tool(tool_name: str) -> tuple[str, list[str], list[str]] | None:
    normalized = str(tool_name or "").strip()
    if not normalized or normalized in _READONLY_INTENT_TOOLS:
        return None
    if normalized in {"task_complete", "task_fail", "ask_human"}:
        return None
    if normalized in _EXECUTION_INTENT_TOOLS:
        return f"requested_{normalized}", ["complete_validation_task"], [normalized, "execute"]
    if normalized in _MUTATION_INTENT_TOOLS:
        tags = [normalized, "mutate_repo"]
        secondary = ["mutate_repo", "complete_validation_task"]
        return f"requested_{normalized}", secondary, tags
    return None


def promote_active_intent_for_tool_call(state: Any, tool_name: str) -> bool:
    promoted = _promoted_intent_for_tool(tool_name)
    if promoted is None:
        return False
    active_intent, secondary, tags = promoted
    current = str(getattr(state, "active_intent", "") or "").strip().lower()
    if current and current not in _WEAK_RUNTIME_INTENTS and current == active_intent:
        return False

    existing_tags = [str(tag).strip() for tag in (getattr(state, "intent_tags", []) or []) if str(tag).strip()]
    phase_tag = _phase_tag(state)
    merged_tags = []
    for tag in [*tags, phase_tag, *existing_tags]:
        if tag and tag not in merged_tags:
            merged_tags.append(tag)

    setattr(state, "active_intent", active_intent)
    setattr(state, "secondary_intents", clip_string_list(secondary, limit=3, item_char_limit=48)[0])
    setattr(state, "intent_tags", clip_string_list(merged_tags, limit=6, item_char_limit=64)[0])
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad["_active_intent_promoted_by_tool"] = {
            "active_intent": active_intent,
            "secondary_intents": list(getattr(state, "secondary_intents", []) or []),
            "intent_tags": list(getattr(state, "intent_tags", []) or []),
            "tool_name": str(tool_name or ""),
            "step_count": int(getattr(state, "step_count", 0) or 0),
        }
    return True


def preserve_promoted_active_intent(
    state: Any,
    derived_intent: str,
    derived_secondary: list[str],
    derived_tags: list[str],
) -> tuple[str, list[str], list[str]]:
    scratchpad = getattr(state, "scratchpad", None)
    marker = scratchpad.get("_active_intent_promoted_by_tool") if isinstance(scratchpad, dict) else None
    if not isinstance(marker, dict):
        return derived_intent, derived_secondary, derived_tags
    promoted_intent = str(marker.get("active_intent") or "").strip()
    if not promoted_intent:
        return derived_intent, derived_secondary, derived_tags
    if str(derived_intent or "").strip().lower() not in _WEAK_RUNTIME_INTENTS:
        return derived_intent, derived_secondary, derived_tags
    promoted_secondary = [
        str(item).strip()
        for item in (marker.get("secondary_intents") or [])
        if str(item).strip()
    ]
    promoted_tags = [
        str(item).strip()
        for item in (marker.get("intent_tags") or [])
        if str(item).strip()
    ]
    return promoted_intent, promoted_secondary, promoted_tags


def _sanitize_task_for_intent_routing(task: str) -> str:
    """Strip terminal prompts from tracebacks so they don't trigger ssh_exec intent."""
    if not task:
        return task
    if _TRACEBACK_HEADER_RE.search(task):
        return _TERMINAL_PROMPT_RE.sub("", task)
    return task
def extract_intent_state(harness: Any, task: str) -> tuple[str, list[str], list[str]]:
    text = (task or "").lower()
    secondary: list[str] = []
    tags: list[str] = []
    requested_tool = infer_requested_tool_name(harness, task)
    author_write_request = looks_like_author_write_request(task)

    # Runtime intent classification can override heuristic extraction for
    # answer-only / research-style tasks that would otherwise be misclassified
    # as execution (e.g. "do a websearch on X then respond back").
    runtime_intent = classify_runtime_intent(
        task,
        recent_messages=list(getattr(harness.state, "recent_messages", []) or []),
        pending_interrupt=getattr(harness.state, "pending_interrupt", None),
    )
    suggestion_only = any(marker in text for marker in READONLY_SUGGESTION_MARKERS)
    if suggestion_only and any(token in text for token in {"inspect", "read", "grep", "find", "search", "list"}):
        primary = "inspect_repo"
        secondary.append("read_artifacts")
    elif runtime_intent.label == "readonly_lookup" and not author_write_request:
        primary = "readonly_lookup"
        secondary.append("answer_only")
        secondary.append("complete_validation_task")
        tags.append("research")
    elif author_write_request:
        primary = "author_write"
        secondary.extend(["mutate_repo", "complete_validation_task"])
        tags.append("write_file")
    elif requested_tool:
        primary = f"requested_{requested_tool}"
        secondary.append("complete_validation_task")
        tags.append(requested_tool)
        if requested_tool == "loop_status":
            secondary.append("call_zero_arg_tool")
        if requested_tool in {"task_complete", "task_fail", "ask_human"}:
            secondary.append("control_tool")
    elif any(token in text for token in {"inspect", "read", "grep", "find", "search", "list"}):
        primary = "inspect_repo"
        secondary.append("read_artifacts")
    elif any(token in text for token in {"write", "edit", "patch", "create", "update", "diff"}):
        primary = "write_file"
        secondary.append("mutate_repo")
    elif "contract" in text or "plan" in text:
        primary = "plan_execution"
        secondary.append("complete_validation_task")
    else:
        primary = "general_task"

    if harness.state.working_memory.failures:
        secondary.append("recover_from_validation_error")

    tags.extend(infer_environment_tags(harness))
    tags.extend(infer_entity_tags(task))
    tags.extend([t for t in harness.state.working_memory.next_actions[-2:] if " " not in t][:2])

    return (
        primary,
        clip_string_list(secondary, limit=3, item_char_limit=48)[0],
        clip_string_list(tags, limit=6, item_char_limit=64)[0],
    )


def infer_environment_tags(harness: Any) -> list[str]:
    phase = str(getattr(harness.state, "current_phase", "") or "").strip().lower()
    tags: list[str] = []
    if phase:
        tags.append(f"{PHASE_TAG_PREFIX}{phase}")
    return tags


def infer_entity_tags(task: str) -> list[str]:
    text = (task or "").lower()
    tags = []
    if "python" in text or ".py" in text:
        tags.append("python")
    if "bash" in text or ".sh" in text:
        tags.append("bash")
    return tags


def infer_requested_tool_name(harness: Any, task: str) -> str:
    text = _sanitize_task_for_intent_routing(task).lower()
    if task_has_local_command_target(task):
        if looks_like_write_patch_request(task):
            return "file_patch"
        if looks_like_write_file_request(task) or looks_like_author_write_request(task):
            return "write_file"
        return "shell_exec"
    # Explicit local-shell override bypasses remote heuristic
    if _LOCAL_SHELL_OVERRIDE_RE.search(task):
        if hasattr(harness, "_looks_like_shell_request") and harness._looks_like_shell_request(task):
            return "shell_exec"
        return "shell_exec"
    from .task_classifier import task_is_local_coding_target
    if task_is_local_coding_target(task):
        if looks_like_write_patch_request(task):
            return "file_patch"
        if looks_like_write_file_request(task) or looks_like_author_write_request(task):
            return "write_file"
        if "file_patch" in text or "patch file" in text:
            return "file_patch"
        if "ast_patch" in text or "structural patch" in text:
            return "ast_patch"
        if any(marker in text for marker in _MEMORY_MARKERS):
            return "memory_update"
        if "read_file" in text or "cat" in text:
            return "read_file"
        if "write_file" in text:
            return "write_file"
        if hasattr(harness, "_looks_like_shell_request") and harness._looks_like_shell_request(task):
            return "shell_exec"
        return ""
    if task_is_local_ssh_file_target(task) or task_is_local_system_target(task):
        if any(marker in text for marker in ("remove", "delete", "clean", "cleanup", "edit", "patch", "update")):
            return "shell_exec"
        if any(marker in text for marker in ("find", "read", "show", "list", "where", "grep", "check")):
            return "read_file"
        if hasattr(harness, "_looks_like_shell_request") and harness._looks_like_shell_request(task):
            return "shell_exec"
        return ""
    remote_task = bool(
        (_REMOTE_TASK_HINT_RE.search(text)
         or _IPV4_RE.search(text)
         or _USER_AT_HOST_RE.search(text))
        and not _REMOTE_NEGATION_RE.search(text)
    )
    if remote_task:
        # Explicitly read-only / non-destructive remote tasks should map to ssh_exec,
        # even if words like "patch" or "replace" appear in instruction text.
        is_read_only_task = any(
            marker in text
            for marker in (
                "non-destructive",
                "read-only",
                "read only",
                "triage",
                "inspect",
                "investigate",
                "do not delete",
                "do not modify",
                "do not patch",
            )
        )
        if is_read_only_task:
            return "ssh_exec"
        if any(marker in text for marker in ("style block", "<style>", "</style>", "between ", "bounded block", "inline style")):
            return "ssh_file_replace_between"
        if any(marker in text for marker in ("read file", "read the file", "cat ", "inspect file", "inspect the file")):
            return "ssh_file_read"
        if any(marker in text for marker in ("write file", "create file", "overwrite", "save file")):
            return "ssh_file_write"
        if any(marker in text for marker in ("patch", "replace exact", "edit file", "edit the file", "replace ")):
            return "ssh_file_patch"
        return "ssh_exec"
    if looks_like_write_patch_request(task):
        return "file_patch"
    if looks_like_write_file_request(task) or looks_like_author_write_request(task):
        return "write_file"
    if "file_patch" in text or "patch file" in text:
        return "file_patch"
    if "ast_patch" in text or "structural patch" in text:
        return "ast_patch"
    if any(marker in text for marker in _MEMORY_MARKERS):
        return "memory_update"
    if "read_file" in text or "cat" in text:
        return "read_file"
    if "write_file" in text:
        return "write_file"
    if hasattr(harness, "_looks_like_shell_request") and harness._looks_like_shell_request(task):
        return "shell_exec"
    return ""


def extract_memory_payload(task: str) -> str:
    text = (task or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    for prefix in _MEMORY_PREFIXES:
        match = re.search(rf"(?<!\w){re.escape(prefix)}(?!\w)", lowered)
        if match is None:
            continue
        remainder = text[match.end() :].strip()
        remainder = remainder.lstrip(":-— ")
        if remainder:
            return remainder
    return ""


def memory_fact_hint(task: str) -> str:
    payload = extract_memory_payload(task)
    if not payload:
        return ""
    clipped, _ = clip_text_value(payload, limit=180)
    return clipped


def completion_next_action() -> str:
    return "Decide whether the current evidence is sufficient; call task_complete when it is."


def next_action_for_task(harness: Any, task: str) -> str:
    memory_hint = memory_fact_hint(task)
    if memory_hint:
        return (
            "Call `memory_update(section='known_facts', content="
            f"{json.dumps(memory_hint, ensure_ascii=True)})` to persist the fact."
        )
    if is_smalltalk(task):
        return completion_next_action()
    return f"{harness.state.current_phase}: gather the next missing fact for {clip_text_value(task, limit=40)[0]}"


def derive_task_contract(task: str) -> str:
    lowered = (task or "").lower()
    memory_hint = memory_fact_hint(task)
    if memory_hint:
        return f"memory_update known_facts: {memory_hint}"
    if "contract" in lowered or "plan" in lowered:
        return "high_fidelity"
    return "general"
