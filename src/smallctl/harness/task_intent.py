from __future__ import annotations

import json
import re
from typing import Any

from ..state import clip_string_list, clip_text_value

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


def extract_intent_state(harness: Any, task: str) -> tuple[str, list[str], list[str]]:
    text = (task or "").lower()
    secondary: list[str] = []
    tags: list[str] = []
    requested_tool = infer_requested_tool_name(harness, task)

    primary = "general_task"
    if requested_tool:
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
    tags = [harness.provider_profile, harness.state.current_phase]
    cwd = harness.state.cwd.lower()
    if "localhost" in cwd:
        tags.append("localhost")
    if "scripts" in cwd:
        tags.append("scripts")
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
    text = (task or "").lower()
    creation_markers = (
        "build",
        "create",
        "make",
        "write",
        "generate",
        "implement",
        "save",
        "produce",
    )
    if (
        "script" in text
        and any(marker in text for marker in creation_markers)
    ) or re.search(r"\b(?:\.py|\.sh|\.bash|\.ps1)\b", text):
        return "write_file"
    if "file_patch" in text or "patch file" in text:
        return "file_patch"
    if any(marker in text for marker in _MEMORY_MARKERS):
        return "memory_update"
    if "read_file" in text or "cat" in text:
        return "read_file"
    if "write_file" in text:
        return "write_file"
    if (
        _REMOTE_TASK_HINT_RE.search(task or "")
        or _IPV4_RE.search(task or "")
        or _USER_AT_HOST_RE.search(task or "")
    ):
        return "ssh_exec"
    if hasattr(harness, "_looks_like_shell_request") and harness._looks_like_shell_request(task):
        return "shell_exec"
    return ""


def extract_memory_payload(task: str) -> str:
    text = (task or "").strip()
    if not text:
        return ""

    lowered = text.lower()
    for prefix in _MEMORY_PREFIXES:
        start = lowered.find(prefix)
        if start == -1:
            continue
        remainder = text[start + len(prefix) :].strip()
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
    return f"{harness.state.current_phase}: gather the next missing fact for {clip_text_value(task, limit=40)[0]}"


def derive_task_contract(task: str) -> str:
    lowered = (task or "").lower()
    memory_hint = memory_fact_hint(task)
    if memory_hint:
        return f"memory_update known_facts: {memory_hint}"
    if "contract" in lowered or "plan" in lowered:
        return "high_fidelity"
    return "general"
