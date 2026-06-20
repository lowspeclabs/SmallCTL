from __future__ import annotations

import re
from pathlib import Path

from ..experience_tags import PHASE_TAG_PREFIX, is_generic_experience_tag
from ..normalization import tokenize as _tokens
from ..state import ExperienceMemory, LoopState, normalize_intent_label

CHAT_SUPPRESSED_MEMORY_TAGS = {
    "shell_exec",
    "ssh_exec",
    "scripts",
    "bash",
    "terminal",
    "command",
    "command_line",
}
LIVE_REMOTE_CORRECTION_PHRASES = (
    "actually use ssh",
    "do it live",
    "redo the remote action",
    "do not rely on past records",
    "don't rely on past records",
    "dont rely on past records",
    "do not rely on prior records",
    "don't rely on prior records",
    "re-run on the host",
    "rerun on the host",
    "run it live",
    "redo it live",
    "fresh ssh",
    "fresh run",
)
LIVE_REMOTE_CORRECTION_HINTS = (
    "actually",
    "again",
    "fresh",
    "live",
    "redo",
    "rerun",
    "re-run",
    "retry",
    "retest",
    "verify",
)


def normalized_goal_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def effective_current_goal(state: LoopState) -> str:
    current_goal = str(getattr(state.working_memory, "current_goal", "") or "").strip()
    if not current_goal:
        return ""
    previous_task = str(state.scratchpad.get("_task_boundary_previous_task") or "").strip()
    has_task_transaction = bool(state.scratchpad.get("_task_transaction"))
    if previous_task and normalized_goal_text(previous_task) == normalized_goal_text(current_goal):
        if not has_task_transaction:
            return ""
    return current_goal


def state_environment_tags(state: LoopState) -> set[str]:
    phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    if not phase:
        return set()
    return {f"{PHASE_TAG_PREFIX}{phase}"}


def is_generic_retrieval_tag(tag: str) -> bool:
    return is_generic_experience_tag(tag)


def prompt_visible_memory_tags(state: LoopState, memory: ExperienceMemory) -> list[str]:
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    active_intent = normalize_intent_label(getattr(state, "active_intent", "") or "")
    state_tags = {
        str(tag).strip().lower()
        for tag in (getattr(state, "intent_tags", []) or [])
        if str(tag).strip()
    }
    visible: list[str] = []
    for tag in getattr(memory, "intent_tags", []) or []:
        normalized = str(tag or "").strip()
        lowered = normalized.lower()
        if not normalized or is_generic_retrieval_tag(lowered):
            continue
        if task_mode == "chat" and lowered in CHAT_SUPPRESSED_MEMORY_TAGS:
            continue
        if lowered.startswith(PHASE_TAG_PREFIX):
            continue
        if lowered == active_intent or lowered in state_tags or lowered.endswith("_exec"):
            visible.append(normalized)
            continue
        if lowered.startswith(("task_", "tool_")):
            visible.append(normalized)
    return visible


def is_generic_terminal_memory(state: LoopState, memory: ExperienceMemory) -> bool:
    if str(memory.tool_name or "").strip().lower() != "task_complete":
        return False
    if normalize_intent_label(memory.intent) != "general_task":
        return False
    if normalize_intent_label(getattr(state, "active_intent", "") or "") != "general_task":
        return False
    if prompt_visible_memory_tags(state, memory):
        return False
    current_goal = effective_current_goal(state)
    if current_goal and (_tokens(current_goal) & _tokens(memory.notes or "")):
        return False
    return True


def query_requests_live_remote_correction(query_text: str) -> bool:
    text = re.sub(r"\s+", " ", str(query_text or "").strip().lower())
    if not text:
        return False
    if any(phrase in text for phrase in LIVE_REMOTE_CORRECTION_PHRASES):
        return True
    has_live_correction_language = any(marker in text for marker in LIVE_REMOTE_CORRECTION_HINTS)
    has_remote_anchor = any(token in text for token in ("ssh", "remote", "host", "server"))
    has_reliance_negation = any(
        phrase in text
        for phrase in (
            "don't rely",
            "do not rely",
            "dont rely",
            "do not trust",
            "don't trust",
        )
    )
    return has_live_correction_language and (has_remote_anchor or has_reliance_negation)


def is_model_terminal_claim(memory: ExperienceMemory) -> bool:
    return (
        str(memory.tool_name or "").strip().lower() == "task_complete"
        and str(getattr(memory, "source", "") or "").strip().lower() == "model_terminal_claim"
    )


def state_entity_tags(state: LoopState) -> set[str]:
    return _tokens(
        " ".join(
            filter(
                None,
                [
                    state.run_brief.original_task,
                    effective_current_goal(state),
                    " ".join(state.working_memory.open_questions),
                ],
            )
        )
    )


def state_target_paths(state: LoopState) -> set[str]:
    paths: set[str] = set()
    for value in list(getattr(state, "files_changed_this_cycle", []) or []):
        text = str(value or "").strip()
        if text:
            paths.add(Path(text).as_posix().lower())
    task_targets = state.scratchpad.get("_task_target_paths")
    if isinstance(task_targets, list):
        for value in task_targets:
            text = str(value or "").strip()
            if text:
                paths.add(Path(text).as_posix().lower())
    write_session = getattr(state, "write_session", None)
    if write_session is not None:
        for key in ("write_target_path", "write_staging_path"):
            text = str(getattr(write_session, key, "") or "").strip()
            if text:
                paths.add(Path(text).as_posix().lower())
    return paths


def state_touched_symbols(state: LoopState) -> set[str]:
    payload = state.scratchpad.get("_touched_symbols")
    if not isinstance(payload, list):
        return set()
    symbols: set[str] = set()
    for value in payload:
        text = str(value or "").strip()
        if not text:
            continue
        symbols |= _tokens(text)
    return symbols


def path_match(left: str, right: str) -> bool:
    lhs = str(left or "").strip().lower()
    rhs = str(right or "").strip().lower()
    if not lhs or not rhs:
        return False
    if lhs == rhs:
        return True
    return lhs.endswith(rhs) or rhs.endswith(lhs)


def durably_stale_ids(state: LoopState, *, key: str) -> set[str]:
    payload = state.scratchpad.get(key)
    if not isinstance(payload, dict):
        return set()
    ids: set[str] = set()
    for raw_id, marker in payload.items():
        item_id = str(raw_id or "").strip()
        if not item_id or not isinstance(marker, dict):
            continue
        if bool(marker.get("stale", False)):
            ids.add(item_id)
    return ids


def is_durably_stale_experience(state: LoopState, memory: ExperienceMemory) -> bool:
    memory_id = str(getattr(memory, "memory_id", "") or "").strip()
    if not memory_id:
        return False
    return memory_id in durably_stale_ids(state, key="_experience_staleness")
