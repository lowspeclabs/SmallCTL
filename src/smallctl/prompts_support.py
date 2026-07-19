from __future__ import annotations

import re
from typing import Any

from .prompt_fragments import (
    _REMOTE_CLEANUP_ACTION_KEYWORDS,
    _REMOTE_CLEANUP_OBJECT_KEYWORDS,
    _REMOTE_CLEANUP_SCOPE_KEYWORDS,
    _REMOTE_CLEANUP_SELF_EVIDENT_KEYWORDS,
)
from .state import LoopState, normalize_intent_label


def _keyword_pattern(keywords: tuple[str, ...]) -> str:
    return "|".join(re.escape(keyword).replace("\\ ", r"[ \t]+") for keyword in keywords)


_REMOTE_CLEANUP_SCOPE_RE = re.compile(
    rf"\b(?:{_keyword_pattern(_REMOTE_CLEANUP_SCOPE_KEYWORDS)})\b", re.IGNORECASE
)
_REMOTE_CLEANUP_SELF_EVIDENT_RE = re.compile(
    rf"\b(?:{_keyword_pattern(_REMOTE_CLEANUP_SELF_EVIDENT_KEYWORDS)})\b", re.IGNORECASE
)
_REMOTE_CLEANUP_ACTION_RE = re.compile(
    rf"\b(?:{_keyword_pattern(_REMOTE_CLEANUP_ACTION_KEYWORDS)})\b", re.IGNORECASE
)
_REMOTE_CLEANUP_OBJECT_RE = re.compile(
    rf"\b(?:{_keyword_pattern(_REMOTE_CLEANUP_OBJECT_KEYWORDS)})\b", re.IGNORECASE
)
_REMOTE_CLEANUP_PROXIMITY_WINDOW = 48


def _readonly_lookup_hint(state: LoopState) -> str | None:
    """Return a prompt hint for answer-only / research tasks."""
    active_intent = normalize_intent_label(getattr(state, "active_intent", "") or "")
    if active_intent != "readonly_lookup":
        return None
    return (
        "ANSWER-ONLY TASK: Your goal is to gather evidence and return a clear, "
        "self-contained answer. When you have enough information, call "
        "`task_complete(message='...')` with the final answer in the message "
        "field. Do not end with plain chat text; wrap the answer in task_complete."
    )


def _graph_step_budget_prompt(scratchpad: Any) -> str | None:
    if not isinstance(scratchpad, dict):
        return None
    try:
        remaining = int(scratchpad.get("_graph_steps_remaining"))
        limit = int(scratchpad.get("_graph_recursion_limit"))
    except (TypeError, ValueError):
        return None
    if remaining <= 0 or limit <= 0:
        return None
    return (
        f"STEP BUDGET: You have approximately {remaining} graph steps remaining out of {limit}. "
        "If you already wrote and verified the file once, do not re-verify unless the latest test output shows a real failure. "
        "When evidence is sufficient, call `task_complete`; when blocked by repeated identical checks, call `task_fail` or escalate instead of looping. "
    )


def _phase_contract_prompt(state: LoopState, available_tool_names: list[str] | None) -> str | None:
    scratchpad = getattr(state, "scratchpad", {})
    existing_contract = scratchpad.get("_phase_contract") if isinstance(scratchpad, dict) else None
    run_brief = getattr(state, "run_brief", None)
    wm = getattr(state, "working_memory", None)
    text = " ".join(
        part
        for part in (
            str(getattr(run_brief, "original_task", "") or ""),
            str(getattr(run_brief, "task_contract", "") or ""),
            " ".join(str(item or "") for item in getattr(run_brief, "acceptance_criteria", []) or []),
            str(getattr(wm, "current_goal", "") or ""),
        )
        if part
    ).lower()
    phased_task = bool(existing_contract) or bool(
        text and any(marker in text for marker in ("phase 1", "phase 2", "phase 3", "phase 4", "phase 5", "phased", "multi-phase", "multiphase"))
    )
    if not phased_task:
        return None
    has_tool = bool(available_tool_names and "phase_contract_update" in available_tool_names)
    setup = ""
    if has_tool and not existing_contract:
        setup = "After reading the spec or roadmap, call `phase_contract_update` to create the contract before implementing or promoting phases. "
    elif has_tool and existing_contract:
        setup = "Use `loop_status` to inspect the active phase contract; call `phase_contract_update` only when the spec or active phase changes. "
    return (
        "PHASED CODING CONTRACT: For phased rollouts, each phase needs expected files, required symbols, checks, and a promotion quality. "
        f"{setup}Do not call `task_complete` for a phase based only on `py_compile`, import-only checks, dependency setup, cleanup commands, or an interactive-loop timeout. "
        "Promotion requires a behavioral verifier that imports the changed code and asserts phase-specific behavior."
    )


def _state_has_remote_cleanup_intent(state: LoopState) -> bool:
    run_brief = getattr(state, "run_brief", None)
    wm = getattr(state, "working_memory", None)
    text = " ".join(
        part
        for part in (
            str(getattr(run_brief, "original_task", "") or ""),
            str(getattr(wm, "current_goal", "") or ""),
        )
        if part
    ).lower()
    if not text:
        return False
    # Require explicit remote scope (ssh/host/server/remote context) so that
    # bare cleanup verbs in unrelated local tasks do not activate the playbook.
    if not _REMOTE_CLEANUP_SCOPE_RE.search(text):
        return False
    # Package-removal verbs are cleanup objects in their own right.
    if _REMOTE_CLEANUP_SELF_EVIDENT_RE.search(text):
        return True
    # Otherwise require a cleanup action near a cleanup object
    # (service/package/user/container removal).
    for match in _REMOTE_CLEANUP_ACTION_RE.finditer(text):
        window = text[
            max(0, match.start() - _REMOTE_CLEANUP_PROXIMITY_WINDOW) : match.end() + _REMOTE_CLEANUP_PROXIMITY_WINDOW
        ]
        if _REMOTE_CLEANUP_OBJECT_RE.search(window):
            return True
    return False


def _is_write_first_task(state: LoopState) -> bool:
    if normalize_intent_label(getattr(state, "active_intent", "")) in {"author_write", "requested_write_file"}:
        return True
    intent_tags = set(getattr(state, "intent_tags", []) or [])
    if "write_file" in intent_tags:
        return True
    task = (getattr(state.run_brief, "original_task", "") or "").lower()
    return bool(task and any(marker in task for marker in ("build a python script", "create a python script", "write a python script")))


def _render_plan_step(step: Any, *, depth: int = 0) -> list[str]:
    indent = "  " * depth
    lines = [f"{indent}- [{step.status}] {step.step_id} {step.title}".strip()]
    if getattr(step, "description", ""):
        lines.append(f"{indent}  {step.description}")
    for note in getattr(step, "notes", []) or []:
        lines.append(f"{indent}  note: {note}")
    for substep in getattr(step, "substeps", []) or []:
        lines.extend(_render_plan_step(substep, depth=depth + 1))
    return lines
