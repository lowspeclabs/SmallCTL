from __future__ import annotations

import re
from typing import Any

from ..state import LoopState
from .verifier_quality import (
    phase_verifier_is_inconclusive,
    verifier_notes_text,
    verifier_quality as _verifier_quality,
)


def phase_promotion_gate_block(
    state: LoopState,
    *,
    message: str,
    verifier_verdict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not looks_like_phase_coding_task(state, message=message):
        return None
    verifier = verifier_verdict if isinstance(verifier_verdict, dict) else {}
    verdict = str(verifier.get("verdict") or "").strip().lower()
    command = str(verifier.get("command") or verifier.get("target") or "").strip()
    failure_mode = str(verifier.get("failure_mode") or "").strip().lower()
    notes = verifier_notes_text(verifier)
    verifier_quality = _verifier_quality(command)

    if verdict != "pass":
        if verdict or failure_mode or notes:
            return phase_gate_payload(
                state,
                verifier,
                reason="phase_promotion_verifier_not_passing",
                verifier_quality=verifier_quality,
                notes=[
                    "Phase promotion requires a passing behavioral verifier, not a failed or inconclusive latest check.",
                    "Fix the first failing phase behavior, then rerun the focused smoke verifier.",
                ],
            )
        return phase_gate_payload(
            state,
            verifier,
            reason="phase_promotion_verifier_missing",
            verifier_quality=verifier_quality,
            notes=[
                "Phase promotion requires a passing behavioral verifier before task_complete.",
                "Run a focused smoke verifier that imports the changed module and exercises the phase behavior.",
            ],
        )

    if verifier_quality["score"] < 3 or phase_verifier_is_inconclusive(verifier, command=command, failure_mode=failure_mode, notes=notes):
        return phase_gate_payload(
            state,
            verifier,
            reason="phase_promotion_behavioral_verifier_required",
            verifier_quality=verifier_quality,
            notes=[
                "The latest passing verifier is too weak for phase promotion.",
                "Verifier quality must be at least `behavioral` for phase promotion.",
                "Syntax checks, import-only checks, dependency setup, cleanup commands, and interactive-loop timeouts do not prove the phase behavior works.",
                "Run a small behavioral smoke verifier that imports the code and asserts the phase-specific behavior without waiting on an interactive loop.",
            ],
        )
    return None


def looks_like_phase_coding_task(state: LoopState, *, message: str) -> bool:
    progress = getattr(state, "challenge_progress", None)
    category = str(getattr(progress, "task_category", "") or "").strip().lower()
    code_change_count = int(getattr(progress, "code_change_count", 0) or 0) if progress is not None else 0
    run_brief = getattr(state, "run_brief", None)
    text_parts = [
        message,
        str(getattr(run_brief, "original_task", "") or ""),
        " ".join(str(item or "") for item in getattr(run_brief, "acceptance_criteria", []) or []),
    ]
    scratchpad = getattr(state, "scratchpad", {})
    if isinstance(scratchpad, dict):
        ledger = scratchpad.get("subtask_ledger")
        if isinstance(ledger, dict):
            active = ledger.get("active_subtask")
            if isinstance(active, dict):
                text_parts.extend([str(active.get("goal") or ""), str(active.get("title") or "")])
    text = "\n".join(part for part in text_parts if part).lower()
    phase_like = bool(re.search(r"\bphase\s*\d+\b|\bphase\b|\bmulti[- ]?phase\b", text))
    return phase_like and (category == "coding" or code_change_count > 0)


def task_involves_interactive_program(state: LoopState) -> bool:
    """Detect if the current task involves an interactive/GUI program like pygame."""
    run_brief = getattr(state, "run_brief", None)
    task_text = str(getattr(run_brief, "original_task", "") or "").lower()
    working_memory = getattr(state, "working_memory", None)
    goal_text = str(getattr(working_memory, "current_goal", "") or "").lower()
    haystack = f"{task_text} {goal_text}"
    interactive_markers = (
        "pygame", "gui", "interactive", "game loop", "event loop",
        "tkinter", "qt", "pyside", "kivy", "arcade", "curses",
        "real-time", "realtime", "animation", "render loop",
    )
    return any(marker in haystack for marker in interactive_markers)


def mutation_expectation_block(state: LoopState, *, message: str) -> dict[str, Any] | None:
    progress = getattr(state, "challenge_progress", None)
    code_change_count = int(getattr(progress, "code_change_count", 0) or 0) if progress is not None else 0
    if code_change_count > 0:
        return None
    if not looks_like_phase_coding_task(state, message=message):
        return None

    run_brief = getattr(state, "run_brief", None)
    working_memory = getattr(state, "working_memory", None)
    text = "\n".join(
        str(part or "")
        for part in [
            getattr(run_brief, "original_task", ""),
            getattr(run_brief, "current_phase_objective", ""),
            getattr(working_memory, "current_goal", ""),
            message,
        ]
        if str(part or "").strip()
    ).lower()
    mutation_expected = bool(
        re.search(r"\b(begin|implement|continue|advance|start)\b.{0,80}\bphase\s*\d+\b", text)
        or re.search(r"\bphase\s*\d+\b.{0,80}\b(begin|implement|continue|advance|start)\b", text)
    )
    if not mutation_expected:
        return None

    completion_text = str(message or "").lower()
    explicit_no_change = any(
        marker in completion_text
        for marker in (
            "no change required",
            "no code change required",
            "diagnostic only",
            "verification only",
            "asked only to verify",
            "user requested no changes",
        )
    )
    if explicit_no_change:
        return None

    return {
        "reason": "mutation_expected_but_no_code_changes",
        "code_change_count": 0,
        "next_required_action": {
            "tool_names": ["file_patch", "file_write", "ast_patch", "ask_human", "task_fail"],
            "notes": [
                "The task asks to begin or implement a coding phase, but no code changes have been made.",
                "Make the first concrete phase change, ask the user if the phase target is ambiguous, or fail if blocked.",
                "Complete with zero changes only when the response explicitly proves no mutation was required.",
            ],
        },
    }


def phase_gate_payload(
    state: LoopState,
    verifier: dict[str, Any],
    *,
    reason: str,
    notes: list[str],
    verifier_quality: dict[str, Any] | None = None,
) -> dict[str, Any]:
    planning_mode = bool(getattr(state, "planning_mode_enabled", False))
    next_action: dict[str, Any] = {
        "tool_name": "request_validation_execution" if planning_mode else "shell_exec",
        "notes": notes,
    }
    if planning_mode:
        next_action["notes"] = [
            *notes,
            "Planning mode cannot execute the verifier directly; request a validation handoff instead of calling `run` or `shell_exec`.",
        ]
    return {
        "reason": reason,
        "last_verifier_verdict": verifier or None,
        "verifier_quality": verifier_quality or {"score": 0, "label": "none"},
        "required_verifier_quality": {"score": 3, "label": "behavioral"},
        "next_required_action": next_action,
    }
