from __future__ import annotations

from typing import Any

from ..state import LoopState
from .control_objectives import (
    clean_objective_title,
    extract_multi_objectives,
    objective_matches_text,
)

MULTI_OBJECTIVE_LEDGER_KEY = "_multi_objective_ledger"


def candidate_multi_objective_texts(state: LoopState) -> list[str]:
    texts: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in texts:
            texts.append(text)

    add(getattr(state.run_brief, "original_task", ""))
    add(getattr(state.run_brief, "current_phase_objective", ""))
    add(getattr(state.working_memory, "current_goal", ""))
    scratchpad = getattr(state, "scratchpad", {})
    if isinstance(scratchpad, dict):
        for key in ("_task_transaction", "_last_task_handoff"):
            payload = scratchpad.get(key)
            if not isinstance(payload, dict):
                continue
            for field in ("raw_task", "effective_task", "current_goal", "user_goal"):
                add(payload.get(field))
    return texts


def ensure_multi_objective_ledger(state: LoopState) -> dict[str, Any] | None:
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return None
    existing = scratchpad.get(MULTI_OBJECTIVE_LEDGER_KEY)
    if isinstance(existing, dict) and isinstance(existing.get("objectives"), list):
        return existing

    best_text = ""
    best_objectives: list[str] = []
    for text in candidate_multi_objective_texts(state):
        objectives = extract_multi_objectives(text)
        if len(objectives) > len(best_objectives):
            best_text = text
            best_objectives = objectives
    if len(best_objectives) < 2:
        return None

    ledger = {
        "status": "active",
        "parent_goal": clean_objective_title(best_text)[:500],
        "objectives": [
            {
                "objective_id": f"O{index}",
                "title": title,
                "status": "pending",
                "evidence": [],
            }
            for index, title in enumerate(best_objectives, start=1)
        ],
    }
    scratchpad[MULTI_OBJECTIVE_LEDGER_KEY] = ledger
    return ledger


def resolved_followup_objective_id(state: LoopState) -> str:
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return ""
    resolved = scratchpad.get("_resolved_followup")
    if not isinstance(resolved, dict):
        return ""
    try:
        index = int(resolved.get("option_index") or 0)
    except (TypeError, ValueError):
        return ""
    return f"O{index}" if index > 0 else ""


def mark_objective_done(objective: dict[str, Any], message: str) -> bool:
    if str(objective.get("status") or "").strip().lower() == "done":
        return False
    objective["status"] = "done"
    evidence = str(message or "").strip()
    if evidence:
        current = objective.get("evidence")
        if not isinstance(current, list):
            current = []
        current.append(evidence[:240])
        objective["evidence"] = current[-4:]
    return True


def multi_objective_completion_block(
    state: LoopState,
    *,
    message: str,
    verifier_verdict: dict[str, Any] | None,
) -> dict[str, Any] | None:
    ledger = ensure_multi_objective_ledger(state)
    if not ledger:
        return None
    objectives = [item for item in ledger.get("objectives", []) if isinstance(item, dict)]
    if not objectives:
        return None

    resolved_objective_id = resolved_followup_objective_id(state)
    completed_now: list[str] = []
    for objective in objectives:
        if (
            resolved_objective_id
            and str(objective.get("objective_id") or "") == resolved_objective_id
        ) or objective_matches_text(objective, message):
            if mark_objective_done(objective, message):
                completed_now.append(str(objective.get("objective_id") or ""))

    remaining = [
        {
            "objective_id": str(item.get("objective_id") or ""),
            "title": str(item.get("title") or ""),
        }
        for item in objectives
        if str(item.get("status") or "").strip().lower() != "done"
    ]
    if not remaining:
        ledger["status"] = "done"
        return None

    return {
        "completed_now": completed_now,
        "remaining_objectives": remaining,
        "ledger": ledger,
        "last_verifier_verdict": verifier_verdict,
    }
