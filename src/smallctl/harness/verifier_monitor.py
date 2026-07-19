from __future__ import annotations

from typing import Any


def track_verifier_rejection(state: Any, verdict: dict[str, Any] | None) -> dict[str, Any]:
    """Track verifier rejections in state scratchpad for loop detection.

    Returns a dict with loop detection info if a loop is detected.
    """
    if not isinstance(verdict, dict):
        return {}

    verdict_str = str(verdict.get("verdict") or "").strip()
    if verdict_str in {"", "pass"}:
        # Reset rejection count on pass/empty — task made progress
        if state.scratchpad.get("_verifier_rejection_count"):
            state.scratchpad["_verifier_rejection_count"] = 0
        return {}

    # Increment rejection count
    rejection_count = int(state.scratchpad.get("_verifier_rejection_count", 0) or 0) + 1
    state.scratchpad["_verifier_rejection_count"] = rejection_count
    state.scratchpad["_last_verifier_rejection"] = dict(verdict)

    # Update same-target streak only when a new rejection is recorded
    command = str(verdict.get("command") or verdict.get("target") or "").strip()
    if command:
        key = "_fama_same_target_streak"
        last = state.scratchpad.get(key)
        if isinstance(last, dict) and last.get("command") == command:
            streak = int(last.get("streak", 0) or 0) + 1
        else:
            streak = 1
        state.scratchpad[key] = {"command": command, "streak": streak}

    result = {
        "rejection_count": rejection_count,
        "is_loop": rejection_count > 3,
        "verdict": verdict,
    }

    return result
