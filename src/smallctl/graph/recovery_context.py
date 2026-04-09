from __future__ import annotations

from typing import Any


def build_goal_recap(harness: Any) -> str:
    state = getattr(harness, "state", None)
    if state is None:
        return ""

    run_brief = getattr(state, "run_brief", None)
    working_memory = getattr(state, "working_memory", None)

    original_task = str(getattr(run_brief, "original_task", "") or "").strip()
    current_goal = str(getattr(working_memory, "current_goal", "") or "").strip()
    phase_focus = str(getattr(run_brief, "current_phase_objective", "") or "").strip()

    parts: list[str] = []
    if original_task:
        parts.append(f"Original task: {original_task}")
    if current_goal and current_goal != original_task:
        parts.append(f"Current goal: {current_goal}")
    if phase_focus and phase_focus not in {original_task, current_goal}:
        parts.append(f"Phase focus: {phase_focus}")

    if not parts:
        return ""
    return "Goal recap: " + " | ".join(parts)
