from __future__ import annotations

import re
import uuid
from typing import Any

from ..state import ExecutionPlan, PlanStep
from ..tools.planning import _refresh_plan_playbook_artifact


def planning_response_looks_like_plan(text: str, *, allow_numbered_list: bool = False) -> bool:
    normalized = (text or "").strip()
    if len(normalized) < 40:
        return False
    lowered = normalized.lower()
    markers = (
        "plan",
        "goal",
        "success criteria",
        "substep",
        "expected artifact",
        "ready for confirmation",
        "ready to proceed",
        "ready for approval",
    )
    if any(marker in lowered for marker in markers):
        return True
    if re.search(r"^\|\s*\d+\s*\|", normalized, flags=re.MULTILINE):
        return True
    if allow_numbered_list and re.search(r"^\s*\d+[.)]\s+\S", normalized, flags=re.MULTILINE):
        return True
    return False


def extract_plan_steps_from_text(text: str) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        table_match = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$", stripped)
        if table_match:
            step_id = f"P{table_match.group(1)}"
            title = table_match.group(2).strip()
            description = table_match.group(3).strip()
            steps.append(PlanStep(step_id=step_id, title=title, description=description))
            continue
        numbered_match = re.match(r"^\d+[.)]\s*(.+)$", stripped)
        if numbered_match:
            step_id = f"P{len(steps) + 1}"
            steps.append(PlanStep(step_id=step_id, title=numbered_match.group(1).strip()))
    return steps


def synthesize_plan_from_text(harness: Any, text: str, *, allow_numbered_list: bool = False) -> ExecutionPlan | None:
    assistant_text = (text or "").strip()
    if not assistant_text or not planning_response_looks_like_plan(assistant_text, allow_numbered_list=allow_numbered_list):
        return None
    goal = str(harness.state.run_brief.original_task or "").strip() or assistant_text.splitlines()[0].strip()
    steps = extract_plan_steps_from_text(assistant_text)
    if not steps:
        return None
    return ExecutionPlan(
        plan_id=f"plan-{uuid.uuid4().hex[:8]}",
        goal=goal,
        summary=assistant_text,
        steps=steps[:6],
        status="draft",
        approved=False,
    )


def persist_planning_playbook(harness: Any, plan: ExecutionPlan) -> None:
    try:
        _refresh_plan_playbook_artifact(state=harness.state, harness=harness, plan=plan)
    except Exception as exc:
        harness.log.warning("failed to persist plan playbook artifact: %s", exc)
