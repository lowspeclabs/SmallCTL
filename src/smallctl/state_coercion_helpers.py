from __future__ import annotations

from typing import Any

from .state_schema import PlanStep


def _compact_plan_step_lines(step: Any, *, depth: int = 0) -> list[str]:
    prefix = "  " * depth
    lines = [f"{prefix}{step.compact_label()}"]
    if step.description:
        lines.append(f"{prefix}  {step.description}")
    if step.notes:
        for note in step.notes:
            lines.append(f"{prefix}  note: {note}")
    if step.evidence_refs:
        lines.append(f"{prefix}  evidence: {', '.join(step.evidence_refs)}")
    if step.claim_refs:
        lines.append(f"{prefix}  claims: {', '.join(step.claim_refs)}")
    for substep in step.substeps:
        lines.extend(_compact_plan_step_lines(substep, depth=depth + 1))
    return lines
