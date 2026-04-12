from __future__ import annotations

from pathlib import Path

from .state import ExecutionPlan, PlanStep, LoopState

_ALLOWED_PLAN_EXPORT_SUFFIXES = {"", ".md", ".txt", ".text"}


def render_plan(plan: ExecutionPlan, *, format: str | None = None) -> str:
    normalized_format = _normalize_plan_format(format)
    if normalized_format == "markdown":
        return _render_markdown_plan(plan)
    return _render_text_plan(plan)


def render_plan_playbook(plan: ExecutionPlan, *, state: LoopState | None = None) -> str:
    """Render a plan as a staged implementation playbook.

    The playbook is intentionally more operational than the canonical plan export:
    it tells the model how to proceed in small, bounded steps instead of trying to
    complete the whole script in one shot.
    """
    stage_lines = [
        "# Plan Playbook",
        "",
        f"Goal: {plan.goal}",
        f"Status: {plan.status}",
    ]
    if plan.summary:
        stage_lines.extend(["", "## Plan Summary", plan.summary.strip()])
    if any([plan.inputs, plan.outputs, plan.constraints, plan.acceptance_criteria, plan.implementation_plan, plan.claim_refs]):
        stage_lines.append("")
        stage_lines.append("## Spec Contract")
        if plan.inputs:
            stage_lines.append("Inputs:")
            stage_lines.extend([f"- {item}" for item in plan.inputs])
        if plan.outputs:
            stage_lines.append("Outputs:")
            stage_lines.extend([f"- {item}" for item in plan.outputs])
        if plan.constraints:
            stage_lines.append("Constraints:")
            stage_lines.extend([f"- {item}" for item in plan.constraints])
        if plan.acceptance_criteria:
            stage_lines.append("Acceptance Criteria:")
            stage_lines.extend([f"- {item}" for item in plan.acceptance_criteria])
        if plan.implementation_plan:
            stage_lines.append("Implementation Plan:")
            stage_lines.extend([f"- {item}" for item in plan.implementation_plan])
        if plan.claim_refs:
            stage_lines.append("Claim References:")
            stage_lines.extend([f"- {item}" for item in plan.claim_refs])
    if plan.requested_output_path:
        stage_lines.extend(["", f"Export target: {plan.requested_output_path}"])
    if state is not None and state.run_brief.original_task:
        stage_lines.extend(["", "## Original Task", state.run_brief.original_task.strip()])
    stage_lines.extend(
        [
            "",
            "## Implementation Order",
            "1. Write the file skeleton first.",
            "2. Add the function and class signatures next.",
            "3. Fill in the code in small increments.",
            "4. Debug and verify with targeted tests or a minimal run.",
            "",
            "## Operating Rules",
            "- Do not attempt a one-shot implementation of the whole script.",
            "- Treat this playbook as the source of truth for the next action.",
            "- Complete the current stage before moving to the next one.",
            "- If a step grows too large, split it into smaller substeps.",
            "- Verify against the acceptance criteria before calling the task complete.",
        ]
    )
    if plan.steps:
        stage_lines.extend(["", "## Plan Steps"])
        for step in plan.steps:
            stage_lines.extend(_render_playbook_step(step))
    else:
        stage_lines.extend(["", "## Plan Steps", "- (no steps yet)"])
    return "\n".join(stage_lines).strip() + "\n"


def write_plan_file(plan: ExecutionPlan, path: str | Path, *, format: str | None = None) -> str:
    output_path, normalized_format = resolve_plan_export_target(path, format=format)
    content = render_plan(plan, format=normalized_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return content


def resolve_plan_export_target(path: str | Path, *, format: str | None = None) -> tuple[Path, str]:
    output_path = Path(path)
    suffix = output_path.suffix.lower()

    if suffix not in _ALLOWED_PLAN_EXPORT_SUFFIXES:
        raise ValueError(
            "Plan export targets must use .md, .txt, or .text; "
            f"refusing to write a plan to '{output_path.name}'."
        )

    normalized_format = _normalize_requested_plan_format(format)
    if normalized_format is None:
        normalized_format = "markdown" if suffix == ".md" else "text"
    elif suffix == ".md" and normalized_format != "markdown":
        raise ValueError("Plan export path ending in .md requires markdown format.")
    elif suffix in {".txt", ".text"} and normalized_format != "text":
        raise ValueError("Plan export path ending in .txt/.text requires text format.")

    return output_path, normalized_format


def _normalize_requested_plan_format(format: str | None) -> str | None:
    normalized = str(format or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"markdown", "md"}:
        return "markdown"
    if normalized in {"text", "txt"}:
        return "text"
    raise ValueError(
        "Unsupported plan export format. Use markdown/md or text/txt."
    )


def _normalize_plan_format(format: str | None) -> str:
    return _normalize_requested_plan_format(format) or "text"


def _render_markdown_plan(plan: ExecutionPlan) -> str:
    lines = [
        "# Execution Plan",
        "",
        f"Goal: {plan.goal}",
        f"Status: {plan.status}",
    ]
    if plan.summary:
        lines.extend(["", f"Summary: {plan.summary}"])
    spec = plan.spec_summary()
    if spec:
        lines.extend(["", f"Spec: {spec}"])
    if plan.requested_output_path:
        lines.extend(["", f"Output: {plan.requested_output_path}"])
    lines.extend(["", "## Steps"])
    if plan.steps:
        for step in plan.steps:
            lines.extend(_render_markdown_step(step))
    else:
        lines.append("- [ ] (no steps yet)")
    return "\n".join(lines).strip() + "\n"


def _render_markdown_step(step: PlanStep, *, depth: int = 0) -> list[str]:
    indent = "  " * depth
    checkbox = _status_to_checkbox(step.status)
    lines = [f"{indent}- {checkbox} {step.step_id} {step.title}".rstrip()]
    if step.description:
        lines.append(f"{indent}  - {step.description}")
    for note in step.notes:
        lines.append(f"{indent}  - note: {note}")
    if step.evidence_refs:
        lines.append(f"{indent}  - evidence: {', '.join(step.evidence_refs)}")
    if step.claim_refs:
        lines.append(f"{indent}  - claims: {', '.join(step.claim_refs)}")
    for substep in step.substeps:
        lines.extend(_render_markdown_step(substep, depth=depth + 1))
    return lines


def _render_text_plan(plan: ExecutionPlan) -> str:
    lines = [
        "Execution Plan",
        "",
        f"Goal: {plan.goal}",
        f"Status: {plan.status}",
    ]
    if plan.summary:
        lines.extend(["", f"Summary: {plan.summary}"])
    spec = plan.spec_summary()
    if spec:
        lines.extend(["", f"Spec: {spec}"])
    if plan.claim_refs:
        lines.extend(["", f"Claims: {', '.join(plan.claim_refs)}"])
    if plan.requested_output_path:
        lines.extend(["", f"Output: {plan.requested_output_path}"])
    lines.extend(["", "Steps:"])
    if plan.steps:
        for step in plan.steps:
            lines.extend(_render_text_step(step))
    else:
        lines.append("[pending] (no steps yet)")
    return "\n".join(lines).strip() + "\n"


def _render_text_step(step: PlanStep, *, depth: int = 0) -> list[str]:
    indent = "  " * depth
    lines = [f"{indent}[{step.status}] {step.step_id} {step.title}".rstrip()]
    if step.description:
        lines.append(f"{indent}  {step.description}")
    for note in step.notes:
        lines.append(f"{indent}  note: {note}")
    if step.evidence_refs:
        lines.append(f"{indent}  evidence: {', '.join(step.evidence_refs)}")
    if step.claim_refs:
        lines.append(f"{indent}  claims: {', '.join(step.claim_refs)}")
    for substep in step.substeps:
        lines.extend(_render_text_step(substep, depth=depth + 1))
    return lines


def _render_playbook_step(step: PlanStep, *, depth: int = 0) -> list[str]:
    indent = "  " * depth
    lines = [f"{indent}- [{step.status}] {step.step_id} {step.title}".rstrip()]
    if step.description:
        lines.append(f"{indent}  - {step.description}")
    if step.notes:
        for note in step.notes:
            lines.append(f"{indent}  - note: {note}")
    if step.evidence_refs:
        lines.append(f"{indent}  - evidence: {', '.join(step.evidence_refs)}")
    if step.claim_refs:
        lines.append(f"{indent}  - claims: {', '.join(step.claim_refs)}")
    for substep in step.substeps:
        lines.extend(_render_playbook_step(substep, depth=depth + 1))
    return lines


def _status_to_checkbox(status: str) -> str:
    return "[x]" if status == "completed" else "[ ]"
