from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..plans import render_plan_playbook, resolve_plan_export_target, write_plan_file
from ..state import ExecutionPlan, PlanInterrupt, PlanStep, LoopState
from .common import needs_human, ok


def _coerce_step_payload(value: Any, *, fallback_step_id: str | None = None) -> PlanStep | None:
    if isinstance(value, PlanStep):
        return value
    if isinstance(value, str):
        title = value.strip()
        if not title:
            return None
        return PlanStep(
            step_id=str(fallback_step_id or title).strip(),
            title=title,
        )
    if not isinstance(value, dict):
        return None
    step_id = str(value.get("step_id", "") or "").strip()
    title = str(value.get("title", "") or "").strip()
    if not step_id and not title:
        return None
    resolved_step_id = step_id or str(fallback_step_id or title).strip()
    return PlanStep(
        step_id=resolved_step_id,
        title=title or step_id or resolved_step_id,
        description=str(value.get("description", "") or ""),
        status=str(value.get("status", "pending") or "pending"),
        notes=[str(item) for item in (value.get("notes") or []) if str(item).strip()],
        depends_on=[str(item) for item in (value.get("depends_on") or []) if str(item).strip()],
        substeps=[
            substep
            for index, item in enumerate((value.get("substeps") or []), start=1)
            if (
                substep := _coerce_step_payload(
                    item,
                    fallback_step_id=f"{resolved_step_id}.{index}",
                )
            )
        ],
        evidence_refs=[str(item) for item in (value.get("evidence_refs") or []) if str(item).strip()],
    )


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _resolve_plan(state: LoopState) -> ExecutionPlan | None:
    return state.draft_plan or state.active_plan


def _sync_plan_state(state: LoopState, plan: ExecutionPlan) -> None:
    if state.draft_plan is not None and state.draft_plan.plan_id == plan.plan_id:
        state.draft_plan = plan
    if state.active_plan is not None and state.active_plan.plan_id == plan.plan_id:
        state.active_plan = plan
    state.sync_plan_mirror()
    state.touch()


def _refresh_plan_playbook_artifact(*, state: LoopState, harness: Any, plan: ExecutionPlan) -> dict[str, Any]:
    if harness is None or not hasattr(harness, "artifact_store"):
        return {"artifact_id": "", "artifact_summary": ""}

    playbook_text = render_plan_playbook(plan, state=state)
    artifact = harness.artifact_store.persist_generated_text(
        kind="plan_playbook",
        source=plan.goal or "plan",
        content=playbook_text,
        summary=f"Plan playbook for {plan.goal or 'current task'}",
        metadata={
            "section": "plan",
            "plan_id": plan.plan_id,
            "goal": plan.goal,
            "plan_status": plan.status,
            "artifact_role": "plan_playbook",
        },
        tool_name="plan_playbook",
    )
    harness.state.artifacts[artifact.artifact_id] = artifact
    harness.state.plan_artifact_id = artifact.artifact_id
    harness.state.plan_resolved = True
    harness.state.retrieval_cache = [artifact.artifact_id]
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_summary": artifact.summary,
    }


def _format_plan_metadata(plan: ExecutionPlan) -> dict[str, Any]:
    active_step = plan.active_step()
    return {
        "plan_id": plan.plan_id,
        "plan_status": plan.status,
        "approved": plan.approved,
        "output_path": plan.requested_output_path,
        "output_format": plan.requested_output_format,
        "active_step": active_step.step_id if active_step else "",
    }


def _normalize_requested_format(format: str | None) -> str | None:
    if format is None:
        return None
    normalized = str(format or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"markdown", "md"}:
        return "markdown"
    if normalized in {"text", "txt"}:
        return "text"
    raise ValueError("Unsupported plan export format. Use markdown/md or text/txt.")


def _suggest_plan_export_path(path: str | None) -> str | None:
    normalized = str(path or "").strip()
    if not normalized:
        return None
    requested_path = Path(normalized)
    suggested_name = f"{requested_path.stem or 'plan'}.md"
    return str(requested_path.with_name(suggested_name))


def _normalize_plan_export_request(
    output_path: str | None,
    output_format: str | None,
) -> tuple[str | None, str | None, dict[str, Any]]:
    normalized_output_path = str(output_path or "").strip() or None
    raw_output_format = str(output_format or "").strip() or None
    warning_metadata: dict[str, Any] = {}

    if not normalized_output_path:
        if not raw_output_format:
            return None, None, warning_metadata
        try:
            return None, _normalize_requested_format(raw_output_format), warning_metadata
        except ValueError as exc:
            warning_metadata["export_warning"] = str(exc)
            warning_metadata["rejected_output_format"] = raw_output_format
            return None, None, warning_metadata

    try:
        _, normalized_output_format = resolve_plan_export_target(
            normalized_output_path,
            format=raw_output_format,
        )
        return normalized_output_path, normalized_output_format, warning_metadata
    except ValueError as exc:
        warning_metadata["export_warning"] = str(exc)

    try:
        _, inferred_output_format = resolve_plan_export_target(
            normalized_output_path,
            format=None,
        )
    except ValueError:
        warning_metadata["rejected_output_path"] = normalized_output_path
        suggestion = _suggest_plan_export_path(normalized_output_path)
        if suggestion:
            warning_metadata["suggested_output_path"] = suggestion
        if raw_output_format:
            try:
                normalized_output_format = _normalize_requested_format(raw_output_format)
            except ValueError:
                warning_metadata["rejected_output_format"] = raw_output_format
                normalized_output_format = None
            else:
                return None, normalized_output_format, warning_metadata
        return None, None, warning_metadata

    if raw_output_format:
        warning_metadata["rejected_output_format"] = raw_output_format
    return normalized_output_path, inferred_output_format, warning_metadata


def _try_write_plan_export(plan: ExecutionPlan) -> str | None:
    if not plan.requested_output_path:
        return None
    try:
        write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
    except ValueError as exc:
        return str(exc)
    return None


async def plan_set(
    *,
    goal: str,
    summary: str = "",
    inputs: list[Any] | None = None,
    outputs: list[Any] | None = None,
    constraints: list[Any] | None = None,
    acceptance_criteria: list[Any] | None = None,
    implementation_plan: list[Any] | None = None,
    steps: list[Any] | None = None,
    output_path: str | None = None,
    plan_output_path: str | None = None,
    output_format: str | None = None,
    plan_output_format: str | None = None,
    state: LoopState,
    harness: Any,
) -> dict:
    requested_output_path = plan_output_path if plan_output_path is not None else output_path
    requested_output_format = plan_output_format if plan_output_format is not None else output_format
    normalized_output_path, normalized_output_format, warning_metadata = _normalize_plan_export_request(
        requested_output_path,
        requested_output_format,
    )

    plan = ExecutionPlan(
        plan_id=f"plan-{uuid.uuid4().hex[:8]}",
        goal=str(goal or "").strip(),
        summary=str(summary or "").strip(),
        inputs=_coerce_string_list(inputs),
        outputs=_coerce_string_list(outputs),
        constraints=_coerce_string_list(constraints),
        acceptance_criteria=_coerce_string_list(acceptance_criteria),
        implementation_plan=_coerce_string_list(implementation_plan),
        steps=[
            step
            for index, item in enumerate((steps or []), start=1)
            if (step := _coerce_step_payload(item, fallback_step_id=f"P{index}")) is not None
        ],
        status="draft",
        requested_output_path=normalized_output_path,
        requested_output_format=normalized_output_format,
        approved=False,
    )
    state.draft_plan = plan
    if state.active_plan is not None and state.active_plan.approved:
        state.active_plan = plan
    state.planning_mode_enabled = True
    state.planner_requested_output_path = plan.requested_output_path or ""
    state.planner_requested_output_format = plan.requested_output_format or ""
    state.acceptance_waived = False
    state.last_verifier_verdict = None
    state.last_failure_class = ""
    state.files_changed_this_cycle = []
    state.repair_cycle_id = ""
    state.stagnation_counters = {}
    if plan.acceptance_criteria:
        state.acceptance_ledger = {
            criterion: state.acceptance_ledger.get(criterion, "pending")
            for criterion in plan.acceptance_criteria
        }
    else:
        state.acceptance_ledger = {}
    state.sync_plan_mirror()
    artifact_info = _refresh_plan_playbook_artifact(state=state, harness=harness, plan=plan)
    state.touch()
    payload = {
        "status": "plan_set",
        "plan": plan,
        **_format_plan_metadata(plan),
        **artifact_info,
        **warning_metadata,
    }
    return ok(payload, metadata=warning_metadata)


async def plan_step_update(
    *,
    step_id: str,
    status: str,
    note: str = "",
    state: LoopState,
    harness: Any,
) -> dict:
    plan = _resolve_plan(state)
    if plan is None:
        return {"success": False, "output": None, "error": "No active or draft plan available.", "metadata": {}}
    step = plan.find_step(str(step_id).strip())
    if step is None:
        return {"success": False, "output": None, "error": f"Unknown plan step: {step_id}", "metadata": {}}
    step.status = str(status or "pending").strip().lower()
    if note.strip():
        step.notes.append(note.strip())
    plan.touch()
    _sync_plan_state(state, plan)
    artifact_info = _refresh_plan_playbook_artifact(state=state, harness=harness, plan=plan)
    export_warning = _try_write_plan_export(plan)
    return ok(
        {
            "status": "step_updated",
            "step_id": step.step_id,
            "step_status": step.status,
            "note": note,
            **artifact_info,
            **({"export_warning": export_warning} if export_warning else {}),
            **_format_plan_metadata(plan),
        }
    )


async def plan_request_execution(
    *,
    question: str,
    state: LoopState,
    harness: Any,
) -> dict:
    plan = _resolve_plan(state)
    if plan is None:
        return {"success": False, "output": None, "error": "No active or draft plan available.", "metadata": {}}
    plan.status = "awaiting_approval"
    plan.touch()
    artifact_info = _refresh_plan_playbook_artifact(state=state, harness=harness, plan=plan)
    export_warning = _try_write_plan_export(plan)
    interrupt = PlanInterrupt(
        question=str(question or "Plan ready. Execute it now?").strip() or "Plan ready. Execute it now?",
        plan_id=plan.plan_id,
    )
    state.planner_interrupt = interrupt
    state.pending_interrupt = {
        "kind": interrupt.kind,
        "question": interrupt.question,
        "plan_id": interrupt.plan_id,
        "approved": interrupt.approved,
        "response_mode": interrupt.response_mode,
    }
    state.touch()
    return needs_human(
        interrupt.question,
        metadata={
            "kind": interrupt.kind,
            "plan_id": interrupt.plan_id,
            **artifact_info,
            **({"export_warning": export_warning} if export_warning else {}),
            **_format_plan_metadata(plan),
        },
    )


async def plan_export(
    *,
    path: str,
    format: str | None = None,
    state: LoopState,
    harness: Any,
) -> dict:
    plan = _resolve_plan(state)
    if plan is None:
        return {"success": False, "output": None, "error": "No active or draft plan available.", "metadata": {}}
    export_path = str(path or "").strip()
    if not export_path:
        return {"success": False, "output": None, "error": "Plan export path is required.", "metadata": {}}
    requested_format = format if format is not None else plan.requested_output_format
    try:
        _, normalized_format = resolve_plan_export_target(export_path, format=requested_format)
    except ValueError as exc:
        return {"success": False, "output": None, "error": str(exc), "metadata": {}}
    plan.requested_output_path = export_path
    plan.requested_output_format = normalized_format
    state.planner_requested_output_path = export_path
    state.planner_requested_output_format = normalized_format
    state.touch()
    content = write_plan_file(plan, Path(export_path), format=normalized_format)
    artifact_info = _refresh_plan_playbook_artifact(state=state, harness=harness, plan=plan)
    return ok(
        {
            "status": "exported",
            "path": export_path,
            "format": plan.requested_output_format,
            "bytes_written": len(content.encode("utf-8")),
            **artifact_info,
            **_format_plan_metadata(plan),
        }
    )


async def plan_subtask(
    *,
    brief: str,
    phase: str = "plan",
    constraints: list[str] | None = None,
    acceptance_criteria: list[str] | None = None,
    harness: Any,
    state: LoopState,
) -> dict:
    del state
    if harness is None:
        return {"success": False, "output": None, "error": "Harness unavailable.", "metadata": {}}
    result = await harness.run_subtask(
        brief=brief,
        phase=phase,
        depth=1,
        metadata={
            "constraints": _coerce_string_list(constraints),
            "acceptance_criteria": _coerce_string_list(acceptance_criteria),
        },
    )
    plan = _resolve_plan(harness.state)
    if plan is not None and result.summary:
        active_step = plan.active_step()
        if active_step is not None:
            active_step.evidence_refs.extend(result.artifact_ids[:3])
            active_step.evidence_refs = list(dict.fromkeys(active_step.evidence_refs))
            active_step.notes.append(result.summary)
            plan.touch()
            _sync_plan_state(harness.state, plan)
    return ok(
        {
            "status": result.status,
            "summary": result.summary,
            "artifact_ids": result.artifact_ids,
            "files_touched": result.files_touched,
            "decisions": result.decisions,
            "remaining_plan": result.remaining_plan,
        }
    )
