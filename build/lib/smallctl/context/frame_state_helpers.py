from __future__ import annotations

from ..state import ContextBrief, LoopState


def dedupe_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def collect_files_in_play(*, state: LoopState, latest_brief: ContextBrief | None) -> list[str]:
    candidates: list[str] = []
    task_targets = state.scratchpad.get("_task_target_paths")
    if isinstance(task_targets, list):
        candidates.extend(str(path or "").strip() for path in task_targets)
    candidates.extend(state.files_changed_this_cycle)
    if latest_brief is not None:
        candidates.extend(latest_brief.files_touched)
    plan = state.active_plan or state.draft_plan
    if plan is not None and plan.requested_output_path:
        candidates.append(str(plan.requested_output_path))
    if state.write_session is not None:
        if state.write_session.write_target_path:
            candidates.append(state.write_session.write_target_path)
        if state.write_session.write_staging_path:
            candidates.append(state.write_session.write_staging_path)
    return dedupe_nonempty(candidates)


def invalidated_fact_hints(state: LoopState) -> list[str]:
    queued = state.scratchpad.get("_invalidated_facts_queue")
    if not isinstance(queued, list):
        return []
    return dedupe_nonempty([str(item).strip() for item in queued if str(item).strip()])


def state_model_name(state: LoopState) -> str:
    scratchpad = getattr(state, "scratchpad", {})
    if isinstance(scratchpad, dict):
        return str(scratchpad.get("_model_name") or "").strip()
    return ""
