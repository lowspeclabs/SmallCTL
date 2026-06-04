from __future__ import annotations

from typing import Any

from ..recovery_metrics import increment_metric
from ..state import LoopState, clip_text_value


def render_remote_mutation_next_action(state: LoopState) -> str:
    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    requirement = scratchpad.get("_remote_mutation_requires_verification")
    if not isinstance(requirement, dict) or not requirement:
        return ""

    host = str(requirement.get("host") or "").strip()
    user = str(requirement.get("user") or "").strip()
    verified_paths = {
        str(path).strip()
        for path in requirement.get("verified_paths", [])
        if str(path).strip()
    }
    guessed_paths = [
        str(path).strip()
        for path in requirement.get("guessed_paths", [])
        if str(path).strip()
    ]
    pending_path = next((path for path in guessed_paths if path not in verified_paths), "")
    if pending_path:
        return "Run " + render_ssh_tool_call(
            "ssh_file_read",
            host=host,
            user=user,
            path=pending_path,
        )

    verified_directories = {
        str(path).strip().rstrip("/")
        for path in requirement.get("verified_directory_empty_checks", [])
        if str(path).strip()
    }
    for check in remote_mutation_directory_checks(requirement):
        directory_path = check["path"]
        if directory_path in verified_directories:
            continue
        command = f"find {directory_path} -mindepth 1 -maxdepth 1 -print -quit"
        return "Run " + render_ssh_tool_call(
            "ssh_exec",
            host=host,
            user=user,
            command=command,
        )
    return ""


def render_ssh_tool_call(tool_name: str, *, host: str, user: str = "", path: str = "", command: str = "") -> str:
    args: list[str] = []
    if host:
        args.append(f"host={host!r}")
    if user:
        args.append(f"user={user!r}")
    if path:
        args.append(f"path={path!r}")
    if command:
        args.append(f"command={command!r}")
    return f"{tool_name}(" + ", ".join(args) + ")"


def remote_mutation_directory_checks(requirement: dict[str, Any]) -> list[dict[str, str]]:
    raw_checks = requirement.get("directory_empty_checks")
    if not isinstance(raw_checks, list):
        return []
    checks: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_checks:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip().rstrip("/")
        if not path or path in seen:
            continue
        seen.add(path)
        glob = str(item.get("glob") or "").strip()
        checks.append({"path": path, "glob": glob})
    return checks


def render_recovery_guidance(state: LoopState, token_budget: int = 500) -> list[str]:
    config = state.scratchpad.get("_recovery_config")
    config = config if isinstance(config, dict) else {}
    if not bool(config.get("reflexion_enabled", True)):
        return []
    active_subtask = None
    ledger = state.subtask_ledger
    if ledger is not None:
        active_subtask = ledger.active()
    latest_failure = state.failure_events[-1] if state.failure_events else None
    if (
        active_subtask is None
        and latest_failure is None
        and not state.reflexion_memory
        and not state.last_failure_class
        and state.write_session is None
        and not isinstance(state.scratchpad.get("_last_schema_validation_hint"), dict)
        and not isinstance(state.scratchpad.get("_read_loop_recovery_payload"), dict)
    ):
        return []

    lines: list[str] = []
    lines.extend(fresh_schema_validation_hint_lines(state))
    lines.extend(fresh_read_loop_recovery_lines(state))
    if active_subtask is not None:
        line = f"Active subtask {active_subtask.subtask_id} [{active_subtask.status}]: {active_subtask.title}"
        if active_subtask.attempts:
            line += f"; attempts={active_subtask.attempts}"
        if active_subtask.next_action:
            line += f"; next={active_subtask.next_action}"
        lines.append(line)
        if active_subtask.blockers:
            lines.append("Latest blocker: " + active_subtask.blockers[-1])
    if latest_failure is not None:
        failure_line = f"Latest failure: {latest_failure.failure_class}"
        if latest_failure.message:
            failure_line += f" - {latest_failure.message}"
        lines.append(failure_line)
        if latest_failure.suggested_next_action:
            lines.append("Next safe action: " + latest_failure.suggested_next_action)
    elif state.last_failure_class:
        lines.append("Latest failure class: " + state.last_failure_class)

    active_subtask_id = str(getattr(active_subtask, "subtask_id", "") or "").strip()
    top_k = max(0, int(config.get("reflexion_inject_top_k", 3) or 3))
    reflections = [
        item
        for item in state.reflexion_memory
        if not active_subtask_id or not item.subtask_id or item.subtask_id == active_subtask_id
    ]
    reflections.sort(
        key=lambda item: (
            float(getattr(item, "score", 0.0) or 0.0),
            float(getattr(item, "timestamp", 0.0) or 0.0),
        ),
        reverse=True,
    )
    injected_reflections = reflections[:top_k]
    for reflection in injected_reflections:
        lines.append(
            f"Lesson {reflection.failure_class}: {reflection.lesson} Avoid: {reflection.avoid} Next: {reflection.next_safe_action}"
        )
    if injected_reflections:
        increment_metric(state, "reflections_injected", len(injected_reflections))
        for reflection in injected_reflections:
            reflection.used_count = int(getattr(reflection, "used_count", 0) or 0) + 1

    completed_limit = max(0, int(config.get("subtask_inject_completed_limit", 3) or 3))
    if ledger is not None and completed_limit:
        completed = [task for task in ledger.subtasks if task.status == "done"][-completed_limit:]
        for task in completed:
            lines.append(f"Completed subtask {task.subtask_id}: {task.title}")

    clipped: list[str] = []
    budget_chars = max(80, int(token_budget or 500) * 4)
    used = 0
    for line in lines:
        text = str(line or "").strip()
        if not text:
            continue
        if len(text) > 240:
            text = text[:239].rstrip() + "..."
        if used + len(text) > budget_chars and clipped:
            break
        clipped.append(text)
        used += len(text)
    return clipped


def fresh_schema_validation_hint_lines(state: LoopState) -> list[str]:
    payload = state.scratchpad.get("_last_schema_validation_hint")
    if not isinstance(payload, dict):
        return []
    created = int(payload.get("created_at_step", 0) or 0)
    if int(state.step_count or 0) - created > 6:
        return []
    excerpt = str(payload.get("schema_excerpt") or "").strip()
    if not excerpt:
        return []
    return ["Latest tool schema hint: " + excerpt.replace("\n", " | ")]


def fresh_read_loop_recovery_lines(state: LoopState) -> list[str]:
    payload = state.scratchpad.get("_read_loop_recovery_payload")
    if not isinstance(payload, dict):
        return []
    created = int(payload.get("created_at_step", 0) or 0)
    if int(state.step_count or 0) - created > 6:
        return []
    tool_name = str(payload.get("tool_name") or "").strip()
    target = str(payload.get("target") or "").strip()
    summary = str(payload.get("last_evidence_summary") or "").strip()
    action = str(payload.get("allowed_next_action") or "").strip()
    line = "Read-loop recovery"
    if tool_name:
        line += f" for {tool_name}"
    if target:
        line += f" target={target}"
    lines = [line + "."]
    if summary:
        lines.append("Prior evidence: " + summary)
    if action:
        lines.append("Allowed next action: " + action)
    return lines


def repair_continuity_lines(state: LoopState) -> list[str]:
    capsule = state.scratchpad.get("_repair_continuity_capsule")
    if not isinstance(capsule, dict):
        return []
    created_at_step = int(capsule.get("created_at_step", 0) or 0)
    current_step = int(state.step_count or 0)
    if current_step - created_at_step > 5:
        return []
    lines: list[str] = []
    command = str(capsule.get("command") or "").strip()
    if command:
        lines.append(f"Failed command: {command}")
    verdict = str(capsule.get("verdict") or "").strip()
    exit_code = capsule.get("exit_code")
    if verdict or exit_code is not None:
        parts: list[str] = []
        if verdict:
            parts.append(f"verdict={verdict}")
        if exit_code is not None:
            parts.append(f"exit={exit_code}")
        lines.append(f"Result: {' | '.join(parts)}")
    failure_mode = str(capsule.get("failure_mode") or "").strip()
    if failure_mode:
        lines.append(f"Suspected cause: {failure_mode}")
    last_attempted_fix = str(capsule.get("last_attempted_fix") or "").strip()
    if last_attempted_fix:
        lines.append(f"Last attempted fix: {last_attempted_fix}")
    next_action = str(
        capsule.get("next_suggested_action") or capsule.get("suggested_next_action") or ""
    ).strip()
    if next_action:
        lines.append(f"Next suggested action: {next_action}")
    return lines


def guard_trip_recovery_lines(state: LoopState) -> list[str]:
    capsule = state.scratchpad.get("_guard_trip_recovery_capsule")
    if not isinstance(capsule, dict):
        return []
    created_at_step = int(capsule.get("created_at_step", 0) or 0)
    current_step = int(state.step_count or 0)
    if current_step - created_at_step > 8:
        return []
    lines: list[str] = []
    goal = str(capsule.get("goal") or "").strip()
    if goal:
        clipped_goal, _ = clip_text_value(goal, limit=220)
        lines.append(f"Interrupted goal: {clipped_goal}")
    failed_tool = str(capsule.get("failed_tool") or "").strip()
    if failed_tool:
        lines.append(f"Guard stopped repeated tool: {failed_tool}")
        lines.append(f"Do not retry {failed_tool} with the same arguments; continue from preserved progress.")
    reason = str(capsule.get("reason") or "").strip()
    if reason and not failed_tool:
        clipped_reason, _ = clip_text_value(reason, limit=220)
        lines.append(f"Guard reason: {clipped_reason}")
    artifact_ids = [
        str(item).strip()
        for item in (capsule.get("preserved_artifact_ids") or [])
        if str(item).strip()
    ]
    if artifact_ids:
        lines.append("Preserved progress artifacts: " + ", ".join(artifact_ids[:6]))
    summary_id = str(capsule.get("summary_id") or "").strip()
    if summary_id:
        lines.append(f"Preserved task summary: {summary_id}")
    return lines
