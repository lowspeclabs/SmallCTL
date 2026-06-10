from __future__ import annotations

from ..remote_scope import has_any_session_ssh_target, remote_scope_is_active
from ..state import LoopState, clip_text_value, normalize_intent_label
from .artifact_visibility import artifact_path_candidates, is_prompt_visible_artifact, is_superseded_artifact
from .frame_state_helpers import dedupe_nonempty

MUTATION_TOOL_NAMES = {
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
}


def coding_anchor_lines(state: LoopState) -> list[str]:
    anchors: list[str] = []
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    is_coding_mode = bool(state.write_session) or bool(state.files_changed_this_cycle) or task_mode in {
        "analysis",
        "local_execute",
        "debug_inspect",
    }
    if not is_coding_mode:
        return anchors

    if state.write_session is not None:
        session = state.write_session
        if session.write_target_path:
            anchors.append(f"target_file={session.write_target_path}")
        anchors.append(f"write_mode={session.write_session_mode}")
        if session.write_sections_completed:
            anchors.append("completed_sections=" + ", ".join(session.write_sections_completed[:5]))
        if session.write_failed_local_patches:
            anchors.append(f"failed_sections={session.write_failed_local_patches} local patch failures")
    if state.files_changed_this_cycle:
        anchors.append("files_changed=" + ", ".join(state.files_changed_this_cycle[-5:]))
    verdict = state.current_verifier_verdict()
    if isinstance(verdict, dict):
        anchors.append("verifier_status=" + str(verdict.get("verdict") or "unknown"))
        command = str(verdict.get("command") or "").strip()
        if command:
            anchors.append(f"verifier_command={command}")
    checklist = state.acceptance_checklist()
    unmet = [
        str(item.get("criterion") or "").strip()
        for item in checklist
        if item.get("satisfied") is False and str(item.get("criterion") or "").strip()
    ]
    if unmet:
        anchors.append("acceptance_deltas=" + "; ".join(unmet[:4]))
    symbols = state.scratchpad.get("_touched_symbols")
    if isinstance(symbols, list):
        touched = [str(symbol).strip() for symbol in symbols if str(symbol).strip()]
        if touched:
            anchors.append("touched_symbols=" + ", ".join(touched[:8]))
    return dedupe_nonempty(anchors)


def artifact_evidence_rows(state: LoopState, *, limit: int = 8) -> list[str]:
    rows: list[str] = []
    for aid, art in state.artifacts.items():
        if is_superseded_artifact(art):
            continue
        if not is_prompt_visible_artifact(art):
            continue
        metadata = getattr(art, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        source_candidates = artifact_path_candidates(art, metadata)
        source = " / ".join(dedupe_nonempty(source_candidates)[:2]) or "observed"
        summary = (getattr(art, "summary", "") or getattr(art, "tool_name", "") or "").strip()
        rows.append(f"  {aid} | {source.replace('|', '/')} | {summary.replace('|', '/')[:110]}")
        if len(rows) >= limit:
            break
    return rows


def render_run_brief(state: LoopState) -> str:
    brief = state.run_brief
    has_content = any(
        [
            brief.original_task,
            brief.task_contract,
            brief.current_phase_objective,
            state.active_intent,
            bool(brief.constraints),
            bool(brief.acceptance_criteria),
        ]
    )
    if not has_content:
        return ""
    parts = ["Run brief:"]
    parts.append(f"  CWD: {state.cwd}")

    if brief.original_task:
        parts.append(f"  Goal: {brief.original_task}")

    if brief.task_contract:
        parts.append(f"  Contract: {brief.task_contract}")

    if brief.current_phase_objective:
        parts.append(f"  Phase focus: {brief.current_phase_objective}")

    if state.active_intent:
        parts.append(f"  Active intent: {normalize_intent_label(state.active_intent)}")

    ssh_sessions = active_ssh_session_labels(state)
    if ssh_sessions and should_render_active_ssh_sessions(state):
        parts.append("  Active SSH sessions: " + " | ".join(ssh_sessions[:3]))

    if brief.constraints:
        parts.append("  Constraints: " + "; ".join(brief.constraints))

    if brief.acceptance_criteria:
        parts.append("  Acceptance criteria: " + "; ".join(brief.acceptance_criteria))

    resolved = state.scratchpad.get("_resolved_followup")
    if isinstance(resolved, dict):
        title = str(resolved.get("option_title") or "").strip()
        index = str(resolved.get("option_index") or "").strip()
        paths = resolved.get("target_paths")
        if isinstance(paths, list):
            cleaned_paths = [str(path).strip() for path in paths if str(path).strip()]
        else:
            cleaned_paths = []
        if title and index:
            if cleaned_paths:
                parts.append(
                    f"  Resolved follow-up: proposal #{index} = {title} in {', '.join(cleaned_paths)}."
                )
            else:
                parts.append(f"  Resolved follow-up: proposal #{index} = {title}.")

    if len(parts) == 1:
        return ""
    return "\n".join(parts)


def run_boundary_lines(state: LoopState) -> list[str]:
    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    transaction = scratchpad.get("_task_transaction")
    handoff = scratchpad.get("_last_task_handoff")
    source: dict[str, object] = {}
    if isinstance(transaction, dict) and transaction:
        source = transaction
    elif isinstance(handoff, dict) and str(handoff.get("status") or "").strip().lower() in {
        "closed",
        "failed",
        "aborted",
        "superseded",
    }:
        source = handoff
    if not source:
        return []

    lines = ["Run boundary:"]
    status = str(source.get("status") or "").strip().lower()
    if status in {"closed", "failed", "aborted", "superseded"}:
        lines.append("  Previous task is closed.")
    turn_type = str(source.get("turn_type") or "").strip()
    if turn_type:
        lines.append(f"  Current turn type: {turn_type}.")
    goal = str(source.get("user_goal") or source.get("current_goal") or source.get("effective_task") or "").strip()
    if goal:
        clipped_goal, _ = clip_text_value(goal, limit=220)
        lines.append(f"  Current goal: {clipped_goal}")
    relevant: list[str] = []
    paths = source.get("allowed_paths")
    if not isinstance(paths, list) or not paths:
        paths = source.get("target_paths")
    if not isinstance(paths, list) or not paths:
        paths = source.get("remote_target_paths")
    if isinstance(paths, list):
        relevant.extend(str(path).strip() for path in paths if str(path).strip())
    artifacts = source.get("allowed_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        artifacts = source.get("last_good_artifact_ids")
    if isinstance(artifacts, list):
        relevant.extend(str(artifact).strip() for artifact in artifacts if str(artifact).strip())
    relevant = dedupe_nonempty(relevant)
    if relevant:
        lines.append("  Relevant prior context: " + "; ".join(relevant[:4]) + ".")
    ignored = source.get("ignored_context")
    if isinstance(ignored, list):
        cleaned_ignored = [str(item).strip() for item in ignored if str(item).strip()]
        if cleaned_ignored:
            lines.append("  Ignore: " + "; ".join(cleaned_ignored[:3]) + ".")
    success = str(source.get("success_condition") or "").strip()
    if success:
        clipped_success, _ = clip_text_value(success, limit=180)
        lines.append(f"  Stop when: {clipped_success}")
    return lines[:7]


def active_ssh_session_labels(state: LoopState) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()

    resolved_remote = state.scratchpad.get("_resolved_remote_followup")
    if isinstance(resolved_remote, dict):
        host = str(resolved_remote.get("host") or "").strip().lower()
        user = str(resolved_remote.get("user") or "").strip()
        if host:
            label = f"{user}@{host}" if user else host
            seen.add(label)
            labels.append(label)

    targets = state.scratchpad.get("_session_ssh_targets")
    if not isinstance(targets, dict):
        return labels

    for key, value in targets.items():
        if not isinstance(value, dict):
            continue
        host = str(value.get("host") or key or "").strip().lower()
        if not host:
            continue
        user = str(value.get("user") or "").strip()
        label = f"{user}@{host}" if user else host
        if label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def continuation_anchor_lines(state: LoopState) -> list[str]:
    handoff = state.scratchpad.get("_last_task_handoff")
    if not isinstance(handoff, dict) or not handoff:
        return []
    if not (
        state.scratchpad.get("_task_boundary_previous_task")
        or state.scratchpad.get("_resolved_followup")
        or state.scratchpad.get("_resolved_remote_followup")
    ):
        return []

    lines: list[str] = []
    next_required_tool = handoff.get("next_required_tool")
    if isinstance(next_required_tool, dict):
        tool_name = str(next_required_tool.get("tool_name") or "").strip()
        if tool_name:
            lines.append(f"next_required_tool={tool_name}")

    last_failed_tool = handoff.get("last_failed_tool")
    if isinstance(last_failed_tool, dict):
        failed_tool_name = str(last_failed_tool.get("tool_name") or "").strip()
        if failed_tool_name:
            lines.append(f"last_failed_tool={failed_tool_name}")
            if bool(last_failed_tool.get("approval_denied")):
                lines.append(
                    f"NOTE: The previous {failed_tool_name} call was denied by user approval. "
                    "If the user indicates approval, re-execute the same call."
                )

    ssh_target = handoff.get("ssh_target")
    if isinstance(ssh_target, dict):
        host = str(ssh_target.get("host") or "").strip().lower()
        user = str(ssh_target.get("user") or "").strip()
        if host:
            lines.append(f"ssh_target={user + '@' if user else ''}{host}")
            lines.append(
                f"Remote target is {host}. If you need to run commands on that host, "
                f"use ssh_exec with host={host}. Do not use any other host."
            )
            targets = state.scratchpad.get("_session_ssh_targets")
            if isinstance(targets, dict):
                target_entry = targets.get(host)
                if isinstance(target_entry, dict):
                    validated_tools = target_entry.get("validated_tools")
                    if isinstance(validated_tools, list):
                        cleaned_tools = [str(item).strip() for item in validated_tools if str(item).strip()]
                    else:
                        cleaned_tools = []
                    last_path = str(target_entry.get("last_validated_path") or "").strip()
                    if cleaned_tools:
                        tool_line = f"validated_remote={user + '@' if user else ''}{host} via {', '.join(cleaned_tools[:2])}"
                        if last_path:
                            tool_line += f" on {last_path}"
                        lines.append(tool_line)

    artifact_ids = handoff.get("last_good_artifact_ids")
    if isinstance(artifact_ids, list):
        cleaned_ids = [str(item).strip() for item in artifact_ids if str(item).strip()]
        if cleaned_ids:
            lines.append("recent_artifacts=" + ", ".join(cleaned_ids[:3]))
    research_artifact_ids = handoff.get("recent_research_artifact_ids")
    if isinstance(research_artifact_ids, list):
        cleaned_research_ids = [str(item).strip() for item in research_artifact_ids if str(item).strip()]
        if cleaned_research_ids:
            lines.append("recent_research_artifacts=" + ", ".join(cleaned_research_ids[:2]))
    return dedupe_nonempty(lines)


def should_render_active_ssh_sessions(state: LoopState) -> bool:
    return remote_scope_is_active(state) or has_any_session_ssh_target(state)


def render_task_ground_truth(state: LoopState) -> str:
    """Render explicit ground-truth about successful tools in the current task."""
    successful_tools: list[str] = []
    for entry in state.tool_history:
        if not isinstance(entry, str):
            continue
        parts = entry.split("|")
        if len(parts) >= 3 and parts[-1] == "success":
            successful_tools.append(parts[0])

    if not successful_tools:
        return (
            "Task ground truth: No tools have succeeded in this task yet. "
            "Do not assume any work is already complete."
        )

    mutation_tools = [tool for tool in successful_tools if tool in MUTATION_TOOL_NAMES]
    if mutation_tools:
        changed = ", ".join(state.files_changed_this_cycle[-5:]) if state.files_changed_this_cycle else "none"
        return (
            "Task ground truth: Mutating operations performed this task: "
            + ", ".join(mutation_tools)
            + ". Files changed this cycle: "
            + changed
            + "."
        )
    return (
        "Task ground truth: Only read/observation tools have succeeded so far ("
        + ", ".join(successful_tools[-5:])
        + "). No mutating operations have been performed in this task yet. "
        "Do not assume any work is already complete."
    )
