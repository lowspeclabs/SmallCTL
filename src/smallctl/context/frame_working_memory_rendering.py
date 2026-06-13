from __future__ import annotations

import re

from ..guards import is_four_b_or_under_model_name
from ..state import LoopState
from .artifact_visibility import is_prompt_visible_artifact, is_superseded_artifact
from .frame_recovery_rendering import guard_trip_recovery_lines, render_remote_mutation_next_action, repair_continuity_lines
from .frame_run_rendering import artifact_evidence_rows, continuation_anchor_lines, render_task_ground_truth, run_boundary_lines
from .frame_session_rendering import render_session_notepad, render_write_session
from .frame_state_helpers import state_model_name
from .observations import build_observation_packets
from .retrieval import LexicalRetriever


def render_working_memory(
    state: LoopState,
    *,
    phase_lines: list[str],
    coding_anchor_lines: list[str],
    fama_capsule_lines: list[str] | None = None,
    recovery_guidance_lines: list[str] | None = None,
) -> str:
    fama_capsule_lines = list(fama_capsule_lines or [])
    recovery_guidance_lines = list(recovery_guidance_lines or [])
    memory = state.working_memory
    plan = state.active_plan or state.draft_plan
    has_content = any(
        [
            memory.current_goal,
            memory.plan,
            memory.decisions,
            memory.open_questions,
            memory.known_facts,
            memory.failures,
            memory.next_actions,
            bool((state.scratchpad.get("_session_notepad") or {}).get("entries"))
            if isinstance(state.scratchpad.get("_session_notepad"), dict)
            else False,
            plan is not None,
            bool(state.artifacts),
            state.write_session is not None,
            bool(phase_lines),
            bool(coding_anchor_lines),
            bool(fama_capsule_lines),
            bool(recovery_guidance_lines),
            bool(run_boundary_lines(state)),
            bool(guard_trip_recovery_lines(state)),
        ]
    )
    if not has_content:
        return ""
    sections = [f"Current CWD: {state.cwd}"]
    task_targets = state.scratchpad.get("_task_target_paths")
    if isinstance(task_targets, list):
        cleaned_targets = [str(path).strip() for path in task_targets if str(path).strip()]
        if cleaned_targets:
            sections.append("Task targets: " + " | ".join(cleaned_targets[:3]))
    ground_truth = render_task_ground_truth(state)
    if ground_truth:
        sections.append(ground_truth)
    boundary_lines = run_boundary_lines(state)
    if boundary_lines:
        sections.extend(boundary_lines)
    if fama_capsule_lines:
        sections.append("FAMA mitigation:")
        sections.extend(f"  {line}" for line in fama_capsule_lines[:5])
    if recovery_guidance_lines:
        sections.append("Recovery guidance:")
        sections.extend(f"  {line}" for line in recovery_guidance_lines[:8])
    repair_lines = repair_continuity_lines(state)
    if repair_lines:
        sections.append("Repair continuity:")
        sections.extend(f"  {line}" for line in repair_lines[:6])
    guard_lines = guard_trip_recovery_lines(state)
    if guard_lines:
        sections.append("Guard trip recovery:")
        sections.extend(f"  {line}" for line in guard_lines[:6])
    current_goal = LexicalRetriever._effective_current_goal(state)
    if current_goal:
        sections.append("Current goal: " + current_goal)
    if memory.plan:
        sections.append("Plan: " + " | ".join(memory.plan))
    if memory.decisions:
        sections.append("Decisions: " + " | ".join(memory.decisions))
    if memory.open_questions:
        sections.append("Open questions: " + " | ".join(memory.open_questions))
    if memory.known_facts:
        filtered_facts = [f for f in memory.known_facts if not _is_low_value_known_fact(f)]
        if filtered_facts:
            sections.append("Known facts: " + " | ".join(filtered_facts))
    sub4b_web_findings = render_sub4b_top_web_findings(state)
    if sub4b_web_findings:
        sections.append(sub4b_web_findings)
    if memory.failures:
        sections.append("Known failures: " + " | ".join(memory.failures))
    if memory.next_actions:
        sections.append("Next actions: " + " | ".join(memory.next_actions))
    remote_mutation_action = render_remote_mutation_next_action(state)
    if remote_mutation_action:
        sections.append("Remote mutation verification pending: " + remote_mutation_action)
    anchor_lines = continuation_anchor_lines(state)
    if anchor_lines:
        sections.append("Continuation anchor:")
        sections.extend(f"  {line}" for line in anchor_lines[:4])
    notepad_section = render_session_notepad(state)
    if notepad_section:
        sections.append(notepad_section)
    if phase_lines:
        sections.extend(phase_lines)
    if coding_anchor_lines:
        sections.append("Coding anchors:")
        sections.extend(f"  {line}" for line in coding_anchor_lines[:8])
    if plan is not None:
        sections.append("Plan summary: " + plan.goal)
        sections.append(f"Plan status: {plan.status}")
        sections.append(
            "Plan resolved: "
            + ("yes" if state.plan_resolved else "no")
            + (f" | Plan artifact: {state.plan_artifact_id}" if state.plan_artifact_id else "")
        )
        if state.plan_artifact_id:
            sections.append(
                f"Plan playbook artifact: {state.plan_artifact_id} (use this as the staged implementation checklist)"
            )
        active_step = plan.active_step()
        if active_step is not None:
            sections.append(f"Active step: {active_step.step_id} [{active_step.status}] {active_step.title}")
        for step in plan.steps[:6]:
            sections.append(f"Plan step: [{step.status}] {step.step_id} {step.title}")
        if plan.requested_output_path:
            sections.append(f"Plan export: {plan.requested_output_path}")
    elif state.plan_resolved and memory.plan:
        sections.append(
            "Plan resolved: yes"
            + (f" | Plan artifact: {state.plan_artifact_id}" if state.plan_artifact_id else "")
        )
        if state.plan_artifact_id:
            sections.append(
                f"Plan playbook artifact: {state.plan_artifact_id} (use this as the staged implementation checklist)"
            )
    if state.contract_phase() == "repair":
        repair_bits = [f"Repair phase: {state.contract_phase()}"]
        if state.last_failure_class:
            repair_bits.append(f"Failure class: {state.last_failure_class}")
        if state.repair_cycle_id:
            repair_bits.append(
                f"System repair cycle: {state.repair_cycle_id} (diagnostic only; not a write_session_id)"
            )
        if state.files_changed_this_cycle:
            repair_bits.append("Files changed this cycle: " + ", ".join(state.files_changed_this_cycle[-5:]))
        if state.stagnation_counters:
            counters = ", ".join(
                f"{name}={count}"
                for name, count in sorted(state.stagnation_counters.items())
                if count
            )
            if counters:
                repair_bits.append(f"Stagnation: {counters}")
        sections.append("Repair focus: " + " | ".join(repair_bits))
    if state.write_session:
        sections.append(render_write_session(state))

    if state.artifacts:
        art_lines = []
        for aid, art in state.artifacts.items():
            if is_superseded_artifact(art):
                continue
            if not is_prompt_visible_artifact(art):
                continue
            summary_snippet = (art.summary or art.tool_name or "").strip()[:90]
            art_lines.append(f"  - {aid}: {summary_snippet}")
        if art_lines:
            if is_four_b_or_under_model_name(state_model_name(state)):
                evidence_rows = artifact_evidence_rows(state)
                if evidence_rows:
                    sections.append(
                        "Available Evidence (pinned for this 4B-or-under model; use these artifact/source cues before rereading):\n"
                        "  artifact_id | source/path | summary\n"
                        + "\n".join(evidence_rows)
                        + "\n  Only page forward with artifact_read(start_line=...) when you need unseen lines."
                    )
            else:
                sections.append(
                    "Available Artifacts (compressed summaries already in context; page forward with artifact_read(start_line=...) only if you need more unseen lines):\n"
                    + "\n".join(art_lines)
                )
    if not sections:
        return ""
    return "Working memory:\n" + "\n".join(sections)


def render_sub4b_top_web_findings(state: LoopState) -> str:
    if not is_four_b_or_under_model_name(state_model_name(state)):
        return ""
    packets = [
        packet
        for packet in build_observation_packets(state, limit=8)
        if packet.tool_name in {"web_search", "web_fetch"} and not packet.stale and str(packet.summary or "").strip()
    ]
    if not packets:
        return ""
    findings: list[str] = []
    for packet in packets[-3:]:
        summary = str(packet.summary or "").strip()
        if not summary:
            continue
        if len(summary) > 180:
            summary = summary[:179].rstrip() + "..."
        if summary not in findings:
            findings.append(summary)
    if not findings:
        return ""
    return "Top web findings: " + " | ".join(findings)


def _is_low_value_known_fact(fact: str) -> bool:
    """Suppress generic tool-schema facts that do not advance the task."""
    text = str(fact or "").lower()
    low_value_patterns = [
        r"\btask_complete\b.*\bkeys\b",
        r"\b\w+\b keys: \w+",
        r"\bparameters\b.*\binclude\b",
        r"\btool takes\b",
        r"\btool accepts\b",
        r"\brequires arguments\b",
        r"\bargument names\b",
        r"\bschema\b.*\bfields\b",
    ]
    return any(re.search(pattern, text) for pattern in low_value_patterns)
