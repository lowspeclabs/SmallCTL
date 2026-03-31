from __future__ import annotations

import json
from typing import Any

from .state import LoopState


def build_system_prompt(
    state: LoopState,
    phase: str,
    *,  # Added type hint for clarity
    available_tool_names: list[str] | None = None,
    strategy_prompt: str | None = None,  # NEW PARAMETER
    manifest: dict[str, Any] | None = None,
) -> str:
    _TOOL_MSG_COMPACT_THRESHOLD: int = 400
    active_profiles = ", ".join(state.active_tool_profiles or ["core"])
    parts = [
        "You are smallctl, an autonomous execution agent. ",
        "RESPONSE STRUCTURE: You MUST start EVERY response with a <think> block for plan and rationale. ",
        "TOOL CALL FORMAT: If tools are available, call them using the JSON format: `{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"val\"}}`. ",
        "CONCISENESS: NEVER re-type detailed tool outputs (like full directory listings or file contents) in your conversational chat. ",
        "CONCISENESS: Summarize findings in 1-2 sentences in chat, then call `task_complete(message='...')` with the definitive answer. ",
        "STRICT: No hallucinations. Do not add descriptions or metadata (like 'Python project config') to file lists unless the tool returned them. ",
        "STRICT: NEVER use text-based tool tags like `<tool_call>` or functional syntax like `dir_list()`. These are FORBIDDEN. ",
        f"Phase: {phase} | Profile: {active_profiles} | CWD: {state.cwd}. ",
        "WORKSPACE: Use relative paths (e.g. 'src/app.py'). ",
        "PRIVILEGES: If a password was provided in the TASK, use it via 'echo PASS | sudo -S COMMAND'. You do NOT have passwordless sudo. ",
        "SHELL: Prefer standard POSIX redirection (e.g., `2>&1`) for robustness. ",
        "MEMORY: Use `memory_update` to persist key facts. ",
        "REDUNDANCY: Prefer the compressed summary or preview first. Use `artifact_read` or `artifact_grep` only when you need the full evidence or line-level detail. ",
        "REDUNDANCY: reuse information you already retrieved. avoid rereading the same path unless the summary is insufficient. ",
        "REDUNDANCY: Do not call `artifact_read` again on an artifact that is already summarized in the tool preview, Working Memory, or Retrieved Artifact Snippets; treat those summaries as the content unless you truly need more detail. ",
        "Efficiency: Use the fewest calls. Do not repeat identical calls. ",
    ]
    if state.scratchpad.get("subtask_depth"):
        parts.append(
            "SUBTASK: Stay in scope. Reuse information you already retrieved. Avoid rereading the same path unless the summary is insufficient. Call task_complete as soon as objective is met."
        )
    if available_tool_names:
        tool_names = ", ".join(sorted(set(available_tool_names)))
        if tool_names:
            parts.append(
                f"Tools: {tool_names}. Use these names exactly. Do not merge calls."
            )
            if "artifact_read" in tool_names:
                parts.append(
                    "ARTIFACTS: Use `artifact_read(artifact_id='A000X')` for paging large outputs. "
                    "When an artifact is truncated, continue from the next chunk with `start_line`/`end_line` instead of rereading from the beginning. "
                    "Use `artifact_grep` for exact line searches and `start_line`/`end_line` for chunks. "
                    "DO NOT call `file_read` on artifacts."
                )
    if state.run_brief.original_task:
        parts.append(
            f"\nTASK: {state.run_brief.original_task}\n"
            "Fulfill all requirements. Once finished, you MUST call `task_complete(message='...')`. "
            "If you already provided a full report in your conversational response, use a concise confirmation in the 'message' field (e.g. 'Task complete as described.') to avoid redundancy."
        )
    if state.run_brief.task_contract:
        parts.append(
            f"\nCONTRACT: {state.run_brief.task_contract}\n"
        )
    if state.working_memory.known_facts:
        facts = "\n".join(f"- {f}" for f in state.working_memory.known_facts)
        parts.append(
            f"\n\n### WORKING MEMORY (PINNED FACTS)\n{facts}\n"
        )
    if getattr(state.working_memory, "current_goal", ""):
        parts.append(f"\n\n### CURRENT GOAL\n- {state.working_memory.current_goal}\n")
    if manifest:
        manifest_json = json.dumps(manifest, indent=2)
        parts.append(
            f"\n\n### CODE INDEX MANIFEST\n"
            "You have access to a pre-built code index. Use it to navigate the codebase before reading files. "
            "Query the index first (e.g. searching symbols, tracing references) to find relevant files and specific line ranges. "
            "Do not read entire large files—use the index to locate the relevant blocks.\n"
            f"```json\n{manifest_json}\n```\n"
        )
    
    prompt = " ".join(parts)
    
    # NEW: Prepended strategy-specific prompt if provided
    if strategy_prompt:
        prompt = f"{strategy_prompt}\n\n---\n{prompt}"
    
    return prompt


def build_planning_prompt(
    state: LoopState,
    phase: str,
    *,
    available_tool_names: list[str] | None = None,
    strategy_prompt: str | None = None,
    manifest: dict[str, Any] | None = None,
) -> str:
    planning_intro = (
        "PLANNING MODE IS ACTIVE. "
        "Gather facts before proposing execution, use planning tools to create and refine a structured plan, "
        "and do not begin normal execution until the user explicitly approves the plan. "
        "If you produce a draft plan, the harness will pause for approval automatically, so do not keep looping on the plan in the same turn. "
        "Do not call `task_complete` to exit planning; use `plan_request_execution` to pause for approval. "
        "Use plan export paths only for plan documents (.md, .txt, .text), never for implementation files like .py. "
        "Exactly one level of subplanning is allowed."
    )
    prompt = build_system_prompt(
        state,
        phase,
        available_tool_names=available_tool_names,
        strategy_prompt=strategy_prompt,
        manifest=manifest,
    )
    planning_sections = [planning_intro]
    plan = state.active_plan or state.draft_plan
    if plan is not None:
        planning_sections.append(
            "Current plan summary:\n"
            f"- plan_id: {plan.plan_id}\n"
            f"- goal: {plan.goal}\n"
            f"- status: {plan.status}\n"
            f"- approved: {plan.approved}\n"
            f"- output_path: {plan.requested_output_path or ''}\n"
            f"- output_format: {plan.requested_output_format or ''}"
        )
        if state.plan_resolved:
            planning_sections.append(
                "Plan state:\n"
                f"- resolved: yes\n"
                f"- plan_artifact_id: {state.plan_artifact_id or ''}\n"
                "The plan is already mirrored in Working Memory. Do not reread the same plan artifact unless the plan materially changed."
            )
        active_step = plan.active_step()
        if active_step is not None:
            planning_sections.append(
                "Active step:\n"
                f"- {active_step.step_id}: {active_step.title} [{active_step.status}]"
            )
        step_lines = []
        for step in plan.steps:
            step_lines.extend(_render_plan_step(step))
        if step_lines:
            planning_sections.append("Plan steps:\n" + "\n".join(step_lines))
    elif state.plan_resolved and state.working_memory.plan:
        planning_sections.append(
            "Plan state:\n"
            f"- resolved: yes\n"
            f"- plan_artifact_id: {state.plan_artifact_id or ''}\n"
            "The plan is already mirrored in Working Memory. Do not reread the same plan artifact unless the plan materially changed."
        )
    if state.planner_requested_output_path:
        planning_sections.append(
            f"Export target: {state.planner_requested_output_path}"
        )
    return f"{prompt}\n\n---\n" + "\n\n".join(planning_sections)


def _render_plan_step(step: Any, *, depth: int = 0) -> list[str]:
    indent = "  " * depth
    lines = [f"{indent}- [{step.status}] {step.step_id} {step.title}".strip()]
    if getattr(step, "description", ""):
        lines.append(f"{indent}  {step.description}")
    for note in getattr(step, "notes", []) or []:
        lines.append(f"{indent}  note: {note}")
    for substep in getattr(step, "substeps", []) or []:
        lines.extend(_render_plan_step(substep, depth=depth + 1))
    return lines
