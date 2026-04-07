from __future__ import annotations

import json
from typing import Any

from .guards import is_small_model_name
from .state import LoopState, clip_text_value


def build_system_prompt(
    state: LoopState,
    phase: str,
    *,  # Added type hint for clarity
    available_tool_names: list[str] | None = None,
    strategy_prompt: str | None = None,  # NEW PARAMETER
    manifest: dict[str, Any] | None = None,
    indexer_mode: bool = False,
) -> str:
    _TOOL_MSG_COMPACT_THRESHOLD: int = 400
    active_profiles = ", ".join(state.active_tool_profiles or ["core"])
    parts = [
        "You are smallctl, an autonomous execution agent. ",
        "RESPONSE STRUCTURE: You MUST start EVERY response with a <think> block for plan and rationale. ",
        "GOAL RETENTION: The user's original task is your primary obligation throughout all turns. Intermediate tool results, assist messages, and artifact reads do NOT satisfy the task unless you have fully answered what was asked. Keep the task goal in view at all times. ",
        "TOOL CALL FORMAT: If tools are available, call them using the JSON format: `{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"val\"}}`. ",
        "CONCISENESS: NEVER re-type detailed tool outputs (like full directory listings or file contents) in your conversational chat. ",
        "CONCISENESS: Summarize findings in 1-2 sentences in chat, then call `task_complete(message='...')` with the definitive answer. ",
        "STRICT: No hallucinations. Do not add descriptions or metadata (like 'Python project config') to file lists unless the tool returned them. ",
        "STRICT: NEVER use text-based tool tags like `<tool_call>` or functional syntax like `dir_list()`. These are FORBIDDEN. ",
        f"Phase: {phase} | Active tool profiles: {active_profiles} | CWD: {state.cwd}. Only the tools exposed for the active profiles are available. ",
        f"Contract phase: {state.contract_phase()}. ",
        "WORKSPACE: Use relative paths (e.g. 'src/app.py'). You should prefer workspace-relative paths and do not start them with a leading slash or backslash. ",
        "PRIVILEGES: Do not invent or guess a sudo password. If privileged access is required, use passwordless sudo or ask the user for help via `ask_human`. ",
        "SHELL: Prefer standard POSIX redirection (e.g., `2>&1`) for robustness. ",
        "MEMORY: Use `memory_update` to persist key facts. ",
        "MEMORY: If the user asks you to save, remember, store, or pin information, call `memory_update` before `task_complete`. ",
        "MEMORY: If `memory_update` says the content already exists, treat it as a no-op and do not call it again for the same fact. Move on to the next step or call `task_complete`. ",
        "REDUNDANCY: Prefer the compressed summary or preview first. Use `artifact_read` or `artifact_grep` only when you need the full evidence or line-level detail. ",
        "REDUNDANCY: reuse information you already retrieved. avoid rereading the same path unless the summary is insufficient. ",
        "REDUNDANCY: Do not call `artifact_read` again on an artifact that is already summarized in the tool preview, Working Memory, or Retrieved Artifact Snippets; treat those summaries as the content unless you truly need more detail. ",
        "REDUNDANCY: If `artifact_read` or `artifact_recall` returns 'not found', treat the evidence as unavailable and re-execute the original tool call (e.g. re-run the shell command) instead of synthesizing or guessing from memory. ",
        "ARTIFACT PAGING: When an artifact is truncated, page forward with `start_line` and `end_line` to get the next unseen chunk. Do not reread earlier chunks unless you need to verify a specific line. ",
        "PLAN HANDOFF: If a plan exists, treat its playbook artifact as the implementation contract. The required order is: 1) write the file skeleton, 2) add function signatures, 3) fill in the code, 4) debug and verify. Do not jump straight to a one-shot full script. ",
        "AUTHORING: In the author phase, prefer one concrete write or read action at a time. If you already have a target file, write or replace it directly instead of bouncing through multiple exploratory calls. Create the target artifact before shell execution; the harness will block shell and SSH commands until there is something concrete to verify. ",
        "AUTHORING: In the repair phase, read the failing file or evidence first, then patch one narrow target and re-run the smallest useful check. ",
        "CHUNKED AUTHORING: When writing large files or complex logic, the harness may initialize a Write Session. "
        "Break the file into logical sections (e.g., imports, constants, classes, main logic). "
        "Use `file_write` or `file_append` with these parameters: "
        "`write_session_id`: Use the ID provided by the harness. "
        "`section_name` or `section_id`: A descriptive name for the current chunk (e.g., 'imports'). "
        "`section_role`: Optional role label for the chunk. "
        "`next_section_name`: The name of the section you will write next. Omit this for the final chunk to finalize the session. "
        "When resuming an active session, prefer `file_write`; the harness will track append/replace behavior from the session metadata. "
        "During local repair, keep the same session and prefer `replace_strategy='overwrite'` so you repair the active section cleanly instead of appending duplicate code. "
        "Complete the entire session before moving to other tasks or verification. ",
        "PLAN HANDOFF: Before calling `task_complete`, ensure the acceptance criteria are satisfied or explicitly waived. Use `loop_status` to check progress and the latest verifier verdict. ",
        "Efficiency: Use the fewest calls. Do not repeat identical calls. Do not repeat the same or near-identical tool call. ",
        f"Once your objective is met, stop exploring and call task_complete(message='...').",
    ]
    if state.contract_phase() == "repair":
        repair_bits = []
        if state.last_failure_class:
            repair_bits.append(f"failure class: {state.last_failure_class}")
        if state.repair_cycle_id:
            repair_bits.append(f"repair cycle: {state.repair_cycle_id}")
        if state.files_changed_this_cycle:
            repair_bits.append(
                "files changed this cycle: " + ", ".join(state.files_changed_this_cycle[-5:])
            )
        if state.stagnation_counters:
            counters = ", ".join(
                f"{name}={count}"
                for name, count in sorted(state.stagnation_counters.items())
                if count
            )
            if counters:
                repair_bits.append(f"stagnation counters: {counters}")
        if repair_bits:
            parts.append("REPAIR FOCUS: " + " | ".join(repair_bits) + ". ")
    if state.write_session:
        session = state.write_session
        recovery_bits = [
            f"session={session.write_session_id}",
            f"mode={session.write_session_mode}",
            f"intent={session.write_session_intent}",
            f"target={session.write_target_path}",
        ]
        if session.write_current_section:
            recovery_bits.append(f"active_section={session.write_current_section}")
        if session.write_next_section:
            recovery_bits.append(f"next_section={session.write_next_section}")
        if session.write_sections_completed:
            recovery_bits.append("completed=" + ", ".join(session.write_sections_completed))
        if session.write_failed_local_patches:
            recovery_bits.append(f"failures={session.write_failed_local_patches}")
        verifier = session.write_last_verifier or {}
        if verifier:
            verifier_bits = [f"verifier={verifier.get('verdict', 'unknown')}"]
            command = str(verifier.get("command", "") or "").strip()
            if command:
                verifier_bits.append(f"command={command}")
            verifier_output, clipped = clip_text_value(str(verifier.get("output", "") or "").strip(), limit=220)
            if verifier_output:
                suffix = " [truncated]" if clipped else ""
                verifier_bits.append(f"output={verifier_output}{suffix}")
            recovery_bits.append("last_verifier=" + " | ".join(verifier_bits))
        parts.append(
            "\n\n### WRITE RECOVERY\n"
            + "\n".join(f"- {item}" for item in recovery_bits)
            + "\nContinue from the active section instead of restarting the file. Resume with `file_write` plus the active session metadata. Do not reread the whole file unless local context is genuinely insufficient."
        )
    verifier_verdict = state.current_verifier_verdict() or {}
    if verifier_verdict:
        verifier_bits = [f"verdict={verifier_verdict.get('verdict', 'unknown')}"]
        verifier_target, clipped = clip_text_value(
            str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip(),
            limit=220,
        )
        if verifier_target:
            suffix = " [truncated]" if clipped else ""
            verifier_bits.append(f"target={verifier_target}{suffix}")
        verifier_output, clipped = clip_text_value(
            str(verifier_verdict.get("key_stderr") or verifier_verdict.get("key_stdout") or "").strip(),
            limit=220,
        )
        if verifier_output:
            suffix = " [truncated]" if clipped else ""
            verifier_bits.append(f"output={verifier_output}{suffix}")
        parts.append("LATEST VERIFIER: " + " | ".join(verifier_bits) + ". ")
        if str(verifier_verdict.get("verdict") or "").strip() not in {"", "pass"} and not state.acceptance_waived:
            parts.append(
                "VERIFIER GUARD: Do not repeat `task_complete` while this verifier is failing. "
                "Use `loop_status` to inspect the blocker, then either run one focused repair step or rerun the check in a zero-exit diagnostic form if the failure itself is the evidence you need. "
            )
    if is_small_model_name(state.scratchpad.get("_model_name")):
        parts.append(
            "SMALL MODEL GUARD: If you seem frozen, hung, or stuck, continue from the last concrete step instead of restarting. "
            "The harness may nudge you to continue, so emit the next tool call or a short progress update immediately."
        )
    if state.scratchpad.get("subtask_depth"):
        parts.append(
            "SUBTASK: Stay in scope. Reuse information you already retrieved. Avoid rereading the same path unless the summary is insufficient. Call task_complete as soon as objective is met."
        )
    if available_tool_names:
        tool_names = ", ".join(sorted(set(available_tool_names)))
        if tool_names:
            parts.append(
                f"Available tools on this turn: {tool_names}. Use these names exactly and do not claim the tool list is unknown. Do not merge calls."
            )
            if "ssh_exec" in available_tool_names:
                parts.append(
                    "NETWORK: Use `ssh_exec` for remote SSH commands and `shell_exec` for local shell work only. "
                    "Do not shell out to `ssh` through `shell_exec` when `ssh_exec` is available."
                )
            if "artifact_read" in tool_names:
                parts.append(
                    "ARTIFACTS: Use `artifact_read(artifact_id='A000X')` for paging large outputs. "
                    "When an artifact is truncated, continue from the next chunk with `start_line`/`end_line` instead of rereading from the beginning. "
                    "Use `artifact_grep` for exact line searches and `start_line`/`end_line` for chunks. "
                    "DO NOT call `file_read` on artifacts."
                )
            if "show_artifact" in tool_names:
                parts.append(
                    "ARTIFACTS: Use `show_artifact(artifact_id='A000X')` when the user explicitly wants the full artifact contents shown directly in the chat/UI."
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
    if _is_write_first_task(state):
        parts.append(
            "\n\n### WRITE-FIRST GUIDANCE\n"
            "The task is to create or edit a file. Use directory listing only to locate the target once, "
            "then move directly to `file_write` or `file_append` instead of repeating `dir_list` on the same path. "
            "If you already know the destination directory, stop exploring and start writing."
        )
    plan = state.active_plan or state.draft_plan
    if plan is not None and state.plan_artifact_id:
        parts.append(
            "\n\n### PLAN PLAYBOOK\n"
            f"Plan artifact: {state.plan_artifact_id}\n"
            "Treat this artifact as the source of truth for staged implementation. "
            "Before a large script change, complete the playbook in order: file skeleton, function signatures, code, then debug."
        )
    if indexer_mode and manifest:
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


def _is_write_first_task(state: LoopState) -> bool:
    if getattr(state, "active_intent", "") == "use_write_file":
        return True
    intent_tags = set(getattr(state, "intent_tags", []) or [])
    if "write_file" in intent_tags:
        return True
    task = (getattr(state.run_brief, "original_task", "") or "").lower()
    return bool(task and any(marker in task for marker in ("build a python script", "create a python script", "write a python script")))


def build_planning_prompt(
    state: LoopState,
    phase: str,
    *,
    available_tool_names: list[str] | None = None,
    strategy_prompt: str | None = None,
    manifest: dict[str, Any] | None = None,
    indexer_mode: bool = False,
) -> str:
    planning_intro = (
        "PLANNING MODE IS ACTIVE. "
        "Gather facts before proposing execution, use planning tools to create and refine a structured plan, "
        "and convert the plan into a playbook artifact that stages implementation into file skeleton, functions, code, and debug steps. "
        "Do not begin normal execution until the user explicitly approves the plan. "
        "If you produce a draft plan, the harness will pause for approval automatically, so do not keep looping on the plan in the same turn. "
        "Do not call `task_complete` to exit planning; use `plan_request_execution` to pause for approval. "
        "Use plan export paths only for plan documents (.md, .txt, .text), never for implementation files like .py. "
        "Exactly one level of subplanning is allowed. "
        "The required plan shape is: goal, inputs, outputs, constraints, acceptance_criteria, implementation_plan, and steps."
    )
    prompt = build_system_prompt(
        state,
        phase,
        available_tool_names=available_tool_names,
        strategy_prompt=strategy_prompt,
        manifest=manifest,
        indexer_mode=indexer_mode,
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
                "The plan is already mirrored in Working Memory. Do not reread the same plan artifact unless the plan materially changed. "
                "If the playbook artifact exists, use it to sequence the next implementation step."
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
            "The plan is already mirrored in Working Memory. Do not reread the same plan artifact unless the plan materially changed. "
            "If the playbook artifact exists, use it to sequence the next implementation step."
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
