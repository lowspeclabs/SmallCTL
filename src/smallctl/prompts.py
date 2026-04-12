from __future__ import annotations

import json
from typing import Any

from .guards import is_small_model_name
from .phases import phase_contract
from .state import LoopState, clip_text_value, normalize_intent_label

_GEMMA_MODEL_MARKERS = (
    "google_gemma-4",
    "google_gemma",
    "gemma-4",
    "gemma-3",
    "gemma/",
)
_EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES = (
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
)
_EXACT_GEMMA_4_26B_A4B_IT_MODEL_SUFFIXES = (
    "google_gemma-4-26b-a4b-it",
)


def is_gemma_model_name(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _GEMMA_MODEL_MARKERS))


def is_exact_small_gemma_4_it_model_name(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(
        normalized
        and any(
            normalized == suffix or normalized.endswith(f"/{suffix}")
            for suffix in _EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES
        )
    )


def is_exact_large_gemma_4_26b_a4b_it_model_name(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(
        normalized
        and any(
            normalized == suffix or normalized.endswith(f"/{suffix}")
            for suffix in _EXACT_GEMMA_4_26B_A4B_IT_MODEL_SUFFIXES
        )
    )


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
    model_name = ""
    scratchpad = getattr(state, "scratchpad", {})
    if isinstance(scratchpad, dict):
        model_name = str(scratchpad.get("_model_name") or "").strip()
    gemma_mode = is_gemma_model_name(model_name)
    exact_small_gemma_mode = is_exact_small_gemma_4_it_model_name(model_name)
    exact_large_gemma_26b_mode = is_exact_large_gemma_4_26b_a4b_it_model_name(model_name)
    if gemma_mode:
        response_structure = (
            "RESPONSE STRUCTURE: This Gemma model may use its native reasoning format. "
            "Do not force a <think> block or add conflicting wrapper tags. "
            "Use the model's normal reasoning flow, then continue with tool calls or the final answer as needed. "
        )
        if exact_small_gemma_mode:
            response_structure += (
                "SMALL GEMMA-4 FORMAT: If you include short reasoning before a tool call, end the reasoning cleanly, "
                "then emit exactly one JSON tool object on its own line with no wrapper tags. "
            )
    else:
        response_structure = (
            "RESPONSE STRUCTURE: You MUST start EVERY response with a <think> block for plan and rationale. "
        )
    if exact_large_gemma_26b_mode:
        contract = phase_contract(phase)
        parts = [
            "You are smallctl, an autonomous execution agent. ",
            response_structure,
            "PRIMARY RULE: Solve the current user task only. Keep the task goal in view and avoid side quests. ",
            "TOOL CALL FORMAT: If tools are available, call them using the JSON format: `{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"val\"}}`. ",
            "STRICT: No hallucinations. Only report what tools actually returned. ",
            "STRICT: NEVER use text-based tool tags like `<tool_call>` or functional syntax like `dir_list()`. ",
            "CONCISENESS: Do not paste long tool output into chat. Summarize briefly, then call `task_complete(message='...')` when done. ",
            "REDUNDANCY: Reuse what you already know. Do not repeat identical or near-identical tool calls. ",
            f"Phase: {phase} | Active tool profiles: {active_profiles} | CWD: {state.cwd}. Only the tools exposed for the active profiles are available. ",
            f"Contract phase: {state.contract_phase()}. ",
            f"Phase contract focus: {contract.focus}. ",
            "WORKSPACE: Prefer workspace-relative paths like `src/app.py`, not absolute paths. ",
            "If the task is complete, stop and call `task_complete(message='...')`. ",
        ]
    else:
        contract = phase_contract(phase)
        parts = [
            "You are smallctl, an autonomous execution agent. ",
            response_structure,
            "GOAL RETENTION: The user's original task is your primary obligation throughout all turns. Intermediate tool results, assist messages, and artifact reads do NOT satisfy the task unless you have fully answered what was asked. Keep the task goal in view at all times. ",
            "TOOL CALL FORMAT: If tools are available, call them using the JSON format: `{\"name\": \"tool_name\", \"arguments\": {\"arg\": \"val\"}}`. ",
            "CONCISENESS: NEVER re-type detailed tool outputs (like full directory listings or file contents) in your conversational chat. ",
            "CONCISENESS: Summarize findings in 1-2 sentences in chat, then call `task_complete(message='...')` with the definitive answer. ",
            "STRICT: No hallucinations. Do not add descriptions or metadata (like 'Python project config') to file lists unless the tool returned them. ",
            "STRICT: NEVER use text-based tool tags like `<tool_call>` or functional syntax like `dir_list()`. These are FORBIDDEN. ",
            f"Phase: {phase} | Active tool profiles: {active_profiles} | CWD: {state.cwd}. Only the tools exposed for the active profiles are available. ",
            f"Contract phase: {state.contract_phase()}. ",
            f"Phase contract focus: {contract.focus}. ",
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
            "SYSTEM IDS: `repair-*` values are system repair cycle IDs for diagnostics only. Never copy a system repair cycle ID into `write_session_id`. ",
            "CHUNKED AUTHORING: When writing large files or complex logic, the harness may initialize a Write Session. "
            "Break the file into logical sections (e.g., imports, constants, classes, main logic). "
            "Use `file_write` or `file_append` for new files, large sections, or chunked authoring. "
            "Use `file_patch` for small exact edits inside an existing file or active staged session. "
            "When using `file_write` or `file_append`, include these parameters: "
            "`write_session_id`: Use the ID provided by the harness. "
            "`section_name` or `section_id`: A descriptive name for the current chunk (e.g., 'imports'). "
            "`section_role`: Optional role label for the chunk. "
            "`next_section_name`: The name of the section you will write next. Omit this for the final chunk to finalize the session. "
            "When resuming an active session, prefer `file_write` for chunk continuation; the harness will track append/replace behavior from the session metadata. "
            "If you need a narrow exact repair inside the staged copy, prefer `file_patch` instead. "
            "If prior chunks are no longer visible because tool previews were compacted or truncated, recover the current staged content first with `file_read(path=target)` before choosing `replace_strategy='overwrite'`. "
            "Do not assume earlier chunks were lost and do not rewrite the whole file from memory unless you have reread the staged copy. "
            "During local repair with `file_write`, keep the same session and prefer `replace_strategy='overwrite'` so you repair the active section cleanly instead of appending duplicate code. "
            "Complete the entire session before moving to other tasks or verification. ",
            "PLAN HANDOFF: Before calling `task_complete`, ensure the acceptance criteria are satisfied or explicitly waived. Use `loop_status` to check progress and the latest verifier verdict. ",
            "Efficiency: Use the fewest calls. Do not repeat identical calls. Do not repeat the same or near-identical tool call. ",
            f"Once your objective is met, stop exploring and call task_complete(message='...').",
        ]
    if exact_small_gemma_mode:
        parts.append(
            "SMALL GEMMA-4 STRICT FORMAT: Never emit `<tool_call>`, `<call>`, `<function=...>`, "
            "`<channel|>`, `<thought>`, angle-bracket function wrappers like `<task_complete(...)>`, "
            "or bare functional syntax like `dir_list()` or `task_complete(message='...')`. "
            "If tools are needed, emit only the JSON object. The backticked task_complete examples in this prompt "
            "describe intent only; do not copy that literal syntax into the response. "
        )
    if state.contract_phase() == "repair":
        repair_bits = []
        if state.last_failure_class:
            repair_bits.append(f"failure class: {state.last_failure_class}")
        if state.repair_cycle_id:
            repair_bits.append(
                f"system repair cycle: {state.repair_cycle_id} (diagnostic only; not a write_session_id)"
            )
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
    current_contract_phase = state.contract_phase()
    if current_contract_phase == "explore":
        parts.append(
            "EXPLORE FOCUS: Collect verified observations, reduce uncertainty, and surface open questions before drafting a plan. "
            "Prefer read-only tools and concise fact capture."
        )
    elif current_contract_phase == "plan":
        parts.append(
            "PLAN FOCUS: Rely on the compressed evidence packet, candidate causes, and handoff artifacts rather than a raw transcript dump. "
            "Turn observations into hypotheses and an executable plan."
        )
    elif current_contract_phase == "author":
        parts.append(
            "AUTHOR FOCUS: Use the approved ExecutionPlan, the target files, and the active write session. "
            "Prefer one bounded implementation change at a time."
        )
    elif current_contract_phase == "execute":
        parts.append(
            "EXECUTE FOCUS: Use the approved plan and evidence support, keep execution bounded to approved actions, and note the verification target."
        )
    elif current_contract_phase == "verify":
        parts.append(
            "VERIFY FOCUS: Compare the observed state against the acceptance criteria and recent evidence. Prefer verification reads over new writes."
        )
    if state.write_session:
        session = state.write_session
        recovery_bits = [
            f"session={session.write_session_id}",
            f"mode={session.write_session_mode}",
            f"intent={session.write_session_intent}",
            f"target={session.write_target_path}",
        ]
        if session.write_staging_path:
            recovery_bits.append(f"staging={session.write_staging_path}")
        if session.write_current_section:
            recovery_bits.append(f"active_section={session.write_current_section}")
        if session.write_next_section:
            recovery_bits.append(f"next_section={session.write_next_section}")
        if session.write_sections_completed:
            recovery_bits.append("completed=" + ", ".join(session.write_sections_completed))
        if session.write_failed_local_patches:
            recovery_bits.append(f"failures={session.write_failed_local_patches}")
        if session.write_pending_finalize:
            recovery_bits.append("pending_finalize=yes")
        else:
            recovery_bits.append("pending_finalize=no")
        if session.write_next_section:
            recovery_bits.append(f"next_action=continue section {session.write_next_section}")
        elif session.write_current_section:
            recovery_bits.append(f"next_action=finish section {session.write_current_section}")
        elif session.write_pending_finalize:
            recovery_bits.append("next_action=finalize staged copy")
        else:
            recovery_bits.append("next_action=continue writing target file")
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
            + f"\nContinue from the active section instead of restarting the file. Resume with `file_write` plus the active session metadata for chunk continuation, or use `file_patch` for a narrow exact edit inside the staged copy. If prior chunks are not fully visible because previews were truncated or compacted, recover the current staged content first with `file_read(path='{session.write_target_path}')` or `artifact_read(artifact_id='{session.write_session_id}__stage')`. The staging path is for read/verify only; the target path is the real write destination. Do not assume the chunks were lost or rewrite the whole file from memory unless you intentionally reread the staged copy and then choose `replace_strategy='overwrite'`."
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
            "The harness may nudge you to continue, so emit the next tool call or a short progress update immediately. "
            "When writing code, keep each chunk under 50 lines and finish one logical section before starting the next. "
            "Use tool names exactly as listed and never invent aliases like `use_shell_exec`."
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
                    "Do not shell out to `ssh` through `shell_exec` when `ssh_exec` is available. "
                    "When the task includes a username, prefer `ssh_exec(target='user@host', command='...')`, "
                    "for example `target='root@192.168.1.63'`, instead of sending only `host='...'`. "
                    "If the user explicitly asks to rerun, recheck, or confirm live, do not rely on retrieved historical notes alone; issue a fresh `ssh_exec` unless the tool is unavailable or blocked."
                )
                if is_small_model_name(state.scratchpad.get("_model_name")):
                    parts.append(
                        "SMALL MODEL TOOL ROUTING: Remote host/IP/user/password mentioned means `ssh_exec`. "
                        "`shell_exec` is local-only."
                    )
            if "shell_exec" in available_tool_names:
                parts.append(
                    "SHELL: For long-running commands, start them with `shell_exec(background=True, command='...')`, then poll with `shell_exec(job_id='...')` every few seconds until the job completes or the timeout window is reached. "
                    "Use the status updates to stay anchored to the original task."
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
            "then move directly to `file_write`, `file_append`, or `file_patch` instead of repeating `dir_list` on the same path. "
            "If you already know the destination directory, stop exploring and start writing."
        )
    plan = state.active_plan or state.draft_plan
    if plan is not None and state.plan_artifact_id:
        claim_bits = ""
        if getattr(plan, "claim_refs", None):
            claim_bits = " Claim refs: " + ", ".join(plan.claim_refs)
        parts.append(
            "\n\n### PLAN PLAYBOOK\n"
            f"Plan artifact: {state.plan_artifact_id}\n"
            "Treat this artifact as the source of truth for staged implementation. "
            "Before a large script change, complete the playbook in order: file skeleton, function signatures, code, then debug."
            f"{claim_bits}"
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
    if normalize_intent_label(getattr(state, "active_intent", "")) == "requested_write_file":
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
