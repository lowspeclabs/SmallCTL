from __future__ import annotations

from typing import Any

from ..retrieval_safety import build_retrieval_safe_text, format_failure_tag, sanitize_retrieval_text
from ..state import LoopState, MemoryEntry, memory_entry_is_stale, normalize_intent_label


def _dedupe_nonempty_texts(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _retrieval_failure_texts(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for text in values:
        normalized.append(format_failure_tag(text))
    return _dedupe_nonempty_texts(normalized)


def _is_recovery_nudge_message(message: Any) -> bool:
    metadata = getattr(message, "metadata", {})
    return isinstance(metadata, dict) and bool(metadata.get("is_recovery_nudge"))


def _retrieval_message_text(message: Any) -> str:
    retrieval_safe_text = str(getattr(message, "retrieval_safe_text", "") or "").strip()
    if retrieval_safe_text:
        return retrieval_safe_text
    return build_retrieval_safe_text(
        role=str(getattr(message, "role", "") or ""),
        content=getattr(message, "content", ""),
        name=getattr(message, "name", ""),
        metadata=getattr(message, "metadata", {}),
    )


def _is_execution_oriented_text(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "shell_exec",
            "ssh_exec",
            "run ",
            "execute",
            "exec ",
            "command",
            "script",
            "terminal",
            "pytest",
            "apt-get",
            "git ",
        )
    )


def build_retrieval_query(state: LoopState, *, retriever_cls: Any) -> str:
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    parts = [
        state.run_brief.original_task,
        state.run_brief.task_contract,
        state.run_brief.current_phase_objective,
    ]
    plan = state.active_plan or state.draft_plan
    if plan is not None:
        parts.append(f"Plan goal: {plan.goal}")
        parts.append(f"Plan status: {plan.status}")
        if plan.requested_output_path:
            parts.append(f"Plan export: {plan.requested_output_path}")
    if task_mode:
        parts.append(f"Task mode: {task_mode}")
    if state.active_intent:
        parts.append(f"Intent: {normalize_intent_label(state.active_intent)}")
    if state.intent_tags:
        parts.append(f"Tags: {' '.join(state.intent_tags)}")
    touched_symbols = state.scratchpad.get("_touched_symbols")
    if isinstance(touched_symbols, list):
        cleaned_symbols = [str(symbol).strip() for symbol in touched_symbols if str(symbol).strip()]
        if cleaned_symbols:
            parts.append("Touched symbols: " + " ".join(cleaned_symbols[:8]))
    current_goal = retriever_cls._effective_current_goal(state)
    if current_goal:
        parts.append(f"Current goal: {current_goal}")
    parts.extend(state.working_memory.plan[-3:])
    parts.extend(state.working_memory.decisions[-3:])
    parts.extend(
        _visible_memory_texts(
            state.working_memory.known_facts,
            state.working_memory.known_fact_meta,
            current_step=state.step_count,
            current_phase=state.current_phase,
        )[-4:]
    )
    parts.extend(state.working_memory.open_questions[-2:])
    parts.extend(
        _retrieval_failure_texts(
            _visible_memory_texts(
                state.working_memory.failures,
                state.working_memory.failure_meta,
                current_step=state.step_count,
                current_phase=state.current_phase,
            )[-4:]
        )
    )
    parts.extend(
        _visible_memory_texts(
            state.working_memory.next_actions,
            state.working_memory.next_action_meta,
            current_step=state.step_count,
            current_phase=state.current_phase,
        )[-3:]
    )
    if task_mode == "chat":
        parts = [part for part in parts if not _is_execution_oriented_text(part)]
    for content in _dedupe_nonempty_texts([
        _retrieval_message_text(message)
        for message in state.recent_messages[-3:]
        if not _is_recovery_nudge_message(message)
    ]):
        parts.append(content)
    return "\n".join(
        sanitize_retrieval_text(part) for part in parts if part
    )


def build_refined_retrieval_query(
    state: LoopState,
    *,
    base_query: str,
    bundle: Any,
    retriever_cls: Any,
) -> str:
    # In repair phase, avoid bloating the refined query with the full original task prompt.
    # Use the current goal / delta instead if the base query has grown excessively long.
    current_goal = retriever_cls._effective_current_goal(state)
    if state.current_phase == "repair" and len(base_query) > 2000 and current_goal:
        parts = [current_goal]
    else:
        parts = [base_query]
    if state.run_brief.task_contract:
        parts.append(f"Contract: {state.run_brief.task_contract}")
    current_goal = retriever_cls._effective_current_goal(state)
    if current_goal:
        parts.append(f"Current goal: {current_goal}")
    if bundle.artifacts:
        top_snippet = bundle.artifacts[0]
        top_artifact = state.artifacts.get(top_snippet.artifact_id)
        if top_artifact:
            parts.append(f"Top artifact: {top_artifact.artifact_id} | {top_artifact.source} | {top_artifact.summary}")
            if top_artifact.path_tags:
                parts.append("Artifact path tags: " + " ".join(top_artifact.path_tags))
            if top_artifact.tool_name:
                parts.append(f"Artifact tool: {top_artifact.tool_name}")
    if bundle.summaries:
        summary = bundle.summaries[0]
        if summary.files_touched:
            parts.append("Summary files: " + " ".join(summary.files_touched[:4]))
        if summary.remaining_plan:
            parts.append("Summary next steps: " + " ".join(summary.remaining_plan[:3]))
        if summary.notes:
            parts.append("Summary notes: " + " ".join(summary.notes[:2]))
    if bundle.experiences:
        memory = bundle.experiences[0]
        if not (
            memory.tool_name == "task_complete"
            and retriever_cls._query_requests_live_remote_correction(base_query)
        ) and not retriever_cls._is_generic_terminal_memory(state, memory):
            parts.append(
                f"Prior outcome: {normalize_intent_label(memory.intent)} / {memory.tool_name} / {memory.outcome}"
            )
            memory_namespace = retriever_cls._resolved_memory_namespace(memory, state=state)
            if (
                memory_namespace in {"ssh_remote", "local_shell", "planning", "debugging", "incidents"}
                and (
                    memory.tool_name in {"task_complete", "task_fail", "memory_update", "artifact_read", "file_read", "dir_list"}
                    or bundle.score_gaps.get("experiences", 999.0) <= 1.0
                )
            ):
                parts.append(f"Memory namespace: {memory_namespace}")
            if memory.failure_mode:
                parts.append(f"Failure mode: {memory.failure_mode}")
            visible_memory_tags = retriever_cls._prompt_visible_memory_tags(state, memory)
            if visible_memory_tags:
                parts.append("Memory tags: " + " ".join(visible_memory_tags[:4]))
    touched_symbols = state.scratchpad.get("_touched_symbols")
    if isinstance(touched_symbols, list):
        cleaned_symbols = [str(symbol).strip() for symbol in touched_symbols if str(symbol).strip()]
        if cleaned_symbols:
            parts.append("Touched symbols: " + " ".join(cleaned_symbols[:8]))
    if state.working_memory.open_questions:
        parts.append("Open questions: " + " ".join(state.working_memory.open_questions[-2:]))
    retrieval_failures = _retrieval_failure_texts(state.working_memory.failures[-2:])
    if retrieval_failures:
        parts.append("Recent failures: " + " ".join(retrieval_failures))
    if state.recent_messages:
        last_user = next(
            (
                message.content
                for message in reversed(state.recent_messages)
                if message.role == "user"
                and message.content
                and not _is_recovery_nudge_message(message)
            ),
            "",
        )
        if last_user:
            parts.append(f"Latest user context: {last_user[:240]}")
    return "\n".join(
        sanitize_retrieval_text(part) for part in parts if part
    )


def _visible_memory_texts(
    values: list[str],
    entries: list[MemoryEntry],
    *,
    current_step: int,
    current_phase: str,
) -> list[str]:
    visible: list[str] = []
    for index, text in enumerate(values):
        entry = entries[index] if index < len(entries) else None
        if entry is not None:
            if memory_entry_is_stale(
                entry,
                current_step=current_step,
                current_phase=current_phase,
            ):
                continue
            if entry.confidence is not None and entry.confidence < 0.6:
                continue
        visible.append(text)
    return visible
