from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
import time
from typing import Any

from ..client import OpenAICompatClient
from ..guards import check_guards
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..models.tool_result import ToolEnvelope
from ..phases import normalize_phase
from ..prompts import build_planning_prompt, build_system_prompt
from ..context.messages import _request_has_full_artifact_intent
from ..context.rendering import render_shell_output
from ..plans import write_plan_file
from ..state import clip_text_value, json_safe_value
from ..state import ExecutionPlan, PlanStep
from .deps import GraphRuntimeDeps
from .routing import LoopRoute
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord, build_operation_id
from ..memory.taxonomy import (
    TOOL_NOT_CALLED,
    PREMATURE_TASK_COMPLETE,
)
from ..tools.planning import plan_request_execution

_REPEATED_TOOL_HISTORY_LIMIT = 12
_IDENTICAL_TOOL_CALL_STREAK_LIMIT = 3
_REPEATED_TOOL_WINDOW = 6
_REPEATED_TOOL_UNIQUE_LIMIT = 3
_UI_TOOL_RESULT_PREVIEW_LIMIT = 1200
_UI_ARTIFACT_READ_PREVIEW_LIMIT = 900
_PROVIDER_HTTP_STATUS_CODES = {500, 502, 503, 504, 530}


class ToolNotFoundError(Exception):
    """Raised when a tool is requested but not found in the registry."""
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool not found: {tool_name}")


def _classify_model_call_error(exc: Exception) -> tuple[str, dict[str, Any]]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    details: dict[str, Any] = {}
    if isinstance(status_code, int):
        details["status_code"] = status_code
        if status_code in _PROVIDER_HTTP_STATUS_CODES:
            return "provider", details
    return "stream", details


def _planning_response_looks_like_plan(text: str) -> bool:
    normalized = (text or "").strip()
    if len(normalized) < 40:
        return False
    lowered = normalized.lower()
    markers = (
        "plan",
        "goal",
        "success criteria",
        "substep",
        "expected artifact",
        "ready for confirmation",
        "ready to proceed",
        "ready for approval",
    )
    if any(marker in lowered for marker in markers):
        return True
    return bool(re.search(r"^\|\s*\d+\s*\|", normalized, flags=re.MULTILINE))


def _is_plan_export_validation_error(error: str | None) -> bool:
    normalized = str(error or "").strip().lower()
    if not normalized:
        return False
    markers = (
        "refusing to write a plan",
        "plan export targets must use .md, .txt, or .text",
        "plan export path ending in .md requires markdown format",
        "plan export path ending in .txt/.text requires text format",
        "unsupported plan export format",
    )
    return any(marker in normalized for marker in markers)


def _build_plan_export_recovery_message(record: ToolExecutionRecord) -> str:
    requested_path = str(
        record.args.get("path")
        or record.args.get("output_path")
        or record.args.get("plan_output_path")
        or ""
    ).strip()
    if requested_path:
        suggested_path = str(Path(requested_path).with_suffix(".md"))
        return (
            "Plan export paths are only for plan documents (.md, .txt, .text). "
            f"Keep implementation targets like `{requested_path}` out of `plan_set` and `plan_export`; "
            f"continue planning without that export, or use `{suggested_path}` for the plan file instead."
        )
    return (
        "Plan exports only support markdown or text plan documents. "
        "Continue planning, and if you still want a plan file, use a `.md`, `.txt`, or `.text` path."
    )


def _extract_plan_steps_from_text(text: str) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        table_match = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|$", stripped)
        if table_match:
            step_id = f"P{table_match.group(1)}"
            title = table_match.group(2).strip()
            description = table_match.group(3).strip()
            steps.append(PlanStep(step_id=step_id, title=title, description=description))
            continue
        numbered_match = re.match(r"^\d+[.)]\s*(.+)$", stripped)
        if numbered_match:
            step_id = f"P{len(steps) + 1}"
            steps.append(PlanStep(step_id=step_id, title=numbered_match.group(1).strip()))
    return steps


def _synthesize_plan_from_text(harness: Any, text: str) -> ExecutionPlan | None:
    assistant_text = (text or "").strip()
    if not assistant_text or not _planning_response_looks_like_plan(assistant_text):
        return None
    goal = str(harness.state.run_brief.original_task or "").strip() or assistant_text.splitlines()[0].strip()
    steps = _extract_plan_steps_from_text(assistant_text)
    if not steps:
        steps = [PlanStep(step_id="P1", title="Review proposed plan")]
    return ExecutionPlan(
        plan_id=f"plan-{uuid.uuid4().hex[:8]}",
        goal=goal,
        summary=assistant_text,
        steps=steps[:6],
        status="draft",
        approved=False,
    )


async def _pause_for_plan_approval(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    question: str = "Plan ready. Execute it now?",
) -> None:
    harness = deps.harness
    plan = harness.state.active_plan or harness.state.draft_plan
    if plan is None:
        return
    await plan_request_execution(question=question, state=harness.state)
    payload = harness.state.pending_interrupt or {
        "kind": "plan_execute_approval",
        "question": question,
        "plan_id": plan.plan_id,
        "approved": False,
        "response_mode": "yes/no/revise",
        "current_phase": harness.state.current_phase,
        "thread_id": graph_state.thread_id,
    }
    graph_state.interrupt_payload = payload
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content=question,
            data={"status_activity": "awaiting plan approval...", "interrupt": payload},
        ),
    )
    graph_state.final_result = {
        "status": "needs_human",
        "message": question,
        "assistant": graph_state.last_assistant_text,
        "thinking": graph_state.last_thinking_text,
        "usage": graph_state.last_usage,
        "interrupt": payload,
    }


HALLUCINATION_MAP = {
    "file_read": "artifact_read",
    "grep": "summarize_report",
    "ls": "dir_list",
}


async def initialize_loop_run(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    task: str,
) -> None:
    harness = deps.harness
    if harness._cancel_requested:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
        )
        graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
        return
    harness.state.pending_interrupt = None
    if graph_state.run_mode == "chat":
        harness.state.scratchpad["_chat_rounds"] = 0
    else:
        harness.state.scratchpad.pop("_chat_rounds", None)
    harness.state.scratchpad.pop("_tool_attempt_history", None)
    harness._runlog("task_start", "task received", task=task)
    await harness._ensure_context_limit()
    harness._initialize_run_brief(task)
    harness._activate_tool_profiles(task)
    harness.state.append_message(ConversationMessage(role="user", content=task))
    harness._log_conversation_state("user_message")
    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.USER, content=task),
    )


async def resume_loop_run(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    human_input: str,
) -> None:
    harness = deps.harness
    pending = harness.state.pending_interrupt
    if not isinstance(pending, dict) or not pending:
        graph_state.final_result = harness._failure(
            "No pending interrupt to resume.",
            error_type="interrupt",
        )
        graph_state.error = graph_state.final_result["error"]
        return
    harness._runlog(
        "interrupt_resume",
        "resuming loop from interrupt",
        thread_id=graph_state.thread_id,
        interrupt=pending,
    )
    # Reset step counter if user explicitly asks to continue after a guard trip
    if human_input.strip().lower() in ("continue", "keep going", "proceed"):
        harness._runlog("step_count_reset", "resetting step count for continuation", old_count=harness.state.step_count)
        harness.state.step_count = 0
        
    harness.state.pending_interrupt = None
    for key in ("_ask_human", "_ask_human_question"):
        harness.state.scratchpad.pop(key, None)
    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=human_input,
            metadata={
                "resumed_from_interrupt": True,
                "interrupt_kind": pending.get("kind", "ask_human"),
            },
        )
    )
    harness._log_conversation_state("resume_user_message")
    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.USER, content=human_input),
    )


async def initialize_planning_run(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    task: str,
) -> None:
    harness = deps.harness
    await initialize_loop_run(graph_state, deps, task=task)
    harness.state.planning_mode_enabled = True
    harness.state.planner_resume_target_mode = "loop"
    harness.state.run_brief.current_phase_objective = f"planning: {task}" if task else "planning"
    if not harness.state.working_memory.current_goal:
        harness.state.working_memory.current_goal = task or harness.state.run_brief.original_task
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content="Planning mode enabled.",
            data={"status_activity": "planning mode active"},
        ),
    )


async def resume_planning_run(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    human_input: str,
) -> None:
    harness = deps.harness
    pending = harness.state.pending_interrupt
    if not isinstance(pending, dict) or not pending:
        graph_state.final_result = harness._failure(
            "No pending planning interrupt to resume.",
            error_type="interrupt",
        )
        graph_state.error = graph_state.final_result["error"]
        return

    harness.state.pending_interrupt = None
    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=human_input,
            metadata={
                "resumed_from_interrupt": True,
                "interrupt_kind": pending.get("kind", "plan_execute_approval"),
            },
        )
    )
    harness._log_conversation_state("resume_planning_message")
    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.USER, content=human_input),
    )

    lowered = human_input.strip().lower()
    if lowered in {"yes", "y", "approve", "approved", "execute", "go ahead", "run it"}:
        plan = harness.state.active_plan or harness.state.draft_plan
        if plan is not None:
            plan.approved = True
            plan.status = "approved"
            plan.touch()
            harness.state.active_plan = plan
            harness.state.draft_plan = plan
            harness.state.planning_mode_enabled = False
            harness.state.current_phase = "execute"
            harness.state.planner_resume_target_mode = "loop"
            harness.state.sync_plan_mirror()
            harness.state.touch()
            if plan.requested_output_path:
                try:
                    write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
                except ValueError as exc:
                    harness.log.warning("skipping invalid plan export during approval: %s", exc)
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Plan approved.",
                data={"status_activity": "plan approved"},
            ),
        )
        graph_state.final_result = {
            "status": "plan_approved",
            "message": "Plan approved.",
            "approved": True,
            "plan": json_safe_value(harness.state.active_plan or harness.state.draft_plan),
        }
        return

    harness.state.planning_mode_enabled = True
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content="Plan revision requested.",
            data={"status_activity": "awaiting plan revision..."},
        ),
    )


async def prepare_loop_step(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> None:
    harness = deps.harness
    start_time = time.perf_counter()
    if harness._cancel_requested:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
        )
        graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
        return
    if graph_state.run_mode == "chat":
        chat_rounds = _coerce_int_value(harness.state.scratchpad.get("_chat_rounds")) + 1
        harness.state.scratchpad["_chat_rounds"] = chat_rounds
        if chat_rounds > 6:
            graph_state.final_result = harness._failure(
                "Chat mode exceeded max tool rounds.",
                error_type="guard",
                details={"max_chat_tool_rounds": 6},
            )
            graph_state.error = graph_state.final_result["error"]
            return
    harness.state.step_count += 1
    harness.state.decay_experiences()
    harness.dispatcher.phase = normalize_phase(harness.state.current_phase)
    harness.state.current_phase = harness.dispatcher.phase

    if graph_state.pending_tool_calls:
        suppressed_plan_reads: list[str] = []
        remaining_calls: list[PendingToolCall] = []
        for pending in graph_state.pending_tool_calls:
            if _should_suppress_resolved_plan_artifact_read(harness, pending):
                artifact_id = _extract_artifact_id_from_args(pending.args)
                if artifact_id:
                    suppressed_plan_reads.append(artifact_id)
                continue
            remaining_calls.append(pending)
        if suppressed_plan_reads:
            graph_state.pending_tool_calls = remaining_calls
            for artifact_id in suppressed_plan_reads:
                if harness.state.scratchpad.get("_plan_artifact_read_suppressed") == artifact_id:
                    continue
                harness.state.scratchpad["_plan_artifact_read_suppressed"] = artifact_id
                harness.state.append_message(
                    ConversationMessage(
                        role="system",
                        content=(
                            f"Plan artifact {artifact_id} is already loaded in Working Memory. "
                            "Reuse the mirrored plan summary instead of rereading it."
                        ),
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "artifact_read",
                            "artifact_id": artifact_id,
                            "recovery_mode": "plan_mirror",
                        },
                    )
                )
                harness._runlog(
                    "artifact_read_suppressed",
                    "suppressed repeated plan artifact read",
                    artifact_id=artifact_id,
                )

    guard_error = check_guards(harness.state, harness.guards)
    if guard_error:
        recovery_hint = _artifact_read_recovery_hint(harness, guard_error)
        if (
            recovery_hint is not None
            and graph_state.run_mode != "chat"
        ):
            artifact_id, query = recovery_hint
            _clear_artifact_read_guard_state(harness, artifact_id)
            graph_state.pending_tool_calls = [
                PendingToolCall(
                    tool_name="artifact_grep",
                    args={
                        "artifact_id": artifact_id,
                        "query": query,
                    },
                    raw_arguments=json.dumps(
                        {"artifact_id": artifact_id, "query": query},
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                )
            ]
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=(
                        f"Auto-advancing repeated `artifact_read` on artifact {artifact_id} "
                        f"to `artifact_grep` with query `{query}`."
                    ),
                    metadata={
                        "recovery_kind": "artifact_read",
                        "artifact_id": artifact_id,
                        "query": query,
                        "recovery_mode": "direct_dispatch",
                    },
                )
            )
            harness._runlog(
                "artifact_read_recovery",
                "scheduled recovery dispatch",
                step=harness.state.step_count,
                artifact_id=artifact_id,
                query=query,
                guard_error=guard_error,
            )
            guard_error = None
        elif (
            recovery_hint is not None
            and graph_state.run_mode == "chat"
        ):
            recovery_armed = harness.state.scratchpad.get("_artifact_read_recovery_nudged")
            if recovery_armed != recovery_hint[0]:
                artifact_id, query = recovery_hint
                msg = (
                    f"You are repeating `artifact_read` on artifact {artifact_id}. "
                    f"Use `artifact_grep` with query `{query}` instead of reading the same artifact again."
                )
                harness.state.scratchpad["_artifact_read_recovery_nudged"] = artifact_id
                harness.state.scratchpad["_artifact_read_recovery_query"] = query
                harness.state.append_message(
                    ConversationMessage(
                        role="user",
                        content=msg,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "artifact_read",
                            "artifact_id": artifact_id,
                            "query": query,
                        },
                    )
                )
                harness._runlog(
                    "artifact_read_recovery",
                    "injected recovery nudge",
                    step=harness.state.step_count,
                    artifact_id=artifact_id,
                    query=query,
                    guard_error=guard_error,
                )
                guard_error = None

    if guard_error:
        harness.state.recent_errors.append(guard_error)
        log_kv(
            harness.log,
            logging.WARNING,
            "harness_guard_triggered",
            step=harness.state.step_count,
            guard_error=guard_error,
        )
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=guard_error),
        )
        graph_state.final_result = harness._failure(guard_error, error_type="guard")
        graph_state.error = graph_state.final_result["error"]
    
    graph_state.latency_metrics["overhead_preparation_duration_sec"] = round(time.perf_counter() - start_time, 3)


async def prepare_prompt(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> list[dict[str, Any]] | None:
    harness = deps.harness
    system_prompt = build_system_prompt(
        harness.state,
        harness.state.current_phase,
        available_tool_names=_available_tool_names(harness, mode="loop"),
        strategy_prompt=harness.strategy_prompt,
        manifest=load_index_manifest(harness.state.cwd),
    )
    try:
        return await harness._build_prompt_messages(system_prompt, event_handler=deps.event_handler)
    except RuntimeError as exc:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=str(exc)),
        )
        graph_state.final_result = harness._failure(str(exc), error_type="prompt_budget")
        graph_state.error = graph_state.final_result["error"]
        return None


async def prepare_chat_prompt(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> list[dict[str, Any]] | None:
    harness = deps.harness
    chat_tool_names = _available_tool_names(harness, mode="chat")
    system_prompt = build_system_prompt(
        harness.state,
        harness.state.current_phase,
        available_tool_names=chat_tool_names,
        strategy_prompt=harness.strategy_prompt,
        manifest=load_index_manifest(harness.state.cwd),
    )
    system_prompt = f"{system_prompt} You may use available tools when needed to answer accurately."
    if "shell_exec" in chat_tool_names:
        system_prompt = (
            f"{system_prompt} "
            "SHELL: `shell_exec` is available for command execution, but it requires user approval before execution."
        )
    try:
        return await harness._build_prompt_messages(system_prompt, event_handler=deps.event_handler)
    except RuntimeError as exc:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=str(exc)),
        )
        graph_state.final_result = harness._failure(str(exc), error_type="prompt_budget")
        graph_state.error = graph_state.final_result["error"]
        return None


async def prepare_planning_prompt(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> list[dict[str, Any]] | None:
    harness = deps.harness
    system_prompt = build_planning_prompt(
        harness.state,
        harness.state.current_phase,
        available_tool_names=_available_tool_names(harness, mode="planning"),
        strategy_prompt=harness.strategy_prompt,
        manifest=load_index_manifest(harness.state.cwd),
    )
    try:
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Gathering planning facts...",
                data={"status_activity": "gathering facts..."},
            ),
        )
        return await harness._build_prompt_messages(system_prompt, event_handler=deps.event_handler)
    except RuntimeError as exc:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=str(exc)),
        )
        graph_state.final_result = harness._failure(str(exc), error_type="prompt_budget")
        graph_state.error = graph_state.final_result["error"]
        return None


def load_index_manifest(cwd: str) -> dict[str, Any] | None:
    path = Path(cwd) / ".smallctl" / "index_manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def select_loop_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    harness = deps.harness
    profiles = set(harness.state.active_tool_profiles)
    phase = harness.state.current_phase
    tools = harness.registry.export_openai_tools(
        phase=phase,
        mode="loop",
        profiles=profiles,
    )
    harness.log.info("select_loop_tools: phase=%s profiles=%s count=%d", phase, profiles, len(tools))
    return tools


def select_chat_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    return deps.harness._chat_mode_tools()


def select_indexer_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    harness = deps.harness
    # Only allow core read tools and indexer write tools
    indexer_profiles = {"indexer", "core", "support"}
    return harness.registry.export_openai_tools(
        mode="indexer",
        profiles=indexer_profiles,
    )


def select_planning_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    harness = deps.harness
    return harness.registry.export_openai_tools(
        phase=harness.state.current_phase,
        mode="planning",
        profiles=set(harness.state.active_tool_profiles),
    )


async def prepare_indexer_prompt(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> list[dict[str, Any]] | None:
    harness = deps.harness
    system_prompt = (
        "You are a high-performance codebase indexer. Your goal is to systematically extract and record every symbol, reference, and import. "
        "EFFICIENCY IS CRITICAL: Use `index_batch_write` to submit ALL symbols, imports, and references for a file segment in a single tool call. "
        "Do not call individual write tools if you have multiple records to submit. "
        "PIPELINE: 1. List files. 2. Read file segments. 3. Extract all relevant metadata. 4. Call `index_batch_write` with the full collection. 5. Move to the next segment or file. "
        "Once all relevant files are indexed, call `index_finalize()`."
    )
    try:
        return await harness._build_prompt_messages(system_prompt, event_handler=deps.event_handler)
    except Exception as exc:
        graph_state.error = str(exc)
        return None


def _available_tool_names(harness: Any, *, mode: str) -> list[str]:
    if mode == "chat":
        tools = harness._chat_mode_tools()
    else:
        tools = harness.registry.export_openai_tools(
            phase=harness.state.current_phase,
            mode=mode,
            profiles=set(harness.state.active_tool_profiles),
        )
    return [
        str(entry["function"]["name"])
        for entry in tools
        if isinstance(entry, dict)
        and "function" in entry
        and isinstance(entry["function"], dict)
        and "name" in entry["function"]
    ]


async def model_call(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> None:
    harness = deps.harness
    chunks: list[dict[str, Any]] = []
    graph_state.pending_tool_calls = []
    graph_state.last_tool_results = []
    graph_state.last_assistant_text = ""
    graph_state.last_thinking_text = ""
    graph_state.last_usage = {}
    start_time = time.perf_counter()
    first_token_time: float | None = None
    # Streaming state machine for real-time tag-based reasoning detection
    inside_tag = False
    buffer = ""
    start_tag = str(harness.thinking_start_tag or "<think>")
    end_tag = str(harness.thinking_end_tag or "</think>")
    
    _CHUNK_ERROR_MAX_RETRIES = 2
    for _model_attempt in range(_CHUNK_ERROR_MAX_RETRIES + 1):
      try:
        chunks = []
        inside_tag = False
        buffer = ""
        async for event in harness.client.stream_chat(messages=messages, tools=tools):
            if harness._cancel_requested:
                await harness._emit(
                    deps.event_handler,
                    UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
                )
                graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
                return
            if event.get("type") == "chunk_error":
                err_msg = event.get("error", "unknown upstream error")
                details = event.get("details")
                retrying = _model_attempt < _CHUNK_ERROR_MAX_RETRIES
                harness._runlog(
                    "stream_chunk_error",
                    "upstream chunk error, will retry" if retrying else "upstream chunk error on final attempt",
                    error=err_msg,
                    attempt=_model_attempt + 1,
                    retrying=retrying,
                    details=details,
                )
                if retrying:
                    await harness._emit(
                        deps.event_handler,
                        UIEvent(
                            event_type=UIEventType.ALERT,
                            content=f"Stream chunk error (retrying): {err_msg}",
                            data={
                                "is_api_error": True,
                                "retrying": True,
                                "attempt": _model_attempt + 1,
                                "details": details,
                            },
                        ),
                    )
                break  # break inner loop → retry outer loop
            if event.get("type") == "chunk":
                data = event.get("data", {})
                choices = data.get("choices") or []
                if not choices:
                    chunks.append(event)
                    continue
                delta = choices[0].get("delta", {})
                reason_field = delta.get("reasoning_content") or delta.get("reasoning")
                content_field = delta.get("content")
                
                if content_field or reason_field:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                
                # A) Explicit reasoning field
                if reason_field:
                    if harness.thinking_visibility:
                        await harness._emit(
                            deps.event_handler,
                            UIEvent(event_type=UIEventType.THINKING, content=reason_field),
                        )
                    harness._runlog("model_token", "thinking token", token=reason_field)
                
                # B) Content-based reasoning with tags
                if content_field:
                    pending = buffer + content_field
                    buffer = ""
                    while pending:
                        if not inside_tag:
                            st_idx = pending.find(start_tag)
                            if st_idx == -1:
                                maybe_part = False
                                for i in range(1, len(start_tag)):
                                    if pending.endswith(start_tag[:i]):
                                        buffer = start_tag[:i]
                                        emittable = pending[:-i]
                                        if emittable:
                                            await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.ASSISTANT, content=emittable))
                                            harness._stream_print(emittable)
                                            harness._runlog("model_token", "assistant token", token=emittable)
                                        maybe_part = True
                                        break
                                if not maybe_part:
                                    await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.ASSISTANT, content=pending))
                                    harness._stream_print(pending)
                                    harness._runlog("model_token", "assistant token", token=pending)
                                pending = ""
                            else:
                                prefix = pending[:st_idx]
                                if prefix:
                                    await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.ASSISTANT, content=prefix))
                                    harness._stream_print(prefix)
                                    harness._runlog("model_token", "assistant token", token=prefix)
                                inside_tag = True
                                pending = pending[st_idx + len(start_tag):]
                        else:
                            et_idx = pending.find(end_tag)
                            if et_idx == -1:
                                maybe_part = False
                                for i in range(1, len(end_tag)):
                                    if pending.endswith(end_tag[:i]):
                                        buffer = end_tag[:i]
                                        emittable = pending[:-i]
                                        if emittable:
                                            if harness.thinking_visibility:
                                                await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.THINKING, content=emittable))
                                            harness._runlog("model_token", "thinking token", token=emittable)
                                        maybe_part = True
                                        break
                                if not maybe_part:
                                    if harness.thinking_visibility:
                                        await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.THINKING, content=pending))
                                    harness._runlog("model_token", "thinking token", token=pending)
                                pending = ""
                            else:
                                thought = pending[:et_idx]
                                if thought:
                                    if harness.thinking_visibility:
                                        await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.THINKING, content=thought))
                                    harness._runlog("model_token", "thinking token", token=thought)
                                inside_tag = False
                                pending = pending[et_idx + len(end_tag):]
            chunks.append(event)
        else:
            # for-loop completed without a break (no chunk_error) → stream finished cleanly
            # Flush any leftover buffer from the stream
            if buffer:
                kind = UIEventType.THINKING if inside_tag else UIEventType.ASSISTANT
                if kind == UIEventType.THINKING:
                    if harness.thinking_visibility:
                        await harness._emit(deps.event_handler, UIEvent(event_type=kind, content=buffer))
                    harness._runlog("model_token", "thinking token", token=buffer)
                else:
                    await harness._emit(deps.event_handler, UIEvent(event_type=kind, content=buffer))
                    harness._stream_print(buffer)
                    harness._runlog("model_token", "assistant token", token=buffer)
            break
        # chunk_error caused the inner-loop break; wait then retry
        if _model_attempt < _CHUNK_ERROR_MAX_RETRIES:
            await asyncio.sleep(float(_model_attempt + 1))
      except asyncio.CancelledError:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
        )
        raise
      except Exception as exc:
        harness.log.exception("stream_chat failed")
        log_kv(harness.log, logging.ERROR, "harness_stream_error", error=str(exc))
        error_type, details = _classify_model_call_error(exc)
        is_api = error_type == "provider"
        content_prefix = "Provider error" if is_api else "Stream error"
        
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ERROR, 
                content=f"{content_prefix}: {exc}",
                data={"is_api_error": is_api}
            ),
        )
        err_msg = str(exc) or type(exc).__name__
        graph_state.final_result = harness._failure(err_msg, error_type=error_type, details=details)
        graph_state.error = graph_state.final_result["error"]
        return
    else:
        # Exhausted all chunk_error retries
        harness._runlog("stream_chunk_error_exhausted", "all chunk error retries exhausted")
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content="Stream error: upstream chunk errors exhausted all retries"),
        )
        graph_state.final_result = harness._failure("Upstream chunk error after retries", error_type="stream")
        graph_state.error = graph_state.final_result["error"]
        return
    if harness.thinking_visibility:
        print()
    stream = OpenAICompatClient.collect_stream(
        chunks,
        reasoning_mode=harness.reasoning_mode,
        thinking_start_tag=harness.thinking_start_tag,
        thinking_end_tag=harness.thinking_end_tag,
    )
    timeline = OpenAICompatClient.collect_timeline(
        chunks,
        reasoning_mode=harness.reasoning_mode,
        thinking_start_tag=harness.thinking_start_tag,
        thinking_end_tag=harness.thinking_end_tag,
    )
    if not harness.thinking_visibility and stream.assistant_text:
        print(stream.assistant_text)
    usage_payload = json_safe_value(stream.usage)
    if not isinstance(usage_payload, dict):
        usage_payload = {}
    if usage_payload:
        harness._apply_usage(usage_payload)

    graph_state.last_usage = usage_payload
    graph_state.last_assistant_text = stream.assistant_text
    graph_state.last_thinking_text = stream.thinking_text
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else duration
    
    graph_state.latency_metrics["model_call_duration_sec"] = round(duration, 3)
    graph_state.latency_metrics["ttft_sec"] = round(ttft, 3)
    
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.METRICS,
            content=f"Model call: {duration:.2f}s (TTFT: {ttft:.2f}s)",
            data={
                "duration_sec": duration,
                "ttft_sec": ttft,
                "usage": usage_payload,
            }
        ),
    )
    
    # Extract Native tool calls
    native_calls = [
        pending
        for payload in stream.tool_calls
        if (pending := PendingToolCall.from_payload(payload)) is not None
    ]
    
    # Extract Inline tool calls (from text body)
    cleaned_text, inline_calls = _extract_inline_tool_calls(stream.assistant_text)
    
    pending_calls = native_calls + inline_calls
    
    # Tool Deduplication with Tool-Specific Policies:
    # 1. Terminal Tools (task_complete, task_fail): First instance wins, ignore subsequent ones.
    # 2. Action Tools: Exact matching (fingerprint of name + sorted arguments).
    # Native calls take priority as they appear first in the list.
    TERMINAL_TOOLS = {"task_complete", "task_fail"}
    seen_fingerprints = set()
    terminal_called = False
    unique_calls: list[PendingToolCall] = []
    
    for call in pending_calls:
        if not call.tool_name:
            continue
            
        is_terminal = call.tool_name in TERMINAL_TOOLS
        
        # Policy for Terminal Tools: Only one completion/failure per turn
        if is_terminal:
            if terminal_called:
                harness._runlog(
                    "tool_deduplication", 
                    "redundant terminal tool call ignored", 
                    tool_name=call.tool_name,
                    source="inline_or_repeat"
                )
                continue
            terminal_called = True
            unique_calls.append(call)
            continue

        # Policy for Action Tools: Exact match fingerprint
        try:
            args_fingerprint = json.dumps(call.args or {}, sort_keys=True)
            fingerprint = f"{call.tool_name}:{args_fingerprint}"
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_calls.append(call)
            else:
                harness._runlog(
                    "tool_deduplication", 
                    "duplicate action tool call suppressed", 
                    tool_name=call.tool_name,
                    fingerprint=fingerprint
                )
        except (TypeError, ValueError):
            # Fallback for non-serializable args
            fingerprint = f"{call.tool_name}:{str(call.args)}"
            if fingerprint not in seen_fingerprints:
                seen_fingerprints.add(fingerprint)
                unique_calls.append(call)
    
    pending_calls = unique_calls

    # STEP 5: Triple-Answer Guard (Progressive/Stability Layer)
    # If the model emits task_complete with a message that matches assistant_text,
    # we strip the assistant_text to avoid redundancy in the logs.
    from ..guards import apply_triple_answer_guard
    final_assistant_text = apply_triple_answer_guard(cleaned_text, pending_calls)

    if not final_assistant_text.strip() and not pending_calls:
        final_assistant_text = _clean_reasoning_fallback_text(stream.thinking_text)

    graph_state.pending_tool_calls = pending_calls
    graph_state.last_assistant_text = final_assistant_text.strip()

    # STEP 6: Post-stream correction.
    # Tokens were already emitted to the UI incrementally during streaming.
    # If inline tool calls were stripped from the text, we need to overwrite
    # the UI assistant bubble with the cleaned version.
    if final_assistant_text.strip() != stream.assistant_text.strip():
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ASSISTANT,
                content=final_assistant_text.strip(),
                data={"kind": "replace"},
            ),
        )
    
    # LAZINESS TRACKING (Baseline)
    if not pending_calls:
        harness.state.inactive_steps += 1
        harness.state.scratchpad["_consecutive_idle"] = int(harness.state.scratchpad.get("_consecutive_idle", 0)) + 1
    else:
        harness.state.scratchpad["_consecutive_idle"] = 0
        
    if int(harness.state.scratchpad.get("_consecutive_idle", 0)) >= 2:
        nudge = (
            "\n[SYSTEM NUDGE]: You have provided 2 consecutive turns without any tool actions. "
            "Please focus on making concrete progress towards the goal (explore/execute) "
            "rather than providing high-level summaries or explanation. "
            "If you are finished, use the task_complete tool."
        )
        harness.state.append_message(ConversationMessage(role="system", content=nudge))
        # Keep it at 1 so the next turn lazy nudges again
        harness.state.scratchpad["_consecutive_idle"] = 1
    
    if final_assistant_text:
        harness._runlog(
            "model_output",
            "assistant output complete",
            assistant_text=final_assistant_text,
        )
    if final_assistant_text or stream.tool_calls:
        harness._record_assistant_message(
            assistant_text=final_assistant_text,
            tool_calls=stream.tool_calls,
        )
        harness._log_conversation_state("assistant_message")
    if stream.thinking_text:
        harness._runlog(
            "model_thinking",
            "thinking output complete",
            thinking_text=stream.thinking_text,
        )
    for entry in timeline:
        if entry.kind == "tool_call":
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.TOOL_CALL,
                    content=entry.content,
                    data=entry.data,
                ),
            )


def _clean_reasoning_fallback_text(text: str) -> str:
    if not text:
        return ""
    import re

    cleaned = re.sub(
        r"</?(?:tool_call|tool_code|call|function|parameter|thinking|reasoning)(?:=[^>]+)?>",
        "",
        str(text),
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


async def interpret_model_output(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    strategy = harness.state.scratchpad.get("strategy", {})
    thought_arch = strategy.get("thought_architecture")
    
    if graph_state.final_result is not None:
        return LoopRoute.FINALIZE

    summarizer_client = getattr(harness, "summarizer_client", None)
    if (
        harness.summarizer
        and graph_state.last_thinking_text
        and len(graph_state.last_thinking_text) > 800
        and summarizer_client
    ):
        
        # Throttling: only distill if we haven't done it too much in this run
        if int(harness.state.scratchpad.get("_distill_count", 0)) < 4:
            harness.state.scratchpad["_distill_count"] = int(harness.state.scratchpad.get("_distill_count", 0)) + 1
            insight = await harness.summarizer.distill_thinking_async(
                client=summarizer_client,
                thinking_text=graph_state.last_thinking_text,
                task=harness.state.run_brief.original_task,
            )
            if insight:
                harness._record_experience(
                    tool_name="reasoning",
                    result=ToolEnvelope(success=True, output=insight),
                    source="summarized",
                    notes=insight,
                )
                # Attach for pruning in history
                if harness.state.recent_messages:
                    last_msg = harness.state.recent_messages[-1]
                    if last_msg.role == "assistant":
                        last_msg.metadata["thinking_insight"] = insight

    if thought_arch == "multi_phase_discovery":
        current_phase = harness.state.current_phase
        if graph_state.pending_tool_calls:
            # If in explore, only allow gathering tools. Block task_complete/file_write.
            if current_phase == "explore":
                blocked = ["task_complete", "task_fail", "file_write"]
                original_calls = list(graph_state.pending_tool_calls)
                graph_state.pending_tool_calls = [c for c in original_calls if c.tool_name not in blocked]
                
                # If the model ONLY called blocked completion tools, it's effectively a 'no tool' completion attempt.
                if original_calls and not graph_state.pending_tool_calls:
                    if all(c.tool_name in ["task_complete", "task_fail"] for c in original_calls):
                        # Depth Lever: Rejection if too early
                        if harness.state.step_count < harness.config.min_exploration_steps:
                             harness.state.append_message(ConversationMessage(
                                 role="user",
                                 content=f"ANTI-LAZINESS: You are trying to finish at step {harness.state.step_count}, but this task requires at least {harness.config.min_exploration_steps} discovery steps. Perform more deep-dive exploration before concluding."
                             ))
                             return LoopRoute.NEXT_STEP

                        # Relax: If it's trying to finish, it probably has the info.
                        # Allow transition to verify if it hasn't happened yet.
                        if current_phase == "explore":
                             harness.state.current_phase = "verify"
                             harness._runlog("phase_transition", "auto-transition to VERIFICATION via premature completion attempt")
                             # Fall through to transition logic below
                if len(graph_state.pending_tool_calls) < len(original_calls):
                    harness.state.append_message(ConversationMessage(
                        role="user",
                        content="You are still in the DISCOVERY phase. Gathering complete? Call `memory_update(section='known_facts', content='...')` to transition to VERIFICATION."
                    ))
                    harness.state.scratchpad["_no_tool_nudges"] = int(harness.state.scratchpad.get("_no_tool_nudges", 0)) + 1
                    return LoopRoute.NEXT_STEP
            
            # If in verify, only allow memory_update
            if current_phase == "verify":
                 blocked = ["task_complete", "task_fail", "file_write", "long_context_lookup", "summarize_report", "artifact_read", "grep"]
                 if any(c.tool_name in blocked for c in graph_state.pending_tool_calls):
                     graph_state.pending_tool_calls = [c for c in graph_state.pending_tool_calls if c.tool_name not in blocked]
                     harness.state.append_message(ConversationMessage(
                        role="user",
                        content="You are in VERIFICATION. List all required constants via `memory_update` then proceed to SYNTHESIS."
                    ))
                     harness.state.scratchpad["_no_tool_nudges"] = int(harness.state.scratchpad.get("_no_tool_nudges", 0)) + 1
                     return LoopRoute.NEXT_STEP

        # Transition logic based on results
        for record in graph_state.last_tool_results:
            if record.tool_name == "memory_update" and record.result.success:
                if current_phase == "explore":
                    harness.state.current_phase = "verify"
                    harness._runlog("phase_transition", "transition to VERIFICATION")
                    harness.state.append_message(ConversationMessage(
                        role="user",
                        content="Transitioning to VERIFICATION phase. Please list/verify all constants."
                    ))
                    return LoopRoute.NEXT_STEP
                elif current_phase == "verify":
                    harness.state.current_phase = "execute"
                    harness._runlog("phase_transition", "transition to SYNTHESIS")
                    harness.state.append_message(ConversationMessage(
                        role="user",
                        content="Transitioning to SYNTHESIS phase. You may now implement the final answer and call task_complete."
                   ))
                    return LoopRoute.NEXT_STEP

    if graph_state.pending_tool_calls:
        for pending in graph_state.pending_tool_calls:
            missing_args = _detect_empty_file_write_payload(harness, pending)
            if missing_args is None:
                missing_args = _detect_missing_required_tool_arguments(harness, pending)
            if missing_args is None:
                continue
            err_msg, details = missing_args
            repair_attempts = int(harness.state.scratchpad.get("_schema_validation_nudges", 0))
            if repair_attempts >= 1:
                harness.state.recent_errors.append(err_msg)
                harness._runlog(
                    "tool_call_validation_error",
                    "tool call missing required arguments",
                    tool_name=pending.tool_name,
                    tool_call_id=pending.tool_call_id,
                    required_fields=details.get("required_fields", []),
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ERROR,
                        content=err_msg,
                        data={"error_type": "schema_validation_error", **details},
                    ),
                )
                graph_state.pending_tool_calls = []
                graph_state.final_result = harness._failure(
                    err_msg,
                    error_type="schema_validation_error",
                    details=details,
                )
                graph_state.error = graph_state.final_result["error"]
                return LoopRoute.FINALIZE

            harness.state.scratchpad["_schema_validation_nudges"] = repair_attempts + 1
            harness.state.recent_errors.append(err_msg)
            repair_message = err_msg
            if not details.get("offending_field"):
                repair_message = _build_schema_repair_message(
                    pending.tool_name,
                    details.get("required_fields", []),
                )
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=repair_message,
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "schema_validation",
                        "tool_name": pending.tool_name,
                        "required_fields": details.get("required_fields", []),
                        "tool_call_id": pending.tool_call_id,
                    },
                )
            )
            harness._runlog(
                "tool_call_repair",
                "injected schema repair nudge",
                tool_name=pending.tool_name,
                tool_call_id=pending.tool_call_id,
                required_fields=details.get("required_fields", []),
                retry_count=repair_attempts + 1,
            )
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content=repair_message,
                    data={
                        "repair_kind": "schema_validation",
                        "tool_name": pending.tool_name,
                        "tool_call_id": pending.tool_call_id,
                        "required_fields": details.get("required_fields", []),
                        "retry_count": repair_attempts + 1,
                    },
                ),
            )
            graph_state.pending_tool_calls = []
            graph_state.last_assistant_text = ""
            graph_state.last_thinking_text = ""
            return LoopRoute.NEXT_STEP

        return LoopRoute.DISPATCH_TOOLS

    nudges = int(harness.state.scratchpad.get("_no_tool_nudges", 0))
    assistant_text = graph_state.last_assistant_text or ""

    if (
        graph_state.run_mode == "planning"
        or harness.state.planning_mode_enabled
    ):
        synthesized_plan = _synthesize_plan_from_text(harness, assistant_text)
        if synthesized_plan is not None:
            harness.state.draft_plan = synthesized_plan
            harness.state.active_plan = synthesized_plan
            harness.state.planning_mode_enabled = True
            harness.state.sync_plan_mirror()
            harness.state.touch()
            await _pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE
        plan = harness.state.active_plan or harness.state.draft_plan
        if plan is not None and plan.status != "approved":
            await _pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE

    planning_request = harness._extract_planning_request(harness.state.run_brief.original_task)
    if (
        graph_state.pending_tool_calls == []
        and planning_request is not None
        and _planning_response_looks_like_plan(assistant_text)
    ):
        synthesized_plan = _synthesize_plan_from_text(harness, assistant_text)
        if synthesized_plan is not None:
            harness.state.draft_plan = synthesized_plan
            harness.state.active_plan = synthesized_plan
            harness.state.planning_mode_enabled = True
            harness.state.sync_plan_mirror()
            harness.state.touch()
            await _pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE
        plan = harness.state.active_plan or harness.state.draft_plan
        if plan is not None:
            await _pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE

    if assistant_text:
        # Let a substantial, tool-backed prose answer finalize before action-format heuristics can
        # mistake its reasoning text for a missing tool call.
        has_facts = bool(harness.state.working_memory.known_facts)
        has_tool_evidence = any(
            message.role == "tool"
            for message in harness.state.recent_messages[-6:]
        )
        if (
            nudges == 0
            and has_facts
            and has_tool_evidence
            and len(assistant_text) > 120
        ):
            harness._runlog(
                "auto_finalize",
                "prose answer with tool evidence; skipping nudge",
                text_len=len(assistant_text),
            )
            graph_state.final_result = {
                "status": "completed",
                "message": {
                    "status": "complete",
                    "message": assistant_text[:500],
                },
                "assistant": assistant_text,
            }
            return LoopRoute.FINALIZE

    # Action Stall Guard: Model justified an action but didn't emit the JSON.
    _ACTION_KEYWORDS = ["call", "run", "execute", "use", "using", "invok", "command", "tool"]
    _HTML_TOOL_TAGS = ["<tool_call>", "<function=", "<parameter="]
    _FUNC_SYNTAX = [f"{t}(" for t in ["shell_exec", "artifact_read", "file_read", "dir_list", "task_complete"]]
    
    low_text = assistant_text.lower()
    thinking_looks_like_action = any(kw in graph_state.last_thinking_text.lower() for kw in _ACTION_KEYWORDS)
    text_looks_like_action_list = any(kw in low_text for kw in _ACTION_KEYWORDS)
    text_has_tool_tags = any(tag in low_text for tag in _HTML_TOOL_TAGS)
    text_has_func_calls = any(fn in low_text for fn in _FUNC_SYNTAX)
    
    if not graph_state.pending_tool_calls and (thinking_looks_like_action or text_looks_like_action_list or text_has_tool_tags or text_has_func_calls):
        if graph_state.run_mode == "planning" or harness.state.planning_mode_enabled:
            synthesized_plan = _synthesize_plan_from_text(harness, assistant_text)
            if synthesized_plan is not None:
                harness.state.draft_plan = synthesized_plan
                harness.state.active_plan = synthesized_plan
                harness.state.planning_mode_enabled = True
                harness.state.sync_plan_mirror()
                harness.state.touch()
                await _pause_for_plan_approval(graph_state, deps)
                return LoopRoute.FINALIZE
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None:
                await _pause_for_plan_approval(graph_state, deps)
                return LoopRoute.FINALIZE
        stalls = int(harness.state.scratchpad.get("_action_stalls", 0))
        if stalls < 1:
             harness.state.scratchpad["_action_stalls"] = stalls + 1
             msg = "### SYSTEM ALERT: You identified or described a tool action, but you did not emit the JSON tool call."
             if text_has_tool_tags or text_has_func_calls:
                 msg = "### FORMAT ERROR: You used text-based tool tags or functional syntax (e.g. <tool_call> or shell_exec()). This is FORBIDDEN. You MUST use the JSON block format."
             
             harness.state.append_message(ConversationMessage(
                 role="user",
                 content=f"{msg}\n\nDO NOT repeat your earlier findings or analysis. Just generate the JSON block immediately after your reasoning. Do not describe what you are going to do; just DO it."
             ))
             harness._record_experience(
                 tool_name="reasoning",
                 result=ToolEnvelope(success=False, error=msg),
                 source="guarded_stall",
                 notes=f"Model described action but missed JSON format. Failure mode: {TOOL_NOT_CALLED}",
             )
             harness._runlog("action_stall", "improper tool format or description", stalls=stalls+1, has_tags=text_has_tool_tags)
             return LoopRoute.NEXT_STEP
             
    # Guard against premature success for "hello" when real work was requested
    if not graph_state.pending_tool_calls and "hello" in low_text and ("task" in low_text or "complete" in low_text):
        if any(v in harness.state.run_brief.original_task.lower() for v in ["ping", "list", "read", "run"]):
             harness.state.append_message(ConversationMessage(
                 role="user",
                 content="### MISSION CHECK: You mention 'hello' or completing the greeting, but a real task is still pending. DO NOT finish yet. Proceed with the primary mission."
             ))
             harness._record_experience(
                 tool_name="task_complete",
                 result=ToolEnvelope(success=False, error="Blocked hello completion attempt"),
                 source="guarded_completion",
                 notes=f"Model attempted 'hello' completion while mission pending. Failure mode: {PREMATURE_TASK_COMPLETE}",
             )
             harness._runlog("premature_completion_blocked", "blocked hello completion", task=harness.state.run_brief.original_task)
             return LoopRoute.NEXT_STEP

    if nudges < 4 and assistant_text:
        harness.state.scratchpad["_no_tool_nudges"] = nudges + 1
        msg = (
            "You reached a conclusion but did not call `task_complete`. "
            "If you are finished, you MUST call `task_complete(message='...')` with your final answer. "
            "Do not repeat your earlier analysis; simply emit the tool call."
        )
        if nudges >= 2:
            msg = "REPEAT WARNING: You are stuck in a loop. You MUST call `task_complete` NOW to save your progress."
            
        harness.state.append_message(
            ConversationMessage(
                role="user",
                content=msg,
                metadata={"is_recovery_nudge": True}
            )
        )
        harness._runlog("no_tool_recovery", "injected recovery nudge", nudge_count=nudges+1)
        return LoopRoute.NEXT_STEP
        
    # If we are stuck but already have tool evidence, finalize with the current answer.
    if nudges >= 4 and graph_state.last_assistant_text and (
        harness.state.current_phase == "execute"
        or any(message.role == "tool" for message in harness.state.recent_messages[-6:])
        or bool(harness.state.working_memory.known_facts)
    ):
        harness._runlog("recovery", "finalizing after multiple no-tool nudges")
        harness.state.scratchpad["_task_complete"] = True
        harness.state.scratchpad["_task_complete_message"] = graph_state.last_assistant_text[:500]
        harness.state.touch()
        graph_state.final_result = {
            "status": "completed",
            "message": {
                "status": "complete",
                "message": graph_state.last_assistant_text[:500],
            },
            "assistant": graph_state.last_assistant_text,
        }
        return LoopRoute.FINALIZE

    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.SYSTEM, content="No tool calls returned; stopping loop."),
    )
    graph_state.final_result = {
        "status": "stopped",
        "reason": "no_tool_calls",
        "assistant": graph_state.last_assistant_text,
        "thinking": graph_state.last_thinking_text,
    }
    return LoopRoute.FINALIZE


async def interpret_chat_output(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    del deps
    if graph_state.pending_tool_calls:
        return LoopRoute.DISPATCH_TOOLS
    graph_state.final_result = {
        "status": "chat_completed",
        "assistant": graph_state.last_assistant_text,
        "thinking": graph_state.last_thinking_text,
        "usage": graph_state.last_usage,
    }
    return LoopRoute.FINALIZE


async def interpret_planning_output(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    if graph_state.final_result is not None:
        return LoopRoute.FINALIZE
    if graph_state.pending_tool_calls:
        return LoopRoute.DISPATCH_TOOLS

    plan = harness.state.active_plan or harness.state.draft_plan
    if plan is None:
        synthesized_plan = _synthesize_plan_from_text(harness, graph_state.last_assistant_text)
        if synthesized_plan is not None:
            harness.state.draft_plan = synthesized_plan
            harness.state.active_plan = synthesized_plan
            harness.state.planning_mode_enabled = True
            harness.state.sync_plan_mirror()
            harness.state.touch()
            await _pause_for_plan_approval(graph_state, deps)
            return LoopRoute.FINALIZE

        harness.state.append_message(
            ConversationMessage(
                role="user",
                content=(
                    "Planning mode is active. Create a structured plan with `plan_set` before trying to execute anything."
                ),
                metadata={"is_recovery_nudge": True, "planner_nudge": True},
            )
        )
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Planning nudge issued.",
                data={"status_activity": "gathering facts..."},
            ),
        )
        return LoopRoute.NEXT_STEP

    if plan.status != "approved":
        await _pause_for_plan_approval(graph_state, deps)
        return LoopRoute.FINALIZE

    graph_state.final_result = {
        "status": "plan_ready",
        "assistant": graph_state.last_assistant_text,
        "thinking": graph_state.last_thinking_text,
        "usage": graph_state.last_usage,
    }
    return LoopRoute.FINALIZE


async def dispatch_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> None:
    harness = deps.harness
    graph_state.last_tool_results = []
    dispatch_start = time.perf_counter()
    for pending in graph_state.pending_tool_calls:
        repeat_error = _detect_repeated_tool_loop(harness, pending)
        if repeat_error is not None:
            recovered = _fallback_repeated_artifact_read(harness, pending)
            if recovered is not None:
                log_kv(
                    harness.log,
                    logging.INFO,
                    "harness_repeated_tool_loop_recovered",
                    step=harness.state.step_count,
                    original_tool_name=pending.tool_name,
                    recovered_tool_name=recovered.tool_name,
                    recovered_args=recovered.args,
                )
                pending = recovered
            else:
                harness.state.recent_errors.append(repeat_error)
                log_kv(
                    harness.log,
                    logging.WARNING,
                    "harness_repeated_tool_loop",
                    step=harness.state.step_count,
                    tool_name=pending.tool_name,
                    arguments=pending.args,
                    error=repeat_error,
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(event_type=UIEventType.ERROR, content=repeat_error),
                )
                graph_state.pending_tool_calls = []
                graph_state.final_result = harness._failure(
                    repeat_error,
                    error_type="guard",
                    details={
                        "tool_name": pending.tool_name,
                        "arguments": json_safe_value(pending.args),
                        "guard": "repeated_tool_loop",
                    },
                )
                graph_state.error = graph_state.final_result["error"]
                return

        hallucination_hint = _detect_hallucinated_tool_call(harness, pending)
        if hallucination_hint:
            log_kv(harness.log, logging.WARNING, "harness_hallucinated_tool_call", tool_name=pending.tool_name)
            await harness._emit(deps.event_handler, UIEvent(event_type=UIEventType.SYSTEM, content=hallucination_hint))
            
            graph_state.last_tool_results.append(
                ToolExecutionRecord(
                    operation_id=f"hallucination:{pending.tool_name}",
                    tool_name=pending.tool_name,
                    args=pending.args,
                    tool_call_id=pending.tool_call_id,
                    result=fake_result,
                )
            )
            continue

        _record_tool_attempt(harness, pending)
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.SYSTEM, content=f"Invoking {pending.tool_name}..."),
        )
        operation_id = build_operation_id(
            thread_id=graph_state.thread_id,
            step_count=harness.state.step_count,
            tool_call_id=pending.tool_call_id,
            tool_name=pending.tool_name,
        )
        existing = _get_tool_execution_record(harness, operation_id)
        replayed = isinstance(existing.get("result"), dict)
        log_kv(
            harness.log,
            logging.INFO,
            "harness_tool_dispatch",
            tool_name=pending.tool_name,
            replayed=replayed,
        )
        if replayed:
            harness._runlog(
                "tool_replay_hit",
                "reusing recorded tool result",
                tool_name=pending.tool_name,
                operation_id=operation_id,
            )
            result = _tool_envelope_from_dict(existing["result"])
        else:
            try:
                # Check for registry presence to satisfy the 'catch ToolNotFoundError' requirement
                if pending.tool_name not in harness.registry.names():
                    raise ToolNotFoundError(pending.tool_name)

                harness._active_dispatch_task = asyncio.create_task(
                    harness._dispatch_tool_call(pending.tool_name, pending.args)
                )
                result = await harness._active_dispatch_task
            except ToolNotFoundError:
                if pending.tool_name in HALLUCINATION_MAP:
                    mapped_tool = HALLUCINATION_MAP[pending.tool_name]
                    # Attempt to extract an ID-like string from arguments to give a better hint
                    raw_id = (
                        pending.args.get("path") or 
                        pending.args.get("artifact_id") or 
                        pending.args.get("pattern") or 
                        "A000X"
                    )
                    # Leniency: if they passed a path like 'A0001', use it as the ID
                    artifact_id = str(raw_id).split("/")[-1]
                    if not artifact_id.startswith("A") and "A" in artifact_id:
                        # try to find the A prefix
                        idx = artifact_id.find("A")
                        artifact_id = artifact_id[idx:]

                    hint = f"Tool '{pending.tool_name}' is unavailable. Use '{mapped_tool}(artifact_id=\"{artifact_id}\")' instead."
                    result = ToolEnvelope(
                        success=True,
                        output=hint,
                        metadata={"interceptor_hit": True, "hallucinated_tool": pending.tool_name}
                    )
                else:
                    # Not in map, fall back to standard failure
                    result = ToolEnvelope(
                        success=False,
                        error=f"Unknown tool: {pending.tool_name}",
                        metadata={"tool_name": pending.tool_name}
                    )
            except asyncio.CancelledError:
                await harness._emit(
                    deps.event_handler,
                    UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
                )
                graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
                return
            finally:
                harness._active_dispatch_task = None
            _store_tool_execution_record(
                harness,
                operation_id=operation_id,
                thread_id=graph_state.thread_id,
                step_count=harness.state.step_count,
                pending=pending,
                result=result,
            )
        log_kv(
            harness.log,
            logging.INFO,
            "harness_tool_result",
            tool_name=pending.tool_name,
            success=result.success,
            replayed=replayed,
        )
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.TOOL_RESULT,
                content=json.dumps(json_safe_value(result.to_dict()), ensure_ascii=True),
                data={
                    "tool_name": pending.tool_name,
                    "tool_call_id": pending.tool_call_id,
                    "success": result.success,
                    "replayed": replayed,
                    "display_text": _format_tool_result_display(
                        tool_name=pending.tool_name,
                        result=result,
                        request_text=harness.state.run_brief.original_task,
                    ),
                },
            ),
        )
        graph_state.last_tool_results.append(
            ToolExecutionRecord(
                operation_id=operation_id,
                tool_name=pending.tool_name,
                args=pending.args,
                tool_call_id=pending.tool_call_id,
                result=result,
                replayed=replayed,
            )
        )
        
        # STOP execution of remaining tools in this turn if this tool needs human input
        if (getattr(result, "status", None) == "needs_human" or 
            result.metadata.get("status") == "needs_human"):
            graph_state.pending_tool_calls = []
            break

    graph_state.pending_tool_calls = []
    dispatch_end = time.perf_counter()
    duration = dispatch_end - dispatch_start
    graph_state.latency_metrics["tool_execution_duration_sec"] = round(duration, 3)
    
    if duration > 0.05:
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.METRICS,
                content=f"Tool execution: {duration:.2f}s",
                data={
                    "duration_sec": duration,
                }
            ),
        )


async def persist_tool_results(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> None:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        stored = _get_tool_execution_record(harness, record.operation_id)
        serialized_message = stored.get("tool_message")
        if isinstance(serialized_message, dict):
            message = _conversation_message_from_dict(serialized_message)
        else:
            message = await harness._record_tool_result(
                tool_name=record.tool_name,
                tool_call_id=record.tool_call_id,
                result=record.result,
                arguments=record.args,
            )
            stored["tool_message"] = message.to_dict()
            artifact_id = message.metadata.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id:
                stored["artifact_id"] = artifact_id
            harness.state.tool_execution_records[record.operation_id] = stored
            harness._record_experience(
                tool_name=record.tool_name,
                result=record.result,
            )

        if _has_matching_tool_message(harness, message):
            continue
        harness.state.append_message(message)
        harness._log_conversation_state("tool_message")


def _auto_update_active_plan_step(harness: Any, *, status: str, note: str = "") -> None:
    plan = getattr(harness.state, "active_plan", None) or getattr(harness.state, "draft_plan", None)
    if plan is None:
        return
    active_step = plan.active_step()
    if active_step is None:
        return
    active_step.status = status
    if note.strip():
        active_step.notes.append(note.strip())
    plan.touch()
    harness.state.sync_plan_mirror()
    harness.state.touch()


async def apply_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        if record.tool_name == "task_complete" and record.result.success:
            _auto_update_active_plan_step(harness, status="completed", note=str(record.result.output or ""))
            await harness._emit(
                deps.event_handler,
                UIEvent(event_type=UIEventType.SYSTEM, content="Task marked complete."),
            )
            graph_state.final_result = {
                "status": "completed",
                "message": record.result.output,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
            return LoopRoute.FINALIZE
        if record.tool_name == "task_fail" and record.result.success:
            _auto_update_active_plan_step(harness, status="blocked", note=str(record.result.output or ""))
            await harness._emit(
                deps.event_handler,
                UIEvent(event_type=UIEventType.ERROR, content="Task marked failed."),
            )
            graph_state.final_result = harness._failure(
                "Task marked failed.",
                error_type="tool",
                details={
                    "tool_name": record.tool_name, 
                    "output": record.result.output,
                    "assistant": graph_state.last_assistant_text,
                    "thinking": graph_state.last_thinking_text,
                    "usage": graph_state.last_usage,
                },
            )
            graph_state.error = graph_state.final_result["error"]
            return LoopRoute.FINALIZE
        if record.tool_name == "ask_human" and record.result.success:
            payload = _build_interrupt_payload(
                harness=harness,
                graph_state=graph_state,
                record=record,
            )
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.SYSTEM,
                    content="Human input requested by model.",
                    data={"interrupt": payload},
                ),
            )
            graph_state.final_result = {
                "status": "needs_human",
                "message": payload.get("question", "Human input requested."),
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE

        # Generic handling for tool-initiated human interrupts (e.g. Sudo password prompts)
        if (getattr(record.result, "status", None) == "needs_human" or 
            record.result.metadata.get("status") == "needs_human"):
            question = record.result.metadata.get("question", "Human input required for tool.")
            payload = {
                "question": question,
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "metadata": {**record.result.metadata, "interrupt_type": "tool_request"},
                "current_phase": "explore",
                "active_profiles": list(harness.state.active_tool_profiles),
                "thread_id": graph_state.thread_id,
                "operation_id": record.operation_id,
                "recent_tool_outcomes": [r.to_summary_dict() for r in graph_state.last_tool_results]
            }
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.SYSTEM,
                    content=f"Tool '{record.tool_name}' requires human input: {question}",
                    data={"interrupt": payload},
                ),
            )
            graph_state.final_result = {
                "status": "needs_human",
                "message": record.result.error,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE

    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP


async def apply_chat_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        # Chat mode should also respect explicit task completion tools
        if record.tool_name == "task_complete" and record.result.success:
            _auto_update_active_plan_step(harness, status="completed", note=str(record.result.output or ""))
            message = str(record.result.output.get("message") if isinstance(record.result.output, dict) else record.result.output)
            graph_state.final_result = {
                "status": "chat_completed",
                "message": message,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
            return LoopRoute.FINALIZE

        if record.tool_name == "task_fail" and record.result.success:
            _auto_update_active_plan_step(harness, status="blocked", note=str(record.result.output or ""))
            message = str(record.result.output.get("message") if isinstance(record.result.output, dict) else record.result.output)
            graph_state.final_result = {
                "status": "chat_failed",
                "message": message,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
            }
            return LoopRoute.FINALIZE

        if record.tool_name == "ask_human" and record.result.success:
            payload = _build_interrupt_payload(
                harness=harness,
                graph_state=graph_state,
                record=record,
            )
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            graph_state.final_result = {
                "status": "needs_human",
                "message": payload.get("question", "Human input requested."),
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE

        # Generic handling for tool-initiated human interrupts (e.g. Sudo password prompts)
        if (getattr(record.result, "status", None) == "needs_human" or 
            record.result.metadata.get("status") == "needs_human"):
            question = record.result.metadata.get("question", "Human input required for tool.")
            payload = {
                "question": question,
                "tool_name": record.tool_name,
                "tool_call_id": record.tool_call_id,
                "metadata": {**record.result.metadata, "interrupt_type": "tool_request"},
                "current_phase": "explore",
                "active_profiles": list(harness.state.active_tool_profiles),
                "thread_id": graph_state.thread_id,
                "operation_id": record.operation_id,
                "recent_tool_outcomes": [r.to_summary_dict() for r in graph_state.last_tool_results]
            }
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.SYSTEM,
                    content=f"Tool '{record.tool_name}' requires human input: {question}",
                    data={"interrupt": payload},
                ),
            )
            graph_state.final_result = {
                "status": "needs_human",
                "message": record.result.error,
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE

    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP


async def apply_planning_tool_outcomes(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
) -> LoopRoute:
    harness = deps.harness
    has_explicit_plan_request = any(
        record.tool_name == "plan_request_execution" and record.result.success
        for record in graph_state.last_tool_results
    )
    for record in graph_state.last_tool_results:
        if not record.result.success and _is_plan_export_validation_error(record.result.error):
            repair_attempts = int(harness.state.scratchpad.get("_plan_export_recovery_nudges", 0))
            if repair_attempts < 1:
                repair_message = _build_plan_export_recovery_message(record)
                harness.state.scratchpad["_plan_export_recovery_nudges"] = repair_attempts + 1
                harness.state.recent_errors.append(str(record.result.error or repair_message))
                harness.state.append_message(
                    ConversationMessage(
                        role="user",
                        content=repair_message,
                        metadata={
                            "is_recovery_nudge": True,
                            "recovery_kind": "plan_export_validation",
                            "tool_name": record.tool_name,
                            "tool_call_id": record.tool_call_id,
                        },
                    )
                )
                harness._runlog(
                    "plan_export_repair",
                    "injected plan export repair nudge",
                    tool_name=record.tool_name,
                    tool_call_id=record.tool_call_id,
                    retry_count=repair_attempts + 1,
                    error=str(record.result.error or ""),
                )
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content=repair_message,
                        data={
                            "repair_kind": "plan_export_validation",
                            "tool_name": record.tool_name,
                            "tool_call_id": record.tool_call_id,
                            "retry_count": repair_attempts + 1,
                        },
                    ),
                )
                graph_state.last_tool_results = []
                graph_state.last_assistant_text = ""
                graph_state.last_thinking_text = ""
                return LoopRoute.NEXT_STEP

        if record.tool_name == "plan_set" and record.result.success:
            plan = harness.state.draft_plan or harness.state.active_plan
            if plan is not None:
                plan.status = "draft"
                plan.touch()
                harness.state.draft_plan = plan
                harness.state.sync_plan_mirror()
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content="Draft plan created.",
                        data={"status_activity": "draft plan created"},
                    ),
                )
                export_warning = str(record.result.metadata.get("export_warning", "") or "").strip()
                if export_warning:
                    await harness._emit(
                        deps.event_handler,
                        UIEvent(
                            event_type=UIEventType.ALERT,
                            content=f"Draft plan created; skipped invalid export hint: {export_warning}",
                            data={
                                "status_activity": "draft plan created",
                                "warning_type": "plan_export_validation",
                                "rejected_output_path": record.result.metadata.get("rejected_output_path", ""),
                                "suggested_output_path": record.result.metadata.get("suggested_output_path", ""),
                            },
                        ),
                    )
                if plan.requested_output_path:
                    try:
                        write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
                    except ValueError as exc:
                        harness.log.warning("skipping invalid plan export after plan_set: %s", exc)
                if not has_explicit_plan_request:
                    await _pause_for_plan_approval(graph_state, deps)
                    return LoopRoute.FINALIZE
            continue
        if record.tool_name == "plan_step_update" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None:
                plan.touch()
                harness.state.sync_plan_mirror()
                active_step = plan.find_step(str(record.args.get("step_id", "")).strip())
                step_label = active_step.step_id if active_step is not None else str(record.args.get("step_id", "")).strip()
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content=f"Plan step updated: {step_label}",
                        data={"status_activity": f"step {step_label} updated"},
                    ),
                )
                if plan.requested_output_path:
                    try:
                        write_plan_file(plan, plan.requested_output_path, format=plan.requested_output_format)
                    except ValueError as exc:
                        harness.log.warning("skipping invalid plan export after step update: %s", exc)
            continue
        if record.tool_name == "plan_export" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None:
                await harness._emit(
                    deps.event_handler,
                    UIEvent(
                        event_type=UIEventType.ALERT,
                        content="Plan file exported.",
                        data={"status_activity": "plan file exported"},
                    ),
                )
            continue
        if record.tool_name == "plan_request_execution" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            payload = {
                "kind": "plan_execute_approval",
                "question": record.result.metadata.get("question", "Plan ready. Execute it now?"),
                "plan_id": plan.plan_id if plan is not None else "",
                "approved": False,
                "response_mode": "yes/no/revise",
                "current_phase": harness.state.current_phase,
                "thread_id": graph_state.thread_id,
            }
            graph_state.interrupt_payload = payload
            harness.state.pending_interrupt = payload
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content=payload["question"],
                    data={"status_activity": "awaiting plan approval...", "interrupt": payload},
                ),
            )
            graph_state.final_result = {
                "status": "needs_human",
                "message": payload["question"],
                "assistant": graph_state.last_assistant_text,
                "thinking": graph_state.last_thinking_text,
                "usage": graph_state.last_usage,
                "interrupt": payload,
            }
            return LoopRoute.FINALIZE
        if record.tool_name == "task_complete" and record.result.success:
            plan = harness.state.active_plan or harness.state.draft_plan
            if plan is not None and plan.approved:
                return LoopRoute.FINALIZE

    graph_state.last_tool_results = []
    return LoopRoute.NEXT_STEP


def _has_completed_tool_backed_answer(harness: Any) -> bool:
    if harness.state.scratchpad.get("_task_complete"):
        return True
    return any(message.role == "tool" for message in harness.state.recent_messages)


def _build_interrupt_payload(
    *,
    harness: Any,
    graph_state: GraphRunState,
    record: ToolExecutionRecord,
) -> dict[str, Any]:
    question = ""
    if isinstance(record.result.output, dict):
        question = str(record.result.output.get("question", "")).strip()
    if not question:
        question = str(harness.state.scratchpad.get("_ask_human_question", "")).strip()
    recent_tool_summary = []
    for item in graph_state.last_tool_results[-3:]:
        summary = {
            "tool_name": item.tool_name,
            "success": item.result.success,
            "replayed": item.replayed,
        }
        if item.result.error:
            summary["error"] = item.result.error
        elif isinstance(item.result.output, dict):
            summary["output"] = {
                key: value
                for key, value in item.result.output.items()
                if key in {"status", "message", "question"}
            }
        recent_tool_summary.append(summary)
    return {
        "kind": "ask_human",
        "question": question,
        "current_phase": harness.state.current_phase,
        "active_profiles": list(harness.state.active_tool_profiles),
        "thread_id": graph_state.thread_id,
        "operation_id": record.operation_id,
        "recent_tool_outcomes": recent_tool_summary,
    }


def _tool_envelope_from_dict(payload: dict[str, Any]) -> ToolEnvelope:
    metadata = json_safe_value(payload.get("metadata") or {})
    if not isinstance(metadata, dict):
        metadata = {}
    return ToolEnvelope(
        success=bool(payload.get("success")),
        output=json_safe_value(payload.get("output")),
        error=None if payload.get("error") is None else str(payload.get("error")),
        metadata=metadata,
    )


def _conversation_message_from_dict(payload: dict[str, Any]) -> ConversationMessage:
    normalized = json_safe_value(payload)
    if not isinstance(normalized, dict):
        normalized = {}
    role = str(normalized.get("role", "tool"))
    content = normalized.get("content")
    if content is not None:
        content = str(content)
    name = normalized.get("name")
    if name is not None:
        name = str(name)
    tool_call_id = normalized.get("tool_call_id")
    if tool_call_id is not None:
        tool_call_id = str(tool_call_id)
    tool_calls = normalized.get("tool_calls")
    metadata = normalized.get("metadata")
    return ConversationMessage(
        role=role,
        content=content,
        name=name,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls if isinstance(tool_calls, list) else [],
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _get_tool_execution_record(harness: Any, operation_id: str) -> dict[str, Any]:
    records = getattr(harness.state, "tool_execution_records", None)
    if not isinstance(records, dict):
        harness.state.tool_execution_records = {}
        return {}
    record = records.get(operation_id)
    return dict(record) if isinstance(record, dict) else {}


def _detect_repeated_tool_loop(harness: Any, pending: PendingToolCall) -> str | None:
    if pending.tool_name in {"task_complete", "task_fail", "ask_human"}:
        _clear_tool_attempt_history(harness)
        return None
    if harness.state.scratchpad.get("_chat_rounds"):
        return None
    history = _tool_attempt_history(harness)
    candidate = {
        "tool_name": pending.tool_name,
        "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args),
    }
    recent_window = history[-(_REPEATED_TOOL_WINDOW - 1) :] + [candidate]
    exact_streak = history[-(_IDENTICAL_TOOL_CALL_STREAK_LIMIT - 1) :] + [candidate]
    if (
        len(exact_streak) >= _IDENTICAL_TOOL_CALL_STREAK_LIMIT
        and len({str(item.get("fingerprint", "")) for item in exact_streak}) == 1
    ):
        return (
            "Guard tripped: repeated tool call loop "
            f"({pending.tool_name} repeated with identical arguments)"
        )
    if len(recent_window) < _REPEATED_TOOL_WINDOW:
        return None
    tool_names = {str(item.get("tool_name", "")) for item in recent_window}
    fingerprints = [str(item.get("fingerprint", "")) for item in recent_window]
    if len(tool_names) == 1 and len(set(fingerprints)) <= _REPEATED_TOOL_UNIQUE_LIMIT:
        return (
            "Guard tripped: repeated tool exploration loop "
            f"({pending.tool_name} cycling through near-identical arguments without progress)"
        )
    return None


def _detect_missing_required_tool_arguments(harness: Any, pending: PendingToolCall) -> tuple[str, dict[str, Any]] | None:
    registry = getattr(harness, "registry", None)
    if registry is None:
        return None
    tool_spec = registry.get(pending.tool_name)
    if tool_spec is None:
        return None
    required = tool_spec.schema.get("required", [])
    if not required:
        return None

    missing_fields = []
    for field in required:
        value = pending.args.get(field)
        if value is None:
            missing_fields.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing_fields.append(field)

    if not missing_fields:
        return None

    message = (
        f"Tool call '{pending.tool_name}' was emitted without arguments. "
        f"Required fields: {', '.join(str(field) for field in missing_fields)}."
    )
    return message, {
        "tool_name": pending.tool_name,
        "tool_call_id": pending.tool_call_id,
        "required_fields": list(missing_fields),
        "raw_arguments": pending.raw_arguments,
    }


def _detect_empty_file_write_payload(
    harness: Any,
    pending: PendingToolCall,
) -> tuple[str, dict[str, Any]] | None:
    """Catch file_write calls that technically parsed but contain no useful code."""
    if pending.tool_name != "file_write":
        return None

    registry = getattr(harness, "registry", None)
    if registry is None:
        return None

    tool_spec = registry.get(pending.tool_name)
    if tool_spec is None:
        return None

    required = tool_spec.schema.get("required", [])
    missing_fields = [field for field in required if field not in pending.args]
    if missing_fields:
        return None

    content = pending.args.get("content")
    if not isinstance(content, str) or not content.strip():
        message = (
            "Tool call 'file_write' contained an empty content field. "
            "Generate the actual code or file contents first, then resend the write with a non-empty content payload."
        )
        return message, {
            "tool_name": pending.tool_name,
            "tool_call_id": pending.tool_call_id,
            "required_fields": list(required),
            "offending_field": "content",
            "raw_arguments": pending.raw_arguments,
        }

    return None


def _build_schema_repair_message(tool_name: str, required_fields: list[Any]) -> str:
    required_text = ", ".join(str(field) for field in required_fields)
    return (
        f"Tool call '{tool_name}' was emitted without arguments. "
        f"Please resend the tool call with these required fields: {required_text}."
    )


def _fallback_repeated_artifact_read(harness: Any, pending: PendingToolCall) -> PendingToolCall | None:
    if pending.tool_name != "artifact_read":
        return None

    artifact_id = _extract_artifact_id_from_args(pending.args)
    if not artifact_id:
        return None

    artifact = _resolve_artifact_record(harness, artifact_id)
    if artifact is None:
        return None

    content = _read_artifact_text(artifact)
    if not content:
        return None

    # Keep this escape hatch narrow: only recover scan-like artifacts with a concrete grep target.
    query = _choose_artifact_grep_query(content)
    if not query:
        return None

    return PendingToolCall(
        tool_name="artifact_grep",
        args={
            "artifact_id": artifact.artifact_id,
            "query": query,
        },
        raw_arguments=json.dumps(
            {"artifact_id": artifact.artifact_id, "query": query},
            ensure_ascii=True,
            sort_keys=True,
        ),
        tool_call_id=pending.tool_call_id,
    )


def _artifact_read_recovery_hint(harness: Any, guard_error: str) -> tuple[str, str] | None:
    if "artifact_read" not in guard_error and "max_consecutive_errors" not in guard_error:
        return None

    if "max_consecutive_errors" in guard_error:
        recent_errors = getattr(harness.state, "recent_errors", [])
        if not recent_errors or not all("artifact_read" in str(err) for err in recent_errors):
            return None

    history = getattr(harness.state, "tool_history", [])
    if not isinstance(history, list) or not history:
        return None

    for fingerprint in reversed(history):
        if not isinstance(fingerprint, str) or not fingerprint.startswith("artifact_read|"):
            continue
        parts = fingerprint.split("|", 2)
        if len(parts) < 2:
            continue
        try:
            args = json.loads(parts[1])
        except Exception:
            continue
        if not isinstance(args, dict):
            continue
        recovered = _fallback_repeated_artifact_read(
            harness,
            PendingToolCall(
                tool_name="artifact_read",
                args=args,
                raw_arguments=json.dumps(args, ensure_ascii=True, sort_keys=True),
            ),
        )
        if recovered is None:
            continue
        artifact_id = str(recovered.args.get("artifact_id", "")).strip()
        query = str(recovered.args.get("query", "")).strip()
        if artifact_id and query:
            return artifact_id, query
    return None


def _extract_artifact_id_from_args(args: dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return None

    for key in ("artifact_id", "path", "id"):
        value = args.get(key)
        if not isinstance(value, str):
            continue
        candidate = Path(value.strip()).stem.strip()
        if candidate:
            return candidate
    return None


def _resolve_artifact_record(harness: Any, artifact_id: str) -> Any | None:
    artifact = harness.state.artifacts.get(artifact_id)
    if artifact is not None:
        return artifact

    if not artifact_id.startswith("A"):
        return None

    try:
        numeric_val = int(artifact_id[1:])
    except ValueError:
        return None

    for aid, record in harness.state.artifacts.items():
        if not isinstance(aid, str) or not aid.startswith("A"):
            continue
        try:
            if int(aid[1:]) == numeric_val:
                return record
        except ValueError:
            continue
    return None


def _read_artifact_text(artifact: Any) -> str:
    content_path = getattr(artifact, "content_path", None)
    if isinstance(content_path, str) and content_path.strip():
        path = Path(content_path)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass

    inline_content = getattr(artifact, "inline_content", None)
    if isinstance(inline_content, str) and inline_content:
        return inline_content
    return ""


def _should_suppress_resolved_plan_artifact_read(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "artifact_read":
        return False
    if not getattr(harness.state, "plan_resolved", False):
        return False
    plan_artifact_id = str(getattr(harness.state, "plan_artifact_id", "") or "").strip()
    if not plan_artifact_id:
        return False
    artifact_id = _extract_artifact_id_from_args(pending.args)
    if artifact_id != plan_artifact_id:
        return False
    return bool(harness.state.active_plan or harness.state.draft_plan or harness.state.working_memory.plan)


def _choose_artifact_grep_query(content: str) -> str | None:
    lowered = content.lower()
    if not any(marker in lowered for marker in ("nmap scan report", "/tcp", "/udp", "host is up")):
        return None
    for query in ("open", "port", "service", "banner", "nmap scan report", "host is up"):
        if query in lowered:
            return query
    return None


def _clear_artifact_read_guard_state(harness: Any, artifact_id: str) -> None:
    if not artifact_id:
        return

    recent_errors = getattr(harness.state, "recent_errors", None)
    if isinstance(recent_errors, list) and recent_errors:
        filtered_errors = [err for err in recent_errors if "artifact_read" not in str(err)]
        if len(filtered_errors) != len(recent_errors):
            harness.state.recent_errors = filtered_errors

    tool_history = getattr(harness.state, "tool_history", None)
    if isinstance(tool_history, list) and tool_history:
        kept_history: list[str] = []
        removed_entries = 0
        for entry in tool_history:
            if not isinstance(entry, str) or not entry.startswith("artifact_read|"):
                kept_history.append(entry)
                continue
            parts = entry.split("|", 2)
            if len(parts) < 3:
                kept_history.append(entry)
                continue
            try:
                args = json.loads(parts[1])
            except Exception:
                kept_history.append(entry)
                continue
            if isinstance(args, dict) and str(args.get("artifact_id", "")).strip() == artifact_id:
                removed_entries += 1
                continue
            kept_history.append(entry)
        if removed_entries:
            harness.state.tool_history = kept_history

    _clear_tool_attempt_history(harness)


def _record_tool_attempt(harness: Any, pending: PendingToolCall) -> None:
    history = _tool_attempt_history(harness)
    history.append(
        {
            "tool_name": pending.tool_name,
            "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args),
        }
    )
    harness.state.scratchpad["_tool_attempt_history"] = history[-_REPEATED_TOOL_HISTORY_LIMIT:]


def _clear_tool_attempt_history(harness: Any) -> None:
    harness.state.scratchpad.pop("_tool_attempt_history", None)


def _tool_attempt_history(harness: Any) -> list[dict[str, str]]:
    history = harness.state.scratchpad.get("_tool_attempt_history")
    if not isinstance(history, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in history[-_REPEATED_TOOL_HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "tool_name": str(item.get("tool_name", "")),
                "fingerprint": str(item.get("fingerprint", "")),
            }
        )
    return normalized


def _tool_call_fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    normalized_args = _normalize_tool_args(tool_name, args)
    return json.dumps({"tool_name": tool_name, "args": normalized_args}, sort_keys=True, ensure_ascii=True)


def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> Any:
    if not isinstance(args, dict):
        return {}
    normalized = json_safe_value(args)
    if not isinstance(normalized, dict):
        return {}
    if tool_name == "shell_exec":
        command = normalized.get("command")
        if command is not None:
            normalized["command"] = _normalize_shell_command(str(command))
    return _normalize_json_like(normalized)


def _normalize_shell_command(command: str) -> str:
    parts = command.strip().split()
    if not parts:
        return ""
    normalized_parts = [_normalize_path_token(part) for part in parts]
    return " ".join(normalized_parts)


def _normalize_path_token(value: str) -> str:
    stripped = value.strip()
    if len(stripped) > 1 and stripped.endswith("/") and "/" in stripped[:-1]:
        return stripped.rstrip("/")
    return stripped


def _normalize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_like(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, str):
        collapsed = " ".join(value.strip().split())
        return _normalize_path_token(collapsed)
    return json_safe_value(value)


def _store_tool_execution_record(
    harness: Any,
    *,
    operation_id: str,
    thread_id: str,
    step_count: int,
    pending: PendingToolCall,
    result: ToolEnvelope,
) -> None:
    existing = _get_tool_execution_record(harness, operation_id)
    existing.update(
        {
            "operation_id": operation_id,
            "thread_id": thread_id,
            "step_count": step_count,
            "tool_name": pending.tool_name,
            "tool_call_id": pending.tool_call_id,
            "args": dict(pending.args),
            "result": result.to_dict(),
        }
    )
    harness.state.tool_execution_records[operation_id] = existing


def _has_matching_tool_message(harness: Any, message: ConversationMessage) -> bool:
    for existing in reversed(harness.state.recent_messages):
        if existing.role != "tool":
            continue
        if existing.name != message.name:
            continue
        if existing.tool_call_id != message.tool_call_id:
            continue
        if existing.content != message.content:
            continue
        if existing.metadata != message.metadata:
            continue
        return True
    return False


def _format_tool_result_display(
    *,
    tool_name: str,
    result: ToolEnvelope,
    request_text: str | None = None,
) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not result.success:
        preview, clipped = clip_text_value(str(result.error or "Tool failed.").strip(), limit=400)
        if clipped:
            return f"{preview}\n... error truncated"
        return preview

    if tool_name == "artifact_read":
        return _format_artifact_read_display(result=result, request_text=request_text)
    if tool_name == "dir_list":
        return _format_dir_list_display(result=result)

    output = result.output
    if isinstance(output, str):
        preview, clipped = clip_text_value(output.strip(), limit=_UI_TOOL_RESULT_PREVIEW_LIMIT)
        lines: list[str] = []
        path = metadata.get("path")
        if isinstance(path, str) and path.strip():
            lines.append(path.strip())
        if preview:
            lines.append(preview)
        text = "\n".join(lines) if lines else "ok"
        if clipped:
            text = f"{text}\n... output truncated"
        return text

    if isinstance(output, dict):
        # 1. Shell-like output (stdout, stderr, exit_code)
        if "stdout" in output or "stderr" in output:
            return _format_shell_output_display(output=output)
            
        # 2. Generic message-based output
        msg = output.get("message") or output.get("output") or output.get("text") or output.get("question")
        if isinstance(msg, str) and msg.strip():
            rendered = msg.strip()
        else:
            # Fallback to full JSON if no obvious single text field found
            rendered = json.dumps(json_safe_value(output), ensure_ascii=True, default=str, indent=2)
            
        preview, clipped = clip_text_value(rendered, limit=_UI_TOOL_RESULT_PREVIEW_LIMIT)
        if clipped:
            return f"{preview}\n... output truncated"
        return preview or "ok"

    if output is None:
        return "ok"

    rendered = json.dumps(json_safe_value(output), ensure_ascii=True, default=str, indent=2)
    preview, clipped = clip_text_value(rendered, limit=_UI_TOOL_RESULT_PREVIEW_LIMIT)
    if clipped:
        return f"{preview}\n... output truncated"
    return preview or "ok"


def _format_dir_list_display(*, result: ToolEnvelope) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    output = result.output
    if not isinstance(output, list):
        return "directory listed"

    lines: list[str] = []
    path = metadata.get("path")
    count = metadata.get("count")
    if isinstance(path, str) and path.strip():
        if isinstance(count, int) and count >= 0:
            lines.append(f"{path.strip()} ({count} items)")
        else:
            lines.append(path.strip())
    elif isinstance(count, int) and count >= 0:
        lines.append(f"{count} items")

    preview_items = output[:8]
    for item in preview_items:
        preview_line = _format_dir_list_item(item)
        if preview_line:
            lines.append(preview_line)

    remaining = len(output) - len(preview_items)
    if remaining > 0:
        lines.append(f"... {remaining} more items")

    text = "\n".join(lines).strip() or "directory listed"
    preview, clipped = clip_text_value(text, limit=420)
    if clipped:
        return f"{preview}\n... output truncated"
    return preview


def _format_artifact_read_display(
    *,
    result: ToolEnvelope,
    request_text: str | None = None,
) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    header_parts: list[str] = []
    full_request = _request_has_full_artifact_intent(request_text)

    artifact_id = metadata.get("artifact_id")
    if isinstance(artifact_id, str) and artifact_id.strip():
        header_parts.append(artifact_id.strip())

    path = metadata.get("path")
    if isinstance(path, str) and path.strip():
        header_parts.append(path.strip())

    line_start = metadata.get("line_start")
    line_end = metadata.get("line_end")
    total_lines = metadata.get("total_lines")
    if isinstance(line_start, int) and isinstance(line_end, int):
        if isinstance(total_lines, int) and total_lines > 0:
            header_parts.append(f"lines {line_start}-{line_end} of {total_lines}")
        else:
            header_parts.append(f"lines {line_start}-{line_end}")
    output = result.output
    preview_source = output if isinstance(output, str) else json.dumps(
        json_safe_value(output),
        ensure_ascii=True,
        default=str,
        indent=2,
    )
    preview, clipped = clip_text_value(
        preview_source, 
        limit=_UI_ARTIFACT_READ_PREVIEW_LIMIT
    )

    lines: list[str] = []
    if header_parts:
        lines.append(" | ".join(header_parts))
    if preview:
        lines.append(preview)
        
    if metadata.get("truncated") or clipped:
        if metadata.get("truncated"):
            if isinstance(line_end, int) and isinstance(total_lines, int) and line_end < total_lines:
                next_start = line_end + 1
                lines.append(
                    f"... continue at start_line={next_start} via "
                    f"artifact_read(artifact_id='{artifact_id}', start_line={next_start})"
                )
            else:
                lines.append("... more available via artifact_read(start_line=..., end_line=..., max_chars=...)")
        elif full_request:
            lines.append("... preview clipped in UI")
        else:
            lines.append("... more available via artifact_read(start_line=..., end_line=..., max_chars=...)")
        
    return "\n\n".join(lines) if lines else "artifact read complete"


def _format_shell_output_display(*, output: dict[str, Any]) -> str:
    return render_shell_output(
        output,
        preview_limit=_UI_TOOL_RESULT_PREVIEW_LIMIT,
        strip_whitespace=True,
    )


def _format_dir_list_item(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item or "").strip()

    name = str(item.get("name") or item.get("path") or "").strip()
    if not name:
        return ""

    parts = [name]
    item_type = str(item.get("type") or "").strip()
    if item_type:
        parts.append(f"[{item_type}]")

    size = item.get("size")
    if isinstance(size, int) and size >= 0:
        parts.append(f"({size} bytes)")

    return " ".join(parts).strip()


def _coerce_int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _detect_hallucinated_tool_call(harness: "Harness", pending: PendingToolCall) -> str | None:
    """Detect if a tool call is missing all its required arguments."""
    meta = harness.registry.get(pending.tool_name)
    if not meta:
        return None
    
    schema = meta.schema or {}
    required = schema.get("required", [])
    if not required:
        return None

    # If the tool requires something but we have nothing, it's a hallucination trap
    if not pending.args:
        return (
            f"Hallucination Warning: Tool '{pending.tool_name}' requires specific parameters "
            f"({', '.join(required)}) but was called with none. Please provide the missing arguments."
        )

    return None

def _extract_inline_tool_calls(text: str) -> tuple[str, list[PendingToolCall]]:
    if not text:
        return "", []
    
    import re
    import json
    import ast
    results: list[PendingToolCall] = []
    cleaned_text = text

    def _try_parse_data(data: Any) -> PendingToolCall | None:
        if not isinstance(data, dict):
            return None
        # Key Hallucination Handling: Support name, tool_name, tool, action
        name = str(data.get("name", data.get("tool_name", data.get("tool", data.get("action", ""))))).strip()
        if not name:
            return None
        # Args Hallucination Handling: Support arguments, args, params, parameters
        args = data.get("arguments", data.get("args", data.get("params", data.get("parameters", {}))))
        payload = {
            "function": {
                "name": name,
                "arguments": json.dumps(args) if isinstance(args, dict) else "{}"
            }
        }
        return PendingToolCall.from_payload(payload)
    
    # 1. Check for XML-style blocks: <tool_code>, <tool_call>, etc.
    xml_patterns = [
        r"<tool_code>(.*?)</tool_code>",
        r"<tool_call>(.*?)</tool_call>",
        r"<call>(.*?)</call>"
    ]
    for pattern in xml_patterns:
        it = re.finditer(pattern, cleaned_text, re.DOTALL)
        offset = 0
        for match in it:
            content = match.group(1).strip()
            
            # 1a. Handle recursive/structured XML (Qwen-style: <function=name><parameter=key>val</parameter></function>)
            struct_fn_match = re.search(r"<function=([\w_-]+)>(.*?)</function>", content, re.DOTALL)
            found = False
            if struct_fn_match:
                tool_name = struct_fn_match.group(1)
                inner_content = struct_fn_match.group(2).strip()
                params = {}
                param_matches = re.findall(r"<parameter=([\w_-]+)>(.*?)</parameter>", inner_content, re.DOTALL)
                for pk, pv in param_matches:
                    params[pk] = pv.strip()
                
                if tool_name:
                    results.append(
                        PendingToolCall(
                            tool_name=tool_name,
                            args=params,
                            raw_arguments=json.dumps(params, ensure_ascii=True, sort_keys=True),
                        )
                    )
                    found = True
            
            if not found:
                # 1b. Attempt JSON inside the tag
                try:
                    data = json.loads(content)
                    pending = _try_parse_data(data)
                    if pending: 
                        results.append(pending)
                        found = True
                except Exception:
                    pass
            
            if not found:
                # 1c. Attempt Function call style: name(arg='val', ...)
                fn_call_regex = r"^([a-zA-Z0-9_-]+)\((.*)\)$"
                fn_match = re.match(fn_call_regex, content, re.DOTALL)
                if fn_match:
                    tool_name = fn_match.group(1)
                    args_str = fn_match.group(2).strip()
                    try:
                        kv_pairs = re.findall(r"([a-zA-Z0-9_-]+)\s*=\s*('[^']*'|\"[^\"]*\"|[0-9.]+)", args_str)
                        args = {k: ast.literal_eval(v) for k, v in kv_pairs}
                        pending = PendingToolCall(
                            tool_name=tool_name,
                            args=args,
                            raw_arguments=args_str,
                        )
                        if pending: 
                            results.append(pending)
                            found = True
                    except Exception:
                        pass
            
            if found:
                # Strip the matched block from cleaned_text
                start, end = match.span()
                cleaned_text = cleaned_text[:start-offset] + cleaned_text[end-offset:]
                offset += (end - start)

    # 1d. Catch structured XML even if NOT wrapped in tags (paranoid)
    struct_fn_matches = list(re.finditer(r"<function=([\w_-]+)>(.*?)</function>", cleaned_text, re.DOTALL))
    offset = 0
    for match in struct_fn_matches:
        tool_name = match.group(1)
        inner_content = match.group(2).strip()
        params = {}
        param_matches = re.findall(r"<parameter=([\w_-]+)>(.*?)</parameter>", inner_content, re.DOTALL)
        for pk, pv in param_matches:
            params[pk] = pv.strip()
        if tool_name:
            results.append(
                PendingToolCall(
                    tool_name=tool_name,
                    args=params,
                    raw_arguments=json.dumps(params, ensure_ascii=True, sort_keys=True),
                )
            )
            start, end = match.span()
            cleaned_text = cleaned_text[:start-offset] + cleaned_text[end-offset:]
            offset += (end - start)

    # 2. Existing JSON extractors (fallbacks if no XML found or in addition)
    # 2a. Check for markdown JSON blocks (allow flexible whitespace)
    json_blocks = list(re.finditer(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL))
    offset = 0
    for match in json_blocks:
        block = match.group(1)
        try:
            data = json.loads(block)
            pending = _try_parse_data(data)
            if pending: 
                results.append(pending)
                start, end = match.span()
                cleaned_text = cleaned_text[:start-offset] + cleaned_text[end-offset:]
                offset += (end - start)
        except Exception:
            pass

    # 2b. Raw outer-level JSON objects
    if "{" in cleaned_text:
        start = cleaned_text.find("{")
        while start != -1:
            brace_count = 0
            end = -1
            for i in range(start, len(cleaned_text)):
                if cleaned_text[i] == "{": brace_count += 1
                elif cleaned_text[i] == "}": brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
            
            if end != -1:
                try:
                    data = json.loads(cleaned_text[start:end])
                    pending = _try_parse_data(data)
                    if pending: 
                        results.append(pending)
                        # Strip it
                        cleaned_text = cleaned_text[:start] + cleaned_text[end:]
                        # Continue searching from same START position (since we shifted text)
                        start = cleaned_text.find("{", start)
                    else:
                        # Continue searching for next potential object
                        start = cleaned_text.find("{", start + 1)
                except Exception:
                     start = cleaned_text.find("{", start + 1)
            else:
                break

    # 2c. Functional style name(arg='val') in raw text
    raw_fn_regex = r"([a-zA-Z0-9_-]+)\(([a-zA-Z0-9_-]+\s*=\s*(?:'[^']*'|\"[^\"]*\"|[0-9.]+)(?:\s*,\s*[a-zA-Z0-9_-]+\s*=\s*(?:'[^']*'|\"[^\"]*\"|[0-9.]+))*)\)"
    matches = list(re.finditer(raw_fn_regex, cleaned_text))
    offset = 0
    for match in matches:
        tool_name = match.group(1)
        args_str = match.group(2)
        try:
            kv_pairs = re.findall(r"([a-zA-Z0-9_-]+)\s*=\s*('[^']*'|\"[^\"]*\"|[0-9.]+)", args_str)
            args = {k: ast.literal_eval(v) for k, v in kv_pairs}
            pending = _try_parse_data({"tool_name": tool_name, "arguments": args})
            if pending: 
                results.append(pending)
                start, end = match.span()
                cleaned_text = cleaned_text[:start-offset] + cleaned_text[end-offset:]
                offset += (end - start)
        except Exception:
            pass

    return cleaned_text, results
