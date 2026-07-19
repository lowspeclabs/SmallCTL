from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from ..client.tool_budgeting import (
    _protected_tool_names,
    _slim_schema_descriptions,
    tool_name as _schema_tool_name,
)
from ..harness.prompt_builder import is_prompt_budget_overflow
from ..harness.tool_visibility import resolve_turn_tool_exposure
from ..models.events import UIEvent, UIEventType
from ..prompts import build_planning_prompt, build_system_prompt
from ..context.step_sandbox import build_step_sandbox_prompt
from .deps import GraphRuntimeDeps
from .state import GraphRunState


def load_index_manifest(cwd: str) -> dict[str, Any] | None:
    path = Path(cwd) / ".smallctl" / "index_manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _available_tool_names(harness: Any, *, mode: str) -> list[str]:
    exposure = resolve_turn_tool_exposure(harness, mode)
    names = exposure.get("names")
    return list(names) if isinstance(names, list) else []


def _exposed_tool_schemas(harness: Any, mode: str) -> list[dict[str, Any]]:
    """Return the tool schemas for this turn, preferring budget-slimmed ones.

    When a prompt-budget overflow forced the one-shot slimming retry, the
    reduced schema set is stashed in the scratchpad so the transport sends the
    same slimmed tools that the retried prompt advertises.
    """
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if isinstance(scratchpad, dict):
        slimmed = scratchpad.get("_prompt_budget_slimmed_tool_schemas")
        if isinstance(slimmed, list) and slimmed:
            return [dict(schema) for schema in slimmed if isinstance(schema, dict)]
    return resolve_turn_tool_exposure(harness, mode)["schemas"]


def _clear_prompt_budget_tool_slimming(harness: Any, *, reason: str) -> bool:
    """Clear residual prompt-budget tool-slimming state.

    The slimming flag and stashed schemas are one-shot per overflow episode:
    once a later prompt builds within budget (or a task boundary resets the
    run), full tool exposure must be restored instead of leaking the reduced
    set into subsequent prompts. Returns True when state was cleared.
    """
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    if not scratchpad.get("_prompt_budget_tool_slimming_active") and not scratchpad.get(
        "_prompt_budget_slimmed_tool_schemas"
    ):
        return False
    scratchpad.pop("_prompt_budget_tool_slimming_active", None)
    scratchpad.pop("_prompt_budget_slimmed_tool_schemas", None)
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "prompt_budget_tool_slimming_cleared",
            "restored full runtime tool exposure after prompt-budget slimming episode",
            reason=reason,
        )
    return True


def _prompt_budget_slimmed_exposure(harness: Any, *, mode: str) -> tuple[list[str], list[dict[str, Any]]]:
    """Deterministically slim runtime tool exposure for a prompt-budget retry.

    Non-protected tools are dropped per the transport budget helper's priority
    order; surviving schemas get truncated descriptions via the shared
    slimming helper. This shrinks both the rendered tool guidance in the
    system prompt and the transported tool schemas.
    """
    exposure = resolve_turn_tool_exposure(harness, mode)
    raw_names = exposure.get("names")
    names = [
        str(name).strip()
        for name in (raw_names if isinstance(raw_names, list) else [])
        if str(name or "").strip()
    ]
    raw_schemas = exposure.get("schemas")
    schemas = [schema for schema in (raw_schemas if isinstance(raw_schemas, list) else []) if isinstance(schema, dict)]
    available = set(names)
    protected = _protected_tool_names(available, requested_tool_name="", mode=mode)
    kept_names = sorted(name for name in available if name in protected)
    slimmed_schemas = [_slim_schema_descriptions(schema) for schema in schemas if _schema_tool_name(schema) in protected]
    return kept_names, slimmed_schemas


async def _retry_prompt_with_slimmed_tools(
    harness: Any,
    deps: GraphRuntimeDeps,
    *,
    mode: str,
    build_prompt: Callable[[list[str]], str],
) -> tuple[list[dict[str, Any]] | None, RuntimeError | None]:
    """Retry a prompt build once with slimmed runtime tool exposure.

    Returns (messages, None) when the slimmed build fits, (None, error) when
    the content is irreducible, and (None, None) when no retry was possible.
    """
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if not isinstance(scratchpad, dict) or scratchpad.get("_prompt_budget_tool_slimming_active"):
        return None, None
    kept_names, slimmed_schemas = _prompt_budget_slimmed_exposure(harness, mode=mode)
    if not kept_names:
        return None, None
    scratchpad["_prompt_budget_tool_slimming_active"] = True
    scratchpad["_prompt_budget_slimmed_tool_schemas"] = slimmed_schemas
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "prompt_budget_tool_slimming_retry",
            "retrying prompt build once with slimmed runtime tool exposure",
            mode=mode,
            kept_tool_names=kept_names,
            dropped_tool_names=sorted(set(_available_tool_names(harness, mode=mode)) - set(kept_names)),
        )
    try:
        messages = await harness._build_prompt_messages(
            build_prompt(kept_names),
            event_handler=deps.event_handler,
        )
    except RuntimeError as exc:
        return None, exc
    return messages, None


async def _resolve_prompt_or_budget_failure(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    mode: str,
    system_prompt: str,
    build_slimmed_prompt: Callable[[list[str]], str],
) -> list[dict[str, Any]] | None:
    """Build prompt messages, retrying overflow once with slimmed tools.

    On irreducible overflow, records the typed prompt_budget failure with the
    diagnostic message produced by the prompt builder.
    """
    harness = deps.harness
    # A new prompt build starts a fresh episode: any slimming state stashed by
    # a previous overflow episode is cleared up front so the full tool set is
    # restored and the one-shot retry is re-armed for this build.
    _clear_prompt_budget_tool_slimming(harness, reason="prompt_build_started")
    try:
        return await harness._build_prompt_messages(system_prompt, event_handler=deps.event_handler)
    except RuntimeError as exc:
        if _overflow_with_passed_verdict_to_success(graph_state, harness, exc):
            return None
        failure_exc = exc
        if is_prompt_budget_overflow(exc):
            retry_messages, retry_exc = await _retry_prompt_with_slimmed_tools(
                harness,
                deps,
                mode=mode,
                build_prompt=build_slimmed_prompt,
            )
            if retry_messages is not None:
                return retry_messages
            if retry_exc is not None:
                failure_exc = retry_exc
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=str(failure_exc)),
        )
        graph_state.final_result = harness._failure(str(failure_exc), error_type="prompt_budget")
        graph_state.error = graph_state.final_result["error"]
        return None


def select_loop_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    harness = deps.harness
    profiles = set(harness.state.active_tool_profiles)
    phase = harness.state.current_phase
    tools = _exposed_tool_schemas(harness, "loop")
    harness.log.info("select_loop_tools: phase=%s profiles=%s count=%d", phase, profiles, len(tools))
    return tools


def select_chat_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    return _exposed_tool_schemas(deps.harness, "chat")


def select_indexer_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    return _exposed_tool_schemas(deps.harness, "indexer")


def select_planning_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    return _exposed_tool_schemas(deps.harness, "planning")


def select_staged_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    harness = deps.harness
    state = harness.state
    plan = state.active_plan or state.draft_plan
    step = plan.find_step(state.active_step_id) if plan is not None and state.active_step_id else None
    allowlist = set(getattr(step, "tool_allowlist", []) or []) if step is not None else set()
    staged_controls = {"step_complete", "step_fail", "loop_status", "ask_human"}
    schemas = resolve_turn_tool_exposure(harness, "loop")["schemas"]
    selected: list[dict[str, Any]] = []
    rejected: list[str] = []
    for schema in schemas:
        function = schema.get("function") if isinstance(schema, dict) else None
        name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if not name:
            continue
        if name == "task_complete":
            rejected.append(name)
            continue
        if name in staged_controls or name in allowlist:
            selected.append(schema)
        else:
            rejected.append(name)
    harness.log.info(
        "staged_tools_selected plan_id=%s step_id=%s step_run_id=%s selected=%s rejected=%s",
        getattr(plan, "plan_id", ""),
        state.active_step_id,
        state.active_step_run_id,
        [str(item.get("function", {}).get("name", "")) for item in selected],
        rejected,
    )
    return selected


def _last_verifier_passed(state: Any) -> bool:
    verdict = getattr(state, "last_verifier_verdict", None) or {}
    if isinstance(verdict, dict):
        return str(verdict.get("verdict") or "").strip().lower() == "pass"
    return str(verdict).strip().lower() == "pass"


def _overflow_with_passed_verdict_to_success(
    graph_state: GraphRunState,
    harness: Any,
    exc: RuntimeError,
) -> bool:
    """If the last verifier passed, treat a prompt-budget overflow as success.

    This prevents a harness that has already accomplished the objective from
    failing just because assembling the next turn's prompt exceeded the token
    budget.
    """
    if not _last_verifier_passed(harness.state):
        return False
    message = "Task objective verified; prompt budget exceeded on next turn."
    graph_state.final_result = {
        "status": "completed",
        "message": {"status": "complete", "message": message},
        "assistant": str(graph_state.last_assistant_text or "").strip() or message,
        "thinking": graph_state.last_thinking_text,
        "usage": graph_state.last_usage,
        "error": None,
    }
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "task_complete_prompt_overflow_after_passed_verifier",
            "auto-completed task after prompt-budget overflow because last verifier passed",
            verifier=str(
                getattr(harness.state, "last_verifier_verdict", {}).get("verdict") or "pass"
            ),
            overflow_message=str(exc),
        )
    return True


async def prepare_prompt(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]] | None:
    harness = deps.harness

    def _build(tool_names: list[str]) -> str:
        return build_system_prompt(
            harness.state,
            harness.state.current_phase,
            available_tool_names=tool_names,
            strategy_prompt=harness.strategy_prompt,
            manifest=load_index_manifest(harness.state.cwd),
            indexer_mode=bool(getattr(harness, "_indexer", False)),
        )

    return await _resolve_prompt_or_budget_failure(
        graph_state,
        deps,
        mode="loop",
        system_prompt=_build(_available_tool_names(harness, mode="loop")),
        build_slimmed_prompt=_build,
    )


async def prepare_staged_prompt(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]] | None:
    harness = deps.harness
    _clear_prompt_budget_tool_slimming(harness, reason="prompt_build_started")
    plan = harness.state.active_plan or harness.state.draft_plan
    step = plan.find_step(harness.state.active_step_id) if plan is not None and harness.state.active_step_id else None
    if step is None:
        graph_state.final_result = harness._failure(
            "Cannot prepare staged prompt without an active plan step.",
            error_type="staged_prompt",
        )
        graph_state.error = graph_state.final_result["error"]
        return None
    try:
        messages = build_step_sandbox_prompt(harness, step)
    except RuntimeError as exc:
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=str(exc)),
        )
        graph_state.final_result = harness._failure(str(exc), error_type="staged_prompt")
        graph_state.error = graph_state.final_result["error"]
        return None
    harness.log.info(
        "staged_prompt_built plan_id=%s step_id=%s step_run_id=%s messages=%d",
        getattr(plan, "plan_id", ""),
        step.step_id,
        harness.state.active_step_run_id,
        len(messages),
    )
    return messages


async def prepare_chat_prompt(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]] | None:
    harness = deps.harness

    def _build(tool_names: list[str]) -> str:
        prompt = build_system_prompt(
            harness.state,
            harness.state.current_phase,
            available_tool_names=tool_names,
            strategy_prompt=harness.strategy_prompt,
            manifest=load_index_manifest(harness.state.cwd),
            indexer_mode=bool(getattr(harness, "_indexer", False)),
        )
        prompt = f"{prompt} You may use available tools when needed to answer accurately."
        if "shell_exec" in tool_names:
            prompt = (
                f"{prompt} "
                "SHELL: `shell_exec` is available for command execution, but it requires user approval before execution."
            )
        return prompt

    return await _resolve_prompt_or_budget_failure(
        graph_state,
        deps,
        mode="chat",
        system_prompt=_build(_available_tool_names(harness, mode="chat")),
        build_slimmed_prompt=_build,
    )


async def prepare_planning_prompt(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]] | None:
    harness = deps.harness

    def _build(tool_names: list[str]) -> str:
        return build_planning_prompt(
            harness.state,
            harness.state.current_phase,
            available_tool_names=tool_names,
            strategy_prompt=harness.strategy_prompt,
            manifest=load_index_manifest(harness.state.cwd),
            indexer_mode=bool(getattr(harness, "_indexer", False)),
        )

    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content="Gathering planning facts...",
            data={"status_activity": "gathering facts..."},
        ),
    )
    return await _resolve_prompt_or_budget_failure(
        graph_state,
        deps,
        mode="planning",
        system_prompt=_build(_available_tool_names(harness, mode="planning")),
        build_slimmed_prompt=_build,
    )


async def prepare_indexer_prompt(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]] | None:
    harness = deps.harness
    _clear_prompt_budget_tool_slimming(harness, reason="prompt_build_started")
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
