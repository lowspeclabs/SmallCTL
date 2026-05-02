from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def select_loop_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    harness = deps.harness
    profiles = set(harness.state.active_tool_profiles)
    phase = harness.state.current_phase
    tools = resolve_turn_tool_exposure(harness, "loop")["schemas"]
    harness.log.info("select_loop_tools: phase=%s profiles=%s count=%d", phase, profiles, len(tools))
    return tools


def select_chat_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    return resolve_turn_tool_exposure(deps.harness, "chat")["schemas"]


def select_indexer_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    return resolve_turn_tool_exposure(deps.harness, "indexer")["schemas"]


def select_planning_tools(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]]:
    del graph_state
    return resolve_turn_tool_exposure(deps.harness, "planning")["schemas"]


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


async def prepare_prompt(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]] | None:
    harness = deps.harness
    system_prompt = build_system_prompt(
        harness.state,
        harness.state.current_phase,
        available_tool_names=_available_tool_names(harness, mode="loop"),
        strategy_prompt=harness.strategy_prompt,
        manifest=load_index_manifest(harness.state.cwd),
        indexer_mode=bool(getattr(harness, "_indexer", False)),
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


async def prepare_staged_prompt(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]] | None:
    harness = deps.harness
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
    chat_tool_names = _available_tool_names(harness, mode="chat")
    system_prompt = build_system_prompt(
        harness.state,
        harness.state.current_phase,
        available_tool_names=chat_tool_names,
        strategy_prompt=harness.strategy_prompt,
        manifest=load_index_manifest(harness.state.cwd),
        indexer_mode=bool(getattr(harness, "_indexer", False)),
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


async def prepare_planning_prompt(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]] | None:
    harness = deps.harness
    system_prompt = build_planning_prompt(
        harness.state,
        harness.state.current_phase,
        available_tool_names=_available_tool_names(harness, mode="planning"),
        strategy_prompt=harness.strategy_prompt,
        manifest=load_index_manifest(harness.state.cwd),
        indexer_mode=bool(getattr(harness, "_indexer", False)),
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


async def prepare_indexer_prompt(graph_state: GraphRunState, deps: GraphRuntimeDeps) -> list[dict[str, Any]] | None:
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
