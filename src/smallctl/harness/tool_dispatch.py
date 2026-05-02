from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope
from .tool_visibility import filter_tools_for_runtime_state
from .task_classifier import (
    classify_runtime_intent,
    classify_task_mode,
    looks_like_author_write_request,
    looks_like_write_file_request,
    looks_like_write_patch_request,
    runtime_policy_for_intent,
)
from .run_mode import (
    is_contextual_affirmative_execution_continuation,
    resolve_mode_task,
    should_enable_complex_write_chat_draft,
)
from .artifact_tracking import file_read_cache_key

_CHAT_WRITE_TOOL_NAMES = {"file_write", "file_patch", "ast_patch"}
_READONLY_CHAT_TOOL_BLOCKLIST = {
    "shell_exec",
    "ssh_exec",
    "ssh_file_read",
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "finalize_write_session",
    "file_write",
    "file_patch",
    "ast_patch",
    "file_delete",
}
_CHAT_TERMINAL_TOOL_NAMES = {"task_complete", "task_fail"}


def _scratchpad(harness: Any) -> dict[str, Any]:
    scratchpad = getattr(harness.state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        harness.state.scratchpad = scratchpad
    return scratchpad


def _refresh_task_mode(harness: Any, task: str) -> str:
    task_mode = classify_task_mode(task)
    harness.state.task_mode = task_mode
    return task_mode


def _refresh_runtime_intent(harness: Any, task: str) -> tuple[str, str]:
    runtime_intent = classify_runtime_intent(
        task,
        recent_messages=getattr(harness.state, "recent_messages", []),
    )
    _scratchpad(harness)["_chat_runtime_intent"] = runtime_intent.label
    return runtime_intent.label, runtime_intent.task_mode


def _has_active_write_session(harness: Any) -> bool:
    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is None:
        return False
    status = str(getattr(session, "status", "") or "").strip().lower()
    return status != "complete"


def _chat_write_tools_allowed(harness: Any, task: str) -> bool:
    if _has_active_write_session(harness):
        return True
    if (
        looks_like_write_patch_request(task)
        or looks_like_write_file_request(task)
        or looks_like_author_write_request(task)
    ):
        return True
    client = getattr(harness, "client", None)
    return should_enable_complex_write_chat_draft(
        task,
        model_name=getattr(client, "model", None),
        cwd=getattr(getattr(harness, "state", None), "cwd", None),
    )


def _filter_chat_tools_for_task_mode(
    harness: Any,
    tools: list[dict[str, Any]],
    *,
    task: str,
    task_mode: str,
) -> list[dict[str, Any]]:
    write_tools_allowed = _chat_write_tools_allowed(harness, task)
    if task_mode == "remote_execute":
        blocked_names = {"shell_exec", "finalize_write_session"}
    elif task_mode == "local_execute":
        blocked_names = {
            "ssh_exec",
            "ssh_file_read",
            "ssh_file_write",
            "ssh_file_patch",
            "ssh_file_replace_between",
            "finalize_write_session",
        }
    else:
        blocked_names = set(_READONLY_CHAT_TOOL_BLOCKLIST)
    if write_tools_allowed:
        blocked_names -= _CHAT_WRITE_TOOL_NAMES
    else:
        blocked_names |= _CHAT_WRITE_TOOL_NAMES
    filtered: list[dict[str, Any]] = []
    for entry in tools:
        function = entry.get("function") if isinstance(entry, dict) else None
        tool_name = str(function.get("name") or "") if isinstance(function, dict) else ""
        if tool_name in blocked_names:
            continue
        filtered.append(entry)
    _scratchpad(harness)["_chat_task_mode"] = task_mode
    return filtered


def _chat_terminal_tools(harness: Any) -> list[dict[str, Any]]:
    tools = harness.registry.export_openai_tools(
        phase=harness.state.current_phase,
        mode="chat",
        profiles=set(harness.state.active_tool_profiles),
    )
    return [
        entry
        for entry in tools
        if isinstance(entry, dict)
        and isinstance(entry.get("function"), dict)
        and str(entry["function"].get("name") or "") in _CHAT_TERMINAL_TOOL_NAMES
    ]


def _task_excerpt(task: str, *, limit: int = 160) -> str:
    text = " ".join(str(task or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _log_chat_tool_selection_error(
    harness: Any,
    *,
    exc: Exception,
    task: str,
    phase: str,
) -> None:
    harness._runlog(
        "chat_tool_selection_error",
        "chat tool exposure failed",
        exception_type=type(exc).__name__,
        error=str(exc),
        phase=phase,
        mode="chat",
        task_excerpt=_task_excerpt(task),
        suppression_reason=str(_scratchpad(harness).get("_chat_tools_suppressed_reason") or ""),
        tool_profiles=list(getattr(harness.state, "active_tool_profiles", []) or []),
        runtime_intent=str(_scratchpad(harness).get("_chat_runtime_intent") or ""),
    )


def chat_mode_requires_tools(harness: Any, task: str) -> bool:
    _refresh_task_mode(harness, task)
    runtime_intent = classify_runtime_intent(
        task,
        recent_messages=getattr(harness.state, "recent_messages", []),
    )
    _scratchpad(harness)["_chat_runtime_intent"] = runtime_intent.label
    runtime_policy = runtime_policy_for_intent(runtime_intent)
    model_name = getattr(harness.client, "model", None)
    if _has_active_write_session(harness):
        return True
    if should_enable_complex_write_chat_draft(
        task,
        model_name=model_name,
        cwd=getattr(harness.state, "cwd", None),
    ):
        return True
    raw_task, resolved_task = resolve_mode_task(harness, task)
    if is_contextual_affirmative_execution_continuation(
        harness,
        raw_task=raw_task,
        resolved_task=resolved_task,
    ):
        return True
    return runtime_policy.chat_requires_tools


def chat_mode_tools(harness: Any) -> list[dict[str, Any]]:
    task = ""
    selection_phase = "current_user_task"
    try:
        task = harness._current_user_task()
        runtime_intent_label, task_mode = _refresh_runtime_intent(harness, task)
        selection_phase = "requires_tools"
        if not chat_mode_requires_tools(harness, task):
            terminal_tools = _chat_terminal_tools(harness)
            _scratchpad(harness)["_chat_tools_exposed"] = bool(terminal_tools)
            _scratchpad(harness)["_chat_tools_suppressed_reason"] = "non_lookup_chat_terminal_only"
            harness._runlog(
                "chat_tool_selection",
                "chat tool exposure reduced to terminal tools",
                task=task,
                reason="non_lookup_chat_terminal_only",
                tool_names=[
                    str(entry["function"]["name"])
                    for entry in terminal_tools
                    if isinstance(entry, dict) and isinstance(entry.get("function"), dict)
                ],
                runtime_intent=runtime_intent_label,
            )
            return terminal_tools

        _scratchpad(harness)["_chat_tools_exposed"] = True
        _scratchpad(harness).pop("_chat_tools_suppressed_reason", None)
        selection_phase = "tool_export"
        tools = harness.registry.export_openai_tools(
            phase=harness.state.current_phase,
            mode="chat",
            profiles=set(harness.state.active_tool_profiles),
        )
        selection_phase = "task_mode_filter"
        tools = _filter_chat_tools_for_task_mode(harness, tools, task=task, task_mode=task_mode)
        selection_phase = "runtime_filter"
        tools = filter_tools_for_runtime_state(
            tools,
            state=harness.state,
            mode="chat",
        )
        selection_phase = "approval_gated_shell"
        shell_spec = harness.registry.get("shell_exec")
        active_tool_names = {
            str(entry["function"]["name"])
            for entry in tools
            if isinstance(entry, dict)
            and isinstance(entry.get("function"), dict)
            and "name" in entry["function"]
        }
        if (
            task_mode == "local_execute"
            and shell_spec is not None
            and shell_spec.profile_allowed(set(harness.state.active_tool_profiles))
            and "shell_exec" not in active_tool_names
        ):
            tools.append(shell_spec.openai_schema())
            harness._runlog(
                "chat_tool_selection",
                "shell execution exposed in chat mode",
                task=task,
                reason="approval_gated_shell",
                runtime_intent=runtime_intent_label,
            )
        return tools
    except Exception as exc:
        _log_chat_tool_selection_error(
            harness,
            exc=exc,
            task=task,
            phase=selection_phase,
        )
        try:
            terminal_tools = _chat_terminal_tools(harness)
        except Exception:
            terminal_tools = []
        _scratchpad(harness)["_chat_tools_exposed"] = bool(terminal_tools)
        _scratchpad(harness)["_chat_tools_suppressed_reason"] = "chat_tool_selection_error"
        return terminal_tools


async def dispatch_tool_call(harness: Any, tool_name: str, args: dict[str, Any]) -> ToolEnvelope:
    cached = maybe_reuse_file_read(harness, tool_name=tool_name, args=args)
    if cached is not None:
        return cached

    if tool_name not in harness.registry.names():
        sanitized = attempt_tool_sanitization(harness, tool_name)
        if sanitized:
            harness._runlog(
                "tool_sanitization",
                "hallucinated tool name detected and split",
                hallucinated=tool_name,
                sanitized=sanitized,
            )
            tool_name = sanitized

    return await harness.dispatcher.dispatch(tool_name, args)


def attempt_tool_sanitization(harness: Any, tool_name: str) -> str | None:
    for registered_name in harness.registry.names():
        if not registered_name:
            continue
        if tool_name.startswith(registered_name) and tool_name != registered_name:
            remainder = tool_name[len(registered_name) :]
            if remainder in harness.registry.names():
                return registered_name
    return None


def maybe_reuse_file_read(harness: Any, *, tool_name: str, args: dict[str, Any]) -> ToolEnvelope | None:
    if tool_name != "file_read":
        return None
    return _reuse_cached_file_read(harness, args)


def _reuse_cached_file_read(harness: Any, args: dict[str, Any]) -> ToolEnvelope | None:
    cache = harness.state.scratchpad.get("file_read_cache")
    if not isinstance(cache, dict):
        return None
    cache_key = file_read_cache_key(harness.state.cwd, args)
    if not cache_key:
        return None
    artifact_id = cache.get(cache_key)
    if not isinstance(artifact_id, str) or not artifact_id:
        return None
    artifact = harness.state.artifacts.get(artifact_id)
    if artifact is None:
        return None
    harness._runlog(
        "tool_cache_hit",
        "reusing prior file_read result",
        tool_name="file_read",
        artifact_id=artifact_id,
        path=artifact.source,
    )
    return ToolEnvelope(
        success=True,
        output={
            "status": "cached",
            "artifact_id": artifact_id,
            "path": artifact.source,
            "summary": artifact.summary,
        },
        metadata={
            "cache_hit": True,
            "artifact_id": artifact_id,
            "path": artifact.source,
            "tool_name": "file_read",
        },
    )
