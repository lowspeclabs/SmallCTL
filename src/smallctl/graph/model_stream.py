from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from ..client import OpenAICompatClient, StreamResult, format_tool_call_text, maybe_parse_tool_args
from ..guards import is_four_b_or_under_model_name
from ..harness.run_mode import should_enable_complex_write_chat_draft
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import clip_text_value, json_safe_value
from ..task_targets import primary_task_target_path
from .deps import GraphRuntimeDeps
from .recovery_context import build_goal_recap
from .state import GraphRunState, PendingToolCall
from .tool_call_parser import (
    _detect_empty_file_write_payload,
    _ensure_chunk_write_session,
    _suggested_chunk_sections,
)
from .write_recovery import (
    build_synthetic_file_write_call,
    can_safely_synthesize,
    infer_write_target_path,
    recover_write_intent,
    recover_content_from_assistant_text,
    write_recovery_kind,
    write_recovery_metadata,
)

_PROVIDER_HTTP_STATUS_CODES = {500, 502, 503, 504, 530}
_CONTEXT_WINDOW_OVERFLOW_RE = re.compile(
    r"n_keep\s*:\s*(\d+)\s*>=\s*n_ctx\s*:\s*(\d+)",
    re.IGNORECASE,
)


@dataclass
class StreamProcessingResult:
    chunks: list[dict[str, Any]] = field(default_factory=list)
    stream: Any = None
    timeline: list[Any] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    ttft: float = 0.0
    halted: bool = False
    halt_reason: str = ""
    halt_details: dict[str, Any] = field(default_factory=dict)


def _classify_model_call_error(exc: Exception) -> tuple[str, dict[str, Any]]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    details: dict[str, Any] = {}
    if isinstance(status_code, int):
        details["status_code"] = status_code
        if status_code in _PROVIDER_HTTP_STATUS_CODES:
            return "provider", details
    return "stream", details


def _with_speaker(harness: Any, data: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = dict(data or {})
    if getattr(harness.state, "planning_mode_enabled", False):
        payload.setdefault("speaker", "planner")
    return payload


def _parse_context_window_overflow(
    error: str,
    details: dict[str, Any] | None = None,
) -> tuple[int, int] | None:
    candidates = [str(error or "")]
    if isinstance(details, dict):
        for key in ("message", "error", "detail"):
            value = details.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)
    for candidate in candidates:
        match = _CONTEXT_WINDOW_OVERFLOW_RE.search(candidate)
        if match is None:
            continue
        return int(match.group(1)), int(match.group(2))
    return None


def _format_partial_tool_calls(tool_calls: list[dict[str, Any]]) -> list[str]:
    summaries: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = str(function.get("name") or "").strip() or "tool_call"
        args_text = str(function.get("arguments") or "")
        args = maybe_parse_tool_args(args_text)
        display = format_tool_call_text(tool_name, args_text, args)
        tool_call_id = str(tool_call.get("id") or "").strip()
        if tool_call_id:
            summaries.append(f"{tool_name} (tool_call_id={tool_call_id}): {display}")
        else:
            summaries.append(f"{tool_name}: {display}")
    return summaries


def _build_tool_specific_recovery_hint(
    *,
    harness: Any,
    partial_tool_calls: list[dict[str, Any]],
) -> str:
    registry = getattr(harness, "registry", None)
    for tool_call in partial_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = str(function.get("name") or "").strip()
        if not tool_name:
            continue

        if tool_name in {"file_write", "file_append"}:
            target_path = primary_task_target_path(harness)
            session = _ensure_chunk_write_session(harness, target_path) if target_path else None
            if session is None:
                session = getattr(getattr(harness, "state", None), "write_session", None)
                if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
                    session = None
            if session is not None:
                section_name = session.write_next_section or session.write_current_section or "imports"
                return (
                    f"For `{tool_name}`, continue Write Session `{session.write_session_id}` for "
                    f"`{session.write_target_path}`. Include required fields `path` and `content`, plus "
                    f"`write_session_id='{session.write_session_id}'` and `section_name='{section_name}'`. "
                    "If additional chunks remain, include `next_section_name='...'`; omit it on the final chunk. "
                    "When resuming an active write session, prefer `file_write` rather than a bare `file_append`. "
                    "Regenerate the complete JSON tool call from scratch."
                )
            path_hint = f" Target path for this task: `{target_path}`." if target_path else ""
            return (
                f"For `{tool_name}`, include both required fields: `path` and `content`."
                f"{path_hint} "
                "If the full implementation is too large for one write, start with a small valid scaffold "
                "such as imports, a module docstring, entrypoints, or TODO stubs, then extend it in later writes. "
                "Empty file writes are currently allowed if the user explicitly asked for that. "
                "Regenerate the complete JSON tool call from scratch."
            )

        if registry is not None:
            tool_spec = registry.get(tool_name)
            if tool_spec is not None:
                required = tool_spec.schema.get("required", [])
                if required:
                    required_text = ", ".join(str(field) for field in required)
                    return (
                        f"For `{tool_name}`, include the required fields: {required_text}. "
                        "Regenerate the complete JSON tool call from scratch."
                    )

        return f"Regenerate the complete `{tool_name}` JSON tool call from scratch with all required arguments."
    return ""


def _build_incomplete_tool_call_recovery_message(
    *,
    harness: Any,
    assistant_text: str,
    partial_tool_calls: list[dict[str, Any]],
) -> str:
    lines = [
        "The previous assistant response was interrupted before a tool call finished streaming.",
        "Please regenerate the full tool call from scratch with complete JSON arguments.",
    ]
    goal_recap = build_goal_recap(harness)
    if goal_recap:
        lines.append(goal_recap)
    clipped_assistant_text, assistant_was_clipped = clip_text_value(assistant_text.strip(), limit=600)
    if clipped_assistant_text:
        prefix = "Assistant text before interruption"
        if assistant_was_clipped:
            prefix += " [truncated]"
        lines.insert(1, f"{prefix}: {clipped_assistant_text}")
    tool_hint = _build_tool_specific_recovery_hint(harness=harness, partial_tool_calls=partial_tool_calls)
    if tool_hint:
        lines.append(tool_hint)
    if partial_tool_calls:
        lines.append("Partial tool call(s) seen before the timeout:")
        for item in _format_partial_tool_calls(partial_tool_calls)[:3]:
            clipped_item, _ = clip_text_value(item, limit=320)
            lines.append(f"- {clipped_item}")
        if len(partial_tool_calls) > 3:
            lines.append(f"- ...and {len(partial_tool_calls) - 3} more")
    return "\n".join(lines)


def _active_text_write_fallback_session(harness: Any) -> Any | None:
    state = getattr(harness, "state", None)
    session = getattr(state, "write_session", None)
    if session is None:
        return None
    if str(getattr(session, "status", "")).strip().lower() == "complete":
        return None

    target_path = primary_task_target_path(harness)
    session_target = str(getattr(session, "write_target_path", "") or "").strip()
    if target_path and session_target:
        try:
            from ..tools.fs import _same_target_path

            if not _same_target_path(session_target, target_path, getattr(state, "cwd", None)):
                return None
        except Exception:
            if session_target != target_path:
                return None
    return session


def _is_sub4b_write_timeout(
    harness: Any,
    *,
    details: dict[str, Any],
    partial_tool_calls: list[dict[str, Any]],
) -> bool:
    """Return True when all of these hold:
    - The error reason is tool_call_continuation_timeout or read_timeout
    - The partial tool call is file_write or file_append
    - The active model is ≤4 B parameters

    Used to trigger an immediate chat-mode fallback on the first chunk
    error rather than burning through all retries.
    """
    reason = str(details.get("reason") or "").strip().lower()
    if reason not in {"tool_call_continuation_timeout", "read_timeout"}:
        return False
    if reason == "read_timeout" and not bool(details.get("tool_call_stream_active", True)):
        return False

    write_tools = {"file_write", "file_append"}
    has_write_call = any(
        str(
            (tc.get("function") or {}).get("name") if isinstance(tc, dict) else ""
        ).strip() in write_tools
        for tc in partial_tool_calls
    )
    if not has_write_call:
        return False

    from ..guards import is_four_b_or_under_model_name

    model_name = getattr(getattr(harness, "client", None), "model", None)
    if model_name is None:
        model_name = harness.state.scratchpad.get("_model_name")
        
    return is_four_b_or_under_model_name(model_name)


def _fallback_section_name(session: Any) -> str:
    for field_name in ("write_next_section", "write_current_section"):
        value = str(getattr(session, field_name, "") or "").strip()
        if value:
            return value
    return "imports"


def _fallback_next_section_name(session: Any, current_section: str) -> str:
    sections = [
        str(section).strip()
        for section in getattr(session, "suggested_sections", []) or []
        if str(section).strip()
    ]
    if not sections:
        return ""
    current = str(current_section or "").strip()
    if not current:
        return sections[0]
    for idx, section in enumerate(sections):
        if section != current:
            continue
        if idx + 1 < len(sections):
            return sections[idx + 1]
        return ""
    return ""


def _build_text_write_fallback_prompt(
    *,
    session: Any,
    current_section: str,
    remaining_sections: list[str],
    task_text: str = "",
) -> str:
    completed_sections = [
        str(section).strip()
        for section in getattr(session, "write_sections_completed", []) or []
        if str(section).strip()
    ]
    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    session_intent = str(getattr(session, "write_session_intent", "") or "").strip()
    remaining_text = ", ".join(remaining_sections) if remaining_sections else "none"
    completed_text = ", ".join(completed_sections) if completed_sections else "none"
    lines = [
        "Write only the code for the current section below.",
        "Return exactly one fenced code block and no prose.",
        "Start the fenced code block immediately and close it as soon as the section is complete.",
    ]
    task_preview = clip_text_value(str(task_text or "").strip(), limit=1200)[0]
    if task_preview:
        lines.extend(
            [
                f"Original task: {task_preview}",
                "Stay aligned with the original task. Do not switch goals, summarize, or explore.",
            ]
        )
    lines.extend(
        [
            "Do not emit tool calls, XML wrappers, or `memory_update`; output code only.",
            "If you start a tool-style tag, JSON object, or explanation, stop and emit the code fence instead.",
            "Keep the chunk small, ideally under 50 lines, and write only one logical section at a time.",
            f"Target path: `{target_path}`.",
            f"Write intent: `{session_intent}`.",
            f"Current section: `{current_section}`.",
            f"Completed sections: {completed_text}.",
            f"Remaining suggested sections: {remaining_text}.",
            "The harness will insert the code into the active write session.",
        ]
    )
    return "\n".join(lines)


def _collect_tool_call_names(tool_calls: list[dict[str, Any]] | None) -> list[str]:
    names: set[str] = set()
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if name:
            names.add(name)
    return sorted(names)


def _fallback_task_text(harness: Any, messages: list[dict[str, Any]]) -> str:
    run_brief = getattr(getattr(harness, "state", None), "run_brief", None)
    task = str(getattr(run_brief, "original_task", "") or "").strip()
    if task:
        return task

    current_user_task = getattr(harness, "_current_user_task", None)
    if callable(current_user_task):
        try:
            text = str(current_user_task() or "").strip()
        except Exception:
            text = ""
        if text and not _looks_like_harness_recovery_message(text):
            return text

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        content = str(message.get("content") or "").strip()
        if content and not _looks_like_harness_recovery_message(content):
            return content
    return ""


def _looks_like_harness_recovery_message(text: str) -> bool:
    cleaned = str(text or "").strip().lower()
    if not cleaned:
        return False
    prefixes = (
        "tool call '",
        "empty payload rejected for `",
        "empty payload rejected for '",
        "you are still in the discovery phase.",
        "you are in verification.",
        "transitioning to verification phase.",
        "transitioning to synthesis phase.",
        "anti-laziness:",
    )
    return cleaned.startswith(prefixes)


def _seed_text_write_fallback_session(
    harness: Any,
    *,
    session: Any | None,
    partial_tool_calls: list[dict[str, Any]] | None,
) -> Any:
    if session is not None:
        return session

    target_path, _, _ = infer_write_target_path(
        harness=harness,
        pending=None,
        assistant_text="",
        partial_tool_calls=partial_tool_calls,
    )
    suggested_sections = _suggested_chunk_sections(target_path) if target_path else []
    first_section = suggested_sections[0] if suggested_sections else ""
    return SimpleNamespace(
        write_session_id="",
        write_target_path=target_path,
        write_session_intent="",
        write_sections_completed=[],
        suggested_sections=suggested_sections,
        write_next_section=first_section,
        write_current_section=first_section,
    )


def _should_attempt_empty_payload_text_fallback(
    harness: Any,
    graph_state: GraphRunState,
    *,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
) -> bool:
    if str(getattr(graph_state, "run_mode", "") or "").strip().lower() != "chat":
        return False

    tool_names = _collect_tool_call_names(tool_calls)
    if not tool_names or any(name not in {"file_write", "file_append"} for name in tool_names):
        return False

    session = _active_text_write_fallback_session(harness)
    if session is not None and str(getattr(session, "write_session_intent", "") or "").strip().lower() in {
        "replace_file",
        "patch_existing",
    }:
        return True

    target_path, _, _ = infer_write_target_path(
        harness=harness,
        pending=None,
        assistant_text="",
        partial_tool_calls=tool_calls,
    )
    if not target_path:
        return False

    model_name = getattr(getattr(harness, "client", None), "model", None)
    if model_name is None:
        model_name = getattr(getattr(harness, "state", None), "scratchpad", {}).get("_model_name")

    task_text = _fallback_task_text(harness, messages)
    if task_text and should_enable_complex_write_chat_draft(
        task_text,
        model_name=model_name,
        cwd=getattr(getattr(harness, "state", None), "cwd", None),
    ):
        return True

    # If we still have a single confident target path, allow the no-tools
    # rescue for <=4b chat-mode writes even when an earlier summary/note lost
    # the path from task text.
    return is_four_b_or_under_model_name(model_name)


def _extract_code_from_fallback_response(
    text: str,
    *,
    target_path: str = "",
    path_confidence: str = "low",
) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    match = re.search(r"```[^\n`]*\n?(.*?)```", cleaned, re.DOTALL)
    if match is not None:
        code = match.group(1).strip()
        return code

    extracted_text, _confidence, _evidence = recover_content_from_assistant_text(
        cleaned,
        target_path=target_path,
        allow_raw_text_targets=False,
        path_confidence=path_confidence,
    )
    if extracted_text:
        return extracted_text
    
    # Do not treat XML tool calls as raw fallback code
    if "<tool_call>" in cleaned or "<function=" in cleaned:
        return ""
        
    return cleaned


def _fallback_response_ready_for_early_exit(
    text: str,
    *,
    target_path: str = "",
    path_confidence: str = "low",
) -> bool:
    cleaned = str(text or "").strip()
    if not cleaned:
        return False

    if re.search(r"```[^\n`]*\n?(.*?)```", cleaned, re.DOTALL):
        return True

    extracted_text, _confidence, evidence = recover_content_from_assistant_text(
        cleaned,
        target_path=target_path,
        allow_raw_text_targets=False,
        path_confidence=path_confidence,
    )
    return bool(extracted_text and "assistant_inline_tool_block" in evidence)


def _build_synthetic_write_tool_call(
    session: Any,
    content: str,
    section_name: str,
    next_section_name: str,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "path": str(getattr(session, "write_target_path", "") or "").strip(),
        "content": str(content),
        "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
        "section_name": str(section_name or "").strip() or "imports",
    }
    if next_section_name:
        args["next_section_name"] = str(next_section_name).strip()
    return {
        "id": f"text_write_fallback_{args['write_session_id'] or 'session'}",
        "type": "function",
        "function": {
            "name": "file_write",
            "arguments": json.dumps(args, ensure_ascii=True, sort_keys=True),
        },
    }


def _record_text_write_fallback_state(
    harness: Any,
    *,
    status: str,
    session: Any,
    current_section: str,
    remaining_sections: list[str],
    prompt: str,
    assistant_text: str = "",
    extracted_code: str = "",
    next_section_name: str = "",
    reason: str = "",
    tool_names: list[str] | None = None,
) -> None:
    harness.state.scratchpad["_last_text_write_fallback"] = {
        "status": status,
        "reason": reason,
        "session_id": str(getattr(session, "write_session_id", "") or ""),
        "target_path": str(getattr(session, "write_target_path", "") or ""),
        "session_intent": str(getattr(session, "write_session_intent", "") or ""),
        "current_section": current_section,
        "next_section_name": next_section_name,
        "remaining_sections": remaining_sections,
        "tool_names": tool_names or [],
        "prompt_preview": clip_text_value(prompt, limit=500)[0],
        "assistant_preview": clip_text_value(assistant_text, limit=800)[0],
        "code_preview": clip_text_value(extracted_code, limit=800)[0],
        "prompt_chars": len(prompt),
        "assistant_chars": len(assistant_text),
        "code_chars": len(extracted_code),
    }


def _build_text_write_fallback_trace(
    *,
    session: Any,
    current_section: str,
    prompt: str,
    assistant_text: str,
    extracted_code: str,
    next_section_name: str,
    tool_names: list[str],
) -> str:
    target_path = str(getattr(session, "write_target_path", "") or "").strip() or "(unknown)"
    prompt_excerpt, prompt_was_clipped = clip_text_value(prompt, limit=1000)
    response_excerpt, response_was_clipped = clip_text_value(assistant_text, limit=1200)
    code_excerpt, code_was_clipped = clip_text_value(extracted_code, limit=1000)

    lines = [
        "Chat-mode fallback activated for a complex write task.",
        f"Target path: `{target_path}`.",
        f"Current section: `{current_section or 'imports'}`.",
        f"Next section: `{next_section_name or 'none'}`.",
        f"Observed tool calls: {', '.join(tool_names) if tool_names else 'none'}.",
        "",
        "Prompt sent to the model:",
        prompt_excerpt + (" [truncated]" if prompt_was_clipped else ""),
        "",
        "Assistant response:",
        response_excerpt + (" [truncated]" if response_was_clipped else ""),
        "",
        "Extraction method:",
        "first fenced code block, otherwise raw response",
        "",
        "Extracted script or key details:",
        code_excerpt if code_excerpt else "(no extractable script found)",
    ]
    if code_was_clipped and code_excerpt:
        lines[-1] += " [truncated]"
    return "\n".join(lines)


async def _emit_text_write_fallback_trace(
    harness: Any,
    deps: GraphRuntimeDeps,
    *,
    session: Any,
    current_section: str,
    prompt: str,
    assistant_text: str,
    extracted_code: str,
    next_section_name: str,
    tool_names: list[str],
) -> None:
    trace_text = _build_text_write_fallback_trace(
        session=session,
        current_section=current_section,
        prompt=prompt,
        assistant_text=assistant_text,
        extracted_code=extracted_code,
        next_section_name=next_section_name,
        tool_names=tool_names,
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ASSISTANT,
            content=trace_text,
            data={
                "kind": "print",
                "artifact_id": "chat-mode-fallback-trace",
                "speaker": "assistant",
            },
        ),
    )


async def _attempt_text_write_fallback(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    messages: list[dict[str, Any]],
    source_chunks: list[dict[str, Any]],
    partial_tool_calls: list[dict[str, Any]],
    session: Any | None,
    reason: str,
    start_time: float,
    first_token_time: float | None,
) -> StreamProcessingResult | None:
    harness = deps.harness
    session_context = _seed_text_write_fallback_session(
        harness,
        session=session,
        partial_tool_calls=partial_tool_calls,
    )
    tool_names = _collect_tool_call_names(partial_tool_calls)
    current_section = _fallback_section_name(session_context)
    remaining_sections = [
        str(section).strip()
        for section in getattr(session_context, "suggested_sections", []) or []
        if str(section).strip()
    ]
    if current_section and remaining_sections:
        try:
            current_index = remaining_sections.index(current_section)
        except ValueError:
            current_index = -1
        if current_index >= 0:
            remaining_sections = remaining_sections[current_index + 1 :]
    task_text = _fallback_task_text(harness, messages)

    fallback_prompt = _build_text_write_fallback_prompt(
        session=session_context,
        current_section=current_section,
        remaining_sections=remaining_sections,
        task_text=task_text,
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content="Chat-mode fallback activated for a complex write task.",
            data={
                "status_activity": "chat-mode fallback active",
                "target_path": str(getattr(session_context, "write_target_path", "") or "").strip(),
                "write_session_id": str(getattr(session_context, "write_session_id", "") or "").strip(),
                "reason": reason,
            },
        ),
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ASSISTANT,
            content="Fallback progress: switching to a code-only rescue pass for the stalled write.",
            data={"status_activity": "fallback rescue in progress"},
        ),
    )
    harness._runlog(
        "stream_text_write_fallback_progress",
        "fallback rescue in progress",
        status_activity="fallback rescue in progress",
        write_session_id=str(getattr(session_context, "write_session_id", "") or ""),
        target_path=str(getattr(session_context, "write_target_path", "") or ""),
        current_section=current_section,
        reason=reason,
    )
    fallback_messages = list(messages) + [
        {
            "role": "system",
            "content": fallback_prompt,
        }
    ]
    graph_state.latency_metrics["text_write_fallback_attempt_count"] = (
        int(graph_state.latency_metrics.get("text_write_fallback_attempt_count", 0) or 0) + 1
    )
    harness._runlog(
        "stream_text_write_fallback_attempt",
        "attempting no-tools write rescue after unusable write call",
        write_session_id=str(getattr(session_context, "write_session_id", "") or ""),
        target_path=str(getattr(session_context, "write_target_path", "") or ""),
        current_section=current_section,
        session_intent=str(getattr(session_context, "write_session_intent", "") or ""),
        reason=reason,
    )
    _record_text_write_fallback_state(
        harness,
        status="attempting",
        reason=reason,
        session=session_context,
        current_section=current_section,
        remaining_sections=remaining_sections,
        prompt=fallback_prompt,
        tool_names=tool_names,
    )
    fallback_chunks: list[dict[str, Any]] = []
    try:
        async for event in harness.client.stream_chat(messages=fallback_messages, tools=[]):
            fallback_chunks.append(event)
            partial_stream = OpenAICompatClient.collect_stream(
                fallback_chunks,
                reasoning_mode=harness.reasoning_mode,
                thinking_start_tag=harness.thinking_start_tag,
                thinking_end_tag=harness.thinking_end_tag,
            )
            if _fallback_response_ready_for_early_exit(
                partial_stream.assistant_text,
                target_path=str(getattr(session_context, "write_target_path", "") or "").strip(),
                path_confidence="high"
                if str(getattr(session_context, "write_target_path", "") or "").strip()
                else "low",
            ):
                harness._runlog(
                    "stream_text_write_fallback_early_exit",
                    "stopped no-tools rescue after receiving a complete recoverable code block",
                    write_session_id=str(getattr(session_context, "write_session_id", "") or ""),
                    target_path=str(getattr(session_context, "write_target_path", "") or ""),
                    reason=reason,
                )
                break
    except Exception as exc:
        harness._runlog(
            "stream_text_write_fallback_failed",
            "no-tools write rescue errored; using original stream",
            error=str(exc),
            write_session_id=str(getattr(session_context, "write_session_id", "") or ""),
            reason=reason,
        )
        _record_text_write_fallback_state(
            harness,
            status="failed",
            reason=f"stream_error: {exc}",
            session=session_context,
            current_section=current_section,
            remaining_sections=remaining_sections,
            prompt=fallback_prompt,
            tool_names=tool_names,
        )
        return None

    fallback_stream = OpenAICompatClient.collect_stream(
        fallback_chunks,
        reasoning_mode=harness.reasoning_mode,
        thinking_start_tag=harness.thinking_start_tag,
        thinking_end_tag=harness.thinking_end_tag,
    )
    fallback_timeline = OpenAICompatClient.collect_timeline(
        fallback_chunks,
        reasoning_mode=harness.reasoning_mode,
        thinking_start_tag=harness.thinking_start_tag,
        thinking_end_tag=harness.thinking_end_tag,
    )
    harness.state.scratchpad["_last_text_write_fallback_assistant_text"] = fallback_stream.assistant_text
    extracted_code = _extract_code_from_fallback_response(
        fallback_stream.assistant_text,
        target_path=str(getattr(session_context, "write_target_path", "") or "").strip(),
        path_confidence="high" if str(getattr(session_context, "write_target_path", "") or "").strip() else "low",
    )
    next_section_name = _fallback_next_section_name(session_context, current_section)
    await _emit_text_write_fallback_trace(
        harness,
        deps,
        session=session_context,
        current_section=current_section,
        prompt=fallback_prompt,
        assistant_text=fallback_stream.assistant_text,
        extracted_code=extracted_code,
        next_section_name=next_section_name,
        tool_names=tool_names,
    )
    fallback_intent = recover_write_intent(
        harness=harness,
        pending=None,
        assistant_text=fallback_stream.assistant_text,
        partial_tool_calls=partial_tool_calls,
    )
    if fallback_intent is not None:
        graph_state.latency_metrics["write_recovery_attempt_count"] = (
            int(graph_state.latency_metrics.get("write_recovery_attempt_count", 0) or 0) + 1
        )
        harness.state.scratchpad["_last_write_recovery"] = write_recovery_metadata(
            fallback_intent,
            status="attempt",
        )
        harness._runlog(
            "write_recovery_attempt",
            "attempting write-intent recovery from stream fallback",
            path=fallback_intent.path,
            confidence=fallback_intent.confidence,
            evidence=fallback_intent.evidence,
            recovery_kind=write_recovery_kind(fallback_intent),
            source=fallback_intent.source,
            reason=reason,
        )
    if fallback_intent is not None and can_safely_synthesize(fallback_intent, harness=harness):
        synthetic_call = build_synthetic_file_write_call(fallback_intent)
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ASSISTANT,
                content="Fallback progress: recovered a usable code block and is finalizing the write.",
                data={"status_activity": "fallback rescue finalizing"},
            ),
        )
        harness._runlog(
            "stream_text_write_fallback_progress",
            "fallback rescue finalizing",
            status_activity="fallback rescue finalizing",
            write_session_id=str(getattr(session_context, "write_session_id", "") or ""),
            target_path=str(getattr(session_context, "write_target_path", "") or ""),
            current_section=fallback_intent.section_name or current_section,
            reason=reason,
        )
        harness._runlog(
            "stream_text_write_fallback_succeeded",
            "converted no-tools write response into synthetic file_write",
            write_session_id=fallback_intent.write_session_id,
            target_path=fallback_intent.path,
            current_section=fallback_intent.section_name or current_section,
            next_section_name=fallback_intent.next_section_name or next_section_name,
            recovery_kind=write_recovery_kind(fallback_intent),
            reason=reason,
        )
        graph_state.latency_metrics["write_recovery_success_count"] = (
            int(graph_state.latency_metrics.get("write_recovery_success_count", 0) or 0) + 1
        )
        if "assistant_fenced_code" in fallback_intent.evidence or "assistant_inline_tool_block" in fallback_intent.evidence:
            graph_state.latency_metrics["write_recovery_from_assistant_code_count"] = (
                int(graph_state.latency_metrics.get("write_recovery_from_assistant_code_count", 0) or 0) + 1
            )
        harness.state.scratchpad["_last_write_recovery"] = write_recovery_metadata(
            fallback_intent,
            status="synthesized",
        )
        _record_text_write_fallback_state(
            harness,
            status="succeeded",
            reason="code_extracted",
            session=session_context,
            current_section=current_section,
            remaining_sections=remaining_sections,
            prompt=fallback_prompt,
            assistant_text=fallback_stream.assistant_text,
            extracted_code=extracted_code,
            next_section_name=next_section_name,
            tool_names=tool_names,
        )
        graph_state.latency_metrics["text_write_fallback_success_count"] = (
            int(graph_state.latency_metrics.get("text_write_fallback_success_count", 0) or 0) + 1
        )
        usage_payload = fallback_stream.usage
        if not isinstance(usage_payload, dict):
            usage_payload = {}
        end_time = time.perf_counter()
        duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else duration
        return StreamProcessingResult(
            chunks=source_chunks,
            stream=StreamResult(
                assistant_text=fallback_stream.assistant_text,
                thinking_text="",
                tool_calls=[synthetic_call],
                usage=usage_payload,
            ),
            timeline=fallback_timeline,
            usage=usage_payload,
            duration=duration,
            ttft=ttft,
            halted=False,
            halt_reason="",
            halt_details={},
        )

    if fallback_intent is not None:
        graph_state.latency_metrics["write_recovery_declined_count"] = (
            int(graph_state.latency_metrics.get("write_recovery_declined_count", 0) or 0) + 1
        )
        if str(fallback_intent.confidence).strip().lower() == "low":
            graph_state.latency_metrics["write_recovery_low_confidence_count"] = (
                int(graph_state.latency_metrics.get("write_recovery_low_confidence_count", 0) or 0) + 1
            )
        harness.state.scratchpad["_last_write_recovery"] = write_recovery_metadata(
            fallback_intent,
            status="declined",
        )
    harness._runlog(
        "stream_text_write_fallback_failed",
        "no-tools write rescue returned no usable code; using original stream",
        write_session_id=str(getattr(session_context, "write_session_id", "") or ""),
        target_path=str(getattr(session_context, "write_target_path", "") or ""),
        reason=reason,
    )
    _record_text_write_fallback_state(
        harness,
        status="failed",
        reason="empty_or_unusable_response",
        session=session_context,
        current_section=current_section,
        remaining_sections=remaining_sections,
        prompt=fallback_prompt,
        assistant_text=fallback_stream.assistant_text,
        tool_names=tool_names,
    )
    return None


async def process_model_stream(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> StreamProcessingResult:
    """Stream a model call, process chunks, and return assembled results."""
    harness = deps.harness
    event_handler = getattr(harness, "event_handler", None)
    echo_to_stdout = event_handler is None
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
    timeout_recovery_nudges = 0
    last_chunk_error_details: dict[str, Any] | None = None
    salvage_partial_stream: StreamResult | None = None
    stream_ended_without_done = False
    stream_ended_without_done_details: dict[str, Any] = {}
    harness.state.scratchpad.pop("_last_incomplete_tool_call", None)
    harness.state.scratchpad.pop("_last_text_write_fallback_assistant_text", None)

    _CHUNK_ERROR_MAX_RETRIES = 2
    # Set to True when we detect a ≤4b model repeatedly unable to stream a
    # file_write payload.  When True we break out of the retry loop after the
    # very first chunk error and fall through to the chat-mode fallback.
    # Persisted in scratchpad so the mode remains active for the entire task;
    # cleared automatically at task boundary via _reset_task_boundary_state.
    _trigger_early_4b_fallback: bool = bool(
        harness.state.scratchpad.get("_sub4b_chat_fallback_active")
    )
    # Set to True when the inner stream loop completes cleanly (no chunk_error).
    # Used to distinguish a successful stream from error-path exits.
    _stream_completed_cleanly: bool = False
    for _model_attempt in range(_CHUNK_ERROR_MAX_RETRIES + 1):
        try:
            chunks = []
            inside_tag = False
            buffer = ""
            _retry_immediately = False
            async for event in harness.client.stream_chat(messages=messages, tools=tools):
                if harness._cancel_requested:
                    await harness._emit(
                        deps.event_handler,
                        UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
                    )
                    graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
                    return StreamProcessingResult(chunks=chunks)
                if event.get("type") == "chunk_error":
                    err_msg = event.get("error", "unknown upstream error")
                    details = event.get("details")
                    if not isinstance(details, dict):
                        details = {}
                    last_chunk_error_details = details
                    retrying = _model_attempt < _CHUNK_ERROR_MAX_RETRIES
                    overflow = _parse_context_window_overflow(err_msg, details)
                    if overflow is not None:
                        n_keep, n_ctx = overflow
                        rebuild_messages = getattr(harness, "_rebuild_messages_after_context_overflow", None)
                        if callable(rebuild_messages):
                            try:
                                replacement_messages = await rebuild_messages(
                                    n_ctx=n_ctx,
                                    n_keep=n_keep,
                                    error_message=err_msg,
                                    event_handler=deps.event_handler,
                                )
                            except RuntimeError as exc:
                                await harness._emit(
                                    deps.event_handler,
                                    UIEvent(event_type=UIEventType.ERROR, content=str(exc)),
                                )
                                graph_state.final_result = harness._failure(
                                    str(exc),
                                    error_type="prompt_budget",
                                )
                                graph_state.error = graph_state.final_result["error"]
                                return StreamProcessingResult(chunks=chunks)
                            if replacement_messages:
                                messages = replacement_messages
                                _retry_immediately = True
                                harness._runlog(
                                    "stream_context_shrink",
                                    "shrinking prompt context after upstream n_keep overflow",
                                    error=err_msg,
                                    attempt=_model_attempt + 1,
                                    retrying=retrying,
                                    details=details,
                                    n_keep=n_keep,
                                    n_ctx=n_ctx,
                                    max_prompt_tokens=getattr(harness.context_policy, "max_prompt_tokens", None),
                                )
                                if retrying:
                                    await harness._emit(
                                        deps.event_handler,
                                        UIEvent(
                                            event_type=UIEventType.ALERT,
                                            content=f"Stream chunk error (shrinking context and retrying): {err_msg}",
                                            data={
                                                "is_api_error": True,
                                                "retrying": True,
                                                "attempt": _model_attempt + 1,
                                                "details": details,
                                                "recovery": "context_shrink",
                                            },
                                        ),
                                    )
                                break
                    if details.get("reason") == "tool_call_continuation_timeout":
                        partial_stream = OpenAICompatClient.collect_stream(
                            chunks,
                            reasoning_mode=harness.reasoning_mode,
                            thinking_start_tag=harness.thinking_start_tag,
                            thinking_end_tag=harness.thinking_end_tag,
                        )
                        salvage_partial_stream = partial_stream
                        if timeout_recovery_nudges < 2:
                            partial_tool_calls = _format_partial_tool_calls(partial_stream.tool_calls)
                            recovery_message = _build_incomplete_tool_call_recovery_message(
                                harness=harness,
                                assistant_text=partial_stream.assistant_text,
                                partial_tool_calls=partial_stream.tool_calls,
                            )
                            recovery_payload = {
                                "kind": "incomplete_tool_call",
                                "attempt": _model_attempt + 1,
                                "error": err_msg,
                                "details": details,
                                "assistant_text": partial_stream.assistant_text,
                                "thinking_text": partial_stream.thinking_text,
                                "partial_tool_calls": partial_tool_calls,
                                "partial_tool_calls_raw": json_safe_value(partial_stream.tool_calls),
                                "message": recovery_message,
                            }
                            harness.state.scratchpad["_last_incomplete_tool_call"] = recovery_payload
                            recovery_metadata = {
                                "is_recovery_nudge": True,
                                "recovery_kind": "incomplete_tool_call",
                                "attempt": _model_attempt + 1,
                                "partial_tool_calls": partial_tool_calls,
                                "tool_call_count": len(partial_tool_calls),
                            }
                            recovery_message_obj = ConversationMessage(
                                role="system",
                                content=recovery_message,
                                metadata=recovery_metadata,
                            )
                            harness.state.append_message(recovery_message_obj)
                            messages.append(recovery_message_obj.to_dict())
                            timeout_recovery_nudges += 1

                        # ── Early-exit path for ≤4b models ──────────────────
                        # These models cannot stream large JSON tool-call
                        # payloads reliably.  Retrying is futile; pivot
                        # immediately to chat-mode fallback on the FIRST
                        # detected write-tool timeout.
                        if not _trigger_early_4b_fallback and _is_sub4b_write_timeout(
                            harness,
                            details=details,
                            partial_tool_calls=partial_stream.tool_calls,
                        ):
                            _trigger_early_4b_fallback = True
                            # Persist so subsequent process_model_stream calls
                            # within the same task also skip straight to fallback.
                            harness.state.scratchpad["_sub4b_chat_fallback_active"] = True
                            harness._runlog(
                                "stream_chunk_error",
                                "sub-4b write timeout: skipping retries, proceeding to chat-mode fallback",
                                error=err_msg,
                                attempt=_model_attempt + 1,
                                retrying=False,
                                details=details,
                            )
                            break  # break inner → fall through to else-clause immediately

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
                if event.get("type") == "backend_wedged":
                    details = event.get("details")
                    if not isinstance(details, dict):
                        details = {}
                    graph_state.latency_metrics["backend_wedged_count"] = (
                        int(graph_state.latency_metrics.get("backend_wedged_count", 0) or 0) + 1
                    )
                    harness._runlog(
                        "backend_wedged",
                        "backend did not emit a first token before timeout",
                        details=details,
                    )
                    await harness._emit(
                        deps.event_handler,
                        UIEvent(
                            event_type=UIEventType.ERROR,
                            content="Backend did not emit a first token before timeout. Automatic recovery did not succeed.",
                            data={"is_api_error": True, "details": details},
                        ),
                    )
                    graph_state.final_result = harness._failure(
                        "Backend did not emit a first token before timeout",
                        error_type="provider",
                        details=details,
                    )
                    graph_state.error = graph_state.final_result["error"]
                    return StreamProcessingResult(chunks=chunks)
                if event.get("type") == "stream_ended_without_done":
                    stream_ended_without_done = True
                    details = event.get("details")
                    if isinstance(details, dict):
                        stream_ended_without_done_details = dict(details)
                    continue
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
                                UIEvent(
                                    event_type=UIEventType.THINKING,
                                    content=reason_field,
                                    data=_with_speaker(harness),
                                ),
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
                                                await harness._emit(
                                                    deps.event_handler,
                                                    UIEvent(
                                                        event_type=UIEventType.ASSISTANT,
                                                        content=emittable,
                                                        data=_with_speaker(harness),
                                                    ),
                                                )
                                                if echo_to_stdout:
                                                    harness._stream_print(emittable)
                                                harness._runlog("model_token", "assistant token", token=emittable)
                                            maybe_part = True
                                            break
                                    if not maybe_part:
                                        await harness._emit(
                                            deps.event_handler,
                                            UIEvent(
                                                event_type=UIEventType.ASSISTANT,
                                                content=pending,
                                                data=_with_speaker(harness),
                                            ),
                                        )
                                        if echo_to_stdout:
                                            harness._stream_print(pending)
                                        harness._runlog("model_token", "assistant token", token=pending)
                                    pending = ""
                                else:
                                    prefix = pending[:st_idx]
                                    if prefix:
                                        await harness._emit(
                                            deps.event_handler,
                                            UIEvent(
                                                event_type=UIEventType.ASSISTANT,
                                                content=prefix,
                                                data=_with_speaker(harness),
                                            ),
                                        )
                                        if echo_to_stdout:
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
                                                    await harness._emit(
                                                        deps.event_handler,
                                                        UIEvent(
                                                            event_type=UIEventType.THINKING,
                                                            content=emittable,
                                                            data=_with_speaker(harness),
                                                        ),
                                                    )
                                                harness._runlog("model_token", "thinking token", token=emittable)
                                            maybe_part = True
                                            break
                                    if not maybe_part:
                                        if harness.thinking_visibility:
                                            await harness._emit(
                                                deps.event_handler,
                                                UIEvent(
                                                    event_type=UIEventType.THINKING,
                                                    content=pending,
                                                    data=_with_speaker(harness),
                                                ),
                                            )
                                        harness._runlog("model_token", "thinking token", token=pending)
                                    pending = ""
                                else:
                                    thought = pending[:et_idx]
                                    if thought:
                                        if harness.thinking_visibility:
                                            await harness._emit(
                                                deps.event_handler,
                                                UIEvent(
                                                    event_type=UIEventType.THINKING,
                                                    content=thought,
                                                    data=_with_speaker(harness),
                                                ),
                                            )
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
                            await harness._emit(
                                deps.event_handler,
                                UIEvent(event_type=kind, content=buffer, data=_with_speaker(harness)),
                            )
                        harness._runlog("model_token", "thinking token", token=buffer)
                    else:
                        await harness._emit(
                            deps.event_handler,
                            UIEvent(event_type=kind, content=buffer, data=_with_speaker(harness)),
                        )
                        if echo_to_stdout:
                            harness._stream_print(buffer)
                        harness._runlog("model_token", "assistant token", token=buffer)
                
                # Inner loop completed cleanly (no chunk_error).
                _stream_completed_cleanly = True
                break

            # chunk_error caused the inner-loop break; wait then retry
            # Skip sleep + retry if an early 4b fallback was triggered.
            if _trigger_early_4b_fallback:
                break  # fall through to the post-loop fallback block
            if _retry_immediately:
                continue
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
                    data={"is_api_error": is_api},
                ),
            )
            err_msg = str(exc) or type(exc).__name__
            graph_state.final_result = harness._failure(err_msg, error_type=error_type, details=details)
            graph_state.error = graph_state.final_result["error"]
            return StreamProcessingResult(chunks=chunks)
    # ── Post-loop fallback / exhaustion handler ──────────────────────────────
    # NOTE: We do NOT use `for…else` here because `else` only fires when the
    # loop exits without a `break`.  Our early-4b-fallback break would skip it.
    # Instead we check an explicit flag: enter this block whenever either
    #   (a) retries were exhausted normally (loop ran to completion), or
    #   (b) _trigger_early_4b_fallback caused an outer break.
    # If the stream completed cleanly, fall through to the happy-path below.
    _enter_fallback_block = (
        not _stream_completed_cleanly
        and (
            _trigger_early_4b_fallback
            or (
                last_chunk_error_details is not None
                and last_chunk_error_details.get("reason") == "tool_call_continuation_timeout"
                and salvage_partial_stream is not None
            )
        )
    )
    if _enter_fallback_block and salvage_partial_stream is not None:
        # Exhausted all chunk_error retries (or early-exit for ≤4b models)
        if (
            last_chunk_error_details
            and last_chunk_error_details.get("reason") == "tool_call_continuation_timeout"
        ):
            session = _active_text_write_fallback_session(harness)
            partial_tool_call_names = {
                str(
                    (tool_call.get("function") or {}).get("name")
                    if isinstance(tool_call, dict)
                    else ""
                ).strip()
                for tool_call in salvage_partial_stream.tool_calls
                if isinstance(tool_call, dict)
            }
            # Standard gate: active write session with replace/patch intent.
            _standard_fallback = (
                session is not None
                and str(getattr(session, "write_session_intent", "")).strip().lower() in {"replace_file", "patch_existing"}
                and any(name in {"file_write", "file_append"} for name in partial_tool_call_names)
            )
            # Extended gate: ≤4b model that timed out on any file_write/file_append,
            # even without an active write session (fast-path from first chunk error).
            _sub4b_fallback = (
                _trigger_early_4b_fallback
                and any(name in {"file_write", "file_append"} for name in partial_tool_call_names)
            )
            should_attempt_text_fallback = _standard_fallback or _sub4b_fallback
            if should_attempt_text_fallback:
                fallback_result = await _attempt_text_write_fallback(
                    graph_state,
                    deps,
                    messages=messages,
                    source_chunks=chunks,
                    partial_tool_calls=salvage_partial_stream.tool_calls,
                    session=session,
                    reason="tool_call_continuation_timeout",
                    start_time=start_time,
                    first_token_time=first_token_time,
                )
                if fallback_result is not None:
                    return fallback_result
        harness._runlog(
            "stream_chunk_error_recovered",
            "salvaging partial stream after tool call continuation timeout",
            details=last_chunk_error_details,
        )
        if echo_to_stdout and harness.thinking_visibility:
            print()
        timeline = OpenAICompatClient.collect_timeline(
            chunks,
            reasoning_mode=harness.reasoning_mode,
            thinking_start_tag=harness.thinking_start_tag,
            thinking_end_tag=harness.thinking_end_tag,
        )
        usage_payload = salvage_partial_stream.usage
        if not isinstance(usage_payload, dict):
            usage_payload = {}
        end_time = time.perf_counter()
        duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else duration
        return StreamProcessingResult(
            chunks=chunks,
            stream=salvage_partial_stream,
            timeline=timeline,
            usage=usage_payload,
            duration=duration,
            ttft=ttft,
            halted=stream_ended_without_done,
            halt_reason="stream_ended_without_done" if stream_ended_without_done else "",
            halt_details=stream_ended_without_done_details,
        )
    if _stream_completed_cleanly:
        # ── Happy path: clean stream, no errors ──────────────────────────────
        if echo_to_stdout and harness.thinking_visibility:
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
        write_calls_with_empty_payload = [
            pending
            for pending in (PendingToolCall.from_payload(tool_call) for tool_call in stream.tool_calls)
            if pending is not None
            and pending.tool_name in {"file_write", "file_append"}
            and _detect_empty_file_write_payload(harness, pending) is not None
        ]
        if write_calls_with_empty_payload and _should_attempt_empty_payload_text_fallback(
            harness,
            graph_state,
            messages=messages,
            tool_calls=stream.tool_calls,
        ):
            fallback_result = await _attempt_text_write_fallback(
                graph_state,
                deps,
                messages=messages,
                source_chunks=chunks,
                partial_tool_calls=stream.tool_calls,
                session=_active_text_write_fallback_session(harness),
                reason="empty_payload",
                start_time=start_time,
                first_token_time=first_token_time,
            )
            if fallback_result is not None:
                return fallback_result
        if echo_to_stdout and not harness.thinking_visibility and stream.assistant_text:
            print(stream.assistant_text)
        usage_payload = stream.usage
        if not isinstance(usage_payload, dict):
            usage_payload = {}
        end_time = time.perf_counter()
        duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else duration
        return StreamProcessingResult(
            chunks=chunks,
            stream=stream,
            timeline=timeline,
            usage=usage_payload,
            duration=duration,
            ttft=ttft,
            halted=stream_ended_without_done,
            halt_reason="stream_ended_without_done" if stream_ended_without_done else "",
            halt_details=stream_ended_without_done_details,
        )
    else:
        # ── All retries exhausted without a salvageable partial stream ────────
        harness._runlog("stream_chunk_error_exhausted", "all chunk error retries exhausted")
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content="Stream error: upstream chunk errors exhausted all retries"),
        )
        graph_state.final_result = harness._failure("Upstream chunk error after retries", error_type="stream")
        graph_state.error = graph_state.final_result["error"]
        return StreamProcessingResult(chunks=chunks)
