from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

from ..client import format_tool_call_text, maybe_parse_tool_args, OpenAICompatClient, StreamResult
from ..task_targets import primary_task_target_path
from .tool_call_parser import _ensure_chunk_write_session
from .write_recovery import (
    build_synthetic_file_write_call,
    recover_content_from_assistant_text,
)

_CONTEXT_WINDOW_OVERFLOW_RE = re.compile(
    r"n_keep\s*:\s*(\d+)\s*>=\s*n_ctx\s*:\s*(\d+)",
    re.IGNORECASE,
)
_PROVIDER_HTTP_STATUS_CODES = {429, 500, 502, 503, 504, 530}


def _classify_model_call_error(exc: Exception) -> tuple[str, dict[str, Any]]:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    details: dict[str, Any] = {}
    if isinstance(status_code, int):
        details["status_code"] = status_code
        if status_code in _PROVIDER_HTTP_STATUS_CODES:
            details["retryable"] = True
        try:
            body = str(getattr(response, "text", "") or "").strip()
        except Exception:
            body = ""
        if body:
            details["body"] = body[:1000]
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


def _fallback_task_text(harness: Any, messages: list[dict[str, Any]]) -> str:
    original_task = str(
        getattr(getattr(harness, "state", None), "run_brief", SimpleNamespace(original_task="")).original_task or ""
    ).strip()
    if original_task:
        return original_task
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if role == "user" and content and not _looks_like_harness_recovery_message(content):
            return content
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content") or "").strip()
        if content and not _looks_like_harness_recovery_message(content):
            return content
    return ""


def _looks_like_harness_recovery_message(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(
        marker in lowered
        for marker in (
            "please regenerate the full tool call from scratch",
            "fallback activated for a complex write task",
            "assistant response was interrupted before a tool call finished streaming",
        )
    )


def _seed_text_write_fallback_session(
    harness: Any,
    *,
    session: Any | None,
    partial_tool_calls: list[dict[str, Any]],
) -> Any:
    fallback_session = session or getattr(getattr(harness, "state", None), "write_session", None)
    target_path = primary_task_target_path(harness)
    if fallback_session is None and partial_tool_calls:
        if target_path:
            fallback_session = _ensure_chunk_write_session(harness, target_path)
    if fallback_session is None and target_path:
        fallback_session = SimpleNamespace(
            write_session_id="",
            write_target_path=target_path,
            write_current_section="complete_file",
            write_next_section="",
            write_session_intent="patch_existing",
            suggested_sections=[],
        )
    return fallback_session


def _should_attempt_empty_payload_text_fallback(
    harness: Any,
    *,
    messages: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]] | None,
) -> bool:
    tool_names = _collect_tool_call_names(tool_calls)
    if not tool_names:
        return False
    if not any(name in {"file_write", "file_append"} for name in tool_names):
        return False
    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is not None:
        return True
    task_text = _fallback_task_text(harness, messages)
    return bool(task_text)


def _collect_tool_call_names(tool_calls: list[dict[str, Any]] | None) -> list[str]:
    names: list[str] = []
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if name:
            names.append(name)
    return names


def _extract_code_from_fallback_response(
    response_text: str,
    *,
    target_path: str,
    path_confidence: str,
) -> str:
    recovered = recover_content_from_assistant_text(
        response_text,
        target_path=target_path,
        allow_raw_text_targets=True,
        path_confidence=path_confidence,
    )
    return str(recovered or "").strip()


def _fallback_response_ready_for_early_exit(
    assistant_text: str,
    *,
    target_path: str,
    path_confidence: str,
) -> bool:
    code = _extract_code_from_fallback_response(
        assistant_text,
        target_path=target_path,
        path_confidence=path_confidence,
    )
    return bool(code)


def _build_synthetic_write_tool_call(fallback_intent: Any) -> dict[str, Any]:
    return build_synthetic_file_write_call(fallback_intent)
