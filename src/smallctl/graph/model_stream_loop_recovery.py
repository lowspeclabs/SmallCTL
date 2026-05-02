from __future__ import annotations

from typing import Any

from ..client import OpenAICompatClient, StreamResult
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..redaction import compact_tool_arguments_for_metadata
from ..state import json_safe_value
from .state import PendingToolCall
from .model_stream_fallback_recovery import (
    _build_incomplete_tool_call_recovery_message,
    _is_sub4b_write_timeout,
)
from .model_stream_fallback_support import _format_partial_tool_calls
from .model_stream_fallback_recovery import _with_speaker


def _compact_raw_arguments_preview(raw_arguments: str, *, limit: int = 240) -> str:
    text = str(raw_arguments or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _build_partial_tool_call_diagnostics(harness: Any, partial_stream: StreamResult) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    registry = getattr(harness, "registry", None)
    for raw_tool_call in partial_stream.tool_calls:
        pending = PendingToolCall.from_payload(raw_tool_call)
        if pending is None:
            continue
        required_fields: list[str] = []
        if registry is not None:
            tool_spec = registry.get(pending.tool_name)
            if tool_spec is not None:
                required_fields = [str(field).strip() for field in tool_spec.schema.get("required", []) if str(field).strip()]
        present_fields = sorted(
            key
            for key, value in dict(pending.args or {}).items()
            if value is not None and (not isinstance(value, str) or value.strip())
        )
        missing_fields = [field for field in required_fields if field not in present_fields]
        diagnostics.append(
            {
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
                "required_fields": required_fields,
                "present_fields": present_fields,
                "missing_required_fields": missing_fields,
                "arguments": json_safe_value(compact_tool_arguments_for_metadata(pending.tool_name, dict(pending.args or {}))),
                "raw_arguments_preview": _compact_raw_arguments_preview(pending.raw_arguments),
            }
        )
    return diagnostics


def _should_add_incomplete_tool_call_recovery_nudge(
    *,
    details: dict[str, Any],
    partial_stream: StreamResult,
) -> bool:
    if str(details.get("type") or "").strip() == "provider_input_validation":
        return False
    if str(details.get("reason") or "").strip() == "tool_call_continuation_timeout":
        return True
    if partial_stream.tool_calls:
        return True
    return bool(str(partial_stream.assistant_text or "").strip())


async def handle_model_stream_chunk_error(
    *,
    harness: Any,
    deps: Any,
    graph_state: Any,
    messages: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    err_msg: str,
    details: dict[str, Any],
    model_attempt: int,
    chunk_error_max_retries: int,
    timeout_recovery_nudges: int,
    trigger_early_4b_fallback: bool,
    salvage_partial_stream: StreamResult | None,
) -> dict[str, Any]:
    retrying = model_attempt < chunk_error_max_retries

    partial_stream = OpenAICompatClient.collect_stream(
        chunks,
        reasoning_mode=harness.reasoning_mode,
        thinking_start_tag=harness.thinking_start_tag,
        thinking_end_tag=harness.thinking_end_tag,
    )
    salvage_partial_stream = partial_stream
    if timeout_recovery_nudges < 2 and _should_add_incomplete_tool_call_recovery_nudge(
        details=details,
        partial_stream=partial_stream,
    ):
        partial_tool_calls = _format_partial_tool_calls(partial_stream.tool_calls)
        tool_call_diagnostics = _build_partial_tool_call_diagnostics(harness, partial_stream)
        recovery_message = _build_incomplete_tool_call_recovery_message(
            harness=harness,
            assistant_text=partial_stream.assistant_text,
            partial_tool_calls=partial_stream.tool_calls,
        )
        recovery_payload = {
            "kind": "incomplete_tool_call",
            "attempt": model_attempt + 1,
            "error": err_msg,
            "details": details,
            "assistant_text": partial_stream.assistant_text,
            "thinking_text": partial_stream.thinking_text,
            "partial_tool_calls": partial_tool_calls,
            "partial_tool_calls_raw": json_safe_value(partial_stream.tool_calls),
            "tool_call_diagnostics": tool_call_diagnostics,
            "message": recovery_message,
        }
        harness.state.scratchpad["_last_incomplete_tool_call"] = recovery_payload
        recovery_metadata = {
            "is_recovery_nudge": True,
            "recovery_kind": "incomplete_tool_call",
            "attempt": model_attempt + 1,
            "partial_tool_calls": partial_tool_calls,
            "tool_call_count": len(partial_tool_calls),
            "tool_call_diagnostics": tool_call_diagnostics,
        }
        recovery_message_obj = ConversationMessage(
            role="system",
            content=recovery_message,
            metadata=recovery_metadata,
        )
        harness.state.append_message(recovery_message_obj)
        messages.append(recovery_message_obj.to_dict())
        timeout_recovery_nudges += 1

    if not trigger_early_4b_fallback and _is_sub4b_write_timeout(
        harness,
        error_text=err_msg,
        error_details=details,
    ):
        trigger_early_4b_fallback = True
        harness.state.scratchpad["_sub4b_chat_fallback_active"] = True
        harness._runlog(
            "stream_chunk_error",
            "sub-4b write timeout: skipping retries, proceeding to chat-mode fallback",
            error=err_msg,
            attempt=model_attempt + 1,
            retrying=False,
            details=details,
        )

    harness._runlog(
        "stream_chunk_error",
        "upstream chunk error, will retry" if retrying else "upstream chunk error on final attempt",
        error=err_msg,
        attempt=model_attempt + 1,
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
                    "attempt": model_attempt + 1,
                    "details": details,
                },
            ),
        )

    return {
        "retrying": retrying,
        "timeout_recovery_nudges": timeout_recovery_nudges,
        "trigger_early_4b_fallback": trigger_early_4b_fallback,
        "salvage_partial_stream": salvage_partial_stream,
    }
