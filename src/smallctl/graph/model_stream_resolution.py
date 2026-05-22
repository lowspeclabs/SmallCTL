from __future__ import annotations

import time
from typing import Any

from ..client import OpenAICompatClient
from ..client.chunk_parser import chunk_contains_tool_call_delta
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import json_safe_value
from .model_stream_fallback import (
    StreamProcessingResult,
    _attempt_remote_write_fallback,
    _attempt_text_write_fallback,
)
from .model_stream_fallback_support import (
    _should_attempt_empty_payload_text_fallback,
)
from .model_stream_fallback_recovery import (
    _active_text_write_fallback_session,
)
from .state import GraphRunState, PendingToolCall
from .deps import GraphRuntimeDeps
from .tool_call_parser import _detect_empty_file_write_payload

_STREAM_CHUNK_ERROR_AUTO_RESUME_SIGNATURE = "_stream_chunk_error_auto_resume_signature"


def _provider_chunk_data(item: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    if item.get("type") == "chunk":
        data = item.get("data", {})
        return data if isinstance(data, dict) else None
    if isinstance(item.get("choices"), list):
        return item
    return None


def _tool_call_stream_boundary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    saw_delta = False
    saw_finish = False
    finish_reasons: list[str] = []
    chunk_count = 0
    for item in chunks:
        data = _provider_chunk_data(item)
        if data is None:
            continue
        chunk_count += 1
        saw_delta = saw_delta or chunk_contains_tool_call_delta(data)
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        finish_reason = choices[0].get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            finish_reasons.append(finish_reason)
            if finish_reason == "tool_calls":
                saw_finish = True
    return {
        "saw_tool_call_delta": saw_delta,
        "saw_tool_calls_finish": saw_finish,
        "finish_reasons": finish_reasons,
        "chunk_count": chunk_count,
    }


def _chunk_error_failure_message(details: dict[str, Any] | None) -> str:
    details = details if isinstance(details, dict) else {}
    if details.get("reason") == "model_unloaded" or details.get("type") == "model_unloaded":
        provider = str(details.get("provider_profile") or "provider").strip() or "provider"
        model = str(details.get("model") or "").strip()
        suffix = f" for {model}" if model else ""
        return f"{provider} model is unloaded{suffix}"
    if details.get("reason") == "openrouter_authentication_failed":
        provider_error = str(details.get("provider_error") or "").strip()
        suffix = f" ({provider_error})" if provider_error else ""
        return (
            "OpenRouter authentication failed: API key is invalid, revoked, "
            f"or belongs to a missing account{suffix}. Update SMALLCTL_API_KEY."
        )
    if details.get("type") == "context_budget_exceeded":
        provider = str(details.get("provider_profile") or "provider").strip() or "provider"
        over_budget = int(details.get("over_budget_tokens", 0) or 0)
        suffix = f" by {over_budget} estimated tokens" if over_budget > 0 else ""
        return f"{provider} prompt exceeded the local context budget before request{suffix}"
    if int(details.get("status_code", 0) or 0) == 400:
        provider = str(details.get("provider_profile") or "provider").strip() or "provider"
        upstream = str(details.get("upstream_provider") or "").strip()
        provider_error = str(details.get("provider_error") or "").strip()
        label = provider
        if upstream and upstream.lower() != provider.lower():
            label = f"{provider}/{upstream}"
        suffix = f": {provider_error}" if provider_error else ""
        return f"{label} input validation failed after retries (HTTP 400{suffix})"
    return "Upstream chunk error after retries"


def _chunk_error_failure_type(details: dict[str, Any] | None) -> str:
    details = details if isinstance(details, dict) else {}
    if details.get("reason") == "model_unloaded" or details.get("type") == "model_unloaded":
        return "provider"
    if details.get("reason") == "openrouter_authentication_failed":
        return "provider"
    if details.get("type") == "context_budget_exceeded":
        return "prompt_budget"
    if int(details.get("status_code", 0) or 0) == 400:
        return "provider"
    return "stream"


def _clear_stream_chunk_error_auto_resume_signature(harness: Any) -> None:
    harness.state.scratchpad.pop(_STREAM_CHUNK_ERROR_AUTO_RESUME_SIGNATURE, None)


def _write_session_auto_resume_signature(session: Any) -> str:
    return "|".join(
        [
            str(getattr(session, "write_session_id", "") or "").strip(),
            str(getattr(session, "write_current_section", "") or "").strip(),
            str(getattr(session, "write_next_section", "") or "").strip(),
            str(getattr(session, "status", "") or "").strip(),
        ]
    )


def _recoverable_active_write_session(harness: Any) -> Any | None:
    session = getattr(harness.state, "write_session", None)
    if session is None:
        return None
    if str(getattr(session, "status", "") or "").strip().lower() == "complete":
        return None
    if not str(getattr(session, "write_session_id", "") or "").strip():
        return None
    if not str(getattr(session, "write_target_path", "") or "").strip():
        return None
    return session


async def resolve_model_stream_result(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    harness: Any,
    chunks: list[dict[str, Any]],
    salvage_partial_stream: Any,
    last_chunk_error_details: dict[str, Any] | None,
    stream_ended_without_done: bool,
    stream_ended_without_done_details: dict[str, Any],
    trigger_early_4b_fallback: bool,
    stream_completed_cleanly: bool,
    echo_to_stdout: bool,
    messages: list[dict[str, Any]],
    start_time: float,
    first_token_time: float | None,
) -> StreamProcessingResult | None:
    if graph_state.final_result is not None:
        return StreamProcessingResult(chunks=chunks)

    if (
        stream_ended_without_done
        and stream_ended_without_done_details.get("reason") == "reasoning_only_stream_stall"
    ):
        failure_message = (
            "Model stream halted after repeated reasoning-only output with no assistant content "
            "or tool call"
        )
        harness._runlog(
            "reasoning_only_stream_exhausted",
            "model stream halted after repeated reasoning-only output",
            details=stream_ended_without_done_details,
            error_type="model_stream_stall",
        )
        await harness._emit(
            deps.event_handler,
            UIEvent(event_type=UIEventType.ERROR, content=f"Stream error: {failure_message}"),
        )
        graph_state.final_result = harness._failure(
            failure_message,
            error_type="model_stream_stall",
            details=stream_ended_without_done_details,
        )
        graph_state.error = graph_state.final_result["error"]
        return StreamProcessingResult(chunks=chunks)

    enter_fallback_block = (
        not stream_completed_cleanly
        and (
            trigger_early_4b_fallback
            or (
                last_chunk_error_details is not None
                and last_chunk_error_details.get("reason") == "tool_call_continuation_timeout"
                and salvage_partial_stream is not None
            )
        )
    )
    if enter_fallback_block and salvage_partial_stream is not None:
        _clear_stream_chunk_error_auto_resume_signature(harness)
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
            has_stalled_write_call = any(name in {"file_write", "file_append"} for name in partial_tool_call_names)
            has_stalled_remote_write_call = "ssh_file_write" in partial_tool_call_names
            standard_fallback = (
                session is not None
                and str(getattr(session, "write_session_intent", "")).strip().lower() in {"replace_file", "patch_existing"}
                and has_stalled_write_call
            )
            sub4b_fallback = (
                trigger_early_4b_fallback
                and has_stalled_write_call
            )
            stalled_write_fallback = (
                has_stalled_write_call
                and _should_attempt_empty_payload_text_fallback(
                    harness,
                    messages=messages,
                    tool_calls=salvage_partial_stream.tool_calls,
                )
            )
            should_attempt_text_fallback = standard_fallback or sub4b_fallback or stalled_write_fallback
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
            if has_stalled_remote_write_call:
                remote_fallback_result = await _attempt_remote_write_fallback(
                    graph_state,
                    deps,
                    messages=messages,
                    source_chunks=chunks,
                    partial_tool_calls=salvage_partial_stream.tool_calls,
                    reason="tool_call_continuation_timeout",
                    start_time=start_time,
                    first_token_time=first_token_time,
                )
                if remote_fallback_result is not None:
                    return remote_fallback_result
        incomplete_payload = harness.state.scratchpad.get("_last_incomplete_tool_call")
        tool_call_diagnostics = []
        if isinstance(incomplete_payload, dict):
            raw_diagnostics = incomplete_payload.get("tool_call_diagnostics")
            if isinstance(raw_diagnostics, list):
                tool_call_diagnostics = raw_diagnostics
        harness._runlog(
            "stream_chunk_error_recovered",
            "salvaging partial stream after tool call continuation timeout",
            details=last_chunk_error_details,
            tool_call_diagnostics=tool_call_diagnostics,
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

    if stream_completed_cleanly:
        _clear_stream_chunk_error_auto_resume_signature(harness)
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
        tool_call_boundary = _tool_call_stream_boundary(chunks)
        observed_tool_call_boundary = (
            bool(tool_call_boundary.get("saw_tool_call_delta"))
            or bool(tool_call_boundary.get("saw_tool_calls_finish"))
        )
        if observed_tool_call_boundary and not stream.tool_calls:
            details = {
                "reason": "tool_call_aggregation_failure",
                **tool_call_boundary,
            }
            harness.state.scratchpad["_last_tool_call_aggregation_failure"] = details
            harness._runlog(
                "tool_call_aggregation_failure",
                "stream advertised tool calls but collector produced none",
                details=details,
            )
            harness.state.append_message(
                ConversationMessage(
                    role="user",
                    content=(
                        "[HARNESS NOTICE]: The prior model stream ended with tool_calls metadata, "
                        "but the harness could not reconstruct any callable tool payloads. Retry the "
                        "tool call now using the available structured tool interface. Do not answer "
                        "with prose unless no tool is needed."
                    ),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "tool_call_aggregation_failure",
                        "details": json_safe_value(details),
                    },
                )
            )
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content="Tool-call stream could not be reconstructed; retrying with a structured tool nudge.",
                    data=json_safe_value(details),
                ),
            )
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
                halted=True,
                halt_reason="tool_call_aggregation_failure",
                halt_details=details,
            )
        write_calls_with_empty_payload = [
            pending
            for pending in (PendingToolCall.from_payload(tool_call) for tool_call in stream.tool_calls)
            if pending is not None
            and pending.tool_name in {"file_write", "file_append"}
            and _detect_empty_file_write_payload(harness, pending) is not None
        ]
        if write_calls_with_empty_payload:
            try:
                should_fallback = _should_attempt_empty_payload_text_fallback(
                    harness,
                    messages=messages,
                    tool_calls=stream.tool_calls,
                )
            except Exception as exc:
                harness._runlog(
                    "empty_payload_fallback_predicate_failed",
                    "empty-payload fallback predicate raised an exception; continuing with normal stream resolution",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                should_fallback = False
            if should_fallback:
                try:
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
                except Exception as exc:
                    harness._runlog(
                        "empty_payload_fallback_failed",
                        "text-write fallback raised an exception; continuing with normal stream resolution",
                        error=str(exc),
                        error_type=type(exc).__name__,
                    )
                    fallback_result = None
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

    recoverable_session = _recoverable_active_write_session(harness)
    if recoverable_session is not None:
        signature = _write_session_auto_resume_signature(recoverable_session)
        last_signature = str(harness.state.scratchpad.get(_STREAM_CHUNK_ERROR_AUTO_RESUME_SIGNATURE) or "")
        if signature and signature != last_signature:
            harness.state.scratchpad[_STREAM_CHUNK_ERROR_AUTO_RESUME_SIGNATURE] = signature
            harness._runlog(
                "stream_chunk_error_auto_resume_scheduled",
                "scheduling internal auto-resume for recoverable write session",
                session_id=str(getattr(recoverable_session, "write_session_id", "") or ""),
                target_path=str(getattr(recoverable_session, "write_target_path", "") or ""),
            )
            harness.state.append_message(
                ConversationMessage(
                    role="system",
                    content=(
                        "Stream retries exhausted, but an active Write Session looks recoverable. "
                        f"Auto-resuming once for Write Session `{recoverable_session.write_session_id}` on "
                        f"`{recoverable_session.write_target_path}`. Continue from current staged progress "
                        "instead of replaying prior chunks."
                    ),
                    metadata={
                        "is_recovery_nudge": True,
                        "recovery_kind": "stream_chunk_error_auto_resume",
                        "session_id": str(getattr(recoverable_session, "write_session_id", "") or ""),
                        "target_path": str(getattr(recoverable_session, "write_target_path", "") or ""),
                    },
                )
            )
            await harness._emit(
                deps.event_handler,
                UIEvent(
                    event_type=UIEventType.ALERT,
                    content="Transient stream failure detected. Auto-resuming write session once.",
                    data={
                        "status_activity": "auto-resume scheduled",
                        "write_session_id": str(getattr(recoverable_session, "write_session_id", "") or ""),
                    },
                ),
            )
            empty_stream = OpenAICompatClient.collect_stream(
                [],
                reasoning_mode=harness.reasoning_mode,
                thinking_start_tag=harness.thinking_start_tag,
                thinking_end_tag=harness.thinking_end_tag,
            )
            end_time = time.perf_counter()
            duration = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else duration
            return StreamProcessingResult(
                chunks=chunks,
                stream=empty_stream,
                timeline=[],
                usage={},
                duration=duration,
                ttft=ttft,
                halted=True,
                halt_reason="stream_chunk_error_auto_resume",
                halt_details={
                    "reason": "stream_chunk_error_auto_resume",
                    "write_session_id": str(getattr(recoverable_session, "write_session_id", "") or ""),
                },
            )

    failure_message = _chunk_error_failure_message(last_chunk_error_details)
    failure_type = _chunk_error_failure_type(last_chunk_error_details)
    harness._runlog(
        "stream_chunk_error_exhausted",
        "all chunk error retries exhausted",
        details=last_chunk_error_details,
        error_type=failure_type,
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.ERROR, content=f"Stream error: {failure_message}"),
    )
    graph_state.final_result = harness._failure(
        failure_message,
        error_type=failure_type,
        details=last_chunk_error_details,
    )
    graph_state.error = graph_state.final_result["error"]
    return StreamProcessingResult(chunks=chunks)
