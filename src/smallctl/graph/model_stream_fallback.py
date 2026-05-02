from __future__ import annotations

import logging
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from ..client import OpenAICompatClient, StreamResult
from ..logging_utils import log_kv
from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import json_safe_value
from .deps import GraphRuntimeDeps
from .state import GraphRunState, PendingToolCall
from .tool_call_parser import (
    _detect_empty_file_write_payload,
    _suggested_chunk_sections,
)
from .model_stream_fallback_support import (
    _collect_tool_call_names,
    _classify_model_call_error,
    _extract_code_from_fallback_response,
    _fallback_response_ready_for_early_exit,
    _fallback_task_text,
    _format_partial_tool_calls,
    _parse_context_window_overflow,
    _seed_text_write_fallback_session,
    _should_attempt_empty_payload_text_fallback,
)
from .model_stream_fallback_recovery import (
    _active_text_write_fallback_session,
    _build_incomplete_tool_call_recovery_message,
    _build_text_write_fallback_prompt,
    _fallback_next_section_name,
    _fallback_section_name,
    _is_sub4b_write_timeout,
    _with_speaker,
)
from .model_stream_fallback_trace import (
    _emit_text_write_fallback_trace,
    _record_text_write_fallback_state,
)
from .write_recovery import (
    build_synthetic_file_write_call,
    can_safely_synthesize,
    infer_write_target_path,
    maybe_finalize_recovered_assistant_write,
    recover_write_intent,
    _maybe_prepend_existing_content,
    write_recovery_kind,
    write_recovery_metadata,
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


def _recover_remote_write_candidate(partial_tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    for raw in reversed(partial_tool_calls):
        pending = PendingToolCall.from_payload(raw)
        if pending is None or pending.tool_name != "ssh_file_write":
            continue
        path = str(pending.args.get("path") or "").strip()
        if not path:
            continue
        candidate = {"path": path}
        for key in (
            "target",
            "host",
            "user",
            "username",
            "port",
            "identity_file",
            "password",
            "encoding",
            "mode",
            "create_parent_dirs",
            "backup",
            "expected_sha256",
            "timeout_sec",
        ):
            value = pending.args.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            candidate[key] = value
        return candidate
    return None


def _looks_like_raw_remote_file_content(path: str, assistant_text: str) -> bool:
    text = str(assistant_text or "").strip()
    if not text:
        return False
    if any(marker in text.lower() for marker in ("<tool_call", "<function=", "```json")):
        return False
    suffix = str(path or "").strip().lower()
    return suffix.endswith((".html", ".css", ".js", ".json", ".svg", ".xml", ".txt", ".md"))


def _extract_remote_write_content(*, path: str, assistant_text: str) -> str:
    extracted = _extract_code_from_fallback_response(
        assistant_text,
        target_path=path,
        path_confidence="high" if path else "low",
    )
    if extracted:
        return extracted
    if _looks_like_raw_remote_file_content(path, assistant_text):
        return str(assistant_text or "").strip()
    return ""


def _build_synthetic_ssh_file_write_call(candidate: dict[str, Any], *, content: str, tool_call_id: str = "") -> dict[str, Any]:
    args = dict(candidate)
    args["content"] = content
    return {
        "id": tool_call_id or "ssh_write_recovery",
        "type": "function",
        "function": {
            "name": "ssh_file_write",
            "arguments": json.dumps(args, ensure_ascii=True, sort_keys=True),
        },
    }


async def _attempt_remote_write_fallback(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    messages: list[dict[str, Any]],
    source_chunks: list[dict[str, Any]],
    partial_tool_calls: list[dict[str, Any]],
    reason: str,
    start_time: float,
    first_token_time: float | None,
) -> StreamProcessingResult | None:
    harness = deps.harness
    candidate = _recover_remote_write_candidate(partial_tool_calls)
    if candidate is None:
        return None

    target_path = str(candidate.get("path") or "").strip()
    prompt_lines = [
        "You are continuing a remote file write after a tool-call stream stalled.",
        f"Remote target path: `{target_path}`." if target_path else "Remote target path: (unknown).",
        "Return only the complete file content for that remote path, with no prose and no tool JSON.",
    ]
    task_text = _fallback_task_text(harness, messages)
    if task_text:
        prompt_lines.extend(["", "User task:", task_text.strip()])
    fallback_messages = list(messages) + [{"role": "system", "content": "\n".join(prompt_lines)}]

    graph_state.latency_metrics["remote_write_fallback_attempt_count"] = (
        int(graph_state.latency_metrics.get("remote_write_fallback_attempt_count", 0) or 0) + 1
    )
    harness._runlog(
        "stream_remote_write_fallback_attempt",
        "attempting no-tools rescue for stalled remote file write",
        target_path=target_path,
        tool_name="ssh_file_write",
        reason=reason,
    )
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content="Chat-mode fallback activated for a stalled remote file write.",
            data={"status_activity": "remote fallback active", "target_path": target_path, "reason": reason},
        ),
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
            if _extract_remote_write_content(path=target_path, assistant_text=partial_stream.assistant_text):
                break
    except Exception as exc:
        harness._runlog(
            "stream_remote_write_fallback_failed",
            "remote write rescue errored; using original stream",
            error=str(exc),
            target_path=target_path,
            reason=reason,
        )
        return None

    fallback_stream = OpenAICompatClient.collect_stream(
        fallback_chunks,
        reasoning_mode=harness.reasoning_mode,
        thinking_start_tag=harness.thinking_start_tag,
        thinking_end_tag=harness.thinking_end_tag,
    )
    recovered_content = _extract_remote_write_content(path=target_path, assistant_text=fallback_stream.assistant_text)
    if not recovered_content:
        harness._runlog(
            "stream_remote_write_fallback_failed",
            "remote write rescue returned no usable file content",
            target_path=target_path,
            reason=reason,
        )
        return None

    fallback_timeline = OpenAICompatClient.collect_timeline(
        fallback_chunks,
        reasoning_mode=harness.reasoning_mode,
        thinking_start_tag=harness.thinking_start_tag,
        thinking_end_tag=harness.thinking_end_tag,
    )
    synthetic_call = _build_synthetic_ssh_file_write_call(candidate, content=recovered_content)
    harness._runlog(
        "stream_remote_write_fallback_succeeded",
        "converted no-tools rescue response into synthetic ssh_file_write",
        target_path=target_path,
        content_chars=len(recovered_content),
        reason=reason,
    )
    graph_state.latency_metrics["remote_write_fallback_success_count"] = (
        int(graph_state.latency_metrics.get("remote_write_fallback_success_count", 0) or 0) + 1
    )
    harness.state.scratchpad["_last_remote_write_fallback"] = {
        "target_path": target_path,
        "content_chars": len(recovered_content),
        "reason": reason,
    }
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
                content="Chat-mode fallback activated for a stalled write task.",
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
    fallback_intent = recover_write_intent(
        harness=harness,
        pending=None,
        assistant_text=fallback_stream.assistant_text,
        partial_tool_calls=partial_tool_calls,
    )
    if fallback_intent is not None:
        maybe_finalize_recovered_assistant_write(fallback_intent)
    resolved_next_section_name = (
        str(getattr(fallback_intent, "next_section_name", "") or "").strip()
        if fallback_intent is not None
        else next_section_name
    )
    await _emit_text_write_fallback_trace(
        harness,
        deps,
        session=session_context,
        current_section=current_section,
        prompt=fallback_prompt,
        assistant_text=fallback_stream.assistant_text,
        extracted_code=extracted_code,
        next_section_name=resolved_next_section_name,
        tool_names=tool_names,
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
        _maybe_prepend_existing_content(fallback_intent, harness=harness)
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
            next_section_name=resolved_next_section_name,
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
            next_section_name=resolved_next_section_name,
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
