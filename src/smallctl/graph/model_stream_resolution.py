from __future__ import annotations

import json
import time
from typing import Any

MAX_DEGENERATE_LOOP_RETRIES = 3

from ..client import OpenAICompatClient
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
from .model_stream_resolution_support import (
    _chunk_error_failure_message,
    _chunk_error_failure_type,
    _tool_call_stream_boundary,
)
from .state import GraphRunState, PendingToolCall
from .deps import GraphRuntimeDeps
from .tool_call_parser import _detect_empty_file_write_payload
from .write_recovery import (
    build_synthetic_file_write_call,
    can_safely_synthesize,
    recover_write_intent,
)

_STREAM_CHUNK_ERROR_AUTO_RESUME_SIGNATURE = "_stream_chunk_error_auto_resume_signature"
_EMPTY_WRITE_FAILURE_COUNT_KEY = "_empty_write_failure_count"
_EMPTY_WRITE_FAILURE_PATH_KEY = "_empty_write_failure_last_path"
_EMPTY_WRITE_STRATEGY_SWITCHED_KEY = "_empty_write_strategy_switched"


def _track_empty_write_failure(harness: Any, path: str) -> int:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {}) or {}
    last_path = scratchpad.get(_EMPTY_WRITE_FAILURE_PATH_KEY, "")
    if last_path != path:
        scratchpad[_EMPTY_WRITE_FAILURE_COUNT_KEY] = 1
        scratchpad[_EMPTY_WRITE_FAILURE_PATH_KEY] = path
    else:
        scratchpad[_EMPTY_WRITE_FAILURE_COUNT_KEY] = scratchpad.get(_EMPTY_WRITE_FAILURE_COUNT_KEY, 0) + 1
    return int(scratchpad[_EMPTY_WRITE_FAILURE_COUNT_KEY])


def _reset_empty_write_failure(harness: Any) -> None:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {}) or {}
    scratchpad.pop(_EMPTY_WRITE_FAILURE_COUNT_KEY, None)
    scratchpad.pop(_EMPTY_WRITE_FAILURE_PATH_KEY, None)


def _try_synthesize_from_stream_text(
    harness: Any,
    stream: Any,
    partial_tool_calls: list[dict[str, Any]],
) -> StreamProcessingResult | None:
    """Fix D: If the model streamed substantive assistant text before an empty tool call,
    try to recover a file_write directly from that text without an expensive fallback chat."""
    assistant_text = str(getattr(stream, "assistant_text", "") or "").strip()
    if len(assistant_text) < 200:
        return None
    intent = recover_write_intent(
        harness=harness,
        pending=None,
        assistant_text=assistant_text,
        partial_tool_calls=partial_tool_calls,
    )
    if intent is None or not can_safely_synthesize(intent, harness=harness):
        return None
    synthetic_call = build_synthetic_file_write_call(intent)
    stream.tool_calls.append(synthetic_call)
    stream.assistant_text = (
        "[Recovered file_write from streamed assistant text after empty payload.]"
    )
    harness._runlog(
        "stream_text_synthesized_from_assistant_text",
        "synthesized file_write directly from streamed assistant text",
        target_path=intent.path,
        content_chars=len(intent.content),
        confidence=intent.confidence,
    )
    return StreamProcessingResult(
        chunks=[],
        stream=stream,
        timeline=[],
        usage=getattr(stream, "usage", {}) or {},
        duration=0.0,
        ttft=0.0,
        halted=False,
        halt_reason="",
        halt_details={},
    )


def _maybe_inject_strategy_switch(harness: Any, path: str, count: int) -> None:
    """Fix E: After 3 empty write failures, suggest shell_exec or subtask approach."""
    if count < 3:
        return
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {}) or {}
    if scratchpad.get(_EMPTY_WRITE_STRATEGY_SWITCHED_KEY):
        return
    scratchpad[_EMPTY_WRITE_STRATEGY_SWITCHED_KEY] = True
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Empty write attempts for `{path}` have failed {count} times. "
                "Switch strategy: use `shell_exec` with a heredoc to write the file, "
                "or break the implementation into smaller sub-tasks (e.g. scaffold HTML structure first, "
                "then add CSS, then add JS). Do not retry another empty `file_write`."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "empty_write_strategy_switch",
                "target_path": path,
                "failure_count": count,
            },
        )
    )
    harness._runlog(
        "empty_write_strategy_switch",
        "injected strategy switch after repeated empty write failures",
        target_path=path,
        failure_count=count,
    )


def _maybe_trigger_escalation_for_empty_writes(harness: Any, path: str, count: int) -> bool:
    """Fix C: After 2 empty write failures, auto-escalate if enabled."""
    if count < 2:
        return False
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return False
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {}) or {}
    seen = scratchpad.get("_escalation_auto_empty_write_fingerprints")
    if not isinstance(seen, list):
        seen = []
    fingerprint = f"empty_write:{path}"
    if fingerprint in seen:
        return False
    seen.append(fingerprint)
    scratchpad["_escalation_auto_empty_write_fingerprints"] = seen[-20:]
    harness.state.scratchpad["_tool_loop_suppression"] = {
        "tool_name": "file_write",
        "error": f"Empty file_write payload failed {count} times for {path}.",
    }
    # Run escalation asynchronously; caller must await
    return True


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
    partial_assistant_text: str = "",
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

    if (
        stream_ended_without_done
        and stream_ended_without_done_details.get("reason") == "model_output_degenerate_loop"
    ):
        harness._runlog(
            "model_output_degenerate_loop_exhausted",
            "model stream halted after degenerate repetition loop",
            details=stream_ended_without_done_details,
            error_type="model_output_degenerate_loop",
        )
        await harness._emit(
            deps.event_handler,
            UIEvent(
                event_type=UIEventType.ALERT,
                content="Model output degenerated into a repetition loop. Recovery nudge injected.",
                data=stream_ended_without_done_details,
            ),
        )
        repeated_phrase = str(stream_ended_without_done_details.get("repeated_phrase") or "")
        # Fix for RCA 8ec35471: after a degenerate loop, re-anchor the model with
        # the current user goal and the available tools instead of only asking it
        # to stop repeating. This reduces the chance that the model emits the
        # repeated phrase again and makes the recovery nudge actionable.
        # Fix for session e9020e29: track consecutive degenerate loops; after
        # the threshold is reached, switch reasoning_mode to "off" so the model
        # is no longer instructed or expected to emit thinking/response tags.
        scratchpad = getattr(harness.state, "scratchpad", {})
        if isinstance(scratchpad, dict):
            degenerate_count = int(scratchpad.get("_consecutive_degenerate_loops", 0)) + 1
            scratchpad["_consecutive_degenerate_loops"] = degenerate_count
        else:
            degenerate_count = 1
        reasoning_mode_switched = False
        if degenerate_count >= MAX_DEGENERATE_LOOP_RETRIES:
            old_reasoning_mode = getattr(harness, "reasoning_mode", "tags")
            if old_reasoning_mode != "off":
                harness.reasoning_mode = "off"
                if isinstance(scratchpad, dict):
                    scratchpad["_thinking_tags_disabled"] = True
                reasoning_mode_switched = True
                harness._runlog(
                    "degenerate_loop_reasoning_mode_disabled",
                    "switched reasoning_mode to off after consecutive degenerate loops",
                    consecutive_loops=degenerate_count,
                    old_mode=old_reasoning_mode,
                )
        recovery_content_parts = [
            "Your previous response degenerated into a loop.",
        ]
        if reasoning_mode_switched:
            recovery_content_parts.append(
                "Thinking markers have been disabled. Do NOT start your response with a "
                "`<think>` or `<thinking>` block, and do not emit `<response>`, `<|channel>`, "
                "`<channel|>`, `<thought>`, or any other angle-bracket control tag. "
                "Respond directly and emit the next tool call as a JSON object."
            )
        else:
            recovery_content_parts.append(
                "Do not emit `<think>`, `<thinking>`, `<response>`, `<|channel>`, `<channel|>`, "
                "`<thought>`, or any other angle-bracket control tag in your next response."
            )
        recovery_content_parts.append("Stop repeating and emit ONE concrete next action as a tool call.")
        user_task = ""
        run_brief = getattr(getattr(harness, "state", None), "run_brief", None)
        if run_brief is not None:
            user_task = str(getattr(run_brief, "original_task", "") or "").strip()
        if not user_task:
            state = getattr(harness, "state", None)
            user_task = str(getattr(state, "current_task", "") or "").strip()
        if user_task:
            recovery_content_parts.append(f"Current goal: {user_task}")
        try:
            # Show the model the exact tools that will be exposed on the next
            # request. Using the exported registry list (current phase, mode and
            # active profiles) keeps the recovery nudge consistent with the
            # actual request payload; chat_mode_tools can over-narrow the list
            # and mislead the model about available capabilities such as ssh_exec.
            exported_tools = harness.registry.export_openai_tools(
                phase=harness.state.current_phase,
                mode="chat",
                profiles=set(harness.state.active_tool_profiles),
            )
            allowed_names = [
                str(entry["function"]["name"])
                for entry in exported_tools
                if isinstance(entry, dict)
                and isinstance(entry.get("function"), dict)
                and "name" in entry["function"]
            ]
        except Exception:
            allowed_names = []
        if not allowed_names:
            try:
                from ..harness.tool_dispatch import chat_mode_tools
                allowed_tools = chat_mode_tools(harness)
                allowed_names = [
                    str(entry["function"]["name"])
                    for entry in allowed_tools
                    if isinstance(entry, dict)
                    and isinstance(entry.get("function"), dict)
                    and "name" in entry["function"]
                ]
            except Exception:
                allowed_names = []
        if allowed_names:
            shown = ", ".join(allowed_names[:8])
            recovery_content_parts.append(f"Available tools now: {shown}")
        else:
            recovery_content_parts.append("If you do not know what to do next, ask a focused clarification or call task_fail.")
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=" ".join(recovery_content_parts),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "model_output_degenerate_loop",
                    "repeated_phrase": repeated_phrase,
                },
            )
        )
        stream = OpenAICompatClient.collect_stream(
            chunks,
            reasoning_mode=harness.reasoning_mode,
            thinking_start_tag=harness.thinking_start_tag,
            thinking_end_tag=harness.thinking_end_tag,
        )
        # If the stream degenerated while the model was emitting an inline tool
        # call, the text collected from chunks may be truncated or empty because
        # the repetition guard halted mid-token. Append any salvaged prefix so
        # parse_tool_calls has a chance to recover the partial tool call.
        partial_text = str(stream_ended_without_done_details.get("partial_assistant_text") or partial_assistant_text or "").strip()
        if partial_text:
            existing_text = str(getattr(stream, "assistant_text", "") or "")
            stream.assistant_text = (existing_text + "\n" + partial_text).strip()
        end_time = time.perf_counter()
        duration = end_time - start_time
        ttft = (first_token_time - start_time) if first_token_time else duration
        return StreamProcessingResult(
            chunks=chunks,
            stream=stream,
            timeline=[],
            usage=getattr(stream, "usage", {}) or {},
            duration=duration,
            ttft=ttft,
            halted=True,
            halt_reason="model_output_degenerate_loop",
            halt_details=stream_ended_without_done_details,
        )

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
            # Determine the target path for failure tracking
            empty_path = ""
            for pending in write_calls_with_empty_payload:
                path_val = str(pending.args.get("path") or "").strip()
                if path_val:
                    empty_path = path_val
                    break
            if not empty_path:
                from ..task_targets import primary_task_target_path
                empty_path = primary_task_target_path(harness) or ""
            failure_count = _track_empty_write_failure(harness, empty_path)

            # Fix D: If the model streamed substantive assistant text, try to synthesize
            # a file_write directly from it before falling back to an expensive new chat.
            synthesized = _try_synthesize_from_stream_text(
                harness, stream, partial_tool_calls=stream.tool_calls
            )
            if synthesized is not None:
                _reset_empty_write_failure(harness)
                return synthesized

            # Fix C: Escalate after 2 consecutive empty write failures
            if failure_count >= 2:
                if _maybe_trigger_escalation_for_empty_writes(harness, empty_path, failure_count):
                    from ..harness.escalation_service import EscalationService
                    try:
                        result = await EscalationService(harness).run(
                            reason=f"Empty file_write payload failed {failure_count} times for `{empty_path}`.",
                            question="What is the smallest safe next evidence-gathering or repair step?",
                            requested_output="next_action",
                            risk_level="medium",
                            source="auto",
                        )
                        if bool(result.get("success")):
                            harness.state.append_message(
                                ConversationMessage(
                                    role="system",
                                    content=(
                                        "Escalation advisor returned bounded recovery advice for repeated empty writes. "
                                        "Treat this as advice only; choose any next action through normal tool policy.\n"
                                        f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
                                    ),
                                    metadata={
                                        "is_recovery_nudge": True,
                                        "recovery_kind": "escalation_advisory",
                                        "source": "auto_empty_write",
                                        "target_path": empty_path,
                                        "escalation_id": result.get("escalation_id"),
                                    },
                                )
                            )
                            harness._runlog(
                                "escalation_auto_empty_write_advisory",
                                "injected escalation advisory after repeated empty write failures",
                                target_path=empty_path,
                                escalation_id=result.get("escalation_id"),
                            )
                    except Exception as exc:
                        harness._runlog(
                            "escalation_auto_empty_write_failed",
                            "escalation service failed for empty writes",
                            error=str(exc),
                            target_path=empty_path,
                        )

            # Fix E: After 3 failures, suggest strategy switch (shell_exec / subtasks)
            _maybe_inject_strategy_switch(harness, empty_path, failure_count)

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
                    _reset_empty_write_failure(harness)
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
