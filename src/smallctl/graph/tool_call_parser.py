from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..client.chunk_parser import extract_response_from_wrapper_tags
from ..state import WriteSession, json_safe_value
from ..task_targets import extract_task_target_paths, primary_task_target_path
from ..write_session_fsm import new_write_session, record_write_session_event
from .state import PendingToolCall
from .tool_model_rules import (
    _clean_reasoning_fallback_text,
    _normalize_model_specific_text,
    _recover_reasoning_only_assistant_text,
    _recover_small_gemma_terminal_message_from_raw_function_syntax,
    _should_discard_qwen_25_structured_inline_tools,
    _strip_empty_qwen_25_execute_wrappers,
    _strip_qwen_25_duplicate_thinking,
)
from .tool_inline_parsing import _extract_inline_tool_calls
from .write_recovery import infer_write_target_path, normalize_write_argument_aliases
from .tool_call_parser_support import (
    _artifact_read_recovery_hint,
    _artifact_read_synthesis_hint,
    _clear_artifact_read_guard_state,
    _clear_tool_attempt_history,
    _coerce_int_or_none,
    _consume_repeat_guard_one_shot_allowance,
    _detect_empty_file_write_payload,
    _detect_timeout_recovered_incomplete_tool_call,
    _detect_missing_required_tool_arguments,
    _detect_patch_existing_stage_read_contract_violation,
    _detect_hallucinated_tool_call,
    _detect_placeholder_tool_call,
    _detect_repeated_tool_loop,
    _dir_list_exploration_progress_is_progress,
    _dir_list_repeat_has_intervening_progress,
    _dir_list_same_path_repeat_is_loop,
    _extract_args_from_fingerprint,
    _extract_artifact_id_from_args,
    _extract_path_from_fingerprint,
    _fallback_repeated_artifact_read,
    _fallback_repeated_file_read,
    _find_full_file_artifact_for_path,
    _file_read_line_progress_is_progress,
    _is_strict_subpath,
    _normalize_json_like,
    _normalize_path_token,
    _normalize_shell_command,
    _normalize_tool_args,
    _normalize_token,
    _placeholder_token,
    _placeholder_value_looks_generic,
    _record_tool_attempt,
    _repeat_loop_limits,
    _requested_artifact_read_target,
    _requested_file_read_range,
    _resolve_artifact_record,
    _resolve_dir_list_path,
    _resolve_file_read_path,
    _should_suppress_resolved_plan_artifact_read,
    _tool_attempt_history,
    _tool_call_fingerprint,
    allow_repeated_tool_call_once,
    _active_write_session_for_target,
    _assistant_declares_read_before_write,
    _assistant_text_target_paths,
    _build_schema_repair_message,
    _declared_read_before_write_reason,
    _detect_oversize_write_payload,
    _ensure_chunk_write_session,
    _infer_write_tool_path,
    _recover_declared_read_before_write,
    _repair_active_write_session_args,
    _salvage_active_write_session_append,
    _should_enter_chunk_mode,
    _suggested_chunk_sections,
)


# ---------------------------------------------------------------------------
# High-level parse entry point
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field

@dataclass
class ToolCallParseResult:
    pending_tool_calls: list[PendingToolCall] = field(default_factory=list)
    final_assistant_text: str = ""
    final_thinking_text: str = ""
    cleaned_text: str = ""


def _text_contains_tool_protocol_markers(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered.strip():
        return False
    markers = (
        "<tool_call",
        "</tool_call>",
        "<tool_code",
        "<call>",
        "</call>",
        "<function=",
        "<function ",
        "</function>",
        "<parameter=",
        "<parameter ",
    )
    return any(marker in lowered for marker in markers)


def _strip_orphan_tool_protocol_markers(text: str) -> str:
    if not text:
        return ""
    stripped = re.sub(
        r"</?(?:tool_call|tool_code|call|function|parameter)(?:=[^>]+)?>",
        "",
        str(text),
        flags=re.IGNORECASE,
    )
    return stripped.strip()


def _strip_tool_protocol_payloads(text: str) -> str:
    if not text:
        return ""
    stripped = re.sub(
        r"<tool_call\b[^>]*>.*?</tool_call>",
        "",
        str(text),
        flags=re.IGNORECASE | re.DOTALL,
    )
    stripped = re.sub(
        r"<tool_code\b[^>]*>.*?</tool_code>",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    stripped = re.sub(
        r"<call\b[^>]*>.*?</call>",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    stripped = re.sub(
        r"<function(?:=[^>]+|\s+name=['\"]?[\w_-]+['\"]?)>.*?</function>",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    stripped = re.sub(
        r"<parameter(?:=[^>]+|\s+name=['\"]?[\w_-]+['\"]?)>.*?</parameter>",
        "",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    stripped = re.sub(
        r"</?(?:tool_call|tool_code|call|function|parameter)(?:=[^>]+)?>",
        "",
        stripped,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def parse_tool_calls(
    stream: Any,
    timeline: list[Any],
    graph_state: Any,
    deps: Any,
    *,
    model_name: str | None = None,
) -> ToolCallParseResult:
    """Extract, deduplicate, and validate tool calls from a completed model stream."""
    harness = deps.harness
    active_model_name = model_name or getattr(getattr(harness, "client", None), "model", None)
    allowed_raw_function_names: set[str] | None = None
    registry = getattr(harness, "registry", None)
    if registry is not None:
        try:
            allowed_raw_function_names = {
                str(name).strip()
                for name in registry.names()
                if str(name).strip()
            }
        except Exception:
            allowed_raw_function_names = None

    # Extract Native tool calls
    native_calls = [
        pending
        for payload in stream.tool_calls
        if (pending := PendingToolCall.from_payload(payload)) is not None
    ]

    # Extract Inline tool calls (from text body)
    assistant_text = _normalize_model_specific_text(stream.assistant_text, model_name=active_model_name)
    assistant_text = _strip_qwen_25_duplicate_thinking(
        assistant_text,
        thinking_text=str(getattr(stream, "thinking_text", "") or ""),
        model_name=active_model_name,
    )
    cleaned_text, inline_calls = _extract_inline_tool_calls(
        assistant_text,
        model_name=active_model_name,
        allowed_raw_function_names=allowed_raw_function_names,
    )
    if _should_discard_qwen_25_structured_inline_tools(
        harness,
        graph_state,
        stream,
        assistant_text,
        model_name=active_model_name,
    ):
        inline_calls = []
        cleaned_text = _strip_empty_qwen_25_execute_wrappers(
            cleaned_text,
            model_name=active_model_name,
        )

    thinking_text = _normalize_model_specific_text(
        str(getattr(stream, "thinking_text", "") or ""),
        model_name=active_model_name,
    )
    thinking_contains_tool_protocol = _text_contains_tool_protocol_markers(thinking_text)
    final_thinking_text = thinking_text
    if thinking_contains_tool_protocol:
        cleaned_thinking_text, thinking_inline_calls = _extract_inline_tool_calls(
            thinking_text,
            model_name=active_model_name,
            allowed_raw_function_names=allowed_raw_function_names,
        )
        if cleaned_thinking_text != thinking_text:
            final_thinking_text = _strip_tool_protocol_payloads(cleaned_thinking_text)
        else:
            stripped_thinking_text = _strip_tool_protocol_payloads(thinking_text)
            if stripped_thinking_text != thinking_text:
                final_thinking_text = stripped_thinking_text
        if not native_calls and not inline_calls and thinking_inline_calls:
            inline_calls.extend(thinking_inline_calls)
            harness._runlog(
                "inline_tool_call_recovered_from_thinking",
                "recovered inline tool call(s) from thinking text",
                recovered_count=len(thinking_inline_calls),
            )
        if final_thinking_text != thinking_text:
            harness._runlog(
                "thinking_tool_protocol_sanitized",
                "sanitized tool protocol noise from thinking text",
                original_length=len(thinking_text),
                sanitized_length=len(final_thinking_text),
            )

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
    if pending_calls and _text_contains_tool_protocol_markers(final_assistant_text):
        final_assistant_text = _strip_orphan_tool_protocol_markers(final_assistant_text)

    recovered_reasoning_text = ""
    allow_reasoning_fallback = not (not pending_calls and thinking_contains_tool_protocol)
    if not final_assistant_text.strip():
        if allow_reasoning_fallback:
            recovered_reasoning_text = _recover_reasoning_only_assistant_text(final_thinking_text)

    if not final_assistant_text.strip() and pending_calls:
        candidate = _clean_reasoning_fallback_text(extract_response_from_wrapper_tags(final_thinking_text))
        if candidate and candidate != "The model returned reasoning text but no final answer.":
            final_assistant_text = apply_triple_answer_guard(candidate, pending_calls)

    if not final_assistant_text.strip() and not pending_calls:
        if allow_reasoning_fallback:
            final_assistant_text = recovered_reasoning_text
            if not final_assistant_text.strip():
                final_assistant_text = _clean_reasoning_fallback_text(final_thinking_text)
    elif not final_assistant_text.strip() and not native_calls:
        final_assistant_text = _recover_small_gemma_terminal_message_from_raw_function_syntax(
            assistant_text,
            pending_calls,
            model_name=active_model_name,
            allowed_tool_names=allowed_raw_function_names,
        )

    return ToolCallParseResult(
        pending_tool_calls=pending_calls,
        final_assistant_text=final_assistant_text.strip(),
        final_thinking_text=final_thinking_text.strip(),
        cleaned_text=cleaned_text,
    )
