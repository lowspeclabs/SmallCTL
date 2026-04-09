from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path
from typing import Any

from ..client.chunk_parser import extract_thinking_from_tags
from ..state import WriteSession, json_safe_value
from ..task_targets import extract_task_target_paths, primary_task_target_path
from ..write_session_fsm import new_write_session, record_write_session_event
from .state import PendingToolCall
from .write_recovery import infer_write_target_path, normalize_write_argument_aliases

_REPEATED_TOOL_HISTORY_LIMIT = 12
_IDENTICAL_TOOL_CALL_STREAK_LIMIT = 3
_REPEATED_TOOL_WINDOW = 6
_REPEATED_TOOL_UNIQUE_LIMIT = 3
_PLACEHOLDER_TOOL_NAME_TOKENS = {
    "tool_name",
    "function_name",
    "action_name",
    "tool",
    "function",
    "action",
    "name",
}
_PLACEHOLDER_ARG_KEY_TOKENS = {
    "arg",
    "args",
    "argument",
    "arguments",
    "param",
    "params",
    "parameter",
    "parameters",
    "value",
    "field",
}
_PLACEHOLDER_ARG_VALUE_TOKENS = {
    "",
    "arg",
    "args",
    "value",
    "parameter",
    "parameters",
    "param",
    "params",
    "tool_name",
    "function_name",
    "action_name",
    "placeholder",
    "string",
    "text",
}
_GLM_BOX_MODEL_MARKERS = (
    "zai-org/glm-4.6v-flash",
    "glm-4.6v-flash",
    "zai-org/glm",
)
_GLM_BOX_TAGS = (
    "<|begin_of_box|>",
    "<|end_of_box|>",
    "<|begin_of_thought|>",
    "<|end_of_thought|>",
)
_GPT_OSS_MODEL_MARKERS = (
    "openai/gpt-oss-20b",
    "gpt-oss-20b",
    "openai/gpt-oss",
)
_QWEN_MODEL_MARKERS = (
    "qwen/",
    "qwen2.5",
    "qwen-2.5",
    "qwen3",
    "qwen-3",
    "qwen3.5",
    "qwen-3.5",
)
_EXACT_QWEN_25_7B_INSTRUCT_MODELS = (
    "qwen/qwen-2.5-7b-instruct",
    "qwen-2.5-7b-instruct",
    "qwen2.5-7b-instruct",
)
_EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES = (
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
)
_RAW_FUNCTION_VALUE_PATTERN = r"(?:'[^']*'|\"[^\"]*\"|[0-9.]+)"
_RAW_FUNCTION_ESCAPED_VALUE_PATTERN = r"(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|[0-9.]+)"
_GPT_OSS_TAGS = (
    "<|channel|>",
    "<|constrain|>",
    "<|message|>",
    "<think",
    "<think<|message|>",
)
_GEMMA_MODEL_MARKERS = (
    "google_gemma-4",
    "google_gemma",
    "gemma-4",
    "gemma-3",
    "gemma/",
)
_GEMMA_TAGS = (
    "<channel|>",
    "<thought>",
    "<|channel|>",
    "<|thought|>",
)


def _normalize_token(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return normalized.strip("_")


def _clean_reasoning_fallback_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(
        r"</?(?:tool_call|tool_code|call|function|parameter|thinking|reasoning)(?:=[^>]+)?>",
        "",
        str(text),
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _model_uses_glm_box_rules(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _GLM_BOX_MODEL_MARKERS))


def _model_uses_gpt_oss_rules(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _GPT_OSS_MODEL_MARKERS))


def _model_uses_gemma_rules(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _GEMMA_MODEL_MARKERS))


def _model_uses_qwen_rules(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(normalized and any(marker in normalized for marker in _QWEN_MODEL_MARKERS))


def _model_is_exact_qwen_25_7b_instruct(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized in _EXACT_QWEN_25_7B_INSTRUCT_MODELS


def _model_is_exact_small_gemma_4_it(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(
        normalized
        and any(
            normalized == suffix or normalized.endswith(f"/{suffix}")
            for suffix in _EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES
        )
    )


def _raw_function_value_pattern(*, model_name: str | None = None) -> str:
    if _model_is_exact_qwen_25_7b_instruct(model_name):
        return _RAW_FUNCTION_ESCAPED_VALUE_PATTERN
    return _RAW_FUNCTION_VALUE_PATTERN


def _raw_function_kv_pattern(*, model_name: str | None = None) -> str:
    value_pattern = _raw_function_value_pattern(model_name=model_name)
    return rf"([a-zA-Z0-9_-]+)\s*=\s*({value_pattern})"


def _raw_function_call_pattern(*, model_name: str | None = None) -> str:
    kv_pattern = _raw_function_kv_pattern(model_name=model_name)
    return rf"([a-zA-Z0-9_-]+)\(({kv_pattern}(?:\s*,\s*{kv_pattern})*)\)"


def _parse_raw_function_call(
    text: str,
    *,
    model_name: str | None = None,
    allowed_tool_names: set[str] | None = None,
) -> PendingToolCall | None:
    candidate_text = str(text or "")
    if _model_uses_gemma_rules(model_name):
        angle_wrapped = re.match(
            rf"^\s*<\s*(?P<body>{_raw_function_call_pattern(model_name=model_name)})\s*>\s*$",
            candidate_text,
            re.DOTALL,
        )
        if angle_wrapped is not None:
            candidate_text = angle_wrapped.group("body")

    match = re.match(
        rf"^\s*{_raw_function_call_pattern(model_name=model_name)}\s*$",
        candidate_text,
        re.DOTALL,
    )
    if match is None:
        return None

    tool_name = match.group(1)
    if allowed_tool_names is not None:
        normalized_tool_name = _normalize_token(tool_name)
        if tool_name not in allowed_tool_names and normalized_tool_name not in _PLACEHOLDER_TOOL_NAME_TOKENS:
            return None
    args_str = match.group(2).strip()
    kv_pairs = re.findall(_raw_function_kv_pattern(model_name=model_name), args_str)
    if not kv_pairs:
        return None

    try:
        args = {key: ast.literal_eval(value) for key, value in kv_pairs}
    except Exception:
        return None

    return PendingToolCall(
        tool_name=tool_name,
        args=args,
        raw_arguments=args_str,
    )


def _recover_small_gemma_terminal_message_from_raw_function_syntax(
    assistant_text: str,
    pending_calls: list[PendingToolCall],
    *,
    model_name: str | None = None,
    allowed_tool_names: set[str] | None = None,
) -> str:
    if not _model_is_exact_small_gemma_4_it(model_name):
        return ""
    if not assistant_text or not pending_calls:
        return ""

    raw_call = _parse_raw_function_call(
        assistant_text,
        model_name=model_name,
        allowed_tool_names=allowed_tool_names,
    )
    if raw_call is None or raw_call.tool_name not in {"task_complete", "task_fail"}:
        return ""

    for call in pending_calls:
        if call.tool_name != raw_call.tool_name:
            continue
        if dict(call.args or {}) != dict(raw_call.args or {}):
            continue
        return str(call.args.get("message", "") or "").strip()
    return ""


def _strip_gpt_oss_channel_prefix(text: str) -> str:
    """Remove the protocol wrapper that gpt-oss emits in assistant content."""
    stripped = re.sub(
        r"^\s*(?:commentary|analysis|final|assistant|tool)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return stripped


def _strip_exact_small_gemma_4_protocol_noise(text: str, *, model_name: str | None) -> str:
    if not _model_is_exact_small_gemma_4_it(model_name):
        return text
    return re.sub(
        r"</?\|?(?:channel|thought)\|?>",
        "",
        str(text or ""),
        flags=re.IGNORECASE,
    )


def _normalize_model_specific_text(text: str, *, model_name: str | None) -> str:
    if not text:
        return ""
    normalized = str(text)
    if _model_uses_qwen_rules(model_name):
        normalized = re.sub(r"^\s*<\|im_start\|>\s*assistant\s*", "", normalized, flags=re.IGNORECASE)
        normalized = normalized.replace("<|im_start|>", "")
        normalized = normalized.replace("<|im_end|>", "")
    if _model_uses_glm_box_rules(model_name):
        for tag in _GLM_BOX_TAGS:
            normalized = normalized.replace(tag, "")
    if _model_uses_gpt_oss_rules(model_name):
        for tag in _GPT_OSS_TAGS:
            normalized = normalized.replace(tag, "")
        normalized = _strip_gpt_oss_channel_prefix(normalized)
    if _model_uses_gemma_rules(model_name):
        for tag in _GEMMA_TAGS:
            normalized = normalized.replace(tag, "")
        normalized = _strip_exact_small_gemma_4_protocol_noise(
            normalized,
            model_name=model_name,
        )
    return normalized.strip()


def _strip_qwen_25_duplicate_thinking(
    text: str,
    *,
    thinking_text: str = "",
    model_name: str | None = None,
) -> str:
    if not _model_is_exact_qwen_25_7b_instruct(model_name):
        return text
    if not text or not str(thinking_text or "").strip():
        return text

    lowered = text.lower()
    if "<think>" not in lowered and "<thinking>" not in lowered:
        return text

    assistant_text, _ = extract_thinking_from_tags(
        text,
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
    )
    return assistant_text.strip()


def _should_discard_qwen_25_structured_inline_tools(
    harness: Any,
    graph_state: Any,
    stream: Any,
    assistant_text: str,
    *,
    model_name: str | None = None,
) -> bool:
    if not _model_is_exact_qwen_25_7b_instruct(model_name):
        return False
    if str(getattr(graph_state, "run_mode", "") or "").strip().lower() != "chat":
        return False
    if getattr(stream, "tool_calls", None):
        return False

    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {}) or {}
    if scratchpad.get("_chat_tools_exposed") is not False:
        return False
    if not str(scratchpad.get("_chat_tools_suppressed_reason") or "").strip():
        return False

    lowered = str(assistant_text or "").lower()
    if any(token in lowered for token in ("<execute", "<tool_call", "<tool_code", "<call>", "<function")):
        return True
    return bool(re.search(r'\{\s*"(?:name|tool_name|tool|action)"\s*:', str(assistant_text or "")))


def _strip_empty_qwen_25_execute_wrappers(text: str, *, model_name: str | None = None) -> str:
    if not _model_is_exact_qwen_25_7b_instruct(model_name):
        return text
    if not text:
        return ""
    cleaned = re.sub(r"<execute>\s*</execute>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _extract_inline_tool_calls(
    text: str,
    *,
    model_name: str | None = None,
    allowed_raw_function_names: set[str] | None = None,
) -> tuple[str, list[PendingToolCall]]:
    if not text:
        return "", []

    results: list[PendingToolCall] = []
    cleaned_text = _strip_exact_small_gemma_4_protocol_noise(
        text,
        model_name=model_name,
    )

    def _parse_bracketed_tool_block(block_text: str) -> PendingToolCall | None:
        if not block_text:
            return None
        match = re.match(r"^\s*\[([A-Za-z0-9_-]+)\]\s*(.*)\s*$", block_text, re.DOTALL)
        if not match:
            return None
        tool_name = match.group(1).strip()
        payload_text = match.group(2).strip()
        if not tool_name or not payload_text:
            return None
        return PendingToolCall.from_payload(
            {
                "function": {
                    "name": tool_name,
                    "arguments": payload_text,
                }
            }
        )

    def _parse_xml_function_block(block_text: str) -> PendingToolCall | None:
        if not block_text:
            return None

        compact_fn_match = re.match(r"^\s*<function=([\w_-]+)>\s*(\{.*\})\s*$", block_text, re.DOTALL)
        if compact_fn_match:
            return PendingToolCall.from_payload(
                {
                    "function": {
                        "name": compact_fn_match.group(1).strip(),
                        "arguments": compact_fn_match.group(2).strip(),
                    }
                }
            )

        struct_patterns = (
            r"<function=([\w_-]+)>(.*?)</function>",
            r"<function\s+name=['\"]?([\w_-]+)['\"]?\s*>(.*?)</function>",
        )
        parameter_patterns = (
            r"<parameter=([\w_-]+)>(.*?)</parameter>",
            r"<parameter\s+name=['\"]?([\w_-]+)['\"]?\s*>(.*?)</parameter>",
        )
        for pattern in struct_patterns:
            struct_fn_match = re.search(pattern, block_text, re.DOTALL)
            if struct_fn_match is None:
                continue
            tool_name = struct_fn_match.group(1).strip()
            inner_content = struct_fn_match.group(2).strip()
            if not tool_name:
                continue
            if inner_content.startswith("{"):
                pending = PendingToolCall.from_payload(
                    {
                        "function": {
                            "name": tool_name,
                            "arguments": inner_content,
                        }
                    }
                )
                if pending is not None:
                    return pending
            params = {}
            for param_pattern in parameter_patterns:
                for pk, pv in re.findall(param_pattern, inner_content, re.DOTALL):
                    params[pk] = pv.strip()
            if params:
                return PendingToolCall(
                    tool_name=tool_name,
                    args=params,
                    raw_arguments=json.dumps(params, ensure_ascii=True, sort_keys=True),
                )
        return None

    def _try_parse_data(data: Any) -> PendingToolCall | None:
        if not isinstance(data, dict):
            return None
        if isinstance(data.get("function"), dict):
            pending = PendingToolCall.from_payload(data)
            if pending is not None:
                return pending
        # Key Hallucination Handling: Support name, tool_name, tool, action
        name = str(data.get("name", data.get("tool_name", data.get("tool", data.get("action", ""))))).strip()
        if not name:
            return None
        # Args Hallucination Handling: Support arguments, args, params, parameters
        args = data.get("arguments", data.get("args", data.get("params", data.get("parameters", {}))))
        if isinstance(args, dict):
            raw_arguments = json.dumps(args)
        elif isinstance(args, str):
            raw_arguments = args
        else:
            raw_arguments = "{}"
        payload = {
            "function": {
                "name": name,
                "arguments": raw_arguments,
            }
        }
        return PendingToolCall.from_payload(payload)

    # 1. Check for XML-style blocks: <tool_code>, <tool_call>, etc.
    xml_patterns = [
        r"<tool_code>(.*?)</tool_code>",
        r"<tool_call>(.*?)</tool_call>",
        r"<call>(.*?)</call>"
    ]
    for pattern in xml_patterns:
        it = re.finditer(pattern, cleaned_text, re.DOTALL)
        offset = 0
        for match in it:
            content = match.group(1).strip()

            found = False
            pending = _parse_xml_function_block(content)
            if pending is not None:
                results.append(pending)
                found = True

            if not found:
                # 1b. Attempt JSON inside the tag
                try:
                    data = json.loads(content)
                    pending = _try_parse_data(data)
                    if pending:
                        results.append(pending)
                        found = True
                except Exception:
                    pass

            if not found:
                # 1c. Attempt Function call style: name(arg='val', ...)
                pending = _parse_raw_function_call(
                    content,
                    model_name=model_name,
                    allowed_tool_names=allowed_raw_function_names,
                )
                if pending is not None:
                    results.append(pending)
                    found = True

            if found:
                # Strip the matched block from cleaned_text
                start, end = match.span()
                cleaned_text = cleaned_text[:start-offset] + cleaned_text[end-offset:]
                offset += (end - start)

    # 1b. Catch bracketed tool wrappers like `[file_write] { ... }`
    bracket_tool_pattern = r"\[\s*([A-Za-z0-9_-]+)\s*\]\s*(\{.*?\})"
    bracket_matches = list(re.finditer(bracket_tool_pattern, cleaned_text, re.DOTALL))
    offset = 0
    for match in bracket_matches:
        block = match.group(0)
        pending = _parse_bracketed_tool_block(block)
        if pending is None:
            continue
        results.append(pending)
        start, end = match.span()
        cleaned_text = cleaned_text[:start-offset] + cleaned_text[end-offset:]
        offset += (end - start)

    # 1d. Catch structured XML even if NOT wrapped in tags (paranoid)
    struct_fn_matches = list(
        re.finditer(
            r"<function(?:=[\w_-]+|\s+name=['\"]?[\w_-]+['\"]?)>.*?</function>",
            cleaned_text,
            re.DOTALL,
        )
    )
    offset = 0
    for match in struct_fn_matches:
        pending = _parse_xml_function_block(match.group(0))
        if pending is not None:
            results.append(pending)
            start, end = match.span()
            cleaned_text = cleaned_text[:start-offset] + cleaned_text[end-offset:]
            offset += (end - start)

    # 2. Existing JSON extractors (fallbacks if no XML found or in addition)
    # 2a. Check for markdown JSON blocks (allow flexible whitespace)
    json_blocks = list(re.finditer(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL))
    offset = 0
    for match in json_blocks:
        block = match.group(1)
        try:
            data = json.loads(block)
            pending = _try_parse_data(data)
            if pending:
                results.append(pending)
                start, end = match.span()
                cleaned_text = cleaned_text[:start-offset] + cleaned_text[end-offset:]
                offset += (end - start)
        except Exception:
            pass

    # 2b. Raw outer-level JSON objects
    if "{" in cleaned_text:
        start = cleaned_text.find("{")
        while start != -1:
            brace_count = 0
            end = -1
            for i in range(start, len(cleaned_text)):
                if cleaned_text[i] == "{": brace_count += 1
                elif cleaned_text[i] == "}": brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

            if end != -1:
                try:
                    data = json.loads(cleaned_text[start:end])
                    pending = _try_parse_data(data)
                    if pending:
                        results.append(pending)
                        # Strip it
                        cleaned_text = cleaned_text[:start] + cleaned_text[end:]
                        # Continue searching from same START position (since we shifted text)
                        start = cleaned_text.find("{", start)
                    else:
                        # Continue searching for next potential object
                        start = cleaned_text.find("{", start + 1)
                except Exception:
                     start = cleaned_text.find("{", start + 1)
            else:
                break

    # 2c. Functional style name(arg='val') in raw text. Only consider standalone
    # lines so ordinary source code like `parser = ArgumentParser(...)` does not
    # get promoted into a tool call.
    standalone_line_regex = r"(?m)^[ \t]*(?P<body>.+?)[ \t]*$"
    matches = list(re.finditer(standalone_line_regex, cleaned_text))
    offset = 0
    for match in matches:
        line_body = _strip_exact_small_gemma_4_protocol_noise(
            match.group("body"),
            model_name=model_name,
        )
        pending = _parse_raw_function_call(
            line_body,
            model_name=model_name,
            allowed_tool_names=allowed_raw_function_names,
        )
        if pending is None:
            continue
        pending = _try_parse_data({"tool_name": pending.tool_name, "arguments": pending.args})
        if pending:
            results.append(pending)
            start, end = match.span()
            cleaned_text = cleaned_text[:start-offset] + cleaned_text[end-offset:]
            offset += (end - start)

    return cleaned_text, results


def _detect_hallucinated_tool_call(harness: Any, pending: PendingToolCall) -> str | None:
    """Detect if a tool call is missing all its required arguments."""
    meta = harness.registry.get(pending.tool_name)
    if not meta:
        return None

    schema = meta.schema or {}
    required = schema.get("required", [])
    if not required:
        return None

    # If the tool requires something but we have nothing, it's a hallucination trap
    if not pending.args:
        return (
            f"Hallucination Warning: Tool '{pending.tool_name}' requires specific parameters "
            f"({', '.join(required)}) but was called with none. Please provide the missing arguments."
        )

    return None


def _placeholder_token(value: Any) -> str:
    return _normalize_token(value)


def _placeholder_value_looks_generic(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        token = _placeholder_token(value)
        return token in _PLACEHOLDER_ARG_VALUE_TOKENS or token.startswith("placeholder")
    if isinstance(value, dict):
        if not value:
            return True
        return all(
            _placeholder_token(key) in _PLACEHOLDER_ARG_KEY_TOKENS
            and _placeholder_value_looks_generic(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return bool(value) and all(_placeholder_value_looks_generic(item) for item in value)
    return False


def _detect_placeholder_tool_call(harness: Any, pending: PendingToolCall) -> tuple[str, dict[str, Any]] | None:
    tool_name = str(getattr(pending, "tool_name", "") or "").strip()
    if not tool_name:
        return None

    registry = getattr(harness, "registry", None)
    if registry is not None:
        try:
            if tool_name in set(registry.names()):
                return None
        except Exception:
            pass

    if _placeholder_token(tool_name) not in _PLACEHOLDER_TOOL_NAME_TOKENS:
        return None

    args = dict(getattr(pending, "args", {}) or {})
    if args:
        placeholder_keys = all(_placeholder_token(key) in _PLACEHOLDER_ARG_KEY_TOKENS for key in args)
        placeholder_values = all(_placeholder_value_looks_generic(value) for value in args.values())
        if not (placeholder_keys and placeholder_values):
            return None

    message = (
        "Placeholder tool schema detected. You emitted the literal tool name "
        f"`{tool_name}` with example arguments instead of a real tool call. "
        "Regenerate the full JSON tool call from scratch using an actual registered tool name "
        "and concrete arguments. Do not send schema examples or placeholder fields like `arg: value`."
    )
    return message, {
        "tool_name": tool_name,
        "tool_call_id": pending.tool_call_id,
        "reason": "placeholder_tool_schema",
        "offending_field": "tool_name",
        "placeholder_arguments": json_safe_value(args),
    }


def _detect_repeated_tool_loop(harness: Any, pending: PendingToolCall) -> str | None:
    if pending.tool_name in {"task_complete", "task_fail", "ask_human"}:
        _clear_tool_attempt_history(harness)
        return None
    if pending.tool_name == "shell_exec":
        command = str(pending.args.get("command") or "").strip()
        job_id = str(pending.args.get("job_id") or "").strip()
        # Polling an existing background job is an expected repeat, not a loop.
        if job_id and not command:
            return None
    if _dir_list_same_path_repeat_is_loop(harness, pending):
        return "Guard tripped: repeated dir_list loop (same path repeated without progress)"
    if _dir_list_exploration_progress_is_progress(harness, pending):
        return None
    if _file_read_line_progress_is_progress(harness, pending):
        return None
    if _artifact_read_line_progress_is_progress(harness, pending):
        return None
    history = _tool_attempt_history(harness)
    candidate = {
        "tool_name": pending.tool_name,
        "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args),
    }
    recent_window = history[-(_REPEATED_TOOL_WINDOW - 1) :] + [candidate]
    exact_streak = history[-(_IDENTICAL_TOOL_CALL_STREAK_LIMIT - 1) :] + [candidate]
    if (
        len(exact_streak) >= _IDENTICAL_TOOL_CALL_STREAK_LIMIT
        and len({str(item.get("fingerprint", "")) for item in exact_streak}) == 1
    ):
        return (
            "Guard tripped: repeated tool call loop "
            f"({pending.tool_name} repeated with identical arguments)"
        )
    if len(recent_window) < _REPEATED_TOOL_WINDOW:
        return None
    tool_names = {str(item.get("tool_name", "")) for item in recent_window}
    fingerprints = [str(item.get("fingerprint", "")) for item in recent_window]
    if len(tool_names) == 1 and len(set(fingerprints)) <= _REPEATED_TOOL_UNIQUE_LIMIT:
        return (
            "Guard tripped: repeated tool exploration loop "
            f"({pending.tool_name} cycling through near-identical arguments without progress)"
        )
    return None


def _dir_list_exploration_progress_is_progress(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "dir_list":
        return False

    candidate_path = _resolve_dir_list_path(harness, pending.args)
    if candidate_path is None:
        return False

    history = _tool_attempt_history(harness)
    recent_paths: list[Path] = []
    for item in reversed(history):
        if str(item.get("tool_name", "")) != "dir_list":
            continue
        fingerprint = str(item.get("fingerprint", ""))
        path = _extract_path_from_fingerprint(harness, fingerprint)
        if path is None:
            continue
        recent_paths.append(path)
        if len(recent_paths) >= 4:
            break

    if not recent_paths:
        return False

    paths = list(reversed(recent_paths))
    paths.append(candidate_path)
    if len(paths) < 2:
        return False

    monotonic_descent = all(_is_strict_subpath(paths[index + 1], paths[index]) for index in range(len(paths) - 1))
    return monotonic_descent


def _dir_list_same_path_repeat_is_loop(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "dir_list":
        return False

    candidate_path = _resolve_dir_list_path(harness, pending.args)
    if candidate_path is None:
        return False

    history = _tool_attempt_history(harness)
    for index in range(len(history) - 1, -1, -1):
        item = history[index]
        if str(item.get("tool_name", "")) != "dir_list":
            continue
        fingerprint = str(item.get("fingerprint", ""))
        path = _extract_path_from_fingerprint(harness, fingerprint)
        if path is None:
            continue
        if path != candidate_path:
            continue
        if not _model_is_exact_small_gemma_4_it(getattr(getattr(harness, "client", None), "model", None)):
            return True
        return not _dir_list_repeat_has_intervening_progress(history, index)
    return False


def _dir_list_repeat_has_intervening_progress(
    history: list[dict[str, str]],
    prior_dir_list_index: int,
) -> bool:
    progress_tools = {"artifact_read", "file_read", "shell_exec", "ssh_exec", "bash_exec"}
    if prior_dir_list_index < 0 or prior_dir_list_index >= len(history):
        return False
    for item in history[prior_dir_list_index + 1 :]:
        if str(item.get("tool_name", "")) in progress_tools:
            return True
    return False


def _resolve_dir_list_path(harness: Any, args: dict[str, Any]) -> Path | None:
    raw_path = args.get("path", ".")
    if not isinstance(raw_path, str):
        return None
    candidate = Path(raw_path.strip() or ".")
    if candidate.is_absolute():
        return candidate.resolve()
    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
    try:
        return (base / candidate).resolve()
    except Exception:
        return (base / candidate)


def _file_read_line_progress_is_progress(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "file_read":
        return False

    candidate_path = _resolve_file_read_path(harness, pending.args)
    if candidate_path is None:
        return False

    candidate_range = _requested_file_read_range(pending.args)
    if candidate_range == (None, None):
        return False

    history = _tool_attempt_history(harness)
    for item in reversed(history):
        if str(item.get("tool_name", "")) != "file_read":
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if not isinstance(args, dict):
            continue
        prior_path = _resolve_file_read_path(harness, args)
        if prior_path is None or prior_path != candidate_path:
            continue
        prior_range = _requested_file_read_range(args)
        return prior_range != candidate_range
    return False


def _artifact_read_line_progress_is_progress(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "artifact_read":
        return False

    candidate_artifact_id = _requested_artifact_read_target(pending.args)
    if not candidate_artifact_id:
        return False

    candidate_range = _requested_file_read_range(pending.args)
    if candidate_range == (None, None):
        return False

    history = _tool_attempt_history(harness)
    for item in reversed(history):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if not isinstance(args, dict):
            continue
        prior_artifact_id = _requested_artifact_read_target(args)
        if not prior_artifact_id or prior_artifact_id != candidate_artifact_id:
            continue
        prior_range = _requested_file_read_range(args)
        return prior_range != candidate_range
    return False


def _resolve_file_read_path(harness: Any, args: dict[str, Any]) -> Path | None:
    raw_path = args.get("path")
    if not isinstance(raw_path, str):
        return None
    candidate = Path(raw_path.strip() or ".")
    if candidate.is_absolute():
        try:
            return candidate.resolve()
        except Exception:
            return candidate
    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
    try:
        return (base / candidate).resolve()
    except Exception:
        return base / candidate


def _requested_file_read_range(args: dict[str, Any]) -> tuple[int | None, int | None]:
    if not isinstance(args, dict):
        return (None, None)
    start_line = args.get("requested_start_line", args.get("start_line"))
    end_line = args.get("requested_end_line", args.get("end_line"))
    return (_coerce_int_or_none(start_line), _coerce_int_or_none(end_line))


def _requested_artifact_read_target(args: dict[str, Any]) -> str:
    if not isinstance(args, dict):
        return ""
    artifact_id = args.get("artifact_id")
    return str(artifact_id or "").strip()


def _coerce_int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_args_from_fingerprint(fingerprint: str) -> dict[str, Any] | None:
    if not fingerprint:
        return None
    try:
        payload = json.loads(fingerprint)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    args = payload.get("args", {})
    return args if isinstance(args, dict) else None


def _extract_path_from_fingerprint(harness: Any, fingerprint: str) -> Path | None:
    args = _extract_args_from_fingerprint(fingerprint)
    if not isinstance(args, dict):
        return None
    return _resolve_dir_list_path(harness, args)


def _is_strict_subpath(child: Path, parent: Path) -> bool:
    if child == parent:
        return False
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _detect_missing_required_tool_arguments(harness: Any, pending: PendingToolCall) -> tuple[str, dict[str, Any]] | None:
    registry = getattr(harness, "registry", None)
    if registry is None:
        return None
    tool_spec = registry.get(pending.tool_name)
    if tool_spec is None:
        return None
    required = tool_spec.schema.get("required", [])
    if not required:
        return None

    missing_fields = []
    for field in required:
        value = pending.args.get(field)
        if value is None:
            missing_fields.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing_fields.append(field)

    if not missing_fields:
        return None

    message = (
        f"Tool call '{pending.tool_name}' was emitted without arguments. "
        f"Required fields: {', '.join(str(field) for field in missing_fields)}."
    )
    return message, {
        "tool_name": pending.tool_name,
        "tool_call_id": pending.tool_call_id,
        "required_fields": list(missing_fields),
        "raw_arguments": pending.raw_arguments,
    }


def _detect_empty_file_write_payload(
    harness: Any,
    pending: PendingToolCall,
) -> tuple[str, dict[str, Any]] | None:
    if pending.tool_name not in {"file_write", "file_append"}:
        return None
    
    # Allow empty writes only if explicitly in a session fallback mode (like stub_and_fill)
    state = getattr(harness, "state", None)
    if state is None:
        return None
    write_session = getattr(state, "write_session", None)
    if write_session and write_session.write_session_mode == "stub_and_fill":
        return None

    content = pending.args.get("content")
    if content is not None and str(content).strip():
        return None

    required_fields: list[str] = []
    path_value = pending.args.get("path")
    if path_value is None or (isinstance(path_value, str) and not path_value.strip()):
        required_fields.append("path")
    if content is None or not str(content).strip():
        required_fields.append("content")

    message = (
        f"Empty payload rejected for `{pending.tool_name}`. Provide concrete content. "
        "If a write session is active, resume it with `file_write` and full session metadata."
    )
    return message, {
        "tool_name": pending.tool_name,
        "tool_call_id": pending.tool_call_id,
        "reason": "empty_payload",
        "required_fields": required_fields,
    }


def _write_policy_value(harness: Any, name: str, default: Any) -> Any:
    config = getattr(harness, "config", None)
    if config is None:
        return default
    return getattr(config, name, default)


def _task_forces_chunk_mode(harness: Any, path: str) -> bool:
    if not path:
        return False
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {}) or {}
    forced_targets = scratchpad.get("_force_chunk_mode_targets")
    if not isinstance(forced_targets, list) or not forced_targets:
        return False

    from ..tools.fs import _resolve

    try:
        candidate = _resolve(path, getattr(harness.state, "cwd", None))
    except Exception:
        candidate = None
    for target in forced_targets:
        try:
            forced_path = _resolve(str(target), getattr(harness.state, "cwd", None))
        except Exception:
            forced_path = None
        if candidate is not None and forced_path is not None and candidate == forced_path:
            return True
        if str(target).strip() == str(path).strip():
            return True
    return False


def _suggested_chunk_sections(path: str) -> list[str]:
    ext = Path(path).suffix.lower()
    if ext == ".py":
        return ["imports", "types/interfaces", "constants/globals", "helpers", "main_logic", "tests/entrypoint"]
    if ext in {".js", ".ts", ".tsx"}:
        return ["imports", "types", "constants", "utils", "main_component/logic", "exports"]
    if ext == ".go":
        return ["package", "imports", "types", "const/var", "helpers", "main_logic"]
    if ext in {".md", ".txt"}:
        return ["header", "overview", "details", "footer"]
    return ["header", "implementation", "footer"]


def _active_write_session_for_target(harness: Any, target_path: str) -> WriteSession | None:
    target = str(target_path or "").strip()
    if not target:
        return None
    session = getattr(getattr(harness, "state", None), "write_session", None)
    if not session or str(getattr(session, "status", "")).strip().lower() == "complete":
        return None

    from ..tools.fs import _resolve

    try:
        target_resolved = _resolve(target, getattr(harness.state, "cwd", None))
        session_resolved = _resolve(session.write_target_path, getattr(harness.state, "cwd", None))
    except Exception:
        target_resolved = None
        session_resolved = None
    if target_resolved is not None and session_resolved is not None and target_resolved == session_resolved:
        return session
    if str(session.write_target_path).strip() == target:
        return session
    return None


def _assistant_text_target_paths(harness: Any, assistant_text: str = "") -> list[str]:
    candidates: list[str] = []
    if assistant_text.strip():
        candidates.append(assistant_text)

    recent_messages = getattr(getattr(harness, "state", None), "recent_messages", [])
    for message in reversed(recent_messages):
        if getattr(message, "role", "") != "assistant":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            candidates.append(content)
        if len(candidates) >= 4:
            break

    ordered: list[str] = []
    seen: set[str] = set()
    for text in candidates:
        for path in extract_task_target_paths(text):
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)
    return ordered


def _infer_write_tool_path(harness: Any, pending: PendingToolCall, *, assistant_text: str = "") -> str:
    if pending.tool_name not in {"file_write", "file_append"}:
        return ""

    explicit_path = str(pending.args.get("path") or "").strip()
    if explicit_path:
        return explicit_path

    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is not None and str(getattr(session, "status", "")).strip().lower() != "complete":
        session_target = str(getattr(session, "write_target_path", "") or "").strip()
        if session_target:
            return session_target

    task_target = str(primary_task_target_path(harness) or "").strip()
    if task_target:
        return task_target

    assistant_paths = _assistant_text_target_paths(harness, assistant_text)
    if assistant_paths:
        return assistant_paths[0]

    return ""


def _repair_active_write_session_args(
    harness: Any,
    pending: PendingToolCall,
    *,
    assistant_text: str = "",
) -> bool:
    if pending.tool_name not in {"file_write", "file_append"}:
        return False

    raw_args = dict(getattr(pending, "args", {}) or {})
    args = normalize_write_argument_aliases(raw_args)
    repaired = args != raw_args

    def _is_blank(value: Any) -> bool:
        return value is None or (isinstance(value, str) and not value.strip())

    def _looks_like_system_repair_cycle_id(value: Any) -> bool:
        return str(value or "").strip().lower().startswith("repair-")

    session = getattr(getattr(harness, "state", None), "write_session", None)
    if session is not None and str(getattr(session, "status", "")).strip().lower() != "complete":
        session_id = str(args.get("write_session_id") or "").strip()
        if session_id and session_id == session.write_session_id:
            if _is_blank(args.get("path")) and str(session.write_target_path or "").strip():
                args["path"] = session.write_target_path
                repaired = True
            if _is_blank(args.get("section_name")) and _is_blank(args.get("section_id")):
                section_name = str(
                    session.write_next_section
                    or session.write_current_section
                    or ""
                ).strip()
                if section_name:
                    args["section_name"] = section_name
                    repaired = True
        elif session_id and _looks_like_system_repair_cycle_id(session_id):
            inferred_path = str(args.get("path") or "").strip()
            if not inferred_path:
                inferred_path, _confidence, _evidence = infer_write_target_path(
                    harness=harness,
                    pending=pending,
                    assistant_text=assistant_text,
                    partial_tool_calls=None,
                )
            session_matches_target = not inferred_path
            if inferred_path:
                session_matches_target = _active_write_session_for_target(harness, inferred_path) is session
            if session_matches_target:
                args["write_session_id"] = session.write_session_id
                repaired = True
                if _is_blank(args.get("path")) and str(session.write_target_path or "").strip():
                    args["path"] = session.write_target_path
                if _is_blank(args.get("section_name")) and _is_blank(args.get("section_id")):
                    section_name = str(
                        session.write_next_section
                        or session.write_current_section
                        or ""
                    ).strip()
                    if section_name:
                        args["section_name"] = section_name

    if _is_blank(args.get("path")):
        inferred_path, _confidence, _evidence = infer_write_target_path(
            harness=harness,
            pending=pending,
            assistant_text=assistant_text,
            partial_tool_calls=None,
        )
        if inferred_path:
            args["path"] = inferred_path
            repaired = True

    if not repaired:
        return False

    pending.args = args
    pending.raw_arguments = json.dumps(args, ensure_ascii=True, sort_keys=True)
    return True


def _ensure_chunk_write_session(harness: Any, target_path: str) -> WriteSession | None:
    target = str(target_path or "").strip()
    if not target:
        return None

    existing = _active_write_session_for_target(harness, target)
    if existing is not None:
        return existing

    from ..guards import is_small_model_name

    model_name = getattr(getattr(harness, "client", None), "model", None)
    if not is_small_model_name(model_name):
        return None
    if not _task_forces_chunk_mode(harness, target):
        return None

    from ..tools.fs import infer_write_session_intent, new_write_session_id

    suggestions = _suggested_chunk_sections(target)
    session = new_write_session(
        session_id=new_write_session_id(),
        target_path=target,
        intent=infer_write_session_intent(target, getattr(harness.state, "cwd", None)),
        mode="chunked_author",
        suggested_sections=suggestions,
        next_section=suggestions[0] if suggestions else "",
    )
    harness.state.write_session = session
    from .tool_outcomes import _register_write_session_stage_artifact
    _register_write_session_stage_artifact(harness, session)
    record_write_session_event(
        harness.state,
        event="session_opened",
        session=session,
        details={"source": "chunk_mode_recovery"},
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "chunk_mode_prearmed",
            "initialized chunked authoring session from recovery path",
            session_id=session.write_session_id,
            target_path=target,
        )
    return session


def _salvage_active_write_session_append(
    harness: Any,
    pending: PendingToolCall,
) -> PendingToolCall | None:
    if pending.tool_name != "file_append":
        return None

    content = pending.args.get("content")
    if content is not None and str(content).strip():
        return None

    active_session = getattr(getattr(harness, "state", None), "write_session", None)
    fallback_target = ""
    if active_session is not None:
        fallback_target = str(getattr(active_session, "write_target_path", "") or "").strip()
    target_path = str(
        pending.args.get("path")
        or primary_task_target_path(harness)
        or fallback_target
    ).strip()
    session = _active_write_session_for_target(harness, target_path) if target_path else None
    if session is None:
        session = getattr(getattr(harness, "state", None), "write_session", None)
        if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
            session = None
    if session is None:
        return None

    payload = getattr(getattr(harness, "state", None), "scratchpad", {}).get("_last_incomplete_tool_call")
    if not isinstance(payload, dict):
        return None
    raw_calls = payload.get("partial_tool_calls_raw")
    if not isinstance(raw_calls, list) or not raw_calls:
        return None

    from ..tools.fs import _same_target_path

    for item in reversed(raw_calls):
        candidate = PendingToolCall.from_payload(item)
        if candidate is None or candidate.tool_name not in {"file_write", "file_append"}:
            continue
        candidate_content = candidate.args.get("content")
        if candidate_content is None or not str(candidate_content).strip():
            continue
        candidate_path = str(candidate.args.get("path") or "").strip()
        if candidate_path and not _same_target_path(session.write_target_path, candidate_path, getattr(harness.state, "cwd", None)):
            continue

        repaired_args: dict[str, Any] = {
            "path": candidate_path or session.write_target_path,
            "content": str(candidate_content),
            "write_session_id": session.write_session_id,
            "section_name": str(
                candidate.args.get("section_name")
                or candidate.args.get("section_id")
                or session.write_next_section
                or session.write_current_section
                or "imports"
            ).strip(),
        }
        for key in (
            "section_id",
            "section_role",
            "next_section_name",
            "replace_strategy",
            "expected_followup_verifier",
        ):
            value = candidate.args.get(key)
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            repaired_args[key] = value

        return PendingToolCall(
            tool_name="file_write",
            args=repaired_args,
            tool_call_id=pending.tool_call_id,
            raw_arguments=json.dumps(repaired_args, ensure_ascii=True, sort_keys=True),
        )
    return None


def _detect_oversize_write_payload(
    harness: Any,
    pending: PendingToolCall,
) -> tuple[str, dict[str, Any]] | None:
    if pending.tool_name != "file_write":
        return None

    model_name = getattr(getattr(harness, "client", None), "model", None)
    from ..guards import is_small_model_name
    is_small = is_small_model_name(model_name)
    
    if not is_small:
        return None

    content = str(pending.args.get("content", ""))
    payload_size = len(content)
    write_session = getattr(harness.state, "write_session", None)
    
    # Hard Gate: If a session is active for this file, you MUST use the session ID.
    if write_session and str(getattr(write_session, "status", "")).strip().lower() != "complete":
        from ..tools.fs import _resolve
        try:
            target_path = _resolve(str(pending.args.get("path") or ""), getattr(harness.state, "cwd", None))
            session_path = _resolve(write_session.write_target_path, getattr(harness.state, "cwd", None))
        except Exception:
            target_path = None
            session_path = None
        if target_path == session_path and (
            not pending.args.get("write_session_id")
            or pending.args.get("write_session_id") != write_session.write_session_id
        ):
            message = (
                f"A write session `{write_session.write_session_id}` is already active for `{write_session.write_target_path}`. "
                "You must include the correct `write_session_id` and section metadata to continue. "
                "Do not attempt to overwrite the file directly during an active session."
            )
            return message, {
                "tool_name": pending.tool_name,
                "tool_call_id": pending.tool_call_id,
                "reason": "session_context_missing",
                "active_session_id": write_session.write_session_id,
            }

    threshold = _write_policy_value(harness, "small_model_hard_write_chars", 4000)
    if payload_size > threshold and not (write_session and pending.args.get("write_session_id")):
        message = (
            f"Write payload for `{pending.tool_name}` exceeds the hard limit of {threshold} characters ({payload_size} chars). "
            "Please use chunked write mode or break your edit into smaller pieces."
        )
        return message, {
            "tool_name": pending.tool_name,
            "tool_call_id": pending.tool_call_id,
            "size": payload_size,
            "threshold": threshold,
            "reason": "payload_too_large",
        }

    return None


def _build_schema_repair_message(
    harness: Any,
    pending: PendingToolCall,
    required_fields: list[Any],
) -> str:
    field_names = [str(field) for field in required_fields if str(field).strip()]
    required_text = ", ".join(field_names) or "path, content"
    if pending.tool_name in {"file_write", "file_append"}:
        target_path = str(pending.args.get("path") or primary_task_target_path(harness) or "").strip()
        target_hint = f" Target path for this task: `{target_path}`." if target_path else ""
        session = _active_write_session_for_target(harness, target_path)
        if session is None:
            session = getattr(getattr(harness, "state", None), "write_session", None)
            if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
                session = None
        if session is not None:
            section_name = session.write_next_section or session.write_current_section or "imports"
            next_hint = (
                f" Resume with section `{section_name}` and include `next_section_name='...'` if more sections remain."
            )
            if session.write_sections_completed and not session.write_next_section:
                next_hint = " Omit `next_section_name` on the final chunk to finalize the session."
            return (
                f"Tool call '{pending.tool_name}' was emitted without arguments. "
                f"Continue Write Session `{session.write_session_id}` for `{session.write_target_path}`. "
                f"Resend `file_write` with these required fields: {required_text}, plus "
                f"`write_session_id='{session.write_session_id}'` and `section_name='{section_name}'`."
                f"{next_hint} Do not switch back to `file_append` or `file_read` unless you truly need local context for a repair."
            )
        return (
            f"Tool call '{pending.tool_name}' was emitted without arguments. "
            f"Please resend the tool call with these required fields: {required_text}."
            f"{target_hint} "
            "If a full implementation is too large, break it down with a small valid scaffold first, "
            "then extend it with later writes."
        )
    return (
        f"Tool call '{pending.tool_name}' was emitted without arguments. "
        f"Please resend the tool call with these required fields: {required_text}."
    )


def _fallback_repeated_artifact_read(harness: Any, pending: PendingToolCall) -> PendingToolCall | None:
    if pending.tool_name != "artifact_read":
        return None

    artifact_id = _extract_artifact_id_from_args(pending.args)
    if not artifact_id:
        return None

    artifact = _resolve_artifact_record(harness, artifact_id)
    if artifact is None:
        return None

    content = _read_artifact_text(artifact)
    if not content:
        return None

    # Keep this escape hatch narrow: only recover scan-like artifacts with a concrete grep target.
    query = _choose_artifact_grep_query(content)
    if not query:
        return None

    return PendingToolCall(
        tool_name="artifact_grep",
        args={
            "artifact_id": artifact.artifact_id,
            "query": query,
        },
        raw_arguments=json.dumps(
            {"artifact_id": artifact.artifact_id, "query": query},
            ensure_ascii=True,
            sort_keys=True,
        ),
        tool_call_id=pending.tool_call_id,
    )


def _fallback_repeated_file_read(harness: Any, pending: PendingToolCall) -> PendingToolCall | None:
    if pending.tool_name != "file_read":
        return None

    candidate_path = _resolve_file_read_path(harness, pending.args)
    if candidate_path is None:
        return None

    artifact = _find_full_file_artifact_for_path(harness, candidate_path)
    if artifact is None:
        return None

    recovered_args: dict[str, Any] = {"artifact_id": artifact.artifact_id}
    start_line, end_line = _requested_file_read_range(pending.args)
    if start_line is not None:
        recovered_args["start_line"] = start_line
    if end_line is not None:
        recovered_args["end_line"] = end_line
    if start_line is None and end_line is None:
        recovered_args["start_line"] = 1

    return PendingToolCall(
        tool_name="artifact_read",
        args=recovered_args,
        raw_arguments=json.dumps(recovered_args, ensure_ascii=True, sort_keys=True),
        tool_call_id=pending.tool_call_id,
    )


def _find_full_file_artifact_for_path(harness: Any, target_path: Path) -> Any | None:
    artifacts = getattr(getattr(harness, "state", None), "artifacts", {})
    if not isinstance(artifacts, dict) or not artifacts:
        return None

    for _, artifact in reversed(list(artifacts.items())):
        if getattr(artifact, "kind", "") != "file_read":
            continue
        metadata = getattr(artifact, "metadata", {})
        if not isinstance(metadata, dict) or not metadata.get("complete_file"):
            continue
        artifact_path = _resolve_file_read_path(
            harness,
            {"path": metadata.get("path") or getattr(artifact, "source", "")},
        )
        if artifact_path is None:
            continue
        if artifact_path == target_path:
            return artifact
    return None


def _artifact_read_recovery_hint(harness: Any, guard_error: str) -> tuple[str, str] | None:
    if "artifact_read" not in guard_error and "max_consecutive_errors" not in guard_error:
        return None

    if "max_consecutive_errors" in guard_error:
        recent_errors = getattr(harness.state, "recent_errors", [])
        if not recent_errors or not all("artifact_read" in str(err) for err in recent_errors):
            return None

    history = getattr(harness.state, "tool_history", [])
    if not isinstance(history, list) or not history:
        return None

    for fingerprint in reversed(history):
        if not isinstance(fingerprint, str) or not fingerprint.startswith("artifact_read|"):
            continue
        parts = fingerprint.split("|", 2)
        if len(parts) < 2:
            continue
        try:
            args = json.loads(parts[1])
        except Exception:
            continue
        if not isinstance(args, dict):
            continue
        recovered = _fallback_repeated_artifact_read(
            harness,
            PendingToolCall(
                tool_name="artifact_read",
                args=args,
                raw_arguments=json.dumps(args, ensure_ascii=True, sort_keys=True),
            ),
        )
        if recovered is None:
            continue
        artifact_id = str(recovered.args.get("artifact_id", "")).strip()
        query = str(recovered.args.get("query", "")).strip()
        if artifact_id and query:
            return artifact_id, query
    return None


def _artifact_read_synthesis_hint(harness: Any, guard_error: str) -> str | None:
    """Return the artifact id that should be synthesized from, if recovery already happened once."""
    if "artifact_read" not in guard_error and "max_consecutive_errors" not in guard_error:
        return None

    if "max_consecutive_errors" in guard_error:
        recent_errors = getattr(harness.state, "recent_errors", [])
        if not recent_errors or not all("artifact_read" in str(err) for err in recent_errors):
            return None

    history = _tool_attempt_history(harness)
    if not history:
        return None

    read_counts: dict[str, int] = {}
    grep_seen: set[str] = set()

    for item in history:
        tool_name = str(item.get("tool_name", ""))
        fingerprint = str(item.get("fingerprint", ""))
        if not tool_name or not fingerprint:
            continue
        try:
            payload = json.loads(fingerprint)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        args = payload.get("args", {})
        if not isinstance(args, dict):
            continue
        artifact_id = str(args.get("artifact_id", "")).strip()
        if not artifact_id:
            continue
        if tool_name == "artifact_read":
            read_counts[artifact_id] = read_counts.get(artifact_id, 0) + 1
        elif tool_name == "artifact_grep":
            grep_seen.add(artifact_id)

    for item in reversed(history):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        fingerprint = str(item.get("fingerprint", ""))
        if not fingerprint:
            continue
        try:
            payload = json.loads(fingerprint)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        args = payload.get("args", {})
        if not isinstance(args, dict):
            continue
        artifact_id = str(args.get("artifact_id", "")).strip()
        if not artifact_id:
            continue
        if read_counts.get(artifact_id, 0) >= 3 and artifact_id in grep_seen:
            return artifact_id

    return None


def _extract_artifact_id_from_args(args: dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return None

    for key in ("artifact_id", "path", "id"):
        value = args.get(key)
        if not isinstance(value, str):
            continue
        candidate = Path(value.strip()).stem.strip()
        if candidate:
            return candidate
    return None


def _resolve_artifact_record(harness: Any, artifact_id: str) -> Any | None:
    artifact = harness.state.artifacts.get(artifact_id)
    if artifact is not None:
        return artifact

    if not artifact_id.startswith("A"):
        return None

    try:
        numeric_val = int(artifact_id[1:])
    except ValueError:
        return None

    for aid, record in harness.state.artifacts.items():
        if not isinstance(aid, str) or not aid.startswith("A"):
            continue
        try:
            if int(aid[1:]) == numeric_val:
                return record
        except ValueError:
            continue
    return None


def _read_artifact_text(artifact: Any) -> str:
    content_path = getattr(artifact, "content_path", None)
    if isinstance(content_path, str) and content_path.strip():
        path = Path(content_path)
        if path.exists():
            try:
                return path.read_text(encoding="utf-8")
            except Exception:
                pass

    inline_content = getattr(artifact, "inline_content", None)
    if isinstance(inline_content, str) and inline_content:
        return inline_content
    return ""


def _should_suppress_resolved_plan_artifact_read(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "artifact_read":
        return False
    if not getattr(harness.state, "plan_resolved", False):
        return False
    plan_artifact_id = str(getattr(harness.state, "plan_artifact_id", "") or "").strip()
    if not plan_artifact_id:
        return False
    artifact_id = _extract_artifact_id_from_args(pending.args)
    if artifact_id != plan_artifact_id:
        return False
    return bool(harness.state.active_plan or harness.state.draft_plan or harness.state.working_memory.plan)


def _choose_artifact_grep_query(content: str) -> str | None:
    lowered = content.lower()
    if not any(marker in lowered for marker in ("nmap scan report", "/tcp", "/udp", "host is up")):
        return None
    for query in ("open", "port", "service", "banner", "nmap scan report", "host is up"):
        if query in lowered:
            return query
    return None


def _clear_artifact_read_guard_state(harness: Any, artifact_id: str) -> None:
    if not artifact_id:
        return

    recent_errors = getattr(harness.state, "recent_errors", None)
    if isinstance(recent_errors, list) and recent_errors:
        filtered_errors = [err for err in recent_errors if "artifact_read" not in str(err)]
        if len(filtered_errors) != len(recent_errors):
            harness.state.recent_errors = filtered_errors

    tool_history = getattr(harness.state, "tool_history", None)
    if isinstance(tool_history, list) and tool_history:
        kept_history: list[str] = []
        removed_entries = 0
        for entry in tool_history:
            if not isinstance(entry, str) or not entry.startswith("artifact_read|"):
                kept_history.append(entry)
                continue
            parts = entry.split("|", 2)
            if len(parts) < 3:
                kept_history.append(entry)
                continue
            try:
                args = json.loads(parts[1])
            except Exception:
                kept_history.append(entry)
                continue
            if isinstance(args, dict) and str(args.get("artifact_id", "")).strip() == artifact_id:
                removed_entries += 1
                continue
            kept_history.append(entry)
        if removed_entries:
            harness.state.tool_history = kept_history

    _clear_tool_attempt_history(harness)


def _record_tool_attempt(harness: Any, pending: PendingToolCall) -> None:
    history = _tool_attempt_history(harness)
    history.append(
        {
            "tool_name": pending.tool_name,
            "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args),
        }
    )
    harness.state.scratchpad["_tool_attempt_history"] = history[-_REPEATED_TOOL_HISTORY_LIMIT:]


def _clear_tool_attempt_history(harness: Any) -> None:
    harness.state.scratchpad.pop("_tool_attempt_history", None)


def _tool_attempt_history(harness: Any) -> list[dict[str, str]]:
    history = harness.state.scratchpad.get("_tool_attempt_history")
    if not isinstance(history, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in history[-_REPEATED_TOOL_HISTORY_LIMIT:]:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "tool_name": str(item.get("tool_name", "")),
                "fingerprint": str(item.get("fingerprint", "")),
            }
        )
    return normalized


def _tool_call_fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    normalized_args = _normalize_tool_args(tool_name, args)
    return json.dumps({"tool_name": tool_name, "args": normalized_args}, sort_keys=True, ensure_ascii=True)


def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> Any:
    if not isinstance(args, dict):
        return {}
    normalized = json_safe_value(args)
    if not isinstance(normalized, dict):
        return {}
    if tool_name == "shell_exec":
        command = normalized.get("command")
        if command is not None:
            normalized["command"] = _normalize_shell_command(str(command))
    return _normalize_json_like(normalized)


def _normalize_shell_command(command: str) -> str:
    parts = command.strip().split()
    if not parts:
        return ""
    normalized_parts = [_normalize_path_token(part) for part in parts]
    return " ".join(normalized_parts)


def _normalize_path_token(value: str) -> str:
    stripped = value.strip()
    if len(stripped) > 1 and stripped.endswith("/") and "/" in stripped[:-1]:
        return stripped.rstrip("/")
    return stripped


def _normalize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_like(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, str):
        collapsed = " ".join(value.strip().split())
        return _normalize_path_token(collapsed)
    return json_safe_value(value)


# ---------------------------------------------------------------------------
# High-level parse entry point
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field

@dataclass
class ToolCallParseResult:
    pending_tool_calls: list[PendingToolCall] = field(default_factory=list)
    final_assistant_text: str = ""
    cleaned_text: str = ""


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

    if not final_assistant_text.strip() and not pending_calls:
        final_assistant_text = _clean_reasoning_fallback_text(stream.thinking_text)
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
        cleaned_text=cleaned_text,
    )


def _should_enter_chunk_mode(harness: Any, pending: PendingToolCall) -> bool:
    """Determine if a write should be redirected to chunked mode."""
    if pending.tool_name != "file_write":
        return False
    
    # Already in a session means we are probably already chunking
    if getattr(harness.state, "write_session", None):
        return False

    from ..guards import is_small_model_name
    model_name = getattr(getattr(harness, "client", None), "model", None)
    if not is_small_model_name(model_name):
        return False

    path = str(pending.args.get("path") or "").strip()
    if _task_forces_chunk_mode(harness, path):
        return True

    content = str(pending.args.get("content", ""))
    payload_size = len(content)

    # Signal 1: Payload exceeds soft threshold
    if payload_size >= _write_policy_value(harness, "small_model_soft_write_chars", 2000):
        return True

    # Signal 2: New file and it's likely to be medium/large based on plan (if available)
    if _write_policy_value(harness, "chunk_mode_new_file_only", True) and path:
        from ..tools.fs import _resolve
        target = _resolve(path, harness.state.cwd)
        if not target.exists():
            # Heuristic: if it's a new file and we are on a small model,
            # be conservative and enter chunk mode if it's > line estimate
            lines = content.count("\n") + 1
            if lines >= _write_policy_value(harness, "new_file_chunk_mode_line_estimate", 100):
                return True

    return False
