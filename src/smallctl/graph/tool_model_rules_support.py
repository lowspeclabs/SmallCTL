from __future__ import annotations

import ast
import re
from typing import Any

from ..client.chunk_parser import extract_response_from_wrapper_tags, extract_thinking_from_tags
from .state import PendingToolCall


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


def _clean_reasoning_fallback_text(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(
        r"</?(?:tool_call|tool_code|call|function|parameter|thinking|reasoning|analysis|plan|execution|response)(?:=[^>]+)?>",
        "",
        str(text),
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _recover_reasoning_only_assistant_text(text: str) -> str:
    wrapped_response = _clean_reasoning_fallback_text(extract_response_from_wrapper_tags(text))
    if wrapped_response:
        return wrapped_response

    cleaned = _clean_reasoning_fallback_text(text)
    if not cleaned:
        return ""

    extracted = _extract_reasoning_final_output_candidate(cleaned)
    if extracted:
        return extracted

    if _looks_like_reasoning_trace(cleaned):
        return "The model returned reasoning text but no final answer."
    return cleaned


def _extract_reasoning_final_output_candidate(text: str) -> str:
    candidate_source = ""
    matches = list(
        re.finditer(
            r"(?is)(?:final output|final answer|final response|response|answer|output)\s*:\s*(.+)",
            str(text or ""),
        )
    )
    if matches:
        candidate_source = matches[-1].group(1).strip()
    else:
        candidate_source = str(text or "").strip()

    quoted = _extract_last_quoted_answer(candidate_source)
    if quoted:
        return quoted

    first_line = candidate_source.splitlines()[0].strip() if candidate_source else ""
    first_line = re.sub(r"\s*\([^)]*\)\s*$", "", first_line).strip()
    first_line = re.split(r"\s+(?:Actually|But|Or|Let's)\b", first_line, maxsplit=1)[0].strip()
    if _looks_like_public_answer(first_line):
        return first_line
    return ""


def _extract_last_quoted_answer(text: str) -> str:
    matches = list(re.finditer(r"(?P<quote>[\"'])(?P<body>[^\n]{1,240}?)(?P=quote)", str(text or "")))
    for match in reversed(matches):
        candidate = match.group("body").strip()
        if _looks_like_public_answer(candidate):
            return candidate
    return ""


def _looks_like_public_answer(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate or "\n" in candidate or len(candidate) > 240:
        return False
    if not re.search(r"[A-Za-z0-9]", candidate):
        return False
    lowered = candidate.lower()
    if any(token in lowered for token in ("thinking process", "final decision", "option a", "option b", "option c")):
        return False
    if re.match(r"^\d+\.\s", candidate):
        return False
    return True


def _looks_like_reasoning_trace(text: str) -> bool:
    lowered = str(text or "").lower()
    if any(
        marker in lowered
        for marker in (
            "thinking process",
            "analyze the request",
            "check constraints",
            "formulate response",
            "final decision",
            "refine for constraint",
            "final output",
            "option a",
            "option b",
            "option c",
            "let's pick",
        )
    ):
        return True
    return bool(
        re.search(r"(?m)^\s*\d+\.\s", str(text or ""))
        or re.search(r"(?m)^\s*[*-]\s+", str(text or ""))
    )


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
    markdown_wrappers = [
        ("**", "**"),
        ("__", "__"),
        ("`", "`"),
    ]
    stripped_wrapper = True
    while stripped_wrapper:
        stripped_wrapper = False
        for prefix, suffix in markdown_wrappers:
            if candidate_text.startswith(prefix) and candidate_text.endswith(suffix):
                inner = candidate_text[len(prefix) : len(candidate_text) - len(suffix)].strip()
                if inner:
                    candidate_text = inner
                    stripped_wrapper = True
                    break
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
        normalized_tool_name = re.sub(r"[^a-z0-9]+", "_", str(tool_name or "").strip().lower()).strip("_")
        if tool_name not in allowed_tool_names and normalized_tool_name not in {
            "tool_name",
            "function_name",
            "action_name",
            "tool",
            "function",
            "action",
            "name",
        }:
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
