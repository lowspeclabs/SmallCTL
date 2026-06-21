from __future__ import annotations

import ast
import json
import re
from typing import Any

from ..state import json_safe_value
from .state import PendingToolCall
from .tool_model_rules import (
    _model_is_lfm25_8b_a1b,
    _parse_raw_function_call,
    _raw_function_call_pattern,
    _strip_exact_small_gemma_4_protocol_noise,
)


_INLINE_TOOL_SCHEMA_KEYS = {
    "function",
    "name",
    "tool_name",
    "tool",
    "tool_call",
    "action",
    "arguments",
    "args",
    "params",
    "parameters",
}


def _inline_json_extra_fields(data: dict[str, Any]) -> dict[str, Any]:
    extras = {
        str(key): value
        for key, value in data.items()
        if str(key) not in _INLINE_TOOL_SCHEMA_KEYS
    }
    safe = json_safe_value(extras)
    return safe if isinstance(safe, dict) else {}


def _try_parse_data(data: Any) -> PendingToolCall | None:
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("function"), dict):
        pending = PendingToolCall.from_payload(data)
        if pending is not None:
            extra_fields = _inline_json_extra_fields(data)
            if extra_fields:
                pending.parser_metadata["inline_json_extra_fields"] = extra_fields
            return pending

    def _pick_name() -> str:
        for key in ("name", "tool_name", "tool_call", "tool", "action"):
            if key in data:
                return str(data[key]).strip()
        return ""

    name = _pick_name()
    if not name:
        return None

    # Determine the arguments container. Some small models emit a flat object
    # such as {"tool_call": "artifact_read", "artifact_id": "A0001"} where
    # every non-name key is an argument.
    explicit_args_keys = ("arguments", "args", "params", "parameters")
    args: Any = None
    for key in explicit_args_keys:
        if key in data:
            args = data[key]
            break
    if (
        args is None
        or not isinstance(args, dict)
        or ("tool_call" in data and not any(k in data for k in explicit_args_keys))
    ):
        inferred_args = {
            str(key): value
            for key, value in data.items()
            if str(key) not in _INLINE_TOOL_SCHEMA_KEYS and str(key) != name
        }
        if inferred_args:
            args = inferred_args
    if args is None:
        args = {}
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
    pending = PendingToolCall.from_payload(payload)
    if pending is not None:
        arg_keys = set(pending.args.keys()) if isinstance(pending.args, dict) else set()
        extra_fields = {
            str(key): value
            for key, value in data.items()
            if str(key) not in _INLINE_TOOL_SCHEMA_KEYS and str(key) not in arg_keys
        }
        safe_extras = json_safe_value(extra_fields)
        if isinstance(safe_extras, dict) and safe_extras:
            pending.parser_metadata["inline_json_extra_fields"] = safe_extras
    return pending


def _try_recover_truncated_inline_json(text: str, start: int) -> dict[str, Any] | None:
    """Attempt to recover a truncated inline JSON tool call by adding closing braces.

    Only recovers when the prefix clearly looks like an inline tool call
    (contains tool/name/action keys and an arguments container) and adding a
    modest number of closing braces yields valid JSON. This salvages tool calls
    that were halted mid-stream by a repetition-loop guard.
    """
    prefix = text[start:]
    lowered_prefix = prefix.lower()
    has_tool_key = any(key in lowered_prefix for key in ('"tool_name"', '"name"', '"action"', '"tool"', '"tool_call"', '"function"'))
    # For flat {"tool_call": "name", ...args} objects, argument keys are not
    # wrapped in an explicit arguments container. Presence of "tool_call" is
    # enough to attempt recovery.
    has_args_key = (
        any(key in lowered_prefix for key in ('"arguments"', '"args"', '"params"', '"parameters"'))
        or '"tool_call"' in lowered_prefix
    )
    if not has_tool_key or not has_args_key:
        return None
    # The model may have been halted inside a string value (e.g. while
    # repeating a word inside a command), so try closing unclosed quotes too.
    for extra_quotes in range(3):
        for extra_braces in range(1, 5):
            try:
                candidate = prefix + ('"' * extra_quotes) + ("}" * extra_braces)
                data = json.loads(candidate)
            except Exception:
                continue
            if _try_parse_data(data):
                return data
    return None


def _extract_orphan_parameter_payload(text: str) -> dict[str, str]:
    if not text:
        return {}
    if "<parameter" not in text.lower():
        return {}
    matches = re.findall(
        r"<parameter(?:=([\w_-]+)|\s+name=['\"]?([\w_-]+)['\"]?)>(.*?)</parameter>",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        return {}
    params: dict[str, str] = {}
    for key_a, key_b, value in matches:
        key = (key_a or key_b or "").strip()
        if not key:
            continue
        params[key] = str(value or "").strip()
    return params


def _infer_tool_name_from_orphan_parameters(
    params: dict[str, str],
    *,
    allowed_raw_function_names: set[str] | None,
) -> str:
    if not isinstance(params, dict) or not params:
        return ""
    keys = {str(key).strip().lower() for key in params}
    if "command" not in keys:
        return ""

    def _tool_allowed(name: str) -> bool:
        if allowed_raw_function_names is None:
            return True
        return name in allowed_raw_function_names

    remote_keys = {
        "host",
        "target",
        "user",
        "username",
        "password",
        "port",
        "identity_file",
    }
    if keys.intersection(remote_keys) and _tool_allowed("ssh_exec"):
        return "ssh_exec"

    shell_keys = {"command", "job_id", "background", "timeout_sec"}
    if keys.issubset(shell_keys) and _tool_allowed("shell_exec"):
        return "shell_exec"

    return ""


def _normalize_gemma_quote_tokens(text: str) -> str:
    """Replace Gemma-4-e2b-it quote control tokens with regular quotes."""
    return str(text or "").replace('<|"|>', '"')


def _parse_gemma_colon_brace_call(
    block_text: str,
    *,
    allowed_raw_function_names: set[str] | None = None,
) -> PendingToolCall | None:
    """Parse Gemma-4-e2b-it `call:tool_name{key: "value", ...}` syntax.

    Some Gemma checkpoints emit tool calls wrapped in `<tool_call>` tags using a
    colon-prefixed, brace-wrapped format with `<|"|>` quote tokens, e.g.:

        <tool_call>call:ssh_exec{command:<|"|>ls<|"|>,host:<|"|>1.2.3.4<|"|>}<tool_call|>

    This parser normalizes the quote tokens, strips the optional `call:` prefix,
    and extracts key/value pairs separated by commas.
    """
    candidate = _normalize_gemma_quote_tokens(block_text).strip()
    if not candidate:
        return None

    if candidate.lower().startswith("call:"):
        candidate = candidate[5:].strip()

    match = re.match(
        r"^\s*([a-zA-Z0-9_-]+)\s*\{\s*(.*?)\s*\}\s*$",
        candidate,
        re.DOTALL,
    )
    if not match:
        return None

    tool_name = match.group(1).strip()
    if allowed_raw_function_names is not None and tool_name not in allowed_raw_function_names:
        return None

    body = match.group(2)
    # Extract key:value pairs, respecting single/double quoted values and
    # tolerating unquoted scalar values. Values may contain shell metacharacters
    # including commas, so comma-splitting must stay inside quotes.
    params: dict[str, str] = {}
    kv_pattern = re.compile(
        r"([a-zA-Z0-9_-]+)\s*:\s*"
        r'("(?:[^"\\]|\\.)*"|'
        r"'(?:[^'\\]|\\.)*'|"
        r"[^,}]+)"
    )
    for key, value in kv_pattern.findall(body):
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            try:
                value = ast.literal_eval(value)
            except Exception:
                value = value[1:-1]
        params[key] = str(value)

    if not params:
        return None

    return PendingToolCall(
        tool_name=tool_name,
        args=params,
        raw_arguments=json.dumps(params, ensure_ascii=True, sort_keys=True),
    )


def _extract_inline_tool_calls(
    text: str,
    *,
    model_name: str | None = None,
    allowed_raw_function_names: set[str] | None = None,
) -> tuple[str, list[PendingToolCall]]:
    if not text:
        return "", []

    results: list[PendingToolCall] = []
    cleaned_text = _normalize_gemma_quote_tokens(
        _strip_exact_small_gemma_4_protocol_noise(
            text,
            model_name=model_name,
        )
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

        # Gemma-style self-closing call tag: <call:tool_name key="value" />
        call_match = re.match(
            r"^\s*<call:([\w_-]+)\s+([^>]*)/?>\s*$",
            block_text,
            re.DOTALL,
        )
        if call_match:
            tool_name = call_match.group(1).strip()
            attr_text = call_match.group(2).strip()
            params: dict[str, str] = {}
            for key, _quote, value in re.findall(
                r'([\w_-]+)\s*=\s*(["\'])(.*?)\2',
                attr_text,
            ):
                params[key] = value
            if params:
                return PendingToolCall(
                    tool_name=tool_name,
                    args=params,
                    raw_arguments=json.dumps(params, ensure_ascii=True, sort_keys=True),
                )

        # Gemma-4-e2b-it `call:tool_name{key: "value", ...}` / `tool_name{...}` syntax
        gemma_brace_call = _parse_gemma_colon_brace_call(
            block_text,
            allowed_raw_function_names=allowed_raw_function_names,
        )
        if gemma_brace_call is not None:
            return gemma_brace_call

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
            # Fallback: small models sometimes use raw parameter tags instead of <parameter> wrappers
            if not params and tool_name in {"file_patch", "ssh_file_patch", "ast_patch"}:
                _RAW_TAG_ALIASES: dict[str, list[str]] = {
                    "target_text": ["target_text", "source", "old_text", "old"],
                    "replacement_text": ["replacement_text", "dest", "new_text", "new", "replacement"],
                    "path": ["path"],
                    "expected_occurrences": ["expected_occurrences"],
                    "write_session_id": ["write_session_id", "session_id"],
                }
                for canonical, aliases in _RAW_TAG_ALIASES.items():
                    if canonical in params:
                        continue
                    for alias in aliases:
                        match = re.search(
                            re.escape(f"<{alias}>") + r"(.*?)" + re.escape(f"</{alias}>"),
                            inner_content,
                            re.DOTALL | re.IGNORECASE,
                        )
                        if match:
                            params[canonical] = match.group(1).strip()
                            break
            if params:
                return PendingToolCall(
                    tool_name=tool_name,
                    args=params,
                    raw_arguments=json.dumps(params, ensure_ascii=True, sort_keys=True),
                )
        return None

    def _parse_direct_xml_tool_tag(block_text: str) -> PendingToolCall | None:
        # Direct self-closing XML tool tag: <tool_name key="value" />
        # Emitted by some Gemma models instead of <call:tool_name ... />.
        match = re.match(
            r"^\s*<([\w_-]+)\s+([^>]*)/?>\s*$",
            block_text,
            re.DOTALL,
        )
        if not match:
            return None
        tool_name = match.group(1).strip()
        if allowed_raw_function_names is not None and tool_name not in allowed_raw_function_names:
            return None
        attr_text = match.group(2).strip()
        params: dict[str, str] = {}
        for key, _quote, value in re.findall(
            r'([\w_-]+)\s*=\s*(["\'])(.*?)\2',
            attr_text,
        ):
            params[key] = value
        if not params:
            return None
        return PendingToolCall(
            tool_name=tool_name,
            args=params,
            raw_arguments=json.dumps(params, ensure_ascii=True, sort_keys=True),
        )

    def _try_parse_lfm_plan_data(data: Any) -> list[PendingToolCall]:
        if not _model_is_lfm25_8b_a1b(model_name) or not isinstance(data, dict):
            return []
        if not any(key in data for key in ("plan", "next_actions", "status_required", "next_step")):
            return []
        actions = data.get("next_actions")
        if not isinstance(actions, list):
            return []
        calls: list[PendingToolCall] = []
        for action in actions:
            pending = _try_parse_data(action)
            if pending is None or not pending.tool_name:
                continue
            if allowed_raw_function_names is not None and pending.tool_name not in allowed_raw_function_names:
                continue
            pending.parser_metadata["lfm_plan_json_recovered"] = True
            calls.append(pending)
        return calls

    def _try_strip_lfm_plan_json_object(start: int) -> bool:
        nonlocal cleaned_text
        brace_count = 0
        end = -1
        for i in range(start, len(cleaned_text)):
            if cleaned_text[i] == "{":
                brace_count += 1
            elif cleaned_text[i] == "}":
                brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
        if end == -1:
            return False
        try:
            data = json.loads(cleaned_text[start:end])
        except Exception:
            return False
        recovered = _try_parse_lfm_plan_data(data)
        if not recovered:
            return False
        results.extend(recovered)
        cleaned_text = cleaned_text[:start] + cleaned_text[end:]
        return True

    xml_patterns = [
        r"<tool_code>(.*?)</tool_code>",
        r"<tool_call>(.*?)</tool_call>",
        r"<\|tool_call>(.*?)</tool_call>",
        r"<tool_call>(.*?)<tool_call\|>",
        r"<\|tool_call>(.*?)<tool_call\|>",
        # Some small models emit an opening `<|tool_call>` tag but forget the
        # closing tag. Capture the rest of the line/string so the brace parser
        # below can still recover the call.
        r"<\|tool_call>([^\n]*?)(?:</tool_call>|<tool_call\|>|\n|$)",
        r"<call>(.*?)</call>",
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
                try:
                    data = json.loads(content)
                    pending = _try_parse_data(data)
                    if pending:
                        results.append(pending)
                        found = True
                except Exception:
                    pass

            if not found:
                pending = _parse_raw_function_call(
                    content,
                    model_name=model_name,
                    allowed_tool_names=allowed_raw_function_names,
                )
                if pending is not None:
                    results.append(pending)
                    found = True

            if found:
                start, end = match.span()
                cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
                offset += end - start

    # Gemma-style self-closing call tags: <call:tool_name key="value" />
    call_tag_matches = list(
        re.finditer(
            r"<call:([\w_-]+)\s+([^>]*)/?>",
            cleaned_text,
            re.DOTALL,
        )
    )
    offset = 0
    for match in call_tag_matches:
        pending = _parse_xml_function_block(match.group(0))
        if pending is not None:
            results.append(pending)
            start, end = match.span()
            cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
            offset += end - start

    # Direct self-closing XML tool tags: <tool_name key="value" />
    # Some Gemma variants emit the tool name directly as the tag name.
    if allowed_raw_function_names:
        direct_tag_pattern = re.compile(
            rf"<({'|'.join(re.escape(name) for name in sorted(allowed_raw_function_names))})\b[^>]*?/>",
            re.DOTALL,
        )
        direct_matches = list(direct_tag_pattern.finditer(cleaned_text))
        offset = 0
        for match in direct_matches:
            pending = _parse_direct_xml_tool_tag(match.group(0))
            if pending is not None:
                results.append(pending)
                start, end = match.span()
                cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
                offset += end - start

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
        cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
        offset += end - start

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
            cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
            offset += end - start

    json_blocks = list(re.finditer(r"```json\s*(.*?)\s*```", cleaned_text, re.DOTALL | re.IGNORECASE))
    offset = 0
    for match in json_blocks:
        block = match.group(1)
        try:
            data = json.loads(block)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        parsed_any = False
        for item in items:
            pending = _try_parse_data(item)
            if pending:
                results.append(pending)
                parsed_any = True
        if parsed_any:
            start, end = match.span()
            cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
            offset += end - start

    # Unlabeled markdown fences that contain JSON objects/arrays are also
    # emitted by some small models. Only strip them when the contents parse as
    # a tool call so that ordinary code blocks stay in the assistant text.
    unlabeled_blocks = list(re.finditer(r"```\s*\n(.*?)\n\s*```", cleaned_text, re.DOTALL))
    offset = 0
    for match in unlabeled_blocks:
        block = match.group(1).strip()
        if not block or block[0] not in {"{", "["}:
            continue
        try:
            data = json.loads(block)
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        parsed_any = False
        for item in items:
            pending = _try_parse_data(item)
            if pending:
                results.append(pending)
                parsed_any = True
        if parsed_any:
            start, end = match.span()
            cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
            offset += end - start

    if "{" in cleaned_text:
        start = cleaned_text.find("{")
        while start != -1 and _try_strip_lfm_plan_json_object(start):
            start = cleaned_text.find("{", start)
        start = cleaned_text.find("{")
        while start != -1:
            brace_count = 0
            end = -1
            for i in range(start, len(cleaned_text)):
                if cleaned_text[i] == "{":
                    brace_count += 1
                elif cleaned_text[i] == "}":
                    brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

            if end != -1:
                try:
                    data = json.loads(cleaned_text[start:end])
                    pending = _try_parse_data(data)
                    if pending:
                        results.append(pending)
                        cleaned_text = cleaned_text[:start] + cleaned_text[end:]
                        start = cleaned_text.find("{", start)
                    else:
                        start = cleaned_text.find("{", start + 1)
                except Exception:
                    start = cleaned_text.find("{", start + 1)
            else:
                # Unbalanced braces: the stream may have been halted mid-tool-call.
                # Try to recover a truncated inline JSON object that looks like a
                # tool call before giving up.
                recovered = _try_recover_truncated_inline_json(cleaned_text, start)
                if recovered:
                    pending = _try_parse_data(recovered)
                    if pending:
                        results.append(pending)
                        cleaned_text = cleaned_text[:start]
                        start = cleaned_text.find("{", start)
                        continue
                break

    # Some small models (e.g. Gemma-4-e2b-it) emit a raw function call at the
    # start of a line and then continue generating text on the same line. Try
    # to extract the call prefix without requiring the whole line to be only
    # the call. We still remove just the matched call portion; any trailing
    # duplicate text is handled by the caller.
    raw_call_regex = _raw_function_call_pattern(model_name=model_name)
    line_call_matches = list(
        re.finditer(rf"(?m)^[ \t]*{raw_call_regex}", cleaned_text)
    )
    offset = 0
    for match in line_call_matches:
        pending = _parse_raw_function_call(
            match.group(0),
            model_name=model_name,
            allowed_tool_names=allowed_raw_function_names,
        )
        if pending is None:
            continue
        pending = _try_parse_data({"tool_name": pending.tool_name, "arguments": pending.args})
        if pending:
            results.append(pending)
            start, end = match.span()
            cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
            offset += end - start

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
            cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
            offset += end - start

    if not results:
        lowered = cleaned_text.lower()
        maybe_orphan_tool_payload = (
            "<parameter" in lowered
            and ("</function>" in lowered or "</tool_call>" in lowered or "<tool_call" in lowered)
        )
        if maybe_orphan_tool_payload:
            orphan_params = _extract_orphan_parameter_payload(cleaned_text)
            inferred_tool_name = _infer_tool_name_from_orphan_parameters(
                orphan_params,
                allowed_raw_function_names=allowed_raw_function_names,
            )
            if inferred_tool_name:
                results.append(
                    PendingToolCall(
                        tool_name=inferred_tool_name,
                        args=orphan_params,
                        raw_arguments=json.dumps(orphan_params, ensure_ascii=True, sort_keys=True),
                    )
                )
                cleaned_text = re.sub(
                    r"</?tool_call>|</?function(?:=[^>]+)?>|<parameter(?:=[^>]+|\s+name=['\"]?[\w_-]+['\"]?)>.*?</parameter>",
                    "",
                    cleaned_text,
                    flags=re.IGNORECASE | re.DOTALL,
                ).strip()

    return cleaned_text, results
