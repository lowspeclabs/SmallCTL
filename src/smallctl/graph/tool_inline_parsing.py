from __future__ import annotations

import json
import re
from typing import Any

from .state import PendingToolCall
from .tool_model_rules import _parse_raw_function_call, _strip_exact_small_gemma_4_protocol_noise


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
        name = str(data.get("name", data.get("tool_name", data.get("tool", data.get("action", ""))))).strip()
        if not name:
            return None
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

    xml_patterns = [
        r"<tool_code>(.*?)</tool_code>",
        r"<tool_call>(.*?)</tool_call>",
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
                cleaned_text = cleaned_text[:start - offset] + cleaned_text[end - offset :]
                offset += end - start
        except Exception:
            pass

    if "{" in cleaned_text:
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
                break

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
