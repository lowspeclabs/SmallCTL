from __future__ import annotations

import json
from typing import Any

from ..context.messages import _request_has_full_artifact_intent, render_dir_list_result
from ..context.rendering import render_shell_output
from ..state import clip_text_value, json_safe_value

_UI_TOOL_RESULT_PREVIEW_LIMIT = 1200
_UI_ARTIFACT_READ_PREVIEW_LIMIT = 900
_DIR_LIST_PREVIEW_ENTRY_LIMIT = 50


def _trim_head_clip_boundary(text: str, *, scan: int = 32) -> str:
    trimmed = text.rstrip()
    if not trimmed:
        return trimmed
    window_start = max(0, len(trimmed) - scan)
    boundary = max(
        trimmed.rfind(" ", window_start),
        trimmed.rfind("\n", window_start),
        trimmed.rfind("\t", window_start),
    )
    if boundary > 0:
        candidate = trimmed[:boundary].rstrip()
        if candidate:
            return candidate
    return trimmed


def _trim_tail_clip_boundary(text: str, *, scan: int = 32) -> str:
    trimmed = text.lstrip()
    if not trimmed:
        return trimmed
    for index, char in enumerate(trimmed[:scan]):
        if char.isspace():
            candidate = trimmed[index + 1 :].lstrip()
            if candidate:
                return candidate
            break
    return trimmed


def _clip_error_display(text: str, *, limit: int) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return "Tool failed."
    if len(normalized) <= limit:
        return normalized

    # Preserve both the front context and the actionable tail of long errors.
    # Bias toward the tail because recovery instructions often appear there.
    separator = "\n...\n"
    available = max(1, limit - len(separator))
    head_budget = max(1, int(available * 0.35))
    tail_budget = max(1, available - head_budget)

    head = _trim_head_clip_boundary(normalized[:head_budget])
    tail = _trim_tail_clip_boundary(normalized[-tail_budget:])
    if not head:
        return tail
    if not tail:
        return head
    return f"{head}{separator}{tail}"


def format_tool_result_display(
    *,
    tool_name: str,
    result: Any,
    request_text: str | None = None,
) -> str:
    from ..models.tool_result import ToolEnvelope

    if not isinstance(result, ToolEnvelope):
        return str(result)

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not result.success:
        return _clip_error_display(str(result.error or "Tool failed."), limit=400)

    if tool_name == "artifact_read":
        return format_artifact_read_display(result=result, request_text=request_text)
    if tool_name == "dir_list":
        return format_dir_list_display(result=result)

    output = result.output
    if isinstance(output, str):
        preview, clipped = clip_text_value(output.strip(), limit=_UI_TOOL_RESULT_PREVIEW_LIMIT)
        lines: list[str] = []
        path = metadata.get("path")
        if isinstance(path, str) and path.strip():
            lines.append(path.strip())
        if preview:
            lines.append(preview)
        text = "\n".join(lines) if lines else "ok"
        if clipped:
            text = f"{text}\n... output truncated"
        return text

    if isinstance(output, dict):
        # 1. Shell-like output (stdout, stderr, exit_code)
        if "stdout" in output or "stderr" in output:
            return format_shell_output_display(output=output)

        # 2. Generic message-based output
        msg = output.get("message") or output.get("output") or output.get("text") or output.get("question")
        if isinstance(msg, str) and msg.strip():
            rendered = msg.strip()
        else:
            # Fallback to full JSON if no obvious single text field found
            rendered = json.dumps(json_safe_value(output), ensure_ascii=True, default=str, indent=2)

        preview, clipped = clip_text_value(rendered, limit=_UI_TOOL_RESULT_PREVIEW_LIMIT)
        if clipped:
            return f"{preview}\n... output truncated"
        return preview or "ok"

    if output is None:
        return "ok"

    rendered = json.dumps(json_safe_value(output), ensure_ascii=True, default=str, indent=2)
    preview, clipped = clip_text_value(rendered, limit=_UI_TOOL_RESULT_PREVIEW_LIMIT)
    if clipped:
        return f"{preview}\n... output truncated"
    return preview or "ok"


def format_dir_list_display(*, result: Any) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    output = result.output
    if not isinstance(output, list):
        return "directory listed"

    text = render_dir_list_result(
        output,
        metadata=metadata,
        max_depth=2,
        max_children=_DIR_LIST_PREVIEW_ENTRY_LIMIT,
        max_items=_DIR_LIST_PREVIEW_ENTRY_LIMIT,
    )
    preview, clipped = clip_text_value(text, limit=420)
    if clipped:
        return f"{preview}\n... output truncated"
    return preview


def format_artifact_read_display(
    *,
    result: Any,
    request_text: str | None = None,
) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    header_parts: list[str] = []
    full_request = _request_has_full_artifact_intent(request_text)

    artifact_id = metadata.get("artifact_id")
    if isinstance(artifact_id, str) and artifact_id.strip():
        header_parts.append(artifact_id.strip())

    path = metadata.get("path")
    if isinstance(path, str) and path.strip():
        header_parts.append(path.strip())

    line_start = metadata.get("line_start")
    line_end = metadata.get("line_end")
    total_lines = metadata.get("total_lines")
    if isinstance(line_start, int) and isinstance(line_end, int):
        if isinstance(total_lines, int) and total_lines > 0:
            header_parts.append(f"lines {line_start}-{line_end} of {total_lines}")
        else:
            header_parts.append(f"lines {line_start}-{line_end}")
    output = result.output
    preview_source = output if isinstance(output, str) else json.dumps(
        json_safe_value(output),
        ensure_ascii=True,
        default=str,
        indent=2,
    )
    preview, clipped = clip_text_value(
        preview_source,
        limit=_UI_ARTIFACT_READ_PREVIEW_LIMIT
    )

    lines: list[str] = []
    if header_parts:
        lines.append(" | ".join(header_parts))
    if preview:
        lines.append(preview)

    if metadata.get("truncated") or clipped:
        if metadata.get("truncated"):
            if isinstance(line_end, int) and isinstance(total_lines, int) and line_end < total_lines:
                next_start = line_end + 1
                lines.append(
                    f"... continue at start_line={next_start} via "
                    f"artifact_read(artifact_id='{artifact_id}', start_line={next_start})"
                )
            else:
                lines.append("... more available via artifact_read(start_line=..., end_line=..., max_chars=...)")
        elif full_request:
            lines.append("... preview clipped in UI")
        else:
            lines.append("... more available via artifact_read(start_line=..., end_line=..., max_chars=...)")

    return "\n\n".join(lines) if lines else "artifact read complete"


def format_shell_output_display(*, output: dict[str, Any]) -> str:
    return render_shell_output(
        output,
        preview_limit=_UI_TOOL_RESULT_PREVIEW_LIMIT,
        strip_whitespace=True,
    )
