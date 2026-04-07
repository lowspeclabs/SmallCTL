from __future__ import annotations

import json
from typing import Any

from ..context.messages import _request_has_full_artifact_intent, render_dir_list_tree
from ..context.rendering import render_shell_output
from ..state import clip_text_value, json_safe_value

_UI_TOOL_RESULT_PREVIEW_LIMIT = 1200
_UI_ARTIFACT_READ_PREVIEW_LIMIT = 900


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
        preview, clipped = clip_text_value(str(result.error or "Tool failed.").strip(), limit=400)
        if clipped:
            return f"{preview}\n... error truncated"
        return preview

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

    lines: list[str] = []
    path = metadata.get("path")
    count = metadata.get("count")
    if isinstance(path, str) and path.strip():
        if isinstance(count, int) and count >= 0:
            lines.append(f"{path.strip()} ({count} items)")
        else:
            lines.append(path.strip())
    elif isinstance(count, int) and count >= 0:
        lines.append(f"{count} items")

    preview_items = output[:8]
    tree_preview = render_dir_list_tree(preview_items, max_depth=2, max_children=8)
    if tree_preview:
        lines.append(tree_preview)

    remaining = len(output) - len(preview_items)
    if remaining > 0:
        lines.append(f"... {remaining} more items")

    text = "\n".join(lines).strip() or "directory listed"
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
