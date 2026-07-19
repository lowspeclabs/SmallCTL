from __future__ import annotations

import json
import logging
from typing import Any

from ..context.rendering import render_shell_failure, render_shell_output
from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from .tool_result_support import is_small_model as _is_small_model

logger = logging.getLogger("smallctl.harness.tool_results")


def _compact_shell_without_artifact(result: ToolEnvelope, *, preview_chars: int) -> str:
    output = result.output if isinstance(result.output, dict) else {}
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not output:
        metadata_output = metadata.get("output")
        if isinstance(metadata_output, dict):
            output = metadata_output
    if isinstance(output, dict):
        if result.success:
            return render_shell_output(output, preview_limit=preview_chars, strip_whitespace=False)
        return render_shell_failure(
            error=result.error,
            output=output,
            preview_limit=preview_chars,
            strip_whitespace=False,
        )
    return str(result.error or result.output or "").strip() or ("ok" if result.success else "Tool failed.")


def _read_file_output_text(result: ToolEnvelope) -> str:
    output = result.output
    if isinstance(output, dict) and isinstance(output.get("content"), str):
        return output["content"]
    if isinstance(output, str):
        return output
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if isinstance(metadata.get("content"), str):
        return metadata["content"]
    return str(output or "")


def _inline_file_read_status(result: ToolEnvelope) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    path = str(metadata.get("path") or "").strip()
    source_path = str(metadata.get("source_path") or "").strip()
    read_from_staging = bool(metadata.get("read_from_staging") or metadata.get("staged_only"))
    complete_file = bool(metadata.get("complete_file"))
    file_content_truncated = bool(metadata.get("truncated"))
    line_start = metadata.get("line_start")
    line_end = metadata.get("line_end")
    total_lines = metadata.get("total_lines")
    lines = [
        "FILE READ STATUS:",
        f"path={path or '(unknown)'}",
    ]
    if source_path and source_path != path:
        lines.append(f"source_path={source_path}")
    if read_from_staging:
        session_id = str(metadata.get("write_session_id") or "").strip()
        detail = f"true; write_session_id={session_id}" if session_id else "true"
        lines.append(f"read_from_active_write_session_staging={detail}")
    lines.extend(
        [
            f"complete_file={'true' if complete_file else 'false'}",
            "display_preview_truncated=false",
            f"file_content_truncated={'true' if file_content_truncated else 'false'}",
        ]
    )
    if isinstance(line_start, int) or isinstance(line_end, int) or isinstance(total_lines, int):
        lines.append(f"lines={line_start}-{line_end} of {total_lines}")
    return "\n".join(lines)


def _format_recovery_hint(metadata: dict[str, Any]) -> str:
    """Append recovery instructions from guard metadata so the model can act on them."""
    next_action = metadata.get("next_required_action")
    next_tool = metadata.get("next_required_tool")
    if not next_action and not next_tool:
        return ""
    hints: list[str] = []
    if next_action:
        if isinstance(next_action, dict):
            action_text = json.dumps(next_action, indent=2, ensure_ascii=False)
        else:
            action_text = str(next_action)
        hints.append(f"Next required action: {action_text}")
    if next_tool:
        if isinstance(next_tool, dict):
            tool_text = json.dumps(next_tool, indent=2, ensure_ascii=False)
        else:
            tool_text = str(next_tool)
        hints.append(f"Next required tool: {tool_text}")
    return "\n\nRecovery hint:\n" + "\n".join(hints)


async def build_tool_result_message(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    tool_call_id: str | None,
) -> ConversationMessage:
    request_text = service.harness.state.run_brief.original_task or service.harness._current_user_task()
    compact_full_file = _is_small_model(service.harness)
    preview_chars = max(180, int(service.harness.context_policy.tool_result_inline_token_limit * 2))

    if not result.success:
        if artifact and tool_name in {"shell_exec", "ssh_exec"}:
            compact_content = service.harness.artifact_store.compact_tool_message(
                artifact,
                result,
                request_text=request_text,
                inline_full_file=not compact_full_file,
                full_file_preview_chars=preview_chars if compact_full_file else None,
            )
        elif tool_name in {"shell_exec", "ssh_exec"}:
            compact_content = _compact_shell_without_artifact(result, preview_chars=preview_chars)
        else:
            compact_content = str(result.error or result.output or "Tool failed.")
    elif tool_name == "artifact_read" and isinstance(result.output, str):
        compact_content = result.output
    elif artifact and tool_name in {"file_read", "ssh_file_read"}:
        compact_content = service.harness.artifact_store.compact_tool_message(
            artifact,
            result,
            request_text=request_text,
            inline_full_file=not compact_full_file,
            full_file_preview_chars=None,
        )
    elif tool_name in {"file_read", "ssh_file_read"}:
        compact_content = f"{_inline_file_read_status(result)}\n\n{_read_file_output_text(result)}"
    else:
        compact_content = (
            service.harness.artifact_store.compact_tool_message(
                artifact,
                result,
                request_text=request_text,
                inline_full_file=not compact_full_file,
                full_file_preview_chars=preview_chars if compact_full_file else None,
            )
            if artifact
            else (
                _compact_shell_without_artifact(result, preview_chars=preview_chars)
                if tool_name in {"shell_exec", "ssh_exec"}
                else str(result.output)
            )
        )

    if not result.success:
        recovery_hint = _format_recovery_hint(result.metadata if isinstance(result.metadata, dict) else {})
        if recovery_hint:
            compact_content = f"{compact_content}{recovery_hint}"

    dry_run_hint = (result.metadata.get("dry_run_hint") if isinstance(result.metadata, dict) else None)
    if dry_run_hint and result.success and tool_name in {"shell_exec", "ssh_exec"}:
        compact_content = f"{compact_content}\n\nHint: {dry_run_hint}"

    if tool_name in {"plan_set", "plan_step_update", "plan_request_execution", "plan_export"}:
        playbook_artifact_id = str(result.metadata.get("artifact_id", "") or "").strip()
        if playbook_artifact_id:
            compact_content = f"Plan playbook captured as Artifact {playbook_artifact_id}.\n\n{compact_content}"

    if artifact:
        service.harness._runlog(
            "artifact_created",
            "tool result processed",
            artifact_id=artifact.artifact_id,
            tool_name=tool_name,
            source=artifact.source,
            size_bytes=artifact.size_bytes,
            inline=bool(artifact.inline_content is not None),
        )
    else:
        service.harness._runlog(
            "tool_result_inlined",
            "tool result kept inline without artifact",
            tool_name=tool_name,
        )

    msg_metadata: dict[str, Any] = {"artifact_id": artifact.artifact_id} if artifact else {}
    if tool_name == "artifact_read" and result.success and isinstance(result.metadata, dict):
        msg_metadata["truncated"] = result.metadata.get("truncated")
        msg_metadata["total_lines"] = result.metadata.get("total_lines")
        msg_metadata["line_start"] = result.metadata.get("line_start")
        msg_metadata["line_end"] = result.metadata.get("line_end")
    if tool_name in {"file_read", "ssh_file_read"} and result.success and isinstance(result.metadata, dict):
        msg_metadata["path"] = result.metadata.get("path")
        msg_metadata["source_path"] = result.metadata.get("source_path")
        msg_metadata["complete_file"] = result.metadata.get("complete_file")
        msg_metadata["file_content_truncated"] = result.metadata.get("truncated")
        msg_metadata["line_start"] = result.metadata.get("line_start")
        msg_metadata["line_end"] = result.metadata.get("line_end")
        msg_metadata["total_lines"] = result.metadata.get("total_lines")

    message = ConversationMessage(
        role="tool",
        name=tool_name,
        tool_call_id=tool_call_id,
        content=compact_content,
        metadata=msg_metadata,
    )

    args_str = json.dumps(result.metadata.get("arguments", {}), sort_keys=True)
    outcome = "success" if result.success else f"error:{result.error}"
    fingerprint = f"{tool_name}|{args_str}|{outcome}"
    service.harness.state.append_tool_history(fingerprint)
    return message
