from __future__ import annotations

import json
import logging
from typing import Any

from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from .tool_result_support import is_small_model as _is_small_model

logger = logging.getLogger("smallctl.harness.tool_results")


async def build_tool_result_message(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    tool_call_id: str | None,
) -> ConversationMessage:
    from ..context.policy import estimate_text_tokens

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
        else:
            compact_content = str(result.error or result.output or "Tool failed.")
    elif tool_name == "artifact_read" and isinstance(result.output, str):
        compact_content = result.output
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
            else str(result.output)
        )

    if tool_name in {"plan_set", "plan_step_update", "plan_request_execution", "plan_export"}:
        playbook_artifact_id = str(result.metadata.get("artifact_id", "") or "").strip()
        if playbook_artifact_id:
            compact_content = f"Plan playbook captured as Artifact {playbook_artifact_id}.\n\n{compact_content}"

    service.harness._runlog(
        "artifact_created",
        "tool result processed",
        artifact_id=artifact.artifact_id if artifact else None,
        tool_name=tool_name,
        source=artifact.source if artifact else tool_name,
        size_bytes=artifact.size_bytes if artifact else 0,
        inline=bool(artifact and artifact.inline_content is not None),
    )

    message = ConversationMessage(
        role="tool",
        name=tool_name,
        tool_call_id=tool_call_id,
        content=compact_content,
        metadata={"artifact_id": artifact.artifact_id} if artifact else {},
    )

    args_str = json.dumps(result.metadata.get("arguments", {}), sort_keys=True)
    outcome = "success" if result.success else f"error:{result.error}"
    fingerprint = f"{tool_name}|{args_str}|{outcome}"
    service.harness.state.append_tool_history(fingerprint)
    return message
