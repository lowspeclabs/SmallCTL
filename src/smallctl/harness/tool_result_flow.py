from __future__ import annotations

import json
import logging
from typing import Any

from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from ..redaction import compact_tool_arguments_for_metadata, redact_sensitive_data
from .tool_result_reuse import handle_reused_artifact_result as _handle_reused_artifact_result
from .tool_result_postprocessing import apply_persisted_artifact_outcome as _apply_persisted_artifact_outcome
from .tool_result_rendering import build_tool_result_message as _build_tool_result_message

logger = logging.getLogger("smallctl.harness.tool_results")


async def _persist_artifact_result(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
    tool_call_id: str | None,
) -> ConversationMessage:
    from ..context.policy import estimate_text_tokens

    artifact = None
    skip_artifact_persist = tool_name == "artifact_read" and not result.success
    reusable_artifact_id = None
    if tool_name in {"artifact_read", "web_fetch"} and result.success and result.metadata.get("artifact_id"):
        reusable_artifact_id = str(result.metadata["artifact_id"])
    if reusable_artifact_id:
        artifact_id = reusable_artifact_id
        artifact = service.harness.state.artifacts.get(artifact_id)
        if artifact is not None and tool_name == "artifact_read" and isinstance(result.output, str):
            artifact.preview_text = result.output[:service.harness.artifact_store.policy.preview_char_limit]

    if artifact is None and not skip_artifact_persist:
        artifact = service.harness.artifact_store.persist_tool_result(
            tool_name=tool_name,
            result=result,
            session_id=str(getattr(service.harness.state, "thread_id", "") or ""),
            tool_call_id=str(tool_call_id or ""),
        )

    if result.success and result.output and artifact:
        out_str = str(result.output)
        tokens = estimate_text_tokens(out_str)
        if tokens > service.harness.context_policy.artifact_summarization_threshold and service.harness.summarizer_client:
            service.harness._runlog(
                "context_summarize_request",
                "requesting summarization for large tool result",
                tool_name=tool_name,
                tokens=tokens,
            )
            try:
                distilled = await service.harness.summarizer.summarize_artifact_async(
                    client=service.harness.summarizer_client,
                    artifact_id=artifact.artifact_id,
                    content=out_str,
                    label=artifact.source or tool_name,
                )
                if distilled:
                    artifact.summary = f"Distilled: {distilled}"
                    artifact.preview_text = distilled[:service.harness.artifact_store.policy.preview_char_limit]
                    artifact.metadata["summarized"] = True
            except Exception as exc:
                logger.warning("Automatic context summarization failed: %s", exc)

    await _apply_persisted_artifact_outcome(
        service,
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        arguments=arguments,
        operation_id=operation_id,
    )

    return await _build_tool_result_message(
        service,
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        tool_call_id=tool_call_id,
    )


async def record_result(
    service: Any,
    *,
    tool_name: str,
    tool_call_id: str | None,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None = None,
    operation_id: str | None = None,
) -> ConversationMessage:
    if isinstance(result.metadata, dict) and result.metadata:
        result.metadata = redact_sensitive_data(result.metadata)

    if arguments is not None and "arguments" not in result.metadata:
        compacted_arguments = compact_tool_arguments_for_metadata(tool_name, arguments)
        result.metadata["arguments"] = redact_sensitive_data(compacted_arguments)

    reused = _handle_reused_artifact_result(
        service,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
        operation_id=operation_id,
        tool_call_id=tool_call_id,
    )
    if reused is not None:
        return reused

    return await _persist_artifact_result(
        service,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
        operation_id=operation_id,
        tool_call_id=tool_call_id,
    )
