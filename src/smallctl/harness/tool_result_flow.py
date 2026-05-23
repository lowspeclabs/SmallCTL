from __future__ import annotations

import json
import logging
from typing import Any

from ..context.policy import estimate_text_tokens
from ..context.rendering import render_shell_failure, render_shell_output
from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from ..redaction import compact_tool_arguments_for_metadata, redact_sensitive_data
from .tool_result_reuse import handle_reused_artifact_result as _handle_reused_artifact_result
from .tool_result_postprocessing import apply_persisted_artifact_outcome as _apply_persisted_artifact_outcome
from .tool_result_rendering import build_tool_result_message as _build_tool_result_message

logger = logging.getLogger("smallctl.harness.tool_results")


def _shell_result_text(result: ToolEnvelope) -> str:
    output = result.output if isinstance(result.output, dict) else {}
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not output:
        metadata_output = metadata.get("output")
        if isinstance(metadata_output, dict):
            output = metadata_output
    if isinstance(output, dict):
        if result.success:
            return render_shell_output(output, preview_limit=None, strip_whitespace=False)
        return render_shell_failure(
            error=result.error,
            output=output,
            preview_limit=None,
            strip_whitespace=False,
        )
    return str(result.error or result.output or "").strip() or ("ok" if result.success else "Tool failed.")


def _prompt_pressure_requires_artifact(service: Any) -> bool:
    prompt_budget = getattr(getattr(service, "harness", None), "state", None)
    prompt_budget = getattr(prompt_budget, "prompt_budget", None)
    pressure_level = str(getattr(prompt_budget, "pressure_level", "") or "").strip().lower()
    if pressure_level in {"medium", "high", "critical", "over_budget", "prompt_budget"}:
        return True
    estimated = int(getattr(prompt_budget, "estimated_prompt_tokens", 0) or 0)
    max_prompt = getattr(prompt_budget, "max_prompt_tokens", None)
    try:
        max_prompt_int = int(max_prompt or 0)
    except (TypeError, ValueError):
        max_prompt_int = 0
    return bool(max_prompt_int > 0 and estimated >= int(max_prompt_int * 0.85))


def _ssh_result_requires_artifact(service: Any, *, result: ToolEnvelope) -> bool:
    rendered = _shell_result_text(result)
    rendered_tokens = estimate_text_tokens(rendered)
    inline_limit = int(getattr(service.harness.context_policy, "tool_result_inline_token_limit", 325) or 325)
    artifact_limit = int(getattr(service.harness.artifact_store.policy, "inline_token_limit", inline_limit) or inline_limit)
    effective_limit = min(inline_limit, artifact_limit)
    if _prompt_pressure_requires_artifact(service):
        effective_limit = max(80, min(effective_limit, inline_limit // 2))
    if rendered_tokens > effective_limit:
        return True
    serialized = json.dumps(result.to_dict(), ensure_ascii=True, default=str)
    if estimate_text_tokens(serialized) > artifact_limit:
        return True
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    return bool(
        metadata.get("truncated")
        or metadata.get("requires_remote_mutation_verification")
        or metadata.get("force_artifact")
    )


def _should_persist_tool_artifact(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
) -> bool:
    if tool_name == "file_read":
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        total_lines = metadata.get("total_lines")
        if total_lines is None:
            output_text = str(result.output or "")
            total_lines = len(output_text.splitlines()) if output_text else 0
        if metadata.get("requested_start_line") is not None or metadata.get("requested_end_line") is not None:
            return True
        output_text = result.output if isinstance(result.output, str) else str(result.output or "")
        inline_limit = int(getattr(service.harness.context_policy, "tool_result_inline_token_limit", 325) or 325)
        if _prompt_pressure_requires_artifact(service):
            inline_limit = max(80, inline_limit // 2)
        # Small file reads can stay inline, but only when the full rendered text fits
        # comfortably in the tool-result budget. Otherwise keep a file_read artifact
        # available so later artifact searches target the file content, not a nearby
        # directory listing artifact.
        if total_lines < 500 and estimate_text_tokens(output_text) <= inline_limit:
            return False
        return True
    if tool_name == "ssh_file_read":
        return False
    if tool_name != "ssh_exec":
        return True
    return _ssh_result_requires_artifact(service, result=result)


async def _persist_artifact_result(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
    tool_call_id: str | None,
) -> ConversationMessage:
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
        if _should_persist_tool_artifact(service, tool_name=tool_name, result=result):
            artifact = service.harness.artifact_store.persist_tool_result(
                tool_name=tool_name,
                result=result,
                session_id=str(getattr(service.harness.state, "thread_id", "") or ""),
                tool_call_id=str(tool_call_id or ""),
            )
            if artifact is not None:
                if not isinstance(result.metadata, dict):
                    result.metadata = {}
                result.metadata["artifact_id"] = artifact.artifact_id

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
    message = reused

    if message is None:
        message = await _persist_artifact_result(
            service,
            tool_name=tool_name,
            result=result,
            arguments=arguments,
            operation_id=operation_id,
            tool_call_id=tool_call_id,
        )

    from ..fama.runtime import observe_tool_result

    await observe_tool_result(
        service,
        tool_name=tool_name,
        result=result,
        arguments=arguments,
        operation_id=operation_id,
    )
    return message
