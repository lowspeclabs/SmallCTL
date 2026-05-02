from __future__ import annotations

import json
from typing import Any

from ..models.conversation import ConversationMessage
from ..models.tool_result import ToolEnvelope
from ..state import json_safe_value
from .tool_result_evidence import record_evidence
from .tool_result_support import auto_mirror_session_anchor as _auto_mirror_session_anchor


def handle_reused_artifact_result(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    operation_id: str | None,
    tool_call_id: str | None,
) -> ConversationMessage | None:
    if result.metadata.get("cache_hit"):
        artifact_id = str(result.metadata.get("artifact_id", "")).strip()
        artifact = service.harness.state.artifacts.get(artifact_id)
        if artifact is None:
            from ..context import format_reused_artifact_message

            compact_content = json.dumps(json_safe_value(result.to_dict()), ensure_ascii=True)
        else:
            from ..context import format_reused_artifact_message

            service.harness.state.retrieval_cache = [artifact.artifact_id]
            compact_content = format_reused_artifact_message(artifact, tool_name=tool_name)
            service.harness._runlog(
                "artifact_reused",
                "tool result satisfied from cached artifact",
                artifact_id=artifact.artifact_id,
                tool_name=tool_name,
                source=artifact.source,
            )
        evidence = record_evidence(
            service,
            tool_name=tool_name,
            result=result,
            artifact=artifact,
            operation_id=operation_id,
            replayed=True,
        )
        _auto_mirror_session_anchor(service.harness, tool_name=tool_name, result=result, arguments=arguments)
        if artifact is not None:
            artifact.metadata.setdefault("evidence_statement", evidence.statement)
        return ConversationMessage(
            role="tool",
            name=tool_name,
            tool_call_id=tool_call_id,
            content=compact_content,
            metadata={"artifact_id": artifact_id, "cache_hit": True},
        )

    if tool_name == "artifact_print" and result.success:
        referenced_artifact_id = str(
            result.metadata.get("artifact_id")
            or result.metadata.get("source_artifact_id")
            or (
                arguments.get("artifact_id")
                if isinstance(arguments, dict)
                else ""
            )
            or ""
        ).strip()
        artifact = service.harness.state.artifacts.get(referenced_artifact_id) if referenced_artifact_id else None
        if artifact is not None:
            from ..context import format_reused_artifact_message

            service.harness.state.retrieval_cache = [artifact.artifact_id]
            service.harness._runlog(
                "artifact_reused",
                "artifact print reused an existing artifact",
                artifact_id=artifact.artifact_id,
                tool_name=tool_name,
                source=artifact.source,
            )
            evidence = record_evidence(
                service,
                tool_name=tool_name,
                result=result,
                artifact=artifact,
                operation_id=operation_id,
                replayed=True,
            )
            _auto_mirror_session_anchor(service.harness, tool_name=tool_name, result=result, arguments=arguments)
            artifact.metadata.setdefault("evidence_statement", evidence.statement)
            return ConversationMessage(
                role="tool",
                name=tool_name,
                tool_call_id=tool_call_id,
                content=format_reused_artifact_message(artifact, tool_name=tool_name),
                metadata={
                    "artifact_id": artifact.artifact_id,
                    "source_artifact_id": artifact.artifact_id,
                    "cache_hit": True,
                },
            )
    return None
