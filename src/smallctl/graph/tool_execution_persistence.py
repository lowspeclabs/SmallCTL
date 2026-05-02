from __future__ import annotations

from typing import Any

from .state import GraphRunState
from .tool_execution_support import (
    _conversation_message_from_dict,
    _get_tool_execution_record,
    _has_matching_tool_message,
    _store_tool_execution_record,
)


async def persist_tool_results(graph_state: GraphRunState, deps: Any) -> None:
    harness = deps.harness
    for record in graph_state.last_tool_results:
        stored = _get_tool_execution_record(harness, record.operation_id)
        serialized_message = stored.get("tool_message")
        if isinstance(serialized_message, dict):
            message = _conversation_message_from_dict(serialized_message)
        else:
            message = await harness._record_tool_result(
                tool_name=record.tool_name,
                tool_call_id=record.tool_call_id,
                result=record.result,
                arguments=record.args,
                operation_id=record.operation_id,
            )
            stored["tool_message"] = message.to_dict()
            artifact_id = message.metadata.get("artifact_id")
            if isinstance(artifact_id, str) and artifact_id:
                stored["artifact_id"] = artifact_id
                artifact = harness.state.artifacts.get(artifact_id)
                if artifact is not None:
                    metadata = dict(getattr(artifact, "metadata", {}) or {})
                    for key in ("plan_id", "step_id", "step_run_id", "step_attempt"):
                        value = stored.get(key)
                        if value not in (None, ""):
                            metadata[key] = value
                    artifact.metadata = metadata
            harness.state.tool_execution_records[record.operation_id] = stored
            harness._record_experience(
                tool_name=record.tool_name,
                result=record.result,
            )

        if _has_matching_tool_message(harness, message):
            continue
        harness.state.append_message(message)
        if harness.state.plan_execution_mode and harness.state.active_step_id:
            harness.state.step_sandbox_history.append(message)
        harness._log_conversation_state("tool_message")
