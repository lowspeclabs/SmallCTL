from __future__ import annotations

from typing import Any

from ..evidence import normalize_tool_result
from ..models.tool_result import ToolEnvelope
from ..state import json_safe_value


def tool_execution_context(service: Any, operation_id: str | None) -> dict[str, Any]:
    if not operation_id:
        return {}
    records = getattr(service.harness.state, "tool_execution_records", None)
    if not isinstance(records, dict):
        return {}
    record = records.get(operation_id)
    return dict(record) if isinstance(record, dict) else {}


def record_evidence(
    service: Any,
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: Any,
    operation_id: str | None,
    replayed: bool = False,
) -> Any:
    context = tool_execution_context(service, operation_id)
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    evidence = normalize_tool_result(
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        operation_id=operation_id or str(context.get("operation_id", "") or ""),
        phase=str(getattr(service.harness.state, "current_phase", "") or context.get("phase", "") or ""),
        evidence_context=context,
        replayed=replayed,
        created_at_step=getattr(service.harness.state, "step_count", 0) or 0,
    )
    reasoning_graph = getattr(service.harness.state, "reasoning_graph", None)
    if reasoning_graph is not None:
        reasoning_graph.evidence_records.append(evidence)
        reasoning_graph.touch_ids()
    replayed_or_cached = bool(
        replayed
        or bool(context.get("replayed"))
        or bool(metadata.get("cache_hit"))
    )
    if not replayed_or_cached:
        clear_stale = getattr(service.harness.state, "_clear_lane_staleness", None)
        if callable(clear_stale):
            evidence_id = str(getattr(evidence, "evidence_id", "") or "").strip()
            if evidence_id:
                clear_stale("_observation_staleness", evidence_id)
            artifact_id = str(getattr(evidence, "artifact_id", "") or "").strip()
            if artifact_id:
                clear_stale("_artifact_staleness", artifact_id)
    if operation_id:
        stored = service.harness.state.tool_execution_records.get(operation_id)
        if isinstance(stored, dict):
            stored["artifact_id"] = str(
                getattr(artifact, "artifact_id", "")
                or result.metadata.get("artifact_id")
                or context.get("artifact_id")
                or stored.get("artifact_id", "")
                or ""
            )
            stored["evidence_id"] = evidence.evidence_id
            stored["evidence_type"] = evidence.evidence_type
            stored["evidence_record"] = json_safe_value(evidence)
            stored["evidence_context"] = context
            service.harness.state.tool_execution_records[operation_id] = stored
    if artifact is not None:
        artifact.metadata.setdefault("evidence_id", evidence.evidence_id)
        artifact.metadata.setdefault("evidence_type", evidence.evidence_type)
        if operation_id:
            artifact.metadata.setdefault("operation_id", operation_id)
    return evidence
