from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .models.tool_result import ToolEnvelope
from .state import ArtifactRecord, EvidenceRecord, json_safe_value

_EVIDENCE_ID_PREFIX = "E"


def normalize_tool_result(
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: ArtifactRecord | None = None,
    operation_id: str = "",
    phase: str = "",
    evidence_context: dict[str, Any] | None = None,
    replayed: bool = False,
) -> EvidenceRecord:
    context = dict(evidence_context or {})
    metadata = _build_metadata(
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        context=context,
        operation_id=operation_id,
        phase=phase,
        replayed=replayed or _coerce_bool(context.get("replayed"), default=False),
    )
    artifact_id = str(
        (artifact.artifact_id if artifact else "")
        or context.get("artifact_id")
        or result.metadata.get("artifact_id")
        or ""
    ).strip()
    source = str((artifact.source if artifact else "") or metadata.get("source") or tool_name).strip()
    replayed_or_cached = bool(
        replayed
        or _coerce_bool(context.get("replayed"), default=False)
        or _coerce_bool(result.metadata.get("cache_hit"), default=False)
    )
    negative = not bool(result.success)
    evidence_type = "replayed_or_cached" if replayed_or_cached else "negative_evidence" if negative else "direct_observation"
    statement = _build_statement(
        tool_name=tool_name,
        result=result,
        artifact=artifact,
        artifact_id=artifact_id,
        replayed=replayed_or_cached,
    )
    evidence_id = _derive_evidence_id(
        artifact_id=artifact_id,
        operation_id=operation_id,
        tool_name=tool_name,
        result=result,
        context=context,
    )
    claim_ids = _coerce_string_list(context.get("claim_ids"))
    decision_ids = _coerce_string_list(context.get("decision_ids"))
    evidence_refs = _coerce_string_list(context.get("evidence_refs"))
    confidence = _derive_confidence(result=result, replayed=replayed_or_cached)
    kind = "negative_observation" if negative else "observation"

    return EvidenceRecord(
        evidence_id=evidence_id,
        kind=kind,
        statement=statement,
        phase=str((phase or context.get("phase") or "")).strip(),
        tool_name=str(tool_name or "").strip(),
        operation_id=str((operation_id or context.get("operation_id") or "")).strip(),
        artifact_id=artifact_id,
        source=source,
        evidence_type=evidence_type,
        confidence=confidence,
        negative=negative,
        replayed=replayed_or_cached,
        claim_ids=claim_ids,
        decision_ids=decision_ids,
        evidence_refs=evidence_refs,
        metadata=metadata,
    )


def _build_metadata(
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: ArtifactRecord | None,
    context: dict[str, Any],
    operation_id: str,
    phase: str,
    replayed: bool,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "tool_name": tool_name,
        "operation_id": str(operation_id or context.get("operation_id") or "").strip(),
        "phase": str(phase or context.get("phase") or "").strip(),
        "success": bool(result.success),
        "status": result.status,
        "error": result.error,
        "cache_hit": bool(result.metadata.get("cache_hit")),
        "replayed": bool(replayed),
        "arguments": json_safe_value(context.get("args") or result.metadata.get("arguments") or {}),
    }
    for key in (
        "path",
        "url",
        "command",
        "status",
        "reason",
        "question",
        "artifact_id",
        "source_artifact_id",
        "line_start",
        "line_end",
        "requested_start_line",
        "requested_end_line",
        "total_lines",
        "complete_file",
        "tool_name",
    ):
        value = result.metadata.get(key)
        if value is not None:
            metadata[key] = json_safe_value(value)
    if artifact is not None:
        metadata["artifact_summary"] = artifact.summary
        metadata["artifact_source"] = artifact.source
        metadata["artifact_kind"] = artifact.kind
        metadata["artifact_id"] = artifact.artifact_id
    if context.get("artifact_id") and "artifact_id" not in metadata:
        metadata["artifact_id"] = str(context.get("artifact_id"))
    if context.get("summary"):
        metadata["summary"] = str(context.get("summary"))
    if context.get("replay_source"):
        metadata["replay_source"] = str(context.get("replay_source"))
    return metadata


def _build_statement(
    *,
    tool_name: str,
    result: ToolEnvelope,
    artifact: ArtifactRecord | None,
    artifact_id: str,
    replayed: bool,
) -> str:
    if not result.success:
        error = _shorten(str(result.error or result.metadata.get("error") or "tool failed"))
        return f"{tool_name} failed: {error}" if error else f"{tool_name} failed"

    if replayed and artifact_id:
        summary = _shorten(artifact.summary if artifact else "")
        if summary:
            return f"{tool_name}: reused {artifact_id}: {summary}"
        return f"{tool_name}: reused {artifact_id}"

    summary = ""
    if artifact is not None:
        summary = artifact.summary
    elif isinstance(result.output, str):
        summary = _shorten(result.output)
    elif isinstance(result.output, dict):
        keys = ", ".join(sorted(result.output.keys())[:5])
        summary = f"{tool_name} keys: {keys}" if keys else tool_name
    elif isinstance(result.output, list):
        summary = f"{tool_name} returned {len(result.output)} items"

    summary = _shorten(summary)
    if summary:
        return f"{tool_name}: {summary}" if not summary.startswith(tool_name) else summary
    return tool_name


def _derive_confidence(*, result: ToolEnvelope, replayed: bool) -> float:
    if not result.success:
        return 0.85
    if replayed:
        return 0.7
    return 0.95


def _derive_evidence_id(
    *,
    artifact_id: str,
    operation_id: str,
    tool_name: str,
    result: ToolEnvelope,
    context: dict[str, Any],
) -> str:
    seed = "|".join(
        [
            artifact_id,
            operation_id,
            tool_name,
            str(bool(result.success)),
            str(result.status or ""),
            str(result.error or ""),
            json.dumps(json_safe_value(context.get("args") or result.metadata.get("arguments") or {}), sort_keys=True, ensure_ascii=True, default=str),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    if artifact_id:
        return f"{_EVIDENCE_ID_PREFIX}-{artifact_id}"
    if operation_id:
        return f"{_EVIDENCE_ID_PREFIX}-{operation_id}"
    return f"{_EVIDENCE_ID_PREFIX}-{digest}"


def _shorten(text: str, *, limit: int = 220) -> str:
    collapsed = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set, frozenset)):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    return []
