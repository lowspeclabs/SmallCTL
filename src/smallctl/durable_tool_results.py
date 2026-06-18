from __future__ import annotations

import json
from typing import Any

from .state_support import json_safe_value


DURABLE_OUTPUT_INLINE_CHAR_LIMIT = 64 * 1024
DURABLE_OUTPUT_PREVIEW_CHAR_LIMIT = 4 * 1024
DURABLE_METADATA_VALUE_CHAR_LIMIT = 4 * 1024


def compact_tool_result_for_durable_state(
    payload: Any,
    *,
    tool_name: str = "",
    artifact_id: str | None = None,
    output_inline_char_limit: int = DURABLE_OUTPUT_INLINE_CHAR_LIMIT,
    preview_char_limit: int = DURABLE_OUTPUT_PREVIEW_CHAR_LIMIT,
) -> dict[str, Any]:
    envelope = json_safe_value(payload if isinstance(payload, dict) else {})
    if not isinstance(envelope, dict):
        envelope = {}
    metadata = envelope.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    resolved_artifact_id = _first_non_empty(
        artifact_id,
        metadata.get("artifact_id"),
        envelope.get("artifact_id"),
    )
    output = json_safe_value(envelope.get("output"))
    output_chars = _serialized_char_count(output)
    metadata = _compact_metadata(metadata, preview_char_limit=preview_char_limit)
    if resolved_artifact_id:
        metadata.setdefault("artifact_id", resolved_artifact_id)

    compacted = dict(envelope)
    compacted["success"] = bool(envelope.get("success"))
    compacted["status"] = envelope.get("status")
    compacted["error"] = None if envelope.get("error") is None else str(envelope.get("error"))
    compacted["metadata"] = metadata
    if output_chars > max(0, int(output_inline_char_limit)):
        preview = _build_output_preview(
            output,
            tool_name=tool_name,
            artifact_id=resolved_artifact_id,
            preview_char_limit=max(1, int(preview_char_limit)),
            original_chars=output_chars,
        )
        compacted["output"] = preview
        metadata["durable_output_compacted"] = {
            "truncated": True,
            "original_chars": output_chars,
            "preview_chars": len(preview),
            "artifact_id": resolved_artifact_id or "",
        }
    else:
        compacted["output"] = output
    return compacted


def _build_output_preview(
    output: Any,
    *,
    tool_name: str,
    artifact_id: str | None,
    preview_char_limit: int,
    original_chars: int,
) -> str:
    parts: list[str] = []
    if artifact_id:
        parts.append(f"[output compacted; full output in artifact {artifact_id}; original_chars={original_chars}]")
    else:
        parts.append(f"[output compacted; original_chars={original_chars}]")
    if tool_name in {"shell_exec", "ssh_exec"} and isinstance(output, dict):
        for key in ("exit_code", "status", "command"):
            if key in output:
                parts.append(f"{key}: {output.get(key)}")
        stdout = str(output.get("stdout") or "")
        stderr = str(output.get("stderr") or "")
        if stdout:
            parts.append("stdout preview:\n" + _clip_text(stdout, preview_char_limit // 2))
        if stderr:
            parts.append("stderr preview:\n" + _clip_text(stderr, preview_char_limit // 2))
    else:
        parts.append(_clip_text(_preview_source(output), preview_char_limit))
    preview = "\n".join(part for part in parts if part)
    return _clip_text(preview, preview_char_limit)


def _compact_metadata(metadata: dict[str, Any], *, preview_char_limit: int) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    for key, value in metadata.items():
        compacted[str(key)] = _compact_metadata_value(value, preview_char_limit=preview_char_limit)
    return compacted


def _compact_metadata_value(value: Any, *, preview_char_limit: int) -> Any:
    safe = json_safe_value(value)
    if isinstance(safe, str):
        if len(safe) > DURABLE_METADATA_VALUE_CHAR_LIMIT:
            return _clip_text(safe, min(preview_char_limit, DURABLE_METADATA_VALUE_CHAR_LIMIT))
        return safe
    if isinstance(safe, dict):
        serialized_chars = _serialized_char_count(safe)
        if serialized_chars > DURABLE_OUTPUT_INLINE_CHAR_LIMIT:
            return {
                "preview": _clip_text(_preview_source(safe), preview_char_limit),
                "truncated": True,
                "original_chars": serialized_chars,
            }
        return {
            str(key): _compact_metadata_value(item, preview_char_limit=preview_char_limit)
            for key, item in safe.items()
        }
    if isinstance(safe, list):
        serialized_chars = _serialized_char_count(safe)
        if serialized_chars > DURABLE_OUTPUT_INLINE_CHAR_LIMIT:
            return {
                "preview": _clip_text(_preview_source(safe), preview_char_limit),
                "truncated": True,
                "original_chars": serialized_chars,
            }
        return [_compact_metadata_value(item, preview_char_limit=preview_char_limit) for item in safe]
    return safe


def _serialized_char_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str))
    except Exception:
        return len(str(value))


def _preview_source(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except Exception:
        return str(value)


def _clip_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    suffix = "\n...[truncated]"
    return value[: max(0, limit - len(suffix))].rstrip() + suffix


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""
