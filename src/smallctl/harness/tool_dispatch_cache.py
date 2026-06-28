from __future__ import annotations

from typing import Any

from ..models.tool_result import ToolEnvelope
from ..state import json_safe_value
from .artifact_tracking import file_read_cache_key, ssh_file_read_cache_key

_READ_ONLY_DEDUP_TOOLS = {"dir_list", "grep", "find_files", "artifact_read", "artifact_grep"}


def maybe_reuse_file_read(harness: Any, *, tool_name: str, args: dict[str, Any]) -> ToolEnvelope | None:
    if tool_name == "file_read":
        return _reuse_cached_file_read(harness, args)
    if tool_name == "ssh_file_read":
        return _reuse_cached_ssh_file_read(harness, args)
    return None


def maybe_reuse_identical_read_call(harness: Any, *, tool_name: str, args: dict[str, Any]) -> ToolEnvelope | None:
    if tool_name not in _READ_ONLY_DEDUP_TOOLS:
        return None
    records = getattr(harness.state, "tool_execution_records", None)
    if not isinstance(records, dict):
        return None
    normalized_args = json_safe_value(args)
    for record in reversed(list(records.values())):
        if not isinstance(record, dict):
            continue
        if str(record.get("tool_name") or "") != tool_name:
            continue
        if json_safe_value(record.get("args")) != normalized_args:
            continue
        result = record.get("result")
        if not isinstance(result, dict) or not result.get("success"):
            continue
        durable_meta = result.get("metadata", {}).get("durable_output_compacted")
        if isinstance(durable_meta, dict) and durable_meta.get("truncated"):
            # The stored result is a preview; do not return it as the full result.
            continue
        artifact_id = str(result.get("metadata", {}).get("artifact_id", "") or "").strip()
        harness._runlog(
            "tool_cache_hit",
            "reusing identical read-only tool result",
            tool_name=tool_name,
            artifact_id=artifact_id,
        )
        return ToolEnvelope(
            success=True,
            output=result.get("output"),
            metadata={
                "cache_hit": True,
                "artifact_id": artifact_id,
                "tool_name": tool_name,
            },
        )
    return None


def _reuse_cached_file_read(harness: Any, args: dict[str, Any]) -> ToolEnvelope | None:
    cache = harness.state.scratchpad.get("file_read_cache")
    if not isinstance(cache, dict):
        return None
    cache_key = file_read_cache_key(harness.state.cwd, args)
    if not cache_key:
        return None
    artifact_id = cache.get(cache_key)
    if not isinstance(artifact_id, str) or not artifact_id:
        return None
    artifact = harness.state.artifacts.get(artifact_id)
    if artifact is None:
        return None
    harness._runlog(
        "tool_cache_hit",
        "reusing prior file_read result",
        tool_name="file_read",
        artifact_id=artifact_id,
        path=artifact.source,
    )
    _mark_cached_file_read_for_repair_cycle(harness, args=args, source=str(artifact.source or ""))
    return ToolEnvelope(
        success=True,
        output={
            "status": "cached",
            "artifact_id": artifact_id,
            "path": artifact.source,
            "summary": artifact.summary,
        },
        metadata={
            "cache_hit": True,
            "artifact_id": artifact_id,
            "path": artifact.source,
            "tool_name": "file_read",
        },
    )


def _mark_cached_file_read_for_repair_cycle(harness: Any, *, args: dict[str, Any], source: str) -> None:
    state = getattr(harness, "state", None)
    if state is None or not str(getattr(state, "repair_cycle_id", "") or "").strip():
        return
    path = str(args.get("path") or source or "").strip()
    if not path:
        return
    try:
        from ..tools.fs_sessions import _record_repair_cycle_read
        from ..tools.fs_write_sessions import _resolve

        _record_repair_cycle_read(state, _resolve(path, getattr(state, "cwd", None)))
    except Exception:
        return


def _reuse_cached_ssh_file_read(harness: Any, args: dict[str, Any]) -> ToolEnvelope | None:
    cache = harness.state.scratchpad.get("ssh_file_read_cache")
    if not isinstance(cache, dict):
        return None
    cache_key = ssh_file_read_cache_key(args)
    if not cache_key:
        return None
    artifact_id = cache.get(cache_key)
    if not isinstance(artifact_id, str) or not artifact_id:
        return None
    artifact = harness.state.artifacts.get(artifact_id)
    if artifact is None:
        return None
    harness._runlog(
        "tool_cache_hit",
        "reusing prior ssh_file_read result",
        tool_name="ssh_file_read",
        artifact_id=artifact_id,
        path=artifact.source,
        host=str(args.get("host") or "").strip(),
    )
    return ToolEnvelope(
        success=True,
        output={
            "status": "cached",
            "artifact_id": artifact_id,
            "path": artifact.source,
            "host": str(args.get("host") or "").strip(),
            "summary": artifact.summary,
        },
        metadata={
            "cache_hit": True,
            "artifact_id": artifact_id,
            "path": artifact.source,
            "host": str(args.get("host") or "").strip(),
            "tool_name": "ssh_file_read",
        },
    )
