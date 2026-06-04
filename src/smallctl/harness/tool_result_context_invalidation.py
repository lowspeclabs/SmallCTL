from __future__ import annotations

from typing import Any


def _emit_context_invalidation(
    service: Any,
    *,
    reason: str,
    paths: list[str] | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if reason == "file_changed" and paths:
        from ..graph.tool_call_parser import allow_repeated_tool_call_once
        for path_entry in paths:
            path_str = str(path_entry or "").strip()
            if not path_str:
                continue
            if ":" in path_str:
                host, sep, remote_path = path_str.partition(":")
                if sep and host and remote_path:
                    allow_repeated_tool_call_once(
                        service.harness,
                        "ssh_file_read",
                        {"host": host, "path": remote_path},
                    )
            else:
                allow_repeated_tool_call_once(
                    service.harness,
                    "file_read",
                    {"path": path_str},
                )
    event = service.harness.state.invalidate_context(
        reason=reason,
        paths=paths,
        details=details,
    )
    runlog = getattr(service.harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "context_invalidated",
            "context invalidation applied",
            reason=event.get("reason", reason),
            paths=event.get("paths", []),
            invalidated_fact_count=event.get("invalidated_fact_count", 0),
            invalidated_memory_count=event.get("invalidated_memory_count", 0),
            invalidated_facts=event.get("invalidated_facts", []),
            invalidated_memory_ids=event.get("invalidated_memory_ids", []),
            invalidated_turn_bundle_count=event.get("invalidated_turn_bundle_count", 0),
            invalidated_turn_bundle_ids=event.get("invalidated_turn_bundle_ids", []),
            invalidated_brief_count=event.get("invalidated_brief_count", 0),
            invalidated_brief_ids=event.get("invalidated_brief_ids", []),
            invalidated_summary_count=event.get("invalidated_summary_count", 0),
            invalidated_summary_ids=event.get("invalidated_summary_ids", []),
            invalidated_artifact_count=event.get("invalidated_artifact_count", 0),
            invalidated_artifact_ids=event.get("invalidated_artifact_ids", []),
            invalidated_observation_count=event.get("invalidated_observation_count", 0),
            invalidated_observation_ids=event.get("invalidated_observation_ids", []),
            details=event.get("details", {}),
        )
