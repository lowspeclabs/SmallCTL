from __future__ import annotations

import json
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models.conversation import ConversationMessage
from ..state import ArtifactRecord, WriteSession
from ..tools.fs import (
    format_write_session_status_block,
    promote_write_session_target,
    restore_write_session_snapshot,
    write_session_status_snapshot,
)
from ..write_session_fsm import (
    record_write_session_event as record_write_session_event_alias,
    transition_write_session,
)
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord
from . import write_session_patch_recovery as _write_session_patch_recovery


def _maybe_schedule_write_recovery_readback(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name != "file_write" or not record.result.success:
        return False

    tool_call_id = str(record.tool_call_id or "").strip()
    if not tool_call_id.startswith("write_recovery_"):
        return False

    config = getattr(harness, "config", None)
    if not bool(getattr(config, "enforce_write_recovery_readback", False)):
        return False

    path = str(record.args.get("path") or "").strip()
    if not path:
        return False

    verifiable_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt", ".json", ".yaml", ".yml"}
    if Path(path).suffix.lower() not in verifiable_extensions:
        return False

    signature = "|".join([tool_call_id, path, "write_recovery_readback"])
    if harness.state.scratchpad.get("_write_recovery_readback_scheduled") == signature:
        return False
    harness.state.scratchpad["_write_recovery_readback_scheduled"] = signature

    graph_state.pending_tool_calls.append(
        PendingToolCall(
            tool_name="file_read",
            args={"path": path},
            raw_arguments=json.dumps({"path": path}, ensure_ascii=True, sort_keys=True),
            source="system",
        )
    )
    from .tool_call_parser import allow_repeated_tool_call_once

    allow_repeated_tool_call_once(harness, "file_read", {"path": path})

    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Synthesized write-recovery for `{path}` succeeded. "
                "Performing a mandatory read-back verification to ensure the file content is correct."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "write_recovery_readback",
                "target_path": path,
            },
        )
    )
    harness._runlog(
        "write_recovery_readback_scheduled",
        "scheduled mandatory read-back after synthesized write success",
        tool_call_id=tool_call_id,
        path=path,
    )
    return True


def _maybe_record_write_session_first_chunk_metric(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> float | None:
    session = getattr(harness.state, "write_session", None)
    if session is None:
        return None
    if record.tool_name not in {"file_write", "file_append", "file_patch", "ast_patch"} or not record.result.success:
        return None
    if str(record.result.metadata.get("write_session_id") or "").strip() != session.write_session_id:
        return None
    if float(getattr(session, "write_first_chunk_at", 0.0) or 0.0) > 0:
        return None
    if not bool(record.result.metadata.get("section_added")):
        return None
    completed_sections = record.result.metadata.get("write_sections_completed")
    if not isinstance(completed_sections, list) or len(completed_sections) != 1:
        return None

    started_at = float(getattr(session, "write_session_started_at", 0.0) or 0.0)
    if started_at <= 0:
        return None

    now = time.time()
    session.write_first_chunk_at = now
    elapsed = round(max(0.0, now - started_at), 3)
    graph_state.latency_metrics["time_to_first_chunk_sec"] = elapsed
    harness._runlog(
        "write_session_first_chunk",
        "recorded first successful chunk for active write session",
        session_id=session.write_session_id,
        target_path=session.write_target_path,
        section_name=str(record.result.metadata.get("write_current_section") or session.write_current_section or ""),
        time_to_first_chunk_sec=elapsed,
    )
    return elapsed


def _register_write_session_stage_artifact(harness: Any, session: Any) -> str | None:
    stage_path = str(getattr(session, "write_staging_path", "") or "").strip()
    if not stage_path or not Path(stage_path).exists():
        return None

    artifact_id = f"{session.write_session_id}__stage"
    stage_filename = Path(stage_path).name
    target_basename = Path(session.write_target_path).name
    # Backward compatibility for prior alias formats.
    legacy_ids = {
        stage_filename,
        f"{session.write_session_id}__{target_basename}__stage.py",
    }

    stage_file = Path(stage_path)
    stat = stage_file.stat()
    try:
        preview_text = stage_file.read_text(encoding="utf-8")
    except Exception:
        preview_text = ""

    record = ArtifactRecord(
        artifact_id=artifact_id,
        kind="file",
        source=stage_path,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        size_bytes=stat.st_size,
        summary=f"Staged content for {session.write_target_path} (session {session.write_session_id})",
        tool_name="file_write",
        content_path=stage_path,
        preview_text=preview_text[:2000] or None,
        metadata={
            "write_session_id": session.write_session_id,
            "target_path": session.write_target_path,
            "is_stage": True,
            "mtime": stat.st_mtime,
        },
    )

    harness.state.artifacts[artifact_id] = record
    for alias in legacy_ids:
        if alias:
            harness.state.artifacts[alias] = record
    return artifact_id


def _invalidate_write_session_stage_artifacts(harness: Any, session: Any, *, target_path: str = "") -> None:
    artifacts = getattr(getattr(harness, "state", None), "artifacts", None)
    if not isinstance(artifacts, dict) or not artifacts:
        return

    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    if not session_id:
        return

    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    authoritative_path = str(target_path or getattr(session, "write_target_path", "") or "").strip()
    target_hash = _target_file_hash(authoritative_path, cwd=cwd)
    invalidated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    seen: set[int] = set()
    for key, artifact in list(artifacts.items()):
        if artifact is None or id(artifact) in seen:
            continue
        seen.add(id(artifact))
        metadata = artifact.metadata if isinstance(getattr(artifact, "metadata", None), dict) else {}
        if str(metadata.get("write_session_id") or "").strip() != session_id:
            continue
        if not bool(metadata.get("is_stage")):
            continue
        metadata.update(
            {
                "stale": True,
                "model_visible": False,
                "artifact_stale_reason": "write_session_promoted",
                "promoted_at": invalidated_at,
                "authoritative_path": authoritative_path,
                "target_path": authoritative_path,
            }
        )
        if target_hash:
            metadata["authoritative_hash"] = target_hash
        artifact.metadata = metadata
        artifact.summary = (
            f"Stale staged content for {authoritative_path or getattr(session, 'write_target_path', '')} "
            f"(session {session_id}); target file is authoritative"
        )
        warning = _stale_stage_warning(metadata)
        preview = str(getattr(artifact, "preview_text", "") or "")
        if warning and preview and not preview.startswith(warning):
            artifact.preview_text = f"{warning}\n\n{preview}"
        elif warning and not preview:
            artifact.preview_text = warning

        try:
            harness._runlog(
                "write_session_stage_artifact_stale",
                "marked staged artifact stale after promotion",
                artifact_id=str(key),
                session_id=session_id,
                target_path=authoritative_path,
            )
        except Exception:
            pass


def _target_file_hash(path: str, *, cwd: str | None = None) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    try:
        base = Path(cwd).resolve() if cwd else Path.cwd().resolve()
        resolved = Path(raw) if Path(raw).is_absolute() else base / raw
        if not resolved.exists() or not resolved.is_file():
            return ""
        return hashlib.sha256(resolved.read_bytes()).hexdigest()
    except Exception:
        return ""


def _stale_stage_warning(metadata: dict[str, Any]) -> str:
    authoritative_path = str(metadata.get("authoritative_path") or metadata.get("target_path") or "").strip()
    if not authoritative_path:
        return "WARNING: This write-session stage artifact is stale because the write session was promoted."
    return (
        "WARNING: This write-session stage artifact is stale because the write session was promoted. "
        f"Use `file_read(path='{authoritative_path}')` for the current authoritative file."
    )


_maybe_emit_patch_existing_first_choice_nudge = _write_session_patch_recovery._maybe_emit_patch_existing_first_choice_nudge
_recover_patch_existing_recovery_session = _write_session_patch_recovery._recover_patch_existing_recovery_session
_maybe_schedule_patch_existing_stage_read_recovery = _write_session_patch_recovery._maybe_schedule_patch_existing_stage_read_recovery
_maybe_schedule_file_patch_read_recovery = _write_session_patch_recovery._maybe_schedule_file_patch_read_recovery
_maybe_emit_write_session_target_path_redirect_nudge = _write_session_patch_recovery._maybe_emit_write_session_target_path_redirect_nudge
_clear_patch_existing_stage_read_autocontinue_count_after_success = _write_session_patch_recovery._clear_patch_existing_stage_read_autocontinue_count_after_success
