from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.conversation import ConversationMessage
from ..normalization import dedupe_keep_tail
from .tool_result_support import is_small_model


def _supersede_prior_read_artifacts(
    service: Any,
    *,
    new_artifact_id: str,
    tool_name: str,
    path: str,
    host: str | None = None,
) -> None:
    """Mark older read artifacts for the same path as superseded by the new one."""
    if not new_artifact_id or not path:
        return
    normalized_path = Path(path).as_posix().lower()
    normalized_host = str(host or "").strip().lower()
    for artifact_id, artifact in list(service.harness.state.artifacts.items()):
        if artifact_id == new_artifact_id:
            continue
        art_tool = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
        if art_tool != tool_name:
            continue
        metadata = getattr(artifact, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        art_path = str(metadata.get("path") or "").strip()
        if not art_path:
            args = metadata.get("arguments")
            if isinstance(args, dict):
                art_path = str(args.get("path") or "").strip()
        if not art_path:
            art_path = str(getattr(artifact, "source", "") or "").strip()
        art_host = str(metadata.get("host") or "").strip().lower()
        if not art_host:
            args = metadata.get("arguments")
            if isinstance(args, dict):
                art_host = str(args.get("host") or "").strip().lower()
        if Path(art_path).as_posix().lower() == normalized_path:
            if normalized_host and art_host and art_host != normalized_host:
                continue
            metadata["superseded_by"] = new_artifact_id


def _mark_prior_read_artifacts_stale(
    service: Any,
    *,
    path: str,
    reason: str = "file_mutated",
) -> None:
    """Mark prior file_read artifacts for the same path as stale after a mutation.

    Unlike superseding (which happens when a newer read artifact replaces an
    older one), staleness is used when the live file has been modified by a
    patch or write and the old snapshot no longer reflects reality.
    """
    if not path:
        return
    normalized_path = Path(path).as_posix().lower()
    for artifact_id, artifact in list(service.harness.state.artifacts.items()):
        art_tool = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
        if art_tool not in {"file_read", "ssh_file_read"}:
            continue
        metadata = getattr(artifact, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        art_path = str(metadata.get("path") or "").strip()
        if not art_path:
            args = metadata.get("arguments")
            if isinstance(args, dict):
                art_path = str(args.get("path") or "").strip()
        if not art_path:
            art_path = str(getattr(artifact, "source", "") or "").strip()
        if Path(art_path).as_posix().lower() == normalized_path:
            if metadata.get("superseded_by"):
                continue
            metadata["stale"] = True
            metadata["artifact_stale_reason"] = reason
            metadata["authoritative_path"] = art_path
            staleness_index = service.harness.state.scratchpad.setdefault("_artifact_staleness", {})
            if isinstance(staleness_index, dict) and artifact_id:
                staleness_index[artifact_id] = {
                    "stale": True,
                    "reason": reason,
                    "paths": [art_path],
                }


def _maybe_emit_artifact_read_eof_overread_nudge(
    service: Any,
    *,
    result: Any,
    artifact: Any,
) -> None:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not bool(metadata.get("eof_overread")):
        return

    artifact_id = str(metadata.get("artifact_id") or getattr(artifact, "artifact_id", "") or "").strip()
    requested_start = int(metadata.get("requested_start_line") or metadata.get("line_start") or 0)
    total_lines = int(metadata.get("artifact_total_lines") or metadata.get("total_lines") or 0)
    if not artifact_id or requested_start <= 0 or total_lines <= 0:
        return

    signature = f"{artifact_id}:{requested_start}:{total_lines}"
    scratchpad = getattr(service.harness.state, "scratchpad", {})
    prior = scratchpad.get("_artifact_read_eof_overread_nudges", [])
    if isinstance(prior, list) and signature in prior:
        return

    if isinstance(prior, list):
        scratchpad["_artifact_read_eof_overread_nudges"] = dedupe_keep_tail(prior + [signature], limit=16)
    else:
        scratchpad["_artifact_read_eof_overread_nudges"] = [signature]

    model_note = (
        " This is a strong hallucination signal for the current small model: trust the reported EOF instead of inventing more lines."
        if is_small_model(service.harness)
        else ""
    )
    content = (
        f"`artifact_read` asked for unseen lines past EOF on artifact `{artifact_id}`: "
        f"requested `start_line={requested_start}` but the artifact only has `{total_lines}` lines."
        f"{model_note} Do not call `artifact_read` again past EOF. "
        "Use the evidence already in context, synthesize the answer, or choose a different tool."
    )
    service.harness.state.append_message(
        ConversationMessage(
            role="system",
            content=content,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "artifact_read_eof_overread",
                "artifact_id": artifact_id,
                "requested_start_line": requested_start,
                "artifact_total_lines": total_lines,
            },
        )
    )
    service.harness._runlog(
        "artifact_read_eof_overread_nudge",
        "nudged model after reading past artifact EOF",
        artifact_id=artifact_id,
        requested_start_line=requested_start,
        artifact_total_lines=total_lines,
    )
