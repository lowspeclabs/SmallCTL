from __future__ import annotations

from pathlib import Path
from typing import Any

from ..context.retrieval_state_helpers import path_match, state_target_paths
from .state import PendingToolCall
from .tool_loop_guard_progress import (
    _coerce_int_or_none,
    _extract_args_from_fingerprint,
    _requested_artifact_read_target,
    _requested_file_read_range,
    _tool_attempt_history,
)


def _artifact_record_for_pending_read(harness: Any, pending: PendingToolCall) -> Any | None:
    if pending.tool_name != "artifact_read":
        return None
    artifact_id = _requested_artifact_read_target(pending.args)
    if not artifact_id:
        return None
    artifacts = getattr(getattr(harness, "state", None), "artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    return artifacts.get(artifact_id)


def _artifact_read_targets_mutation_result_loop(harness: Any, pending: PendingToolCall) -> bool:
    artifact = _artifact_record_for_pending_read(harness, pending)
    if artifact is None:
        return False
    tool_name = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
    if tool_name not in {"ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}:
        return False
    artifact_id = _requested_artifact_read_target(pending.args)
    if not artifact_id:
        return False
    prior_reads = 0
    for item in reversed(_tool_attempt_history(harness)[-6:]):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if isinstance(args, dict) and _requested_artifact_read_target(args) == artifact_id:
            prior_reads += 1
    return prior_reads >= 1


def _artifact_read_past_eof_is_loop(harness: Any, pending: PendingToolCall) -> bool:
    artifact = _artifact_record_for_pending_read(harness, pending)
    if artifact is None:
        return False
    start_line, _end_line = _requested_file_read_range(pending.args)
    if start_line is None or start_line <= 0:
        return False
    total_lines = None
    metadata = getattr(artifact, "metadata", {})
    if isinstance(metadata, dict):
        raw_total = metadata.get("total_lines") or metadata.get("artifact_total_lines")
        total_lines = _coerce_int_or_none(raw_total)
    if total_lines is None:
        content_path = str(getattr(artifact, "content_path", "") or "").strip()
        if content_path:
            try:
                total_lines = len(Path(content_path).read_text(encoding="utf-8").splitlines())
            except OSError:
                total_lines = None
    if total_lines is None or start_line <= total_lines:
        return False
    artifact_id = _requested_artifact_read_target(pending.args)
    for item in reversed(_tool_attempt_history(harness)[-6:]):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if not isinstance(args, dict) or _requested_artifact_read_target(args) != artifact_id:
            continue
        prior_start, _prior_end = _requested_file_read_range(args)
        if prior_start is not None and prior_start > total_lines:
            return True
    return False


def _ssh_file_read_after_remote_mutation_is_progress(
    harness: Any, pending: PendingToolCall
) -> bool:
    """Allow ssh_file_read right after a successful ssh_file_patch/write/replace_between
    on the same remote path - the model needs to see the mutated state."""
    if pending.tool_name != "ssh_file_read":
        return False
    read_path = str(pending.args.get("path") or "").strip()
    read_host = str(pending.args.get("host") or "").strip().lower()
    if not read_path:
        return False
    for item in reversed(_tool_attempt_history(harness)[-6:]):
        tool_name = str(item.get("tool_name", ""))
        if tool_name == "ssh_file_read":
            continue
        if tool_name not in {"ssh_file_patch", "ssh_file_replace_between", "ssh_file_write"}:
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if not isinstance(args, dict):
            continue
        mutated_path = str(args.get("path") or "").strip()
        mutated_host = str(args.get("host") or "").strip().lower()
        if mutated_path == read_path and (not read_host or mutated_host == read_host):
            return True
    return False


def _artifact_read_target_path_re_read_is_loop(harness: Any, pending: PendingToolCall) -> bool:
    """
    Detect re-reading a user-specified target/instruction file (e.g. AGENTS.md)
    after a prior read already covered the requested range. Small models often
    return to the instruction file as a crutch instead of acting on tool output.
    """
    if pending.tool_name != "artifact_read":
        return False

    artifact_id = _requested_artifact_read_target(pending.args)
    if not artifact_id:
        return False

    artifact = _artifact_record_for_pending_read(harness, pending)
    if artifact is None:
        return False

    source_path = str(
        getattr(artifact, "source", "")
        or getattr(artifact, "path", "")
        or ""
    ).strip()
    if not source_path:
        return False

    state = getattr(harness, "state", None)
    if state is None:
        return False
    target_paths = state_target_paths(state)
    normalized_source = Path(source_path).as_posix().lower()
    if not target_paths or not any(
        path_match(normalized_source, Path(target).as_posix().lower())
        for target in target_paths
    ):
        return False

    # Check if the requested range has already been covered by a prior read
    # of the same artifact (by artifact_id) or by a file_read of the same path.
    start_line, end_line = _requested_file_read_range(pending.args)
    if start_line is None:
        start_line = 1

    history = _tool_attempt_history(harness)
    for item in reversed(history[-6:]):
        tool_name = str(item.get("tool_name", ""))
        if tool_name not in {"artifact_read", "file_read"}:
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if not isinstance(args, dict):
            continue

        if tool_name == "artifact_read":
            if _requested_artifact_read_target(args) != artifact_id:
                continue
        else:  # file_read
            read_path = str(args.get("path") or "").strip()
            if not read_path:
                continue
            if Path(read_path).as_posix().lower() != normalized_source:
                continue

        prior_start, prior_end = _requested_file_read_range(args)
        if prior_start is None:
            prior_start = 1
        if prior_end is not None and end_line is not None:
            if prior_start <= start_line and prior_end >= end_line:
                return True
        elif prior_end is None and end_line is None:
            # Both are full reads; any prior full read of the same target file is a loop.
            return True

    return False
