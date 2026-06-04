from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .tool_loop_guard_progress import (
    _coerce_int_or_none,
    _requested_artifact_read_target,
    _requested_file_read_range,
)

_ARTIFACT_COVERAGE_SCRATCHPAD_KEY = "_artifact_read_coverage"


def artifact_read_result_is_new_range(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    artifact_id = _requested_artifact_read_target(args)
    if not artifact_id:
        return False
    span = artifact_read_effective_span(record)
    if span is not None:
        span_artifact_id, start_line, end_line, _total_lines, eof_overread = span
        if eof_overread:
            return False
        artifact_id = span_artifact_id
        coverage = artifact_coverage_entry(harness, artifact_id)
        if coverage is not None:
            return span_adds_unseen_lines(
                start_line=start_line,
                end_line=end_line,
                ranges=coverage.get("ranges", []),
            )
    candidate_range = _requested_file_read_range(args)
    state = getattr(harness, "state", None)
    if state is None:
        return True
    history = getattr(state, "scratchpad", {}).get("_progress_read_history", [])
    if not isinstance(history, list):
        return True
    for item in reversed(history[-12:]):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        if str(item.get("artifact_id", "")) != artifact_id:
            continue
        prior_range = (_coerce_int_or_none(item.get("start_line")), _coerce_int_or_none(item.get("end_line")))
        if prior_range == candidate_range:
            return False
    return True


def ssh_file_read_result_is_new_range(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    path = str(args.get("path") or "").strip()
    if not path:
        return False
    candidate_range = _requested_file_read_range(args)
    state = getattr(harness, "state", None)
    if state is None:
        return True
    history = getattr(state, "scratchpad", {}).get("_progress_read_history", [])
    if not isinstance(history, list):
        return True
    for item in reversed(history[-12:]):
        if str(item.get("tool_name", "")) != "ssh_file_read":
            continue
        if str(item.get("path", "")) != path:
            continue
        prior_range = (_coerce_int_or_none(item.get("start_line")), _coerce_int_or_none(item.get("end_line")))
        if prior_range == candidate_range:
            return False
    return True


def file_read_result_is_new_range(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    path = str(args.get("path") or "").strip()
    if not path:
        return False
    candidate_range = _requested_file_read_range(args)
    state = getattr(harness, "state", None)
    if state is None:
        return True
    history = getattr(state, "scratchpad", {}).get("_progress_read_history", [])
    if not isinstance(history, list):
        return True
    for item in reversed(history[-12:]):
        if str(item.get("tool_name", "")) != "file_read":
            continue
        if str(item.get("path", "")) != path:
            continue
        prior_range = (_coerce_int_or_none(item.get("start_line")), _coerce_int_or_none(item.get("end_line")))
        if prior_range == candidate_range:
            return False
    return True


def ssh_exec_read_is_new(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not command:
        return False
    normalized_command = re.sub(r"\s+", " ", command.lower())
    state = getattr(harness, "state", None)
    if state is None:
        return True
    history = getattr(state, "scratchpad", {}).get("_progress_read_history", [])
    if not isinstance(history, list):
        return True
    for item in reversed(history[-12:]):
        if str(item.get("tool_name", "")) != "ssh_exec":
            continue
        prior_command = str(item.get("command", "") or "").strip()
        if re.sub(r"\s+", " ", prior_command.lower()) == normalized_command:
            return False
    return True


def record_progress_read(harness: Any, record: Any) -> None:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return
    history = scratchpad.setdefault("_progress_read_history", [])
    if not isinstance(history, list):
        return
    entry: dict[str, Any] = {"tool_name": record.tool_name}
    if record.tool_name == "artifact_read":
        entry["artifact_id"] = _requested_artifact_read_target(args)
    elif record.tool_name == "ssh_file_read":
        entry["path"] = str(args.get("path") or "").strip()
    elif record.tool_name == "file_read":
        entry["path"] = str(args.get("path") or "").strip()
    elif record.tool_name == "ssh_exec":
        entry["command"] = str(args.get("command") or "").strip()
    result = getattr(record, "result", None)
    metadata = getattr(result, "metadata", {}) if result is not None else {}
    if isinstance(metadata, dict):
        if metadata.get("path") and not entry.get("path"):
            entry["path"] = str(metadata.get("path") or "").strip()
        entry["source_path"] = str(metadata.get("source_path") or "").strip()
        entry["read_from_staging"] = bool(metadata.get("read_from_staging") or metadata.get("staged_only"))
        entry["complete_file"] = bool(metadata.get("complete_file"))
        entry["file_content_truncated"] = bool(metadata.get("truncated"))
        entry["total_lines"] = metadata.get("total_lines")
        entry["line_start"] = metadata.get("line_start")
        entry["line_end"] = metadata.get("line_end")
    start_line, end_line = _requested_file_read_range(args)
    if start_line is not None:
        entry["start_line"] = start_line
    if end_line is not None:
        entry["end_line"] = end_line
    history.append(entry)
    if len(history) > 24:
        del history[: len(history) - 24]
    if record.tool_name == "artifact_read":
        record_artifact_read_coverage(harness, record)


def artifact_read_effective_span(record: Any) -> tuple[str, int, int, int | None, bool] | None:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}

    artifact_id = str(
        metadata.get("artifact_id")
        or metadata.get("source_artifact_id")
        or _requested_artifact_read_target(args)
        or ""
    ).strip()
    if not artifact_id:
        return None

    requested_start, requested_end = _requested_file_read_range(args)
    start_line = _coerce_int_or_none(
        metadata.get("line_start", metadata.get("requested_start_line", requested_start))
    )
    end_line = _coerce_int_or_none(
        metadata.get("line_end", metadata.get("requested_end_line", requested_end))
    )
    total_lines = _coerce_int_or_none(metadata.get("total_lines", metadata.get("artifact_total_lines")))
    eof_overread = bool(metadata.get("eof_overread"))

    if start_line is None:
        return None
    if end_line is None and total_lines is not None and not bool(metadata.get("truncated")):
        end_line = total_lines
    if end_line is None:
        return None
    if total_lines is not None:
        end_line = min(end_line, total_lines)
    if start_line < 1 or end_line < start_line:
        return None
    return artifact_id, start_line, end_line, total_lines, eof_overread


def artifact_coverage_map(harness: Any) -> dict[str, dict[str, Any]]:
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    if not isinstance(scratchpad, dict):
        return {}
    coverage = scratchpad.setdefault(_ARTIFACT_COVERAGE_SCRATCHPAD_KEY, {})
    if not isinstance(coverage, dict):
        coverage = {}
        scratchpad[_ARTIFACT_COVERAGE_SCRATCHPAD_KEY] = coverage
    return coverage


def artifact_coverage_entry(harness: Any, artifact_id: str) -> dict[str, Any] | None:
    coverage = artifact_coverage_map(harness)
    entry = coverage.get(str(artifact_id or "").strip())
    return entry if isinstance(entry, dict) else None


def span_adds_unseen_lines(*, start_line: int, end_line: int, ranges: Any) -> bool:
    normalized_ranges = normalize_line_ranges(ranges)
    if not normalized_ranges:
        return True

    cursor = start_line
    for prior_start, prior_end in normalized_ranges:
        if prior_end < cursor:
            continue
        if prior_start > cursor:
            return True
        cursor = max(cursor, prior_end + 1)
        if cursor > end_line:
            return False
    return cursor <= end_line


def normalize_line_ranges(ranges: Any) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    if not isinstance(ranges, list):
        return normalized
    for item in ranges:
        if isinstance(item, dict):
            start_line = _coerce_int_or_none(item.get("start_line"))
            end_line = _coerce_int_or_none(item.get("end_line"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start_line = _coerce_int_or_none(item[0])
            end_line = _coerce_int_or_none(item[1])
        else:
            continue
        if start_line is None or end_line is None or start_line < 1 or end_line < start_line:
            continue
        normalized.append((start_line, end_line))
    normalized.sort()
    merged: list[tuple[int, int]] = []
    for start_line, end_line in normalized:
        if not merged or start_line > merged[-1][1] + 1:
            merged.append((start_line, end_line))
            continue
        prior_start, prior_end = merged[-1]
        merged[-1] = (prior_start, max(prior_end, end_line))
    return merged


def record_artifact_read_coverage(harness: Any, record: Any) -> None:
    span = artifact_read_effective_span(record)
    if span is None:
        return
    artifact_id, start_line, end_line, total_lines, eof_overread = span
    coverage = artifact_coverage_map(harness)
    entry = coverage.setdefault(artifact_id, {"ranges": []})
    if not isinstance(entry, dict):
        entry = {"ranges": []}
        coverage[artifact_id] = entry
    if total_lines is not None:
        entry["total_lines"] = total_lines
    state = getattr(harness, "state", None)
    if state is not None:
        entry["last_read_step"] = int(getattr(state, "step_count", 0) or 0)
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if isinstance(metadata, dict):
        entry["truncated"] = bool(metadata.get("truncated"))
    output = getattr(getattr(record, "result", None), "output", "")
    if isinstance(output, str) and output:
        entry["preview"] = output[:1200]
    if eof_overread:
        entry["eof_overread"] = True
        return
    ranges = normalize_line_ranges(entry.get("ranges", []))
    ranges.append((start_line, end_line))
    entry["ranges"] = [
        {"start_line": merged_start, "end_line": merged_end}
        for merged_start, merged_end in normalize_line_ranges(ranges)
    ]
    total = _coerce_int_or_none(entry.get("total_lines"))
    if total is not None and total > 0:
        entry["complete"] = coverage_is_complete(ranges=entry["ranges"], total_lines=total)


def coverage_is_complete(*, ranges: Any, total_lines: int) -> bool:
    if total_lines < 1:
        return False
    normalized_ranges = normalize_line_ranges(ranges)
    return bool(normalized_ranges and normalized_ranges[0][0] <= 1 and normalized_ranges[0][1] >= total_lines)


def next_unread_artifact_line(harness: Any, artifact_id: str) -> int | None:
    entry = artifact_coverage_entry(harness, artifact_id)
    if entry is None:
        return 1
    total_lines = _coerce_int_or_none(entry.get("total_lines"))
    cursor = 1
    for start_line, end_line in normalize_line_ranges(entry.get("ranges", [])):
        if start_line > cursor:
            return cursor
        cursor = max(cursor, end_line + 1)
    if total_lines is not None and cursor > total_lines:
        return None
    return cursor


def artifact_read_is_past_eof(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    artifact_id = _requested_artifact_read_target(args)
    if not artifact_id:
        return False
    start_line, _end_line = _requested_file_read_range(args)
    if start_line is None or start_line <= 0:
        return False
    state = getattr(harness, "state", None)
    if state is None:
        return False
    artifacts = getattr(state, "artifacts", {})
    if not isinstance(artifacts, dict):
        return False
    artifact = artifacts.get(artifact_id)
    if artifact is None:
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
    if total_lines is not None and start_line > total_lines:
        return True
    return False


def artifact_read_is_continuation_page(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    artifact_id = _requested_artifact_read_target(args)
    if not artifact_id:
        return False
    start_line, end_line = _requested_file_read_range(args)
    if start_line is None or start_line <= 1:
        return False
    coverage = artifact_coverage_entry(harness, artifact_id)
    if coverage is None:
        return False
    normalized = normalize_line_ranges(coverage.get("ranges", []))
    if not normalized or normalized[-1][1] < start_line - 1:
        return False
    # Only count as a continuation page if it extends beyond prior coverage
    if end_line is not None and end_line <= normalized[-1][1]:
        return False
    return True


def ssh_file_read_is_past_eof(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    start_line, _end_line = _requested_file_read_range(args)
    if start_line is None or start_line <= 0:
        return False
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    total_lines = _coerce_int_or_none(metadata.get("total_lines"))
    if total_lines is not None and start_line > total_lines:
        return True
    return False
