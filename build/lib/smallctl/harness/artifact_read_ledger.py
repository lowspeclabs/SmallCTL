from __future__ import annotations

from typing import Any

from ..context.artifact_read_coverage import ARTIFACT_COVERAGE_SCRATCHPAD_KEY
from ..models.tool_result import ToolEnvelope


def _coerce_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_artifact_read_ranges(ranges: Any) -> list[tuple[int, int]]:
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


def _artifact_read_coverage_is_complete(*, ranges: Any, total_lines: int) -> bool:
    if total_lines < 1:
        return False
    normalized = _normalize_artifact_read_ranges(ranges)
    return bool(normalized and normalized[0][0] <= 1 and normalized[0][1] >= total_lines)


def record_artifact_read_ledger(
    service: Any,
    *,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None,
    artifact: Any,
) -> None:
    state = getattr(service.harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None) if state is not None else None
    if not isinstance(scratchpad, dict):
        return
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    artifact_id = str(metadata.get("artifact_id") or getattr(artifact, "artifact_id", "") or "").strip()
    if not artifact_id:
        return

    args = arguments if isinstance(arguments, dict) else {}
    requested_start = _coerce_int_or_none(args.get("start_line"))
    requested_end = _coerce_int_or_none(args.get("end_line"))
    start_line = _coerce_int_or_none(metadata.get("line_start", metadata.get("requested_start_line", requested_start)))
    end_line = _coerce_int_or_none(metadata.get("line_end", metadata.get("requested_end_line", requested_end)))
    total_lines = _coerce_int_or_none(metadata.get("total_lines", metadata.get("artifact_total_lines")))
    if start_line is None:
        return
    if end_line is None and total_lines is not None and not bool(metadata.get("truncated")):
        end_line = total_lines
    if end_line is None:
        return
    if total_lines is not None:
        end_line = min(end_line, total_lines)
    if start_line < 1 or end_line < start_line:
        return

    coverage = scratchpad.setdefault(ARTIFACT_COVERAGE_SCRATCHPAD_KEY, {})
    if not isinstance(coverage, dict):
        coverage = {}
        scratchpad[ARTIFACT_COVERAGE_SCRATCHPAD_KEY] = coverage
    entry = coverage.setdefault(artifact_id, {"ranges": []})
    if not isinstance(entry, dict):
        entry = {"ranges": []}
        coverage[artifact_id] = entry

    if total_lines is not None:
        entry["total_lines"] = total_lines
    entry["last_read_step"] = int(getattr(state, "step_count", 0) or 0)
    entry["truncated"] = bool(metadata.get("truncated"))
    if getattr(artifact, "source", ""):
        entry["source"] = str(getattr(artifact, "source", ""))
    output = result.output if isinstance(result.output, str) else ""
    if output:
        entry["preview"] = output[:1200]

    if bool(metadata.get("eof_overread")):
        entry["eof_overread"] = True
        return

    ranges = _normalize_artifact_read_ranges(entry.get("ranges", []))
    ranges.append((start_line, end_line))
    entry["ranges"] = [
        {"start_line": merged_start, "end_line": merged_end}
        for merged_start, merged_end in _normalize_artifact_read_ranges(ranges)
    ]
    total = _coerce_int_or_none(entry.get("total_lines"))
    if total is not None and total > 0:
        entry["complete"] = _artifact_read_coverage_is_complete(
            ranges=entry["ranges"],
            total_lines=total,
        )
