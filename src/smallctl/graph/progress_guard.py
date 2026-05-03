from __future__ import annotations

import re
from typing import Any

from ..models.conversation import ConversationMessage
from ..shell_utils import is_read_only_shell_evidence_action as _is_read_only_shell_evidence_action
from ..state import json_safe_value
from ..harness.task_transactions import recovery_context_lines, transaction_from_scratchpad
from .tool_loop_guard_progress import (
    _coerce_int_or_none,
    _requested_artifact_read_target,
    _requested_file_read_range,
    _tool_attempt_history,
)

_MUTATION_TOOLS = {
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
}

_READ_TOOLS = {
    "artifact_read",
    "ssh_file_read",
    "file_read",
}
_REMOTE_PATH_RE = re.compile(r"(?<![\w/])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+")
_ARTIFACT_COVERAGE_SCRATCHPAD_KEY = "_artifact_read_coverage"

def _is_ssh_exec_read_command(record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not command:
        return False
    return _is_read_only_shell_evidence_action(command)


def _turn_has_actionable_progress(harness: Any, graph_state: Any) -> bool:
    """Return True if the current turn changed actionable state."""
    last_tool_results = getattr(graph_state, "last_tool_results", []) or []
    last_assistant_text = str(getattr(graph_state, "last_assistant_text", "") or "").strip()

    # 1. Task completion
    for record in last_tool_results:
        if record.tool_name == "task_complete" and record.result.success:
            return True

    # 2. Successful mutation with changed=True
    for record in last_tool_results:
        if record.tool_name in _MUTATION_TOOLS:
            if record.result.success and (record.result.metadata or {}).get("changed") is True:
                return True

    # 3. Plan step state change
    if _plan_step_changed(harness):
        return True

    # 4. Successful verifier with a new verdict
    for record in last_tool_results:
        if record.tool_name in {"shell_exec", "ssh_exec"}:
            metadata = record.result.metadata or {}
            verdict = str(metadata.get("verdict") or metadata.get("status") or "").strip()
            if verdict:
                prior = _prior_turn_verdict(harness)
                if verdict != prior:
                    return True

    # 4b. Novel remote SSH observations count as progress even when the command failed.
    for record in last_tool_results:
        if record.tool_name == "ssh_exec" and _ssh_exec_has_novel_remote_observation(harness, record):
            return True

    # 5. Successful read of a new artifact/ssh_file range or ssh_exec read command
    for record in last_tool_results:
        if record.tool_name == "artifact_read" and record.result.success:
            if _artifact_read_is_past_eof(harness, record):
                return False
            if _artifact_read_result_is_new_range(harness, record):
                return True
        if record.tool_name == "ssh_file_read" and record.result.success:
            if _ssh_file_read_is_past_eof(harness, record):
                return False
            if _ssh_file_read_result_is_new_range(harness, record):
                return True
        if record.tool_name == "ssh_exec" and record.result.success:
            if _is_ssh_exec_read_command(record):
                if _ssh_exec_read_is_new(harness, record):
                    return True
            else:
                return True

    # 6. Any other successful non-read, non-mutation, non-exec tool
    #    (shell_exec/ssh_exec are handled above; identical calls are caught by loop guards)
    for record in last_tool_results:
        if record.result.success and record.tool_name not in _READ_TOOLS and record.tool_name not in _MUTATION_TOOLS and record.tool_name not in {"shell_exec", "ssh_exec"}:
            return True

    # 7. No-tool turn with non-repeating assistant text
    if not last_tool_results:
        if last_assistant_text and not _assistant_text_is_repeat(harness, last_assistant_text):
            return True

    return False


def _prior_turn_verdict(harness: Any) -> str:
    state = getattr(harness, "state", None)
    if state is None:
        return ""
    return str(getattr(state, "scratchpad", {}).get("_progress_prior_verdict", "") or "").strip()


def _artifact_read_result_is_new_range(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    artifact_id = _requested_artifact_read_target(args)
    if not artifact_id:
        return False
    span = _artifact_read_effective_span(record)
    if span is not None:
        span_artifact_id, start_line, end_line, _total_lines, eof_overread = span
        if eof_overread:
            return False
        artifact_id = span_artifact_id
        coverage = _artifact_coverage_entry(harness, artifact_id)
        if coverage is not None:
            return _span_adds_unseen_lines(
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


def _ssh_file_read_result_is_new_range(harness: Any, record: Any) -> bool:
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


def _ssh_exec_read_is_new(harness: Any, record: Any) -> bool:
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


def _record_progress_read(harness: Any, record: Any) -> None:
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
    elif record.tool_name == "ssh_exec":
        entry["command"] = str(args.get("command") or "").strip()
    start_line, end_line = _requested_file_read_range(args)
    if start_line is not None:
        entry["start_line"] = start_line
    if end_line is not None:
        entry["end_line"] = end_line
    history.append(entry)
    if len(history) > 24:
        del history[: len(history) - 24]
    if record.tool_name == "artifact_read":
        _record_artifact_read_coverage(harness, record)


def _artifact_read_effective_span(record: Any) -> tuple[str, int, int, int | None, bool] | None:
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


def _artifact_coverage_map(harness: Any) -> dict[str, dict[str, Any]]:
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    if not isinstance(scratchpad, dict):
        return {}
    coverage = scratchpad.setdefault(_ARTIFACT_COVERAGE_SCRATCHPAD_KEY, {})
    if not isinstance(coverage, dict):
        coverage = {}
        scratchpad[_ARTIFACT_COVERAGE_SCRATCHPAD_KEY] = coverage
    return coverage


def _artifact_coverage_entry(harness: Any, artifact_id: str) -> dict[str, Any] | None:
    coverage = _artifact_coverage_map(harness)
    entry = coverage.get(str(artifact_id or "").strip())
    return entry if isinstance(entry, dict) else None


def _span_adds_unseen_lines(*, start_line: int, end_line: int, ranges: Any) -> bool:
    normalized_ranges = _normalize_line_ranges(ranges)
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


def _normalize_line_ranges(ranges: Any) -> list[tuple[int, int]]:
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


def _record_artifact_read_coverage(harness: Any, record: Any) -> None:
    span = _artifact_read_effective_span(record)
    if span is None:
        return
    artifact_id, start_line, end_line, total_lines, eof_overread = span
    coverage = _artifact_coverage_map(harness)
    entry = coverage.setdefault(artifact_id, {"ranges": []})
    if not isinstance(entry, dict):
        entry = {"ranges": []}
        coverage[artifact_id] = entry
    if total_lines is not None:
        entry["total_lines"] = total_lines
    if eof_overread:
        entry["eof_overread"] = True
        return
    ranges = _normalize_line_ranges(entry.get("ranges", []))
    ranges.append((start_line, end_line))
    entry["ranges"] = [
        {"start_line": merged_start, "end_line": merged_end}
        for merged_start, merged_end in _normalize_line_ranges(ranges)
    ]
    total = _coerce_int_or_none(entry.get("total_lines"))
    if total is not None and total > 0:
        entry["complete"] = _coverage_is_complete(ranges=entry["ranges"], total_lines=total)


def _coverage_is_complete(*, ranges: Any, total_lines: int) -> bool:
    if total_lines < 1:
        return False
    normalized_ranges = _normalize_line_ranges(ranges)
    return bool(normalized_ranges and normalized_ranges[0][0] <= 1 and normalized_ranges[0][1] >= total_lines)


def _next_unread_artifact_line(harness: Any, artifact_id: str) -> int | None:
    entry = _artifact_coverage_entry(harness, artifact_id)
    if entry is None:
        return 1
    total_lines = _coerce_int_or_none(entry.get("total_lines"))
    cursor = 1
    for start_line, end_line in _normalize_line_ranges(entry.get("ranges", [])):
        if start_line > cursor:
            return cursor
        cursor = max(cursor, end_line + 1)
    if total_lines is not None and cursor > total_lines:
        return None
    return cursor


def _ssh_exec_remote_paths(record: Any) -> list[str]:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not command:
        return []
    paths: list[str] = []
    for match in _REMOTE_PATH_RE.finditer(command):
        path = match.group(0)
        if path not in paths:
            paths.append(path)
    return paths[:8]


def _ssh_exec_observation_entries(harness: Any) -> list[dict[str, Any]]:
    state = getattr(harness, "state", None)
    if state is None:
        return []
    scratchpad = getattr(state, "scratchpad", {})
    history = scratchpad.get("_progress_ssh_observation_history", [])
    return history if isinstance(history, list) else []


def _ssh_exec_has_novel_remote_observation(harness: Any, record: Any) -> bool:
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    host = str(args.get("host") or metadata.get("host") or "").strip().lower()
    failure_class = str(metadata.get("ssh_error_class") or metadata.get("failure_kind") or "").strip()
    auth_mode = str(metadata.get("ssh_auth_mode") or "").strip()
    reached_remote_host = (
        bool(getattr(record.result, "success", False))
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    )
    prior_entries = _ssh_exec_observation_entries(harness)
    if not host:
        return False

    if failure_class:
        if not any(
            str(item.get("host") or "").strip().lower() == host
            and str(item.get("failure_class") or "").strip() == failure_class
            for item in prior_entries
            if isinstance(item, dict)
        ):
            return True

    for path in _ssh_exec_remote_paths(record):
        if not any(
            str(item.get("host") or "").strip().lower() == host
            and path in (item.get("paths") or [])
            for item in prior_entries
            if isinstance(item, dict)
        ):
            return True

    if reached_remote_host and auth_mode:
        if not any(
            str(item.get("host") or "").strip().lower() == host
            and bool(item.get("reached_remote_host"))
            and str(item.get("auth_mode") or "").strip() == auth_mode
            for item in prior_entries
            if isinstance(item, dict)
        ):
            return True

    return False


def _record_ssh_exec_observation(harness: Any, record: Any) -> None:
    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return
    history = scratchpad.setdefault("_progress_ssh_observation_history", [])
    if not isinstance(history, list):
        return
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    history.append(
        {
            "host": str(args.get("host") or metadata.get("host") or "").strip().lower(),
            "failure_class": str(metadata.get("ssh_error_class") or metadata.get("failure_kind") or "").strip(),
            "paths": _ssh_exec_remote_paths(record),
            "auth_mode": str(metadata.get("ssh_auth_mode") or "").strip(),
            "reached_remote_host": (
                bool(getattr(record.result, "success", False))
                or bool(metadata.get("ssh_transport_succeeded"))
                or str(metadata.get("failure_kind") or "").strip() == "remote_command"
            ),
        }
    )
    if len(history) > 32:
        del history[: len(history) - 32]


def _artifact_read_is_past_eof(harness: Any, record: Any) -> bool:
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
                from pathlib import Path
                total_lines = len(Path(content_path).read_text(encoding="utf-8").splitlines())
            except OSError:
                total_lines = None
    if total_lines is not None and start_line > total_lines:
        return True
    return False


def _ssh_file_read_is_past_eof(harness: Any, record: Any) -> bool:
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


def _extract_args_from_fingerprint(fingerprint: str) -> dict[str, Any] | None:
    if not fingerprint:
        return None
    try:
        import json

        payload = json.loads(fingerprint)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    args = payload.get("args", {})
    return args if isinstance(args, dict) else None


def _plan_step_changed(harness: Any) -> bool:
    state = getattr(harness, "state", None)
    if state is None:
        return False
    plan = getattr(state, "active_plan", None)
    if plan is None:
        return False
    current_step = ""
    try:
        current_step = str(plan.current_step_index or plan.current_step or "").strip()
    except Exception:
        pass
    prior_step = str(getattr(state, "scratchpad", {}).get("_progress_prior_plan_step", "") or "").strip()
    return current_step != "" and current_step != prior_step


def _assistant_text_is_repeat(harness: Any, text: str) -> bool:
    state = getattr(harness, "state", None)
    if state is None:
        return False
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not normalized:
        return False
    prior_texts: list[str] = []
    for message in reversed(getattr(state, "recent_messages", [])):
        if getattr(message, "role", "") != "assistant":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        prior_normalized = re.sub(r"\s+", " ", content.lower())
        if prior_normalized == normalized:
            return True
        prior_texts.append(prior_normalized)
        if len(prior_texts) >= 2:
            break
    return False


def _update_progress_tracking(harness: Any, graph_state: Any) -> None:
    """Evaluate this turn and update the no-actionable-progress counter."""
    state = getattr(harness, "state", None)
    if state is None:
        return

    counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
    is_progress = _turn_has_actionable_progress(harness, graph_state)

    scratchpad = getattr(state, "scratchpad", {})
    if is_progress:
        counters["no_actionable_progress"] = 0
        # Update prior-state snapshots for next comparison
        if isinstance(scratchpad, dict):
            # Save current verdict from the most recent verifier result
            verdict: str = ""
            for record in getattr(graph_state, "last_tool_results", []) or []:
                if record.tool_name in {"shell_exec", "ssh_exec"}:
                    meta = record.result.metadata or {}
                    v = str(meta.get("verdict") or meta.get("status") or "").strip()
                    if v:
                        verdict = v
            if not verdict:
                current_verifier = getattr(state, "current_verifier_verdict", None)
                verifier = current_verifier() if callable(current_verifier) else None
                if isinstance(verifier, dict):
                    verdict = str(verifier.get("verdict") or verifier.get("status") or "").strip()
            scratchpad["_progress_prior_verdict"] = verdict
            # Save current plan step
            plan = getattr(state, "active_plan", None)
            if plan is not None:
                try:
                    scratchpad["_progress_prior_plan_step"] = str(
                        plan.current_step_index or plan.current_step or ""
                    ).strip()
                except Exception:
                    pass
    else:
        counters["no_actionable_progress"] = int(counters.get("no_actionable_progress", 0)) + 1

    # Record successful reads for next-turn range comparison
    for record in getattr(graph_state, "last_tool_results", []) or []:
        if record.tool_name in {"artifact_read", "ssh_file_read"} and record.result.success:
            _record_progress_read(harness, record)
        if record.tool_name == "ssh_exec" and record.result.success and _is_ssh_exec_read_command(record):
            _record_progress_read(harness, record)
        if record.tool_name == "ssh_exec":
            _record_ssh_exec_observation(harness, record)

    state.stagnation_counters = counters


def _build_progress_stagnation_nudge(harness: Any) -> str:
    goal = str(
        getattr(getattr(getattr(harness, "state", None), "run_brief", None), "original_task", "")
        or ""
    ).strip()
    goal_note = f" for `{goal}`" if goal else ""
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    transaction = transaction_from_scratchpad(scratchpad if isinstance(scratchpad, dict) else {})
    tx_lines = recovery_context_lines(transaction)
    last_action = _last_stalled_action(harness)
    last_action_note = f" Last stalled action: {last_action}." if last_action else ""
    tx_note = (" " + " ".join(tx_lines)) if tx_lines else ""
    return (
        "You have made no actionable progress in the last few turns. "
        f"Use the evidence already visible in context{goal_note}.{tx_note}{last_action_note} "
        "Perform the next concrete mutation, run a focused verifier, or call "
        "`task_complete(message='...')` if the task is finished. "
        "Do not repeat the same analysis or read operations. "
        "Choose exactly one: A. Explain the blocker and stop. B. Try a different specific fix. C. Ask for missing information."
    )


def _last_stalled_action(harness: Any) -> str:
    history = _tool_attempt_history(harness)
    if not history:
        return ""
    item = history[-1]
    tool_name = str(item.get("tool_name") or "").strip()
    if not tool_name:
        return ""
    return tool_name


def _stagnation_thresholds_for_phase(harness: Any) -> tuple[int, int]:
    """Return (nudge_start, trip_threshold) based on current phase.

    Research-heavy phases get higher thresholds to avoid false-positives
    during legitimate diagnostic or exploratory work.
    """
    state = getattr(harness, "state", None)
    if state is None:
        return 3, 5
    phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    if phase in {"explore", "repair"}:
        return 5, 7
    return 3, 5


def _check_progress_stagnation(harness: Any, graph_state: Any) -> str | None:
    """Check no-actionable-progress cycles and inject nudge or return guard error.

    Returns a guard error string if the stagnation limit has been reached,
    or None (after optionally injecting a nudge message into state).
    """
    state = getattr(harness, "state", None)
    if state is None:
        return None

    counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
    cycle_count = int(counters.get("no_actionable_progress", 0))
    nudge_start, trip_threshold = _stagnation_thresholds_for_phase(harness)

    if cycle_count < nudge_start:
        return None

    if nudge_start <= cycle_count < trip_threshold:
        # Inject recovery nudge and continue
        state.append_message(
            ConversationMessage(
                role="user",
                content=_build_progress_stagnation_nudge(harness),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "no_actionable_progress",
                    "cycle_count": cycle_count,
                },
            )
        )
        harness._runlog(
            "progress_stagnation_nudge",
            f"injected recovery nudge after {cycle_count} no-progress cycles",
            cycle_count=cycle_count,
        )
        return None

    # cycle_count >= trip_threshold -> trip guard
    return (
        f"Progress stagnation guard tripped: no actionable progress made in {cycle_count} steps. "
        "The model is repeating analysis or read-only operations without moving the task forward."
    )


_COMPLETION_CONFABULATION_PATTERNS = [
    re.compile(r"already\s+(performed|completed|done|finished)", re.IGNORECASE),
    re.compile(r"was\s+already\s+(performed|completed|done|finished)", re.IGNORECASE),
    re.compile(r"previous\s+task_complete", re.IGNORECASE),
    re.compile(r"prior\s+task_complete", re.IGNORECASE),
    re.compile(r"redesign\s+was\s+already", re.IGNORECASE),
    re.compile(r"successful\s+redesign\s+was", re.IGNORECASE),
    re.compile(r"task\s+is\s+already\s+complete", re.IGNORECASE),
    re.compile(r"work\s+is\s+already\s+done", re.IGNORECASE),
    re.compile(r"already\s+succeeded\s+in", re.IGNORECASE),
]


def _check_completion_confabulation(harness: Any, graph_state: Any) -> str | None:
    """Detect if the model falsely believes work was already completed in this task.

    Returns a guard error string if confabulation is detected, or None (after
    optionally injecting a recovery nudge into state).
    """
    state = getattr(harness, "state", None)
    if state is None:
        return None

    # If mutations have actually occurred, there is nothing to confabulate.
    if state.files_changed_this_cycle:
        return None

    for entry in state.tool_history:
        if not isinstance(entry, str):
            continue
        parts = entry.split("|")
        if len(parts) >= 3 and parts[-1] == "success" and parts[0] in _MUTATION_TOOLS:
            return None

    text_to_check = " ".join(
        [
            str(getattr(graph_state, "last_assistant_text", "") or ""),
            str(getattr(graph_state, "last_thinking_text", "") or ""),
        ]
    )
    if not text_to_check.strip():
        return None

    for pattern in _COMPLETION_CONFABULATION_PATTERNS:
        if pattern.search(text_to_check):
            break
    else:
        return None

    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    if scratchpad.get("_confabulation_nudged"):
        # Already nudged once for this task; don't spam.
        return None

    scratchpad["_confabulation_nudged"] = True
    state.scratchpad = scratchpad

    state.append_message(
        ConversationMessage(
            role="user",
            content=(
                "GROUND TRUTH CHECK: No mutating operations have been performed in this task. "
                "Do not assume any work was already completed. Start implementation now using the "
                "evidence already in context."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "completion_confabulation",
            },
        )
    )
    harness._runlog(
        "completion_confabulation_nudge",
        "injected confabulation recovery nudge",
    )
    return None
