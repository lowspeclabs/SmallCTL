from __future__ import annotations

from collections import Counter
from typing import Any

from .signals import FamaFailureKind, FamaSignal, current_step, get_fama_state

_LOOP_COUNTERS = ("no_progress", "no_actionable_progress", "repeat_command", "repeat_patch")
_READ_LOOP_TOOLS = {"artifact_read", "file_read", "dir_list", "ssh_file_read", "web_fetch"}
_REMOTE_CONFUSION_REASONS = {
    "remote_path_requires_ssh_exec",
    "remote_path_requires_typed_ssh_file_tool",
    "remote_mutation_requires_verification",
}
_BAD_ARG_ERROR_MARKERS = (
    "tool arguments must be an object",
    "missing required field",
    "expected type",
    "schema validation",
    "invalid tool arguments",
    "validation error",
)
_BAD_ARG_REASONS = {"schema_validation", "validation_error", "bad_tool_args", "invalid_arguments"}
_OUTPUT_MISREAD_REASONS = {
    "lookup_answer_missing",
    "answer_missing_from_latest_output",
    "tool_output_contradiction",
    "task_complete_contradicts_tool_output",
}


def detect_early_stop_from_result(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    if str(tool_name or "").strip() != "task_complete":
        return None
    if bool(getattr(result, "success", False)):
        return None

    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    reason = str(metadata.get("reason") or "").strip()
    if reason in {"task_complete_blocked_in_staged_execution", "session_incomplete"}:
        verifier = _metadata_verifier(metadata)
        if not _verifier_failed(verifier):
            return None

    evidence = _early_stop_evidence(metadata, error=str(getattr(result, "error", "") or ""), state=state)
    if not evidence:
        return None
    return FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="tool_result",
        evidence=evidence,
        step=current_step(state),
        tool_name="task_complete",
        operation_id=operation_id,
    )


def detect_repeated_tool_loop(state: Any, *, threshold: int = 3) -> FamaSignal | None:
    counters = getattr(state, "stagnation_counters", None)
    if not isinstance(counters, dict):
        return None
    tripped: list[tuple[str, int]] = []
    threshold = max(1, int(threshold or 3))
    for name in _LOOP_COUNTERS:
        try:
            count = int(counters.get(name, 0) or 0)
        except (TypeError, ValueError):
            count = 0
        if count >= threshold:
            tripped.append((name, count))
    if not tripped:
        return None

    repeated_tool = _repeated_tool_from_history(state, threshold=threshold)
    evidence_bits = [f"{name}={count}" for name, count in tripped]
    if repeated_tool:
        evidence_bits.append(f"repeated_tool={repeated_tool}")
    return FamaSignal(
        kind=FamaFailureKind.LOOPING,
        severity=2,
        source="loop_guard",
        evidence="; ".join(evidence_bits),
        step=current_step(state),
        tool_name=repeated_tool,
    )


def detect_remote_local_confusion(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    reason = str(metadata.get("reason") or "").strip()
    if reason not in _REMOTE_CONFUSION_REASONS:
        return None
    path = str(metadata.get("path") or "").strip()
    suggested = str(metadata.get("suggested_tool") or "").strip()
    evidence = f"reason={reason}"
    if path:
        evidence += f"; path={path}"
    if suggested:
        evidence += f"; suggested_tool={suggested}"
    return FamaSignal(
        kind=FamaFailureKind.REMOTE_LOCAL_CONFUSION,
        severity=2,
        source="dispatcher",
        evidence=evidence,
        step=current_step(state),
        tool_name=str(tool_name or "").strip() or None,
        operation_id=operation_id,
    )


def detect_write_session_stall(state: Any, *, threshold: int = 3) -> FamaSignal | None:
    session = getattr(state, "write_session", None)
    if session is None:
        return None
    status = str(getattr(session, "status", "") or "").strip().lower()
    if status == "complete":
        return None
    scratchpad = getattr(state, "scratchpad", None)
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    schema_failure = scratchpad.get("_last_write_session_schema_failure")
    counters = getattr(state, "stagnation_counters", None)
    counters = counters if isinstance(counters, dict) else {}
    threshold = max(1, int(threshold or 3))
    stalled_counters: list[str] = []
    for name in ("no_actionable_progress", "repeat_patch", "repeat_command"):
        try:
            count = int(counters.get(name, 0) or 0)
        except (TypeError, ValueError):
            count = 0
        if count >= threshold:
            stalled_counters.append(f"{name}={count}")
    if not isinstance(schema_failure, dict) and not stalled_counters:
        return None

    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    target = str(getattr(session, "write_target_path", "") or "").strip()
    evidence_bits = [f"session_id={session_id}", f"status={status or 'open'}"]
    if target:
        evidence_bits.append(f"target={target}")
    if isinstance(schema_failure, dict):
        tool = str(schema_failure.get("tool_name") or "").strip()
        if tool:
            evidence_bits.append(f"schema_failure_tool={tool}")
    evidence_bits.extend(stalled_counters)
    return FamaSignal(
        kind=FamaFailureKind.WRITE_SESSION_STALL,
        severity=2,
        source="write_session",
        evidence="; ".join(item for item in evidence_bits if item),
        step=current_step(state),
        tool_name=str(schema_failure.get("tool_name") or "").strip()
        if isinstance(schema_failure, dict)
        else None,
    )


def record_bad_tool_arg_failure(state: Any, *, tool_name: str, result: Any) -> int:
    if not _is_bad_tool_arg_failure(result):
        return 0
    payload = get_fama_state(state)
    counts = payload.get("bad_tool_arg_counts")
    if not isinstance(counts, dict):
        counts = {}
    key = _bad_arg_key(tool_name, result)
    try:
        count = int(counts.get(key, 0) or 0) + 1
    except (TypeError, ValueError):
        count = 1
    counts[key] = count
    payload["bad_tool_arg_counts"] = dict(list(counts.items())[-24:])
    return count


def detect_bad_tool_args(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
    threshold: int = 2,
) -> FamaSignal | None:
    if not _is_bad_tool_arg_failure(result):
        return None
    threshold = max(1, int(threshold or 2))
    count = _bad_arg_observation_count(state, tool_name=tool_name, result=result)
    schema_nudges = _scratchpad_int(state, "_schema_validation_nudges", 0)
    if max(count, schema_nudges) < threshold:
        return None
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    reason = str(metadata.get("reason") or metadata.get("error_type") or "validation_error").strip()
    evidence = f"repeated schema/validation failures for {tool_name or 'tool'}"
    if reason:
        evidence += f"; reason={reason}"
    return FamaSignal(
        kind=FamaFailureKind.BAD_TOOL_ARGS,
        severity=2,
        source="dispatcher",
        evidence=evidence,
        step=current_step(state),
        tool_name=str(tool_name or "").strip() or None,
        operation_id=operation_id,
    )


def detect_tool_output_misread(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    if str(tool_name or "").strip() != "task_complete":
        return None
    if bool(getattr(result, "success", False)):
        return None
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    if _early_stop_evidence(metadata, error=str(getattr(result, "error", "") or ""), state=state):
        return None
    reason = str(metadata.get("reason") or "").strip()
    error = str(getattr(result, "error", "") or "")
    lowered_error = error.lower()
    if reason not in _OUTPUT_MISREAD_REASONS and not any(
        marker in lowered_error
        for marker in (
            "missing from the latest tool output",
            "contradicts the latest tool output",
            "not supported by the latest tool output",
            "answer was not found in tool output",
        )
    ):
        return None
    evidence = f"task_complete contradicted or misread tool output"
    if reason:
        evidence += f"; reason={reason}"
    return FamaSignal(
        kind=FamaFailureKind.TOOL_OUTPUT_MISREAD,
        severity=2,
        source="tool_result",
        evidence=evidence,
        step=current_step(state),
        tool_name="task_complete",
        operation_id=operation_id,
    )


def detect_backend_stream_halt(state: Any, *, threshold: int = 2) -> FamaSignal | None:
    scratchpad = getattr(state, "scratchpad", None)
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    halted = bool(scratchpad.get("_last_stream_halted_without_done"))
    recovery = scratchpad.get("_last_backend_recovery")
    recovery = recovery if isinstance(recovery, dict) else {}
    recovery_reason = str(
        scratchpad.get("_last_stream_halt_reason")
        or recovery.get("reason")
        or recovery.get("status")
        or ""
    ).strip()
    recovery_text = " ".join(str(value).lower() for value in recovery.values())
    streamish_recovery = any(marker in recovery_text for marker in ("stream", "first_token", "timeout", "halt"))
    if not halted and not streamish_recovery:
        return None
    payload = get_fama_state(state)
    try:
        count = int(payload.get("backend_stream_halt_count", 0) or 0) + 1
    except (TypeError, ValueError):
        count = 1
    payload["backend_stream_halt_count"] = count
    threshold = max(1, int(threshold or 2))
    if count < threshold:
        return None
    evidence = f"backend stream halted before tool/final output; count={count}"
    if recovery_reason:
        evidence += f"; reason={recovery_reason}"
    return FamaSignal(
        kind=FamaFailureKind.BACKEND_STREAM_HALT,
        severity=3,
        source="backend_recovery",
        evidence=evidence,
        step=current_step(state),
    )


def detect_context_drift(state: Any) -> FamaSignal | None:
    if _has_human_gate(state):
        return None
    scratchpad = getattr(state, "scratchpad", None)
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    handoff = scratchpad.get("_last_task_handoff")
    handoff = handoff if isinstance(handoff, dict) else {}
    if not bool(scratchpad.get("_task_divergence_nudged")) and not bool(handoff.get("divergence_nudged")):
        return None
    task_mode = str(handoff.get("task_mode") or "").strip()
    original_task = str(handoff.get("raw_user_task") or handoff.get("original_task") or "").strip()
    evidence = "current action diverged from active task boundary"
    if task_mode:
        evidence += f"; task_mode={task_mode}"
    if original_task:
        evidence += f"; original_task={original_task[:120]}"
    return FamaSignal(
        kind=FamaFailureKind.CONTEXT_DRIFT,
        severity=2,
        source="task_boundary",
        evidence=evidence,
        step=current_step(state),
    )


def latest_verifier_passed(state: Any, *, result: Any | None = None) -> bool:
    metadata = getattr(result, "metadata", None) if result is not None else None
    metadata = metadata if isinstance(metadata, dict) else {}
    verifier = _metadata_verifier(metadata)
    if verifier is not None:
        return _verdict_is_pass(verifier.get("verdict"))
    verifier_verdict = str(metadata.get("verifier_verdict") or "").strip()
    if verifier_verdict:
        return _verdict_is_pass(verifier_verdict)
    current_verifier = getattr(state, "current_verifier_verdict", None)
    verdict = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
    if isinstance(verdict, dict) and verdict:
        return _verdict_is_pass(verdict.get("verdict"))
    return False


def _repeated_tool_from_history(state: Any, *, threshold: int) -> str | None:
    history = getattr(state, "tool_history", None)
    if not isinstance(history, list) or not history:
        return None
    counts = Counter(str(item or "") for item in history if str(item or "").strip())
    for fingerprint, count in counts.most_common():
        if count < threshold:
            continue
        tool_name = fingerprint.split("|", 1)[0].strip()
        if tool_name:
            return tool_name
    return None


def _early_stop_evidence(metadata: dict[str, Any], *, error: str, state: Any) -> str:
    verifier = _metadata_verifier(metadata)
    if _verifier_failed(verifier):
        return _verifier_evidence(verifier)

    lowered_error = error.lower()
    if "latest verifier verdict is still failing" in lowered_error:
        return "task_complete rejected because the latest verifier verdict is failing"

    pending = metadata.get("pending_acceptance_criteria")
    if isinstance(pending, list) and pending:
        return "task_complete rejected with pending acceptance criteria"
    checklist = metadata.get("acceptance_checklist")
    if _checklist_has_pending(checklist):
        return "task_complete rejected with unsatisfied acceptance checklist"

    scratchpad = getattr(state, "scratchpad", None)
    scratch_verdict = scratchpad.get("_last_verifier_verdict") if isinstance(scratchpad, dict) else None
    if _verifier_failed(scratch_verdict):
        return _verifier_evidence(scratch_verdict)
    return ""


def _metadata_verifier(metadata: dict[str, Any]) -> dict[str, Any] | None:
    verdict = metadata.get("last_verifier_verdict")
    return verdict if isinstance(verdict, dict) and verdict else None


def _verifier_failed(verifier: Any) -> bool:
    if not isinstance(verifier, dict) or not verifier:
        return False
    verdict = str(verifier.get("verdict") or "").strip().lower()
    return bool(verdict) and verdict != "pass"


def _verdict_is_pass(value: Any) -> bool:
    return str(value or "").strip().lower() == "pass"


def _verifier_evidence(verifier: dict[str, Any]) -> str:
    verdict = str(verifier.get("verdict") or "unknown").strip()
    target = str(verifier.get("command") or verifier.get("target") or "").strip()
    if target:
        return f"task_complete rejected with verifier verdict {verdict}: {target}"
    return f"task_complete rejected with verifier verdict {verdict}"


def _checklist_has_pending(checklist: Any) -> bool:
    if not isinstance(checklist, list):
        return False
    for item in checklist:
        if isinstance(item, dict) and not bool(item.get("satisfied")):
            return True
    return False


def _is_bad_tool_arg_failure(result: Any) -> bool:
    if bool(getattr(result, "success", False)):
        return False
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    reason = str(metadata.get("reason") or metadata.get("error_type") or metadata.get("source") or "").strip().lower()
    if reason in _BAD_ARG_REASONS:
        return True
    error = str(getattr(result, "error", "") or "").lower()
    return any(marker in error for marker in _BAD_ARG_ERROR_MARKERS)


def _bad_arg_observation_count(state: Any, *, tool_name: str, result: Any) -> int:
    payload = get_fama_state(state)
    counts = payload.get("bad_tool_arg_counts")
    if not isinstance(counts, dict):
        return 0
    try:
        return int(counts.get(_bad_arg_key(tool_name, result), 0) or 0)
    except (TypeError, ValueError):
        return 0


def _bad_arg_key(tool_name: str, result: Any) -> str:
    error = str(getattr(result, "error", "") or "").strip().lower()
    if "missing required field" in error:
        bucket = "missing_required"
    elif "expected type" in error:
        bucket = "type_mismatch"
    elif "object" in error:
        bucket = "not_object"
    else:
        metadata = getattr(result, "metadata", None)
        metadata = metadata if isinstance(metadata, dict) else {}
        bucket = str(metadata.get("reason") or metadata.get("error_type") or "validation").strip().lower()
    return f"{str(tool_name or '').strip()}:{bucket}"


def _scratchpad_int(state: Any, key: str, default: int) -> int:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return default
    try:
        return int(scratchpad.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _has_human_gate(state: Any) -> bool:
    pending = getattr(state, "pending_interrupt", None)
    if isinstance(pending, dict) and pending:
        return True
    scratchpad = getattr(state, "scratchpad", None)
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    for key in ("_human_approval_pending", "_shell_human_retry_state"):
        if scratchpad.get(key):
            return True
    return False
