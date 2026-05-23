from __future__ import annotations

from collections import Counter
from typing import Any

from ..diagnostic_tasks import diagnostic_failure_task
from .signals import FamaFailureKind, FamaSignal, current_step, get_fama_state

_LOOP_COUNTERS = ("no_progress", "no_actionable_progress", "repeat_command", "repeat_patch")
_READ_LOOP_TOOLS = {"artifact_read", "file_read", "dir_list", "ssh_file_read", "web_fetch"}
_REMOTE_CONFUSION_REASONS = {
    "remote_path_requires_ssh_exec",
    "remote_path_requires_typed_ssh_file_tool",
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
WRONG_PATH_MARKERS = (
    "no such file or directory",
    "cannot access",
    "not found",
    "filenotfounderror",
    "path does not exist",
)
_TEST_OUTPUT_MARKERS = (
    "assertionerror",
    "traceback",
    "failed (failures=",
    "failed (errors=",
    "ran ",
    "pytest",
    "unittest",
)
WRITE_TOOLS = {"file_write", "file_append", "ssh_file_write"}
_TEST_FAILURE_MARKERS = (
    "failed",
    "error",
    "traceback",
    "assertionerror",
    "pytest",
    "test failed",
)
_ZERO_TEST_MARKERS = (
    "ran 0 tests",
    "no tests ran",
    "collected 0 items",
    "0 tests collected",
    "no tests collected",
)


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
        failure_class="verifier_failed"
        if "verifier verdict" in evidence.lower() or "latest verifier" in evidence.lower()
        else "completion_blocked",
        next_safe_action="Read the blocking verifier or acceptance evidence, patch one narrow cause, then retry the focused check.",
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

    repeated_tool, repeated_fingerprint = _repeated_tool_from_history(state, threshold=threshold)
    evidence_bits = [f"{name}={count}" for name, count in tripped]
    if repeated_tool:
        evidence_bits.append(f"repeated_tool={repeated_tool}")
    if repeated_fingerprint:
        evidence_bits.append(f"repeated_fingerprint={repeated_fingerprint[:160]}")
    if repeated_tool in _READ_LOOP_TOOLS:
        _record_read_loop_recovery_payload(
            state,
            tool_name=repeated_tool,
            fingerprint=repeated_fingerprint,
        )
    return FamaSignal(
        kind=FamaFailureKind.LOOPING,
        severity=2,
        source="loop_guard",
        evidence="; ".join(evidence_bits),
        step=current_step(state),
        tool_name=repeated_tool,
        failure_class="repeated_action" if repeated_tool else "no_progress",
        next_safe_action=_next_action_for_repeated_loop(
            repeated_tool=repeated_tool,
            counters=tripped,
        ),
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
        failure_class="wrong_path",
        next_safe_action="Verify the correct local or remote path/scope, then retry with the verified target.",
    )


def detect_remote_verification_pending(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    reason = str(metadata.get("reason") or "").strip()
    if reason != "remote_mutation_requires_verification":
        return None
    if bool(getattr(result, "success", False)):
        return None
    if str(tool_name or "").strip() != "task_complete":
        return None
    host = str(metadata.get("host") or "").strip()
    pending = metadata.get("pending_paths")
    if not isinstance(pending, list):
        pending = metadata.get("guessed_paths")
    pending_paths = [str(path).strip() for path in pending or [] if str(path).strip()]
    evidence = "remote mutation completion blocked until remote read-back verification"
    if host:
        evidence += f"; host={host}"
    if pending_paths:
        evidence += f"; pending_path={pending_paths[0]}"
    return FamaSignal(
        kind=FamaFailureKind.REMOTE_VERIFICATION_PENDING,
        severity=2,
        source="tool_result",
        evidence=evidence,
        step=current_step(state),
        tool_name="task_complete",
        operation_id=operation_id,
        failure_class="remote_verification_pending",
        next_safe_action="Use the required remote read-back verification tool/path to verify the mutation, then retry completion.",
    )


def detect_wrong_path(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    if str(tool_name or "").strip() == "task_complete":
        return None
    if bool(getattr(result, "success", False)):
        return None
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    if str(metadata.get("reason") or "").strip() in _REMOTE_CONFUSION_REASONS:
        return None
    combined = _result_text(result, metadata=metadata).lower()
    if str(tool_name or "").strip() in {"shell_exec", "ssh_exec"} and _looks_like_test_failure_output(combined):
        return None
    if _is_patch_target_miss(tool_name, combined):
        return None
    marker = next((item for item in WRONG_PATH_MARKERS if item in combined), "")
    if not marker:
        return None
    path = _path_from_metadata_or_args(metadata)
    evidence = f"path failure marker={marker}"
    if path:
        evidence += f"; path={path}"
    return FamaSignal(
        kind=FamaFailureKind.REMOTE_LOCAL_CONFUSION,
        severity=2,
        source="tool_result",
        evidence=evidence,
        step=current_step(state),
        tool_name=str(tool_name or "").strip() or None,
        operation_id=operation_id,
        failure_class="wrong_path",
        next_safe_action="Run a narrow directory/file check in the correct scope, then retry with the verified path.",
    )


def _looks_like_test_failure_output(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    if "not found in" in lowered and any(marker in lowered for marker in _TEST_OUTPUT_MARKERS):
        return True
    if "assertionerror" in lowered and any(marker in lowered for marker in ("failed", "traceback", "ran ")):
        return True
    return False


def _is_patch_target_miss(tool_name: str, combined_result_text: str) -> bool:
    tool = str(tool_name or "").strip()
    if tool not in {"file_patch", "ast_patch", "ssh_file_patch"}:
        return False
    text = str(combined_result_text or "").lower()
    patch_miss_markers = (
        "patch target text was not found",
        "target text was not found",
        "replacement bounds were not found",
        "bounded region was not found",
        "class `",
        "function `",
        "method `",
    )
    return any(marker in text for marker in patch_miss_markers)


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
        failure_class="write_session_stall",
        next_safe_action="Inspect the active write session state and continue or repair only the narrow stalled section.",
    )


def looks_like_empty_write(tool_name: str, args: dict[str, Any] | None, result: Any) -> bool:
    if str(tool_name or "").strip() not in WRITE_TOOLS:
        return False
    args = args if isinstance(args, dict) else {}
    has_content_arg = "content" in args or "text" in args
    content = str(args.get("content") or args.get("text") or "")
    if has_content_arg and not content.strip():
        return True
    output_text = _result_text(result).lower()
    return "0 bytes" in output_text or "empty write" in output_text


def detect_empty_write(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    arguments: dict[str, Any] | None = None,
    operation_id: str | None = None,
) -> FamaSignal | None:
    if not looks_like_empty_write(tool_name, arguments, result):
        return None
    args = arguments if isinstance(arguments, dict) else {}
    path = str(args.get("path") or args.get("target") or "").strip()
    evidence = f"{tool_name or 'write tool'} produced empty or near-empty content"
    if path:
        evidence += f"; path={path}"
    return FamaSignal(
        kind=FamaFailureKind.WRITE_SESSION_STALL,
        severity=2,
        source="tool_result",
        evidence=evidence,
        step=current_step(state),
        tool_name=str(tool_name or "").strip() or None,
        operation_id=operation_id,
        failure_class="empty_write",
        next_safe_action="Verify file size/content, then patch or retry the smallest write target.",
    )


def detect_verifier_failure_from_result(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    has_result_verifier = any(
        key in metadata
        for key in ("last_verifier_verdict", "verifier", "verifier_verdict")
    )
    if str(tool_name or "").strip() not in {"shell_exec", "ssh_exec"} and not has_result_verifier:
        return None
    verifier = _verifier_from_result_or_state(state, result=result)
    if not _verifier_failed(verifier):
        return None
    if diagnostic_failure_task(state):
        return None
    failure_class = detect_test_failure_from_verdict(verifier) or "verifier_failed"
    # Distinguish timeout / infinite-loop from ordinary test failures
    if failure_class == "verifier_failed":
        failure_mode = str(verifier.get("failure_mode") or "").strip().lower()
        if failure_mode == "environment":
            stdout = str(verifier.get("key_stdout") or "").strip()
            stderr = str(verifier.get("key_stderr") or "").strip()
            error_text = " ".join([stdout, stderr]).lower()
            if "timed out" in error_text or "timeout" in error_text:
                if not stdout and not stderr:
                    failure_class = "infinite_loop_suspected"
                else:
                    failure_class = "verifier_timeout"
    return FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="verifier",
        evidence=_verifier_evidence(verifier),
        step=current_step(state),
        tool_name=str(tool_name or "").strip() or None,
        operation_id=operation_id,
        failure_class=failure_class,
        next_safe_action="Read the failing output and patch one narrow cause, then rerun the smallest check.",
    )


def detect_test_failure_from_verdict(verdict: Any) -> str | None:
    if not isinstance(verdict, dict) or not verdict:
        return None
    command = str(verdict.get("command") or verdict.get("target") or "").lower()
    output = " ".join(
        str(verdict.get(key) or "").lower()
        for key in ("key_stdout", "key_stderr", "failure_mode")
    )
    if _looks_like_zero_tests(output):
        return "zero_tests_discovered"
    if "pytest" in command or "test" in command:
        return "test_failed"
    if any(marker in output for marker in _TEST_FAILURE_MARKERS) and "test" in output:
        return "test_failed"
    return None


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
        failure_class="tool_schema_invalid",
        next_safe_action="Use the tool schema and emit one minimal valid call.",
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
        failure_class="hallucinated_assumption",
        next_safe_action="Ground the next action in the latest tool output instead of unsupported assumptions.",
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
        failure_class="backend_stream_failure",
        next_safe_action="Recover with a smaller, explicit next action before retrying generation.",
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
        failure_class="context_missing",
        next_safe_action="Return to the active user task boundary and fetch only the missing evidence needed for the next action.",
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


def _repeated_tool_from_history(state: Any, *, threshold: int) -> tuple[str | None, str | None]:
    history = getattr(state, "tool_history", None)
    if not isinstance(history, list) or not history:
        return None, None
    counts = Counter(str(item or "") for item in history if str(item or "").strip())
    for fingerprint, count in counts.most_common():
        if count < threshold:
            continue
        tool_name = fingerprint.split("|", 1)[0].strip()
        if tool_name:
            return tool_name, fingerprint
    return None, None


def _record_read_loop_recovery_payload(
    state: Any,
    *,
    tool_name: str,
    fingerprint: str | None,
) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    target = _read_loop_target_from_fingerprint(fingerprint)
    scratchpad["_read_loop_recovery_payload"] = {
        "tool_name": tool_name,
        "fingerprint": str(fingerprint or "")[:240],
        "target": target,
        "created_at_step": current_step(state),
        "last_evidence_summary": _read_loop_evidence_summary(state, tool_name=tool_name, target=target),
        "allowed_next_action": (
            f"Do not call {tool_name} with the same target again. Use visible evidence to patch, "
            "run a focused verifier, inspect a different missing target, or complete if enough evidence is present."
        ),
    }


def _read_loop_target_from_fingerprint(fingerprint: str | None) -> str:
    text = str(fingerprint or "").strip()
    if not text:
        return ""
    if "|" in text:
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if len(parts) >= 2:
            return parts[1][:180]
    return text[:180]


def _read_loop_evidence_summary(state: Any, *, tool_name: str, target: str) -> str:
    if tool_name == "artifact_read":
        artifacts = getattr(state, "artifacts", None)
        if isinstance(artifacts, dict) and target:
            artifact = artifacts.get(target)
            title = str(getattr(artifact, "title", "") or "").strip()
            metadata = getattr(artifact, "metadata", None)
            if isinstance(metadata, dict):
                lines = metadata.get("line_count") or metadata.get("total_lines")
                if title and lines:
                    return f"Artifact {target} ({title}) has {lines} lines; use already-read preview/coverage before rereading."
                if title:
                    return f"Artifact {target} ({title}) was already read recently."
        return f"Recent artifact_read calls already targeted {target or 'the same artifact/range'}."
    if target:
        return f"Recent {tool_name} calls already targeted {target}."
    return f"Recent {tool_name} calls repeated without producing new evidence."


def _next_action_for_repeated_loop(
    *,
    repeated_tool: str | None,
    counters: list[tuple[str, int]],
) -> str:
    tool_name = str(repeated_tool or "").strip()
    counter_names = {name for name, _count in counters}
    if tool_name in _READ_LOOP_TOOLS:
        return (
            f"Do not call {tool_name} with the same target again. Use the already-visible evidence "
            "to make the next patch, run a focused verifier, or inspect a different missing target."
        )
    if tool_name in {"shell_exec", "ssh_exec"}:
        return (
            f"Do not rerun the same {tool_name} command unchanged. Read its prior output, then change "
            "one variable: path, command, patch, or verifier scope."
        )
    if tool_name in WRITE_TOOLS or "repeat_patch" in counter_names:
        return (
            "Do not retry the same write/patch unchanged. Inspect the current staged/target content, "
            "then make the smallest different patch."
        )
    if "no_progress" in counter_names or "no_actionable_progress" in counter_names:
        return (
            "Stop the current loop, state one concrete missing evidence item, and take a different "
            "tool action that can produce it."
        )
    return "Reuse the prior evidence and take the next smallest different action."


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


def _verifier_from_result_or_state(state: Any, *, result: Any | None) -> dict[str, Any] | None:
    metadata = getattr(result, "metadata", None) if result is not None else None
    metadata = metadata if isinstance(metadata, dict) else {}
    verifier = _metadata_verifier(metadata)
    if verifier is not None:
        return verifier
    nested = metadata.get("verifier")
    if isinstance(nested, dict) and nested:
        return nested
    verifier_verdict = str(metadata.get("verifier_verdict") or "").strip()
    if verifier_verdict:
        return {
            "verdict": verifier_verdict,
            "command": str(metadata.get("verifier_command") or metadata.get("command") or ""),
            "target": str(metadata.get("verifier_target") or metadata.get("target") or ""),
            "exit_code": metadata.get("verifier_exit_code"),
            "key_stdout": str(metadata.get("verifier_stdout") or ""),
            "key_stderr": str(metadata.get("verifier_stderr") or ""),
            "failure_mode": str(metadata.get("failure_mode") or ""),
        }
    current_verifier = getattr(state, "current_verifier_verdict", None)
    verdict = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
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
    output = " ".join(
        str(verifier.get(key) or "").lower()
        for key in ("key_stdout", "key_stderr", "failure_mode")
    )
    qualifier = "; zero tests discovered" if _looks_like_zero_tests(output) else ""
    if target:
        return f"task_complete rejected with verifier verdict {verdict}: {target}{qualifier}"
    return f"task_complete rejected with verifier verdict {verdict}{qualifier}"


def _looks_like_zero_tests(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _ZERO_TEST_MARKERS)


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


def _result_text(result: Any, *, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata if isinstance(metadata, dict) else getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    parts = [
        str(getattr(result, "error", "") or ""),
        str(getattr(result, "output", "") or ""),
    ]
    for key in (
        "error",
        "message",
        "reason",
        "path",
        "target",
        "stderr",
        "stdout",
        "verifier_stderr",
        "verifier_stdout",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            parts.append(str(value))
    output = getattr(result, "output", None)
    if isinstance(output, dict):
        for key in ("stderr", "stdout", "message", "error"):
            value = output.get(key)
            if value not in (None, ""):
                parts.append(str(value))
    return "\n".join(parts)


def _path_from_metadata_or_args(metadata: dict[str, Any]) -> str:
    for key in ("path", "target", "file", "filename"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    args = metadata.get("arguments")
    if isinstance(args, dict):
        for key in ("path", "target", "file", "filename"):
            value = str(args.get(key) or "").strip()
            if value:
                return value
    return ""

def detect_tool_plan_hard_route(state: Any) -> bool:
    """Set scratchpad flag when evidence-starved conditions warrant hard-routing to ToolPlan."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    metrics = scratchpad.get("_recovery_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    # Condition 1: model tried patch before reading (evidence_before_patch)
    if int(metrics.get("tool_plan_evidence_before_patch_count", 0) or 0) > 0:
        scratchpad["_fama_force_tool_plan_next_turn"] = True
        return True

    # Condition 2: repeated reads without progress
    if int(metrics.get("tool_plan_repeated_read_count", 0) or 0) > 1:
        scratchpad["_fama_force_tool_plan_next_turn"] = True
        return True

    # Condition 3: wrong path count > 0
    if int(metrics.get("tool_plan_wrong_path_count", 0) or 0) > 0:
        scratchpad["_fama_force_tool_plan_next_turn"] = True
        return True

    # Condition 4: active subtask evidence_count < 2 and failure_count > 0
    subtask_ledger = getattr(state, "subtask_ledger", None)
    if subtask_ledger is not None:
        try:
            active = subtask_ledger.infer_or_create_active_subtask()
            evidence_count = len(getattr(active, "evidence_items", []) or [])
            failure_count = len(getattr(active, "failure_events", []) or [])
            if evidence_count < 2 and failure_count > 0:
                scratchpad["_fama_force_tool_plan_next_turn"] = True
                return True
        except Exception:
            pass

    return False
