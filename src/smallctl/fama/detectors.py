from __future__ import annotations

import re
from typing import Any

from ..diagnostic_tasks import diagnostic_failure_task
from .detector_classifiers import (
    WRONG_PATH_MARKERS,
    WRITE_TOOLS,
    _BAD_ARG_ERROR_MARKERS,
    _BAD_ARG_REASONS,
    _LOOP_COUNTERS,
    _OUTPUT_MISREAD_REASONS,
    _READ_LOOP_TOOLS,
    _REMOTE_CONFUSION_REASONS,
    _TEST_FAILURE_MARKERS,
    _is_patch_target_miss,
    _looks_like_test_failure_output,
    _looks_like_zero_tests,
    _metadata_verifier,
    _verdict_is_pass,
    _verifier_failed,
)
from .signals import FamaFailureKind, FamaSignal, current_step, get_fama_state
from .detectors_support import (
    _scratchpad_int,
    _early_stop_evidence,
    _next_action_for_repeated_loop,
    _record_read_loop_recovery_payload,
    _repeated_tool_from_history,
    _verifier_evidence,
    _verifier_from_result_or_state,
    _has_human_gate,
    _result_text,
    _path_from_metadata_or_args,
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
    if reason in {"task_complete_blocked_in_staged_execution", "session_incomplete",
                  "remote_mutation_requires_verification"}:
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


def detect_identical_tool_loop(state: Any, *, threshold: int = 3) -> FamaSignal | None:
    history = getattr(state, "tool_history", None)
    if not isinstance(history, list) or len(history) < max(1, int(threshold or 3)):
        return None
    threshold = max(1, int(threshold or 3))
    recent = [str(item or "").strip() for item in history[-threshold:]]
    if any(not item for item in recent):
        return None
    if len(set(recent)) != 1:
        return None
    fingerprint = recent[-1]
    tool_name = fingerprint.split("|", 1)[0].strip() or None
    next_action = (
        "Do not call it again unchanged. Explain the blocker or try a different specific fix that can change the prompt/output state."
    )
    return FamaSignal(
        kind=FamaFailureKind.LOOPING,
        severity=2,
        source="loop_guard",
        evidence=fingerprint,
        step=current_step(state),
        tool_name=tool_name,
        failure_class="repeated_action",
        next_safe_action=next_action,
    )


def detect_generic_stuck_loop(
    state: Any,
    *,
    threshold: int = 3,
) -> FamaSignal | None:
    """Fire after N consecutive tool failures or no-progress turns, regardless of fingerprint.

    Unlike detect_identical_tool_loop / detect_repeated_tool_loop, this detector
    does not require the exact same tool/arguments to repeat. It catches models
    that churn on slightly different failing commands or stall without making
    progress.
    """
    counters = getattr(state, "stagnation_counters", None) or {}
    no_progress = int(counters.get("no_actionable_progress", 0) or 0)
    threshold = max(1, int(threshold or 3))

    # Count consecutive failed tool outcomes at the tail of tool_history.
    # tool_history entries are formatted as: tool_name|args_json|outcome
    consecutive_failures = 0
    tool_name: str | None = None
    for entry in reversed(getattr(state, "tool_history", []) or []):
        parts = str(entry or "").split("|")
        if len(parts) < 3:
            break
        this_tool = parts[0].strip() or None
        outcome = parts[2].strip().lower()
        if outcome.startswith("error") or outcome in {"false", "failure", "fail"}:
            consecutive_failures += 1
            if tool_name is None:
                tool_name = this_tool
        elif outcome in {"success", "true", "ok"}:
            break
        else:
            break

    recent_errors = getattr(state, "recent_errors", None) or []
    recent_error_count = sum(1 for item in recent_errors[-threshold:] if str(item or "").strip())

    # Require both no-progress and some evidence of repeated failure.
    if no_progress < threshold:
        return None
    if consecutive_failures < 2 and recent_error_count < 2:
        return None

    evidence_bits = [f"no_actionable_progress={no_progress}", f"consecutive_failures={consecutive_failures}"]
    if recent_error_count:
        evidence_bits.append(f"recent_errors={recent_error_count}")
    if tool_name:
        evidence_bits.append(f"last_tool={tool_name}")

    return FamaSignal(
        kind=FamaFailureKind.LOOPING,
        severity=2,
        source="loop_guard",
        evidence="; ".join(evidence_bits),
        step=current_step(state),
        tool_name=tool_name,
        failure_class="no_progress",
        next_safe_action=(
            "Progress has stalled. Gather one fresh piece of evidence, change approach, "
            "or ask for help instead of repeating similar actions."
        ),
    )


def detect_interactive_installer_stall(state: Any, *, threshold: int = 2) -> FamaSignal | None:
    history = [str(item or "").strip() for item in getattr(state, "tool_history", []) or [] if str(item or "").strip()]
    threshold = max(1, int(threshold or 2))
    if len(history) < (threshold * 2):
        return None

    pairs: list[tuple[str, str, str, str]] = []
    for index in range(len(history) - 1):
        send_entry = history[index]
        read_entry = history[index + 1]
        send_parts = send_entry.split("|")
        read_parts = read_entry.split("|")
        if len(send_parts) < 3 or len(read_parts) < 4:
            continue
        if send_parts[0].strip() != "ssh_session_send" or read_parts[0].strip() != "ssh_session_read":
            continue
        send_session = send_parts[1].strip()
        read_session = read_parts[1].strip()
        prompt = read_parts[2].strip()
        status = read_parts[3].strip()
        if not send_session or send_session != read_session:
            continue
        pairs.append((send_session, send_parts[2].strip(), prompt, status))

    if len(pairs) < threshold + 1:
        return None

    tail = pairs[-(threshold + 1):]
    session_id = tail[-1][0]
    prompt = tail[-1][2]
    status = tail[-1][3]
    if not prompt:
        return None
    if any(item[0] != session_id or item[2] != prompt or item[3] != status for item in tail):
        return None

    return FamaSignal(
        kind=FamaFailureKind.INTERACTIVE_SESSION_STALL,
        severity=2,
        source="interactive_session",
        evidence=(
            f"ssh_session_read returned the same prompt after repeated sends; session={session_id}; prompt={prompt}; status={status}"
        ),
        step=current_step(state),
        tool_name="ssh_session_read",
        failure_class="interactive_session_stall",
        next_safe_action=(
            "Do not keep polling the same prompt. Retry one `ssh_session_send` with explicit submit semantics, inspect the exact prompt state, or explain the blocker."
        ),
    )


def detect_weak_verifier_logic(
    state: Any,
    *,
    tool_name: str,
    result: Any,
) -> FamaSignal | None:
    if str(tool_name or "").strip() != "ssh_exec":
        return None
    if not bool(getattr(result, "success", False)):
        return None
    scratchpad = getattr(state, "scratchpad", None)
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    verdict = scratchpad.get("_last_verifier_verdict") or getattr(state, "last_verifier_verdict", None)
    verdict = verdict if isinstance(verdict, dict) else {}
    if str(verdict.get("verdict") or "").strip().lower() != "pass":
        return None
    combined = _result_text(result).lower()
    if not any(marker in combined for marker in ("(y/n)", "continue?", "password:", "awaiting input", "prompt")):
        return None
    return FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="verifier",
        evidence="Verifier passed even though ssh_exec output still shows an interactive prompt.",
        step=current_step(state),
        tool_name="ssh_exec",
        failure_class="verifier_failed",
        next_safe_action="Do not treat the install as complete. Continue the interactive flow or verify the real post-install outcome first.",
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


def detect_patch_target_not_found(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    if tool_name not in {"file_patch", "ssh_file_patch"}:
        return None
    if bool(getattr(result, "success", False)):
        return None
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    error_kind = str(metadata.get("error_kind") or "").strip().lower()
    error_text = str(getattr(result, "error", "") or metadata.get("error") or "").strip().lower()
    if error_kind != "patch_target_not_found" and "target text" not in error_text:
        return None
    return FamaSignal(
        kind=FamaFailureKind.BAD_TOOL_ARGS,
        severity=2,
        source="tool_result",
        evidence=str(getattr(result, "error", "") or "patch target text was not found"),
        step=current_step(state),
        tool_name=tool_name,
        operation_id=operation_id,
        failure_class="patch_target_not_found",
        next_safe_action=(
            "Use the fresh file read, choose a smaller exact span copied from current content, "
            "or verify the intended edit is already applied before retrying file_patch."
        ),
        suggested_mitigations=["patch_target_not_found_capsule", "evidence_reuse_capsule"],
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
    evidence = "task_complete contradicted or misread tool output"
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


def detect_model_output_degenerate_loop(state: Any) -> FamaSignal | None:
    """Fire immediately when the most recent stream halt was a degenerate loop.

    Unlike the generic backend_stream_halt detector, this does not wait for a
    repeat count: a model emitting unrecoverable control-tag repetition is a
    severe failure mode that warrants an immediate mitigation capsule.
    """
    scratchpad = getattr(state, "scratchpad", None)
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    halt_reason = str(scratchpad.get("_last_stream_halt_reason") or "").strip()
    if halt_reason != "model_output_degenerate_loop":
        return None
    halt_details = scratchpad.get("_last_stream_halt_details")
    halt_details = halt_details if isinstance(halt_details, dict) else {}
    repeated_phrase = str(halt_details.get("repeated_phrase") or "").strip()
    evidence = "model output degenerated into a repetition loop"
    if repeated_phrase:
        evidence += f"; repeated_phrase={repeated_phrase[:120]}"
    return FamaSignal(
        kind=FamaFailureKind.BACKEND_STREAM_HALT,
        severity=3,
        source="backend_recovery",
        evidence=evidence,
        step=current_step(state),
        failure_class="backend_stream_failure",
        next_safe_action="Emit exactly one concrete next action; do not repeat reasoning tags or re-derive context from scratch.",
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
    # Degenerate loops are handled by the dedicated detector above.
    if recovery_reason == "model_output_degenerate_loop":
        return None
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

    # Condition 5: 3+ repeated same-family shell failures warrant tool-plan evidence gathering
    repeated_signal = detect_repeated_failure_pattern(state, threshold=3)
    if repeated_signal is not None and str(repeated_signal.tool_name or "").strip() in {"shell_exec", "ssh_exec"}:
        scratchpad["_fama_force_tool_plan_next_turn"] = True
        return True

    # Condition 6: 3+ repeated filesystem tool failures (e.g., file_read on missing path)
    if repeated_signal is not None and str(repeated_signal.tool_name or "").strip() in {"file_read", "file_write", "dir_list"}:
        scratchpad["_fama_force_tool_plan_next_turn"] = True
        return True

    return False


def _detect_repeated_failure_pattern_from_scratchpad(
    state: Any,
    *,
    threshold: int = 3,
) -> FamaSignal | None:
    """Legacy fallback using _repeated_failure_observations scratchpad."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    observations = scratchpad.get("_repeated_failure_observations")
    if not isinstance(observations, list):
        return None
    current_step_val = current_step(state)
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        count = int(obs.get("count", 0) or 0)
        if count < threshold:
            continue
        last_step = int(obs.get("last_step", 0) or 0)
        if current_step_val - last_step > 5:
            continue
        tool_name = str(obs.get("tool_name") or "").strip()
        pattern = str(obs.get("pattern") or "").strip()
        if not tool_name or not pattern:
            continue
        evidence = f"{tool_name} failed with {pattern} ({count} attempts)"
        failure_class = pattern.replace(" ", "_")
        return FamaSignal(
            kind=FamaFailureKind.LOOPING,
            severity=2,
            source="tool_result",
            evidence=evidence,
            step=current_step_val,
            tool_name=tool_name,
            failure_class=failure_class,
            next_safe_action=(
                f"The same `{tool_name}` call has failed {count} times with the same error. "
                "Stop retrying the identical call. Use ask_human if the path or command is ambiguous, "
                "or switch to a fundamentally different approach."
            ),
            suggested_mitigations=["interactive_tui_capsule", "evidence_reuse_capsule"],
        )
    return None


def detect_repeated_failure_pattern(
    state: Any,
    *,
    threshold: int = 3,
) -> FamaSignal | None:
    """Detect when the same tool fails with the same error pattern >= threshold times.

    Scans the last N tool results directly from state.tool_execution_records.
    Counts identical (tool_name, failure_class, target_host) tuples within the last 5 turns.
    """
    records = getattr(state, "tool_execution_records", None)
    if not isinstance(records, dict) or not records:
        # Fallback to legacy scratchpad observations (used by some tests)
        return _detect_repeated_failure_pattern_from_scratchpad(state, threshold=threshold)

    current_step_val = current_step(state)
    # Collect the last 5 tool results in chronological order
    items = [
        record for record in records.values()
        if isinstance(record, dict) and int(record.get("step_count", 0) or 0) > 0
    ]
    items.sort(key=lambda r: int(r.get("step_count", 0) or 0))
    recent = items[-5:]

    from collections import Counter
    failures: list[tuple[str, str, str]] = []
    for record in recent:
        result = record.get("result")
        if not isinstance(result, dict):
            continue
        if bool(result.get("success")):
            continue
        tool_name = str(record.get("tool_name") or "").strip()
        if not tool_name:
            continue
        metadata = result.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        failure_class = str(metadata.get("failure_mode") or metadata.get("failure_kind") or metadata.get("error_kind") or result.get("status") or "").strip().lower()
        args = record.get("args")
        args = args if isinstance(args, dict) else {}
        host = str(args.get("host") or metadata.get("host") or "").strip().lower()
        failures.append((tool_name, failure_class or "unknown", host))

    if not failures:
        return None

    counts = Counter(failures)
    most_common = counts.most_common(1)
    if not most_common:
        return None
    (key, count) = most_common[0]
    if count < threshold:
        return None

    tool_name, failure_class, host = key
    evidence = f"{tool_name} failed with {failure_class} on {host or 'unknown host'} ({count} attempts in last 5 turns)"
    return FamaSignal(
        kind=FamaFailureKind.LOOPING,
        severity=2,
        source="tool_result",
        evidence=evidence,
        step=current_step_val,
        tool_name=tool_name,
        failure_class=failure_class,
        next_safe_action=(
            f"The same `{tool_name}` call has failed {count} times with the same error. "
            "Stop retrying the identical call. Use ask_human if the path or command is ambiguous, "
            "or switch to a fundamentally different approach."
        ),
        suggested_mitigations=["interactive_tui_capsule", "evidence_reuse_capsule"],
    )


def detect_verifier_failure_mode(
    state: Any,
    *,
    operation_id: str | None = None,
) -> FamaSignal | None:
    """Detect a persistent failure_mode from the last verifier verdict.

    If the harness has set failure_mode (e.g., 'logic', 'path') in
    last_verifier_verdict and the model has not yet resolved it, emit a
    FAMA signal so the failure-aware mitigation system can inject a
    targeted capsule.
    """
    verdict = getattr(state, "last_verifier_verdict", None)
    if not isinstance(verdict, dict):
        return None
    failure_mode = str(verdict.get("failure_mode") or "").strip()
    if not failure_mode:
        return None
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    key = f"_fama_verifier_failure_mode:{failure_mode}"
    count = int(scratchpad.get(key, 0) or 0) + 1
    scratchpad[key] = count
    threshold = 1 if str(getattr(state, "task_mode", "")).startswith("remote") else 2
    if count < threshold:
        return None
    return FamaSignal(
        kind=FamaFailureKind.EARLY_STOP,
        severity=2,
        source="verifier",
        evidence=f"last_verifier_verdict shows failure_mode={failure_mode} (observed {count} times)",
        step=current_step(state),
        tool_name="task_complete",
        operation_id=operation_id,
        failure_class=failure_mode,
        next_safe_action="Read the verifier output carefully, identify the exact failure mode, and patch one narrow cause before retrying.",
        suggested_mitigations=["repair_debug_scaffold", "acceptance_checklist_capsule"],
    )


def detect_preflight_contradiction(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    """Detect when a preflight guard claims a file is NOT FOUND but a same-session
    ssh_file_write verified the exact path on the same host/user."""
    if str(tool_name or "").strip() != "ssh_exec":
        return None
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    preflight = metadata.get("preflight_probes")
    if not isinstance(preflight, dict):
        return None
    if preflight.get("script_exists"):
        return None
    script_path = str(preflight.get("script_path") or "").strip()
    host = str(metadata.get("host") or "").strip().lower()
    user = str(metadata.get("user") or "").strip() or None
    if not script_path or not host:
        return None
    from ..tools.shell_support import _remote_installer_preflight_has_verified_write
    if not _remote_installer_preflight_has_verified_write(
        state, host=host, user=user, script_path=script_path
    ):
        return None
    evidence = (
        f"preflight reported NOT FOUND for {script_path} on {host}, "
        f"but ssh_file_write verified it in the same session"
    )
    return FamaSignal(
        kind=FamaFailureKind.PREFLIGHT_CONTRADICTION,
        severity=2,
        source="preflight",
        evidence=evidence,
        step=current_step(state),
        tool_name="ssh_exec",
        operation_id=operation_id,
        failure_class="preflight_contradiction",
        next_safe_action="The installer path was already verified by ssh_file_write. Retry the installer command without additional path checks.",
        suggested_mitigations=["preflight_contradiction_capsule"],
    )


def detect_repeated_remote_installer_failure(
    state: Any,
    *,
    threshold: int = 2,
) -> FamaSignal | None:
    """Detect repeated remote installer preflight or execution failures."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    preflights = scratchpad.get("_remote_installer_preflight")
    if not isinstance(preflights, dict):
        return None
    failure_count = 0
    last_host = ""
    last_script = ""
    for entry in preflights.values():
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "").strip()
        if status in {"missing_critical_files", "corrupt", "required"}:
            failure_count += 1
            last_host = str(entry.get("host") or "").strip()
            last_script = str(entry.get("script_path") or "").strip()
    if failure_count < threshold:
        return None
    evidence = f"remote installer preflight/execution failed {failure_count} times"
    if last_host:
        evidence += f"; host={last_host}"
    if last_script:
        evidence += f"; script={last_script}"
    return FamaSignal(
        kind=FamaFailureKind.REPEATED_REMOTE_INSTALLER_FAILURE,
        severity=2,
        source="preflight",
        evidence=evidence,
        step=current_step(state),
        tool_name="ssh_exec",
        failure_class="repeated_remote_installer_failure",
        next_safe_action="The remote installer has failed repeatedly. Verify the remote environment state (apt sources, DNS, python3), repair any broken state, and only then retry.",
        suggested_mitigations=["repeated_remote_installer_failure_capsule", "evidence_reuse_capsule"],
    )


def detect_ssh_host_key_verification_failure(
    state: Any,
    *,
    threshold: int = 2,
) -> FamaSignal | None:
    """Detect repeated SSH host-key verification failures and recommend local recovery."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    recovery_state = scratchpad.get("_ssh_auth_recovery_state")
    if not isinstance(recovery_state, dict):
        return None
    threshold = max(1, int(threshold or 2))
    for key, record in recovery_state.items():
        if not isinstance(record, dict):
            continue
        if str(record.get("last_error_class") or "").strip() != "host_key_verification":
            continue
        consecutive_count = int(record.get("consecutive_count") or 0)
        if consecutive_count < threshold:
            continue
        host = str(record.get("host") or key or "").strip()
        if not host:
            continue
        return FamaSignal(
            kind=FamaFailureKind.SSH_HOST_KEY_VERIFICATION,
            severity=2,
            source="tool_result",
            evidence=f"ssh host-key verification failed {consecutive_count} times for {host}",
            step=current_step(state),
            tool_name="ssh_exec",
            failure_class="ssh_host_key_verification",
            next_safe_action=(
                f"SSH host key changed for {host}. Do not patch known_hosts line by line. "
                f"Use approved local `ssh-keygen -R {host} -f ~/.ssh/known_hosts`, then retry SSH only after approval."
            ),
            suggested_mitigations=["ssh_host_key_recovery_capsule"],
        )
    return None


_SSH_HOST_KEY_CHANGED_RE = re.compile(
    r"REMOTE HOST IDENTIFICATION HAS CHANGED|Host key verification failed|Offending .* key in .*known_hosts",
    re.IGNORECASE | re.DOTALL,
)
_SSH_KEYGEN_REMOVE_RE = re.compile(r"ssh-keygen\s+[^\n]*\s-R\s+['\"]?(?P<host>[A-Za-z0-9._:-]+)", re.IGNORECASE)
_OFFENDING_KNOWN_HOST_RE = re.compile(
    r"Offending .* key in (?P<path>[^:\r\n]+known_hosts):(?P<line>\d+)",
    re.IGNORECASE,
)


def detect_ssh_host_key_verification_failure_from_result(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    arguments: dict[str, Any] | None = None,
    operation_id: str | None = None,
) -> FamaSignal | None:
    """Detect an SSH known_hosts trust failure directly from the current tool result."""
    if str(tool_name or "").strip() not in {"ssh_exec", "ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}:
        return None
    if bool(getattr(result, "success", False)):
        return None
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    combined = _result_text(result, metadata=metadata)
    if not _SSH_HOST_KEY_CHANGED_RE.search(combined):
        return None
    args = arguments if isinstance(arguments, dict) else {}
    host = str(args.get("host") or metadata.get("host") or "").strip()
    if not host:
        remove_match = _SSH_KEYGEN_REMOVE_RE.search(combined)
        if remove_match:
            host = str(remove_match.group("host") or "").strip()
    if not host:
        host = _host_key_failure_host_from_scratchpad(state)
    host_label = host or "the host"
    offending = _OFFENDING_KNOWN_HOST_RE.search(combined)
    known_hosts_hint = ""
    if offending:
        known_hosts_hint = f"; offending local file {offending.group('path')} line {offending.group('line')}"
    return FamaSignal(
        kind=FamaFailureKind.SSH_HOST_KEY_VERIFICATION,
        severity=2,
        source="tool_result",
        evidence=f"ssh host-key verification failed for {host_label}{known_hosts_hint}",
        step=current_step(state),
        tool_name=str(tool_name or "").strip(),
        operation_id=operation_id,
        failure_class="ssh_host_key_verification",
        next_safe_action=(
            f"SSH host key changed for {host_label}. Treat `~/.ssh/known_hosts` as a local harness file, "
            "not a remote file. Ask approval, run local `ssh-keygen -R <host> -f ~/.ssh/known_hosts`, "
            "then retry SSH only after trust is fixed."
        ),
        suggested_mitigations=["ssh_host_key_recovery_capsule"],
    )


def _host_key_failure_host_from_scratchpad(state: Any) -> str:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return ""
    recovery_state = scratchpad.get("_ssh_auth_recovery_state")
    if not isinstance(recovery_state, dict):
        return ""
    for key, record in recovery_state.items():
        if not isinstance(record, dict):
            continue
        if str(record.get("last_error_class") or "").strip() != "host_key_verification":
            continue
        host = str(record.get("host") or key or "").strip()
        if "@" in host:
            host = host.rsplit("@", 1)[-1].strip()
        if host:
            return host
    return ""


def detect_upstream_install_source_invalid(state: Any) -> FamaSignal | None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    diagnosis = scratchpad.get("_install_source_diagnosis")
    if not isinstance(diagnosis, dict):
        return None
    if not bool(diagnosis.get("public_dns_nxdomain")) or not bool(diagnosis.get("network_ok")):
        return None
    invalid_fetch_count = int(diagnosis.get("invalid_fetch_count", 0) or 0)
    resolve_fail_count = int(diagnosis.get("resolve_fail_count", 0) or 0)
    if invalid_fetch_count <= 0 and resolve_fail_count <= 0:
        return None
    task_text = " ".join(
        str(part or "")
        for part in (
            getattr(getattr(state, "run_brief", None), "original_task", ""),
            getattr(getattr(state, "working_memory", None), "current_goal", ""),
        )
    ).lower()
    if not re.search(r"\b(?:install|setup|deploy|configure)\b", task_text):
        return None
    host = str(diagnosis.get("source_host") or diagnosis.get("nxdomain_host") or "").strip()
    evidence_parts = []
    if invalid_fetch_count > 0:
        evidence_parts.append(f"installer fetches were invalid {invalid_fetch_count} time(s)")
    if host:
        evidence_parts.append(f"package host {host} returned NXDOMAIN on public DNS")
    if resolve_fail_count > 0 and host:
        evidence_parts.append(f"package resolution also failed locally for {host}")
    evidence_parts.append("general network access still works")
    return FamaSignal(
        kind=FamaFailureKind.UPSTREAM_INSTALL_SOURCE_INVALID,
        severity=3,
        source="tool_result",
        evidence="; ".join(evidence_parts),
        step=current_step(state),
        tool_name="web_search",
        failure_class="upstream_install_source_invalid",
        next_safe_action=(
            "Stop local DNS repair and repeated installer retries. Research the current official install path, "
            "ask for approval for an alternate/manual path, or explain the blocker."
        ),
        suggested_mitigations=["source_invalid_install_capsule", "evidence_reuse_capsule", "tool_exposure_narrowing"],
    )


def detect_stale_success_claim(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    """Detect when the model claims task_complete success but it was blocked
    or the objective verifier has not actually passed."""
    if str(tool_name or "").strip() != "task_complete":
        return None
    if bool(getattr(result, "success", False)):
        return None
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    reason = str(metadata.get("reason") or "").strip()
    if reason not in {
        "task_complete_blocked_in_staged_execution",
        "session_incomplete",
        "missing_supported_claim",
        "remote_mutation_requires_verification",
    }:
        return None
    # Check if the model's last assistant message claimed success
    messages = getattr(state, "messages", [])
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages[-6:]):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "").lower()
        if any(phrase in content for phrase in (
            "successfully installed", "installation complete", "pihole is installed",
            "done", "completed successfully", "finished installing",
        )):
            evidence = f"model claimed success but task_complete was blocked; reason={reason}"
            return FamaSignal(
                kind=FamaFailureKind.STALE_SUCCESS_CLAIM,
                severity=2,
                source="tool_result",
                evidence=evidence,
                step=current_step(state),
                tool_name="task_complete",
                operation_id=operation_id,
                failure_class="stale_success_claim",
                next_safe_action="Do not claim success before the objective verifier passes. Verify the actual install outcome with a service check or version command.",
                suggested_mitigations=["acceptance_checklist_capsule", "evidence_reuse_capsule"],
            )
    return None


def detect_objective_verifier_mismatch(
    state: Any,
    *,
    operation_id: str | None = None,
) -> FamaSignal | None:
    """Detect when the last verifier checks something tangential to the user objective
    (e.g. verifying script existence instead of install success)."""
    verdict = getattr(state, "last_verifier_verdict", None)
    if not isinstance(verdict, dict):
        return None
    command = str(verdict.get("command") or verdict.get("target") or "").lower()
    task = str(getattr(getattr(state, "run_brief", None), "original_task", "") or "").strip().lower()
    if not task or not command:
        return None
    # Heuristic: install task but verifier only checks file existence
    install_indicators = ("install", "setup", "deploy", "configure")
    if not any(ind in task for ind in install_indicators):
        return None
    file_only_checks = ("ls -la ", "test -f ", "test -e ", "cat ")
    if not any(cmd in command for cmd in file_only_checks):
        return None
    # Check if the verifier passed
    verdict_label = str(verdict.get("verdict") or "").strip().lower()
    if verdict_label != "pass":
        return None
    evidence = (
        f"verifier only checked file existence ({command}) for an install task; "
        f"objective={task[:80]}"
    )
    return FamaSignal(
        kind=FamaFailureKind.OBJECTIVE_MISMATCH,
        severity=2,
        source="verifier",
        evidence=evidence,
        step=current_step(state),
        tool_name="task_complete",
        operation_id=operation_id,
        failure_class="objective_mismatch",
        next_safe_action="The verifier must match the user objective. For install tasks, verify service status, version, or listening port—not just file existence.",
        suggested_mitigations=["acceptance_checklist_capsule"],
    )


def detect_preexisting_state_as_success(
    state: Any,
    *,
    operation_id: str | None = None,
) -> FamaSignal | None:
    """Detect when the model may be confusing pre-existing remote state with
    successful completion from its own actions."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    # Look for evidence that pre-existing state was found early in the session
    preexisting = scratchpad.get("_preexisting_remote_state_observed")
    if not isinstance(preexisting, dict):
        return None
    # Only trigger if the model later attempted task_complete
    last_tool = None
    messages = getattr(state, "messages", [])
    if isinstance(messages, list):
        for msg in reversed(messages[-10:]):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls") or []
                if isinstance(tool_calls, list) and tool_calls:
                    last_tool = str(tool_calls[-1].get("function", {}).get("name") or "").strip()
                    break
    if last_tool != "task_complete":
        return None
    host = str(preexisting.get("host") or "").strip()
    path = str(preexisting.get("path") or "").strip()
    evidence = "pre-existing remote state may have been treated as task completion"
    if host:
        evidence += f"; host={host}"
    if path:
        evidence += f"; path={path}"
    return FamaSignal(
        kind=FamaFailureKind.PREEXISTING_STATE_AS_SUCCESS,
        severity=2,
        source="task_result",
        evidence=evidence,
        step=current_step(state),
        tool_name="task_complete",
        operation_id=operation_id,
        failure_class="preexisting_state_as_success",
        next_safe_action="Distinguish 'state already existed' from 'I caused the state'. Verify that your actions produced the intended outcome, not that it was already present.",
        suggested_mitigations=["preexisting_state_as_success_capsule", "acceptance_checklist_capsule", "evidence_reuse_capsule"],
    )


def detect_apt_deb822_preflight_blocked(
    state: Any,
    *,
    tool_name: str,
    result: Any,
    operation_id: str | None = None,
) -> FamaSignal | None:
    """Detect when an apt operation is blocked by the deb822 preflight guard.

    If the validator has already been marked clean for this host/user but the
    gate still blocks, emit a high-severity contradiction signal. Otherwise
    emit a low-severity guidance signal for the first occurrence.
    """
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    if metadata.get("reason") != "apt_deb822_preflight_required":
        return None
    host = str(metadata.get("host") or "").strip().lower()
    user = str(metadata.get("user") or "").strip().lower()
    from ..tools.shell_support import _is_deb822_preflight_clean
    if _is_deb822_preflight_clean(state, host=host, user=user):
        return FamaSignal(
            kind=FamaFailureKind.PREFLIGHT_CONTRADICTION,
            severity=3,
            source="preflight",
            evidence=f"apt_deb822 preflight blocked for {host}/{user} but validator was already marked clean in session",
            step=current_step(state),
            tool_name=str(tool_name or "").strip() or None,
            operation_id=operation_id,
            failure_class="apt_deb822_preflight_contradiction",
            next_safe_action="The deb822 validator passed earlier but the apt preflight gate is still blocking. Escalate to ask_human or a larger model.",
            suggested_mitigations=["preflight_contradiction_capsule"],
        )
    return FamaSignal(
        kind=FamaFailureKind.PREFLIGHT_CONTRADICTION,
        severity=1,
        source="preflight",
        evidence=f"apt_deb822 preflight required for {host}/{user}",
        step=current_step(state),
        tool_name=str(tool_name or "").strip() or None,
        operation_id=operation_id,
        failure_class="apt_deb822_preflight_required",
        next_safe_action="Run the standalone deb822 validator first, then retry the apt command.",
        suggested_mitigations=["preflight_contradiction_capsule"],
    )


def detect_loop_rewrite(
    state: Any,
    *,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> FamaSignal | None:
    """Detect when the same file is rewritten multiple times with similar content."""
    if tool_name not in {"file_write", "ssh_file_write"}:
        return None
    args = arguments if isinstance(arguments, dict) else {}
    path = str(args.get("path") or "").strip()
    content = str(args.get("content") or "").strip()
    if not path or not content:
        return None
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    key = f"_loop_rewrite:{path}"
    history = scratchpad.get(key)
    if not isinstance(history, list):
        history = []
    # Store a hash of the content to detect similar rewrites
    import hashlib
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    history.append(content_hash)
    scratchpad[key] = history[-5:]
    if len(history) < 3:
        return None
    # Check if last 3 hashes are identical or very similar
    if len(set(history[-3:])) == 1:
        return FamaSignal(
            kind=FamaFailureKind.LOOPING,
            severity=2,
            source="tool_result",
            evidence=f"{tool_name} has rewritten {path} {len(history)} times with identical content",
            step=current_step(state),
            tool_name=tool_name,
            failure_class="repeated_action",
            next_safe_action="Stop rewriting the same file. Use file_patch for narrow edits, or verify the existing content before overwriting.",
            suggested_mitigations=["micro_plan_capsule", "evidence_reuse_capsule"],
        )
    return None


def detect_verifier_path_misclassification(
    state: Any,
    *,
    operation_id: str | None = None,
) -> FamaSignal | None:
    """Detect when a verifier path failure contradicts a recent ssh_file_write verification."""
    verdict = getattr(state, "last_verifier_verdict", None)
    if not isinstance(verdict, dict):
        return None
    failure_mode = str(verdict.get("failure_mode") or "").strip().lower()
    if failure_mode != "path":
        return None
    command = str(verdict.get("command") or "").strip()
    if not command:
        return None
    # Extract path from command
    words = command.split()
    path = ""
    for word in words:
        if word.startswith("/"):
            path = word
            break
    if not path:
        return None
    recent_messages = getattr(state, "recent_messages", None) or []
    if not isinstance(recent_messages, list):
        return None
    for msg in reversed(recent_messages):
        if msg is None:
            continue
        # Support both dict and object-style messages (e.g. SimpleNamespace)
        if isinstance(msg, dict):
            role = msg.get("role")
            name = msg.get("name")
            metadata = msg.get("metadata") or {}
        else:
            role = getattr(msg, "role", None)
            name = getattr(msg, "name", None)
            metadata = getattr(msg, "metadata", None) or {}
        if role != "tool" or name != "ssh_file_write":
            continue
        if not isinstance(metadata, dict):
            continue
        if not metadata.get("success"):
            continue
        msg_path = str(metadata.get("path") or "").strip()
        if msg_path == path:
            return FamaSignal(
                kind=FamaFailureKind.TOOL_OUTPUT_MISREAD,
                severity=2,
                source="verifier",
                evidence=f"verifier reported path failure for {path}, but ssh_file_write confirmed it exists",
                step=current_step(state),
                tool_name="ssh_exec",
                operation_id=operation_id,
                failure_class="verifier_misclassification",
                next_safe_action="The path was verified by ssh_file_write. The verifier failure is a false negative. Retry the command or use a different verification approach.",
                suggested_mitigations=["evidence_reuse_capsule", "acceptance_checklist_capsule"],
            )
    return None
