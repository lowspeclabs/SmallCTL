from __future__ import annotations

import logging
from typing import Any

from .config import (
    done_gate_on_failure,
    fama_enabled,
    max_active_mitigations,
    signal_window,
)
from .detectors import detect_early_stop_from_result, latest_verifier_passed
from .fingerprints import active_done_gate_fingerprints, passing_verifier_fingerprint
from .detectors import (
    detect_backend_stream_halt,
    detect_bad_tool_args,
    detect_context_drift,
    detect_empty_write,
    detect_repeated_tool_loop,
    detect_remote_local_confusion,
    detect_remote_verification_pending,
    detect_tool_plan_hard_route,
    detect_verifier_failure_from_result,
    detect_tool_output_misread,
    detect_wrong_path,
    detect_write_session_stall,
    record_bad_tool_arg_failure,
)
from .judge import maybe_run_llm_judge
from .reflexion_bridge import record_fama_failure_event
from .router import route_signal
from .signals import FamaSignal, current_step, get_fama_state, push_fama_signal
from .state import activate_mitigations, active_mitigations, clear_mitigations, expire_mitigations
from ..recovery_metrics import increment_metric, increment_metric_bucket

logger = logging.getLogger("smallctl.fama")


async def observe_tool_result(
    service: Any,
    *,
    tool_name: str,
    result: Any,
    arguments: dict[str, Any] | None = None,
    operation_id: str | None = None,
) -> None:
    harness = getattr(service, "harness", None)
    state = getattr(harness, "state", None)
    config = getattr(harness, "config", None)
    if state is None or not fama_enabled(config):
        return
    try:
        if latest_verifier_passed(state, result=result) and _verifier_pass_matches_active_done_gate(
            state,
            result=result,
        ):
            cleared = clear_mitigations(
                state,
                {"done_gate", "acceptance_checklist_capsule"},
                reason="verifier_passed",
            )
            for mitigation in cleared:
                _runlog(
                    harness,
                    "fama_mitigation_expired",
                    "FAMA mitigation cleared",
                    mitigation=mitigation.name,
                    reason="verifier_passed",
                    step=current_step(state),
                )

        # Fix 5 (RCA 8b79ca76): successful shell/ssh re-execution can also
        # clear done_gate when it verifies a previously failing target.
        if not latest_verifier_passed(state, result=result):
            if _successful_execution_clears_done_gate(
                state,
                result=result,
                tool_name=tool_name,
                arguments=arguments,
            ):
                cleared = clear_mitigations(
                    state,
                    {"done_gate", "acceptance_checklist_capsule"},
                    reason="re_verification_passed",
                )
                for mitigation in cleared:
                    _runlog(
                        harness,
                        "fama_mitigation_expired",
                        "FAMA mitigation cleared",
                        mitigation=mitigation.name,
                        reason="re_verification_passed",
                        step=current_step(state),
                    )

        early_stop = detect_early_stop_from_result(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if early_stop is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=early_stop, dedupe=False)

        verifier_failure = None
        if str(tool_name or "").strip() != "task_complete":
            verifier_failure = detect_verifier_failure_from_result(
                state,
                tool_name=tool_name,
                result=result,
                operation_id=operation_id,
            )
        if verifier_failure is not None:
            await _handle_observed_signal(
                harness,
                state=state,
                config=config,
                signal=verifier_failure,
                dedupe=True,
            )
            # Circuit-breaker: SSH auth impossibility should not trap the agent
            if _is_ssh_auth_impossibility(result):
                cleared = clear_mitigations(
                    state,
                    {"done_gate", "acceptance_checklist_capsule"},
                    reason="ssh_auth_impossible",
                )
                for mitigation in cleared:
                    _runlog(
                        harness,
                        "fama_mitigation_expired",
                        "FAMA mitigation cleared",
                        mitigation=mitigation.name,
                        reason="ssh_auth_impossible",
                        step=current_step(state),
                    )
                # Reset remote intent so local tools become available
                state.task_mode = "local_execute"
                state.active_intent = "general_task"
                _runlog(
                    harness,
                    "fama_ssh_auth_circuit_breaker",
                    "SSH auth failure triggered circuit breaker; released done_gate and reset to local_execute",
                    step=current_step(state),
                )

        record_bad_tool_arg_failure(state, tool_name=tool_name, result=result)
        bad_args = detect_bad_tool_args(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if bad_args is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=bad_args, dedupe=True)

        output_misread = detect_tool_output_misread(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if output_misread is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=output_misread, dedupe=True)

        remote_verification = detect_remote_verification_pending(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if remote_verification is not None:
            await _handle_observed_signal(
                harness,
                state=state,
                config=config,
                signal=remote_verification,
                dedupe=True,
            )

        empty_write = detect_empty_write(
            state,
            tool_name=tool_name,
            result=result,
            arguments=arguments,
            operation_id=operation_id,
        )
        if empty_write is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=empty_write, dedupe=True)

        remote_confusion = detect_remote_local_confusion(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if remote_confusion is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=remote_confusion, dedupe=True)

        wrong_path = detect_wrong_path(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if wrong_path is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=wrong_path, dedupe=True)
    except Exception as exc:
        logger.warning("FAMA observe failed: %s", exc)
        _runlog(
            harness,
            "fama_observe_error",
            "FAMA observation failed",
            exception_type=type(exc).__name__,
            source="observe_tool_result",
        )


def expire_for_turn(harness: Any, *, mode: str) -> None:
    state = getattr(harness, "state", None)
    config = getattr(harness, "config", None)
    if state is None or not fama_enabled(config):
        return
    step = current_step(state)
    expired = expire_mitigations(state, step=step)
    for mitigation in expired:
        _runlog(
            harness,
            "fama_mitigation_expired",
            "FAMA mitigation expired",
            mitigation=mitigation.name,
            reason="ttl",
            step=step,
            mode=mode,
        )
    for mitigation in active_mitigations(state):
        _runlog(
            harness,
            "fama_mitigation_ttl",
            "FAMA mitigation active",
            mitigation=mitigation.name,
            activated_step=mitigation.activated_step,
            expires_after_step=mitigation.expires_after_step,
            remaining_steps=max(0, mitigation.expires_after_step - step),
            reason=mitigation.reason,
            mode=mode,
        )
    threshold = int(getattr(config, "loop_guard_stagnation_threshold", 3) or 3)
    for signal in (
        detect_repeated_tool_loop(state, threshold=threshold),
        detect_write_session_stall(state, threshold=threshold),
        detect_backend_stream_halt(state, threshold=2),
        detect_context_drift(state),
    ):
        if signal is not None:
            _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    detect_tool_plan_hard_route(state)


def _handle_signal(
    harness: Any,
    *,
    state: Any,
    config: Any,
    signal: FamaSignal,
    dedupe: bool,
) -> bool:
    signature = _signal_signature(signal)
    if dedupe and _signature_seen(state, signature):
        _runlog(
            harness,
            "fama_signal_suppressed",
            "FAMA signal suppressed by deduplication",
            signature=signature,
            kind=signal.kind.value,
            severity=signal.severity,
            source=signal.source,
            step=signal.step,
            tool_name=signal.tool_name,
            evidence=signal.evidence,
        )
        return False
    push_fama_signal(state, signal, window=signal_window(config))
    _mark_signature_seen(state, signature)
    increment_metric_bucket(state, "fama_signals_by_kind", signal.kind.value)
    if signal.failure_class == "repeated_action" or signal.kind.value == "looping":
        increment_metric(state, "repeated_action_count")
    try:
        record_fama_failure_event(harness, state=state, signal=signal)
    except Exception as exc:
        logger.warning("FAMA recovery bridge failed: %s", exc)
    _runlog(
        harness,
        "fama_signal_detected",
        "FAMA signal detected",
        kind=signal.kind.value,
        severity=signal.severity,
        source=signal.source,
        step=signal.step,
        tool_name=signal.tool_name,
        failure_class=signal.failure_class,
    )
    if signal.severity >= 2:
        _append_context_invalidation(state, signal)
    if signal.kind.value == "early_stop" and not done_gate_on_failure(config):
        return True
    mitigations = route_signal(signal, state=state, config=config)
    activated = activate_mitigations(
        state,
        mitigations,
        max_active=max_active_mitigations(config),
    )
    _runlog(
        harness,
        "fama_signal_to_mitigation",
        "FAMA signal routed to mitigations",
        signal_kind=signal.kind.value,
        signal_source=signal.source,
        signal_evidence=signal.evidence,
        signal_step=signal.step,
        activated_mitigations=[mitigation.name for mitigation in activated],
        active_mitigations=[mitigation.name for mitigation in active_mitigations(state)],
    )
    for mitigation in activated:
        _runlog(
            harness,
            "fama_mitigation_activated",
            "FAMA mitigation activated",
            mitigation=mitigation.name,
            reason=mitigation.reason,
            expires_after_step=mitigation.expires_after_step,
        )
    return True


async def _handle_observed_signal(
    harness: Any,
    *,
    state: Any,
    config: Any,
    signal: FamaSignal,
    dedupe: bool,
) -> None:
    if not _handle_signal(harness, state=state, config=config, signal=signal, dedupe=dedupe):
        return
    judge_signal = await maybe_run_llm_judge(harness, state=state, config=config, base_signal=signal)
    if judge_signal is not None:
        _handle_signal(harness, state=state, config=config, signal=judge_signal, dedupe=True)


def _signal_signature(signal: FamaSignal) -> str:
    tool_name = str(signal.tool_name or "")
    evidence = str(signal.evidence or "")
    if signal.kind.value == "looping":
        repeated = ""
        if "repeated_tool=" in evidence:
            repeated = evidence.split("repeated_tool=", 1)[1].split(";", 1)[0].strip()
        return f"{signal.kind.value}:{signal.source}:{tool_name or repeated}"
    if signal.kind.value == "write_session_stall":
        session_id = ""
        if "session_id=" in evidence:
            session_id = evidence.split("session_id=", 1)[1].split(";", 1)[0].strip()
        return f"{signal.kind.value}:{session_id}"
    if signal.kind.value == "remote_local_confusion":
        reason = evidence.split(";", 1)[0].strip()
        return f"{signal.kind.value}:{tool_name}:{reason}"
    if signal.kind.value in {"bad_tool_args", "tool_output_misread", "backend_stream_halt", "context_drift"}:
        reason = evidence.split(";", 1)[0].strip()
        return f"{signal.kind.value}:{signal.source}:{tool_name}:{reason}"
    return f"{signal.kind.value}:{signal.source}:{tool_name}:{evidence}"


def _signature_seen(state: Any, signature: str) -> bool:
    seen = get_fama_state(state).get("seen_signatures")
    return isinstance(seen, list) and signature in seen


def _mark_signature_seen(state: Any, signature: str) -> None:
    payload = get_fama_state(state)
    seen = [str(item) for item in payload.get("seen_signatures", []) if str(item).strip()]
    if signature not in seen:
        seen.append(signature)
    payload["seen_signatures"] = seen[-24:]


def _append_context_invalidation(state: Any, signal: FamaSignal) -> None:
    paths = _signal_paths(signal)
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    history = scratchpad.get("_context_invalidations")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "reason": "fama_failure_detected",
            "paths": paths,
            "fama_signal": signal.kind.value,
            "step": signal.step,
            "source": signal.source,
        }
    )
    scratchpad["_context_invalidations"] = history[-40:]


def _signal_paths(signal: FamaSignal) -> list[str]:
    paths: list[str] = []
    evidence = str(signal.evidence or "")
    for marker in ("path=", "target="):
        if marker not in evidence:
            continue
        value = evidence.split(marker, 1)[1].split(";", 1)[0].strip()
        if value and value not in paths:
            paths.append(value)
    return paths


def _verifier_pass_matches_active_done_gate(state: Any, *, result: Any | None) -> bool:
    fingerprints = active_done_gate_fingerprints(state)
    if not fingerprints:
        return True
    verifier_fingerprint = passing_verifier_fingerprint(state, result=result)
    if not verifier_fingerprint:
        return True
    return verifier_fingerprint in fingerprints


def _successful_execution_clears_done_gate(
    state: Any,
    *,
    result: Any,
    tool_name: str,
    arguments: dict[str, Any] | None,
) -> bool:
    """Return True when a successful shell/ssh execution re-verifies a previously failing target."""
    if str(tool_name or "").strip() not in {"shell_exec", "ssh_exec"}:
        return False
    if not bool(getattr(result, "success", False)):
        return False
    fingerprints = active_done_gate_fingerprints(state)
    if not fingerprints:
        return False
    command = str((arguments or {}).get("command", "")).strip()
    if not command:
        return False
    from .fingerprints import normalize_verifier_target

    cmd_fp = normalize_verifier_target(command)
    if cmd_fp in fingerprints:
        return True
    # Path-based matching: if the successful command references the same
    # target path as any active done_gate fingerprint, consider it a
    # successful re-verification.
    for fp in fingerprints:
        if not fp:
            continue
        for token in fp.split():
            if "/" in token and token in cmd_fp:
                return True
            if any(
                token.endswith(ext)
                for ext in (
                    ".py",
                    ".js",
                    ".ts",
                    ".sh",
                    ".go",
                    ".rs",
                    ".java",
                )
            ):
                if token in cmd_fp:
                    return True
    return False


def _is_ssh_auth_impossibility(result: Any) -> bool:
    """Return True when an ssh_exec result failed with an authentication error."""
    if bool(getattr(result, "success", True)):
        return False
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    tool_name = str(metadata.get("tool_name") or "").strip()
    if tool_name != "ssh_exec":
        return False
    error = str(getattr(result, "error", "") or "").lower()
    stderr = ""
    output = metadata.get("output")
    if isinstance(output, dict):
        stderr = str(output.get("stderr") or "").lower()
    else:
        stderr = str(metadata.get("stderr") or "").lower()
    combined = f"{error}\n{stderr}"
    return "permission denied" in combined and ("publickey" in combined or "password" in combined)


def _runlog(harness: Any, event: str, message: str, **data: Any) -> None:
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(event, message, **data)
