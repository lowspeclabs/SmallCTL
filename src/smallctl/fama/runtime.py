from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any

from .config import (
    done_gate_on_failure,
    fama_enabled,
    max_active_mitigations,
    signal_window,
)
from .detectors import detect_early_stop_from_result, latest_verifier_passed
from .fingerprints import (
    active_done_gate_fingerprints,
    install_verifier_passes_objective,
    passing_verifier_fingerprint,
)
from .detectors import (
    detect_apt_deb822_preflight_blocked,
    detect_backend_stream_halt,
    detect_bad_tool_args,
    detect_context_drift,
    detect_debian_13_installer_readiness,
    detect_empty_write,
    detect_generic_stuck_loop,
    detect_interactive_installer_stall,
    detect_model_output_degenerate_loop,
    detect_objective_verifier_mismatch,
    detect_preflight_contradiction,
    detect_preexisting_state_as_success,
    detect_patch_target_not_found,
    detect_stale_success_claim,
    detect_tool_output_misread,
    detect_repeated_remote_installer_failure,
    detect_repeated_tool_loop,
    detect_remote_local_confusion,
    detect_remote_verification_pending,
    detect_ssh_host_key_verification_failure,
    detect_ssh_host_key_verification_failure_from_result,
    detect_tool_plan_hard_route,
    detect_upstream_install_source_invalid,
    detect_verifier_failure_from_result,
    detect_wrong_path,
    detect_write_session_stall,
    record_bad_tool_arg_failure,
)
from .judge import maybe_run_llm_judge
from .reflexion_bridge import record_fama_failure_event
from .router import route_signal
from .signals import FamaFailureKind, FamaSignal, current_step, get_fama_state, push_fama_signal
from .state import activate_mitigations, active_mitigations, clear_mitigations, expire_mitigations
from ..recovery_metrics import increment_metric, increment_metric_bucket
from ..logging_utils import runlog as _runlog

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

        if latest_verifier_passed(state, result=result) and install_verifier_passes_objective(
            state, result=result
        ):
            cleared = clear_mitigations(
                state,
                {"done_gate", "acceptance_checklist_capsule"},
                reason="install_verifier_passed",
            )
            for mitigation in cleared:
                _runlog(
                    harness,
                    "fama_mitigation_expired",
                    "FAMA mitigation cleared",
                    mitigation=mitigation.name,
                    reason="install_verifier_passed",
                    step=current_step(state),
                )

        # A re-verification may add harmless shell syntax (for example,
        # ``&& echo VERIFIED_OK``), causing exact verifier fingerprints to
        # differ even though it checks the same target.
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
            # Recovery outcomes are intentionally typed. A failed remote login
            # is not evidence that the task became local.
            if _is_local_route_contradiction(result, tool_name=tool_name):
                previous_mode = str(getattr(state, "task_mode", "") or "")
                state.task_mode = "local_execute"
                state.active_intent = "requested_shell_exec"
                _record_pinned_recovery(
                    state,
                    kind="route_contradiction",
                    blocker="A local workspace command was submitted through ssh_exec.",
                    next_allowed_action="Retry once with shell_exec.",
                    required_tool="shell_exec",
                    result=result,
                )
                _runlog(harness, "fama_ssh_transport_circuit_breaker", "Local route contradiction requires shell_exec retry", previous_task_mode=previous_mode, next_task_mode="local_execute", failure_kind="route_contradiction", required_tool="shell_exec", retry_eligible=True, next_required_action="retry_with_shell_exec")
            elif _is_ssh_transport_impossibility(result, tool_name=tool_name):
                cleared = clear_mitigations(
                    state,
                    {"done_gate", "acceptance_checklist_capsule"},
                    reason="ssh_transport_impossible",
                )
                for mitigation in cleared:
                    _runlog(
                        harness,
                        "fama_mitigation_expired",
                        "FAMA mitigation cleared",
                        mitigation=mitigation.name,
                        reason="ssh_transport_impossible",
                        step=current_step(state),
                    )
                previous_mode = str(getattr(state, "task_mode", "") or "")
                state.task_mode = "remote_execute"
                state.active_intent = "requested_ssh_exec"
                metadata = getattr(result, "metadata", None)
                metadata = metadata if isinstance(metadata, dict) else {}
                _record_pinned_recovery(
                    state,
                    kind="ssh_auth_blocker",
                    blocker="SSH authentication or transport failed before remote execution.",
                    next_allowed_action="Provide corrected non-secret SSH credentials or resolve remote connectivity.",
                    required_tool="ask_human",
                    result=result,
                )
                _runlog(
                    harness,
                    "fama_ssh_transport_circuit_breaker",
                    "SSH transport failure requires remote credential or connectivity remediation",
                    step=current_step(state),
                    previous_task_mode=previous_mode,
                    next_task_mode="remote_execute",
                    failure_kind=str(metadata.get("failure_kind") or "transport"),
                    attempted_command_scope="remote",
                    required_tool="ask_human",
                    retry_eligible=False,
                    next_required_action="request_remote_credential_or_connectivity_remediation",
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

        patch_target_not_found = detect_patch_target_not_found(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if patch_target_not_found is not None:
            await _handle_observed_signal(
                harness,
                state=state,
                config=config,
                signal=patch_target_not_found,
                dedupe=True,
            )

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

        if str(tool_name or "").strip() == "ssh_session_read":
            metadata = getattr(result, "metadata", None)
            metadata = metadata if isinstance(metadata, dict) else {}
            if bool(metadata.get("interactive_output_unchanged")):
                interactive_stall = detect_interactive_installer_stall(state, threshold=2)
                if interactive_stall is not None:
                    await _handle_observed_signal(
                        harness,
                        state=state,
                        config=config,
                        signal=interactive_stall,
                        dedupe=True,
                    )

        preflight_contradiction = detect_preflight_contradiction(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if preflight_contradiction is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=preflight_contradiction, dedupe=True)

        apt_deb822_block = detect_apt_deb822_preflight_blocked(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if apt_deb822_block is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=apt_deb822_block, dedupe=True)

        debian_readiness = detect_debian_13_installer_readiness(state, threshold=1)
        if debian_readiness is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=debian_readiness, dedupe=True)

        stale_success = detect_stale_success_claim(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if stale_success is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=stale_success, dedupe=True)

        host_key_failure_from_result = detect_ssh_host_key_verification_failure_from_result(
            state,
            tool_name=tool_name,
            result=result,
            arguments=arguments,
            operation_id=operation_id,
        )
        if host_key_failure_from_result is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=host_key_failure_from_result, dedupe=True)

        host_key_failure = detect_ssh_host_key_verification_failure(state, threshold=2)
        if host_key_failure is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=host_key_failure, dedupe=True)
    except Exception as exc:
        logger.warning("FAMA observe failed: %s", exc, exc_info=True)
        import traceback
        _runlog(
            harness,
            "fama_observe_error",
            "FAMA observation failed",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            traceback=traceback.format_exc(limit=8),
            source="observe_tool_result",
            tool_name=tool_name,
        )


async def observe_guard_trip(
    harness: Any,
    *,
    guard_error: str,
    tool_history_tail: list[str] | None = None,
    grouped_errors: list[dict[str, Any]] | None = None,
) -> None:
    """Route a guard trip diagnosis into FAMA so a mitigation capsule is injected.

    The signal is classified as LOOPING when the guard error mentions repeated
    loops, repeated tools, or stagnation; otherwise it is treated as CONTEXT_DRIFT
    (e.g. max_consecutive_errors where the model has lost the thread).
    """
    state = getattr(harness, "state", None)
    config = getattr(harness, "config", None)
    if state is None or not fama_enabled(config):
        return
    try:
        error_text = str(guard_error or "").lower()
        is_looping = any(
            marker in error_text
            for marker in (
                "loop detected",
                "repeated tool",
                "repeated tool call loop",
                "stagnation",
                "stuck",
            )
        )
        kind = FamaFailureKind.LOOPING if is_looping else FamaFailureKind.CONTEXT_DRIFT

        # Try to identify the most repeated failing tool from grouped errors.
        repeated_tool: str | None = None
        max_count = 0
        for group in grouped_errors or []:
            if not isinstance(group, dict):
                continue
            sig = str(group.get("signature") or "").strip()
            count = int(group.get("count", 0) or 0)
            if sig and count > max_count:
                max_count = count
                tool_candidate = sig.split(":", 1)[0].strip()
                if tool_candidate:
                    repeated_tool = tool_candidate

        # Fall back to the most recent failing tool in the history tail.
        if not repeated_tool and tool_history_tail:
            for entry in reversed(tool_history_tail):
                entry_text = str(entry or "").strip()
                if "|" in entry_text:
                    candidate = entry_text.split("|", 1)[0].strip()
                    if candidate and candidate != "Guard tripped":
                        repeated_tool = candidate
                        break

        evidence = str(guard_error or "guard tripped").strip()
        if repeated_tool:
            evidence = f"repeated_tool={repeated_tool}; {evidence}"

        if kind == FamaFailureKind.LOOPING:
            next_safe_action = (
                "Do not retry the same failing command or tool unchanged. "
                "Pick one concrete different action: gather fresh evidence, "
                "change an argument, or explain the blocker before continuing."
            )
        else:
            next_safe_action = (
                "Stop and re-read the current state. Choose one bounded read or "
                "diagnostic action to rebuild context before mutating or finishing."
            )

        signal = FamaSignal(
            kind=kind,
            severity=2,
            source="guard_trip",
            evidence=evidence,
            step=current_step(state),
            tool_name=repeated_tool,
            failure_class="repeated_action" if kind == FamaFailureKind.LOOPING else "context_missing",
            next_safe_action=next_safe_action,
        )
        await _handle_observed_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    except Exception as exc:
        logger.warning("FAMA guard-trip observation failed: %s", exc)


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
        detect_generic_stuck_loop(state, threshold=threshold),
        detect_interactive_installer_stall(state, threshold=2),
        detect_write_session_stall(state, threshold=threshold),
        detect_model_output_degenerate_loop(state),
        detect_backend_stream_halt(state, threshold=2),
        detect_context_drift(state),
        detect_repeated_remote_installer_failure(state, threshold=2),
        detect_upstream_install_source_invalid(state),
        detect_objective_verifier_mismatch(state),
        detect_preexisting_state_as_success(state),
    ):
        if signal is not None:
            _handle_signal(harness, state=state, config=config, signal=signal, dedupe=True)
    detect_tool_plan_hard_route(state)


def _model_claims_hallucinated_bug(state: Any) -> bool:
    """Detect if the model's recent output claims an error absent from the actual verifier output.

    Returns True when the most recent assistant text references a specific exception
    or error message that does not appear in the raw verifier output stored in
    `_last_failed_verifier`. This prevents the harness from silently deduplicating
    verifier failures while the model chases a hallucinated bug.
    """
    scratchpad = getattr(state, "scratchpad", None) or {}
    verifier = scratchpad.get("_last_failed_verifier") if isinstance(scratchpad, dict) else None
    if not isinstance(verifier, dict):
        return False

    raw_output = str(verifier.get("raw_output") or "").strip()
    if not raw_output:
        return False

    # Collect recent assistant text
    recent_text = ""
    for msg in getattr(state, "recent_messages", [])[-4:]:
        if getattr(msg, "role", "") == "assistant":
            content = getattr(msg, "content", "") or ""
            recent_text += " " + content

    if not recent_text.strip():
        return False

    # Extract specific error claims: ExceptionType: description
    _ERROR_CLAIM_RE = re.compile(
        r"\b(AttributeError|NameError|TypeError|ValueError|AssertionError|"
        r"IndexError|KeyError|ImportError|ModuleNotFoundError|RuntimeError|"
        r"OSError|IOError|ZeroDivisionError|FileNotFoundError|"
        r"NotImplementedError|OverflowError|RecursionError)\s*:\s*[^\n]+",
        re.IGNORECASE,
    )
    claimed_errors = set()
    for match in _ERROR_CLAIM_RE.finditer(recent_text):
        claimed_errors.add(match.group(0).strip())

    if not claimed_errors:
        return False

    raw_lower = raw_output.lower()
    for error in claimed_errors:
        if error.lower() not in raw_lower:
            # Model claimed an error that does not exist in the actual verifier output
            return True

    return False


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
        # Carve-out: loop_guard looping signals escalate after repeated suppression
        if signal.kind.value == "looping" and signal.source == "loop_guard":
            payload = get_fama_state(state)
            key = f"_fama_loop_guard_suppression_count:{signature}"
            try:
                count = int(payload.get(key, 0)) + 1
            except (TypeError, ValueError):
                count = 1
            payload[key] = count
            if count >= 3:
                _runlog(
                    harness,
                    "fama_signal_escalated",
                    "FAMA looping signal escalated after repeated suppression",
                    signature=signature,
                    kind=signal.kind.value,
                    severity=signal.severity,
                    source=signal.source,
                    step=signal.step,
                    tool_name=signal.tool_name,
                    evidence=signal.evidence,
                    suppression_count=count,
                )
                # fall through to process the signal
            else:
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
                    suppression_count=count,
                )
                return False
        else:
            # Carve-out: repeated verifier failures escalate so the harness
            # does not silently swallow legitimate recurring test/verifier errors.
            if signal.source == "verifier" and signal.kind.value == "early_stop":
                payload = get_fama_state(state)
                key = f"_fama_verifier_suppression_count:{signature}"
                try:
                    count = int(payload.get(key, 0)) + 1
                except (TypeError, ValueError):
                    count = 1
                payload[key] = count

                # Detect hallucinated bugs on first suppression and escalate immediately
                is_hallucination = _model_claims_hallucinated_bug(state)
                if is_hallucination or count >= 3:
                    _runlog(
                        harness,
                        "fama_signal_escalated",
                        "FAMA verifier signal escalated after hallucination or repeated suppression",
                        signature=signature,
                        kind=signal.kind.value,
                        severity=signal.severity,
                        source=signal.source,
                        step=signal.step,
                        tool_name=signal.tool_name,
                        evidence=signal.evidence,
                        suppression_count=count,
                        hallucination=is_hallucination,
                    )
                    # Escalate severity and force re-processing
                    signal = dataclasses.replace(
                        signal,
                        severity=3,
                        evidence=f"{signal.evidence} (hallucination={is_hallucination}, repeated={count}x)",
                    )
                    # Hard circuit-breaker: when the same (failure_class, tool_name)
                    # combination has escalated 3+ times, clear done_gate and
                    # force a strategic pivot so the model stops cycling.
                    payload_pivot = get_fama_state(state)
                    pivot_key = f"_fama_pivot_count:{signal.failure_class}:{signal.tool_name}"
                    pivot_count = int(payload_pivot.get(pivot_key, 0) or 0) + 1
                    payload_pivot[pivot_key] = pivot_count
                    if pivot_count >= 3:
                        cleared_pivot = clear_mitigations(
                            state,
                            {"done_gate", "acceptance_checklist_capsule"},
                            reason="dead_end_pivot",
                        )
                        for mit in cleared_pivot:
                            _runlog(
                                harness,
                                "fama_circuit_breaker",
                                "FAMA circuit breaker: forcing strategic pivot after repeated same-failure escalation",
                                failure_class=signal.failure_class,
                                tool_name=signal.tool_name,
                                pivot_count=pivot_count,
                                cleared=mit.name,
                            )
                    # fall through to process the signal
                else:
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
                        suppression_count=count,
                    )
                    return False
            else:
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
    # Clear suppression counts on successful pass-through
    if signal.kind.value == "looping" and signal.source == "loop_guard":
        payload = get_fama_state(state)
        key = f"_fama_loop_guard_suppression_count:{signature}"
        payload.pop(key, None)
    if signal.source == "verifier" and signal.kind.value == "early_stop":
        payload = get_fama_state(state)
        key = f"_fama_verifier_suppression_count:{signature}"
        payload.pop(key, None)
    push_fama_signal(state, signal, window=signal_window(config))
    _mark_signature_seen(state, signature)
    increment_metric_bucket(state, "fama_signals_by_kind", signal.kind.value)
    if signal.failure_class == "repeated_action" or signal.kind.value == "looping":
        increment_metric(state, "repeated_action_count")
    if signal.kind.value == "ssh_host_key_verification":
        host = _extract_ssh_host_key_failure_host(signal)
        if host:
            _runlog(
                harness,
                "ssh_host_key_recovery_required",
                f"SSH host key changed for {host}",
                host=host,
                suggested_command=f"ssh-keygen -R {host} -f ~/.ssh/known_hosts",
            )
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


def _is_ssh_transport_impossibility(result: Any, *, tool_name: str = "") -> bool:
    """Return True when ssh_exec failed before useful remote verification ran."""
    if bool(getattr(result, "success", True)):
        return False
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    resolved_tool_name = str(tool_name or metadata.get("tool_name") or "").strip()
    if resolved_tool_name != "ssh_exec":
        return False
    error = str(getattr(result, "error", "") or "").lower()
    stderr = ""
    exit_code = None
    output = metadata.get("output")
    if isinstance(output, dict):
        stderr = str(output.get("stderr") or "").lower()
        exit_code = output.get("exit_code")
    else:
        stderr = str(metadata.get("stderr") or "").lower()
        exit_code = metadata.get("exit_code")
    combined = f"{error}\n{stderr}"
    if metadata.get("failure_kind") == "remote_command":
        return False
    if metadata.get("ssh_transport_succeeded") is True:
        return False
    transport_markers = (
        "permission denied (publickey",
        "permission denied, please try again",
        "no route to host",
        "connection refused",
        "connection timed out",
        "network is unreachable",
        "could not resolve hostname",
        "name or service not known",
        "temporary failure in name resolution",
    )
    if any(marker in combined for marker in transport_markers):
        return True
    try:
        if int(exit_code) == 255 and not stderr:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _is_local_route_contradiction(result: Any, *, tool_name: str) -> bool:
    metadata = getattr(result, "metadata", None)
    return bool(
        tool_name == "ssh_exec"
        and isinstance(metadata, dict)
        and metadata.get("reason") == "local_command_requires_shell_exec"
    )


def _record_pinned_recovery(
    state: Any,
    *,
    kind: str,
    blocker: str,
    next_allowed_action: str,
    required_tool: str,
    result: Any,
) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    metadata = getattr(result, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    scratchpad["_pinned_recovery"] = {
        "recovery_id": f"{kind}:{current_step(state)}",
        "kind": kind,
        "current_blocker": blocker,
        "next_allowed_action": next_allowed_action,
        "required_tool": required_tool,
        "target": str(metadata.get("host") or ""),
        "command_fingerprint": str(metadata.get("command_fingerprint") or ""),
        "creation_step": current_step(state),
        "source_event": "fama_ssh_transport_circuit_breaker",
    }


def _is_ssh_auth_impossibility(result: Any) -> bool:
    """Compatibility wrapper for tests/imports using the old helper name."""
    return _is_ssh_transport_impossibility(result)


def _extract_ssh_host_key_failure_host(signal: FamaSignal) -> str:
    """Extract the target host from a host-key verification FAMA signal."""
    evidence = str(signal.evidence or "")
    marker = " for "
    if marker in evidence:
        return evidence.rsplit(marker, 1)[-1].split(";", 1)[0].strip()
    return ""
