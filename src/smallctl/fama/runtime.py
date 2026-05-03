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
from .detectors import (
    detect_backend_stream_halt,
    detect_bad_tool_args,
    detect_context_drift,
    detect_repeated_tool_loop,
    detect_remote_local_confusion,
    detect_tool_output_misread,
    detect_write_session_stall,
    record_bad_tool_arg_failure,
)
from .judge import maybe_run_llm_judge
from .router import route_signal
from .signals import FamaSignal, current_step, get_fama_state, push_fama_signal
from .state import activate_mitigations, active_mitigations, clear_mitigations, expire_mitigations

logger = logging.getLogger("smallctl.fama")


async def observe_tool_result(
    service: Any,
    *,
    tool_name: str,
    result: Any,
    arguments: dict[str, Any] | None = None,
    operation_id: str | None = None,
) -> None:
    del arguments
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

        early_stop = detect_early_stop_from_result(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if early_stop is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=early_stop, dedupe=False)

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

        remote_confusion = detect_remote_local_confusion(
            state,
            tool_name=tool_name,
            result=result,
            operation_id=operation_id,
        )
        if remote_confusion is not None:
            await _handle_observed_signal(harness, state=state, config=config, signal=remote_confusion, dedupe=True)
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
    expired = expire_mitigations(state, step=current_step(state))
    for mitigation in expired:
        _runlog(
            harness,
            "fama_mitigation_expired",
            "FAMA mitigation expired",
            mitigation=mitigation.name,
            reason="ttl",
            step=current_step(state),
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


def _handle_signal(
    harness: Any,
    *,
    state: Any,
    config: Any,
    signal: FamaSignal,
    dedupe: bool,
) -> bool:
    if dedupe and _signature_seen(state, _signal_signature(signal)):
        return False
    push_fama_signal(state, signal, window=signal_window(config))
    _mark_signature_seen(state, _signal_signature(signal))
    _runlog(
        harness,
        "fama_signal_detected",
        "FAMA signal detected",
        kind=signal.kind.value,
        severity=signal.severity,
        source=signal.source,
        step=signal.step,
        tool_name=signal.tool_name,
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
    fingerprints = _active_done_gate_fingerprints(state)
    if not fingerprints:
        return True
    verifier_fingerprint = _passing_verifier_fingerprint(state, result=result)
    if not verifier_fingerprint:
        return True
    return verifier_fingerprint in fingerprints


def _active_done_gate_fingerprints(state: Any) -> set[str]:
    fingerprints: set[str] = set()
    for mitigation in active_mitigations(state):
        if mitigation.name not in {"done_gate", "acceptance_checklist_capsule"}:
            continue
        fingerprint = _fingerprint_from_reason(mitigation.reason)
        if fingerprint:
            fingerprints.add(fingerprint)
    return fingerprints


def _fingerprint_from_reason(reason: str) -> str:
    text = str(reason or "").strip()
    marker = "verifier verdict "
    if marker not in text:
        return ""
    tail = text.split(marker, 1)[1]
    if ":" not in tail:
        return ""
    return _normalize_verifier_target(tail.split(":", 1)[1])


def _passing_verifier_fingerprint(state: Any, *, result: Any | None) -> str:
    metadata = getattr(result, "metadata", None) if result is not None else None
    metadata = metadata if isinstance(metadata, dict) else {}
    verifier = metadata.get("last_verifier_verdict")
    if not isinstance(verifier, dict) or not verifier:
        current_verifier = getattr(state, "current_verifier_verdict", None)
        verifier = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
    if isinstance(verifier, dict) and str(verifier.get("verdict") or "").strip().lower() == "pass":
        return _normalize_verifier_target(str(verifier.get("command") or verifier.get("target") or ""))
    if str(metadata.get("verifier_verdict") or "").strip().lower() == "pass":
        return _normalize_verifier_target(
            str(metadata.get("verifier_command") or metadata.get("verifier_target") or "")
        )
    return ""


def _normalize_verifier_target(value: str) -> str:
    text = " ".join(str(value or "").strip().split())
    return text.casefold()


def _runlog(harness: Any, event: str, message: str, **data: Any) -> None:
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(event, message, **data)
