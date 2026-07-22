from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


FAMA_SCRATCHPAD_KEY = "_fama"
FAMA_VERSION = 1
DEFAULT_SIGNAL_WINDOW = 8


class FamaFailureKind(str, Enum):
    EARLY_STOP = "early_stop"
    LOOPING = "looping"
    INTERACTIVE_SESSION_STALL = "interactive_session_stall"
    REMOTE_LOCAL_CONFUSION = "remote_local_confusion"
    REMOTE_VERIFICATION_PENDING = "remote_verification_pending"
    TOOL_OUTPUT_MISREAD = "tool_output_misread"
    BAD_TOOL_ARGS = "bad_tool_args"
    WRITE_SESSION_STALL = "write_session_stall"
    BACKEND_STREAM_HALT = "backend_stream_halt"
    CONTEXT_DRIFT = "context_drift"
    PREFLIGHT_CONTRADICTION = "preflight_contradiction"
    STALE_SUCCESS_CLAIM = "stale_success_claim"
    OBJECTIVE_MISMATCH = "objective_mismatch"
    REPEATED_REMOTE_INSTALLER_FAILURE = "repeated_remote_installer_failure"
    UPSTREAM_INSTALL_SOURCE_INVALID = "upstream_install_source_invalid"
    PREEXISTING_STATE_AS_SUCCESS = "preexisting_state_as_success"
    SSH_HOST_KEY_VERIFICATION = "ssh_host_key_verification"
    WRONG_PATH = "wrong_path"


FAILURE_CLASSES = {
    "tool_schema_invalid": "Model emitted malformed JSON, wrong tool name, or invalid args.",
    "tool_execution_failed": "Tool ran but returned success=false or exception.",
    "patch_target_not_found": "A file patch target_text did not match the current file content.",
    "completion_blocked": "Model attempted to finish while completion gates were still blocked.",
    "wrong_path": "Path does not exist, wrong cwd, absolute path issue, remote/local mismatch, or stale file path.",
    "remote_verification_pending": "Remote mutation appears complete but still needs remote read-back verification.",
    "empty_write": "file_write/file_append/SSH write produced empty or near-empty content unexpectedly.",
    "write_session_stall": "Chunked write session repeated or failed to progress.",
    "repeated_action": "Same or near-same tool call repeated without new evidence.",
    "verifier_failed": "Verifier/test/check says acceptance criteria are not met.",
    "test_failed": "Unit/integration command failed.",
    "zero_tests_discovered": "A test command ran successfully but discovered zero tests.",
    "context_missing": "Model needs evidence it has not actually retrieved.",
    "hallucinated_assumption": "Model asserted something unsupported by tool output.",
    "human_resteer": "User corrected direction or clarified after model drift.",
    "same_scope_iteration": "User continued or refined the same task without implying model drift.",
    "backend_stream_failure": "Model/backend stream failed, truncated, or wedged.",
    "no_progress": "Progress guard sees no new files, evidence, or subtask movement.",
    "preflight_contradiction": "Preflight guard contradicted verified evidence (e.g. wrote file then claimed NOT FOUND).",
    "stale_success_claim": "Model claimed success after task_complete was blocked or before verification passed.",
    "objective_mismatch": "Verifier or success check does not match the user objective.",
    "repeated_remote_installer_failure": "Remote installer preflight or execution failed repeatedly.",
    "upstream_install_source_invalid": "Installer fetches or package hosts are invalid upstream while general connectivity still works.",
    "preexisting_state_as_success": "Model treated pre-existing state as successful task completion.",
    "ssh_host_key_verification": "SSH host key verification failed repeatedly.",
}


DEFAULT_FAILURE_CLASS_BY_KIND: dict[FamaFailureKind, str] = {
    FamaFailureKind.EARLY_STOP: "completion_blocked",
    FamaFailureKind.LOOPING: "repeated_action",
    FamaFailureKind.INTERACTIVE_SESSION_STALL: "interactive_session_stall",
    FamaFailureKind.REMOTE_LOCAL_CONFUSION: "wrong_path",
    FamaFailureKind.REMOTE_VERIFICATION_PENDING: "remote_verification_pending",
    FamaFailureKind.TOOL_OUTPUT_MISREAD: "hallucinated_assumption",
    FamaFailureKind.BAD_TOOL_ARGS: "tool_schema_invalid",
    FamaFailureKind.WRITE_SESSION_STALL: "write_session_stall",
    FamaFailureKind.BACKEND_STREAM_HALT: "backend_stream_failure",
    FamaFailureKind.CONTEXT_DRIFT: "context_missing",
    FamaFailureKind.PREFLIGHT_CONTRADICTION: "preflight_contradiction",
    FamaFailureKind.STALE_SUCCESS_CLAIM: "stale_success_claim",
    FamaFailureKind.OBJECTIVE_MISMATCH: "objective_mismatch",
    FamaFailureKind.REPEATED_REMOTE_INSTALLER_FAILURE: "repeated_remote_installer_failure",
    FamaFailureKind.UPSTREAM_INSTALL_SOURCE_INVALID: "upstream_install_source_invalid",
    FamaFailureKind.PREEXISTING_STATE_AS_SUCCESS: "preexisting_state_as_success",
    FamaFailureKind.SSH_HOST_KEY_VERIFICATION: "ssh_host_key_verification",
    FamaFailureKind.WRONG_PATH: "wrong_path",
}


@dataclass(slots=True)
class FamaSignal:
    kind: FamaFailureKind
    severity: int
    source: str
    evidence: str
    step: int
    tool_name: str | None = None
    operation_id: str | None = None
    suggested_mitigations: list[str] = field(default_factory=list)
    failure_class: str | None = None
    next_safe_action: str | None = None


@dataclass(slots=True)
class ActiveMitigation:
    name: str
    reason: str
    source_signal: str
    activated_step: int
    expires_after_step: int
    priority: int = 50


def current_step(state: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(getattr(state, "step_count", default)))
    except (TypeError, ValueError):
        return default


def get_fama_state(state: Any) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        try:
            state.scratchpad = scratchpad
        except Exception:
            return _empty_fama_state()

    payload = scratchpad.get(FAMA_SCRATCHPAD_KEY)
    if not isinstance(payload, dict):
        payload = _empty_fama_state()
        scratchpad[FAMA_SCRATCHPAD_KEY] = payload
    payload.setdefault("version", FAMA_VERSION)
    if not isinstance(payload.get("signals"), list):
        payload["signals"] = []
    if not isinstance(payload.get("active_mitigations"), list):
        payload["active_mitigations"] = []
    if not isinstance(payload.get("seen_signatures"), list):
        payload["seen_signatures"] = []
    if not isinstance(payload.get("last_observed_step"), int):
        payload["last_observed_step"] = current_step(state)
    return payload


def push_fama_signal(state: Any, signal: FamaSignal, *, window: int = DEFAULT_SIGNAL_WINDOW) -> None:
    payload = get_fama_state(state)
    signals = [item for item in payload.get("signals", []) if isinstance(item, dict)]
    signals.append(signal_to_dict(signal))
    limit = max(1, int(window or DEFAULT_SIGNAL_WINDOW))
    payload["signals"] = signals[-limit:]
    payload["last_observed_step"] = signal.step


def signal_to_dict(signal: FamaSignal) -> dict[str, Any]:
    return {
        "kind": signal.kind.value,
        "severity": max(1, min(3, int(signal.severity))),
        "source": str(signal.source),
        "evidence": str(signal.evidence),
        "step": int(signal.step),
        "tool_name": signal.tool_name,
        "operation_id": signal.operation_id,
        "suggested_mitigations": list(signal.suggested_mitigations),
        "failure_class": signal.failure_class
        or DEFAULT_FAILURE_CLASS_BY_KIND.get(signal.kind),
        "next_safe_action": signal.next_safe_action,
    }


def signal_from_dict(payload: dict[str, Any]) -> FamaSignal | None:
    try:
        kind = FamaFailureKind(str(payload.get("kind") or ""))
    except ValueError:
        return None
    suggested = payload.get("suggested_mitigations")
    return FamaSignal(
        kind=kind,
        severity=max(1, min(3, _to_int(payload.get("severity"), 1))),
        source=str(payload.get("source") or ""),
        evidence=str(payload.get("evidence") or ""),
        step=_to_int(payload.get("step"), 0),
        tool_name=_optional_str(payload.get("tool_name")),
        operation_id=_optional_str(payload.get("operation_id")),
        suggested_mitigations=[str(item) for item in suggested] if isinstance(suggested, list) else [],
        failure_class=_optional_str(payload.get("failure_class")),
        next_safe_action=_optional_str(payload.get("next_safe_action")),
    )


def mitigation_to_dict(mitigation: ActiveMitigation) -> dict[str, Any]:
    return {
        "name": str(mitigation.name),
        "reason": str(mitigation.reason),
        "source_signal": str(mitigation.source_signal),
        "activated_step": int(mitigation.activated_step),
        "expires_after_step": int(mitigation.expires_after_step),
        "priority": int(mitigation.priority),
    }


def mitigation_from_dict(payload: dict[str, Any]) -> ActiveMitigation | None:
    name = str(payload.get("name") or "").strip()
    if not name:
        return None
    return ActiveMitigation(
        name=name,
        reason=str(payload.get("reason") or ""),
        source_signal=str(payload.get("source_signal") or ""),
        activated_step=_to_int(payload.get("activated_step"), 0),
        expires_after_step=_to_int(payload.get("expires_after_step"), 0),
        priority=_to_int(payload.get("priority"), 50),
    )


def _empty_fama_state() -> dict[str, Any]:
    return {
        "version": FAMA_VERSION,
        "signals": [],
        "active_mitigations": [],
        "seen_signatures": [],
        "last_observed_step": 0,
    }


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
