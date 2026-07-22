from __future__ import annotations

import re
import logging
from typing import Any

from .config import default_ttl_steps
from .detector_classifiers import _READ_LOOP_TOOLS
from .signals import ActiveMitigation, FamaFailureKind, FamaSignal, current_step
from ..logging_utils import log_kv, synthetic_trace_id

_LOGGER = logging.getLogger("smallctl.fama.router")


_AUTH_FAILURE_EVIDENCE_RE = re.compile(
    r"(?:permission denied|authentication failed|publickey|password required|"
    r"connection refused|host key verification|no route to host|"
    r"could not resolve hostname|name or service not known)",
    re.IGNORECASE,
)


MITIGATION_RULES: dict[FamaFailureKind, list[str]] = {
    FamaFailureKind.EARLY_STOP: ["done_gate", "acceptance_checklist_capsule"],
    FamaFailureKind.LOOPING: ["micro_plan_capsule", "evidence_reuse_capsule", "tool_exposure_narrowing", "evidence_gathering_needed"],
    FamaFailureKind.INTERACTIVE_SESSION_STALL: ["interactive_installer_stall_capsule", "tool_exposure_narrowing", "evidence_reuse_capsule"],
    FamaFailureKind.REMOTE_LOCAL_CONFUSION: ["remote_scope_capsule", "remote_tool_exposure_guard"],
    FamaFailureKind.REMOTE_VERIFICATION_PENDING: ["remote_scope_capsule", "remote_verification_pending_capsule"],
    FamaFailureKind.TOOL_OUTPUT_MISREAD: ["evidence_reuse_capsule", "acceptance_checklist_capsule"],
    FamaFailureKind.BAD_TOOL_ARGS: ["micro_plan_capsule"],
    FamaFailureKind.WRITE_SESSION_STALL: ["write_session_recovery_capsule", "outline_only_recovery"],
    FamaFailureKind.BACKEND_STREAM_HALT: ["micro_plan_capsule", "outline_only_recovery"],
    FamaFailureKind.CONTEXT_DRIFT: ["micro_plan_capsule", "evidence_gathering_needed", "evidence_gathering_needed_hard_route", "remote_scope_capsule"],
    FamaFailureKind.PREFLIGHT_CONTRADICTION: ["preflight_contradiction_capsule", "micro_plan_capsule"],
    FamaFailureKind.STALE_SUCCESS_CLAIM: ["acceptance_checklist_capsule", "evidence_reuse_capsule"],
    FamaFailureKind.OBJECTIVE_MISMATCH: ["acceptance_checklist_capsule", "evidence_reuse_capsule"],
    FamaFailureKind.REPEATED_REMOTE_INSTALLER_FAILURE: ["repeated_remote_installer_failure_capsule", "preflight_contradiction_capsule", "evidence_reuse_capsule", "micro_plan_capsule"],
    FamaFailureKind.UPSTREAM_INSTALL_SOURCE_INVALID: ["source_invalid_install_capsule", "evidence_reuse_capsule", "tool_exposure_narrowing", "micro_plan_capsule"],
    FamaFailureKind.PREEXISTING_STATE_AS_SUCCESS: ["preexisting_state_as_success_capsule", "acceptance_checklist_capsule", "evidence_reuse_capsule"],
    FamaFailureKind.SSH_HOST_KEY_VERIFICATION: ["ssh_host_key_recovery_capsule"],
    FamaFailureKind.WRONG_PATH: ["micro_plan_capsule", "evidence_reuse_capsule"],
}


_REMOTE_INSTALL_TOOLS = {
    "ssh_exec",
    "ssh_session_read",
    "ssh_session_send",
    "ssh_session_start",
    "http_get",
    "file_download",
}

_REMOTE_INSTALL_FAILURE_CLASSES = {
    "repeated_remote_installer_failure",
    "interactive_session_stall",
    "verifier_failed",
    "tool_execution_failed",
    "no_progress",
    "repeated_action",
    "wrong_path",
    "remote_verification_pending",
}

_REMOTE_INSTALL_MARKERS = (
    "remote installer",
    "installer",
    "ssh_exec",
    "ssh session",
    "apt",
    "curl",
    "wget",
    "download",
)


def _task_requires_file_mutation(state: Any) -> bool:
    """Mirror of progress_guard_support._current_task_requires_file_mutation.

    Kept local to avoid a circular import from graph.progress_guard_support.
    """
    if state is None:
        return False
    active_intent = str(getattr(state, "active_intent", "") or "").strip()
    if active_intent in {"requested_file_patch", "requested_write_file"}:
        return True
    texts: list[str] = []
    run_brief = getattr(state, "run_brief", None)
    texts.append(str(getattr(run_brief, "original_task", "") or ""))
    working_memory = getattr(state, "working_memory", None)
    texts.append(str(getattr(working_memory, "current_goal", "") or ""))
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        handoff = scratchpad.get("_last_task_handoff")
        if isinstance(handoff, dict):
            texts.append(str(handoff.get("effective_task") or ""))
            texts.append(str(handoff.get("current_goal") or ""))
    task_text = " ".join(texts).lower()
    mutation_verb = any(verb in task_text for verb in ("patch", "fix", "repair", "update", "modify"))
    file_target = any(
        marker in task_text
        for marker in ("file", "file_patch", ".html", ".py", ".js", ".ts", "/var/www", "do not do a direct overwrite")
    )
    return mutation_verb and file_target


def _is_remote_install_failure(signal: FamaSignal, state: Any) -> bool:
    if str(getattr(state, "task_mode", "") or "").strip().lower() != "remote_execute":
        return False
    tool_name = str(signal.tool_name or "").strip()
    evidence_lower = str(signal.evidence or "").lower()
    has_install_marker = any(marker in evidence_lower for marker in _REMOTE_INSTALL_MARKERS)
    # Remote tools (e.g. ssh_exec) are used for many non-install tasks. Only
    # treat the failure as an install failure when the evidence actually
    # mentions install-related markers (apt, curl, wget, download, installer,
    # or the tool name itself as recorded by loop detectors).
    if tool_name in _REMOTE_INSTALL_TOOLS:
        return has_install_marker
    failure_class = str(signal.failure_class or "").strip()
    if failure_class in _REMOTE_INSTALL_FAILURE_CLASSES:
        return has_install_marker
    if tool_name and has_install_marker:
        return True
    return False


def route_signal(signal: FamaSignal, *, state: Any, config: Any) -> list[ActiveMitigation]:
    names = list(MITIGATION_RULES.get(signal.kind, []))
    # SSH auth / environment failures: replace done_gate with a
    # fallback suggestion so the model tries a different approach
    # instead of retrying the same failing SSH command.
    if signal.tool_name == "ssh_exec" and signal.failure_class == "verifier_failed":
        if _AUTH_FAILURE_EVIDENCE_RE.search(signal.evidence):
            names = [n for n in names if n not in {"done_gate", "acceptance_checklist_capsule"}]
            if "remote_auth_failure_capsule" not in names:
                names.append("remote_auth_failure_capsule")
            if "micro_plan_capsule" not in names:
                names.append("micro_plan_capsule")
    if signal.failure_class == "zero_tests_discovered" and "zero_test_recovery_capsule" not in names:
        names.append("zero_test_recovery_capsule")
    if signal.failure_class == "patch_target_not_found" and "patch_target_not_found_capsule" not in names:
        names.append("patch_target_not_found_capsule")
    # Timeout / infinite-loop signatures should not trap the agent behind done_gate.
    # A hanging script needs a rewrite, not an acceptance checklist.
    if signal.failure_class in {"verifier_timeout", "infinite_loop_suspected"}:
        names = [n for n in names if n not in {"done_gate", "acceptance_checklist_capsule"}]
        if "rewrite_suggestion_capsule" not in names:
            names.append("rewrite_suggestion_capsule")
    if not names:
        return []
    # Mutation-required read-loop breaker: when the model is looping on read
    # tools for a task that requires a file mutation, stop telling it to gather
    # more evidence and force a concrete patch/action instead.
    if (
        signal.kind == FamaFailureKind.LOOPING
        and signal.source == "loop_guard"
        and str(signal.tool_name or "").strip() in _READ_LOOP_TOOLS
        and _task_requires_file_mutation(state)
    ):
        state_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
        if state_phase in {"execute", "author", "repair"}:
            names = [n for n in names if n != "evidence_gathering_needed"]
            if "mutation_loop_breaker" not in names:
                names.append("mutation_loop_breaker")
    # P1.2: when source is loop_guard, ensure tool_exposure_narrowing is present
    # to break the repeat cycle directly
    if signal.source == "loop_guard" and "tool_exposure_narrowing" not in names:
        names = list(names) + ["tool_exposure_narrowing"]
    # Remote-install failures should always surface an install-specific capsule
    # (repeated_remote_installer_failure_capsule or interactive_installer_stall_capsule)
    # so the model does not cycle on generic advice.
    if _is_remote_install_failure(signal, state):
        install_capsules = {"repeated_remote_installer_failure_capsule", "interactive_installer_stall_capsule"}
        if not (install_capsules & set(names)):
            names = ["repeated_remote_installer_failure_capsule"] + list(names)
        # Suppress generic completion-gate advice for repeated install failures;
        # the installer needs environment repair, not a verifier checklist.
        names = [n for n in names if n not in {"done_gate", "acceptance_checklist_capsule"}]
        if "micro_plan_capsule" not in names:
            names.append("micro_plan_capsule")
    # Severity-3 escalation for repeated early_stop: pivot from blocking
    # completion to admitting defeat, so the model does not cycle
    # indefinitely on an unfixable external blocker.
    if signal.kind == FamaFailureKind.EARLY_STOP and signal.severity >= 3:
        names = [n for n in names if n != "done_gate"]
        if "dead_end_pivot_capsule" not in names:
            names.append("dead_end_pivot_capsule")
        if "micro_plan_capsule" not in names:
            names.append("micro_plan_capsule")
    step = current_step(state)
    ttl = default_ttl_steps(config)
    pending_interrupt = getattr(state, "pending_interrupt", None)
    if isinstance(pending_interrupt, dict) and pending_interrupt:
        # If a human interrupt is pending, keep acceptance/done_gate mitigations
        # alive longer so they don't expire while waiting for user input.
        ttl += max(3, ttl)
    expires_after_step = step + ttl
    source_signal = f"{signal.kind.value}:{signal.step}:{signal.tool_name or ''}"
    reason = signal.evidence
    if signal.kind == FamaFailureKind.SSH_HOST_KEY_VERIFICATION and signal.next_safe_action:
        reason = signal.next_safe_action
    mitigations = [
        ActiveMitigation(
            name=name,
            reason=reason,
            source_signal=source_signal,
            activated_step=step,
            expires_after_step=expires_after_step,
        )
        for name in names
    ]
    if _LOGGER.isEnabledFor(logging.DEBUG):
        trace_id = synthetic_trace_id(state, suffix="fama")
        log_kv(
            _LOGGER,
            logging.DEBUG,
            "fama_signal_routed",
            trace_id=trace_id,
            signal=signal.kind.value,
            detector=signal.source,
            mitigations=[m.name for m in mitigations],
            affected_tools=sorted({str(signal.tool_name or "")}),
            duration_steps=ttl,
        )
    return mitigations
