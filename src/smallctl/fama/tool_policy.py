from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

from ..diagnostic_tasks import (
    diagnostic_completion_reports_failure,
    diagnostic_failure_completion_allowed,
)
from ..models.tool_result import ToolEnvelope
from .config import done_gate_on_failure, fama_enabled
from .fingerprints import active_done_gate_fingerprints, normalize_verifier_target
from .signals import get_fama_state
from .state import active_mitigation_names

# Read-only status/presence inquiry: the answer to the question may be that the
# resource is absent or failed, which is valid negative intelligence.
_READ_ONLY_STATUS_INQUIRY_RE = re.compile(
    r"\b(?:is|are)\s+\S+(?:\s+\S+){0,6}\s+(?:up(?:\s+and\s+running)?|running|installed|active|enabled|disabled)\b"
    r"|\b(?:status|state)\s+of\b"
    r"|\bcheck\s+(?:if|whether)\s+\S+(?:\s+\S+){0,6}\s+(?:is|are)\s+(?:up(?:\s+and\s+running)?|running|installed|active|enabled|disabled)\b",
    re.IGNORECASE,
)

# Action/repair intent markers that turn a status question into a task that
# must actually fix or change state before completing.
_ACTION_INTENT_RE = re.compile(
    r"\b(?:start|restart|stop|enable|disable|fix|repair|install|uninstall|reinstall|deploy|"
    r"set\s+up|make\s+sure|ensure|get\b.+?\brunning|bring\b.+?\bup|configure|troubleshoot|resolve)\b",
    re.IGNORECASE,
)

# Service / binary / package presence probes whose failure is informative.
_STATUS_PROBE_RE = re.compile(
    r"\b(?:systemctl\s+(?:status|is-active|is-enabled)|service\s+\S+\s+status|rc-service\s+\S+\s+status|"
    r"which\s+\S+|whereis\s+\S+|type\s+\S+|command\s+-v\s+\S+|dpkg\s+-l\s+\S+|"
    r"apt\s+(?:list|show|search)\s+\S+|rpm\s+-q\s+\S+|apk\s+info\s+\S+|pgrep\b|pidof\b|"
    r"\S+\s+(?:--version|version|info|status|--help))\b",
    re.IGNORECASE,
)

_LOCAL_MUTATING_TOOLS = {"shell_exec", "file_write", "file_append", "file_patch", "ast_patch", "file_delete"}
_READ_LOOP_TOOLS = {"artifact_read", "file_read", "dir_list", "ssh_file_read", "web_fetch"}
_REPAIR_TOOLS = _LOCAL_MUTATING_TOOLS | _READ_LOOP_TOOLS | {
    "grep",
    "find_files",
    "finalize_write_session",
    "ssh_exec",
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
}
_INTERACTIVE_SSH_TOOLS = {"ssh_session_start", "ssh_session_read", "ssh_session_send", "ssh_session_close"}
# When an interactive installer stall is detected, only these tools stay exposed.
# Everything else (especially ssh_exec and pipe-to-shell retries) is hidden so the
# model must send one exact answer to the existing session instead of restarting.
_INTERACTIVE_INSTALLER_STALL_ALLOWED_TOOLS = {"ssh_session_read", "ssh_session_send", "ssh_session_close"}


def _task_text(state: Any) -> str:
    bits: list[str] = []
    run_brief = getattr(state, "run_brief", None)
    working_memory = getattr(state, "working_memory", None)
    for value in (
        getattr(run_brief, "original_task", ""),
        getattr(run_brief, "effective_task", ""),
        getattr(working_memory, "current_goal", ""),
        getattr(state, "active_intent", ""),
        " ".join(str(item) for item in (getattr(state, "secondary_intents", []) or [])),
        " ".join(str(item) for item in (getattr(state, "intent_tags", []) or [])),
    ):
        value = str(value or "").strip()
        if value:
            bits.append(value)
    return " ".join(bits).casefold()


def _read_only_status_inquiry(state: Any) -> bool:
    text = _task_text(state)
    if not text:
        return False
    if _ACTION_INTENT_RE.search(text):
        return False
    return bool(_READ_ONLY_STATUS_INQUIRY_RE.search(text))


def _latest_verifier_is_status_probe(state: Any) -> bool:
    verifier = _latest_verifier(state)
    if not isinstance(verifier, dict):
        return False
    command = str(verifier.get("command") or verifier.get("target") or "").strip()
    if not command:
        return False
    return bool(_STATUS_PROBE_RE.search(command))


def _done_gate_diagnostic_failure_exemption(state: Any) -> bool:
    verifier = _latest_verifier(state)
    if not isinstance(verifier, dict):
        return False
    verdict = str(verifier.get("verdict") or "").strip().lower()
    if verdict not in {"fail", "failed", "error"}:
        return False
    return _read_only_status_inquiry(state) and _latest_verifier_is_status_probe(state)


def apply_fama_tool_exposure(
    schemas: list[dict[str, Any]],
    *,
    state: Any,
    mode: str,
    config: Any = None,
) -> list[dict[str, Any]]:
    hidden_tools = fama_hidden_tools_for_exposure(schemas, state=state, mode=mode, config=config)
    if not hidden_tools:
        return list(schemas)
    return [schema for schema in schemas if _tool_name(schema) not in hidden_tools]


def fama_hidden_tools_for_exposure(
    schemas: list[dict[str, Any]],
    *,
    state: Any,
    mode: str,
    config: Any = None,
) -> set[str]:
    del mode
    config = _effective_config(state, config)
    if not fama_enabled(config):
        return set()
    active = active_mitigation_names(state)
    exported = {_tool_name(schema) for schema in schemas}
    hidden_tools: set[str] = set()
    if "done_gate" in active and "task_complete" in exported:
        if _latest_verifier_is_remote_transport_failure(state):
            pass
        # Repair-phase exemption: allow task_complete so the model can exit
        # after successfully patching a file that previously caused a timeout.
        elif _repair_file_write_exemption(state):
            pass  # keep task_complete visible
        elif _done_gate_diagnostic_failure_exemption(state):
            pass  # read-only status inquiry: failure is the answer
        else:
            hidden_tools.add("task_complete")
    if "done_gate" in active and "task_fail" in exported and (_REPAIR_TOOLS & exported):
        if _latest_verifier_is_remote_transport_failure(state):
            pass
        elif _has_pending_ssh_auth_recovery(state):
            pass
        # Dead-end escape hatch: if the same verifier has rejected 5+ times
        # on the same target, let the model call task_fail to report the
        # blocker rather than cycling indefinitely.
        elif _verifier_rejection_count(state) < 5 or _same_target_rejection_streak(state) < 4:
            hidden_tools.add("task_fail")
    if "remote_tool_exposure_guard" in active:
        # Don't hide local mutating tools when the current task explicitly
        # references local paths — the model needs them to complete the task.
        if not _current_task_has_explicit_local_targets(state):
            hidden_tools.update(_LOCAL_MUTATING_TOOLS & exported)
    if "tool_exposure_narrowing" in active:
        repeated_tool = _latest_loop_repeated_tool(state)
        if repeated_tool in _READ_LOOP_TOOLS and repeated_tool in exported:
            hidden_tools.add(repeated_tool)
    # Interactive SSH installer stall: the prompt has not advanced despite prior
    # sends. Force the model to send one exact answer to the existing session and
    # then read the new state. Block every other tool so it cannot retry the
    # installer with ssh_exec or curl|bash.
    if "interactive_installer_stall_capsule" in active:
        hidden_tools.update(exported - _INTERACTIVE_INSTALLER_STALL_ALLOWED_TOOLS)
    # Repair-phase read-only loop breaker: if the model has made repeated
    # read-only calls without mutation progress in repair phase, suppress
    # further reads and force mutation action.
    state_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    if state_phase == "repair":
        stagnation = getattr(state, "stagnation_counters", None)
        if isinstance(stagnation, dict):
            no_progress = int(stagnation.get("no_actionable_progress") or 0)
            if no_progress >= 2:
                for read_tool in _READ_LOOP_TOOLS:
                    if read_tool in exported:
                        # Don't hide file_read if a repair cycle is actively
                        # waiting for a read snapshot; hiding it would deadlock
                        # the recovery.
                        if read_tool == "file_read" and getattr(state, "repair_cycle_id", None):
                            continue
                        hidden_tools.add(read_tool)
                # Also suppress meta read tools that don't mutate
                for meta_tool in ("artifact_grep", "artifact_print", "log_note"):
                    if meta_tool in exported:
                        hidden_tools.add(meta_tool)
    if not _interactive_ssh_tools_exposed(state):
        hidden_tools.update(_INTERACTIVE_SSH_TOOLS & exported)
    return hidden_tools


def fama_hidden_tool_reasons_for_exposure(
    schemas: list[dict[str, Any]],
    *,
    state: Any,
    mode: str,
    config: Any = None,
) -> dict[str, list[str]]:
    hidden_tools = fama_hidden_tools_for_exposure(schemas, state=state, mode=mode, config=config)
    if not hidden_tools:
        return {}
    active = active_mitigation_names(state)
    exported = {_tool_name(schema) for schema in schemas}
    reasons: dict[str, list[str]] = {tool: [] for tool in hidden_tools}
    if "done_gate" in active:
        if "task_complete" in hidden_tools:
            reasons["task_complete"].append("done_gate_requires_verified_acceptance")
        if "task_fail" in hidden_tools:
            reasons["task_fail"].append("done_gate_preserves_repair_path")
    if "remote_tool_exposure_guard" in active:
        for tool in sorted(hidden_tools & _LOCAL_MUTATING_TOOLS):
            reasons[tool].append("remote_tool_exposure_guard_hides_local_mutation")
    if "tool_exposure_narrowing" in active:
        repeated_tool = _latest_loop_repeated_tool(state)
        if repeated_tool in hidden_tools:
            reasons[repeated_tool].append("tool_exposure_narrowing_repeated_tool")
    if "interactive_installer_stall_capsule" in active:
        for tool in sorted(hidden_tools):
            if tool in _INTERACTIVE_INSTALLER_STALL_ALLOWED_TOOLS:
                continue
            reasons[tool].append("interactive_installer_stall_narrows_to_send")
    state_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    stagnation = getattr(state, "stagnation_counters", None)
    if state_phase == "repair" and isinstance(stagnation, dict):
        no_progress = int(stagnation.get("no_actionable_progress") or 0)
        if no_progress >= 2:
            for tool in sorted(hidden_tools & (_READ_LOOP_TOOLS | {"artifact_grep", "artifact_print", "log_note"})):
                reasons[tool].append("repair_read_only_loop_breaker")
    if not _interactive_ssh_tools_exposed(state):
        for tool in sorted(hidden_tools & _INTERACTIVE_SSH_TOOLS):
            reasons[tool].append("interactive_ssh_tools_not_exposed")
    for tool in sorted(hidden_tools):
        if not reasons[tool]:
            if tool not in exported:
                reasons[tool].append("not_exported")
            else:
                reasons[tool].append("fama_policy")
    return reasons


def _has_pending_ssh_auth_recovery(state: Any) -> bool:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    recovery_state = scratchpad.get("_ssh_auth_recovery_state")
    if not isinstance(recovery_state, dict):
        return False
    return any(isinstance(record, dict) for record in recovery_state.values())


def enforce_fama_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    state: Any,
    mode: str,
    config: Any = None,
) -> ToolEnvelope | None:
    config = _effective_config(state, config)
    if not fama_enabled(config) or not done_gate_on_failure(config):
        return None
    if str(tool_name or "").strip() != "task_complete":
        return None
    if "done_gate" not in active_mitigation_names(state):
        return None
    verifier = _latest_verifier(state)
    verdict = str((verifier or {}).get("verdict") or "").strip().lower()
    if verdict == "pass" or bool(getattr(state, "acceptance_waived", False)):
        return None
    if _latest_verifier_is_remote_transport_failure(state):
        return None
    message = str((arguments or {}).get("message") or "")
    if diagnostic_failure_completion_allowed(state, message=message, verifier=verifier):
        return None
    if _done_gate_diagnostic_failure_exemption(state) and diagnostic_completion_reports_failure(message, verifier):
        return None
    required_fps = active_done_gate_fingerprints(state)
    actual_fp = normalize_verifier_target(str((verifier or {}).get("command") or (verifier or {}).get("target") or ""))
    verdict_label = str((verifier or {}).get("verdict") or "").strip().lower()
    failure_mode = str((verifier or {}).get("failure_mode") or "").strip()
    semantic_failure = str((verifier or {}).get("key_stderr") or (verifier or {}).get("key_stdout") or "").strip()
    if not semantic_failure and verdict_label == "fail":
        semantic_failure = "The latest verifier returned a failure verdict."
    error_msg = (
        "FAMA done_gate blocked task_complete because the latest verifier or "
        "acceptance evidence is not satisfied yet."
    )
    if verdict_label == "fail" and failure_mode:
        error_msg += f" Verdict: FAIL (failure_mode={failure_mode}). {semantic_failure}"
    elif verdict_label == "fail":
        error_msg += f" Verdict: FAIL. {semantic_failure}"
    return ToolEnvelope(
        success=False,
        error=error_msg,
        metadata={
            "reason": "fama_done_gate",
            "active_mitigation": "done_gate",
            "last_verifier_verdict": verifier,
            "required_fingerprints": sorted(required_fps),
            "actual_fingerprint": actual_fp,
            "fingerprint_match": bool(actual_fp and actual_fp in required_fps),
            "mode": str(mode or ""),
            "next_required_action": "run verification, satisfy acceptance, or call task_fail with the blocker",
        },
    )


def _interactive_ssh_tools_exposed(state: Any) -> bool:
    active = active_mitigation_names(state)
    # A detected interactive installer stall is the strongest signal that the
    # interactive SSH session tools must be visible so the model can send one
    # exact answer to the existing session.
    if "interactive_installer_stall_capsule" in active:
        return True
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        if bool(
            scratchpad.get("_expose_interactive_session_tools")
            or scratchpad.get("_expose_interactive_ssh_tools")
        ):
            return True
    verifier = _latest_verifier(state)
    if isinstance(verifier, dict):
        failure_text = " ".join(
            str(verifier.get(key) or "")
            for key in ("failure_mode", "ssh_error_class", "key_stderr", "key_stdout", "reason", "message")
        ).lower()
        if "interactive_installer_blocked" in failure_text or "error opening terminal: unknown" in failure_text:
            return True
    return False


def _repair_file_write_exemption(state: Any) -> bool:
    state_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    active_intent = str(getattr(state, "active_intent", "") or "").strip().lower()
    return state_phase == "repair" and active_intent in {"requested_file_patch", "requested_write_file"}


def _latest_verifier_is_remote_transport_failure(state: Any) -> bool:
    verifier = _latest_verifier(state)
    if not verifier:
        return False
    if str(verifier.get("tool") or "").strip() != "ssh_exec":
        return False
    try:
        if int(verifier.get("exit_code")) == 255:
            return True
    except (TypeError, ValueError):
        pass
    combined = "\n".join(
        str(verifier.get(key) or "")
        for key in ("key_stdout", "key_stderr", "failure_mode")
    ).lower()
    return any(
        marker in combined
        for marker in (
            "no route to host",
            "connection refused",
            "connection timed out",
            "network is unreachable",
            "could not resolve hostname",
            "permission denied (publickey",
        )
    )


def _current_task_has_explicit_local_targets(state: Any) -> bool:
    """Return True if the current task or goal references local-only paths."""
    if state is None:
        return False
    texts: list[str] = []
    run_brief = getattr(state, "run_brief", None)
    if run_brief is not None:
        texts.append(str(getattr(run_brief, "original_task", "") or ""))
        texts.append(str(getattr(run_brief, "effective_task", "") or ""))
    working_memory = getattr(state, "working_memory", None)
    if working_memory is not None:
        texts.append(str(getattr(working_memory, "current_goal", "") or ""))
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        handoff = scratchpad.get("_last_task_handoff")
        if isinstance(handoff, dict):
            texts.append(str(handoff.get("effective_task") or ""))
            texts.append(str(handoff.get("current_goal") or ""))
    combined = " ".join(texts).lower()
    local_markers = ("./", "../", "/home/", "/tmp/", "/var/", "/opt/", "/usr/")
    return any(marker in combined for marker in local_markers)


def _tool_name(entry: dict[str, Any]) -> str:
    function = entry.get("function") if isinstance(entry, dict) else None
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip()


def _effective_config(state: Any, config: Any) -> Any:
    if config is not None:
        return config
    scratchpad = getattr(state, "scratchpad", None)
    payload = scratchpad.get("_fama_config") if isinstance(scratchpad, dict) else None
    if not isinstance(payload, dict):
        return None
    return SimpleNamespace(
        fama_enabled=bool(payload.get("enabled", True)),
        fama_mode=str(payload.get("mode") or "lite"),
        fama_done_gate_on_failure=bool(payload.get("done_gate_on_failure", True)),
    )


def _latest_verifier(state: Any) -> dict[str, Any] | None:
    current_verifier = getattr(state, "current_verifier_verdict", None)
    verifier = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
    return verifier if isinstance(verifier, dict) and verifier else None


def _latest_loop_repeated_tool(state: Any) -> str:
    payload = get_fama_state(state)
    signals = payload.get("signals")
    if not isinstance(signals, list):
        return ""
    for signal in reversed(signals):
        if not isinstance(signal, dict):
            continue
        if str(signal.get("kind") or "") != "looping":
            continue
        tool_name = str(signal.get("tool_name") or "").strip()
        if tool_name:
            return tool_name
        evidence = str(signal.get("evidence") or "")
        marker = "repeated_tool="
        if marker in evidence:
            return evidence.split(marker, 1)[1].split(";", 1)[0].strip()
    return ""


def _verifier_rejection_count(state: Any) -> int:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return 0
    return int(scratchpad.get("_verifier_rejection_count", 0) or 0)


def _same_target_rejection_streak(state: Any) -> int:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return 0
    verdict = scratchpad.get("_last_verifier_rejection")
    if not isinstance(verdict, dict):
        return 0
    command = str(verdict.get("command") or verdict.get("target") or "").strip()
    if not command:
        return 0
    key = "_fama_same_target_streak"
    last = scratchpad.get(key)
    if isinstance(last, dict) and last.get("command") == command:
        streak = int(last.get("streak", 0) or 0) + 1
    else:
        streak = 1
    scratchpad[key] = {"command": command, "streak": streak}
    return streak
