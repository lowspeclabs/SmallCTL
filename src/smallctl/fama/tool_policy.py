from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..diagnostic_tasks import diagnostic_failure_completion_allowed
from ..models.tool_result import ToolEnvelope
from .config import done_gate_on_failure, fama_enabled
from .fingerprints import active_done_gate_fingerprints, normalize_verifier_target
from .signals import get_fama_state
from .state import active_mitigation_names

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
        # Repair-phase exemption: allow task_complete so the model can exit
        # after successfully patching a file that previously caused a timeout.
        state_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
        active_intent = str(getattr(state, "active_intent", "") or "").strip().lower()
        if state_phase == "repair" and active_intent in {"requested_file_patch", "requested_write_file"}:
            pass  # keep task_complete visible
        else:
            hidden_tools.add("task_complete")
    if "done_gate" in active and "task_fail" in exported and (_REPAIR_TOOLS & exported):
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
                        hidden_tools.add(read_tool)
                # Also suppress meta read tools that don't mutate
                for meta_tool in ("artifact_grep", "artifact_print", "log_note"):
                    if meta_tool in exported:
                        hidden_tools.add(meta_tool)
    if not _interactive_ssh_tools_exposed(state):
        hidden_tools.update(_INTERACTIVE_SSH_TOOLS & exported)
    return hidden_tools


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
    message = str((arguments or {}).get("message") or "")
    if diagnostic_failure_completion_allowed(state, message=message, verifier=verifier):
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
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        return bool(
            scratchpad.get("_expose_interactive_session_tools")
            or scratchpad.get("_expose_interactive_ssh_tools")
        )
    return False


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
