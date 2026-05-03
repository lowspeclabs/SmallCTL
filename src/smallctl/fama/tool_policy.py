from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ..models.tool_result import ToolEnvelope
from .config import done_gate_on_failure, fama_enabled
from .signals import get_fama_state
from .state import active_mitigation_names

_LOCAL_MUTATING_TOOLS = {"shell_exec", "file_write", "file_append", "file_patch", "ast_patch", "file_delete"}
_READ_LOOP_TOOLS = {"artifact_read", "file_read", "dir_list", "ssh_file_read", "web_fetch"}


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
        hidden_tools.add("task_complete")
    if "remote_tool_exposure_guard" in active:
        hidden_tools.update(_LOCAL_MUTATING_TOOLS & exported)
    if "tool_exposure_narrowing" in active:
        repeated_tool = _latest_loop_repeated_tool(state)
        if repeated_tool in _READ_LOOP_TOOLS and repeated_tool in exported:
            hidden_tools.add(repeated_tool)
    return hidden_tools


def enforce_fama_tool_call(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    state: Any,
    mode: str,
    config: Any = None,
) -> ToolEnvelope | None:
    del arguments
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
    return ToolEnvelope(
        success=False,
        error=(
            "FAMA done_gate blocked task_complete because the latest verifier or "
            "acceptance evidence is not satisfied yet."
        ),
        metadata={
            "reason": "fama_done_gate",
            "active_mitigation": "done_gate",
            "last_verifier_verdict": verifier,
            "mode": str(mode or ""),
            "next_required_action": "run verification, satisfy acceptance, or call task_fail with the blocker",
        },
    )


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
