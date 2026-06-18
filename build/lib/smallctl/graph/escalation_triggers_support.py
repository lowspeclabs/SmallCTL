from __future__ import annotations

from typing import Any

_APT_SOURCES_FAILURE_CLASSES = {"apt_sources_malformed", "apt_sources_deb822"}
_APT_TIP_TAG = "{apt_sources_tip}"
_APT_CONFIRMED_TAG = "{apt_sources_confirmed}"


def _fama_signal_classes_from_scratchpad(scratchpad: dict[str, Any]) -> list[str]:
    fama = scratchpad.get("_fama")
    if not isinstance(fama, dict):
        return []
    signals = fama.get("signals")
    if not isinstance(signals, list):
        return []
    classes: list[str] = []
    for item in signals[-8:]:
        if not isinstance(item, dict):
            continue
        for key in ("failure_class", "kind"):
            value = str(item.get(key) or "").strip()
            if value:
                classes.append(value)
    return classes


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _has_active_verifier_failure(harness: Any) -> bool:
    """Return True if the latest verifier verdict is a failure."""
    verifier = getattr(getattr(harness, "state", None), "last_verifier_verdict", None)
    if not isinstance(verifier, dict):
        return False
    return str(verifier.get("verdict") or "").strip().lower() in {"fail", "failed", "error"}


def _consecutive_verifier_failure_class(state: Any, threshold: int = 3) -> str | None:
    """Return the failure class if the last `threshold` failure events are all verifier failures of the same class."""
    failure_events = getattr(state, "failure_events", None)
    if not isinstance(failure_events, list) or len(failure_events) < threshold:
        return None
    recent = failure_events[-threshold:]
    classes = [str(getattr(e, "failure_class", "") or "").strip() for e in recent]
    if not all(classes):
        return None
    first = classes[0]
    if not all(c == first for c in classes):
        return None
    if first not in {
        "verifier_failed",
        "verifier_timeout",
        "infinite_loop_suspected",
        "test_failed",
        "syntax_error",
        "import_error",
    }:
        return None
    return first


def _patch_stall_trigger(state: Any, graph_state: Any, *, threshold: int) -> str:
    counters = getattr(state, "stagnation_counters", None)
    counters = counters if isinstance(counters, dict) else {}
    if _safe_int(counters.get("repeat_patch"), 0) >= threshold:
        return "repeat_patch"
    if _safe_int(counters.get("no_actionable_progress"), 0) >= threshold and _last_turn_touched_patch_path(graph_state):
        return "no_actionable_progress_after_patch"

    failure_events = getattr(state, "failure_events", None)
    if isinstance(failure_events, list):
        for event in reversed(failure_events[-8:]):
            failure_class = str(getattr(event, "failure_class", "") or "").strip()
            fama_kind = str(getattr(event, "fama_kind", "") or "").strip()
            if failure_class == "write_session_stall" or fama_kind == "write_session_stall":
                return "write_session_stall"

    scratchpad = getattr(state, "scratchpad", None)
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    for signal in _fama_signal_classes_from_scratchpad(scratchpad):
        if signal == "write_session_stall":
            return "write_session_stall"

    if scratchpad.get("_repair_cycle_escalation_ready") and _last_turn_touched_patch_path(graph_state):
        return "repair_cycle_exhausted"
    return ""


def _latest_patch_stall_path(state: Any, graph_state: Any) -> str:
    for record in reversed(getattr(graph_state, "last_tool_results", []) or []):
        args = record.args if isinstance(getattr(record, "args", None), dict) else {}
        path = str(args.get("path") or "").strip()
        if path:
            return path
    changed = getattr(state, "files_changed_this_cycle", None)
    if isinstance(changed, list) and changed:
        return str(changed[-1] or "").strip()
    return ""


def _last_turn_touched_patch_path(graph_state: Any) -> bool:
    for record in getattr(graph_state, "last_tool_results", []) or []:
        if record.tool_name in {"file_patch", "ast_patch", "file_write", "file_append"}:
            return True
    return False


def _has_apt_sources_tip(state: Any) -> str:
    """Return existing confirmed apt-sources tip from working memory if present."""
    wm = getattr(state, "working_memory", None)
    if wm is None:
        return ""
    for fact in getattr(wm, "known_facts", []) or []:
        text = str(fact or "").strip()
        if _APT_TIP_TAG in text and _APT_CONFIRMED_TAG in text:
            return text
    return ""


def _store_apt_sources_tip(state: Any, tip: str, classification: str = "") -> None:
    """Store a confirmed apt-sources tip in working memory.

    Format: {apt_sources_tip} {classification} <tip_text> {apt_sources_confirmed}
    """
    wm = getattr(state, "working_memory", None)
    if wm is None:
        return
    if _APT_TIP_TAG not in tip:
        tip = f"{_APT_TIP_TAG} {tip}"
    if classification and f"class:{classification}" not in tip:
        tip = f"{tip} class:{classification}"
    if _APT_CONFIRMED_TAG not in tip:
        tip = f"{tip} {_APT_CONFIRMED_TAG}"
    existing = list(getattr(wm, "known_facts", []) or [])
    if tip not in existing:
        existing.append(tip)
        # Keep last 12 facts
        wm.known_facts = existing[-12:]
        state.touch()


def _get_pending_apt_sources_tip(state: Any) -> dict[str, Any]:
    """Retrieve pending apt-sources escalation result from scratchpad."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    return scratchpad.get("_pending_apt_sources_tip") or {}


def _clear_pending_apt_sources_tip(state: Any) -> None:
    """Clear pending apt-sources tip from scratchpad."""
    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict) and "_pending_apt_sources_tip" in scratchpad:
        del scratchpad["_pending_apt_sources_tip"]
        state.touch()


def _maybe_confirm_apt_sources_tip(state: Any, record: Any) -> str:
    """Check if a successful apt command confirms a pending tip.

    If a pending tip exists in scratchpad and apt succeeds, extract the tip
    and classification from the escalation response, save to working memory,
    and clear the pending tip.
    """
    from ..graph.state import ToolExecutionRecord
    if not isinstance(record, ToolExecutionRecord):
        return ""
    if not record.result.success:
        return ""
    tool_name = str(getattr(record, "tool_name", "") or "").strip()
    if tool_name not in {"ssh_exec", "shell_exec"}:
        return ""
    command = str(record.args.get("command") or "").strip().lower()
    if "apt" not in command:
        return ""

    # Verify apt actually succeeded (no errors in stderr)
    stderr = str(record.result.error or "").lower()
    stdout = str(getattr(record.result, "stdout", "") or "").lower()
    combined = f"{stderr} {stdout}"
    if (
        "malformed" in combined
        and "sources" in combined
        and ("e:" in combined or "apt" in combined)
    ):
        # Apt still failing - don't confirm yet
        return ""
    if "the list of sources could not be read" in combined:
        # Apt still failing - don't confirm yet
        return ""

    pending = _get_pending_apt_sources_tip(state)
    if not pending:
        return ""

    # Extract tip from escalation response
    next_action = pending.get("recommended_next_action", {})
    tip_text = ""
    if isinstance(next_action, dict):
        tip_text = str(next_action.get("reason") or "").strip()
    if not tip_text:
        repair_plan = pending.get("repair_plan", "")
        tip_text = str(repair_plan or "").strip()

    # Extract classification from response
    classification = str(pending.get("classification") or pending.get("failure_diagnosis") or "").strip()
    if not classification:
        classification = "apt_sources_malformed"

    if tip_text and len(tip_text) > 10:
        _store_apt_sources_tip(state, tip_text, classification)
        _clear_pending_apt_sources_tip(state)
        runlog = getattr(state, "_runlog", None)
        if callable(runlog):
            runlog(
                "apt_sources_tip_confirmed",
                "apt command succeeded; saved confirmed tip to working memory",
                classification=classification,
                tip_preview=tip_text[:120],
            )
        return f"{_APT_TIP_TAG} class:{classification} {tip_text} {_APT_CONFIRMED_TAG}"

    _clear_pending_apt_sources_tip(state)
    return ""
