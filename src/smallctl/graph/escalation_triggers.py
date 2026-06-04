from __future__ import annotations

import json
from typing import Any

from ..models.conversation import ConversationMessage
from ..state import json_safe_value
from ..harness.escalation_service import EscalationService
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord
from .tool_loop_guards import _tool_call_fingerprint

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
    # Only look at the last `threshold` events
    recent = failure_events[-threshold:]
    # All must have a failure_class
    classes = [str(getattr(e, "failure_class", "") or "").strip() for e in recent]
    if not all(classes):
        return None
    # All must be the same class
    first = classes[0]
    if not all(c == first for c in classes):
        return None
    # Must be a verifier-related failure class
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

async def _maybe_auto_trigger_escalation_for_tool_loop(
    *,
    harness: Any,
    pending: PendingToolCall,
    repeat_error: str,
    force: bool = False,
) -> bool:
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not force and not bool(getattr(config, "escalation_auto_trigger", False)):
        return False
    if not _has_active_verifier_failure(harness):
        return False

    fingerprint = _tool_call_fingerprint(pending.tool_name, pending.args)
    seen = harness.state.scratchpad.get("_escalation_auto_tool_loop_fingerprints")
    if not isinstance(seen, list):
        seen = []
    if fingerprint in seen:
        return False

    harness.state.scratchpad["_tool_loop_suppression"] = {
        "tool_name": pending.tool_name,
        "arguments": json_safe_value(pending.args),
        "error": repeat_error,
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason=f"Repeated `{pending.tool_name}` call was blocked by the tool-loop guard.",
        question="What is the smallest safe next evidence-gathering or repair step?",
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    harness.state.scratchpad["_escalation_auto_tool_loop_fingerprints"] = seen[-20:]
    if not bool(result.get("success")):
        return False

    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice. Treat this as advice only; "
                "choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_tool_loop",
                "tool_name": pending.tool_name,
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_tool_loop_advisory",
            "injected escalation advisory after repeated tool loop",
            tool_name=pending.tool_name,
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


async def _maybe_auto_trigger_escalation_for_same_tool_failures(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Auto-escalate when the same tool fails multiple times in one turn with different arguments.

    Only triggers when there is an active verifier failure."""
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return False
    if not _has_active_verifier_failure(harness):
        return False

    failure_counts: dict[str, int] = {}
    for record in graph_state.last_tool_results:
        if not record.result.success:
            failure_counts[record.tool_name] = failure_counts.get(record.tool_name, 0) + 1

    for tool_name, count in failure_counts.items():
        if count < 3:
            continue

        seen_key = f"_escalation_auto_same_tool_fingerprints:{tool_name}"
        seen = harness.state.scratchpad.get(seen_key)
        if isinstance(seen, list) and harness.state.step_count in seen:
            continue

        harness.state.scratchpad["_tool_loop_suppression"] = {
            "tool_name": tool_name,
            "error": f"Same tool failed {count} times in one turn with different arguments.",
        }
        from ..harness.escalation_service import EscalationService

        result = await EscalationService(harness).run(
            reason=f"`{tool_name}` failed {count} times in one turn with different arguments.",
            question="What is the smallest safe next evidence-gathering or repair step?",
            requested_output="next_action",
            risk_level="medium",
            source="auto",
        )
        if not isinstance(seen, list):
            seen = []
        seen.append(harness.state.step_count)
        harness.state.scratchpad[seen_key] = seen[-10:]

        if not bool(result.get("success")):
            continue

        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    "Escalation advisor returned bounded recovery advice. Treat this as advice only; "
                    "choose any next action through normal tool policy.\n"
                    f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "escalation_advisory",
                    "source": "auto_same_tool_failures",
                    "tool_name": tool_name,
                    "escalation_id": result.get("escalation_id"),
                },
            )
        )
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "escalation_auto_same_tool_failures_advisory",
                "injected escalation advisory after same-tool repeated failures",
                tool_name=tool_name,
                escalation_id=result.get("escalation_id"),
                verdict=result.get("verdict"),
            )
        return True

    return False


async def _maybe_auto_trigger_escalation_for_patch_stall(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Auto-escalate after patch/write-session stalls once policy evidence exists.

    Only triggers when there is an active verifier failure (i.e. the patch
    is failing to fix a verifier/test failure)."""
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return False
    if not _has_active_verifier_failure(harness):
        return False

    state = getattr(harness, "state", None)
    if state is None:
        return False
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False

    threshold = max(3, _safe_int(getattr(config, "escalation_repeated_failure_threshold", 3), 3))
    trigger = _patch_stall_trigger(state, graph_state, threshold=threshold)
    if not trigger:
        return False

    fingerprint = "|".join(
        [
            trigger,
            str(getattr(state, "step_count", 0) or 0),
            _latest_patch_stall_path(state, graph_state),
        ]
    )
    seen = scratchpad.get("_escalation_auto_patch_stall_fingerprints")
    if not isinstance(seen, list):
        seen = []
    if fingerprint in seen:
        return False

    scratchpad["_tool_loop_suppression"] = {
        "tool_name": "file_patch",
        "error": f"Patch/write-session stall detected: {trigger}.",
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason=f"Patch/write-session stall detected after file mutation: {trigger}.",
        question="What is the smallest safe next evidence-gathering or repair step?",
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    scratchpad["_escalation_auto_patch_stall_fingerprints"] = seen[-20:]
    if not bool(result.get("success")):
        return False

    state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice for a patch stall. "
                "Treat this as advice only; choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_patch_stall",
                "trigger": trigger,
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_patch_stall_advisory",
            "injected escalation advisory after patch/write-session stall",
            trigger=trigger,
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


async def _maybe_auto_trigger_escalation_for_completion_block(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Handle explicit escalation requests stuck behind repeated task_complete blocks."""
    state = getattr(harness, "state", None)
    if state is None:
        return False
    if not _user_requested_escalation(state):
        return False

    threshold = max(2, _safe_int(getattr(getattr(harness, "config", None), "escalation_repeated_failure_threshold", 2), 2))
    if _consecutive_task_complete_verification_blocks(state, threshold=threshold) < threshold:
        return False

    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    fingerprint = f"completion_block|{getattr(state, 'step_count', 0)}"
    seen = scratchpad.get("_escalation_auto_completion_block_fingerprints")
    if not isinstance(seen, list):
        seen = []
    if fingerprint in seen:
        return False

    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return _emit_escalation_config_blocker(
            state,
            scratchpad,
            seen,
            fingerprint,
            "Escalation was explicitly requested, but escalation is disabled. Configure escalation_enabled, escalation_endpoint, and escalation_model.",
        )
    if not str(getattr(config, "escalation_endpoint", "") or "").strip() or not str(getattr(config, "escalation_model", "") or "").strip():
        return _emit_escalation_config_blocker(
            state,
            scratchpad,
            seen,
            fingerprint,
            "Escalation was explicitly requested, but escalation_endpoint or escalation_model is missing.",
        )
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return _emit_escalation_config_blocker(
            state,
            scratchpad,
            seen,
            fingerprint,
            "Escalation was explicitly requested, but escalation_auto_trigger is disabled. Call escalate_to_bigger_model or enable escalation_auto_trigger.",
        )

    scratchpad["_tool_loop_suppression"] = {
        "tool_name": "task_complete",
        "error": "Repeated task_complete calls were blocked by post-change verification while escalation was requested.",
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason="User explicitly requested escalation after repeated task_complete verification blocks.",
        question="What is the smallest safe next step to resolve the stuck completion/verification loop?",
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    scratchpad["_escalation_auto_completion_block_fingerprints"] = seen[-20:]
    if not bool(result.get("success")):
        return False

    state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice for a repeated task_complete block. "
                "Treat this as advice only; choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_completion_block",
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_completion_block_advisory",
            "injected escalation advisory after repeated task_complete block",
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


def _emit_escalation_config_blocker(
    state: Any,
    scratchpad: dict[str, Any],
    seen: list[str],
    fingerprint: str,
    message: str,
) -> bool:
    seen.append(fingerprint)
    scratchpad["_escalation_auto_completion_block_fingerprints"] = seen[-20:]
    scratchpad["_last_escalation"] = {
        "step_count": getattr(state, "step_count", 0),
        "trigger": "completion_block_config_error",
        "verdict": "config_error",
    }
    state.append_message(
        ConversationMessage(
            role="system",
            content=message,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_config_blocker",
                "source": "auto_completion_block",
            },
        )
    )
    recent_errors = getattr(state, "recent_errors", None)
    if isinstance(recent_errors, list):
        recent_errors.append(message)
    return True


def _consecutive_task_complete_verification_blocks(state: Any, *, threshold: int) -> int:
    failure_events = getattr(state, "failure_events", None)
    if not isinstance(failure_events, list):
        return 0
    count = 0
    for event in reversed(failure_events):
        tool_name = str(getattr(event, "tool_name", "") or "").strip()
        failure_class = str(getattr(event, "failure_class", "") or "").strip()
        if tool_name == "task_complete" and failure_class == "post_change_verification_required":
            count += 1
            if count >= threshold:
                return count
            continue
        if count:
            break
    return count


def _user_requested_escalation(state: Any) -> bool:
    for messages_attr in ("conversation_history", "transcript_messages", "recent_messages"):
        messages = getattr(state, messages_attr, None)
        if not isinstance(messages, list):
            continue
        for message in reversed(messages[-20:]):
            if not isinstance(message, dict):
                role = getattr(message, "role", None)
                content = getattr(message, "content", None)
            else:
                role = message.get("role")
                content = message.get("content")
            if str(role or "").strip().lower() != "user":
                continue
            lowered = str(content or "").lower()
            if any(marker in lowered for marker in ("escalate", "bigger model", "larger model", "stronger model")):
                return True
    return False


def _patch_stall_trigger(state: Any, graph_state: GraphRunState, *, threshold: int) -> str:
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

    # Mandatory escalation after 2 repair cycles on the same target
    if scratchpad.get("_repair_cycle_escalation_ready") and _last_turn_touched_patch_path(graph_state):
        return "repair_cycle_exhausted"
    return ""


def _latest_patch_stall_path(state: Any, graph_state: GraphRunState) -> str:
    for record in reversed(getattr(graph_state, "last_tool_results", []) or []):
        args = record.args if isinstance(getattr(record, "args", None), dict) else {}
        path = str(args.get("path") or "").strip()
        if path:
            return path
    changed = getattr(state, "files_changed_this_cycle", None)
    if isinstance(changed, list) and changed:
        return str(changed[-1] or "").strip()
    return ""


def _last_turn_touched_patch_path(graph_state: GraphRunState) -> bool:
    for record in getattr(graph_state, "last_tool_results", []) or []:
        if record.tool_name in {"file_patch", "ast_patch", "file_write", "file_append"}:
            return True
    return False


async def _maybe_auto_trigger_escalation_for_verifier_stall(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Auto-escalate after 3 consecutive verifier failures of the same class.

    This forces escalation when the small model is stuck in a verifier
    failure loop (e.g., same test/import/runtime error repeated)."""
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return False
    if not _has_active_verifier_failure(harness):
        return False

    state = getattr(harness, "state", None)
    if state is None:
        return False

    threshold = max(3, _safe_int(getattr(config, "escalation_repeated_failure_threshold", 3), 3))
    failure_class = _consecutive_verifier_failure_class(state, threshold=threshold)
    if not failure_class:
        return False

    fingerprint = f"verifier_stall|{failure_class}|{getattr(state, 'step_count', 0)}"
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    seen = scratchpad.get("_escalation_auto_verifier_stall_fingerprints")
    if not isinstance(seen, list):
        seen = []
    if fingerprint in seen:
        return False

    scratchpad["_tool_loop_suppression"] = {
        "tool_name": "verifier",
        "error": f"Verifier stalled with {threshold} consecutive {failure_class} failures.",
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason=f"{threshold} consecutive verifier failures of class '{failure_class}'. Small model is stuck in a repair loop.",
        question="What is the smallest safe next evidence-gathering or repair step?",
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    scratchpad["_escalation_auto_verifier_stall_fingerprints"] = seen[-20:]
    if not bool(result.get("success")):
        return False

    state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice for repeated verifier failures. "
                "Treat this as advice only; choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_verifier_stall",
                "failure_class": failure_class,
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_verifier_stall_advisory",
            "injected escalation advisory after repeated verifier failures",
            failure_class=failure_class,
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


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


_APT_SOURCES_FAILURE_CLASSES = {"apt_sources_malformed", "apt_sources_deb822"}
_APT_TIP_TAG = "{apt_sources_tip}"
_APT_CONFIRMED_TAG = "{apt_sources_confirmed}"


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


def _maybe_confirm_apt_sources_tip(state: Any, record: ToolExecutionRecord) -> str:
    """Check if a successful apt command confirms a pending tip.

    If a pending tip exists in scratchpad and apt succeeds, extract the tip
    and classification from the escalation response, save to working memory,
    and clear the pending tip.
    """
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


async def _maybe_auto_trigger_escalation_for_apt_sources_failure(
    *,
    harness: Any,
    graph_state: GraphRunState,
) -> bool:
    """Auto-escalate when apt commands fail due to malformed sources.list files.

    If a tip already exists in working memory, nudge the model to try it
    instead of escalating again.
    """
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "escalation_enabled", False)):
        return False
    if not bool(getattr(config, "escalation_auto_trigger", False)):
        return False

    state = getattr(harness, "state", None)
    if state is None:
        return False
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False

    # Look for apt sources failure in recent tool results.
    # IMPORTANT: apt commands often return exit_code=0 (SSH success) but write
    # errors to stderr (e.g., "E: Malformed entry 1 in sources file"). We must
    # inspect stderr even when record.result.success is True.
    failure_count = 0
    for record in getattr(graph_state, "last_tool_results", []) or []:
        if not isinstance(record, ToolExecutionRecord):
            continue
        tool_name = str(getattr(record, "tool_name", "") or "").strip()
        if tool_name not in {"ssh_exec", "shell_exec"}:
            continue
        command = str(record.args.get("command") or "").strip().lower()
        if "apt" not in command:
            continue
        failure_class = str(
            getattr(record.result, "failure_class", "")
            or getattr(record, "failure_class", "")
            or ""
        ).strip()
        stderr = str(record.result.error or "").lower()
        stdout = str(getattr(record.result, "stdout", "") or "").lower()
        combined = f"{stderr} {stdout}"
        if failure_class in _APT_SOURCES_FAILURE_CLASSES:
            failure_count += 1
            continue
        # Heuristic: apt errors in stderr even with exit code 0
        if (
            "malformed" in combined
            and "sources" in combined
            and ("e:" in combined or "apt" in combined)
        ):
            failure_count += 1
            continue
        # Broader catch-all for apt sources failures
        if "the list of sources could not be read" in combined:
            failure_count += 1

    if failure_count == 0:
        return False

    # Check if we already have a tip
    existing_tip = _has_apt_sources_tip(state)
    if existing_tip:
        # Nudge the model to use the existing tip instead of escalating
        state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    "APT SOURCES FAILURE RECOVERY: A previous escalation provided a tip for this exact failure. "
                    f"Try this first: {existing_tip}\n"
                    "Apply this tip before calling escalate_to_bigger_model again."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "apt_sources_tip",
                    "source": "working_memory",
                },
            )
        )
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "apt_sources_tip_nudge",
                "nudged model to use existing apt-sources tip from working memory",
                tip=existing_tip[:120],
            )
        return True

    # Check if we already escalated for this failure mode recently
    seen_key = "_escalation_auto_apt_sources_fingerprints"
    seen = scratchpad.get(seen_key)
    if not isinstance(seen, list):
        seen = []
    fingerprint = f"apt_sources_failure:{state.step_count}"
    if fingerprint in seen:
        return False

    scratchpad["_tool_loop_suppression"] = {
        "tool_name": "ssh_exec",
        "error": "apt sources malformed; escalating to bigger model for deb822 guidance.",
    }
    from ..harness.escalation_service import EscalationService

    result = await EscalationService(harness).run(
        reason="apt command reports malformed sources file error in stderr (exit code may be 0 but apt failed).",
        question=(
            "How should I fix a malformed apt sources file on a modern Debian/Ubuntu system that uses deb822 .sources format? "
            "Include a 'classification' field in your response describing the type of fix (e.g., deb822_format_correction, sources_list_replacement, etc.)."
        ),
        requested_output="next_action",
        risk_level="medium",
        source="auto",
    )
    seen.append(fingerprint)
    scratchpad[seen_key] = seen[-10:]

    if not bool(result.get("success")):
        return False

    # Store the escalation result as pending; only save to working memory after verification
    scratchpad["_pending_apt_sources_tip"] = result
    state.touch()

    state.append_message(
        ConversationMessage(
            role="system",
            content=(
                "Escalation advisor returned bounded recovery advice for apt sources failure. "
                "Treat this as advice only; choose any next action through normal tool policy.\n"
                f"{json.dumps(json_safe_value(result), ensure_ascii=True, sort_keys=True)}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "escalation_advisory",
                "source": "auto_apt_sources_failure",
                "escalation_id": result.get("escalation_id"),
            },
        )
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "escalation_auto_apt_sources_advisory",
            "injected escalation advisory after apt sources failure",
            escalation_id=result.get("escalation_id"),
            verdict=result.get("verdict"),
        )
    return True


