from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ..models.conversation import ConversationMessage
from .state import PendingToolCall

_FAMA_OBSERVE_TASKS: set[asyncio.Task[Any]] = set()

_WRITE_SESSION_GUARD_ERROR_KINDS = {
    "patch_over_rewrite_guard",
    "patch_existing_requires_explicit_replace_strategy",
    "Patch-existing write sessions",
    "chunked_write_overwrite_new_section_after_progress",
    "write_session_staging_path_used_as_target",
}


def _dispatch_stagnation_recovery(harness: Any, guard_error: str) -> None:
    """Inject stagnation recovery nudge and reset counters."""
    from ..graph.progress_guard_support import _current_task_requires_file_mutation
    from ..harness.task_transactions import recovery_context_lines, transaction_from_scratchpad

    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    transaction = transaction_from_scratchpad(scratchpad if isinstance(scratchpad, dict) else {})
    tx_lines = recovery_context_lines(transaction)
    tx_note = (" " + " ".join(tx_lines)) if tx_lines else ""
    phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    active_intent = str(getattr(state, "active_intent", "") or "").strip().lower()
    mutation_required = _current_task_requires_file_mutation(state)
    if mutation_required or (phase == "repair" and active_intent in {"requested_file_patch", "requested_write_file"}):
        repair_directive = (
            " You MUST now call a concrete mutation tool (`file_patch`, `file_write`, or `ast_patch`) "
            "or run a focused verifier (`shell_exec`). Do not read, analyze, or plan further."
        )
        if isinstance(scratchpad, dict):
            scratchpad["_read_only_loop_gate_active"] = True
            scratchpad["_read_only_loop_gate_triggered_at"] = int(getattr(state, "step_count", 0) or 0)
    else:
        repair_directive = " Try a different tool, check permissions, or rethink your approach instead of repeating the same action."
    harness._runlog(
        "recovery_decision",
        "selected stagnation recovery nudge",
        status="scheduled",
        recovery_kind="stagnation",
        guard_error=guard_error,
        step=getattr(harness.state, "step_count", 0),
        current_phase=getattr(harness.state, "current_phase", ""),
        active_intent=getattr(harness.state, "active_intent", ""),
    )
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"System: {guard_error}."
                f"{repair_directive}"
                f"{tx_note}"
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "stagnation",
                "guard_error": guard_error,
            },
        )
    )
    harness._runlog(
        "stagnation_recovery",
        "injected strategy pivot nudge",
        step=harness.state.step_count,
    )
    counters = harness.state.stagnation_counters
    if isinstance(counters, dict):
        # Reset the counter that actually trips the progress-stagnation guard
        # (progress_guard._check_progress_stagnation reads only
        # "no_actionable_progress"); without this the trip re-fires every step.
        for name in ("no_actionable_progress", "no_progress", "repeat_command", "repeat_patch"):
            if name in counters:
                counters[name] = 0
    # tool_history is intentionally retained: clearing it disarms the
    # repeated-action guard (guards.py) and hides the loop from FAMA.
    _observe_stagnation_guard_trip(harness, guard_error)


def _observe_stagnation_guard_trip(harness: Any, guard_error: str) -> None:
    """Route the stagnation guard trip into FAMA.

    The caller clears ``guard_error`` immediately after dispatch, which would
    otherwise skip ``observe_guard_trip`` entirely for stagnation recoveries.
    Emit the observation here, before that happens.
    """
    try:
        from ..fama.runtime import observe_guard_trip

        coro = observe_guard_trip(
            harness,
            guard_error=guard_error,
            tool_history_tail=list((getattr(harness.state, "tool_history", []) or [])[-8:]),
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            task = loop.create_task(coro)
            _FAMA_OBSERVE_TASKS.add(task)
            task.add_done_callback(_FAMA_OBSERVE_TASKS.discard)
        else:
            asyncio.run(coro)
    except Exception as exc:
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "guard_trip_fama_observation_failed",
                "FAMA guard-trip observation failed",
                error=str(exc),
            )


def _dispatch_artifact_read_recovery(harness: Any, graph_state: Any, recovery_hint: tuple[str, str]) -> None:
    """Dispatch artifact_grep as recovery for repeated artifact_read."""
    artifact_id, query = recovery_hint
    from .tool_call_parser import _clear_artifact_read_guard_state
    _clear_artifact_read_guard_state(harness, artifact_id)
    harness._runlog(
        "recovery_decision",
        "selected direct artifact_read recovery dispatch",
        status="scheduled",
        recovery_kind="artifact_read",
        recovery_mode="direct_dispatch",
        tool_name="artifact_grep",
        artifact_id=artifact_id,
        query=query,
        step=getattr(harness.state, "step_count", 0),
    )
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="artifact_grep",
            args={
                "artifact_id": artifact_id,
                "query": query,
            },
            source="system",
        )
    ]
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Auto-advancing repeated `artifact_read` on artifact {artifact_id} "
                f"to `artifact_grep` with query `{query}`."
            ),
            metadata={
                "recovery_kind": "artifact_read",
                "artifact_id": artifact_id,
                "query": query,
                "recovery_mode": "direct_dispatch",
            },
        )
    )
    harness._runlog(
        "artifact_read_recovery",
        "scheduled recovery dispatch",
        step=harness.state.step_count,
        artifact_id=artifact_id,
        query=query,
    )


def _inject_artifact_read_recovery_nudge(harness: Any, recovery_hint: tuple[str, str]) -> None:
    """Inject a user nudge for artifact_read recovery in chat mode."""
    artifact_id, query = recovery_hint
    recovery_armed = harness.state.scratchpad.get("_artifact_read_recovery_nudged")
    if recovery_armed == artifact_id:
        return
    harness._runlog(
        "recovery_decision",
        "selected artifact_read recovery nudge",
        status="scheduled",
        recovery_kind="artifact_read",
        recovery_mode="chat_nudge",
        tool_name="artifact_grep",
        artifact_id=artifact_id,
        query=query,
        step=getattr(harness.state, "step_count", 0),
    )
    msg = (
        f"You are repeating `artifact_read` on artifact {artifact_id}. "
        f"Use `artifact_grep` with query `{query}` instead of reading the same artifact again."
    )
    harness.state.scratchpad["_artifact_read_recovery_nudged"] = artifact_id
    harness.state.scratchpad["_artifact_read_recovery_query"] = query
    harness.state.append_message(
        ConversationMessage(
            role="user",
            content=msg,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "artifact_read",
                "artifact_id": artifact_id,
                "query": query,
            },
        )
    )
    harness._runlog(
        "artifact_read_recovery",
        "injected recovery nudge",
        step=harness.state.step_count,
        artifact_id=artifact_id,
        query=query,
    )


def _extract_write_session_guard_target_path(guard_error: str, recent_errors: list[str]) -> str | None:
    """Find the most likely target path from a write-session guard trip."""
    # Prefer the most recent write-session-related error with a path.
    for error in reversed(recent_errors):
        text = str(error or "")
        if not any(kind in text for kind in _WRITE_SESSION_GUARD_ERROR_KINDS):
            if "file_write to `" not in text:
                continue
        m = re.search(r"file_write to `([^`]+)`", text)
        if m:
            return m.group(1).strip()
        # Also accept backtick-wrapped paths that look like files.
        m = re.search(r"`([^`]+\.[a-zA-Z0-9]+)`", text)
        if m:
            return m.group(1).strip()
    # Fallback to the guard error text itself.
    m = re.search(r"file_write to `([^`]+)`", str(guard_error or ""))
    if m:
        return m.group(1).strip()
    return None


def _is_write_session_guard_trip(guard_error: str, recent_errors: list[str]) -> bool:
    """Return True if max_consecutive_errors is dominated by write-session failures."""
    if "max_consecutive_errors" not in str(guard_error or "").lower():
        return False
    ws_errors = 0
    for error in recent_errors:
        text = str(error or "")
        if any(kind in text for kind in _WRITE_SESSION_GUARD_ERROR_KINDS):
            ws_errors += 1
        elif text.startswith("file_write:") and "file_write to `" in text:
            # A file_write failure that is not explicitly a write-session error
            # but still counts as part of the stuck write loop.
            ws_errors += 1
    # Need at least 3 related errors or a majority of the configured threshold.
    return ws_errors >= 3


def _dispatch_write_session_guard_recovery(
    harness: Any,
    graph_state: Any,
    guard_error: str,
) -> bool:
    """Recover from max_consecutive_errors caused by a stuck write session.

    Aborts the active write session, clears error counters, and schedules a
    fresh file_read so the model can choose a correct write shape instead of
    looping on implicit chunked/patch-existing semantics.
    """
    recent_errors = list(getattr(harness.state, "recent_errors", []) or [])
    if not _is_write_session_guard_trip(guard_error, recent_errors):
        return False

    target_path = _extract_write_session_guard_target_path(guard_error, recent_errors)
    if not target_path:
        return False

    session = getattr(harness.state, "write_session", None)
    aborted_session_id: str | None = None
    if session is not None:
        session_target = str(getattr(session, "write_target_path", "") or "").strip()
        if session_target:
            try:
                from ..tools.fs import _same_target_path

                matches = _same_target_path(
                    session_target, target_path, getattr(harness.state, "cwd", None)
                )
            except Exception:
                matches = session_target == target_path
            if matches:
                from .write_session_outcomes_support import _abort_write_session

                aborted_session_id = str(
                    getattr(session, "write_session_id", "") or ""
                ).strip()
                _abort_write_session(harness, session)

    # Clear the error state that caused the guard trip so the runtime can continue.
    harness.state.recent_errors = []
    harness.state.tool_history = []
    harness.state.stagnation_counters = {}
    scratchpad = getattr(harness.state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        scratchpad.pop("_tool_attempt_history", None)

    # Schedule a fresh read of the target so the model has current content.
    read_args = {"path": target_path}
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args=read_args,
            raw_arguments=json.dumps(read_args, ensure_ascii=True, sort_keys=True),
            source="system",
        )
    ]
    from .tool_call_parser import allow_repeated_tool_call_once

    allow_repeated_tool_call_once(harness, "file_read", read_args)

    session_note = (
        f" Active Write Session `{aborted_session_id}` has been aborted."
        if aborted_session_id
        else ""
    )
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Recovery: repeated `file_write` failures to `{target_path}` were caused by a stuck "
                f"patch-existing/chunked write session.{session_note} "
                "Read the current file content, then choose exactly one approach: "
                "use `file_write(path=..., replace_strategy='overwrite')` to replace the entire file, "
                "or use `file_patch` for a narrow exact edit. Do not retry implicit chunked writes."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "write_session_guard_recovery",
                "target_path": target_path,
                "aborted_session_id": aborted_session_id or "",
            },
        )
    )
    harness._runlog(
        "write_session_guard_recovery",
        "aborted stuck write session and scheduled fresh file_read after max_consecutive_errors",
        target_path=target_path,
        aborted_session_id=aborted_session_id or "",
        step=getattr(harness.state, "step_count", 0),
    )
    return True
