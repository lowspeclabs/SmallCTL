from __future__ import annotations

from typing import Any

from ..harness.task_transactions import recovery_context_lines, transaction_from_scratchpad


def _current_task_requires_file_mutation(state: Any | None) -> bool:
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


def _prior_turn_verdict(harness: Any) -> str:
    state = getattr(harness, "state", None)
    if state is None:
        return ""
    return str(getattr(state, "scratchpad", {}).get("_progress_prior_verdict", "") or "").strip()


def _record_command(record: Any) -> str:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    return str(args.get("command") or "").strip()


def _current_verifier_payload(state: Any) -> dict[str, Any] | None:
    current_verifier = getattr(state, "current_verifier_verdict", None)
    verifier = current_verifier() if callable(current_verifier) else getattr(state, "last_verifier_verdict", None)
    return verifier if isinstance(verifier, dict) and verifier else None


def _record_verifier_verdict(harness: Any, record: Any) -> str:
    metadata = record.result.metadata or {}
    verdict = str(metadata.get("verdict") or metadata.get("status") or "").strip()
    if verdict:
        return verdict

    state = getattr(harness, "state", None)
    if state is None:
        return ""
    verifier = _current_verifier_payload(state)
    if verifier is None:
        return ""
    if str(verifier.get("tool") or "").strip() != str(getattr(record, "tool_name", "") or "").strip():
        return ""
    verifier_command = str(verifier.get("command") or "").strip()
    record_command = _record_command(record)
    if verifier_command and record_command and verifier_command != record_command:
        return ""
    return str(verifier.get("verdict") or verifier.get("status") or "").strip()


def _read_repeats_fully_read_target_after_failed_verifier(harness: Any, record: Any) -> bool:
    from .progress_guard_constants import _LAST_FAILED_VERIFIER_KEY

    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    if not isinstance(scratchpad.get(_LAST_FAILED_VERIFIER_KEY), dict):
        return False
    if not _current_task_requires_file_mutation(state):
        return False
    if str(getattr(record, "tool_name", "") or "") != "file_read":
        return False
    args = getattr(record, "args", None)
    if not isinstance(args, dict):
        args = {}
    requested_path = str(args.get("path") or "").strip()
    if not requested_path:
        return False
    history = scratchpad.get("_progress_read_history")
    if not isinstance(history, list):
        return False
    for item in reversed(history[-12:]):
        if not isinstance(item, dict):
            continue
        if str(item.get("tool_name") or "") != "file_read":
            continue
        if str(item.get("path") or "").strip() != requested_path:
            continue
        if bool(item.get("complete_file")) and not bool(item.get("file_content_truncated")):
            return True
    return False


def _latest_complete_file_read_note(scratchpad: dict[str, Any]) -> str:
    history = scratchpad.get("_progress_read_history")
    if not isinstance(history, list):
        return ""
    for item in reversed(history[-8:]):
        if str(item.get("tool_name") or "") != "file_read":
            continue
        if not bool(item.get("complete_file")):
            continue
        if bool(item.get("file_content_truncated")):
            continue
        path = str(item.get("path") or "").strip()
        source_path = str(item.get("source_path") or "").strip()
        read_from_staging = bool(item.get("read_from_staging")) and source_path and source_path != path
        if path:
            if read_from_staging:
                return (
                    f" The last `file_read` fully read the active write-session staged copy for `{path}` "
                    f"from `{source_path}`; that staged read is not proof that the authoritative target file is empty."
                )
            return (
                f" The last `file_read` fully read `{path}`; if a prior message showed `...` or `[truncated]`, "
                "that was chat display compaction, not evidence that the file content is missing."
            )
        return (
            " The last `file_read` fully read the file; if a prior message showed `...` or `[truncated]`, "
            "that was chat display compaction, not evidence that the file content is missing."
        )
    return ""


def _latest_failed_verifier_note(scratchpad: dict[str, Any]) -> str:
    from .progress_guard_constants import _LAST_FAILED_VERIFIER_KEY
    verifier = scratchpad.get(_LAST_FAILED_VERIFIER_KEY)
    if not isinstance(verifier, dict):
        return ""
    command = str(verifier.get("command") or "").strip()
    summary = verifier.get("summary")
    lines = [str(item).strip() for item in summary if str(item).strip()] if isinstance(summary, list) else []
    raw_output = str(verifier.get("raw_output") or "").strip()
    if not command and not lines and not raw_output:
        return ""
    note = ""
    if command:
        note += f" Last verifier failed: `{command}`."
    if lines:
        joined = " | ".join(lines[:8])
        note += f" Failure summary: {joined}."
    if raw_output:
        truncated = raw_output[-1200:] if len(raw_output) > 1200 else raw_output
        note += (
            "\n[REPAIR GROUNDING] The exact verifier output is shown below. "
            "Your next patch MUST address the concrete failure shown here. Do not invent errors that do not appear:\n"
            f"```\n{truncated}\n```"
        )
    note += " Do not reread unchanged evidence; patch the concrete failure or rerun the focused verifier."
    return note


def _latest_blocker_note(scratchpad: dict[str, Any]) -> str:
    blocker = scratchpad.get("_latest_execution_blocker")
    if not isinstance(blocker, dict):
        return ""
    salient = str(blocker.get("salient_error") or "").strip()
    blocker_class = str(blocker.get("blocker_class") or "").strip()
    command = str(blocker.get("command") or "").strip()
    if not salient and not blocker_class:
        return ""
    note = f" Latest blocker: {salient or blocker_class}."
    if command:
        note += f" Failed command: `{command}`."
    if not bool(blocker.get("is_interactive_prompt")):
        note += " Do not keep applying stale interactive-prompt/stdin advice unless the latest blocker is actually a prompt."
    return note


def _last_stalled_action(harness: Any) -> str:
    from .tool_loop_guard_progress import _tool_attempt_history
    history = _tool_attempt_history(harness)
    if not history:
        return ""
    item = history[-1]
    tool_name = str(item.get("tool_name") or "").strip()
    if not tool_name:
        return ""
    return tool_name


def _stagnation_thresholds_for_phase(harness: Any) -> tuple[int, int]:
    """Return (nudge_start, trip_threshold) based on current phase.

    Research-heavy phases get higher thresholds to avoid false-positives
    during legitimate diagnostic or exploratory work.
    """
    from ..remote_scope import remote_scope_is_active
    state = getattr(harness, "state", None)
    if state is None:
        return 3, 5
    if remote_scope_is_active(state):
        return 7, 10
    phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    if phase in {"explore", "repair"}:
        return 3, 6
    return 3, 5


def _build_progress_stagnation_nudge(harness: Any) -> str:
    goal = str(
        getattr(getattr(getattr(harness, "state", None), "run_brief", None), "original_task", "")
        or ""
    ).strip()
    goal_note = f" for `{goal}`" if goal else ""
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    transaction = transaction_from_scratchpad(scratchpad if isinstance(scratchpad, dict) else {})
    tx_lines = recovery_context_lines(transaction)
    last_action = _last_stalled_action(harness)
    last_action_note = f" Last stalled action: {last_action}." if last_action else ""
    blocker_note = _latest_blocker_note(scratchpad if isinstance(scratchpad, dict) else {})
    verifier_note = _latest_failed_verifier_note(scratchpad if isinstance(scratchpad, dict) else {})
    file_read_note = _latest_complete_file_read_note(scratchpad if isinstance(scratchpad, dict) else {})
    tx_note = (" " + " ".join(tx_lines)) if tx_lines else ""
    mutation_note = ""
    if _current_task_requires_file_mutation(state):
        mutation_note = (
            " For a requested file patch, memory notes, artifact searches, and repeated reads are not progress; "
            "call `ssh_file_patch` / `ssh_file_replace_between` for a remote file, `file_patch` for a local file, "
            "or `task_fail(message='...')` with the concrete blocker."
        )
    return (
        "You have made no actionable progress in the last few turns. "
        f"Use the evidence already visible in context{goal_note}.{tx_note}{last_action_note}{blocker_note}{verifier_note}{file_read_note} "
        f"{mutation_note} "
        "Perform the next concrete mutation, run a focused verifier, or call "
        "`task_complete(message='...')` if the task is finished. "
        "Do not repeat the same analysis or read operations. "
        "Choose exactly one: A. Explain the blocker and stop. B. Try a different specific fix. C. Ask for missing information."
    )
