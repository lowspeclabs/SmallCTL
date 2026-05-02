from __future__ import annotations

import re
from typing import Any

from ..models.conversation import ConversationMessage
from ..shell_utils import is_read_only_shell_evidence_action as _is_read_only_shell_evidence_action
from ..state import json_safe_value
from .tool_loop_guard_progress import (
    _coerce_int_or_none,
    _requested_artifact_read_target,
    _requested_file_read_range,
    _tool_attempt_history,
)

_MUTATION_TOOLS = {
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
}

_READ_TOOLS = {
    "artifact_read",
    "ssh_file_read",
    "file_read",
}
_REMOTE_PATH_RE = re.compile(r"(?<![\w/])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+")

def _is_ssh_exec_read_command(record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not command:
        return False
    return _is_read_only_shell_evidence_action(command)


def _turn_has_actionable_progress(harness: Any, graph_state: Any) -> bool:
    """Return True if the current turn changed actionable state."""
    last_tool_results = getattr(graph_state, "last_tool_results", []) or []
    last_assistant_text = str(getattr(graph_state, "last_assistant_text", "") or "").strip()

    # 1. Task completion
    for record in last_tool_results:
        if record.tool_name == "task_complete" and record.result.success:
            return True

    # 2. Successful mutation with changed=True
    for record in last_tool_results:
        if record.tool_name in _MUTATION_TOOLS:
            if record.result.success and (record.result.metadata or {}).get("changed") is True:
                return True

    # 3. Plan step state change
    if _plan_step_changed(harness):
        return True

    # 4. Successful verifier with a new verdict
    for record in last_tool_results:
        if record.tool_name in {"shell_exec", "ssh_exec"}:
            metadata = record.result.metadata or {}
            verdict = str(metadata.get("verdict") or metadata.get("status") or "").strip()
            if verdict:
                prior = _prior_turn_verdict(harness)
                if verdict != prior:
                    return True

    # 4b. Novel remote SSH observations count as progress even when the command failed.
    for record in last_tool_results:
        if record.tool_name == "ssh_exec" and _ssh_exec_has_novel_remote_observation(harness, record):
            return True

    # 5. Successful read of a new artifact/ssh_file range or ssh_exec read command
    for record in last_tool_results:
        if record.tool_name == "artifact_read" and record.result.success:
            if _artifact_read_is_past_eof(harness, record):
                return False
            if _artifact_read_result_is_new_range(harness, record):
                return True
        if record.tool_name == "ssh_file_read" and record.result.success:
            if _ssh_file_read_is_past_eof(harness, record):
                return False
            if _ssh_file_read_result_is_new_range(harness, record):
                return True
        if record.tool_name == "ssh_exec" and record.result.success:
            if _is_ssh_exec_read_command(record):
                if _ssh_exec_read_is_new(harness, record):
                    return True
            else:
                return True

    # 6. Any other successful non-read, non-mutation, non-exec tool
    #    (shell_exec/ssh_exec are handled above; identical calls are caught by loop guards)
    for record in last_tool_results:
        if record.result.success and record.tool_name not in _READ_TOOLS and record.tool_name not in _MUTATION_TOOLS and record.tool_name not in {"shell_exec", "ssh_exec"}:
            return True

    # 7. No-tool turn with non-repeating assistant text
    if not last_tool_results:
        if last_assistant_text and not _assistant_text_is_repeat(harness, last_assistant_text):
            return True

    return False


def _prior_turn_verdict(harness: Any) -> str:
    state = getattr(harness, "state", None)
    if state is None:
        return ""
    return str(getattr(state, "scratchpad", {}).get("_progress_prior_verdict", "") or "").strip()


def _artifact_read_result_is_new_range(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    artifact_id = _requested_artifact_read_target(args)
    if not artifact_id:
        return False
    candidate_range = _requested_file_read_range(args)
    state = getattr(harness, "state", None)
    if state is None:
        return True
    history = getattr(state, "scratchpad", {}).get("_progress_read_history", [])
    if not isinstance(history, list):
        return True
    for item in reversed(history[-12:]):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        if str(item.get("artifact_id", "")) != artifact_id:
            continue
        prior_range = (_coerce_int_or_none(item.get("start_line")), _coerce_int_or_none(item.get("end_line")))
        if prior_range == candidate_range:
            return False
    return True


def _ssh_file_read_result_is_new_range(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    path = str(args.get("path") or "").strip()
    if not path:
        return False
    candidate_range = _requested_file_read_range(args)
    state = getattr(harness, "state", None)
    if state is None:
        return True
    history = getattr(state, "scratchpad", {}).get("_progress_read_history", [])
    if not isinstance(history, list):
        return True
    for item in reversed(history[-12:]):
        if str(item.get("tool_name", "")) != "ssh_file_read":
            continue
        if str(item.get("path", "")) != path:
            continue
        prior_range = (_coerce_int_or_none(item.get("start_line")), _coerce_int_or_none(item.get("end_line")))
        if prior_range == candidate_range:
            return False
    return True


def _ssh_exec_read_is_new(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not command:
        return False
    normalized_command = re.sub(r"\s+", " ", command.lower())
    state = getattr(harness, "state", None)
    if state is None:
        return True
    history = getattr(state, "scratchpad", {}).get("_progress_read_history", [])
    if not isinstance(history, list):
        return True
    for item in reversed(history[-12:]):
        if str(item.get("tool_name", "")) != "ssh_exec":
            continue
        prior_command = str(item.get("command", "") or "").strip()
        if re.sub(r"\s+", " ", prior_command.lower()) == normalized_command:
            return False
    return True


def _record_progress_read(harness: Any, record: Any) -> None:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return
    history = scratchpad.setdefault("_progress_read_history", [])
    if not isinstance(history, list):
        return
    entry: dict[str, Any] = {"tool_name": record.tool_name}
    if record.tool_name == "artifact_read":
        entry["artifact_id"] = _requested_artifact_read_target(args)
    elif record.tool_name == "ssh_file_read":
        entry["path"] = str(args.get("path") or "").strip()
    elif record.tool_name == "ssh_exec":
        entry["command"] = str(args.get("command") or "").strip()
    start_line, end_line = _requested_file_read_range(args)
    if start_line is not None:
        entry["start_line"] = start_line
    if end_line is not None:
        entry["end_line"] = end_line
    history.append(entry)
    if len(history) > 24:
        del history[: len(history) - 24]


def _ssh_exec_remote_paths(record: Any) -> list[str]:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not command:
        return []
    paths: list[str] = []
    for match in _REMOTE_PATH_RE.finditer(command):
        path = match.group(0)
        if path not in paths:
            paths.append(path)
    return paths[:8]


def _ssh_exec_observation_entries(harness: Any) -> list[dict[str, Any]]:
    state = getattr(harness, "state", None)
    if state is None:
        return []
    scratchpad = getattr(state, "scratchpad", {})
    history = scratchpad.get("_progress_ssh_observation_history", [])
    return history if isinstance(history, list) else []


def _ssh_exec_has_novel_remote_observation(harness: Any, record: Any) -> bool:
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    host = str(args.get("host") or metadata.get("host") or "").strip().lower()
    failure_class = str(metadata.get("ssh_error_class") or metadata.get("failure_kind") or "").strip()
    auth_mode = str(metadata.get("ssh_auth_mode") or "").strip()
    reached_remote_host = (
        bool(getattr(record.result, "success", False))
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    )
    prior_entries = _ssh_exec_observation_entries(harness)
    if not host:
        return False

    if failure_class:
        if not any(
            str(item.get("host") or "").strip().lower() == host
            and str(item.get("failure_class") or "").strip() == failure_class
            for item in prior_entries
            if isinstance(item, dict)
        ):
            return True

    for path in _ssh_exec_remote_paths(record):
        if not any(
            str(item.get("host") or "").strip().lower() == host
            and path in (item.get("paths") or [])
            for item in prior_entries
            if isinstance(item, dict)
        ):
            return True

    if reached_remote_host and auth_mode:
        if not any(
            str(item.get("host") or "").strip().lower() == host
            and bool(item.get("reached_remote_host"))
            and str(item.get("auth_mode") or "").strip() == auth_mode
            for item in prior_entries
            if isinstance(item, dict)
        ):
            return True

    return False


def _record_ssh_exec_observation(harness: Any, record: Any) -> None:
    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return
    history = scratchpad.setdefault("_progress_ssh_observation_history", [])
    if not isinstance(history, list):
        return
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    history.append(
        {
            "host": str(args.get("host") or metadata.get("host") or "").strip().lower(),
            "failure_class": str(metadata.get("ssh_error_class") or metadata.get("failure_kind") or "").strip(),
            "paths": _ssh_exec_remote_paths(record),
            "auth_mode": str(metadata.get("ssh_auth_mode") or "").strip(),
            "reached_remote_host": (
                bool(getattr(record.result, "success", False))
                or bool(metadata.get("ssh_transport_succeeded"))
                or str(metadata.get("failure_kind") or "").strip() == "remote_command"
            ),
        }
    )
    if len(history) > 32:
        del history[: len(history) - 32]


def _artifact_read_is_past_eof(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    artifact_id = _requested_artifact_read_target(args)
    if not artifact_id:
        return False
    start_line, _end_line = _requested_file_read_range(args)
    if start_line is None or start_line <= 0:
        return False
    state = getattr(harness, "state", None)
    if state is None:
        return False
    artifacts = getattr(state, "artifacts", {})
    if not isinstance(artifacts, dict):
        return False
    artifact = artifacts.get(artifact_id)
    if artifact is None:
        return False
    total_lines = None
    metadata = getattr(artifact, "metadata", {})
    if isinstance(metadata, dict):
        raw_total = metadata.get("total_lines") or metadata.get("artifact_total_lines")
        total_lines = _coerce_int_or_none(raw_total)
    if total_lines is None:
        content_path = str(getattr(artifact, "content_path", "") or "").strip()
        if content_path:
            try:
                from pathlib import Path
                total_lines = len(Path(content_path).read_text(encoding="utf-8").splitlines())
            except OSError:
                total_lines = None
    if total_lines is not None and start_line > total_lines:
        return True
    return False


def _ssh_file_read_is_past_eof(harness: Any, record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    start_line, _end_line = _requested_file_read_range(args)
    if start_line is None or start_line <= 0:
        return False
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    total_lines = _coerce_int_or_none(metadata.get("total_lines"))
    if total_lines is not None and start_line > total_lines:
        return True
    return False


def _extract_args_from_fingerprint(fingerprint: str) -> dict[str, Any] | None:
    if not fingerprint:
        return None
    try:
        import json

        payload = json.loads(fingerprint)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    args = payload.get("args", {})
    return args if isinstance(args, dict) else None


def _plan_step_changed(harness: Any) -> bool:
    state = getattr(harness, "state", None)
    if state is None:
        return False
    plan = getattr(state, "active_plan", None)
    if plan is None:
        return False
    current_step = ""
    try:
        current_step = str(plan.current_step_index or plan.current_step or "").strip()
    except Exception:
        pass
    prior_step = str(getattr(state, "scratchpad", {}).get("_progress_prior_plan_step", "") or "").strip()
    return current_step != "" and current_step != prior_step


def _assistant_text_is_repeat(harness: Any, text: str) -> bool:
    state = getattr(harness, "state", None)
    if state is None:
        return False
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not normalized:
        return False
    prior_texts: list[str] = []
    for message in reversed(getattr(state, "recent_messages", [])):
        if getattr(message, "role", "") != "assistant":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        prior_normalized = re.sub(r"\s+", " ", content.lower())
        if prior_normalized == normalized:
            return True
        prior_texts.append(prior_normalized)
        if len(prior_texts) >= 2:
            break
    return False


def _update_progress_tracking(harness: Any, graph_state: Any) -> None:
    """Evaluate this turn and update the no-actionable-progress counter."""
    state = getattr(harness, "state", None)
    if state is None:
        return

    counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
    is_progress = _turn_has_actionable_progress(harness, graph_state)

    scratchpad = getattr(state, "scratchpad", {})
    if is_progress:
        counters["no_actionable_progress"] = 0
        # Update prior-state snapshots for next comparison
        if isinstance(scratchpad, dict):
            # Save current verdict from the most recent verifier result
            verdict: str = ""
            for record in getattr(graph_state, "last_tool_results", []) or []:
                if record.tool_name in {"shell_exec", "ssh_exec"}:
                    meta = record.result.metadata or {}
                    v = str(meta.get("verdict") or meta.get("status") or "").strip()
                    if v:
                        verdict = v
            if not verdict:
                current_verifier = getattr(state, "current_verifier_verdict", None)
                verifier = current_verifier() if callable(current_verifier) else None
                if isinstance(verifier, dict):
                    verdict = str(verifier.get("verdict") or verifier.get("status") or "").strip()
            scratchpad["_progress_prior_verdict"] = verdict
            # Save current plan step
            plan = getattr(state, "active_plan", None)
            if plan is not None:
                try:
                    scratchpad["_progress_prior_plan_step"] = str(
                        plan.current_step_index or plan.current_step or ""
                    ).strip()
                except Exception:
                    pass
    else:
        counters["no_actionable_progress"] = int(counters.get("no_actionable_progress", 0)) + 1

    # Record successful reads for next-turn range comparison
    for record in getattr(graph_state, "last_tool_results", []) or []:
        if record.tool_name in {"artifact_read", "ssh_file_read"} and record.result.success:
            _record_progress_read(harness, record)
        if record.tool_name == "ssh_exec" and record.result.success and _is_ssh_exec_read_command(record):
            _record_progress_read(harness, record)
        if record.tool_name == "ssh_exec":
            _record_ssh_exec_observation(harness, record)

    state.stagnation_counters = counters


def _build_progress_stagnation_nudge(harness: Any) -> str:
    goal = str(
        getattr(getattr(getattr(harness, "state", None), "run_brief", None), "original_task", "")
        or ""
    ).strip()
    goal_note = f" for `{goal}`" if goal else ""
    return (
        "You have made no actionable progress in the last few turns. "
        f"Use the evidence already visible in context{goal_note}. "
        "Perform the next concrete mutation, run a focused verifier, or call "
        "`task_complete(message='...')` if the task is finished. "
        "Do not repeat the same analysis or read operations."
    )


def _stagnation_thresholds_for_phase(harness: Any) -> tuple[int, int]:
    """Return (nudge_start, trip_threshold) based on current phase.

    Research-heavy phases get higher thresholds to avoid false-positives
    during legitimate diagnostic or exploratory work.
    """
    state = getattr(harness, "state", None)
    if state is None:
        return 3, 5
    phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    if phase in {"explore", "repair"}:
        return 5, 7
    return 3, 5


def _check_progress_stagnation(harness: Any, graph_state: Any) -> str | None:
    """Check no-actionable-progress cycles and inject nudge or return guard error.

    Returns a guard error string if the stagnation limit has been reached,
    or None (after optionally injecting a nudge message into state).
    """
    state = getattr(harness, "state", None)
    if state is None:
        return None

    counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
    cycle_count = int(counters.get("no_actionable_progress", 0))
    nudge_start, trip_threshold = _stagnation_thresholds_for_phase(harness)

    if cycle_count < nudge_start:
        return None

    if nudge_start <= cycle_count < trip_threshold:
        # Inject recovery nudge and continue
        state.append_message(
            ConversationMessage(
                role="user",
                content=_build_progress_stagnation_nudge(harness),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "no_actionable_progress",
                    "cycle_count": cycle_count,
                },
            )
        )
        harness._runlog(
            "progress_stagnation_nudge",
            f"injected recovery nudge after {cycle_count} no-progress cycles",
            cycle_count=cycle_count,
        )
        return None

    # cycle_count >= trip_threshold -> trip guard
    return (
        f"Progress stagnation guard tripped: no actionable progress made in {cycle_count} steps. "
        "The model is repeating analysis or read-only operations without moving the task forward."
    )


_COMPLETION_CONFABULATION_PATTERNS = [
    re.compile(r"already\s+(performed|completed|done|finished)", re.IGNORECASE),
    re.compile(r"was\s+already\s+(performed|completed|done|finished)", re.IGNORECASE),
    re.compile(r"previous\s+task_complete", re.IGNORECASE),
    re.compile(r"prior\s+task_complete", re.IGNORECASE),
    re.compile(r"redesign\s+was\s+already", re.IGNORECASE),
    re.compile(r"successful\s+redesign\s+was", re.IGNORECASE),
    re.compile(r"task\s+is\s+already\s+complete", re.IGNORECASE),
    re.compile(r"work\s+is\s+already\s+done", re.IGNORECASE),
    re.compile(r"already\s+succeeded\s+in", re.IGNORECASE),
]


def _check_completion_confabulation(harness: Any, graph_state: Any) -> str | None:
    """Detect if the model falsely believes work was already completed in this task.

    Returns a guard error string if confabulation is detected, or None (after
    optionally injecting a recovery nudge into state).
    """
    state = getattr(harness, "state", None)
    if state is None:
        return None

    # If mutations have actually occurred, there is nothing to confabulate.
    if state.files_changed_this_cycle:
        return None

    for entry in state.tool_history:
        if not isinstance(entry, str):
            continue
        parts = entry.split("|")
        if len(parts) >= 3 and parts[-1] == "success" and parts[0] in _MUTATION_TOOLS:
            return None

    text_to_check = " ".join(
        [
            str(getattr(graph_state, "last_assistant_text", "") or ""),
            str(getattr(graph_state, "last_thinking_text", "") or ""),
        ]
    )
    if not text_to_check.strip():
        return None

    for pattern in _COMPLETION_CONFABULATION_PATTERNS:
        if pattern.search(text_to_check):
            break
    else:
        return None

    scratchpad = state.scratchpad if isinstance(getattr(state, "scratchpad", None), dict) else {}
    if scratchpad.get("_confabulation_nudged"):
        # Already nudged once for this task; don't spam.
        return None

    scratchpad["_confabulation_nudged"] = True
    state.scratchpad = scratchpad

    state.append_message(
        ConversationMessage(
            role="user",
            content=(
                "GROUND TRUTH CHECK: No mutating operations have been performed in this task. "
                "Do not assume any work was already completed. Start implementation now using the "
                "evidence already in context."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "completion_confabulation",
            },
        )
    )
    harness._runlog(
        "completion_confabulation_nudge",
        "injected confabulation recovery nudge",
    )
    return None
