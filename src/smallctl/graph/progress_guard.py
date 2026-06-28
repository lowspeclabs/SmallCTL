from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from ..models.conversation import ConversationMessage
from ..remote_scope import remote_scope_is_active
from ..shell_utils import is_read_only_shell_evidence_action as _is_read_only_shell_evidence_action
from ..state import json_safe_value
from .tool_loop_guard_progress import (
    _coerce_int_or_none,
    _requested_artifact_read_target,
    _requested_file_read_range,
    _tool_attempt_history,
)
from .progress_guard_constants import (
    _COMPLETION_CONFABULATION_PATTERNS,
    _DETERMINISTIC_READ_FAILURES_KEY,
    _FAILED_MUTATION_REPAIR_PROGRESS_BUDGET,
    _FAILED_MUTATION_REPAIR_PROGRESS_KEY,
    _LAST_FAILED_VERIFIER_KEY,
    _MUTATION_TASK_READ_BUDGET_PER_FILE,
    _MUTATION_TASK_READ_ONLY_LOOP_THRESHOLD,
    _MUTATION_TOOLS,
    _PATCH_META_TOOLS,
    _PATCH_TARGET_NOT_FOUND_COUNTS_KEY,
    _PATCH_TARGET_NOT_FOUND_SUPPRESS_AFTER,
    _READ_ONLY_LOOP_TOOLS,
    _READ_TOOLS,
    _STALE_VERIFIER_KEY,
)
from .progress_guard_coverage import (
    artifact_coverage_entry as _artifact_coverage_entry,
    artifact_coverage_map as _artifact_coverage_map,
    artifact_read_effective_span as _artifact_read_effective_span,
    artifact_read_is_continuation_page as _artifact_read_is_continuation_page,
    artifact_read_is_past_eof as _artifact_read_is_past_eof,
    artifact_read_result_is_new_range as _artifact_read_result_is_new_range,
    coverage_is_complete as _coverage_is_complete,
    file_read_result_is_new_range as _file_read_result_is_new_range,
    next_unread_artifact_line as _next_unread_artifact_line,
    normalize_line_ranges as _normalize_line_ranges,
    record_artifact_read_coverage as _record_artifact_read_coverage,
    record_progress_read as _record_progress_read,
    span_adds_unseen_lines as _span_adds_unseen_lines,
    ssh_exec_read_is_new as _ssh_exec_read_is_new,
    ssh_file_read_is_past_eof as _ssh_file_read_is_past_eof,
    ssh_file_read_result_is_new_range as _ssh_file_read_result_is_new_range,
)
from .progress_guard_ssh import (
    record_ssh_exec_observation as _record_ssh_exec_observation,
    ssh_exec_has_novel_partial_output as _ssh_exec_has_novel_partial_output,
    ssh_exec_has_novel_remote_observation as _ssh_exec_has_novel_remote_observation,
    ssh_exec_observation_entries as _ssh_exec_observation_entries,
    ssh_exec_output_fingerprint as _ssh_exec_output_fingerprint,
    ssh_exec_remote_paths as _ssh_exec_remote_paths,
)
from .verifier_utils import (
    _command_looks_like_verifier,
    _summarize_verifier_failure,
    _verifier_output_text,
)
from .progress_guard_support import (
    _build_progress_stagnation_nudge,
    _current_verifier_payload,
    _current_task_requires_file_mutation,
    _latest_blocker_note,
    _latest_complete_file_read_note,
    _latest_failed_verifier_note,
    _last_stalled_action,
    _prior_turn_verdict,
    _read_repeats_fully_read_target_after_failed_verifier,
    _record_verifier_verdict,
    _stagnation_thresholds_for_phase,
)

def _is_shell_read_command(record: Any) -> bool:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not command:
        return False
    return _is_read_only_shell_evidence_action(command)


def _turn_is_read_only(last_tool_results: list[Any]) -> bool:
    """Return True if every tool result this turn is a read/search operation."""
    if not last_tool_results:
        return False
    saw_read_tool = False
    for record in last_tool_results:
        tool_name = str(getattr(record, "tool_name", "") or "")
        if tool_name in _MUTATION_TOOLS or tool_name in {"task_complete", "task_fail"}:
            return False
        if tool_name in {"shell_exec", "ssh_exec"}:
            if not _is_shell_read_command(record):
                return False
        if tool_name in _READ_ONLY_LOOP_TOOLS:
            saw_read_tool = True
    return saw_read_tool


def _record_deterministic_read_failure(harness: Any, record: Any) -> None:
    if getattr(record, "tool_name", "") != "ssh_file_read":
        return
    result = getattr(record, "result", None)
    if result is None or getattr(result, "success", False):
        return
    error = str(getattr(result, "error", "") or "").strip().lower()
    if "not found" not in error and "no such file" not in error:
        return
    args = getattr(record, "args", None)
    if not isinstance(args, dict):
        args = {}
    key = {
        "tool_name": "ssh_file_read",
        "host": str(args.get("host") or "").strip(),
        "user": str(args.get("user") or "").strip(),
        "path": str(args.get("path") or "").strip(),
    }
    if not key["path"]:
        return
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    failures = scratchpad.get(_DETERMINISTIC_READ_FAILURES_KEY)
    if not isinstance(failures, list):
        failures = []
    failures.append(key)
    scratchpad[_DETERMINISTIC_READ_FAILURES_KEY] = failures[-16:]


def _mutation_task_read_budget_exhausted(harness: Any, record: Any) -> bool:
    if str(getattr(record, "tool_name", "") or "") != "file_read":
        return False
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    args = getattr(record, "args", None)
    if not isinstance(args, dict):
        args = {}
    path = str(args.get("path") or "").strip()
    if not path:
        return False
    history = scratchpad.get("_progress_read_history")
    if not isinstance(history, list):
        return False
    prior_same_file_reads = 0
    for item in reversed(history):
        if not isinstance(item, dict):
            continue
        tool_name = str(item.get("tool_name") or "")
        if tool_name in _MUTATION_TOOLS or tool_name in {"shell_exec", "ssh_exec"}:
            break
        if tool_name == "file_read" and str(item.get("path") or "").strip() == path:
            prior_same_file_reads += 1
    return prior_same_file_reads >= _MUTATION_TASK_READ_BUDGET_PER_FILE


def _turn_has_actionable_progress(harness: Any, graph_state: Any) -> bool:
    """Return True if the current turn changed actionable state."""
    last_tool_results = getattr(graph_state, "last_tool_results", []) or []
    last_assistant_text = str(getattr(graph_state, "last_assistant_text", "") or "").strip()
    mutation_required = _current_task_requires_file_mutation(getattr(harness, "state", None))

    # 1. Task completion
    for record in last_tool_results:
        if record.tool_name == "task_complete" and record.result.success:
            return True

    # 2. Successful mutation with changed=True
    for record in last_tool_results:
        if record.tool_name in _MUTATION_TOOLS:
            if record.result.success and (record.result.metadata or {}).get("changed") is True:
                return True

    # 2b. A novel failed patch/write attempt can still be actionable repair
    # progress: the model used the right class of tool against a concrete
    # verifier failure and learned that its target text was stale. Keep this
    # bounded so varied-but-wrong patch attempts still trip stagnation.
    for record in last_tool_results:
        if _failed_mutation_attempt_is_repair_progress(harness, record, mutation_required=mutation_required):
            return True

    # 3. Plan step state change
    if _plan_step_changed(harness):
        return True

    # 4. Successful verifier with a new verdict
    for record in last_tool_results:
        if record.tool_name in {"shell_exec", "ssh_exec"}:
            verdict = _record_verifier_verdict(harness, record)
            if verdict:
                prior = _prior_turn_verdict(harness)
                if verdict != prior:
                    return True

    # 4b. Novel remote SSH observations count as progress even when the command failed.
    for record in last_tool_results:
        if record.tool_name == "ssh_exec" and _ssh_exec_has_novel_remote_observation(harness, record):
            return True

    # 4c. Partial remote output on failure/timeout is still useful once.
    for record in last_tool_results:
        if record.tool_name == "ssh_exec" and not record.result.success:
            if _ssh_exec_has_novel_partial_output(harness, record):
                return True

    # 5. Successful read of a new artifact/ssh_file/file range or ssh_exec read command
    for record in last_tool_results:
        if record.tool_name == "artifact_read" and record.result.success:
            if _artifact_read_is_past_eof(harness, record):
                return False
            # If this read is continuing pagination on the same artifact,
            # treat it as progress so the guard doesn't fire mid-read.
            if _artifact_read_is_continuation_page(harness, record):
                return True
            if _artifact_read_result_is_new_range(harness, record):
                return True
        if record.tool_name == "ssh_file_read" and record.result.success:
            if _ssh_file_read_is_past_eof(harness, record):
                return False
            if _ssh_file_read_result_is_new_range(harness, record):
                return True
        if record.tool_name == "file_read" and record.result.success:
            if _read_repeats_fully_read_target_after_failed_verifier(harness, record):
                continue
            if mutation_required and _mutation_task_read_budget_exhausted(harness, record):
                continue
            if _file_read_result_is_new_range(harness, record):
                return True
        if record.tool_name == "ssh_exec" and record.result.success:
            if _is_shell_read_command(record):
                if _ssh_exec_read_is_new(harness, record):
                    return True
            else:
                return True
        if record.tool_name == "shell_exec" and record.result.success:
            if not _is_shell_read_command(record):
                return True

    # 6. Any other successful non-read, non-mutation, non-exec tool
    #    (shell_exec/ssh_exec are handled above; identical calls are caught by loop guards)
    for record in last_tool_results:
        if mutation_required and record.tool_name in _PATCH_META_TOOLS:
            continue
        if record.result.success and record.tool_name not in _READ_TOOLS and record.tool_name not in _MUTATION_TOOLS and record.tool_name not in {"shell_exec", "ssh_exec"}:
            return True

    # 7. No-tool turn with non-repeating assistant text
    if not last_tool_results:
        if last_assistant_text and not _assistant_text_is_repeat(harness, last_assistant_text):
            return True

    return False


def _record_failed_verifier(state: Any, record: Any) -> None:
    if getattr(record, "tool_name", "") not in {"shell_exec", "ssh_exec"}:
        return
    result = getattr(record, "result", None)
    if result is None or getattr(result, "success", False):
        return
    args = getattr(record, "args", {}) if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not command:
        return
    if not _command_looks_like_verifier(command):
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    output_text = _verifier_output_text(record)
    scratchpad[_LAST_FAILED_VERIFIER_KEY] = {
        "tool_name": str(getattr(record, "tool_name", "") or "").strip(),
        "command": command,
        "summary": _summarize_verifier_failure(output_text or str(getattr(result, "error", "") or "")),
        "raw_output": output_text,
    }


def _failed_mutation_attempt_is_repair_progress(
    harness: Any,
    record: Any,
    *,
    mutation_required: bool,
) -> bool:
    if getattr(record, "tool_name", "") not in _MUTATION_TOOLS:
        return False
    result = getattr(record, "result", None)
    if result is None or getattr(result, "success", False):
        return False

    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None) if state is not None else None
    if not isinstance(scratchpad, dict):
        return False
    if not mutation_required and not scratchpad.get(_LAST_FAILED_VERIFIER_KEY):
        return False
    if not _failed_mutation_is_patch_target_mismatch(record):
        return False

    path = _mutation_record_path(record)
    if not path:
        return False
    fingerprint = _failed_mutation_fingerprint(record, path)
    if not fingerprint:
        return False

    history = scratchpad.get(_FAILED_MUTATION_REPAIR_PROGRESS_KEY)
    if not isinstance(history, list):
        history = []
    for item in history:
        if isinstance(item, dict) and str(item.get("fingerprint") or "") == fingerprint:
            return False

    same_target_count = sum(
        1
        for item in history
        if isinstance(item, dict)
        and str(item.get("tool_name") or "") == str(getattr(record, "tool_name", "") or "")
        and str(item.get("path") or "") == path
    )
    if same_target_count >= _FAILED_MUTATION_REPAIR_PROGRESS_BUDGET:
        return False

    history.append(
        {
            "tool_name": str(getattr(record, "tool_name", "") or ""),
            "path": path,
            "fingerprint": fingerprint,
        }
    )
    scratchpad[_FAILED_MUTATION_REPAIR_PROGRESS_KEY] = history[-24:]
    return True


def _maybe_suppress_file_patch_after_target_not_found(state: Any, record: Any) -> None:
    if getattr(record, "tool_name", "") != "file_patch":
        return
    result = getattr(record, "result", None)
    if result is None or getattr(result, "success", False):
        return
    if not _failed_mutation_is_patch_target_mismatch(record):
        return

    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    path = _mutation_record_path(record)
    if not path:
        return
    fingerprint = _failed_mutation_fingerprint(record, path)
    if not fingerprint:
        return

    counts = scratchpad.get(_PATCH_TARGET_NOT_FOUND_COUNTS_KEY)
    if not isinstance(counts, dict):
        counts = {}
    count = int(counts.get(fingerprint, 0) or 0) + 1
    counts[fingerprint] = count
    scratchpad[_PATCH_TARGET_NOT_FOUND_COUNTS_KEY] = dict(list(counts.items())[-24:])

    if count >= _PATCH_TARGET_NOT_FOUND_SUPPRESS_AFTER:
        scratchpad["_repeated_tool_loop_suppressed_tool"] = "file_patch"
        scratchpad["_repeated_tool_loop_suppressed_ttl"] = max(
            int(scratchpad.get("_repeated_tool_loop_suppressed_ttl", 0) or 0),
            2,
        )
        scratchpad["_last_file_patch_suppression"] = {
            "reason": "repeated_patch_target_not_found",
            "path": path,
            "fingerprint": fingerprint,
            "count": count,
        }


def _failed_mutation_is_patch_target_mismatch(record: Any) -> bool:
    result = getattr(record, "result", None)
    metadata = getattr(result, "metadata", {}) if result is not None else {}
    if not isinstance(metadata, dict):
        metadata = {}
    error_kind = str(metadata.get("error_kind") or metadata.get("failure_kind") or "").strip().lower()
    if error_kind in {"patch_target_not_found", "target_not_found", "replace_target_not_found"}:
        return True
    error = str(getattr(result, "error", "") or "").strip().lower()
    return (
        "patch target text was not found" in error
        or "target text was not found" in error
        or "target text not found" in error
    )


def _mutation_record_path(record: Any) -> str:
    args = getattr(record, "args", {}) if isinstance(getattr(record, "args", None), dict) else {}
    result = getattr(record, "result", None)
    metadata = getattr(result, "metadata", {}) if result is not None else {}
    if not isinstance(metadata, dict):
        metadata = {}
    return str(metadata.get("path") or args.get("path") or "").strip()


def _failed_mutation_fingerprint(record: Any, path: str) -> str:
    args = getattr(record, "args", {}) if isinstance(getattr(record, "args", None), dict) else {}
    if not isinstance(args, dict):
        args = {}
    relevant_args = {
        key: args.get(key)
        for key in (
            "target_text",
            "replacement_text",
            "old_text",
            "new_text",
            "before",
            "after",
            "content",
        )
        if key in args
    }
    if not relevant_args:
        relevant_args = dict(args)
        relevant_args.pop("path", None)
    payload = {
        "tool_name": str(getattr(record, "tool_name", "") or ""),
        "path": path,
        "args": json_safe_value(relevant_args),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8", errors="replace")).hexdigest()[:16]
def _mark_verifier_stale_after_mutation(state: Any, record: Any) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    verifier = _current_verifier_payload(state)
    if verifier is None:
        return
    metadata = getattr(getattr(record, "result", None), "metadata", {}) or {}
    args = getattr(record, "args", {}) if isinstance(getattr(record, "args", None), dict) else {}
    paths = []
    for value in (metadata.get("path"), args.get("path")):
        text = str(value or "").strip()
        if text and text not in paths:
            paths.append(text)
    scratchpad[_STALE_VERIFIER_KEY] = {
        "reason": "file_changed_after_verifier",
        "tool_name": str(getattr(record, "tool_name", "") or "").strip(),
        "paths": paths,
        "prior_verdict": json_safe_value(verifier),
    }


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


def _verifier_success_nudge_key(command: str, changed_paths: list[str]) -> str:
    nudge_context = json.dumps(
        {
            "command": re.sub(r"\s+", " ", str(command or "")).strip(),
            "changed_paths": sorted(str(path).strip() for path in changed_paths if str(path).strip()),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"_verifier_nudge_{hashlib.sha256(nudge_context.encode('utf-8')).hexdigest()[:16]}"


def _maybe_inject_verifier_success_nudge(state: Any, graph_state: Any) -> None:
    """Inject a completion nudge when a verifier succeeds on a recently mutated file."""
    if not getattr(state, "files_changed_this_cycle", None):
        return
    changed_paths = [str(p).strip() for p in state.files_changed_this_cycle if str(p).strip()]
    if not changed_paths:
        return
    for record in getattr(graph_state, "last_tool_results", []) or []:
        if record.tool_name not in {"shell_exec", "ssh_exec"}:
            continue
        if not record.result.success:
            continue
        args = record.args if isinstance(getattr(record, "args", None), dict) else {}
        command = str(args.get("command") or "").strip()
        if not command or not _command_looks_like_verifier(command):
            continue
        # Check exit code 0
        output = getattr(getattr(record, "result", None), "output", None)
        exit_code = None
        if isinstance(output, dict):
            exit_code = output.get("exit_code")
        if exit_code is None:
            metadata = getattr(getattr(record, "result", None), "metadata", {}) or {}
            if isinstance(metadata, dict):
                exit_code = metadata.get("exit_code")
        if exit_code != 0:
            continue
        # Check if verifier targets a recently mutated file
        if not any(str(path).strip() in command for path in changed_paths):
            continue
        # Avoid duplicate nudges for the same verifier context across restored sessions.
        scratchpad = getattr(state, "scratchpad", {})
        if not isinstance(scratchpad, dict):
            scratchpad = {}
        nudge_key = _verifier_success_nudge_key(command, changed_paths)
        if scratchpad.get(nudge_key):
            continue
        scratchpad[nudge_key] = True
        state.scratchpad = scratchpad
        state.append_message(
            ConversationMessage(
                role="user",
                content=(
                    "The verification command succeeded with exit code 0. "
                    "If the task requirements are satisfied, call `task_complete(message=...)` now "
                    "instead of performing additional reads."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "verifier_success_completion_prompt",
                    "command": command,
                },
            )
        )
        break


def _update_remote_install_stall_counter(state: Any, graph_state: Any, counters: dict[str, Any]) -> None:
    """Track repeated remote download/install failures to detect install stalls."""
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    if task_mode != "remote_execute":
        return

    remote_install_tools = {"ssh_exec", "http_get", "file_download"}
    for record in getattr(graph_state, "last_tool_results", []) or []:
        tool_name = str(getattr(record, "tool_name", "") or "").strip()
        if tool_name not in remote_install_tools:
            continue
        result = getattr(record, "result", None)
        if result is None or getattr(result, "success", False):
            continue
        args = getattr(record, "args", {}) if isinstance(getattr(record, "args", None), dict) else {}
        host = str(args.get("host") or "").strip().lower()
        command = str(args.get("command") or "").strip().lower()
        url = str(args.get("url") or "").strip().lower()

        # Determine objective category
        category = ""
        if "git" in command or "git" in url:
            category = "git"
        elif "curl" in command or "wget" in command:
            category = "curl"
        elif tool_name in {"http_get", "file_download"} or url:
            category = "download"
        elif command:
            category = "remote_command"

        key = f"remote_install:{host}:{category}"
        prior = counters.get("_remote_install_stall_history", [])
        if not isinstance(prior, list):
            prior = []
        prior.append(key)
        prior = prior[-5:]
        counters["_remote_install_stall_history"] = prior
        count = sum(1 for item in prior if item == key)
        counters["remote_install_stall"] = max(int(counters.get("remote_install_stall", 0) or 0), count)
        if count >= 2:
            break


def _update_progress_tracking(harness: Any, graph_state: Any) -> None:
    """Evaluate this turn and update the no-actionable-progress counter."""
    state = getattr(harness, "state", None)
    if state is None:
        return

    counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
    is_progress = _turn_has_actionable_progress(harness, graph_state)

    scratchpad = getattr(state, "scratchpad", {})
    mutation_required = _current_task_requires_file_mutation(state)
    read_only_turn = _turn_is_read_only(getattr(graph_state, "last_tool_results", []) or [])
    if mutation_required and read_only_turn:
        counters["consecutive_read_only_turns"] = int(counters.get("consecutive_read_only_turns", 0)) + 1
    else:
        counters["consecutive_read_only_turns"] = 0

    for record in getattr(graph_state, "last_tool_results", []) or []:
        if record.tool_name in _MUTATION_TOOLS and record.result.success and (record.result.metadata or {}).get("changed") is True:
            _mark_verifier_stale_after_mutation(state, record)
            if isinstance(scratchpad, dict) and scratchpad.get("_read_only_loop_gate_active"):
                scratchpad.pop("_read_only_loop_gate_active", None)
                scratchpad.pop("_read_only_loop_gate_triggered_at", None)
                scratchpad.pop("_read_only_loop_gate_nudged", None)
                harness._runlog(
                    "read_only_loop_gate_cleared",
                    "cleared read-only loop hard gate after successful mutation",
                )
        _record_failed_verifier(state, record)
        _maybe_suppress_file_patch_after_target_not_found(state, record)
    _maybe_inject_verifier_success_nudge(state, graph_state)
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
                        scratchpad.pop(_STALE_VERIFIER_KEY, None)
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

    # Activate the read-only loop hard gate for mutation-required tasks that have
    # done too many consecutive read/search turns, even if individual reads were
    # novel enough to count as actionable progress.
    if (
        mutation_required
        and int(counters.get("consecutive_read_only_turns", 0)) >= _MUTATION_TASK_READ_ONLY_LOOP_THRESHOLD
        and isinstance(scratchpad, dict)
        and not scratchpad.get("_read_only_loop_gate_active")
    ):
        scratchpad["_read_only_loop_gate_active"] = True
        scratchpad["_read_only_loop_gate_triggered_at"] = int(getattr(state, "step_count", 0) or 0)
        harness._runlog(
            "read_only_loop_gate_activated",
            "activated read-only loop hard gate for mutation-required task",
            consecutive_read_only_turns=counters["consecutive_read_only_turns"],
        )

    # Update remote-install stall counter
    _update_remote_install_stall_counter(state, graph_state, counters)

    # Record successful reads for next-turn range comparison
    for record in getattr(graph_state, "last_tool_results", []) or []:
        if record.tool_name in {"artifact_read", "ssh_file_read", "file_read"} and record.result.success:
            _record_progress_read(harness, record)
        _record_deterministic_read_failure(harness, record)
        if record.tool_name == "ssh_exec" and record.result.success and _is_shell_read_command(record):
            _record_progress_read(harness, record)
        if record.tool_name == "ssh_exec":
            _record_ssh_exec_observation(harness, record)

    state.stagnation_counters = counters


def _check_progress_stagnation(harness: Any, graph_state: Any) -> str | None:
    """Check no-actionable-progress cycles and inject nudge or return guard error.

    Returns a guard error string if the stagnation limit has been reached,
    or None (after optionally injecting a nudge message into state).
    """
    state = getattr(harness, "state", None)
    if state is None:
        return None

    counters = state.stagnation_counters if isinstance(getattr(state, "stagnation_counters", None), dict) else {}
    consecutive_read_only = int(counters.get("consecutive_read_only_turns", 0))
    if (
        consecutive_read_only >= _MUTATION_TASK_READ_ONLY_LOOP_THRESHOLD
        and _current_task_requires_file_mutation(state)
    ):
        return (
            f"Read-only loop guard tripped: {consecutive_read_only} consecutive read/search turns "
            "on a task that requires a file mutation. Stop reading and apply a concrete patch or write."
        )

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


def _check_completion_confabulation(harness: Any, graph_state: Any) -> str | None:
    """Detect if the model falsely believes work was already completed in this task.

    Returns a guard error string if confabulation is detected, or None (after
    optionally injecting a recovery nudge into state).
    """
    state = getattr(harness, "state", None)
    if state is None:
        return None

    # Read-only / research / analysis tasks do not require mutations. Treating
    # a completed answer as a confabulation pushes the model into a useless
    # loop, so skip the nudge entirely for those intents.
    active_intent = str(getattr(state, "active_intent", "") or "").strip().lower()
    task_mode = str(getattr(state, "task_mode", "") or "").strip().lower()
    if active_intent == "readonly_lookup" or task_mode in {"analysis", "debug_inspect", "chat"}:
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
