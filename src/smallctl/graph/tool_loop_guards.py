from __future__ import annotations

import json
import math
import re
from typing import Any

from ..docker_retry_normalization import docker_reload_target, docker_retry_family
from ..repeat_loop_policy import strict_identical_limit, strict_window_limit
from ..state import json_safe_value
from .state import PendingToolCall
from ..harness.task_transactions import recovery_context_lines, transaction_from_scratchpad
from .tool_loop_guard_constants import (
    _DETERMINISTIC_READ_FAILURES_KEY,
    _DIR_LIST_IDENTICAL_TOOL_CALL_STREAK_LIMIT,
    _DIR_LIST_REPEATED_TOOL_UNIQUE_LIMIT,
    _DIR_LIST_REPEATED_TOOL_WINDOW,
    _EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES,
    _IDENTICAL_TOOL_CALL_STREAK_LIMIT,
    _INCOMPLETE_TOOL_CALL_SCRATCHPAD_KEY,
    _PLACEHOLDER_ARG_KEY_TOKENS,
    _PLACEHOLDER_ARG_VALUE_TOKENS,
    _PLACEHOLDER_TOOL_NAME_TOKENS,
    _REPEATED_TOOL_HISTORY_LIMIT,
    _REPEATED_TOOL_UNIQUE_LIMIT,
    _REPEATED_TOOL_WINDOW,
    _STRICT_LOOP_GUARD_IDENTICAL_LIMIT,
    _STRICT_LOOP_GUARD_TOOLS,
    _STRICT_LOOP_GUARD_UNIQUE_LIMIT,
    _STRICT_LOOP_GUARD_WINDOW_LIMIT,
)
from .tool_loop_guard_progress import (
    _artifact_read_line_progress_is_progress,
    _dir_list_exploration_progress_is_progress,
    _dir_list_repeat_has_intervening_progress,
    _dir_list_same_path_repeat_is_loop,
    _extract_args_from_fingerprint,
    _extract_path_from_fingerprint,
    _file_read_line_progress_is_progress,
    _is_strict_subpath,
    _model_is_exact_small_gemma_4_it,
    _requested_file_read_range,
    _resolve_dir_list_path,
    _resolve_file_read_path,
    _tool_attempt_history,
)
from .tool_loop_guard_read_predicates import (
    _artifact_read_past_eof_is_loop,
    _artifact_read_target_path_re_read_is_loop,
    _artifact_read_targets_mutation_result_loop,
    _ssh_file_read_after_remote_mutation_is_progress,
)
from .progress_guard_ssh import ssh_exec_read_targets
from .tool_loop_guards_support import (
    _directive_hint_for_repeated_tool,
    _normalize_token,
    _placeholder_token,
    _placeholder_value_looks_generic,
    _tool_call_fingerprint,
    _semantic_tool_call_fingerprint,
    _cwd_for_fingerprint,
    _normalize_tool_args,
    _normalize_shell_command,
    _normalize_path_token,
    _normalize_local_path_for_fingerprint,
    _normalize_json_like,
)


def _timeout_recovered_incomplete_tool_call_context(harness: Any, pending: PendingToolCall) -> dict[str, Any] | None:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return None
    payload = scratchpad.get(_INCOMPLETE_TOOL_CALL_SCRATCHPAD_KEY)
    if not isinstance(payload, dict):
        return None
    details = payload.get("details")
    if not isinstance(details, dict):
        return None
    if str(details.get("reason") or "").strip() != "tool_call_continuation_timeout":
        return None
    diagnostics = payload.get("tool_call_diagnostics")
    if not isinstance(diagnostics, list):
        diagnostics = []
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        if str(item.get("tool_name") or "").strip() != pending.tool_name:
            continue
        merged = dict(item)
        merged["reason"] = "tool_call_continuation_timeout"
        merged["provider_profile"] = str(details.get("provider_profile") or "").strip()
        merged["timeout_message"] = str(details.get("message") or "").strip()
        merged["attempt"] = details.get("attempt")
        return merged
    return None


def _detect_timeout_recovered_incomplete_tool_call(
    harness: Any,
    pending: PendingToolCall,
) -> tuple[str, dict[str, Any]] | None:
    context = _timeout_recovered_incomplete_tool_call_context(harness, pending)
    if context is None:
        return None

    present_fields = [str(field).strip() for field in context.get("present_fields", []) if str(field).strip()]
    missing_fields = [str(field).strip() for field in context.get("missing_required_fields", []) if str(field).strip()]
    provider_profile = str(context.get("provider_profile") or "provider").strip() or "provider"
    timeout_message = str(context.get("timeout_message") or "").strip()
    present_text = ", ".join(present_fields) if present_fields else "none"
    missing_text = ", ".join(missing_fields) if missing_fields else "unknown"
    message = (
        f"Recovered partial `{pending.tool_name}` tool call after {provider_profile} timed out waiting for tool-call continuation. "
        f"Present fields: {present_text}. Missing required fields: {missing_text}. "
        "Regenerate the full tool call from scratch instead of treating this as a hallucinated call."
    )
    if timeout_message:
        message = f"{message} Upstream detail: {timeout_message}."
    details = {
        "tool_name": pending.tool_name,
        "tool_call_id": pending.tool_call_id,
        "required_fields": list(context.get("required_fields", [])),
        "present_fields": present_fields,
        "missing_required_fields": missing_fields,
        "reason": "tool_call_continuation_timeout",
        "provider_profile": provider_profile,
        "timeout_message": timeout_message,
        "attempt": context.get("attempt"),
        "arguments": json_safe_value(context.get("arguments") or {}),
        "raw_arguments_preview": str(context.get("raw_arguments_preview") or ""),
        "suppress_retrieval_persistence": True,
    }
    return message, details


def _detect_hallucinated_tool_call(harness: Any, pending: PendingToolCall) -> str | None:
    if _timeout_recovered_incomplete_tool_call_context(harness, pending) is not None:
        return None
    registry = getattr(harness, "registry", None)
    if registry is None:
        return None
    get_meta = getattr(registry, "get", None)
    if not callable(get_meta):
        return None
    meta = get_meta(pending.tool_name)
    if not meta:
        return None

    schema = meta.schema or {}
    required = schema.get("required", [])
    if not required:
        return None

    if not pending.args:
        return (
            f"Hallucination Warning: Tool '{pending.tool_name}' requires specific parameters "
            f"({', '.join(required)}) but was called with none. Please provide the missing arguments."
        )

    return None


def _repeat_loop_limits(harness: Any, pending: PendingToolCall) -> tuple[int, int, int]:
    del harness
    identical_limit = _IDENTICAL_TOOL_CALL_STREAK_LIMIT
    window_limit = _REPEATED_TOOL_WINDOW
    unique_limit = _REPEATED_TOOL_UNIQUE_LIMIT
    if pending.tool_name == "artifact_print":
        return (2, 4, 2)
    if pending.tool_name == "dir_list":
        return (
            _DIR_LIST_IDENTICAL_TOOL_CALL_STREAK_LIMIT,
            _DIR_LIST_REPEATED_TOOL_WINDOW,
            _DIR_LIST_REPEATED_TOOL_UNIQUE_LIMIT,
        )
    if pending.tool_name == "file_read":
        return (
            strict_identical_limit(pending.tool_name, _STRICT_LOOP_GUARD_IDENTICAL_LIMIT),
            strict_window_limit(pending.tool_name, _STRICT_LOOP_GUARD_WINDOW_LIMIT),
            _STRICT_LOOP_GUARD_UNIQUE_LIMIT,
        )
    if pending.tool_name in _STRICT_LOOP_GUARD_TOOLS:
        return (
            _STRICT_LOOP_GUARD_IDENTICAL_LIMIT,
            _STRICT_LOOP_GUARD_WINDOW_LIMIT,
            _STRICT_LOOP_GUARD_UNIQUE_LIMIT,
        )
    return (identical_limit, window_limit, unique_limit)


def _format_repeated_tool_loop_message(harness: Any, pending: PendingToolCall, message: str) -> str:
    hint = _directive_hint_for_repeated_tool(harness, pending)
    parts = [message]
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    transaction = transaction_from_scratchpad(scratchpad if isinstance(scratchpad, dict) else {})
    tx_lines = recovery_context_lines(transaction)
    if tx_lines:
        parts.append(" ".join(tx_lines))
    tool_name = str(getattr(pending, "tool_name", "") or "").strip()
    if tool_name:
        parts.append(f"Last repeated action: `{tool_name}`.")
    if hint:
        parts.append(hint)
    parts.append(
        "Choose exactly one: A. Explain the blocker and stop. B. Try a different specific fix. C. Ask for missing information."
    )
    return " ".join(parts)


def _detect_placeholder_tool_call(harness: Any, pending: PendingToolCall) -> tuple[str, dict[str, Any]] | None:
    tool_name = str(getattr(pending, "tool_name", "") or "").strip()
    if not tool_name:
        return None

    registry = getattr(harness, "registry", None)
    if registry is not None:
        try:
            if tool_name in set(registry.names()):
                return None
        except Exception:
            pass

    if _placeholder_token(tool_name) not in _PLACEHOLDER_TOOL_NAME_TOKENS:
        return None

    args = dict(getattr(pending, "args", {}) or {})
    if args:
        placeholder_keys = all(_placeholder_token(key) in _PLACEHOLDER_ARG_KEY_TOKENS for key in args)
        placeholder_values = all(_placeholder_value_looks_generic(value) for value in args.values())
        if not (placeholder_keys and placeholder_values):
            return None

    message = (
        "Placeholder tool schema detected. You emitted the literal tool name "
        f"`{tool_name}` with example arguments instead of a real tool call. "
        "Regenerate the full JSON tool call from scratch using an actual registered tool name "
        "and concrete arguments. Do not send schema examples or placeholder fields like `arg: value`."
    )
    return message, {
        "tool_name": tool_name,
        "tool_call_id": pending.tool_call_id,
        "reason": "placeholder_tool_schema",
        "offending_field": "tool_name",
        "placeholder_arguments": json_safe_value(args),
    }


def _detect_command_placeholder(harness: Any, pending: PendingToolCall) -> tuple[str, dict[str, Any]] | None:
    """Reject shell/SSH commands that still contain angle-bracket placeholders.

    Small models sometimes emit a command like
    ``docker run ... <original_docker_run_command>`` where the placeholder is
    meant to be replaced by the user. Executing it produces a cryptic shell
    syntax error, so treat it as a missing-argument validation failure instead.
    """
    tool_name = str(getattr(pending, "tool_name", "") or "").strip()
    if tool_name not in {"shell_exec", "ssh_exec"}:
        return None
    args = dict(getattr(pending, "args", {}) or {})
    command = str(args.get("command", "") or "").strip()
    if not command:
        return None
    # Match a single-word placeholder inside angle brackets, e.g.
    # <original_docker_run_command>, <path>, <container_name>.
    match = re.search(r"<[A-Za-z_][A-Za-z0-9_-]*>", command)
    if not match:
        return None
    placeholder = match.group(0)
    message = (
        f"The `{tool_name}` command still contains a placeholder: `{placeholder}`. "
        "Replace it with the actual value before running the command."
    )
    return message, {
        "tool_name": tool_name,
        "tool_call_id": pending.tool_call_id,
        "reason": "command_contains_placeholder",
        "offending_field": "command",
        "placeholder": placeholder,
        "command": command,
    }


def _detect_unknown_tool_call(harness: Any, pending: PendingToolCall) -> tuple[str, dict[str, Any]] | None:
    tool_name = str(getattr(pending, "tool_name", "") or "").strip()
    if not tool_name:
        return None

    registry = getattr(harness, "registry", None)
    if registry is None:
        return None

    names_fn = getattr(registry, "names", None)
    if not callable(names_fn):
        return None

    try:
        available_tools = set(names_fn())
    except Exception:
        return None

    if tool_name in available_tools:
        return None

    from .nodes import HALLUCINATION_MAP

    if tool_name in HALLUCINATION_MAP:
        mapped_tool = HALLUCINATION_MAP[tool_name]
        args = dict(getattr(pending, "args", {}) or {})
        raw_id = args.get("path") or args.get("artifact_id") or args.get("pattern") or "A000X"
        artifact_id = str(raw_id).split("/")[-1]
        if not artifact_id.startswith("A") and "A" in artifact_id:
            idx = artifact_id.find("A")
            artifact_id = artifact_id[idx:]

        hint = f"Tool '{tool_name}' is unavailable. Use '{mapped_tool}(artifact_id=\"{artifact_id}\")' instead."
        return hint, {
            "tool_name": tool_name,
            "tool_call_id": pending.tool_call_id,
            "reason": "hallucinated_tool_mapped",
            "offending_field": "tool_name",
            "mapped_tool": mapped_tool,
        }

    available_list = ", ".join(sorted(available_tools))
    hint = f"Unknown tool '{tool_name}'. Available tools: {available_list}"
    return hint, {
        "tool_name": tool_name,
        "tool_call_id": pending.tool_call_id,
        "reason": "unknown_tool",
        "offending_field": "tool_name",
    }


def _record_tool_attempt(harness: Any, pending: PendingToolCall) -> None:
    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    history = scratchpad.setdefault("_tool_attempt_history", [])
    if not isinstance(history, list):
        return
    cwd = _cwd_for_fingerprint(harness)
    semantic_fingerprint = _semantic_tool_call_fingerprint(pending.tool_name, pending.args, cwd=cwd)
    history.append(
        {
            "tool_name": pending.tool_name,
            "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args, cwd=cwd),
            "semantic_fingerprint": semantic_fingerprint,
        }
    )
    if len(history) > _REPEATED_TOOL_HISTORY_LIMIT:
        del history[: len(history) - _REPEATED_TOOL_HISTORY_LIMIT]


def _clear_tool_attempt_history(harness: Any) -> None:
    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    if isinstance(scratchpad, dict):
        scratchpad.pop("_tool_attempt_history", None)


def _ssh_file_read_repeats_deterministic_not_found(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "ssh_file_read":
        return False
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    failures = scratchpad.get(_DETERMINISTIC_READ_FAILURES_KEY)
    if not isinstance(failures, list):
        return False
    path = str(pending.args.get("path") or "").strip()
    host = str(pending.args.get("host") or "").strip()
    user = str(pending.args.get("user") or "").strip()
    if not path:
        return False
    for item in reversed(failures[-8:]):
        if not isinstance(item, dict):
            continue
        if (
            str(item.get("tool_name") or "") == "ssh_file_read"
            and str(item.get("path") or "").strip() == path
            and str(item.get("host") or "").strip() == host
            and str(item.get("user") or "").strip() == user
        ):
            return True
    return False


def _repair_phase_mutation_tool_repeat_is_loop(harness: Any, pending: PendingToolCall) -> bool:
    phase = str(getattr(getattr(harness, "state", None), "current_phase", "") or "").strip().lower()
    if phase != "repair":
        return False
    tool_name = str(getattr(pending, "tool_name", "") or "").strip()
    if tool_name not in {
        "ssh_file_replace_between",
        "ssh_file_patch",
        "ssh_file_write",
        "file_patch",
        "ast_patch",
        "file_write",
    }:
        return False
    history = _tool_attempt_history(harness)
    if len(history) < 1:
        return False
    cwd = _cwd_for_fingerprint(harness)
    candidate = {
        "tool_name": pending.tool_name,
        "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args, cwd=cwd),
        "semantic_fingerprint": _semantic_tool_call_fingerprint(pending.tool_name, pending.args, cwd=cwd),
    }
    last = history[-1]
    return (
        str(last.get("tool_name", "")) == candidate["tool_name"]
        and _history_fingerprint(last) == candidate["semantic_fingerprint"]
    )


def _history_fingerprint(item: dict[str, Any]) -> str:
    return str(item.get("semantic_fingerprint") or item.get("fingerprint") or "")


def _file_read_immediate_full_repeat_is_loop(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "file_read":
        return False
    if _requested_file_read_range(pending.args) != (None, None):
        return False
    history = _tool_attempt_history(harness)
    if not history:
        return False

    cwd = _cwd_for_fingerprint(harness)
    candidate_fingerprint = _semantic_tool_call_fingerprint("file_read", pending.args, cwd=cwd)
    progress_tools = {
        "file_write",
        "file_append",
        "file_patch",
        "ast_patch",
        "file_delete",
        "shell_exec",
        "ssh_exec",
    }
    for item in reversed(history[-6:]):
        tool_name = str(item.get("tool_name", ""))
        if tool_name in progress_tools:
            return False
        if tool_name != "file_read":
            continue
        args = _extract_args_from_fingerprint(_history_fingerprint(item))
        if isinstance(args, dict) and _requested_file_read_range(args) != (None, None):
            continue
        return _history_fingerprint(item) == candidate_fingerprint
    return False


def _file_read_failed_repeat_is_loop(harness: Any, pending: PendingToolCall) -> bool:
    """Detect repeated file_read calls that previously failed (e.g., path not found)."""
    if pending.tool_name != "file_read":
        return False
    history = _tool_attempt_history(harness)
    if not history:
        return False
    cwd = _cwd_for_fingerprint(harness)
    candidate_fingerprint = _semantic_tool_call_fingerprint("file_read", pending.args, cwd=cwd)
    for item in reversed(history[-6:]):
        tool_name = str(item.get("tool_name", ""))
        if tool_name != "file_read":
            continue
        if _history_fingerprint(item) != candidate_fingerprint:
            continue
        # Check if this prior attempt was recorded as a failure
        if item.get("success") is False:
            return True
    return False


def _detect_repeated_tool_loop(harness: Any, pending: PendingToolCall) -> str | None:
    if pending.tool_name in {"task_complete", "task_fail", "ask_human"}:
        _clear_tool_attempt_history(harness)
        return None
    if pending.tool_name == "shell_exec":
        command = str(pending.args.get("command") or "").strip()
        job_id = str(pending.args.get("job_id") or "").strip()
        if job_id and not command:
            return None
    exhausted_docker_error = _detect_exhausted_docker_registry_family(harness, pending)
    if exhausted_docker_error:
        return exhausted_docker_error
    exhausted_reload_error = _detect_exhausted_docker_reload(harness, pending)
    if exhausted_reload_error:
        return exhausted_reload_error
    if _consume_repeat_guard_one_shot_allowance(harness, pending):
        return None
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    if pending.tool_name == "web_fetch" and isinstance(scratchpad, dict):
        exhausted = scratchpad.get("_web_fetch_budget_exhausted")
        if isinstance(exhausted, dict) and exhausted.get("terminal"):
            detail = str(exhausted.get("error") or "web_fetch budget exhausted for this run").strip()
            return _format_repeated_tool_loop_message(
                harness,
                pending,
                f"Guard tripped: {detail}. Do not retry web_fetch in this run; use existing search results, artifacts, or another strategy.",
            )
    if isinstance(scratchpad, dict):
        nudge_key = f"generic_loop:{pending.tool_name}:{json.dumps(json_safe_value(pending.args), sort_keys=True)}"
        if scratchpad.get("_generic_loop_nudged") == nudge_key:
            return _format_repeated_tool_loop_message(
                harness,
                pending,
                f"Guard tripped: repeated tool call loop ({pending.tool_name} repeated with identical arguments after prior nudge)",
            )
    if _repair_phase_mutation_tool_repeat_is_loop(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            f"Guard tripped: repeated {pending.tool_name} in repair phase with identical arguments",
        )
    if _ssh_file_read_repeats_deterministic_not_found(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            (
                "Guard tripped: repeated deterministic missing remote file read "
                "(ssh_file_read path was already reported not found)"
            ),
        )
    if _ssh_exec_repeats_remote_read_target(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            "Guard tripped: repeated remote file read loop (the same remote file was already inspected; use visible evidence to patch, verify, or inspect a different target)",
        )
    if _ssh_exec_immediate_identical_probe_is_loop(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            "Guard tripped: immediate duplicate SSH diagnostic probe (use the existing result and inspect a different layer or make a justified change)",
        )
    if _dir_list_same_path_repeat_is_loop(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            "Guard tripped: repeated dir_list loop (same path repeated without progress)",
        )
    if _dir_list_exploration_progress_is_progress(harness, pending):
        return None
    if _file_read_line_progress_is_progress(harness, pending):
        return None
    if _file_read_immediate_full_repeat_is_loop(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            (
                "Guard tripped: repeated full file_read loop "
                "(same canonical path already read and no mutation or verifier ran since)"
            ),
        )
    if _file_read_failed_repeat_is_loop(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            (
                "Guard tripped: repeated file_read failure loop "
                "(same path failed previously; do not retry the same failing read)"
            ),
        )
    if _artifact_read_targets_mutation_result_loop(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            (
                "Guard tripped: repeated read-only loop "
                "(successful SSH file mutation artifact already contains the write confirmation)"
            ),
        )
    if _artifact_read_past_eof_is_loop(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            "Guard tripped: repeated artifact_read EOF overread loop",
        )
    if _artifact_read_target_path_re_read_is_loop(harness, pending):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            (
                "Guard tripped: repeated read-only loop "
                "(re-reading an instruction/target file that was already read; "
                "use the most recent tool output to fix the blocker instead)"
            ),
        )
    if _artifact_read_line_progress_is_progress(harness, pending):
        return None
    if _ssh_file_read_after_remote_mutation_is_progress(harness, pending):
        return None
    from ..challenge_progress import terminal_readiness_state
    if terminal_readiness_state(getattr(harness, "state", None)):
        if pending.tool_name in {"dir_list", "file_read", "loop_status", "artifact_read", "artifact_grep", "artifact_print"}:
            return _format_repeated_tool_loop_message(
                harness, pending,
                "Guard tripped: terminal readiness reached. Stop exploratory reads and call task_complete or task_fail."
            )
    cwd = _cwd_for_fingerprint(harness)
    history = _tool_attempt_history(harness)
    candidate = {
        "tool_name": pending.tool_name,
        "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args, cwd=cwd),
        "semantic_fingerprint": _semantic_tool_call_fingerprint(pending.tool_name, pending.args, cwd=cwd),
    }
    identical_limit, window_limit, unique_limit = _repeat_loop_limits(harness, pending)
    recent_window = history[-(window_limit - 1) :] + [candidate]
    exact_streak = history[-(identical_limit - 1) :] + [candidate]
    if (
        len(exact_streak) >= identical_limit
        and len({_history_fingerprint(item) for item in exact_streak}) == 1
    ):
        return _format_repeated_tool_loop_message(
            harness,
            pending,
            (
                "Guard tripped: repeated tool call loop "
                f"({pending.tool_name} repeated with identical arguments)"
            ),
        )
    if pending.tool_name in _STRICT_LOOP_GUARD_TOOLS:
        identical_occurrences = sum(
            1
            for item in recent_window
            if _history_fingerprint(item) == str(candidate.get("semantic_fingerprint", ""))
        )
        if identical_occurrences >= identical_limit:
            return _format_repeated_tool_loop_message(
                harness,
                pending,
                (
                    "Guard tripped: repeated tool call loop "
                    f"({pending.tool_name} repeated {identical_occurrences} times with identical arguments)"
                ),
            )
    if len(recent_window) < window_limit:
        return None

    tool_names = {str(item.get("tool_name", "")) for item in recent_window}
    fingerprints = [_history_fingerprint(item) for item in recent_window]

    if len(set(fingerprints)) <= unique_limit:
        if len(tool_names) == 1:
            return _format_repeated_tool_loop_message(
                harness,
                pending,
                (
                    "Guard tripped: repeated tool exploration loop "
                    f"({pending.tool_name} cycling through near-identical arguments without progress)"
                ),
            )
        else:
            tools_str = ", ".join(sorted(tool_names))
            return _format_repeated_tool_loop_message(
                harness,
                pending,
                (
                    "Guard tripped: cyclic multi-tool loop "
                    f"(cycling between tools [{tools_str}] with recurring identical arguments without progress)"
                ),
            )
    return None


def _ssh_exec_repeats_remote_read_target(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "ssh_exec":
        return False
    candidate = PendingToolCall(tool_name="ssh_exec", args=pending.args)
    targets = set(ssh_exec_read_targets(candidate))
    if not targets:
        return False
    history = _tool_attempt_history(harness)
    occurrences = {target: 1 for target in targets}
    for item in history[-11:]:
        if str(item.get("tool_name") or "") != "ssh_exec":
            continue
        args = _extract_args_from_fingerprint(_history_fingerprint(item))
        if not args:
            continue
        prior = PendingToolCall(tool_name="ssh_exec", args=args)
        for target in targets.intersection(ssh_exec_read_targets(prior)):
            occurrences[target] += 1
    return any(count >= 3 for count in occurrences.values())


def _ssh_exec_immediate_identical_probe_is_loop(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "ssh_exec":
        return False
    command = str(pending.args.get("command") or "").strip()
    if not command or re.search(
        r"\b(?:rm|mv|cp|chmod|chown|mkdir|touch|tee|truncate|sed\s+-i|"
        r"docker\s+compose\s+(?:up|down|restart|stop|start|pull|build)|"
        r"systemctl\s+(?:start|stop|restart|reload|enable|disable))\b|>>?",
        command,
        re.IGNORECASE,
    ):
        return False
    history = _tool_attempt_history(harness)
    if not history or str(history[-1].get("tool_name") or "") != "ssh_exec":
        return False
    candidate = _semantic_tool_call_fingerprint(
        pending.tool_name,
        pending.args,
        cwd=_cwd_for_fingerprint(harness),
    )
    return _history_fingerprint(history[-1]) == candidate


def _detect_exhausted_docker_registry_family(harness: Any, pending: PendingToolCall) -> str | None:
    if pending.tool_name not in {"shell_exec", "ssh_exec"}:
        return None
    command = str(pending.args.get("command") or "").strip()
    if not command:
        return None
    family = docker_retry_family(command)
    if not family:
        return None
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    exhausted = scratchpad.get("_docker_registry_exhausted_families", [])
    if not isinstance(exhausted, list) or family not in exhausted:
        return None
    image_ref = family.split("::", 1)[-1]
    return (
        "Guard tripped: repeated Docker registry/image resolution loop "
        f"for `{image_ref}`. This image reference already failed multiple times. "
        "Do not retry the same image ref. Verify the exact repo/tag from trusted docs or web search, "
        "check `docker image ls`, or switch to a different image/package."
    )


def _detect_exhausted_docker_reload(harness: Any, pending: PendingToolCall) -> str | None:
    if pending.tool_name not in {"shell_exec", "ssh_exec"}:
        return None
    target = docker_reload_target(str(pending.args.get("command") or ""))
    if not target:
        return None
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    failed = scratchpad.get("_last_failed_verifier") if isinstance(scratchpad, dict) else None
    attempts = scratchpad.get("_successful_reload_attempts", {}) if isinstance(scratchpad, dict) else {}
    if not isinstance(failed, dict) or not isinstance(attempts, dict):
        return None
    failure_fingerprint = "|".join(
        str(failed.get(key) or "").strip().lower()
        for key in ("command", "failure_mode", "key_stderr", "key_stdout")
    )
    if int(attempts.get(f"{target}|{failure_fingerprint}", 0) or 0) < 2:
        return None
    service = target.split(":", 1)[-1]
    return (
        "Guard tripped: two equivalent successful Docker reload/restart attempts left acceptance unchanged. "
        f"Do not reload `{service}` again. Recreate only the affected service/container, then rerun the direct acceptance check."
    )


def allow_repeated_tool_call_once(harness: Any, tool_name: str, args: dict[str, Any]) -> None:
    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return
    fingerprint = _tool_call_fingerprint(tool_name, args, cwd=_cwd_for_fingerprint(harness))
    allowances = scratchpad.setdefault("_repeat_guard_one_shot_fingerprints", [])
    if isinstance(allowances, list) and fingerprint not in allowances:
        allowances.append(fingerprint)


def _consume_repeat_guard_one_shot_allowance(harness: Any, pending: PendingToolCall) -> bool:
    state = getattr(harness, "state", None)
    if state is None:
        return False
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return False
    allowances = scratchpad.get("_repeat_guard_one_shot_fingerprints", [])
    if not isinstance(allowances, list):
        return False
    fingerprint = _tool_call_fingerprint(pending.tool_name, pending.args, cwd=_cwd_for_fingerprint(harness))
    if fingerprint not in allowances:
        return False
    allowances.remove(fingerprint)
    if not allowances:
        scratchpad.pop("_repeat_guard_one_shot_fingerprints", None)
    return True
