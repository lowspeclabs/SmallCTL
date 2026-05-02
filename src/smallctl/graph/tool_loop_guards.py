from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ..docker_retry_normalization import docker_retry_family
from ..guards import is_four_b_or_under_model_name
from ..state import json_safe_value
from .state import PendingToolCall
from .tool_loop_guard_progress import (
    _artifact_read_line_progress_is_progress,
    _coerce_int_or_none,
    _dir_list_exploration_progress_is_progress,
    _dir_list_repeat_has_intervening_progress,
    _dir_list_same_path_repeat_is_loop,
    _extract_args_from_fingerprint,
    _extract_path_from_fingerprint,
    _file_read_line_progress_is_progress,
    _is_strict_subpath,
    _model_is_exact_small_gemma_4_it,
    _requested_artifact_read_target,
    _requested_file_read_range,
    _resolve_dir_list_path,
    _resolve_file_read_path,
    _tool_attempt_history,
)

_REPEATED_TOOL_HISTORY_LIMIT = 24
_IDENTICAL_TOOL_CALL_STREAK_LIMIT = 3
_REPEATED_TOOL_WINDOW = 12
_REPEATED_TOOL_UNIQUE_LIMIT = 5
_STRICT_LOOP_GUARD_TOOLS = {
    "dir_list",
    "file_read",
    "artifact_read",
    "artifact_grep",
    "artifact_print",
    "web_search",
    "web_fetch",
}
_STRICT_LOOP_GUARD_IDENTICAL_LIMIT = 3
_STRICT_LOOP_GUARD_WINDOW_LIMIT = 6
_STRICT_LOOP_GUARD_UNIQUE_LIMIT = 3
_DIR_LIST_LOOP_GUARD_STRICTNESS_MULTIPLIER = 1.5
_DIR_LIST_IDENTICAL_TOOL_CALL_STREAK_LIMIT = max(
    2,
    math.ceil(_STRICT_LOOP_GUARD_IDENTICAL_LIMIT / _DIR_LIST_LOOP_GUARD_STRICTNESS_MULTIPLIER),
)
_DIR_LIST_REPEATED_TOOL_WINDOW = max(
    4,
    math.ceil(_STRICT_LOOP_GUARD_WINDOW_LIMIT / _DIR_LIST_LOOP_GUARD_STRICTNESS_MULTIPLIER),
)
_DIR_LIST_REPEATED_TOOL_UNIQUE_LIMIT = max(
    2,
    math.ceil(_STRICT_LOOP_GUARD_UNIQUE_LIMIT / _DIR_LIST_LOOP_GUARD_STRICTNESS_MULTIPLIER),
)
_PLACEHOLDER_TOOL_NAME_TOKENS = {
    "tool_name",
    "function_name",
    "action_name",
    "tool",
    "function",
    "action",
    "name",
}
_PLACEHOLDER_ARG_KEY_TOKENS = {
    "arg",
    "args",
    "argument",
    "arguments",
    "param",
    "params",
    "parameter",
    "parameters",
    "value",
    "field",
}
_PLACEHOLDER_ARG_VALUE_TOKENS = {
    "",
    "arg",
    "args",
    "value",
    "parameter",
    "parameters",
    "param",
    "params",
    "tool_name",
    "function_name",
    "action_name",
    "placeholder",
    "string",
    "text",
}
_EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES = (
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
)
_INCOMPLETE_TOOL_CALL_SCRATCHPAD_KEY = "_last_incomplete_tool_call"


def _normalize_token(value: Any) -> str:
    import re

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return normalized.strip("_")


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
    if pending.tool_name in _STRICT_LOOP_GUARD_TOOLS:
        return (
            _STRICT_LOOP_GUARD_IDENTICAL_LIMIT,
            _STRICT_LOOP_GUARD_WINDOW_LIMIT,
            _STRICT_LOOP_GUARD_UNIQUE_LIMIT,
        )
    return (identical_limit, window_limit, unique_limit)


def _model_name_for_loop_guard(harness: Any) -> str:
    client_model = str(getattr(getattr(harness, "client", None), "model", "") or "").strip()
    if client_model:
        return client_model
    state = getattr(harness, "state", None)
    scratchpad = getattr(state, "scratchpad", {}) if state is not None else {}
    if isinstance(scratchpad, dict):
        return str(scratchpad.get("_model_name") or "").strip()
    return ""


def _directive_hint_for_repeated_tool(harness: Any, pending: PendingToolCall) -> str:
    if not is_four_b_or_under_model_name(_model_name_for_loop_guard(harness)):
        return ""

    args = dict(getattr(pending, "args", {}) or {})
    tool_name = str(getattr(pending, "tool_name", "") or "").strip()
    path = str(args.get("path") or args.get("target_path") or "").strip()
    artifact_id = str(args.get("artifact_id") or args.get("id") or "").strip()
    path_note = f" for `{path}`" if path else ""
    artifact_note = f" `{artifact_id}`" if artifact_id else ""

    if tool_name == "file_read":
        return (
            f"Directive Hint: Stop rereading the same file{path_note}. Use the evidence already in Working Memory. "
            "If an edit is needed, call `file_patch` or `ast_patch`; if proof is needed, call `shell_exec` with a focused verifier; "
            "if the answer is ready, call `task_complete`."
        )
    if tool_name == "dir_list":
        return (
            f"Directive Hint: Stop listing the same directory{path_note}. Pick the visible target path and move to `file_read`, "
            "`file_patch`, or `shell_exec` for the next concrete step."
        )
    if tool_name in {"artifact_read", "artifact_print", "artifact_grep"}:
        return (
            f"Directive Hint: Stop repeating `{tool_name}` on artifact{artifact_note}. Use the pinned evidence table and current artifact summary. "
            "Only call `artifact_read` again with a higher `start_line` when you need unseen lines; otherwise patch, use a different source/query, "
            "call `loop_status`, ask for help if the required tool is unavailable, or complete."
        )
    if tool_name in {"file_patch", "ast_patch"}:
        return (
            f"Directive Hint: Stop retrying the same patch{path_note}. Change the patch arguments based on the last failure, "
            "read the target once for exact context, or run `shell_exec` to verify the current state."
        )
    if tool_name == "shell_exec":
        return (
            "Directive Hint: Stop rerunning the same command. Use the existing command output or artifact; "
            "only run a different focused command if it will produce new evidence."
        )
    return (
        f"Directive Hint: Stop repeating `{tool_name}` with near-identical arguments. Use the current evidence and choose a different next action: "
        "`file_patch`, `ast_patch`, `shell_exec`, or `task_complete` as appropriate."
    )


def _format_repeated_tool_loop_message(harness: Any, pending: PendingToolCall, message: str) -> str:
    hint = _directive_hint_for_repeated_tool(harness, pending)
    if not hint:
        return message
    return f"{message}. {hint}"


def _artifact_record_for_pending_read(harness: Any, pending: PendingToolCall) -> Any | None:
    if pending.tool_name != "artifact_read":
        return None
    artifact_id = _requested_artifact_read_target(pending.args)
    if not artifact_id:
        return None
    artifacts = getattr(getattr(harness, "state", None), "artifacts", {})
    if not isinstance(artifacts, dict):
        return None
    return artifacts.get(artifact_id)


def _artifact_read_targets_mutation_result_loop(harness: Any, pending: PendingToolCall) -> bool:
    artifact = _artifact_record_for_pending_read(harness, pending)
    if artifact is None:
        return False
    tool_name = str(getattr(artifact, "tool_name", "") or getattr(artifact, "kind", "") or "").strip()
    if tool_name not in {"ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}:
        return False
    artifact_id = _requested_artifact_read_target(pending.args)
    if not artifact_id:
        return False
    prior_reads = 0
    for item in reversed(_tool_attempt_history(harness)[-6:]):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if isinstance(args, dict) and _requested_artifact_read_target(args) == artifact_id:
            prior_reads += 1
    return prior_reads >= 1


def _artifact_read_past_eof_is_loop(harness: Any, pending: PendingToolCall) -> bool:
    artifact = _artifact_record_for_pending_read(harness, pending)
    if artifact is None:
        return False
    start_line, _end_line = _requested_file_read_range(pending.args)
    if start_line is None or start_line <= 0:
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
                total_lines = len(Path(content_path).read_text(encoding="utf-8").splitlines())
            except OSError:
                total_lines = None
    if total_lines is None or start_line <= total_lines:
        return False
    artifact_id = _requested_artifact_read_target(pending.args)
    for item in reversed(_tool_attempt_history(harness)[-6:]):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if not isinstance(args, dict) or _requested_artifact_read_target(args) != artifact_id:
            continue
        prior_start, _prior_end = _requested_file_read_range(args)
        if prior_start is not None and prior_start > total_lines:
            return True
    return False


def _placeholder_token(value: Any) -> str:
    return _normalize_token(value)


def _placeholder_value_looks_generic(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        token = _placeholder_token(value)
        return token in _PLACEHOLDER_ARG_VALUE_TOKENS or token.startswith("placeholder")
    if isinstance(value, dict):
        if not value:
            return True
        return all(
            _placeholder_token(key) in _PLACEHOLDER_ARG_KEY_TOKENS
            and _placeholder_value_looks_generic(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return bool(value) and all(_placeholder_value_looks_generic(item) for item in value)
    return False


def _tool_call_fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    normalized_args = _normalize_tool_args(tool_name, args)
    return json.dumps({"tool_name": tool_name, "args": normalized_args}, sort_keys=True, ensure_ascii=True)


def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> Any:
    if not isinstance(args, dict):
        return args
    normalized = {str(key): _normalize_json_like(value) for key, value in args.items()}
    if tool_name in {"shell_exec", "bash_exec", "ssh_exec"} and "command" in normalized:
        normalized["command"] = _normalize_shell_command(str(normalized["command"]))
    return normalized


def _normalize_shell_command(command: str) -> str:
    import re

    command = str(command or "").strip()
    command = re.sub(r"\s+", " ", command)
    return command


def _normalize_path_token(value: str) -> str:
    return str(value or "").strip()


def _normalize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_like(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_like(item) for item in value]
    return value


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
    history.append(
        {
            "tool_name": pending.tool_name,
            "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args),
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
    if _consume_repeat_guard_one_shot_allowance(harness, pending):
        return None
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
    if _artifact_read_line_progress_is_progress(harness, pending):
        return None
    history = _tool_attempt_history(harness)
    candidate = {
        "tool_name": pending.tool_name,
        "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args),
    }
    identical_limit, window_limit, unique_limit = _repeat_loop_limits(harness, pending)
    recent_window = history[-(window_limit - 1) :] + [candidate]
    exact_streak = history[-(identical_limit - 1) :] + [candidate]
    if (
        len(exact_streak) >= identical_limit
        and len({str(item.get("fingerprint", "")) for item in exact_streak}) == 1
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
            if str(item.get("fingerprint", "")) == str(candidate.get("fingerprint", ""))
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
    fingerprints = [str(item.get("fingerprint", "")) for item in recent_window]

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


def allow_repeated_tool_call_once(harness: Any, tool_name: str, args: dict[str, Any]) -> None:
    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return
    fingerprint = _tool_call_fingerprint(tool_name, args)
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
    fingerprint = _tool_call_fingerprint(pending.tool_name, pending.args)
    if fingerprint not in allowances:
        return False
    allowances.remove(fingerprint)
    if not allowances:
        scratchpad.pop("_repeat_guard_one_shot_fingerprints", None)
    return True
