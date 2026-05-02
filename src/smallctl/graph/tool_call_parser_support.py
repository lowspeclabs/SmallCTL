from __future__ import annotations

import json
from typing import Any

from ..state import WriteSession, json_safe_value
from ..task_targets import extract_task_target_paths, primary_task_target_path
from .state import PendingToolCall
from .tool_artifact_recovery import (
    _artifact_read_recovery_hint,
    _artifact_read_synthesis_hint,
    _clear_artifact_read_guard_state,
    _extract_artifact_id_from_args,
    _fallback_repeated_artifact_read,
    _fallback_repeated_file_read,
    _find_full_file_artifact_for_path,
    _read_artifact_text,
    _resolve_artifact_record,
    _should_suppress_resolved_plan_artifact_read,
    _choose_artifact_grep_query,
)
from .tool_loop_guards import (
    _coerce_int_or_none,
    _consume_repeat_guard_one_shot_allowance,
    _detect_timeout_recovered_incomplete_tool_call,
    _detect_hallucinated_tool_call,
    _detect_placeholder_tool_call,
    _detect_repeated_tool_loop,
    _dir_list_exploration_progress_is_progress,
    _dir_list_repeat_has_intervening_progress,
    _dir_list_same_path_repeat_is_loop,
    _extract_args_from_fingerprint,
    _extract_path_from_fingerprint,
    _file_read_line_progress_is_progress,
    _is_strict_subpath,
    _normalize_json_like,
    _normalize_path_token,
    _normalize_shell_command,
    _normalize_tool_args,
    _normalize_token,
    _placeholder_token,
    _placeholder_value_looks_generic,
    _record_tool_attempt,
    _repeat_loop_limits,
    _requested_artifact_read_target,
    _requested_file_read_range,
    _resolve_dir_list_path,
    _resolve_file_read_path,
    _tool_attempt_history,
    _tool_call_fingerprint,
    _clear_tool_attempt_history,
    allow_repeated_tool_call_once,
)
from .tool_write_session_policy import (
    _active_write_session_for_target,
    _ensure_chunk_write_session,
    _should_enter_chunk_mode,
    _suggested_chunk_sections,
)
from .tool_write_session_support import (
    _assistant_declares_read_before_write,
    _assistant_text_target_paths,
    _build_schema_repair_message,
    _declared_read_before_write_reason,
    _detect_oversize_write_payload,
    _infer_write_tool_path,
    _recover_declared_read_before_write,
    _repair_active_write_session_args,
    _salvage_active_write_session_append,
)

_REPEATED_TOOL_HISTORY_LIMIT = 24
_IDENTICAL_TOOL_CALL_STREAK_LIMIT = 6
_REPEATED_TOOL_WINDOW = 12
_REPEATED_TOOL_UNIQUE_LIMIT = 6
_STRICT_LOOP_GUARD_TOOLS = {"dir_list", "file_read", "artifact_read"}
_REPEAT_GUARD_ONE_SHOT_FINGERPRINTS_KEY = "_repeat_guard_one_shot_fingerprints"
_PATCH_EXISTING_STAGE_READ_CONTRACT_KEY = "_patch_existing_stage_read_contract"
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


def _detect_missing_required_tool_arguments(harness: Any, pending: PendingToolCall) -> tuple[str, dict[str, Any]] | None:
    registry = getattr(harness, "registry", None)
    if registry is None:
        return None
    tool_spec = registry.get(pending.tool_name)
    if tool_spec is None:
        return None
    required = tool_spec.schema.get("required", [])
    if not required:
        return None

    missing_fields = []
    for field in required:
        value = pending.args.get(field)
        if value is None:
            missing_fields.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing_fields.append(field)

    if not missing_fields:
        return None

    message = (
        f"Tool call '{pending.tool_name}' was emitted without arguments. "
        f"Required fields: {', '.join(str(field) for field in missing_fields)}."
    )
    return message, {
        "tool_name": pending.tool_name,
        "tool_call_id": pending.tool_call_id,
        "required_fields": list(missing_fields),
        "raw_arguments": pending.raw_arguments,
    }


def _detect_empty_file_write_payload(
    harness: Any,
    pending: PendingToolCall,
) -> tuple[str, dict[str, Any]] | None:
    if pending.tool_name not in {"file_write", "file_append"}:
        return None

    state = getattr(harness, "state", None)
    if state is None:
        return None
    write_session = getattr(state, "write_session", None)
    if write_session and write_session.write_session_mode == "stub_and_fill":
        return None

    content = pending.args.get("content")
    if content is not None and str(content).strip():
        return None

    required_fields: list[str] = []
    path_value = pending.args.get("path")
    if path_value is None or (isinstance(path_value, str) and not path_value.strip()):
        required_fields.append("path")
    if content is None or not str(content).strip():
        required_fields.append("content")

    message = (
        f"Empty payload rejected for `{pending.tool_name}`. Provide concrete content. "
        "If a write session is active, resume it with `file_write` and full session metadata for chunk continuation, "
        "or use `file_patch` for a narrow exact edit inside the staged/current copy, or `ast_patch` for a narrow structural edit. "
        "For localized edits to an existing file, do not retry a full `file_write` unless you intend a complete rewrite. The target path remains the canonical "
        "destination; the staged copy is the read/verify source while the session is active."
    )
    return message, {
        "tool_name": pending.tool_name,
        "tool_call_id": pending.tool_call_id,
        "reason": "empty_payload",
        "required_fields": required_fields,
    }


def _detect_patch_existing_stage_read_contract_violation(
    harness: Any,
    pending: PendingToolCall,
) -> tuple[str, dict[str, Any]] | None:
    state = getattr(harness, "state", None)
    if state is None:
        return None
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    contract = scratchpad.get(_PATCH_EXISTING_STAGE_READ_CONTRACT_KEY)
    if not isinstance(contract, dict):
        return None

    session = getattr(state, "write_session", None)
    if session is None or str(getattr(session, "status", "") or "").strip().lower() == "complete":
        scratchpad.pop(_PATCH_EXISTING_STAGE_READ_CONTRACT_KEY, None)
        return None

    contract_session_id = str(contract.get("session_id") or "").strip()
    if contract_session_id and str(getattr(session, "write_session_id", "") or "").strip() != contract_session_id:
        scratchpad.pop(_PATCH_EXISTING_STAGE_READ_CONTRACT_KEY, None)
        return None

    contract_target = str(contract.get("target_path") or getattr(session, "write_target_path", "") or "").strip()
    if not contract_target:
        scratchpad.pop(_PATCH_EXISTING_STAGE_READ_CONTRACT_KEY, None)
        return None

    if pending.tool_name not in {"file_write", "file_append", "file_patch", "ast_patch"}:
        return None

    from ..tools.fs import _same_target_path

    pending_path = str(pending.args.get("path") or "").strip()
    if pending_path and not _same_target_path(contract_target, pending_path, getattr(state, "cwd", None)):
        return None

    if pending.tool_name in {"file_patch", "ast_patch"}:
        scratchpad.pop(_PATCH_EXISTING_STAGE_READ_CONTRACT_KEY, None)
        return None

    strategy = str(pending.args.get("replace_strategy") or "auto").strip().lower() or "auto"
    if strategy == "overwrite":
        scratchpad.pop(_PATCH_EXISTING_STAGE_READ_CONTRACT_KEY, None)
        return None

    staging_path = str(contract.get("staging_path") or getattr(session, "write_staging_path", "") or "").strip()
    message = (
        "Patch-existing recovery already reread the staged copy. "
        "The next same-target write must choose one explicit repair shape: "
        "`file_patch` for a narrow exact edit inside the staged copy, `ast_patch` for a narrow structural edit, or "
        "`file_write` with `replace_strategy='overwrite'` to replace the entire staged file. "
        "Do not send another implicit first-chunk `file_write`/`file_append` with `replace_strategy='auto'`."
    )
    return message, {
        "tool_name": pending.tool_name,
        "tool_call_id": pending.tool_call_id,
        "reason": "patch_existing_stage_read_requires_explicit_shape",
        "target_path": contract_target,
        "write_session_id": contract_session_id or str(getattr(session, "write_session_id", "") or "").strip(),
        "staging_path": staging_path,
        "required_followup_tools": ["file_patch", "ast_patch", "file_write"],
        "allowed_replace_strategy": "overwrite",
        "offending_field": "replace_strategy",
    }


def _record_tool_attempt(harness: Any, pending: PendingToolCall) -> None:
    history = _tool_attempt_history(harness)
    history.append(
        {
            "tool_name": pending.tool_name,
            "fingerprint": _tool_call_fingerprint(pending.tool_name, pending.args),
        }
    )
    harness.state.scratchpad["_tool_attempt_history"] = history[-_REPEATED_TOOL_HISTORY_LIMIT:]


def _clear_tool_attempt_history(harness: Any) -> None:
    harness.state.scratchpad.pop("_tool_attempt_history", None)


def _normalize_shell_command(command: str) -> str:
    parts = command.strip().split()
    if not parts:
        return ""
    normalized_parts = [_normalize_path_token(part) for part in parts]
    return " ".join(normalized_parts)


def _normalize_path_token(value: str) -> str:
    stripped = value.strip()
    if len(stripped) > 1 and stripped.endswith("/") and "/" in stripped[:-1]:
        return stripped.rstrip("/")
    return stripped


def _normalize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_like(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, str):
        collapsed = " ".join(value.strip().split())
        return _normalize_path_token(collapsed)
    return json_safe_value(value)


def _tool_call_fingerprint(tool_name: str, args: dict[str, Any]) -> str:
    normalized_args = _normalize_tool_args(tool_name, args)
    return json.dumps({"tool_name": tool_name, "args": normalized_args}, sort_keys=True, ensure_ascii=True)


def _normalize_tool_args(tool_name: str, args: dict[str, Any]) -> Any:
    if not isinstance(args, dict):
        return {}
    normalized = json_safe_value(args)
    if not isinstance(normalized, dict):
        return {}
    if tool_name == "shell_exec":
        command = normalized.get("command")
        if command is not None:
            normalized["command"] = _normalize_shell_command(str(command))
    return _normalize_json_like(normalized)
