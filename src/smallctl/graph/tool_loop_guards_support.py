from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..guards import is_four_b_or_under_model_name, is_seven_b_or_under_model_name
from .state import PendingToolCall
from .tool_loop_guard_constants import (
    _PLACEHOLDER_ARG_KEY_TOKENS,
    _PLACEHOLDER_ARG_VALUE_TOKENS,
)


def _normalize_token(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower())
    return normalized.strip("_")


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


def _tool_call_fingerprint(tool_name: str, args: dict[str, Any], *, cwd: str | None = None) -> str:
    normalized_args = _normalize_tool_args(tool_name, args, cwd=cwd)
    return json.dumps({"tool_name": tool_name, "args": normalized_args}, sort_keys=True, ensure_ascii=True)


def _semantic_tool_call_fingerprint(tool_name: str, args: dict[str, Any], *, cwd: str | None = None) -> str:
    return _tool_call_fingerprint(tool_name, args, cwd=cwd)


def _cwd_for_fingerprint(harness: Any) -> str | None:
    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    return str(cwd) if isinstance(cwd, str) and cwd else None


def _normalize_tool_args(tool_name: str, args: dict[str, Any], *, cwd: str | None = None) -> Any:
    if not isinstance(args, dict):
        return args
    normalized = {str(key): _normalize_json_like(value) for key, value in args.items()}
    if tool_name in {"shell_exec", "bash_exec", "ssh_exec"} and "command" in normalized:
        normalized["command"] = _normalize_shell_command(str(normalized["command"]))
    if tool_name in {
        "file_read",
        "dir_list",
        "dir_tree",
        "file_write",
        "file_append",
        "file_patch",
        "ast_patch",
        "file_delete",
    }:
        for key in ("path", "target_path"):
            value = normalized.get(key)
            if isinstance(value, str) and value.strip():
                normalized[key] = _normalize_local_path_for_fingerprint(value, cwd=cwd)
    return normalized


def _normalize_shell_command(command: str) -> str:
    command = str(command or "").strip()
    command = re.sub(r"\s+", " ", command)
    return command


def _normalize_path_token(value: str) -> str:
    return str(value or "").strip()


def _normalize_local_path_for_fingerprint(value: str, *, cwd: str | None = None) -> str:
    candidate = Path(str(value or "").strip() or ".")
    if not candidate.is_absolute():
        base = Path(cwd) if cwd else Path.cwd()
        candidate = base / candidate
    try:
        return str(candidate.resolve())
    except Exception:
        return str(candidate)


def _normalize_json_like(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_like(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_like(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_like(item) for item in value]
    return value


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
    model_name = _model_name_for_loop_guard(harness)
    is_small = is_seven_b_or_under_model_name(model_name)
    if not is_small and not is_four_b_or_under_model_name(model_name):
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
    if tool_name == "ssh_exec":
        return (
            "Directive Hint: Stop rerunning similar SSH commands. Combine remaining discovery into a single `ssh_exec` using `&&` or `;`. "
            "Prefer `ssh_file_read` over `ssh_exec cat` for reading remote files. Then synthesize findings and move forward."
        )
    if tool_name == "ssh_file_read":
        return (
            f"Directive Hint: Stop re-reading the same remote file{path_note}. Use the evidence already in Working Memory. "
            "If you need to verify state, run one focused `ssh_exec` command. Otherwise synthesize and proceed."
        )
    return (
        f"Directive Hint: Stop repeating `{tool_name}` with near-identical arguments. Use the current evidence and choose a different next action: "
        "`file_patch`, `ast_patch`, `shell_exec`, or `task_complete` as appropriate."
    )
