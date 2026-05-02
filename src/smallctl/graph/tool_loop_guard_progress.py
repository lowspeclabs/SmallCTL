from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .state import PendingToolCall


def _model_is_exact_small_gemma_4_it(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return bool(
        normalized
        and any(
            normalized == suffix or normalized.endswith(f"/{suffix}")
            for suffix in (
                "gemma-4-e2b-it",
                "gemma-4-e4b-it",
            )
        )
    )


def _tool_attempt_history(harness: Any) -> list[dict[str, str]]:
    state = getattr(harness, "state", None)
    if state is None:
        return []
    scratchpad = getattr(state, "scratchpad", {})
    history = scratchpad.get("_tool_attempt_history", [])
    if not isinstance(history, list):
        return []
    return history


def _coerce_int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _extract_args_from_fingerprint(fingerprint: str) -> dict[str, Any] | None:
    if not fingerprint:
        return None
    try:
        payload = json.loads(fingerprint)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    args = payload.get("args", {})
    return args if isinstance(args, dict) else None


def _requested_file_read_range(args: dict[str, Any]) -> tuple[int | None, int | None]:
    if not isinstance(args, dict):
        return (None, None)
    start_line = args.get("requested_start_line", args.get("start_line"))
    end_line = args.get("requested_end_line", args.get("end_line"))
    return (_coerce_int_or_none(start_line), _coerce_int_or_none(end_line))


def _requested_artifact_read_target(args: dict[str, Any]) -> str:
    if not isinstance(args, dict):
        return ""
    artifact_id = args.get("artifact_id")
    return str(artifact_id or "").strip()


def _resolve_dir_list_path(harness: Any, args: dict[str, Any]) -> Path | None:
    raw_path = args.get("path", ".")
    if not isinstance(raw_path, str):
        return None
    candidate = Path(raw_path.strip() or ".")
    if candidate.is_absolute():
        return candidate.resolve()
    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
    try:
        return (base / candidate).resolve()
    except Exception:
        return (base / candidate)


def _resolve_file_read_path(harness: Any, args: dict[str, Any]) -> Path | None:
    raw_path = args.get("path")
    if not isinstance(raw_path, str):
        return None
    candidate = Path(raw_path.strip() or ".")
    if candidate.is_absolute():
        try:
            return candidate.resolve()
        except Exception:
            return candidate
    cwd = getattr(getattr(harness, "state", None), "cwd", None)
    base = Path(cwd) if isinstance(cwd, str) and cwd else Path.cwd()
    try:
        return (base / candidate).resolve()
    except Exception:
        return base / candidate


def _is_strict_subpath(child: Path, parent: Path) -> bool:
    if child == parent:
        return False
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _extract_path_from_fingerprint(harness: Any, fingerprint: str) -> Path | None:
    args = _extract_args_from_fingerprint(fingerprint)
    if not isinstance(args, dict):
        return None
    return _resolve_dir_list_path(harness, args)


def _dir_list_exploration_progress_is_progress(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "dir_list":
        return False

    candidate_path = _resolve_dir_list_path(harness, pending.args)
    if candidate_path is None:
        return False

    history = _tool_attempt_history(harness)
    recent_paths: list[Path] = []
    for item in reversed(history):
        if str(item.get("tool_name", "")) != "dir_list":
            continue
        fingerprint = str(item.get("fingerprint", ""))
        path = _extract_path_from_fingerprint(harness, fingerprint)
        if path is None:
            continue
        recent_paths.append(path)
        if len(recent_paths) >= 4:
            break

    if not recent_paths:
        return False

    paths = list(reversed(recent_paths))
    paths.append(candidate_path)
    if len(paths) < 2:
        return False

    monotonic_descent = all(_is_strict_subpath(paths[index + 1], paths[index]) for index in range(len(paths) - 1))
    return monotonic_descent


def _dir_list_same_path_repeat_is_loop(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "dir_list":
        return False

    candidate_path = _resolve_dir_list_path(harness, pending.args)
    if candidate_path is None:
        return False

    history = _tool_attempt_history(harness)
    for index in range(len(history) - 1, -1, -1):
        item = history[index]
        if str(item.get("tool_name", "")) != "dir_list":
            continue
        fingerprint = str(item.get("fingerprint", ""))
        path = _extract_path_from_fingerprint(harness, fingerprint)
        if path is None:
            continue
        if path != candidate_path:
            continue
        if not _model_is_exact_small_gemma_4_it(getattr(getattr(harness, "client", None), "model", None)):
            return True
        return not _dir_list_repeat_has_intervening_progress(history, index)
    return False


def _dir_list_repeat_has_intervening_progress(
    history: list[dict[str, str]],
    prior_dir_list_index: int,
) -> bool:
    progress_tools = {"artifact_read", "file_read", "shell_exec", "ssh_exec", "bash_exec"}
    if prior_dir_list_index < 0 or prior_dir_list_index >= len(history):
        return False
    for item in history[prior_dir_list_index + 1 :]:
        if str(item.get("tool_name", "")) in progress_tools:
            return True
    return False


def _file_read_line_progress_is_progress(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "file_read":
        return False

    candidate_path = _resolve_file_read_path(harness, pending.args)
    if candidate_path is None:
        return False

    candidate_range = _requested_file_read_range(pending.args)
    if candidate_range == (None, None):
        return False

    history = _tool_attempt_history(harness)
    seen_ranges = set()
    has_prior_reads = False

    for item in reversed(history[-12:]):
        if str(item.get("tool_name", "")) != "file_read":
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if not isinstance(args, dict):
            continue
        prior_path = _resolve_file_read_path(harness, args)
        if prior_path is None or prior_path != candidate_path:
            continue
        prior_range = _requested_file_read_range(args)
        if prior_range == candidate_range:
            return False  # We already requested this exact range recently; not progress.
        seen_ranges.add(prior_range)
        has_prior_reads = True

    return has_prior_reads


def _artifact_read_line_progress_is_progress(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "artifact_read":
        return False

    candidate_artifact_id = _requested_artifact_read_target(pending.args)
    if not candidate_artifact_id:
        return False

    candidate_range = _requested_file_read_range(pending.args)
    if candidate_range == (None, None):
        return False

    history = _tool_attempt_history(harness)
    seen_ranges = set()
    has_prior_reads = False

    for item in reversed(history[-12:]):
        if str(item.get("tool_name", "")) != "artifact_read":
            continue
        args = _extract_args_from_fingerprint(str(item.get("fingerprint", "")))
        if not isinstance(args, dict):
            continue
        prior_artifact_id = _requested_artifact_read_target(args)
        if not prior_artifact_id or prior_artifact_id != candidate_artifact_id:
            continue
        prior_range = _requested_file_read_range(args)
        if prior_range == candidate_range:
            return False  # We already requested this exact range recently; not progress.
        seen_ranges.add(prior_range)
        has_prior_reads = True

    return has_prior_reads
