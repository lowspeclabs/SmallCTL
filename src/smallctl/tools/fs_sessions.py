from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from ..state import LoopState


def _normalize_section_name(section_name: str | None, section_id: str | None) -> str:
    return str(section_name or section_id or "unnamed").strip() or "unnamed"


def _normalize_replace_strategy(replace_strategy: str | None) -> str:
    strategy = str(replace_strategy or "auto").strip().lower()
    if strategy in {"overwrite", "replace", "rewrite"}:
        return "overwrite"
    if strategy == "append":
        return "append"
    return "auto"


def _write_session_can_finalize(session: Any) -> bool:
    mode = str(getattr(session, "write_session_mode", "") or "").strip().lower()
    return mode in {"chunked_author", "local_repair", "stub_and_fill"}


def _section_name_allows_full_file_finalization(section_name: str) -> bool:
    normalized = str(section_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized in {
        "final_content",
        "full_file",
        "complete_file",
        "entire_file",
        "full_script",
        "final_file",
    }


def _normalize_chunk_section_label(value: str) -> str:
    normalized = str(value or "").strip().lower()
    for token in ("-", " ", "/", "\\"):
        normalized = normalized.replace(token, "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _chunk_section_labels_match(left: str, right: str) -> bool:
    left_normalized = _normalize_chunk_section_label(left)
    right_normalized = _normalize_chunk_section_label(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    left_tokens = {token for token in left_normalized.split("_") if token}
    right_tokens = {token for token in right_normalized.split("_") if token}
    if not left_tokens or not right_tokens:
        return False
    return left_tokens.issubset(right_tokens) or right_tokens.issubset(left_tokens)


def _chunk_section_completed(completed: set[str], label: str) -> bool:
    return any(_chunk_section_labels_match(item, label) for item in completed)


def _infer_next_suggested_section(session: Any, current_section: str) -> str:
    suggestions = [
        str(item or "").strip()
        for item in (getattr(session, "suggested_sections", []) or [])
        if str(item or "").strip()
    ]
    if not suggestions:
        return ""

    normalized_current = _normalize_chunk_section_label(current_section)
    normalized_suggestions = [
        _normalize_chunk_section_label(item)
        for item in suggestions
    ]
    completed = {
        _normalize_chunk_section_label(item)
        for item in (getattr(session, "write_sections_completed", []) or [])
        if str(item or "").strip()
    }
    if normalized_current:
        completed.add(normalized_current)

    current_index = next(
        (
            index
            for index, normalized in enumerate(normalized_suggestions)
            if _chunk_section_labels_match(normalized_current, normalized)
        ),
        -1,
    )
    if current_index >= 0:
        index = current_index
        for next_index in range(index + 1, len(suggestions)):
            if not _chunk_section_completed(completed, normalized_suggestions[next_index]):
                return suggestions[next_index]
        return ""

    for suggestion, normalized in zip(suggestions, normalized_suggestions):
        if not _chunk_section_completed(completed, normalized):
            return suggestion
    return ""


def _write_session_should_finalize(
    session: Any,
    *,
    section_name: str,
    next_section_name: str,
    replace_strategy: str,
    content: str,
) -> bool:
    if next_section_name:
        return False
    if not _write_session_can_finalize(session):
        return False
    if _section_name_allows_full_file_finalization(section_name):
        return True
    if replace_strategy == "overwrite" and _looks_like_full_script_content(content, section_name):
        return True

    suggestions = [
        str(item or "").strip()
        for item in (getattr(session, "suggested_sections", []) or [])
        if str(item or "").strip()
    ]
    if not suggestions:
        return True

    normalized_section = _normalize_chunk_section_label(section_name)
    normalized_suggestions = [
        _normalize_chunk_section_label(item)
        for item in suggestions
    ]
    if normalized_section and _chunk_section_labels_match(normalized_section, normalized_suggestions[-1]):
        return True

    completed = {
        _normalize_chunk_section_label(item)
        for item in (getattr(session, "write_sections_completed", []) or [])
        if str(item or "").strip()
    }
    if normalized_section:
        completed.add(normalized_section)
    return all(_chunk_section_completed(completed, normalized) for normalized in normalized_suggestions)


def _looks_like_full_script_content(content: str, section_name: str) -> bool:
    normalized_section = str(section_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    likely_scaffold_sections = {
        "imports",
        "imports_and_types",
        "imports_and_class",
        "imports_and_classes",
        "imports_and_core",
        "imports_and_implementation",
        "helpers",
        "helper",
        "utils",
        "utilities",
        "constants",
        "types",
        "implementation",
        "core",
        "body",
        "main",
        "main_file",
    }
    if normalized_section not in likely_scaffold_sections:
        return False
    text = str(content or "")
    if len(text) < 300 and text.count("\n") < 12:
        return False
    markers = 0
    lowered = text.lower()
    for token in ("\nclass ", "\ndef ", "\nasync def ", "\nif __name__", "\nunittest.", "\npytest", "\nfrom ", "\nimport "):
        if token in lowered:
            markers += 1
    top_level_defs = sum(
        1
        for line in text.splitlines()
        if line.startswith(("def ", "async def ", "class "))
    )
    if top_level_defs >= 2:
        markers += 1
    return markers >= 3


def _append_unique_section(completed_sections: list[str], section_name: str) -> bool:
    normalized = str(section_name or "").strip()
    if not normalized or normalized in completed_sections:
        return False
    completed_sections.append(normalized)
    return True


def _clone_section_ranges(value: dict[str, dict[str, int]] | None) -> dict[str, dict[str, int]]:
    if not isinstance(value, dict):
        return {}
    cloned: dict[str, dict[str, int]] = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            continue
        try:
            start = int(item.get("start", -1))
            end = int(item.get("end", -1))
        except (TypeError, ValueError):
            continue
        if start < 0 or end < start:
            continue
        cloned[str(key)] = {"start": start, "end": end}
    return cloned


def new_write_session_id() -> str:
    return f"ws_{uuid.uuid4().hex[:6]}"


def infer_write_session_intent(path: str, cwd: str | None = None) -> str:
    from .fs_write_sessions import _resolve

    try:
        return "patch_existing" if _resolve(path, cwd).exists() else "replace_file"
    except Exception:
        return "replace_file"


def _same_target_path(left: str, right: str, cwd: str | None = None) -> bool:
    from .fs_write_sessions import _resolve

    try:
        return _resolve(left, cwd) == _resolve(right, cwd)
    except Exception:
        return str(left) == str(right)


def _repair_cycle_reads(state: LoopState | None) -> list[str]:
    if state is None:
        return []
    reads = state.scratchpad.setdefault("_repair_cycle_reads", [])
    if not isinstance(reads, list):
        reads = []
        state.scratchpad["_repair_cycle_reads"] = reads
    return [str(item) for item in reads if str(item).strip()]


def _record_repair_cycle_read(state: LoopState | None, path: Path) -> None:
    if state is None or not state.repair_cycle_id:
        return
    reads = _repair_cycle_reads(state)
    normalized = str(path.resolve()) if hasattr(path, "resolve") else str(path)
    if normalized not in reads:
        reads.append(normalized)
        state.scratchpad["_repair_cycle_reads"] = reads


def _repair_cycle_allows_patch(state: LoopState | None, path: Path) -> bool:
    if state is None or not state.repair_cycle_id:
        return True
    reads = set(_repair_cycle_reads(state))
    normalized = str(path.resolve()) if hasattr(path, "resolve") else str(path)
    if normalized in reads:
        return True
    return _latest_path_evidence_allows_repair_patch(state, path)


def _latest_path_evidence_allows_repair_patch(state: LoopState, path: Path) -> bool:
    records = getattr(state, "tool_execution_records", None)
    if not isinstance(records, dict) or not records:
        return False

    for record in reversed(list(records.values())):
        if not isinstance(record, dict):
            continue
        tool_name = str(record.get("tool_name") or "").strip()
        if tool_name not in {"file_read", "file_write", "file_append", "file_patch", "ast_patch", "file_delete"}:
            continue
        if not _record_targets_path(record, path, getattr(state, "cwd", None)):
            continue

        result = record.get("result")
        result_success = isinstance(result, dict) and bool(result.get("success"))
        if tool_name == "file_read":
            return result_success
        if tool_name in {"file_patch", "ast_patch"} and not result_success:
            return False
        if tool_name in {"file_write", "file_append", "file_patch", "ast_patch", "file_delete"}:
            return False
    return False


def _record_targets_path(record: dict[str, Any], path: Path, cwd: str | None) -> bool:
    args = record.get("args")
    metadata = {}
    result = record.get("result")
    if isinstance(result, dict) and isinstance(result.get("metadata"), dict):
        metadata = result["metadata"]

    candidates: list[str] = []
    if isinstance(args, dict):
        candidates.extend(
            str(args.get(key) or "").strip()
            for key in ("path", "target_path")
            if str(args.get(key) or "").strip()
        )
    candidates.extend(
        str(metadata.get(key) or "").strip()
        for key in ("requested_path", "path", "target_path")
        if str(metadata.get(key) or "").strip()
    )

    for candidate in candidates:
        if _same_target_path(candidate, str(path), cwd):
            return True
    return False


def _repair_cycle_read_required_metadata(
    state: LoopState | None,
    path: Path,
    *,
    requested_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_path = str(path.resolve()) if hasattr(path, "resolve") else str(path)
    repair_cycle_id = str(getattr(state, "repair_cycle_id", "") or "").strip()
    metadata: dict[str, Any] = {
        "path": normalized_path,
        "system_repair_cycle_id": repair_cycle_id,
        "required_read_paths": _repair_cycle_reads(state),
        "error_kind": "repair_cycle_read_required",
        "recovery_hint": (
            "This is a new repair-cycle read requirement. Prior file_read artifacts do not satisfy "
            "the current repair cycle; call file_read on this path to refresh the disk snapshot."
        ),
        "next_required_tool": {
            "tool_name": "file_read",
            "required_arguments": {"path": requested_path or normalized_path},
            "reason": "new_repair_cycle_requires_fresh_disk_snapshot",
            "system_repair_cycle_id": repair_cycle_id,
        },
    }
    if requested_path is not None:
        metadata["requested_path"] = requested_path
    if extra:
        metadata.update(extra)
    return metadata


def _mark_repeat_patch(state: LoopState | None) -> None:
    if state is None:
        return
    counters = state.stagnation_counters if isinstance(state.stagnation_counters, dict) else {}
    counters["repeat_patch"] = int(counters.get("repeat_patch", 0)) + 1
    state.stagnation_counters = counters


def _mark_repeat_command(state: LoopState | None) -> None:
    if state is None:
        return
    counters = state.stagnation_counters if isinstance(state.stagnation_counters, dict) else {}
    counters["repeat_command"] = int(counters.get("repeat_command", 0)) + 1
    state.stagnation_counters = counters


def _record_file_change(state: LoopState | None, path: Path) -> None:
    if state is None:
        return
    normalized = str(path.resolve()) if hasattr(path, "resolve") else str(path)
    if normalized in state.files_changed_this_cycle:
        _mark_repeat_patch(state)
    changed = [item for item in state.files_changed_this_cycle if item != normalized]
    changed.append(normalized)
    state.files_changed_this_cycle = changed[-12:]
    try:
        state.invalidate_context(
            reason="file_changed",
            paths=[normalized],
            details={"state_change": f"File changed: {normalized}"},
        )
    except Exception:
        pass
    state.touch()


def _repair_cycle_session_id_failure(
    *,
    supplied_id: str,
    path: str,
    state: LoopState | None,
) -> dict[str, Any]:
    from .fs_write_session_policy import _write_session_resume_metadata
    from .common import fail

    session = getattr(state, "write_session", None) if state is not None else None
    if session is not None and str(getattr(session, "status", "")).strip().lower() == "complete":
        session = None

    metadata: dict[str, Any] = {
        "path": path,
        "error_kind": "repair_cycle_used_as_write_session_id",
        "supplied_write_session_id": supplied_id,
        "system_repair_cycle_id": str(getattr(state, "repair_cycle_id", "") or "").strip(),
    }
    if session is not None:
        metadata["active_write_session_id"] = str(getattr(session, "write_session_id", "") or "").strip()
        metadata["next_required_tool"] = _write_session_resume_metadata(session, path=path)
        return fail(
            f"`{supplied_id}` looks like a system repair cycle ID, not a `write_session_id`. "
            f"Resume the active Write Session with `write_session_id='{session.write_session_id}'` for `{session.write_target_path}`.",
            metadata=metadata,
        )

    return fail(
        f"`{supplied_id}` looks like a system repair cycle ID, not a `write_session_id`. "
        "There is no active write session to resume; omit `write_session_id` for a direct write or inspect `loop_status` for the current blocker.",
        metadata=metadata,
    )
