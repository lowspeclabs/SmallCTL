from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any

from ..state import LoopState
from ..write_session_fsm import record_write_session_event
from .common import fail, ok

FILE_MUTATING_TOOLS = {"file_write", "file_append", "file_delete"}


def is_file_mutating_tool(tool_name: str) -> bool:
    return str(tool_name or "").strip() in FILE_MUTATING_TOOLS


def _resolve(path: str, cwd: str | None = None) -> Path:
    base = Path(cwd) if cwd else Path.cwd()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _normalized_path_str(path: Path) -> str:
    try:
        return str(path.resolve())
    except Exception:
        return str(path)


def _looks_like_system_repair_cycle_id(value: str | None) -> bool:
    return str(value or "").strip().lower().startswith("repair-")


def _suspicious_temp_root_details(path: str) -> dict[str, str] | None:
    raw = str(path or "").strip()
    normalized = raw.replace("\\", "/")
    if not raw or (normalized != "/temp" and not normalized.startswith("/temp/")):
        return None

    suffix = normalized[len("/temp"):]
    relative_suffix = suffix.lstrip("/")
    return {
        "path": raw,
        "suggested_tmp_path": f"/tmp{suffix}",
        "suggested_relative_path": "./temp" if not relative_suffix else f"./temp/{relative_suffix}",
    }


def _guard_suspicious_temp_root_path(path: str) -> dict[str, Any] | None:
    details = _suspicious_temp_root_details(path)
    if details is None:
        return None
    return fail(
        f"Path `{details['path']}` points at the root-level `/temp` directory, which is usually a typo in this harness. "
        f"Use `{details['suggested_tmp_path']}` for a system temp file or `{details['suggested_relative_path']}` for a workspace-local temp path instead.",
        metadata={
            "path": details["path"],
            "error_kind": "suspicious_temp_root_path",
            "suggested_tmp_path": details["suggested_tmp_path"],
            "suggested_relative_path": details["suggested_relative_path"],
        },
    )


def _write_session_resume_metadata(session: Any, *, path: str) -> dict[str, Any]:
    section_name = str(
        getattr(session, "write_next_section", "")
        or getattr(session, "write_current_section", "")
        or "imports"
    ).strip() or "imports"
    return {
        "tool_name": "file_write",
        "required_fields": ["path", "content", "write_session_id", "section_name"],
        "required_arguments": {
            "path": str(getattr(session, "write_target_path", "") or path or "").strip(),
            "write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
            "section_name": section_name,
        },
        "optional_fields": ["next_section_name"],
    }


def _repair_cycle_session_id_failure(
    *,
    supplied_id: str,
    path: str,
    state: LoopState | None,
) -> dict[str, Any]:
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
    normalized = _normalized_path_str(path)
    if normalized not in reads:
        reads.append(normalized)
        state.scratchpad["_repair_cycle_reads"] = reads


def _repair_cycle_allows_patch(state: LoopState | None, path: Path) -> bool:
    if state is None or not state.repair_cycle_id:
        return True
    reads = set(_repair_cycle_reads(state))
    return _normalized_path_str(path) in reads


def _record_file_change(state: LoopState | None, path: Path) -> None:
    if state is None:
        return
    normalized = _normalized_path_str(path)
    if normalized in state.files_changed_this_cycle:
        _mark_repeat_patch(state)
    changed = [item for item in state.files_changed_this_cycle if item != normalized]
    changed.append(normalized)
    state.files_changed_this_cycle = changed[-12:]
    state.touch()


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


def _same_target_path(left: str, right: str, cwd: str | None = None) -> bool:
    try:
        return _resolve(left, cwd) == _resolve(right, cwd)
    except Exception:
        return str(left) == str(right)


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
    import uuid

    return f"ws_{uuid.uuid4().hex[:6]}"


def infer_write_session_intent(path: str, cwd: str | None = None) -> str:
    try:
        return "patch_existing" if _resolve(path, cwd).exists() else "replace_file"
    except Exception:
        return "replace_file"


def _write_session_dir(cwd: str | None = None) -> Path:
    base = Path(cwd) if cwd else Path.cwd()
    return (base / ".smallctl" / "write_sessions").resolve()


def _session_file_stem(session_id: str, target: Path, label: str) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in (target.stem or "target"))
    return f"{session_id}__{stem}__{label}"


def _session_stage_path(session_id: str, target: Path, cwd: str | None = None) -> Path:
    suffix = target.suffix or ".txt"
    return _write_session_dir(cwd) / f"{_session_file_stem(session_id, target, 'stage')}{suffix}"


def _session_original_snapshot_path(session_id: str, target: Path, cwd: str | None = None) -> Path:
    suffix = target.suffix or ".txt"
    return _write_session_dir(cwd) / f"{_session_file_stem(session_id, target, 'original')}{suffix}"


def _session_attempt_snapshot_path(session_id: str, target: Path, cwd: str | None = None) -> Path:
    suffix = target.suffix or ".txt"
    return _write_session_dir(cwd) / f"{_session_file_stem(session_id, target, 'attempt')}{suffix}"


def _read_text_file(path: Path, *, encoding: str = "utf-8") -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding=encoding)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _write_text_file(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def _active_session_staging_path(
    state: LoopState | None,
    path: str,
    cwd: str | None = None,
) -> Path | None:
    session = getattr(state, "write_session", None) if state is not None else None
    if session is None or str(getattr(session, "status", "")).strip().lower() == "complete":
        return None
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    if not staging_path:
        return None
    try:
        target = _resolve(path, cwd)
        session_target = _resolve(session.write_target_path, cwd)
    except Exception:
        return None
    if target != session_target:
        return None
    staging = Path(staging_path)
    return staging if staging.exists() else None


def _ensure_write_session_files(
    session: Any,
    target: Path,
    *,
    cwd: str | None = None,
    encoding: str = "utf-8",
) -> Path:
    if str(getattr(session, "write_session_intent", "") or "").strip().lower() not in {"replace_file", "patch_existing"}:
        session.write_session_intent = infer_write_session_intent(str(target), cwd)

    storage_dir = _write_session_dir(cwd)
    storage_dir.mkdir(parents=True, exist_ok=True)

    target_exists = target.exists()
    if target_exists or getattr(session, "write_target_existed_at_start", False):
        session.write_target_existed_at_start = True

    if not str(getattr(session, "write_original_snapshot_path", "") or "").strip():
        session.write_original_snapshot_path = str(
            _session_original_snapshot_path(session.write_session_id, target, cwd)
        )
    original_snapshot = Path(session.write_original_snapshot_path)
    if target_exists and not original_snapshot.exists():
        _write_text_file(original_snapshot, _read_text_file(target, encoding=encoding), encoding=encoding)

    if session.write_session_intent == "patch_existing" and not (target_exists or original_snapshot.exists()):
        session.write_session_intent = "replace_file"

    if not str(getattr(session, "write_staging_path", "") or "").strip():
        session.write_staging_path = str(_session_stage_path(session.write_session_id, target, cwd))
    staging_path = Path(session.write_staging_path)
    if not staging_path.exists():
        seed_content = ""
        if session.write_session_intent == "patch_existing":
            if original_snapshot.exists():
                seed_content = _read_text_file(original_snapshot, encoding=encoding)
            elif target_exists:
                seed_content = _read_text_file(target, encoding=encoding)
        _write_text_file(staging_path, seed_content, encoding=encoding)

    if not str(getattr(session, "write_last_attempt_snapshot_path", "") or "").strip():
        session.write_last_attempt_snapshot_path = str(
            _session_attempt_snapshot_path(session.write_session_id, target, cwd)
        )
    return staging_path


def active_write_session_source_path(
    state: LoopState | None,
    path: str,
    cwd: str | None = None,
) -> str | None:
    staging = _active_session_staging_path(state, path, cwd)
    return str(staging) if staging is not None else None


def write_session_verify_path(session: Any, cwd: str | None = None) -> str:
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    if staging_path and Path(staging_path).exists():
        return staging_path
    return str(_resolve(session.write_target_path, cwd))


def write_session_status_snapshot(
    session: Any,
    *,
    cwd: str | None = None,
    finalized: bool = False,
    encoding: str = "utf-8",
) -> dict[str, str]:
    staging_hash = str(getattr(session, "write_last_staged_hash", "") or "").strip()
    if not staging_hash:
        staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
        if staging_path and Path(staging_path).exists():
            staging_hash = _content_hash(_read_text_file(Path(staging_path), encoding=encoding))
        elif finalized:
            target = _resolve(session.write_target_path, cwd)
            if target.exists():
                staging_hash = _content_hash(_read_text_file(target, encoding=encoding))
    next_section = str(getattr(session, "write_next_section", "") or "").strip() or "-"
    return {
        "type": "write_session",
        "id": str(getattr(session, "write_session_id", "") or "").strip(),
        "mode": str(getattr(session, "write_session_mode", "") or "").strip() or "chunked_author",
        "next": next_section,
        "staged_hash": staging_hash or "missing",
        "finalized": "yes" if finalized else "no",
    }


def format_write_session_status_block(snapshot: dict[str, str]) -> str:
    ordered_keys = ("type", "id", "mode", "next", "staged_hash", "finalized")
    parts = [f"{key}={snapshot.get(key, '')}" for key in ordered_keys]
    return "WRITE_SESSION_STATUS " + " ".join(parts)


def promote_write_session_target(
    session: Any,
    *,
    cwd: str | None = None,
) -> tuple[bool, str]:
    target = _resolve(session.write_target_path, cwd)
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    if not staging_path:
        return False, f"Write session `{session.write_session_id}` has no staging file to finalize."
    staging = Path(staging_path)
    if not staging.exists():
        return False, f"Staging file is missing for write session `{session.write_session_id}`."
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(target)
    except Exception as exc:
        return False, f"Unable to finalize staged write for `{session.write_target_path}`: {exc}"
    return True, str(target)


def restore_write_session_snapshot(
    session: Any,
    *,
    cwd: str | None = None,
    encoding: str = "utf-8",
) -> tuple[bool, str]:
    snapshot_path = str(getattr(session, "write_last_attempt_snapshot_path", "") or "").strip()
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    if not snapshot_path or not staging_path:
        return False, "Write session snapshot is unavailable."
    snapshot = Path(snapshot_path)
    staging = Path(staging_path)
    if not snapshot.exists():
        return False, f"Write session snapshot is missing: {snapshot}"
    try:
        restored_content = _read_text_file(snapshot, encoding=encoding)
        _write_text_file(staging, restored_content, encoding=encoding)
    except Exception as exc:
        return False, f"Unable to restore staged write for `{session.write_target_path}`: {exc}"
    session.write_last_staged_hash = _content_hash(restored_content)
    session.write_sections_completed = list(getattr(session, "write_last_attempt_sections", []) or [])
    session.write_section_ranges = _clone_section_ranges(
        getattr(session, "write_last_attempt_ranges", {}) or {}
    )
    return True, str(staging)


def _replace_known_section(
    staged_content: str,
    section_ranges: dict[str, dict[str, int]],
    section_name: str,
    new_content: str,
) -> tuple[str, dict[str, dict[str, int]]]:
    current_range = section_ranges.get(section_name) or {}
    start = int(current_range.get("start", 0))
    end = int(current_range.get("end", start))
    updated_content = staged_content[:start] + new_content + staged_content[end:]
    delta = len(new_content) - (end - start)
    updated_ranges: dict[str, dict[str, int]] = {}
    for name, item in section_ranges.items():
        if name == section_name or not isinstance(item, dict):
            continue
        item_start = int(item.get("start", 0))
        item_end = int(item.get("end", item_start))
        if item_start >= end:
            item_start += delta
            item_end += delta
        updated_ranges[name] = {"start": item_start, "end": item_end}
    updated_ranges[section_name] = {"start": start, "end": start + len(new_content)}
    return updated_content, updated_ranges


def _workspace_relative_hint(path: str, cwd: str | None = None) -> str | None:
    raw = str(path or "").strip()
    if not raw:
        return None

    candidate = Path(raw)
    base = Path(cwd) if cwd else Path.cwd()

    if candidate.is_absolute():
        try:
            relative = candidate.resolve().relative_to(base.resolve())
        except Exception:
            trimmed = raw.lstrip("\\/")
            if not trimmed:
                return None
            workspace_candidate = (base / Path(trimmed)).resolve()
            if not (workspace_candidate.exists() or workspace_candidate.parent.exists()):
                return None
            try:
                relative = workspace_candidate.relative_to(base.resolve())
            except Exception:
                return None
        return "." if str(relative) == "." else f"./{relative}"

    if raw[0] not in {"\\", "/"}:
        return None

    trimmed = raw.lstrip("\\/")
    if not trimmed:
        return None
    suggested = (base / Path(trimmed)).resolve()
    try:
        relative = suggested.relative_to(base.resolve())
    except ValueError:
        return None
    return str(relative)


def _missing_path_error(*, requested_path: str, resolved_path: Path, cwd: str | None = None) -> str:
    message = f"File does not exist: {resolved_path}"
    suggestion = _workspace_relative_hint(requested_path, cwd)
    if suggestion:
        message = (
            f"{message}. The requested path {requested_path!r} was treated as absolute. "
            f"If you meant a workspace-relative path, retry with {suggestion!r}."
        )
    return message


def _missing_dir_error(*, requested_path: str, resolved_path: Path, cwd: str | None = None) -> str:
    message = f"Directory does not exist: {resolved_path}"
    suggestion = _workspace_relative_hint(requested_path, cwd)
    if suggestion:
        message = (
            f"{message}. The requested path {requested_path!r} was treated as absolute. "
            f"If you meant a workspace-relative path, retry with {suggestion!r}."
        )
    return message


def _build_dir_tree(
    path: Path,
    *,
    depth: int,
    max_depth: int,
    max_children: int,
    remaining_nodes: list[int],
) -> dict[str, Any]:
    node = {
        "name": path.name or str(path),
        "path": str(path),
        "type": "dir" if path.is_dir() else "file",
        "size": path.stat().st_size if path.is_file() else None,
    }
    if not path.is_dir() or depth >= max_depth or remaining_nodes[0] <= 0:
        return node

    try:
        children = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    except Exception:
        return node

    preview_children: list[dict[str, Any]] = []
    child_limit = min(max_children, len(children))
    for child in children[:child_limit]:
        if remaining_nodes[0] <= 0:
            break
        remaining_nodes[0] -= 1
        preview_children.append(
            _build_dir_tree(
                child,
                depth=depth + 1,
                max_depth=max_depth,
                max_children=max_children,
                remaining_nodes=remaining_nodes,
            )
        )

    if preview_children:
        node["children"] = preview_children
        node["children_count"] = len(children)
        if len(children) > len(preview_children):
            node["children_truncated"] = True
    return node


async def file_read(
    path: str,
    cwd: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    max_bytes: int = 100_000,
    state: LoopState | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    target = _resolve(path, cwd)
    source = _active_session_staging_path(state, path, cwd) or target
    if not source.exists():
        return fail(
            _missing_path_error(requested_path=path, resolved_path=target, cwd=cwd),
            metadata={"path": str(target), "requested_path": path},
        )
    try:
        source_size = source.stat().st_size
        raw = source.read_bytes()
        raw = raw[:max_bytes]
        text = raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return fail(f"Unable to read file: {exc}")

    lines = text.splitlines()
    total_lines = len(lines)
    requested_start = start_line
    requested_end = end_line
    if start_line is not None and end_line is not None and end_line < start_line:
        return fail(
            f"Invalid line range: start_line ({start_line}) cannot be greater than end_line ({end_line})",
            metadata={
                "path": str(target),
                "bytes": len(raw),
                "requested_start_line": requested_start,
                "requested_end_line": requested_end,
                "max_bytes": max_bytes,
                "total_lines": total_lines,
            },
        )
    s = 0 if start_line is None else max(start_line - 1, 0)
    e = len(lines) if end_line is None else min(end_line, len(lines))
    sliced = "\n".join(lines[s:e])
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    complete_file = (
        total_lines == 0
        or (
            (requested_start is None or requested_start <= 1)
            and (requested_end is None or requested_end >= total_lines)
            and source_size <= max_bytes
        )
    )
    truncated = source_size > max_bytes
    _record_repair_cycle_read(state, target)
    if source != target:
        _record_repair_cycle_read(state, source)
    return ok(
        sliced,
        metadata={
            "path": str(target),
            "source_path": str(source),
            "bytes": len(raw),
            "elapsed_ms": elapsed_ms,
            "requested_start_line": requested_start,
            "requested_end_line": requested_end,
            "max_bytes": max_bytes,
            "line_start": s + 1 if lines else 0,
            "line_end": e,
            "total_lines": total_lines,
            "complete_file": complete_file,
            "truncated": truncated,
            "read_from_staging": source != target,
        },
    )


async def file_write(
    path: str,
    content: str,
    cwd: str | None = None,
    encoding: str = "utf-8",
    state: LoopState | None = None,
    session_id: str | None = None,
    write_session_id: str | None = None,
    section_name: str | None = None,
    section_id: str | None = None,
    section_role: str | None = None,
    next_section_name: str | None = None,
    replace_strategy: str | None = None,
    expected_followup_verifier: str | None = None,
) -> dict[str, Any]:
    suspicious_path = _guard_suspicious_temp_root_path(path)
    if suspicious_path is not None:
        return suspicious_path

    target = _resolve(path, cwd)
    if not write_session_id and session_id:
        write_session_id = session_id

    if not content and not write_session_id:
        return fail(
            "File write received empty content. If you intended to create an empty file, please use `\"\"` (or use shell commands like touch). "
            "If you forgot the content, please retry with the full content payload."
        )
    
    if write_session_id:
        if _looks_like_system_repair_cycle_id(write_session_id):
            return _repair_cycle_session_id_failure(
                supplied_id=str(write_session_id or "").strip(),
                path=path,
                state=state,
            )
        if state is None or state.write_session is None:
            return fail(f"No active write session found for session ID `{write_session_id}`. Start a session or write directly.")
        session = state.write_session
        if session.write_session_id != write_session_id:
            return fail(f"Session ID mismatch: expected `{session.write_session_id}`, got `{write_session_id}`.")
        if not _same_target_path(session.write_target_path, path, cwd):
            if not session.write_sections_completed:
                # Allow path correction if no sections completed yet
                parent_logger = getattr(state, "log", logging.getLogger("smallctl.tools.fs"))
                parent_logger.info(f"Correcting session target path from `{session.write_target_path}` to `{path}`")
                session.write_target_path = path
                # Reset snapshot/staging paths so they are re-initialized for the new target
                session.write_staging_path = ""
                session.write_original_snapshot_path = ""
                session.write_last_attempt_snapshot_path = ""
                session.write_target_existed_at_start = False
            else:
                return fail(f"Session target path mismatch: expected `{session.write_target_path}`, got `{path}`.")

        normalized_section_name = _normalize_section_name(section_name, section_id)
        normalized_next_section = str(next_section_name or "").strip()
        strategy = _normalize_replace_strategy(replace_strategy)
        staging_path = _ensure_write_session_files(
            session,
            target,
            cwd=cwd,
            encoding=encoding,
        )
        staged_content = _read_text_file(staging_path, encoding=encoding)
        previous_sections = list(session.write_sections_completed)
        previous_ranges = _clone_section_ranges(session.write_section_ranges)
        _write_text_file(
            Path(session.write_last_attempt_snapshot_path),
            staged_content,
            encoding=encoding,
        )
        session.write_last_attempt_sections = previous_sections
        session.write_last_attempt_ranges = previous_ranges

        current_range = previous_ranges.get(normalized_section_name)
        if current_range:
            updated_content, updated_ranges = _replace_known_section(
                staged_content,
                previous_ranges,
                normalized_section_name,
                content,
            )
            effective_strategy = "replace_section"
        elif strategy == "overwrite" and not previous_sections:
            updated_content = content
            updated_ranges = {
                normalized_section_name: {"start": 0, "end": len(content)}
            }
            effective_strategy = "overwrite"
        else:
            if (
                session.write_session_intent == "patch_existing"
                and not previous_sections
                and strategy == "auto"
            ):
                return fail(
                    "Patch-existing write sessions require an explicit `replace_strategy` of "
                    "`overwrite` to replace the file or `append` to add a new tracked section."
                )
            start = len(staged_content)
            updated_content = staged_content + content
            updated_ranges = _clone_section_ranges(previous_ranges)
            updated_ranges[normalized_section_name] = {
                "start": start,
                "end": start + len(content),
            }
            effective_strategy = "append"
        final_chunk = not normalized_next_section and _write_session_can_finalize(session)

        try:
            _write_text_file(staging_path, updated_content, encoding=encoding)
        except Exception as exc:
            return fail(f"Unable to write section `{normalized_section_name}` to `{path}`: {exc}")

        session.write_last_staged_hash = _content_hash(updated_content)
        session.write_section_ranges = updated_ranges
        session.write_current_section = normalized_section_name
        session.write_next_section = normalized_next_section
        section_added = _append_unique_section(session.write_sections_completed, normalized_section_name)
        if section_added and session.write_first_chunk_at <= 0:
            session.write_first_chunk_at = time.time()
            record_write_session_event(
                state,
                event="first_chunk_written",
                session=session,
                details={"section_name": normalized_section_name},
            )

        _record_file_change(state, target)

        status_snapshot = write_session_status_snapshot(
            session,
            cwd=cwd,
            finalized=False,
            encoding=encoding,
        )
        status_block = format_write_session_status_block(status_snapshot)

        msg = f"Section `{normalized_section_name}` written to `{path}`."
        if normalized_next_section:
            msg += f" Waiting for next section: `{normalized_next_section}`."
        elif final_chunk:
            msg += " Final section candidate recorded. Awaiting verifier."
        else:
            msg += " Session remains active for local repair."
        msg += f" Staged copy: `{staging_path}`."
        msg += f"\n{status_block}"
            
        return ok(msg, metadata={
            "path": str(target),
            "bytes": len(content.encode(encoding)),
            "staging_path": str(staging_path),
            "write_session_intent": session.write_session_intent,
            "write_session_handle_type": "write_session",
            "write_session_id": write_session_id,
            "write_current_section": normalized_section_name,
            "write_next_section": normalized_next_section,
            "write_sections_completed": session.write_sections_completed,
            "write_section_ranges": session.write_section_ranges,
            "write_session_staged_hash": session.write_last_staged_hash,
            "write_session_status_block": status_block,
            "write_session_finalized": False,
            "write_session_final_chunk": final_chunk,
            "section_name": normalized_section_name,
            "section_id": str(section_id or normalized_section_name),
            "section_role": str(section_role or ""),
            "section_added": section_added,
            "replace_strategy": effective_strategy,
            "expected_followup_verifier": str(expected_followup_verifier or ""),
            "staged_only": True,
        })

    if not _repair_cycle_allows_patch(state, target):
        _mark_repeat_patch(state)
        return fail(
            "Repair cycle requires reading the target file before patching it again.",
            metadata={
                "path": str(target),
                "system_repair_cycle_id": getattr(state, "repair_cycle_id", ""),
                "required_read_paths": _repair_cycle_reads(state),
            },
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
    except Exception as exc:
        return fail(f"Unable to write file: {exc}")
    _record_file_change(state, target)
    return ok("written", metadata={"path": str(target), "bytes": len(content.encode(encoding))})


async def file_append(
    path: str,
    content: str,
    cwd: str | None = None,
    encoding: str = "utf-8",
    state: LoopState | None = None,
    session_id: str | None = None,
    write_session_id: str | None = None,
    section_name: str | None = None,
    section_id: str | None = None,
    section_role: str | None = None,
    next_section_name: str | None = None,
    replace_strategy: str | None = None,
    expected_followup_verifier: str | None = None,
) -> dict[str, Any]:
    suspicious_path = _guard_suspicious_temp_root_path(path)
    if suspicious_path is not None:
        return suspicious_path

    if not write_session_id and session_id:
        write_session_id = session_id
    if write_session_id:
        return await file_write(
            path=path,
            content=content,
            cwd=cwd,
            encoding=encoding,
            state=state,
            session_id=session_id,
            write_session_id=write_session_id,
            section_name=section_name,
            section_id=section_id,
            section_role=section_role,
            next_section_name=next_section_name,
            replace_strategy=replace_strategy or "append",
            expected_followup_verifier=expected_followup_verifier,
        )
    target = _resolve(path, cwd)
    if not _repair_cycle_allows_patch(state, target):
        _mark_repeat_patch(state)
        return fail(
            "Repair cycle requires reading the target file before patching it again.",
            metadata={
                "path": str(target),
                "system_repair_cycle_id": getattr(state, "repair_cycle_id", ""),
                "required_read_paths": _repair_cycle_reads(state),
            },
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding=encoding) as fh:
            fh.write(content)
    except Exception as exc:
        return fail(f"Unable to append file: {exc}")
    _record_file_change(state, target)
    return ok("appended", metadata={"path": str(target)})


async def file_delete(
    path: str,
    cwd: str | None = None,
    state: LoopState | None = None,
) -> dict[str, Any]:
    suspicious_path = _guard_suspicious_temp_root_path(path)
    if suspicious_path is not None:
        return suspicious_path

    target = _resolve(path, cwd)
    if not _repair_cycle_allows_patch(state, target):
        _mark_repeat_patch(state)
        return fail(
            "Repair cycle requires reading the target file before patching it again.",
            metadata={
                "path": str(target),
                "system_repair_cycle_id": getattr(state, "repair_cycle_id", ""),
                "required_read_paths": _repair_cycle_reads(state),
            },
        )
    try:
        if not target.exists():
            return fail(f"File does not exist: {target}")
        target.unlink()
    except Exception as exc:
        return fail(f"Unable to delete file: {exc}")
    _record_file_change(state, target)
    return ok("deleted", metadata={"path": str(target)})


async def dir_list(path: str = ".", cwd: str | None = None) -> dict[str, Any]:
    target = _resolve(path, cwd)
    if not target.exists():
        return fail(
            _missing_dir_error(requested_path=path, resolved_path=target, cwd=cwd),
            metadata={"path": str(target), "requested_path": path},
    )
    if not target.is_dir():
        return fail(f"Path is not a directory: {target}")
    items: list[dict[str, Any]] = []
    root_limit = 120
    children = sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
    for p in children[:root_limit]:
        items.append(
            _build_dir_tree(
                p,
                depth=0,
                max_depth=2,
                max_children=8,
                remaining_nodes=[root_limit],
            )
        )
    return ok(
        items,
        metadata={
            "path": str(target),
            "count": len(items),
            "truncated": len(children) > len(items),
            "tree_depth": 2,
            "tree_children_limit": 8,
        },
    )


async def dir_tree(
    path: str = ".",
    cwd: str | None = None,
    max_depth: int = 3,
    max_entries: int = 500,
) -> dict[str, Any]:
    root = _resolve(path, cwd)
    if not root.exists() or not root.is_dir():
        return fail(f"Invalid directory: {root}")

    entries: list[dict[str, Any]] = []
    root_depth = len(root.parts)
    for p in root.rglob("*"):
        depth = len(p.parts) - root_depth
        if depth > max_depth:
            continue
        entries.append(
            {
                "path": str(p),
                "relative": str(p.relative_to(root)),
                "depth": depth,
                "type": "dir" if p.is_dir() else "file",
            }
        )
        if len(entries) >= max_entries:
            break
    return ok(entries, metadata={"path": str(root), "count": len(entries), "truncated": len(entries) >= max_entries})
