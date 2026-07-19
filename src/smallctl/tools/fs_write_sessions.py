from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from ..state import LoopState
from ..write_session_fsm import new_write_session, record_write_session_event
from .fs_paths import _same_target_path
from .fs_sessions import (
    _append_unique_section,
    _clone_section_ranges,
    _infer_next_suggested_section,
    _is_finalization_marker,
    _looks_like_complete_html_document,
    _suggested_chunk_sections,
    infer_write_session_intent,
    new_write_session_id,
)


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
    temp_name = ""
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    fd_open = True
    try:
        if path.exists():
            try:
                os.chmod(temp_name, path.stat().st_mode)
            except OSError:
                pass
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            fd_open = False
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(path)
    except Exception:
        if fd_open:
            try:
                os.close(fd)
            except OSError:
                pass
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


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


def _resolve_patch_source(
    state: LoopState | None,
    path: str,
    *,
    cwd: str | None = None,
    encoding: str = "utf-8",
    write_session_id: str | None = None,
) -> tuple[Path, Path, Any, bool]:
    target = _resolve(path, cwd)
    session = getattr(state, "write_session", None) if state is not None else None
    if session is None or str(getattr(session, "status", "")).strip().lower() == "complete":
        return target, target, session, False

    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    if write_session_id:
        if not session_id:
            raise LookupError(
                f"No active write session found for session ID `{write_session_id}`."
            )
        if session_id != write_session_id:
            raise LookupError(
                f"Session ID mismatch: expected `{session_id}`, got `{write_session_id}`."
            )

    session_target = _resolve(str(getattr(session, "write_target_path", "") or ""), cwd)
    if target != session_target:
        if write_session_id:
            raise ValueError(
                f"Write session `{write_session_id}` is targeting `{session_target}`, not `{target}`."
            )
        return target, target, session, False

    staging_path = _ensure_write_session_files(session, target, cwd=cwd, encoding=encoding)
    return staging_path, target, session, True


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
        "checkpointed_sections": ",".join(write_session_checkpointed_sections(session)) or "-",
        "next_legal_operation": write_session_next_legal_operation(session),
        "staged_hash": staging_hash or "missing",
        "finalized": "yes" if finalized else "no",
    }


def format_write_session_status_block(snapshot: dict[str, str]) -> str:
    ordered_keys = (
        "type",
        "id",
        "mode",
        "next",
        "checkpointed_sections",
        "next_legal_operation",
        "staged_hash",
        "finalized",
    )
    parts = [f"{key}={snapshot.get(key, '')}" for key in ordered_keys]
    return "WRITE_SESSION_STATUS " + " ".join(parts)


def write_session_checkpointed_sections(session: Any) -> list[str]:
    sections: list[str] = []
    for item in getattr(session, "write_sections_completed", []) or []:
        section = str(item or "").strip()
        if section and section not in sections:
            sections.append(section)
    return sections


def write_session_next_legal_operation(session: Any) -> str:
    status = str(getattr(session, "status", "") or "open").strip().lower() or "open"
    if status == "complete":
        return "none:complete"
    next_section = str(getattr(session, "write_next_section", "") or "").strip()
    if next_section:
        return f"append_section:{next_section}"
    if bool(getattr(session, "write_sections_completed", []) or []):
        return "finalize"
    suggested = list(getattr(session, "suggested_sections", []) or [])
    first = str(suggested[0] if suggested else "").strip() or "full_file"
    return f"append_section:{first}"


def write_session_contract(session: Any) -> dict[str, Any]:
    return {
        "active_write_session_id": str(getattr(session, "write_session_id", "") or "").strip(),
        "target_path": str(getattr(session, "write_target_path", "") or "").strip(),
        "checkpointed_sections": write_session_checkpointed_sections(session),
        "next_legal_operation": write_session_next_legal_operation(session),
    }


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
        shutil.copy2(staging, target)
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


def _resolve(path: str, cwd: str | None = None) -> Path:
    base = Path(cwd) if cwd else Path.cwd()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


_IMPLICIT_SESSION_SOFT_CHUNK_CHARS = 2000
_IMPLICIT_SESSION_SMALL_HTML_CHARS = 20000


def maybe_create_implicit_write_session(
    state: LoopState | None,
    path: str,
    content: str,
    *,
    section_name: str | None,
    next_section_name: str | None,
    replace_strategy: str | None,
    cwd: str | None = None,
) -> Any | None:
    """Create a new single-path write session when chunking is implied by the call.

    Returns the created session or None if the call should remain a direct write.
    """
    if state is None or not path:
        return None

    existing = resolve_write_session_for_path(state, path, cwd)
    if existing is not None:
        return existing

    active = getattr(state, "write_session", None)
    if active is not None and str(getattr(active, "status", "") or "").strip().lower() not in {"complete"}:
        # Allow concurrent sessions only when the active session is for a different path.
        # If the active session targets the same path, resolve_write_session_for_path
        # would have returned it above, so reaching here means a different path.
        active_target = str(getattr(active, "write_target_path", "") or "").strip()
        if active_target and not _same_target_path(active_target, path, cwd):
            # Safe to create a concurrent session for a different path.
            pass
        else:
            # Defer concurrent sessions: if another path has an active session, keep
            # this call as a direct write for now.
            return None

    strategy = str(replace_strategy or "auto").strip().lower()
    normalized_section = str(section_name or "").strip()
    raw_next = str(next_section_name or "").strip()
    has_next = bool(raw_next) and not _is_finalization_marker(raw_next)
    content_len = len(str(content or ""))

    if strategy == "overwrite" and not has_next:
        # An explicit one-shot overwrite with no declared follow-up section is a
        # direct write; do not trap it in a staged session that may never finalize.
        return None

    # A present section label (other than a full-file marker) implies chunked authoring.
    implies_chunking = bool(normalized_section) and normalized_section.lower() not in {
        "full_file",
        "final_content",
        "complete_file",
        "entire_file",
        "final_file",
    }

    # A next-section hint always implies chunked authoring.
    if has_next:
        implies_chunking = True

    target = _resolve(path, cwd)
    is_html = target.suffix.lower() in {".html", ".htm"}
    looks_complete_html = _looks_like_complete_html_document(content)

    if not implies_chunking and is_html:
        # Small complete HTML artifacts stay direct writes.
        if content_len < _IMPLICIT_SESSION_SMALL_HTML_CHARS and looks_complete_html:
            return None
        if content_len < _IMPLICIT_SESSION_SMALL_HTML_CHARS and strategy == "overwrite":
            return None

    if not implies_chunking and content_len < _IMPLICIT_SESSION_SOFT_CHUNK_CHARS:
        return None

    if strategy == "overwrite" and not implies_chunking and not looks_complete_html:
        # A plain overwrite of a large non-HTML payload is likely a direct write.
        return None

    intent = infer_write_session_intent(path, cwd)
    suggestions = _suggested_chunk_sections(path)
    first_section = normalized_section or (suggestions[0] if suggestions else "full_file")
    session_next = str(next_section_name or "").strip()
    if _is_finalization_marker(session_next):
        session_next = ""
    session = new_write_session(
        session_id=new_write_session_id(),
        target_path=path,
        intent=intent,
        mode="chunked_author",
        suggested_sections=suggestions,
        next_section=session_next,
    )
    session.write_current_section = first_section
    if not session.write_next_section and suggestions:
        # If the caller supplied a section name, infer the next suggestion after it.
        inferred = _infer_next_suggested_section(session, first_section)
        session.write_next_section = inferred
    _store_active_write_session(state, session)
    record_write_session_event(
        state,
        event="implicit_session_created",
        session=session,
        details={
            "path": str(target),
            "trigger": "section_name" if normalized_section else ("next_section_name" if has_next else "payload_size"),
            "first_section": first_section,
        },
    )
    return session


def _canonical_session_key(path: str, cwd: str | None = None) -> str:
    return str(_resolve(path, cwd))


def _store_active_write_session(state: LoopState, session: Any) -> None:
    """Store session in the path map and keep state.write_session as the alias."""
    state.write_session = session
    target = str(getattr(session, "write_target_path", "") or "").strip()
    if not target:
        return
    key = _canonical_session_key(target, getattr(state, "cwd", None))
    state.active_write_sessions_by_path[key] = session


def _remove_active_write_session(state: LoopState | None, session: Any | None) -> None:
    if state is None or session is None:
        return
    target = str(getattr(session, "write_target_path", "") or "").strip()
    if target:
        key = _canonical_session_key(target, getattr(state, "cwd", None))
        state.active_write_sessions_by_path.pop(key, None)
    if getattr(state, "write_session", None) is session:
        state.write_session = None


def _set_active_write_session_alias(state: LoopState | None, session: Any | None) -> None:
    """Update state.write_session to point to the most recently touched session.

    If the provided session is None, try to pick a non-terminal session from the
    active map as a fallback so the alias never silently points to a stale object.
    """
    if state is None:
        return
    if session is not None:
        state.write_session = session
        return
    candidates = [
        s
        for s in (getattr(state, "active_write_sessions_by_path", {}) or {}).values()
        if s is not None and str(getattr(s, "status", "") or "").strip().lower() not in {"complete"}
    ]
    if candidates:
        # Prefer the most recently started non-terminal session.
        candidates.sort(key=lambda s: float(getattr(s, "write_session_started_at", 0.0) or 0.0), reverse=True)
        state.write_session = candidates[0]
    else:
        state.write_session = None


def resolve_write_session_for_path(
    state: LoopState | None,
    path: str,
    cwd: str | None = None,
) -> Any | None:
    """Return the active non-terminal write session whose canonical target matches path.

    First consults the durable multi-session map, then falls back to
    ``state.write_session`` for compatibility with checkpoints and tests that
    have not yet populated the map.
    """
    if state is None or path is None:
        return None

    key = _canonical_session_key(path, cwd)
    active_map = getattr(state, "active_write_sessions_by_path", None)
    if isinstance(active_map, dict):
        session = active_map.get(key)
        if session is not None and str(getattr(session, "status", "") or "").strip().lower() not in {"complete"}:
            _set_active_write_session_alias(state, session)
            return session

    session = getattr(state, "write_session", None)
    if session is None:
        return None
    status = str(getattr(session, "status", "") or "").strip().lower()
    if status in {"complete"}:
        return None
    target = str(getattr(session, "write_target_path", "") or "").strip()
    if not target:
        return None
    if _same_target_path(target, path, cwd):
        # Populate the path map from the legacy alias on first use.
        _store_active_write_session(state, session)
        return session
    return None
