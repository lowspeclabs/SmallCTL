from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

from ..state import LoopState
from .fs_sessions import _append_unique_section, _clone_section_ranges, infer_write_session_intent


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
