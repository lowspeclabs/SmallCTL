from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail, ok
from .fs_loop_guard import clear_loop_guard_verification_requirement
from .fs_sessions import _record_repair_cycle_read


def _resolve(path: str, cwd: str | None = None) -> Path:
    base = Path(cwd) if cwd else Path.cwd()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _active_session_staging_path(
    state: LoopState | None,
    path: str,
    cwd: str | None = None,
) -> Path | None:
    session = getattr(state, "write_session", None) if state is not None else None
    if session is None or str(getattr(session, "status", "")).strip().lower() == "complete":
        return None
    staging_path = str(getattr(session, "write_staging_path", "") or "").strip()
    try:
        target = _resolve(path, cwd)
        session_target = _resolve(session.write_target_path, cwd)
    except Exception:
        return None
    if target != session_target:
        return None
    staging = Path(staging_path)
    target_exists = target.exists()
    try:
        first_chunk_at = float(getattr(session, "write_first_chunk_at", 0.0) or 0.0)
    except (TypeError, ValueError):
        first_chunk_at = 0.0
    has_staged_progress = bool(
        getattr(session, "write_sections_completed", None)
        or getattr(session, "write_last_staged_hash", None)
        or getattr(session, "write_section_ranges", None)
        or first_chunk_at > 0.0
    )
    if staging_path and staging.exists():
        if target_exists and staging.stat().st_size == 0 and not has_staged_progress:
            return None
        return staging
    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    if not session_id:
        return None
    try:
        from .fs_write_sessions import _session_stage_path

        expected_staging = _session_stage_path(session_id, target, cwd)
    except Exception:
        expected_staging = None
    if expected_staging is not None and expected_staging.exists():
        if target_exists and expected_staging.stat().st_size == 0 and not has_staged_progress:
            return None
        try:
            session.write_staging_path = str(expected_staging)
        except Exception:
            pass
        return expected_staging
    if target_exists and not has_staged_progress:
        return None
    try:
        from .fs import _ensure_write_session_files

        restored = _ensure_write_session_files(session, target, cwd=cwd)
    except Exception:
        return None
    if restored.exists():
        if not target_exists or restored.stat().st_size > 0:
            return restored
        try:
            if target.stat().st_size == 0:
                return restored
        except OSError:
            pass
    return None


def active_write_session_source_path(
    state: LoopState | None,
    path: str,
    cwd: str | None = None,
) -> str | None:
    staging = _active_session_staging_path(state, path, cwd)
    return str(staging) if staging is not None else None


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
    if complete_file:
        clear_loop_guard_verification_requirement(
            state,
            path=str(target),
            cwd=cwd,
        )
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
            "total_items": len(children),
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
