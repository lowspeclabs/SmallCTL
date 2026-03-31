from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .common import fail, ok


def _resolve(path: str, cwd: str | None = None) -> Path:
    base = Path(cwd) if cwd else Path.cwd()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def _workspace_relative_hint(path: str, cwd: str | None = None) -> str | None:
    candidate = Path(path)
    if candidate.is_absolute() or not path or path[0] not in {"\\", "/"}:
        return None
    trimmed = path.lstrip("\\/")
    if not trimmed:
        return None
    base = Path(cwd) if cwd else Path.cwd()
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


async def file_read(
    path: str,
    cwd: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
    max_bytes: int = 100_000,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    target = _resolve(path, cwd)
    if not target.exists():
        return fail(
            _missing_path_error(requested_path=path, resolved_path=target, cwd=cwd),
            metadata={"path": str(target), "requested_path": path},
        )
    try:
        raw = target.read_bytes()
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
        )
    )
    return ok(
        sliced,
        metadata={
            "path": str(target),
            "bytes": len(raw),
            "elapsed_ms": elapsed_ms,
            "requested_start_line": requested_start,
            "requested_end_line": requested_end,
            "max_bytes": max_bytes,
            "line_start": s + 1 if lines else 0,
            "line_end": e,
            "total_lines": total_lines,
            "complete_file": complete_file,
        },
    )


async def file_write(
    path: str,
    content: str,
    cwd: str | None = None,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    target = _resolve(path, cwd)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding=encoding)
    except Exception as exc:
        return fail(f"Unable to write file: {exc}")
    return ok("written", metadata={"path": str(target), "bytes": len(content.encode(encoding))})


async def file_append(
    path: str,
    content: str,
    cwd: str | None = None,
    encoding: str = "utf-8",
) -> dict[str, Any]:
    target = _resolve(path, cwd)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding=encoding) as fh:
            fh.write(content)
    except Exception as exc:
        return fail(f"Unable to append file: {exc}")
    return ok("appended", metadata={"path": str(target)})


async def file_delete(path: str, cwd: str | None = None) -> dict[str, Any]:
    target = _resolve(path, cwd)
    try:
        if not target.exists():
            return fail(f"File does not exist: {target}")
        target.unlink()
    except Exception as exc:
        return fail(f"Unable to delete file: {exc}")
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
    items = []
    for p in sorted(target.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        items.append(
            {
                "name": p.name,
                "path": str(p),
                "type": "dir" if p.is_dir() else "file",
                "size": p.stat().st_size if p.is_file() else None,
            }
        )
    return ok(items, metadata={"path": str(target), "count": len(items)})


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
