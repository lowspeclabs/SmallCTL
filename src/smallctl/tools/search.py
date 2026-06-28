from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

from .common import fail, ok


async def grep(
    pattern: str,
    path: str = ".",
    regex: bool = True,
    case_sensitive: bool = False,
    max_results: int = 200,
) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.exists():
        return fail(f"Path does not exist: {root}")
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        matcher = re.compile(pattern, flags) if regex else None
    except re.error as exc:
        return fail(
            f"Invalid regex: {exc}",
            metadata={"pattern": pattern, "regex": True, "error_kind": "invalid_regex"},
        )
    results: list[dict[str, Any]] = []

    candidates = [root] if root.is_file() else root.rglob("*")
    for file_path in candidates:
        if not file_path.is_file():
            continue
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for idx, line in enumerate(lines, start=1):
            matched = bool(matcher.search(line)) if matcher else pattern in line
            if not matched:
                continue
            results.append({"path": str(file_path), "line": idx, "text": line})
            if len(results) >= max_results:
                return ok(results, metadata={"truncated": True, "count": len(results)})
    return ok(results, metadata={"truncated": False, "count": len(results)})


async def find_files(
    pattern: str,
    path: str = ".",
    regex: bool = False,
    max_results: int = 500,
) -> dict[str, Any]:
    root = Path(path).resolve()
    if not root.exists():
        return fail(f"Path does not exist: {root}")
    results: list[str] = []
    try:
        rx = re.compile(pattern) if regex else None
    except re.error as exc:
        return fail(
            f"Invalid regex: {exc}",
            metadata={"pattern": pattern, "regex": True, "error_kind": "invalid_regex"},
        )
    use_glob = not regex and any(char in pattern for char in "*?[]")
    candidates = [root] if root.is_file() else root.rglob("*")
    for file_path in candidates:
        if not file_path.is_file():
            continue
        rel = file_path.name if root.is_file() else str(file_path.relative_to(root))
        if rx:
            matched = bool(rx.search(rel))
        elif use_glob:
            matched = fnmatch.fnmatch(rel, pattern)
        else:
            matched = pattern in rel
        if matched:
            results.append(str(file_path))
            if len(results) >= max_results:
                return ok(results, metadata={"truncated": True, "count": len(results)})
    return ok(results, metadata={"truncated": False, "count": len(results)})
