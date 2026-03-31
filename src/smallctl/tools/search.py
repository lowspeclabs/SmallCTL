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
    matcher = re.compile(pattern, flags) if regex else None
    results: list[dict[str, Any]] = []

    for file_path in root.rglob("*"):
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
    rx = re.compile(pattern) if regex else None
    use_glob = not regex and any(char in pattern for char in "*?[]")
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        rel = str(file_path.relative_to(root))
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
