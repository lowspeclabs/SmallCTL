from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any


def resolve_path(path: str, cwd: str | None = None) -> str:
    base = Path(cwd) if cwd else Path.cwd()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        return str(candidate.resolve())
    except Exception:
        return str(candidate)


def content_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()


def count_added_lines(before: str, after: str) -> int:
    added = 0
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm=""):
        if line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
    return added


def tail_excerpt(text: str, *, tail_lines: int) -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max(1, int(tail_lines)) :])
