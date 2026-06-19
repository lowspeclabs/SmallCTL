from __future__ import annotations

import difflib
import hashlib
from pathlib import Path


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


def content_similarity(left: str, right: str) -> float:
    """Return a 0-1 similarity ratio between two text blocks."""
    return difflib.SequenceMatcher(
        None,
        str(left or ""),
        str(right or ""),
        autojunk=False,
    ).ratio()


def is_substantially_new_content(
    new_content: str,
    existing_content: str,
    *,
    similarity_threshold: float = 0.85,
    min_new_chars: int = 50,
) -> bool:
    """Decide whether new_content is a meaningful extension or replacement.

    Returns True when the new content is not mostly identical to the existing
    content and adds a non-trivial amount of material. This is used by chunked
    write loop guards to allow legitimate section extensions while still
    blocking identical or near-identical rewrites.
    """
    new_content = str(new_content or "")
    existing_content = str(existing_content or "")
    if not new_content:
        return False
    similarity = content_similarity(new_content, existing_content)
    if similarity >= similarity_threshold:
        return False
    if len(new_content) <= len(existing_content) and similarity >= 0.7:
        return False
    added = max(0, len(new_content) - len(existing_content))
    if added >= min_new_chars:
        return True
    # Also allow if the new content is materially rewritten (low similarity)
    # even when it is not longer.
    return similarity < 0.5
