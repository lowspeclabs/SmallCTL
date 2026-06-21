from __future__ import annotations

from pathlib import Path
from typing import Any


def _same_target_path(left: Any, right: Any, cwd: str | None = None) -> bool:
    """Return True when two path strings resolve to the same filesystem path."""
    if left == right:
        return True
    try:
        base = Path(cwd) if cwd else Path.cwd()
        left_path = Path(left)
        if not left_path.is_absolute():
            left_path = base / left_path
        right_path = Path(right)
        if not right_path.is_absolute():
            right_path = base / right_path
        return left_path.resolve() == right_path.resolve()
    except Exception:
        return str(left) == str(right)
