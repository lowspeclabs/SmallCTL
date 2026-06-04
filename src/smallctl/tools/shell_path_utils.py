from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Any


def _safe_resolve_path(path: str | Path, *, cwd: str | None = None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = Path(cwd or Path.cwd()) / candidate
    try:
        return candidate.resolve()
    except Exception:
        return candidate.absolute()


def _is_within_path(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _same_target_path(a: str, b: str, cwd: str | None = None) -> bool:
    if a == b:
        return True
    try:
        ra = _safe_resolve_path(a, cwd=cwd)
        rb = _safe_resolve_path(b, cwd=cwd)
        return ra == rb
    except Exception:
        return False


def _target_path_aliases(target_path: str, *, cwd: str | None = None) -> list[str]:
    aliases: set[str] = set()
    raw = str(target_path or "").strip()
    if not raw:
        return []

    aliases.add(raw)
    if raw.startswith("./"):
        aliases.add(raw[2:])
    elif not raw.startswith("/"):
        aliases.add(f"./{raw}")

    try:
        base = Path(cwd).resolve() if cwd else Path.cwd().resolve()
        resolved = (Path(raw) if Path(raw).is_absolute() else (base / raw)).resolve()
        aliases.add(str(resolved))
        try:
            rel = resolved.relative_to(base)
        except Exception:
            rel = None
        if rel is not None:
            rel_str = str(rel)
            if rel_str:
                aliases.add(rel_str)
                aliases.add(f"./{rel_str}")
    except Exception:
        pass

    return [alias for alias in aliases if alias]


def _path_alias_mentioned(command: str, alias: str) -> bool:
    if not alias:
        return False
    pattern = rf"(?<![A-Za-z0-9_./-]){re.escape(alias)}(?![A-Za-z0-9_./-])"
    return bool(re.search(pattern, command))


def _token_path_candidates(token: str) -> list[str]:
    normalized = str(token or "").strip().strip("'\"`")
    if not normalized:
        return []

    while normalized.startswith("("):
        normalized = normalized[1:].strip()
    while normalized.endswith((";", "|", "&", ",", ")")):
        normalized = normalized[:-1].strip()
    if not normalized:
        return []

    candidates = [normalized]
    if "=" in normalized and not normalized.startswith("="):
        _, value = normalized.split("=", 1)
        value = value.strip().strip("'\"`")
        while value.endswith((";", "|", "&", ",", ")")):
            value = value[:-1].strip()
        if value:
            candidates.append(value)
    return candidates
