from __future__ import annotations

import re
from pathlib import Path

SYMBOL_CAPTURE_LIMIT = 24

_SYMBOL_TOKEN_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]{0,80}$")
_SYMBOL_LINE_PATTERNS = (
    re.compile(r"^\s*(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b"),
    re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*\("),
    re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)\b"),
    re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*="),
)


def extract_symbol_candidates_from_text(text: str) -> list[str]:
    symbols: list[str] = []
    if not text:
        return symbols
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(("+++", "---")):
            continue
        if line[0] in {"+", "-"} and len(line) > 1:
            line = line[1:].lstrip()
        for pattern in _SYMBOL_LINE_PATTERNS:
            match = pattern.search(line)
            if match is None:
                continue
            token = normalize_symbol_token(match.group(1))
            if token and token not in symbols:
                symbols.append(token)
    return symbols


def extract_symbol_candidates_from_path(path: str) -> list[str]:
    normalized = str(path or "").strip()
    if not normalized:
        return []
    stem = Path(normalized).stem.strip()
    if stem.lower() in {"", "__init__", "__main__", "index", "main"}:
        return []
    token = normalize_symbol_token(stem)
    return [token] if token else []


def extract_symbol_candidates_from_file(path: str, *, cwd: str) -> list[str]:
    normalized = str(path or "").strip()
    if not normalized:
        return []
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = Path(cwd or ".") / candidate
    try:
        if not candidate.exists() or not candidate.is_file():
            return []
        content = candidate.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []
    # We only need top-level-ish anchors for prompt stability; avoid scanning huge files.
    lines = content.splitlines()[:500]
    return extract_symbol_candidates_from_text("\n".join(lines))


def normalize_symbol_token(value: str) -> str:
    token = str(value or "").strip().strip("`'\".,:;()[]{}<>")
    if not token:
        return ""
    if token.lower() in {"path", "file", "content", "target", "replacement"}:
        return ""
    if not _SYMBOL_TOKEN_RE.match(token):
        return ""
    return token
