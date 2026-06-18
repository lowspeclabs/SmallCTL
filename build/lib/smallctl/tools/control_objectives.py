from __future__ import annotations

import re
from typing import Any

_ISSUE_LIST_CONTEXT_RE = re.compile(
    r"\b(?:fix|address|resolve|handle|correct).{0,80}\b(?:issues|bugs|findings|items|problems)\b",
    re.IGNORECASE | re.DOTALL,
)
_ISSUE_BULLET_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+)(?P<body>.+?)\s*$")
_SEVERITY_PREFIX_RE = re.compile(r"^(?:critical|high|medium|low|p[0-3])\s*:\s+", re.IGNORECASE)
_OBJECTIVE_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*(?:\(\))?")
_OBJECTIVE_STOPWORDS = {
    "about",
    "after",
    "again",
    "already",
    "also",
    "before",
    "both",
    "cannot",
    "complete",
    "could",
    "does",
    "done",
    "fail",
    "fails",
    "fix",
    "fixed",
    "following",
    "from",
    "handle",
    "high",
    "ignore",
    "ignores",
    "into",
    "issue",
    "issues",
    "large",
    "look",
    "medium",
    "method",
    "missing",
    "only",
    "patch",
    "patches",
    "probably",
    "prompt",
    "report",
    "safe",
    "should",
    "state",
    "still",
    "test",
    "that",
    "there",
    "this",
    "with",
    "works",
}


def clean_objective_title(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip()).strip(" \"'")
    return text[:260]


def extract_multi_objectives(text: str) -> list[str]:
    raw_text = str(text or "")
    if not raw_text.strip():
        return []
    raw_text = re.sub(
        r"\s+([-*]\s+(?:critical|high|medium|low|p[0-3])\s*:)",
        r"\n\1",
        raw_text,
        flags=re.IGNORECASE,
    )
    has_issue_context = _ISSUE_LIST_CONTEXT_RE.search(raw_text) is not None
    objectives: list[str] = []
    current: list[str] = []
    severity_count = 0
    in_code_block = False

    def flush() -> None:
        if not current:
            return
        title = clean_objective_title(" ".join(current))
        current.clear()
        if title and title not in objectives:
            objectives.append(title)

    for raw_line in raw_text.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            flush()
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        match = _ISSUE_BULLET_RE.match(line)
        if match:
            body = str(match.group("body") or "").strip()
            is_severity_item = _SEVERITY_PREFIX_RE.match(body) is not None
            if not has_issue_context and not is_severity_item:
                flush()
                continue
            flush()
            if is_severity_item:
                severity_count += 1
            current.append(body)
            continue
        if current and line.startswith((" ", "\t")) and line.strip():
            current.append(line.strip())
        elif current and not line.strip():
            flush()
    flush()
    if len(objectives) < 2:
        return []
    if not has_issue_context and severity_count < 2:
        return []
    return objectives


def objective_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _OBJECTIVE_TOKEN_RE.findall(str(text or "").lower()):
        token = raw.strip("()")
        if len(token) < 4 or token in _OBJECTIVE_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def objective_matches_text(objective: dict[str, Any], text: str) -> bool:
    title = str(objective.get("title") or "").strip().lower()
    haystack = str(text or "").strip().lower()
    if not title or not haystack:
        return False
    if title in haystack:
        return True
    title_tokens = objective_tokens(title)
    if not title_tokens:
        return False
    matched = title_tokens.intersection(objective_tokens(haystack))
    required = max(2, min(4, len(title_tokens) // 3 or 1))
    return len(matched) >= required
