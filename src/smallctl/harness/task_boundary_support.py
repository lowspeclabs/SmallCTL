from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ..task_targets import extract_task_target_paths

_SYSTEM_FOLLOW_UP_SPLIT_RE = re.compile(r"\nFollow-up:\s*", re.IGNORECASE)
_INLINE_CONTINUE_TASK_PREFIX_RE = re.compile(
    r"^\s*Continue current task:\s*(?P<body>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_INLINE_USER_WRAP_MARKER_RE = re.compile(
    r"\.(?:\s*User\s+(?P<kind>follow-up|correction):\s*)",
    re.IGNORECASE,
)
_NUMBERED_OPTION_RE = re.compile(r"^\s*(\d+)[.)]\s+(.+?)\s*$")
_INLINE_NUMBERED_OPTION_RE = re.compile(r"(?:^|\s)(\d+)[.)]\s+(.+?)(?=(?:\s+\d+[.)]\s+)|$)")
_MARKDOWN_OPTION_RE = re.compile(
    r"^\s*(?:\*\*)?(?:option|proposal)\s+(\d+)\s*(?:[-—–:]|\*\*)\s*(.*?)(?:\*\*)?\s*$",
    re.IGNORECASE,
)
_OPTION_ACTION_WORDS = re.compile(
    r"\b(stream|streaming|md5|hash|patch|edit|modify|fix|update|implement|add|replace|refactor|test|skip|handle|read|write|calculate)\b",
    re.IGNORECASE,
)
_TARGET_NEGATION_RE = re.compile(r"\b(?:instead\s+of|rather\s+than|without|avoid|do\s+not|don't|dont)\b", re.IGNORECASE)
_TARGET_REPLACEMENT_RE = re.compile(
    r"\b(?:rewrite|rebuild|replace|convert|migrate|switch|change\s+to|move\s+to|port\s+to)\b",
    re.IGNORECASE,
)
_TARGET_LANGUAGE_RE = re.compile(r"\b(?:rust|go|typescript|javascript|python|bash|shell)\b", re.IGNORECASE)


def normalize_task_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def collapse_task_chain(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part.strip() for part in _SYSTEM_FOLLOW_UP_SPLIT_RE.split(text) if part.strip()]
    candidate = parts[-1] if parts else text
    inline = canonicalize_inline_task_wrapper(candidate)
    return inline if inline else candidate


def base_task_from_task_chain(value: Any) -> str:
    text = collapse_task_chain(value)
    parsed = parse_inline_task_wrapper(text)
    if parsed is None:
        return text
    base = str(parsed.get("base") or "").strip()
    return base or text


def is_remote_followup_wrapper(value: Any) -> bool:
    return normalize_task_text(collapse_task_chain(value)).startswith("continue remote task over ssh")


def canonicalize_inline_task_wrapper(value: Any) -> str:
    parsed = parse_inline_task_wrapper(value)
    if parsed is None:
        return ""
    base = str(parsed.get("base") or "").strip()
    latest_suffix_text = str(parsed.get("latest_suffix_text") or "").strip()
    latest_suffix_kind = str(parsed.get("latest_suffix_kind") or "").strip().lower()
    if not base:
        return str(value or "").strip()
    if not latest_suffix_text:
        return f"Continue current task: {base}"

    label = "User correction" if latest_suffix_kind == "correction" else "User follow-up"
    return f"Continue current task: {base}. {label}: {latest_suffix_text}"


def parse_inline_task_wrapper(value: Any) -> dict[str, str] | None:
    text = str(value or "").strip()
    if not text:
        return None

    current = text
    saw_wrapper = False
    latest_suffix_kind = ""
    latest_suffix_text = ""
    while True:
        match = _INLINE_CONTINUE_TASK_PREFIX_RE.match(current)
        if match is None:
            break
        saw_wrapper = True
        body = str(match.group("body") or "").strip()
        if not body:
            break
        suffix_markers = list(_INLINE_USER_WRAP_MARKER_RE.finditer(body))
        if suffix_markers:
            suffix = suffix_markers[-1]
            suffix_text = body[suffix.end() :].strip()
            suffix_kind = str(suffix.group("kind") or "").strip().lower()
            if suffix_text:
                latest_suffix_text = suffix_text
                latest_suffix_kind = suffix_kind
            current = body[: suffix.start()].strip().rstrip(".")
            if not current:
                break
            continue
        current = body
        break

    if not saw_wrapper:
        return None

    base = current.strip()
    return {
        "base": base or text,
        "latest_suffix_kind": latest_suffix_kind,
        "latest_suffix_text": latest_suffix_text,
    }


def clean_option_title(value: str) -> str:
    title = str(value or "").strip()
    if not title:
        return ""
    bold = re.match(r"^\*\*(.+?)\*\*(?:\s*[-:]\s*(.*))?$", title)
    if bold:
        head = str(bold.group(1) or "").strip()
        tail = str(bold.group(2) or "").strip()
        return f"{head} - {tail}" if tail else head
    return title


def extract_action_options_from_text(text: str, inherited_paths: list[str]) -> list[dict[str, Any]]:
    options: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    def _append_option(index: int, raw_title: str) -> None:
        title = clean_option_title(raw_title)
        if not title or not _OPTION_ACTION_WORDS.search(title):
            return
        key = (index, title.lower())
        if key in seen:
            return
        seen.add(key)
        paths = extract_task_target_paths(title) or inherited_paths
        options.append(
            {
                "index": index,
                "title": title,
                "target_paths": list(paths),
            }
        )

    for line in str(text or "").splitlines():
        match = _NUMBERED_OPTION_RE.match(line)
        if not match:
            match = _MARKDOWN_OPTION_RE.match(line)
        if not match:
            continue
        _append_option(int(match.group(1)), str(match.group(2) or ""))
    if options:
        return options
    flattened = re.sub(r"\s+", " ", str(text or "").strip())
    for match in _INLINE_NUMBERED_OPTION_RE.finditer(flattened):
        try:
            index = int(match.group(1))
        except (TypeError, ValueError):
            continue
        _append_option(index, str(match.group(2) or ""))
    return options


def merge_action_options(
    existing: list[dict[str, Any]],
    extracted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for option in [*existing, *extracted]:
        if not isinstance(option, dict):
            continue
        title = str(option.get("title") or "").strip()
        try:
            index = int(option.get("index") or 0)
        except (TypeError, ValueError):
            index = 0
        if not title or index <= 0:
            continue
        key = (index, title.lower())
        if key in seen:
            continue
        seen.add(key)
        target_paths = option.get("target_paths")
        if isinstance(target_paths, list):
            cleaned_paths = [str(path).strip() for path in target_paths if str(path).strip()]
        else:
            cleaned_paths = []
        merged.append({"index": index, "title": title, "target_paths": cleaned_paths})
    return merged


def blocks_inherited_target(suffix: str, inherited_paths: list[str]) -> bool:
    text = str(suffix or "").strip().lower()
    if not text or not inherited_paths:
        return False
    path_names = {str(path).strip().lower() for path in inherited_paths if str(path).strip()}
    path_basenames = {Path(path).name.lower() for path in path_names if path}
    mentions_inherited_path = any(path in text for path in path_names | path_basenames)
    mentions_generic_code_target = bool(re.search(r"\b(?:python\s+file|script|file|module|code)\b", text))
    if _TARGET_NEGATION_RE.search(text) and (mentions_inherited_path or mentions_generic_code_target):
        return True
    if _TARGET_NEGATION_RE.search(text) and (_TARGET_REPLACEMENT_RE.search(text) or _TARGET_LANGUAGE_RE.search(text)):
        return True
    return False


def normalize_remote_host(value: Any) -> str:
    return str(value or "").strip().lower()


def coerce_remote_target(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    host = normalize_remote_host(value.get("host"))
    if not host:
        return None
    user = str(value.get("user") or "").strip()
    return {"host": host, "user": user}


def merge_remote_targets(targets: list[dict[str, str]]) -> list[dict[str, str]]:
    merged: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for target in targets:
        normalized = coerce_remote_target(target)
        if normalized is None:
            continue
        key = (normalized["host"], normalized["user"].lower())
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged


def format_remote_target(target: dict[str, Any]) -> str:
    host = normalize_remote_host(target.get("host"))
    user = str(target.get("user") or "").strip()
    if not host:
        return ""
    return f"{user}@{host}" if user else host
