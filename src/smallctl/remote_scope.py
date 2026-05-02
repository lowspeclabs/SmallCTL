from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .task_targets import extract_task_target_paths

_REMOTE_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w/])/(?:(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+(?:\.[A-Za-z0-9._-]+)?)"
)
_IPV4_HOST_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_USER_AT_HOST_RE = re.compile(
    r"\b[A-Za-z0-9._-]+@(?:[A-Za-z0-9.-]+|\d{1,3}(?:\.\d{1,3}){3})\b",
    re.IGNORECASE,
)
_REMOTE_WEB_FOLLOWUP_HINTS = (
    "page",
    "pages",
    "home page",
    "homepage",
    "site",
    "website",
    "theme",
    "style",
    "styling",
    "design",
    "layout",
    "landing",
    "explainer",
    "nginx",
    "html",
)
_LOCAL_ONLY_HINTS = (
    " locally",
    " local repo",
    " in the repo",
    " in this repo",
    " README.md",
)
_REMOTE_CONTINUATION_QUERY_HINTS = (
    "?",
    " what ",
    " which ",
    " is ",
    " are ",
    " has ",
    " have ",
    " check ",
    " confirm ",
    " enabled ",
    " config",
    " nginx",
    " demo-site",
    " site",
    " page",
    " pages",
)
_REMOTE_REFERENCE_STOPWORDS = {
    "continue",
    "current",
    "task",
    "user",
    "follow",
    "followup",
    "remote",
    "over",
    "ssh",
    "root",
    "host",
    "server",
    "html",
    "var",
    "www",
    "the",
    "this",
    "that",
    "with",
    "from",
    "into",
    "read",
    "site",
    "page",
    "pages",
}


def _scratchpad(state: Any | None) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None) if state is not None else None
    return scratchpad if isinstance(scratchpad, dict) else {}


def _normalized_mode(value: Any) -> str:
    return str(value or "").strip().lower()


def _extract_remote_absolute_paths(*texts: Any) -> list[str]:
    collected: list[str] = []
    seen: set[str] = set()
    for text_value in texts:
        text = str(text_value or "")
        if not text:
            continue
        for match in _REMOTE_ABSOLUTE_PATH_RE.finditer(text):
            normalized = str(match.group(0) or "").strip()
            if not normalized.startswith("/") or normalized in seen:
                continue
            seen.add(normalized)
            collected.append(normalized)
    return collected


def _looks_like_remote_task_text(*texts: Any) -> bool:
    for text_value in texts:
        text = str(text_value or "").strip().lower()
        if not text:
            continue
        if "ssh" in text or "remote host" in text or "remote server" in text:
            return True
        if _USER_AT_HOST_RE.search(text) is not None:
            return True
        if _IPV4_HOST_RE.search(text) is not None and any(
            hint in text for hint in ("host", "server", "ssh", "remote")
        ):
            return True
    return False


def _remote_paths_overlap(left: list[str], right: list[str]) -> bool:
    left_norm = {str(path).strip().lower() for path in left if str(path).strip()}
    right_norm = {str(path).strip().lower() for path in right if str(path).strip()}
    if not left_norm or not right_norm:
        return False
    if left_norm & right_norm:
        return True
    left_names = {Path(path).name.lower() for path in left_norm}
    right_names = {Path(path).name.lower() for path in right_norm}
    return bool(left_names & right_names)


def _confirmed_session_targets(state: Any | None) -> list[dict[str, str]]:
    scratchpad = _scratchpad(state)
    targets = scratchpad.get("_session_ssh_targets")
    if not isinstance(targets, dict):
        return []
    collected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for key, value in targets.items():
        if not isinstance(value, dict) or not bool(value.get("confirmed")):
            continue
        host = str(value.get("host") or key or "").strip().lower()
        if not host:
            continue
        user = str(value.get("user") or "").strip()
        token = (host, user)
        if token in seen:
            continue
        seen.add(token)
        collected.append({"host": host, "user": user})
    return collected


def has_any_session_ssh_target(state: Any | None) -> bool:
    scratchpad = _scratchpad(state)
    targets = scratchpad.get("_session_ssh_targets")
    if not isinstance(targets, dict):
        return False
    return any(isinstance(value, dict) and str(value.get("host") or key or "").strip() for key, value in targets.items())


def confirmed_ssh_target_count(state: Any | None) -> int:
    return len(_confirmed_session_targets(state))


def has_single_confirmed_ssh_target(state: Any | None) -> bool:
    return confirmed_ssh_target_count(state) == 1


def _handoff_remote_anchor_count(handoff: dict[str, Any]) -> int:
    count = 0
    ssh_target = handoff.get("ssh_target")
    if isinstance(ssh_target, dict) and str(ssh_target.get("host") or "").strip():
        count += 1
    ssh_targets = handoff.get("ssh_targets")
    if isinstance(ssh_targets, list) and any(
        isinstance(target, dict) and str(target.get("host") or "").strip() for target in ssh_targets
    ):
        count += 1
    remote_target_paths = handoff.get("remote_target_paths")
    if isinstance(remote_target_paths, list) and any(str(path).strip() for path in remote_target_paths):
        count += 1
    next_required_tool = handoff.get("next_required_tool")
    if isinstance(next_required_tool, dict) and str(next_required_tool.get("tool_name") or "").strip() == "ssh_exec":
        count += 1
    return count


def handoff_supports_remote_continuation(state: Any | None) -> bool:
    scratchpad = _scratchpad(state)
    handoff = scratchpad.get("_last_task_handoff")
    if not isinstance(handoff, dict) or not handoff:
        return False

    has_remote_anchors = _handoff_remote_anchor_count(handoff) > 0
    if has_remote_anchors and confirmed_ssh_target_count(state) > 0:
        return True

    task_mode = _normalized_mode(handoff.get("task_mode"))
    if task_mode != "remote_execute" and not _looks_like_remote_task_text(
        handoff.get("effective_task"),
        handoff.get("current_goal"),
        handoff.get("raw_task"),
    ):
        return False

    if has_remote_anchors:
        return True
    return confirmed_ssh_target_count(state) > 0


def remote_scope_is_active(state: Any | None) -> bool:
    if state is None:
        return False
    if _normalized_mode(getattr(state, "task_mode", "")) == "remote_execute":
        return True
    if _normalized_mode(getattr(state, "active_intent", "")) == "requested_ssh_exec":
        return True
    scratchpad = _scratchpad(state)
    resolved_remote = scratchpad.get("_resolved_remote_followup")
    if isinstance(resolved_remote, dict) and resolved_remote:
        return True
    return handoff_supports_remote_continuation(state)


def recent_remote_target_paths(state: Any | None) -> list[str]:
    scratchpad = _scratchpad(state)
    handoff = scratchpad.get("_last_task_handoff")
    if not isinstance(handoff, dict):
        return []
    paths = handoff.get("remote_target_paths")
    if not isinstance(paths, list):
        return []
    return [str(path).strip() for path in paths if str(path).strip()]


def _remote_reference_terms(paths: list[str]) -> set[str]:
    terms: set[str] = set()
    for path in paths:
        name = Path(str(path).strip()).name.lower()
        stem = Path(str(path).strip()).stem.lower()
        for candidate in (name, stem):
            token = candidate.strip()
            if len(token) >= 4:
                terms.add(token)
        simplified = re.sub(r"-(?:page-)?\d+$", "", stem).strip("-_ ")
        if len(simplified) >= 4:
            terms.add(simplified)
        stripped_page = re.sub(r"-page$", "", simplified).strip("-_ ")
        if len(stripped_page) >= 4:
            terms.add(stripped_page)
    return terms


def _handoff_reference_terms(state: Any | None) -> set[str]:
    scratchpad = _scratchpad(state)
    handoff = scratchpad.get("_last_task_handoff")
    if not isinstance(handoff, dict):
        return set()

    terms = set(_remote_reference_terms(recent_remote_target_paths(state)))
    raw_text = " ".join(
        str(handoff.get(key) or "")
        for key in ("effective_task", "current_goal", "raw_task")
    ).lower()
    for token in re.findall(r"[a-z0-9][a-z0-9._/-]{2,}", raw_text):
        for candidate in re.split(r"[/._-]+", token):
            cleaned = candidate.strip().lower()
            if len(cleaned) < 4 or cleaned in _REMOTE_REFERENCE_STOPWORDS or cleaned.isdigit():
                continue
            terms.add(cleaned)
    return terms


def task_matches_remote_continuation(state: Any | None, task: str) -> bool:
    text = " ".join(str(task or "").strip().lower().split())
    if not text:
        return False
    if not handoff_supports_remote_continuation(state):
        return False

    explicit_local_targets = extract_task_target_paths(text)
    remote_paths = recent_remote_target_paths(state)
    explicit_remote_paths = _extract_remote_absolute_paths(text)
    if explicit_remote_paths:
        return bool(remote_paths and _remote_paths_overlap(explicit_remote_paths, remote_paths))
    if explicit_local_targets:
        return False

    if any(marker in text for marker in ("./", "../", "/home/")):
        return False
    if any(hint.lower() in f" {text} " for hint in _LOCAL_ONLY_HINTS):
        return False

    if not remote_paths:
        return False
    if confirmed_ssh_target_count(state) != 1:
        return False

    reference_terms = _remote_reference_terms(remote_paths)
    mentions_reference = any(term in text for term in reference_terms)
    has_web_hint = any(hint in text for hint in _REMOTE_WEB_FOLLOWUP_HINTS)
    has_remote_web_context = any(
        path.startswith("/var/www/") or path.endswith(".html") or path.endswith(".htm")
        for path in remote_paths
    )
    if mentions_reference or (has_remote_web_context and has_web_hint):
        return True

    handoff_terms = _handoff_reference_terms(state)
    overlaps_handoff = any(term in text for term in handoff_terms)
    has_query_hint = any(hint in f" {text} " for hint in _REMOTE_CONTINUATION_QUERY_HINTS)
    return overlaps_handoff and has_query_hint
