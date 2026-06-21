from __future__ import annotations

import re
from typing import Any

from .task_classifier_constants import (
    ANALYSIS_MARKERS,
    DEBUG_MARKERS,
    IP_ADDRESS_PATTERN,
    PLAN_ONLY_PHRASES,
    READONLY_FILE_TARGETS,
    READONLY_SUGGESTION_MARKERS,
    REMOTE_HINTS_WORD_BOUNDARIES_RE,
    WEB_LOOKUP_MARKERS,
)


_LOCAL_SSH_FILE_MARKERS = (
    "known_hosts",
    "known hosts",
    "authorized_keys",
    "authorized keys",
    "~/.ssh",
    "/.ssh/",
    ".ssh/",
)

_LOCAL_SCOPE_MARKERS = (
    "current user",
    "current user's",
    "this user",
    "this user's",
    "my user",
    "my user's",
    "this host",
    "this machine",
    "local",
    "locally",
    "on this host",
    "on this machine",
)

_EXPLICIT_REMOTE_EXECUTION_MARKERS = (
    "ssh to",
    "connect to",
    "remote host",
    "remote server",
    "target host",
    "over ssh",
    "via ssh",
)

_LOCAL_COMMAND_TARGET_RE = re.compile(
    r"(?:^|[\s`'\"])(?:\./|\.\./|/)[^\s`'\"]+\.(?:py|sh|bash|js|ts|tsx|jsx|rb|pl|lua)\b",
    re.IGNORECASE,
)

_REMOTE_EXECUTION_NEGATION_RE = re.compile(
    r"\b(?:do\s+not|never|don't|dont)\b[^.;\n]*\b(?:ssh\s+to|connect\s+to|use\s+ssh|ssh_exec)\b"
    r"|"
    r"\b(?:no\s+ssh|no\s+ssh_exec)\b",
    re.IGNORECASE,
)


def looks_like_execution_followup(text: str) -> bool:
    followup_phrases = (
        "use the command",
        "use that command",
        "run it",
        "run that",
        "execute it",
        "execute that",
        "try again",
        "use the shell command",
        "run the shell command",
        "execute the shell command",
    )
    return any(phrase in text for phrase in followup_phrases)


def looks_like_readonly_chat_request(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if looks_like_execution_followup(text):
        return False
    readonly_markers = (
        "what",
        "which",
        "show",
        "read",
        "find",
        "search",
        "grep",
        "list",
        "current",
        "status",
        "where",
        "how many",
        "inspect",
        "check",
        "look at",
        "can you see",
        "tell me",
        "summarize",
    )
    readonly_targets = (
        "file",
        "files",
        "folder",
        "directory",
        "repo",
        "repository",
        "cwd",
        "working directory",
        "log",
        "logs",
        "artifact",
        "artifacts",
        "process",
        "cpu",
        "ram",
        "memory",
        "host",
        "system",
        "status",
        "code",
        "source",
        "src",
        "web",
        "website",
        "internet",
        "online",
        "docs",
        "documentation",
        "pricing",
        "release",
        "releases",
        "announcement",
        "news",
    )
    has_readonly_marker = any(marker in text for marker in readonly_markers)
    has_target = any(target in text for target in readonly_targets)
    return has_readonly_marker and has_target


def _has_explicit_remote_execution_scope(text: str) -> bool:
    if any(marker in text for marker in _EXPLICIT_REMOTE_EXECUTION_MARKERS):
        return True
    return bool("@" in text and any(marker in text for marker in ("ssh", "scp", "sftp")))


def task_has_local_scope_markers(task: str) -> bool:
    """Return True when the task text contains explicit local-scope markers."""
    text = str(task or "").strip().lower()
    if not text:
        return False
    if any(marker in text for marker in _LOCAL_SCOPE_MARKERS):
        return True
    return any(marker in text for marker in _LOCAL_SSH_FILE_MARKERS)


def task_has_local_command_target(task: str) -> bool:
    """Detect a local executable/script path used as the command target."""
    text = str(task or "").strip().lower()
    if not text:
        return False
    if _LOCAL_COMMAND_TARGET_RE.search(text):
        return True
    return any(marker in text for marker in ("run ./", "execute ./", "python ./", "python3 ./"))


def task_is_local_ssh_file_target(task: str) -> bool:
    """Detect tasks scoped to this user's local SSH metadata files."""
    text = str(task or "").strip().lower()
    if not text:
        return False
    has_ssh_file = any(marker in text for marker in _LOCAL_SSH_FILE_MARKERS)
    if not has_ssh_file:
        return False
    has_local_scope = any(marker in text for marker in _LOCAL_SCOPE_MARKERS)
    if has_local_scope:
        return True
    # A bare known_hosts cleanup normally treats the IP as data in the file,
    # not as an SSH destination. Explicit remote phrasing still wins.
    return not _has_explicit_remote_execution_scope(text)

def is_smalltalk(task: str) -> bool:
    text = task.strip().lower()
    smalltalk = {
        "hi",
        "hello",
        "hey",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "how are you",
        "what's up",
        "whats up",
    }
    return text in smalltalk


def looks_like_plan_only_request(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    return any(phrase in text for phrase in PLAN_ONLY_PHRASES)


def has_remote_execution_target(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if task_has_local_command_target(text) and not _has_explicit_remote_execution_scope(text):
        return False
    if "working only inside" in text or "only inside ./temp" in text:
        return False
    if _REMOTE_EXECUTION_NEGATION_RE.search(text):
        return False
    for match in IP_ADDRESS_PATTERN.finditer(text):
        ip = match.group(0)
        if ip.startswith(("0.", "127.")):
            continue
        return True
    if "@" in text and any(marker in text for marker in ("ssh", "scp", "sftp")):
        return True
    return _has_explicit_remote_execution_scope(text)


def looks_like_debug_inspection_request(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if "tell me what failed" in text or "what failed" in text:
        return True
    has_debug_marker = any(marker in text for marker in DEBUG_MARKERS)
    has_read_signal = any(
        marker in text
        for marker in (
            "inspect",
            "read",
            "show",
            "tell me",
            "summarize",
            "check",
            "look at",
        )
    )
    from .task_classifier_content_lookup import needs_loop_for_content_lookup
    return has_debug_marker and (has_read_signal or needs_loop_for_content_lookup(task))


def looks_like_analysis_request(task: str) -> bool:
    text = task.strip().lower()
    if not text:
        return False
    if any(marker in text for marker in ANALYSIS_MARKERS):
        return True
    if looks_like_readonly_chat_request(task) and not looks_like_debug_inspection_request(task):
        return True
    from .task_classifier_content_lookup import needs_loop_for_content_lookup
    if needs_loop_for_content_lookup(task) and not looks_like_debug_inspection_request(task):
        return any(target in text for target in READONLY_FILE_TARGETS)
    return False


def task_is_local_system_target(task: str) -> bool:
    """Detect local system administration tasks (e.g. cleaning SSH keys, local config)."""
    text = str(task or "").strip().lower()
    if not text:
        return False
    if task_is_local_ssh_file_target(text):
        return True
    # Strong local-system markers that indicate the user wants local ops
    strong_local_system_markers = (
        "local .ssh",
        "local ssh",
        "authorized_keys",
        "known_hosts",
        "on this host",
        "on this machine",
        "locally",
        "local config",
        "local system",
    )
    # Weaker markers that need additional context
    weak_local_markers = (
        "clean up",
        "cleanup",
        "remove",
        "delete",
    )
    has_strong_marker = any(m in text for m in strong_local_system_markers)
    has_weak_marker = any(m in text for m in weak_local_markers)
    # Exclude only if there are REAL remote execution targets (IPs, user@host ssh patterns)
    # Do NOT exclude just because the word "host" or "remote" appears — those can be local context
    has_real_remote_target = _has_explicit_remote_execution_scope(text)
    if has_strong_marker and not has_real_remote_target:
        return True
    # Weak markers only count if there's also a local-system context word
    if has_weak_marker and not has_real_remote_target:
        local_context_words = ("local", "this host", "here", "on this machine")
        has_local_context = any(w in text for w in local_context_words)
        if has_local_context:
            return True
    return False


def task_is_local_coding_target(task: str) -> bool:
    text = str(task or "").strip()
    if not text:
        return False
    lowered = text.lower()
    has_py_target = bool(re.search(r'(?:\.\/)?[^\s\'"]+\.py', text))
    if not has_py_target:
        return False
    # Extract .py paths and check if any point to a local filesystem location
    py_paths = re.findall(r'[^\s\'"]+\.py', text)
    has_local_style_path = any(
        p.startswith(('./', '../')) or (not p.startswith('/') and not p.startswith('~/'))
        for p in py_paths
    )
    # Strong coding indicators that should override any SSH mentions in instructions
    strong_coding_markers = (
        "build a self-contained python script",
        "build a self-contained python",
        "python script at `./temp/",
        "python script at ./temp/",
        "embedded csv string",
        "embedded markdown",
        "embedded json",
        "embedded sample list",
        "unittest",
    )
    has_strong_coding = any(m in lowered for m in strong_coding_markers)
    has_explicit_remote = has_remote_execution_target(text)
    if has_strong_coding and not has_explicit_remote:
        # Don't let SSH credential instructions in the prompt override a clear coding task
        return True
    # If the task is clearly asking for analysis/suggestions, don't treat as coding
    if any(marker in lowered for marker in READONLY_SUGGESTION_MARKERS):
        return False
    # Local-style paths override weak remote hints (e.g. "fake hosts" in task description)
    if has_local_style_path and not has_explicit_remote:
        return True
    # Fallback: any local .py path without remote indicators is a coding target
    if has_explicit_remote:
        return False
    # Additional weak coding markers
    weak_coding_markers = ("unittest", "python script", "py script", ".py file", "python3")
    has_weak_coding = any(m in lowered for m in weak_coding_markers)
    return has_py_target and has_weak_coding
