from __future__ import annotations

import re
from typing import Any

from .task_classifier_constants import (
    ANALYSIS_MARKERS,
    DEBUG_MARKERS,
    IP_ADDRESS_PATTERN,
    PLAN_ONLY_PHRASES,
    READONLY_FILE_TARGETS,
    REMOTE_HINTS_WORD_BOUNDARIES_RE,
)

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
    if IP_ADDRESS_PATTERN.search(text):
        return True
    if "@" in text and any(marker in text for marker in ("ssh", "scp", "sftp")):
        return True
    # Use word-boundary matching to avoid false positives on words like "hosts", "serverless"
    return bool(REMOTE_HINTS_WORD_BOUNDARIES_RE.search(text))


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
    from .task_classifier import looks_like_readonly_chat_request
    if looks_like_readonly_chat_request(task) and not looks_like_debug_inspection_request(task):
        return True
    from .task_classifier_content_lookup import needs_loop_for_content_lookup
    if needs_loop_for_content_lookup(task) and not looks_like_debug_inspection_request(task):
        return any(target in text for target in READONLY_FILE_TARGETS)
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
    if "list improvements" in lowered or "suggest improvements" in lowered:
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
