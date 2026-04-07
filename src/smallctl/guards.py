from __future__ import annotations

from dataclasses import dataclass
import re

from .state import LoopState
from typing import TYPE_CHECKING
import difflib

if TYPE_CHECKING:
    from .graph.state import PendingToolCall


@dataclass
class GuardConfig:
    max_steps: int = 35
    max_tokens: int | None = None
    max_consecutive_errors: int = 5
    max_repeated_actions: int = 3


_SMALL_MODEL_SIZE_RE = re.compile(r"(?<![xX])\b(?P<size>\d+(?:\.\d+)?)b\b", re.IGNORECASE)
_SMALL_MODEL_HINTS = ("tiny", "mini", "small", "compact")


def is_small_model_name(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    if not normalized:
        return False

    if any(hint in normalized for hint in _SMALL_MODEL_HINTS):
        return True

    for match in _SMALL_MODEL_SIZE_RE.finditer(normalized):
        try:
            size = float(match.group("size"))
        except ValueError:
            continue
        if size <= 14.0:
            return True

    return False


def is_four_b_or_under_model_name(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    if not normalized:
        return False

    for match in _SMALL_MODEL_SIZE_RE.finditer(normalized):
        try:
            size = float(match.group("size"))
        except ValueError:
            continue
        if size <= 4.0:
            return True

    return False


def check_guards(state: LoopState, cfg: GuardConfig) -> str | None:
    if state.step_count >= cfg.max_steps:
        return f"Guard tripped: max_steps ({cfg.max_steps})"
    if cfg.max_tokens is not None and state.token_usage >= cfg.max_tokens:
        return f"Guard tripped: max_tokens ({cfg.max_tokens})"
    if len(state.recent_errors) >= cfg.max_consecutive_errors:
        return f"Guard tripped: max_consecutive_errors ({cfg.max_consecutive_errors}) - Errors: {state.recent_errors}"
    
    no_progress = int(state.stagnation_counters.get("no_progress", 0))
    if no_progress >= 2:
        return f"Guard tripped: stagnation limit - no progress made in {no_progress} steps."
    
    # Repeat detection (Detects loops even if some steps succeed)
    if state.tool_history:
        from collections import Counter
        counts = Counter(state.tool_history)
        for fingerprint, count in counts.items():
            if count >= cfg.max_repeated_actions:
                tool_name = fingerprint.split("|")[0]
                return f"Guard tripped: loop detected - tool '{tool_name}' repeated {count} times with identical args and outcome."

    return None


def apply_triple_answer_guard(
    assistant_text: str,
    pending_tool_calls: list[PendingToolCall],
    threshold: float = 0.9,
) -> str:
    """Detect 'Triple-Answer' (model repeats itself in text and tool).
    
    If any task_complete or task_fail tool call has a message that matches
    the assistant_text by > threshold, we strip the redundant text.
    """
    if not assistant_text or not pending_tool_calls:
        return assistant_text
    
    # Normalize assistant text for comparison
    clean_assistant = assistant_text.strip().lower()
    if not clean_assistant:
        return assistant_text

    for call in pending_tool_calls:
        if call.tool_name in ("task_complete", "task_fail"):
            message = str(call.args.get("message", "")).strip().lower()
            if not message:
                continue
                
            # Quick check: if exactly equal, 100% match
            if message == clean_assistant:
                return ""
            
            # Fuzzy match
            matcher = difflib.SequenceMatcher(None, clean_assistant, message)
            ratio = matcher.ratio()
            
            if ratio >= threshold:
                # Redundancy detected! Return empty string to strip the text
                return ""
                
    return assistant_text
