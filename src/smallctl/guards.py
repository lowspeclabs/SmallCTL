from __future__ import annotations

from dataclasses import dataclass

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


def check_guards(state: LoopState, cfg: GuardConfig) -> str | None:
    if state.step_count >= cfg.max_steps:
        return f"Guard tripped: max_steps ({cfg.max_steps})"
    if cfg.max_tokens is not None and state.token_usage >= cfg.max_tokens:
        return f"Guard tripped: max_tokens ({cfg.max_tokens})"
    if len(state.recent_errors) >= cfg.max_consecutive_errors:
        return f"Guard tripped: max_consecutive_errors ({cfg.max_consecutive_errors}) - Errors: {state.recent_errors}"
    
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
