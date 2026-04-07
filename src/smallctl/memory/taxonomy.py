from __future__ import annotations

from typing import Any

# Recommended Failure Modes from memory-upgrade.md
TOOL_NOT_CALLED = "tool_not_called"
WRONG_TOOL_CALLED = "wrong_tool_called"
SCHEMA_VALIDATION_ERROR = "schema_validation_error"
ZERO_ARG_TOOL_ARG_LEAK = "zero_arg_tool_arg_leak"
REPEATED_TOOL_LOOP = "repeated_tool_loop"
PREMATURE_TASK_COMPLETE = "premature_task_complete"
PHASE_MISMATCH = "phase_mismatch"
RETRIEVAL_NOISE = "retrieval_noise"
STALE_MEMORY_APPLIED = "stale_memory_applied"
ENVIRONMENT_MISMATCH = "environment_mismatch"
UNKNOWN_FAILURE = "unknown_failure"

ALL_FAILURE_MODES = {
    TOOL_NOT_CALLED,
    WRONG_TOOL_CALLED,
    SCHEMA_VALIDATION_ERROR,
    ZERO_ARG_TOOL_ARG_LEAK,
    REPEATED_TOOL_LOOP,
    PREMATURE_TASK_COMPLETE,
    PHASE_MISMATCH,
    RETRIEVAL_NOISE,
    STALE_MEMORY_APPLIED,
    ENVIRONMENT_MISMATCH,
    UNKNOWN_FAILURE,
}


def normalize_failure_mode(error: Any, *, tool_name: str = "", success: bool = False) -> str:
    if success:
        return ""

    text = str(error or "").lower()
    normalized_tool = str(tool_name or "").lower()

    if "missing required field" in text or "expected type" in text or "schema" in text:
        return SCHEMA_VALIDATION_ERROR
    if "unknown tool" in text:
        return WRONG_TOOL_CALLED
    if "zero-argument" in text or ("scratch_list" in normalized_tool and "missing" in text):
        return ZERO_ARG_TOOL_ARG_LEAK
    if "loop" in text:
        return REPEATED_TOOL_LOOP
    if "premature" in text:
        return PREMATURE_TASK_COMPLETE
    if "phase" in text:
        return PHASE_MISMATCH
    if "not called" in text:
        return TOOL_NOT_CALLED
    return UNKNOWN_FAILURE
