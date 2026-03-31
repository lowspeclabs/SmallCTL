from __future__ import annotations

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
