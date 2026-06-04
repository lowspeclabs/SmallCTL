from __future__ import annotations

import math

_REPEATED_TOOL_HISTORY_LIMIT = 24
_IDENTICAL_TOOL_CALL_STREAK_LIMIT = 3
_REPEATED_TOOL_WINDOW = 12
_REPEATED_TOOL_UNIQUE_LIMIT = 5
_STRICT_LOOP_GUARD_TOOLS = {
    "dir_list",
    "file_read",
    "ssh_file_read",
    "artifact_read",
    "artifact_grep",
    "artifact_print",
    "web_search",
    "web_fetch",
}
_STRICT_LOOP_GUARD_IDENTICAL_LIMIT = 3
_STRICT_LOOP_GUARD_WINDOW_LIMIT = 6
_STRICT_LOOP_GUARD_UNIQUE_LIMIT = 3
_DIR_LIST_LOOP_GUARD_STRICTNESS_MULTIPLIER = 1.5
_DIR_LIST_IDENTICAL_TOOL_CALL_STREAK_LIMIT = max(
    2,
    math.ceil(_STRICT_LOOP_GUARD_IDENTICAL_LIMIT / _DIR_LIST_LOOP_GUARD_STRICTNESS_MULTIPLIER),
)
_DIR_LIST_REPEATED_TOOL_WINDOW = max(
    4,
    math.ceil(_STRICT_LOOP_GUARD_WINDOW_LIMIT / _DIR_LIST_LOOP_GUARD_STRICTNESS_MULTIPLIER),
)
_DIR_LIST_REPEATED_TOOL_UNIQUE_LIMIT = max(
    2,
    math.ceil(_STRICT_LOOP_GUARD_UNIQUE_LIMIT / _DIR_LIST_LOOP_GUARD_STRICTNESS_MULTIPLIER),
)
_DETERMINISTIC_READ_FAILURES_KEY = "_deterministic_read_failures"
_PLACEHOLDER_TOOL_NAME_TOKENS = {
    "tool_name",
    "function_name",
    "action_name",
    "tool",
    "function",
    "action",
    "name",
}
_PLACEHOLDER_ARG_KEY_TOKENS = {
    "arg",
    "args",
    "argument",
    "arguments",
    "param",
    "params",
    "parameter",
    "parameters",
    "value",
    "field",
}
_PLACEHOLDER_ARG_VALUE_TOKENS = {
    "",
    "arg",
    "args",
    "value",
    "parameter",
    "parameters",
    "param",
    "params",
    "tool_name",
    "function_name",
    "action_name",
    "placeholder",
    "string",
    "text",
}
_EXACT_GEMMA_4_SMALL_IT_MODEL_SUFFIXES = (
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
)
_INCOMPLETE_TOOL_CALL_SCRATCHPAD_KEY = "_last_incomplete_tool_call"
