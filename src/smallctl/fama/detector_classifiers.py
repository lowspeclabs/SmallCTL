from __future__ import annotations

_LOOP_COUNTERS = ("no_progress", "no_actionable_progress", "repeat_command", "repeat_patch")
_READ_LOOP_TOOLS = {"artifact_read", "file_read", "dir_list", "ssh_file_read", "web_fetch"}
_REMOTE_CONFUSION_REASONS = {
    "remote_path_requires_ssh_exec",
    "remote_path_requires_typed_ssh_file_tool",
}
_BAD_ARG_ERROR_MARKERS = (
    "tool arguments must be an object",
    "missing required field",
    "expected type",
    "schema validation",
    "invalid tool arguments",
    "validation error",
)
_BAD_ARG_REASONS = {"schema_validation", "validation_error", "bad_tool_args", "invalid_arguments"}
_OUTPUT_MISREAD_REASONS = {
    "lookup_answer_missing",
    "answer_missing_from_latest_output",
    "tool_output_contradiction",
    "task_complete_contradicts_tool_output",
}
WRONG_PATH_MARKERS = (
    "no such file or directory",
    "cannot access",
    "not found",
    "filenotfounderror",
    "path does not exist",
)
_TEST_OUTPUT_MARKERS = (
    "assertionerror",
    "traceback",
    "failed (failures=",
    "failed (errors=",
    "ran ",
    "pytest",
    "unittest",
)
WRITE_TOOLS = {"file_write", "file_append", "ssh_file_write"}
_TEST_FAILURE_MARKERS = (
    "failed",
    "error",
    "traceback",
    "assertionerror",
    "pytest",
    "test failed",
)
_ZERO_TEST_MARKERS = (
    "ran 0 tests",
    "no tests ran",
    "collected 0 items",
    "0 tests collected",
    "no tests collected",
)
_PATCH_MISS_MARKERS = (
    "patch target text was not found",
    "target text was not found",
    "replacement bounds were not found",
    "bounded region was not found",
    "class `",
    "function `",
    "method `",
)


def _looks_like_test_failure_output(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    if "not found in" in lowered and any(marker in lowered for marker in _TEST_OUTPUT_MARKERS):
        return True
    if "assertionerror" in lowered and any(marker in lowered for marker in ("failed", "traceback", "ran ")):
        return True
    return False


def _is_patch_target_miss(tool_name: str, combined_result_text: str) -> bool:
    tool = str(tool_name or "").strip()
    if tool not in {"file_patch", "ast_patch", "ssh_file_patch"}:
        return False
    text = str(combined_result_text or "").lower()
    return any(marker in text for marker in _PATCH_MISS_MARKERS)


def _looks_like_zero_tests(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in _ZERO_TEST_MARKERS)


def _verdict_is_pass(value: Any) -> bool:
    return str(value or "").strip().lower() == "pass"


def _verifier_failed(verifier: Any) -> bool:
    if not isinstance(verifier, dict) or not verifier:
        return False
    verdict = str(verifier.get("verdict") or "").strip().lower()
    return bool(verdict) and verdict != "pass"


def _metadata_verifier(metadata: dict[str, Any]) -> dict[str, Any] | None:
    verdict = metadata.get("last_verifier_verdict")
    return verdict if isinstance(verdict, dict) and verdict else None
