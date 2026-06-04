from __future__ import annotations

import re

_MUTATION_TOOLS = {
    "ssh_file_write",
    "ssh_file_patch",
    "ssh_file_replace_between",
    "file_write",
    "file_append",
    "file_patch",
    "ast_patch",
}

_READ_TOOLS = {
    "artifact_read",
    "ssh_file_read",
    "file_read",
}
_PATCH_META_TOOLS = {
    "artifact_grep",
    "artifact_print",
    "log_note",
    "memory_update",
}
_DETERMINISTIC_READ_FAILURES_KEY = "_deterministic_read_failures"
_FAILED_MUTATION_REPAIR_PROGRESS_KEY = "_failed_mutation_repair_progress"
_FAILED_MUTATION_REPAIR_PROGRESS_BUDGET = 3
_PATCH_TARGET_NOT_FOUND_COUNTS_KEY = "_patch_target_not_found_counts"
_PATCH_TARGET_NOT_FOUND_SUPPRESS_AFTER = 2
_STALE_VERIFIER_KEY = "_last_verifier_stale_after_mutation"
_LAST_FAILED_VERIFIER_KEY = "_last_failed_verifier"

_COMPLETION_CONFABULATION_PATTERNS = (
    re.compile(r"\b(?:already\s+(?:done|complete|finished|fixed|patched|written)|"
               r"(?:task|work)\s+(?:is\s+)?(?:already\s+)?complete|"
               r"(?:no\s+)?(?:further\s+)?action\s+(?:is\s+)?(?:required|needed)|"
               r"(?:nothing\s+)?(?:else\s+)?(?:to\s+)?(?:do|fix|change|write))\b", re.IGNORECASE),
    re.compile(r"\b(?:successfully\s+(?:patched|fixed|updated|written|created|deployed|installed))\b", re.IGNORECASE),
    re.compile(r"\b(?:patch\s+applied|fix\s+verified|all\s+tests\s+pass(?:ed|ing)?|"
               r"ready\s+for\s+(?:review|production|deployment))\b", re.IGNORECASE),
)
