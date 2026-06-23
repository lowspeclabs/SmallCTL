from __future__ import annotations

import re
from typing import Any

from .common import fail


_CURL_RE = re.compile(r"\bcurl\b", re.IGNORECASE)
_PIPE_TO_SHELL_RE = re.compile(
    r"(?:curl|wget)\b[^|]*\|\s*(?:bash|sh|dash|zsh|ksh)\b",
    re.IGNORECASE,
)
_EXECUTE_DOWNLOADED_RE = re.compile(
    r"(?:curl|wget)\b.*(?:-[oO]\s+\S+|--output\s+\S+|\>\s*\S+).*\b(?:bash|sh|dash|zsh|ksh)\b",
    re.IGNORECASE,
)


def _curl_command_has_fail_flag(command: str) -> bool:
    """Return True if the curl invocation uses -f, --fail, or --fail-with-body."""
    raw = str(command or "")
    # Split the command on the curl token and inspect the first curl invocation.
    for match in _CURL_RE.finditer(raw):
        start = match.start()
        # Look at the remainder of the command up to the next shell control token
        # or pipe/redirect that ends the curl invocation.
        remainder = raw[start:]
        end_match = re.search(r"(?=[|;&<>])", remainder)
        curl_chunk = remainder[: end_match.start()] if end_match else remainder
        # Now look for the fail flag as a standalone token in this chunk.
        if re.search(r"(^|\s)(-[\w]*f[\w]*|--fail(?:-with-body)?)(?=\s|$)", curl_chunk, re.IGNORECASE):
            return True
    return False


def _looks_like_script_fetch_or_pipe(command: str) -> bool:
    """Return True when curl/wget fetches something that will be executed."""
    raw = str(command or "")
    return bool(_PIPE_TO_SHELL_RE.search(raw) or _EXECUTE_DOWNLOADED_RE.search(raw))


def _curl_fail_flag_guard(command: str, *, tool_name: str) -> dict[str, Any] | None:
    """Block curl/wget fetches that will be executed unless curl uses -f/--fail.

    Without -f, curl exits 0 on HTTP errors and can pipe empty/404 HTML into a
    shell, producing the exact failure mode seen on Debian 13 installer runs.
    """
    raw = str(command or "").strip()
    if not raw or not _CURL_RE.search(raw):
        return None
    if not _looks_like_script_fetch_or_pipe(raw):
        return None
    if _curl_command_has_fail_flag(raw):
        return None
    return fail(
        f"`{tool_name}` blocked a curl/wget fetch that will be executed because curl is missing "
        "the `-f` / `--fail` flag. Without it, HTTP errors exit 0 and empty or 404 responses can be "
        "piped into a shell. Retry with `curl -fsSL ...` (or add `-f` / `--fail`).",
        metadata={
            "reason": "curl_missing_fail_flag",
            "command": raw,
            "next_required_action": "Use curl -fsSL <URL> when fetching scripts or keys.",
        },
    )


def _curl_pipe_to_shell_guard(command: str, *, tool_name: str) -> dict[str, Any] | None:
    """Deprecated alias kept for backward compatibility.

    Prefer `_curl_fail_flag_guard` which checks the actual flag rather than
    blocking all curl-to-shell invocations.
    """
    return _curl_fail_flag_guard(command, tool_name=tool_name)
