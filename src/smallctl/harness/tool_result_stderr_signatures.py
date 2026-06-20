from __future__ import annotations

import hashlib
import re
from typing import Any


def stderr_text(result: Any) -> str:
    stderr = ""
    if isinstance(getattr(result, "output", None), dict):
        stderr = str(result.output.get("stderr") or "")
    if not stderr and getattr(result, "error", None):
        stderr = str(result.error)
    return stderr


# Lines that are known program output/noise rather than actionable errors.
# These should not become the stderr signature used by the circuit breaker.
_CURL_PROGRESS_HEADER_RE = re.compile(r"^%\s+Total\s+%\s+Received\s+%\s+Xferd")
_CURL_PROGRESS_SUBHEADER_RE = re.compile(r"^Dload\s+Upload\s+Total\s+Spent\s+Left\s+Speed$")
_CURL_PROGRESS_STAT_RE = re.compile(
    r"^\s*\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+--:--:--"
)
_APT_CLI_WARNING = (
    "WARNING: apt does not have a stable CLI interface. Use with caution in scripts."
)


def _is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped == _APT_CLI_WARNING:
        return True
    if _CURL_PROGRESS_HEADER_RE.match(stripped):
        return True
    if _CURL_PROGRESS_SUBHEADER_RE.match(stripped):
        return True
    if _CURL_PROGRESS_STAT_RE.match(stripped):
        return True
    # Curl size/speed summary line, e.g. "100  1320  100  1320    0     0   5743      0"
    if re.match(r"^\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s*$", stripped):
        return True
    return False


def _error_precedence_key(line: str) -> int:
    """Lower values sort earlier; prefer explicit Error lines over warnings/noise."""
    stripped = line.strip()
    if stripped.startswith("Error:"):
        return 0
    if stripped.startswith("ERROR:"):
        return 1
    if stripped.startswith("Fatal:"):
        return 2
    if stripped.startswith("failed:") or stripped.startswith("Failed:"):
        return 3
    return 10


def stderr_signature_line(result: Any) -> str | None:
    text = stderr_text(result)
    lines = [line for line in text.splitlines() if not _is_noise_line(line)]
    if not lines:
        return None
    # Prefer the most specific error line; fall back to the first non-noise line.
    preferred = min(lines, key=_error_precedence_key)
    stripped = preferred.strip()
    return re.sub(r"\s+", " ", stripped)[:240]


def stderr_signature_key(result: Any) -> str | None:
    line = stderr_signature_line(result)
    if not line:
        return None
    return hashlib.sha1(line.lower().encode("utf-8", errors="replace")).hexdigest()[:12]
