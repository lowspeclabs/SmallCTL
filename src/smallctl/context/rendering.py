from __future__ import annotations

import re
from typing import Any

from ..state import clip_text_value


def render_shell_output(
    output: dict[str, Any],
    *,
    preview_limit: int | None = None,
    strip_whitespace: bool = False,
) -> str:
    stdout = str(output.get("stdout") or "")
    stderr = str(output.get("stderr") or "")
    exit_code = output.get("exit_code")
    progress_updates = output.get("progress_updates")

    parts: list[str] = []
    if isinstance(progress_updates, list) and progress_updates:
        progress_text = "\n".join(str(item).strip() for item in progress_updates if str(item).strip())
        if progress_text:
            parts.append(f"--- [PROGRESS] ---\n{progress_text}")
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"--- [STDERR] ---\n{stderr}")
    if exit_code is not None:
        parts.append(f"--- [EXIT CODE: {exit_code}] ---")

    text = "\n\n".join(parts) or "ok"
    if strip_whitespace:
        text = text.strip()
    if preview_limit is None:
        return text

    preview, clipped = clip_text_value(text, limit=preview_limit)
    if clipped:
        return f"{preview}\n... output truncated"
    return preview


def render_shell_failure(
    *,
    error: str | None,
    output: dict[str, Any] | None = None,
    preview_limit: int | None = None,
    strip_whitespace: bool = False,
) -> str:
    error_text = str(error or "").strip()
    failure_summary = _failure_summary(error_text=error_text, output=output)
    transcript = ""
    if isinstance(output, dict):
        transcript = render_shell_output(
            output,
            preview_limit=None,
            strip_whitespace=strip_whitespace,
        ).strip()
        if transcript == "ok":
            transcript = ""

    parts: list[str] = []
    if failure_summary:
        parts.append(f"--- [FAILURE SUMMARY] ---\n{failure_summary}")
    if error_text:
        parts.append(error_text)
    if transcript:
        if not error_text or not transcript.startswith(error_text):
            parts.append(transcript)
        elif transcript != error_text:
            parts[-1] = transcript

    text = "\n\n".join(parts) or "Shell command failed."
    if preview_limit is None:
        return text
    preview, clipped = clip_text_value(text, limit=preview_limit)
    if clipped:
        return f"{preview}\n... output truncated"
    return preview


_UNITTEST_ERROR_RE = re.compile(r"^ERROR:\s+.+", re.MULTILINE)
_TEST_SUMMARY_RE = re.compile(
    r"^(?:FAILED\s*\([^)]+\)|(?:=+\s*)?\d+\s+failed\b.*|NO TESTS RAN\b.*)",
    re.IGNORECASE | re.MULTILINE,
)
_EXCEPTION_LINE_RE = re.compile(
    r"^(?:[A-Za-z_][\w.]*Error|AssertionError|Exception|SystemExit|KeyboardInterrupt):\s+.+",
    re.MULTILINE,
)


def _failure_summary(*, error_text: str, output: dict[str, Any] | None) -> str:
    lines: list[str] = []
    if isinstance(output, dict):
        exit_code = output.get("exit_code")
        if exit_code not in (None, "", 0, "0"):
            lines.append(f"Command failed with exit code {exit_code}.")
        combined = "\n".join(
            str(output.get(key) or "").strip()
            for key in ("stdout", "stderr")
            if str(output.get(key) or "").strip()
        )
    else:
        combined = ""
    if error_text:
        combined = "\n".join(part for part in (combined, error_text) if part)

    if not combined:
        return "\n".join(lines)

    matches: list[str] = []
    for regex in (_TEST_SUMMARY_RE, _UNITTEST_ERROR_RE, _EXCEPTION_LINE_RE):
        matches.extend(match.group(0).strip() for match in regex.finditer(combined))

    seen = {line.lower() for line in lines}
    for line in matches:
        normalized = line.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line[:240])
        if len(lines) >= 7:
            break
    return "\n".join(lines)
