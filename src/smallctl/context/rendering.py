from __future__ import annotations

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

    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"--- [STDERR] ---\n{stderr}")
    if exit_code is not None and exit_code != 0:
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
    transcript = ""
    if isinstance(output, dict):
        transcript = render_shell_output(
            output,
            preview_limit=preview_limit,
            strip_whitespace=strip_whitespace,
        ).strip()
        if transcript == "ok":
            transcript = ""

    parts: list[str] = []
    if error_text:
        parts.append(error_text)
    if transcript:
        if not error_text or not transcript.startswith(error_text):
            parts.append(transcript)
        elif transcript != error_text:
            parts[-1] = transcript

    return "\n\n".join(parts) or "Shell command failed."
