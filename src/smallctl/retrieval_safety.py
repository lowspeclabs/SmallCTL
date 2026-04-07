from __future__ import annotations

import re
from typing import Any

from .memory.taxonomy import SCHEMA_VALIDATION_ERROR, normalize_failure_mode


def raw_schema_error_present(text: str) -> bool:
    lowered = str(text or "").lower()
    return "missing required field" in lowered or "expected type" in lowered


def extract_failure_tool_name(text: str) -> str:
    prefix, separator, _ = str(text or "").partition(":")
    candidate = prefix.strip().lower()
    if separator and candidate and re.fullmatch(r"[a-z0-9_./-]+", candidate):
        return candidate
    return ""


def format_failure_tag(text: str, *, tool_name: str = "") -> str:
    raw_text = str(text or "").strip()
    if not raw_text:
        return ""
    resolved_tool_name = str(tool_name or extract_failure_tool_name(raw_text)).strip().lower()
    failure_mode = normalize_failure_mode(raw_text, tool_name=resolved_tool_name, success=False)
    if not failure_mode:
        return raw_text
    if resolved_tool_name:
        return f"{resolved_tool_name}: {failure_mode}"
    return failure_mode


def build_retrieval_safe_text(
    *,
    role: str,
    content: str | None,
    name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    normalized_content = str(content or "").strip()
    if not normalized_content:
        return ""

    metadata = metadata if isinstance(metadata, dict) else {}
    tool_name = str(name or metadata.get("tool_name") or "").strip().lower()
    recovery_kind = str(metadata.get("recovery_kind") or metadata.get("repair_kind") or "").strip().lower()

    if recovery_kind == "schema_validation":
        return format_failure_tag(normalized_content, tool_name=tool_name)
    if str(role or "").strip().lower() == "tool" and raw_schema_error_present(normalized_content):
        return format_failure_tag(normalized_content, tool_name=tool_name)
    if metadata.get("is_recovery_nudge") and raw_schema_error_present(normalized_content):
        return format_failure_tag(normalized_content, tool_name=tool_name)
    if (
        raw_schema_error_present(normalized_content)
        and normalize_failure_mode(normalized_content, success=False) == SCHEMA_VALIDATION_ERROR
    ):
        return SCHEMA_VALIDATION_ERROR

    if tool_name in ("shell_exec", "ssh_exec", "run_command"):
        lower_content = normalized_content.lower()
        if "<html" in lower_content or "<!doctype html" in lower_content or "404 not found" in lower_content:
            return f"{tool_name}: HTTP/HTML Error (raw output suppressed for memory safety)"

    if len(normalized_content) > 1024:
        return normalized_content[:1024]
    return normalized_content
