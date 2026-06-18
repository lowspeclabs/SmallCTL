from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

from .common import fail


def supported_ast_patch_operations() -> list[str]:
    return [
        "add_import",
        "replace_function",
        "insert_in_function",
        "update_call_keyword",
        "add_dataclass_field",
    ]


def unsupported_language_failure(
    *,
    path: Path,
    requested_path: str,
    language: str,
    operation: str,
) -> dict[str, Any]:
    return fail(
        f"Language `{language}` is not supported by `ast_patch` yet. Use `language='python'` for v1.",
        metadata={
            "path": str(path),
            "requested_path": requested_path,
            "language": language,
            "operation": operation,
            "error_kind": "unsupported_language",
            "supported_languages": ["python"],
        },
    )


def build_ast_patch_metadata(
    *,
    path: Path,
    requested_path: str,
    source_path: Path,
    session: Any | None,
    staged_only: bool,
    language: str,
    operation: str,
    target: dict[str, Any],
    payload: dict[str, Any],
    changed: bool,
    updated_text: str,
    original_text: str,
    matched_node_count: int,
    touched_symbols: list[str],
    dry_run: bool,
    expected_followup_verifier: str | None,
    staging_path: Path | None,
    status_block: str | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path),
        "requested_path": requested_path,
        "source_path": str(source_path),
        "staged_only": staged_only,
        "language": language,
        "operation": operation,
        "target": target,
        "payload": payload,
        "changed": changed,
        "matched_node_count": matched_node_count,
        "touched_symbols": list(touched_symbols),
        "diff_preview": build_diff_preview(original_text, updated_text) if changed else "",
        "bytes": len((updated_text if changed else original_text).encode("utf-8")),
        "dry_run": dry_run,
        "expected_followup_verifier": str(expected_followup_verifier or ""),
    }
    if staging_path is not None:
        metadata["staging_path"] = str(staging_path)
    if session is not None:
        metadata["write_session_id"] = str(getattr(session, "write_session_id", "") or "").strip()
        metadata["write_session_status_block"] = status_block or ""
    return metadata


def dedupe_symbols(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        candidate = str(value or "").strip()
        if not candidate or candidate in deduped:
            continue
        deduped.append(candidate)
    return deduped


def build_diff_preview(before: str, after: str, *, limit: int = 1400) -> str:
    if before == after:
        return ""
    preview = "\n".join(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=1,
        )
    ).strip()
    if len(preview) <= limit:
        return preview
    return preview[: limit - 14].rstrip() + "\n...[truncated]"
