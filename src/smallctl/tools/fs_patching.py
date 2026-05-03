from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..state import clip_text_value


def _count_exact_occurrences(text: str, target_text: str) -> int:
    if target_text == "":
        return 0
    return text.count(target_text)


def _build_patch_text_preview(text: str, *, limit: int = 120) -> dict[str, Any]:
    preview, clipped = clip_text_value(text, limit=limit)
    return {
        "preview": preview.replace("\n", "\\n"),
        "clipped": clipped,
        "bytes": len(text.encode("utf-8")),
        "line_count": text.count("\n") + (1 if text else 0),
    }


def _build_patch_best_match(
    content: str,
    target_text: str,
    *,
    limit: int = 240,
) -> dict[str, Any] | None:
    """Return a compact nearest snippet for exact patch misses."""

    if not content or not target_text:
        return None

    content_lines = content.splitlines()
    if not content_lines:
        return None
    target_lines = target_text.splitlines() or [target_text]
    window_sizes = sorted({max(1, len(target_lines) - 1), max(1, len(target_lines)), len(target_lines) + 1})
    normalized_target = " ".join(target_text.split())

    best: dict[str, Any] | None = None
    best_score = -1.0
    for window_size in window_sizes:
        if window_size < 1:
            continue
        for start_index in range(0, max(1, len(content_lines) - window_size + 1)):
            snippet_lines = content_lines[start_index : start_index + window_size]
            if not snippet_lines:
                continue
            snippet = "\n".join(snippet_lines)
            exact_score = SequenceMatcher(None, target_text, snippet).ratio()
            normalized_snippet = " ".join(snippet.split())
            normalized_score = (
                SequenceMatcher(None, normalized_target, normalized_snippet).ratio()
                if normalized_target and normalized_snippet
                else 0.0
            )
            score = max(exact_score, normalized_score)
            if score <= best_score:
                continue
            preview, clipped = clip_text_value(snippet, limit=limit)
            best_score = score
            best = {
                "preview": preview,
                "clipped": clipped,
                "chars": len(snippet),
                "start_line": start_index + 1,
                "end_line": start_index + len(snippet_lines),
                "similarity": round(score, 3),
                "match_basis": "whitespace_normalized" if normalized_score > exact_score else "exact",
            }
    return best


def _build_patch_ambiguity_hint(
    *,
    actual_occurrences: int,
    expected_occurrences: int,
) -> str:
    if actual_occurrences <= 0:
        return (
            "The target text was not found. Read the smallest relevant slice first, then retry with an exact "
            "substring from the current file contents."
        )
    if actual_occurrences == 1:
        return "The target text matched once, but the patch still failed. Read the smallest relevant slice and retry."
    if expected_occurrences == 1:
        return (
            f"The target text matched {actual_occurrences} times. Read a smaller slice and make `target_text` more "
            "specific, or set `expected_occurrences` to the exact number of matches only if you intend to replace "
            "every one of them."
        )
    return (
        f"The target text matched {actual_occurrences} times and `expected_occurrences` was {expected_occurrences}. "
        "Read a smaller slice and make the target more specific, or keep the explicit multi-match replacement if "
        "every match should be updated."
    )


def _build_patch_failure_metadata(
    *,
    path: Path,
    requested_path: str,
    error_kind: str,
    source_path: Path | None = None,
    staged_only: bool = False,
    session: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "path": str(path),
        "requested_path": requested_path,
        "error_kind": error_kind,
        "staged_only": staged_only,
    }
    if source_path is not None:
        metadata["source_path"] = str(source_path)
    if session is not None:
        metadata["write_session_id"] = str(getattr(session, "write_session_id", "") or "").strip()
    if extra:
        metadata.update(extra)
    return metadata


def _build_patch_failure_message(
    *,
    requested_path: str,
    source_path: Path | None,
    staged_only: bool,
    error_kind: str,
    actual_occurrences: int | None = None,
    expected_occurrences: int | None = None,
) -> str:
    target_label = f"`{requested_path}`" if requested_path else "the requested path"
    if staged_only and source_path is not None:
        stage_label = f"`{source_path}`"
        if error_kind == "patch_target_not_found":
            return f"Patch target text was not found in active staged copy {stage_label} for target {target_label}."
        if error_kind == "patch_occurrence_mismatch":
            return (
                f"Patch target text occurred {actual_occurrences} times in active staged copy {stage_label} "
                f"for target {target_label}, but expected {expected_occurrences}."
            )
    if error_kind == "patch_target_not_found":
        return f"Patch target text was not found in `{requested_path}`."
    if error_kind == "patch_occurrence_mismatch":
        return (
            f"Patch target text occurred {actual_occurrences} times in `{requested_path}`, "
            f"but expected {expected_occurrences}."
        )
    return f"Patch failed for {target_label}."


def _apply_exact_patch(
    text: str,
    target_text: str,
    replacement_text: str,
    *,
    expected_occurrences: int = 1,
) -> tuple[str, int]:
    actual_occurrences = text.count(target_text) if target_text else 0
    if actual_occurrences != expected_occurrences:
        raise ValueError(str(actual_occurrences))
    return text.replace(target_text, replacement_text, expected_occurrences), actual_occurrences


def _build_patch_metadata(
    *,
    path: Path,
    requested_path: str,
    target_text: str,
    replacement_text: str,
    occurrence_count: int,
    expected_occurrences: int,
    source_path: Path,
    staged_only: bool,
    encoding: str = "utf-8",
    session: Any | None = None,
    staging_path: Path | None = None,
    status_block: str | None = None,
) -> dict[str, Any]:
    from .fs import _read_text_file

    metadata: dict[str, Any] = {
        "path": str(path),
        "requested_path": requested_path,
        "bytes": len(_read_text_file(source_path, encoding=encoding).encode(encoding)),
        "target_text_bytes": len(target_text.encode(encoding)),
        "replacement_text_bytes": len(replacement_text.encode(encoding)),
        "occurrence_count": occurrence_count,
        "expected_occurrences": expected_occurrences,
        "source_path": str(source_path),
        "staged_only": staged_only,
        "target_text_preview": _build_patch_text_preview(target_text),
        "replacement_text_preview": _build_patch_text_preview(replacement_text),
    }
    if staging_path is not None:
        metadata["staging_path"] = str(staging_path)
    if session is not None:
        metadata["write_session_id"] = str(getattr(session, "write_session_id", "") or "").strip()
        metadata["write_session_status_block"] = status_block or ""
    return metadata
