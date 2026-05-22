from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from difflib import unified_diff
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from ..state import clip_text_value


def _count_exact_occurrences(text: str, target_text: str) -> int:
    if target_text == "":
        return 0
    return text.count(target_text)


PatchMode = str


@dataclass(frozen=True)
class PatchPlan:
    mode: PatchMode
    old_text: str
    new_text: str
    occurrence_count: int
    replacement_count: int
    match_spans: list[tuple[int, int]]
    selected_occurrence: int | None
    diff: str
    diff_preview: str
    diff_clipped: bool
    old_sha256: str
    new_sha256: str


def _text_sha256(text: str, *, encoding: str = "utf-8") -> str:
    return hashlib.sha256(text.encode(encoding)).hexdigest()


def _build_unified_patch_diff(
    old_text: str,
    new_text: str,
    *,
    fromfile: str,
    tofile: str,
) -> str:
    return "".join(
        unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def _build_patch_diff_metadata(plan: PatchPlan, *, dry_run: bool) -> dict[str, Any]:
    return {
        "diff": plan.diff,
        "diff_preview": plan.diff_preview,
        "diff_clipped": plan.diff_clipped,
        "old_sha256": plan.old_sha256,
        "new_sha256": plan.new_sha256,
        "dry_run": dry_run,
        "patch_mode": plan.mode,
        "actual_occurrences": plan.occurrence_count,
        "replacement_count": plan.replacement_count,
        "match_spans": [{"start": start, "end": end} for start, end in plan.match_spans],
        "selected_occurrence": plan.selected_occurrence,
    }


def _normalize_occurrence_index(occurrence_index: int | None, occurrence_count: int) -> int | None:
    if occurrence_index is None:
        return None
    try:
        selected = int(occurrence_index)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid_occurrence_index") from exc
    if selected < 1 or selected > occurrence_count:
        raise ValueError("invalid_occurrence_index")
    return selected


def _apply_exact_patch_plan(
    text: str,
    target_text: str,
    replacement_text: str,
    *,
    expected_occurrences: int = 1,
    occurrence_index: int | None = None,
    fromfile: str = "before",
    tofile: str = "after",
    encoding: str = "utf-8",
    diff_limit: int = 8000,
) -> PatchPlan:
    match_spans = (
        [(match.start(), match.end()) for match in re.finditer(re.escape(target_text), text)]
        if target_text
        else []
    )
    actual_occurrences = len(match_spans)
    if actual_occurrences != expected_occurrences:
        raise ValueError(str(actual_occurrences))

    selected_occurrence = _normalize_occurrence_index(occurrence_index, actual_occurrences)
    if selected_occurrence is None:
        new_text = text.replace(target_text, replacement_text, expected_occurrences)
        replacement_count = actual_occurrences
    else:
        start, end = match_spans[selected_occurrence - 1]
        new_text = text[:start] + replacement_text + text[end:]
        replacement_count = 1

    diff = _build_unified_patch_diff(text, new_text, fromfile=fromfile, tofile=tofile)
    diff_preview, diff_clipped = clip_text_value(diff, limit=diff_limit)
    return PatchPlan(
        mode="exact",
        old_text=text,
        new_text=new_text,
        occurrence_count=actual_occurrences,
        replacement_count=replacement_count,
        match_spans=match_spans,
        selected_occurrence=selected_occurrence,
        diff=diff,
        diff_preview=diff_preview,
        diff_clipped=diff_clipped,
        old_sha256=_text_sha256(text, encoding=encoding),
        new_sha256=_text_sha256(new_text, encoding=encoding),
    )


def _compile_regex_patch_pattern(
    pattern_text: str,
    *,
    case_insensitive: bool = False,
    multiline: bool = False,
    dotall: bool = False,
) -> re.Pattern[str]:
    flags = 0
    if case_insensitive:
        flags |= re.IGNORECASE
    if multiline:
        flags |= re.MULTILINE
    if dotall:
        flags |= re.DOTALL
    return re.compile(pattern_text, flags)


def _apply_regex_patch_plan(
    text: str,
    pattern_text: str,
    replacement_text: str,
    *,
    expected_occurrences: int = 1,
    occurrence_index: int | None = None,
    case_insensitive: bool = False,
    multiline: bool = False,
    dotall: bool = False,
    fromfile: str = "before",
    tofile: str = "after",
    encoding: str = "utf-8",
    diff_limit: int = 8000,
) -> PatchPlan:
    pattern = _compile_regex_patch_pattern(
        pattern_text,
        case_insensitive=case_insensitive,
        multiline=multiline,
        dotall=dotall,
    )
    matches = list(pattern.finditer(text))
    if any(match.start() == match.end() for match in matches):
        raise ValueError("empty_regex_match")
    match_spans = [(match.start(), match.end()) for match in matches]
    actual_occurrences = len(match_spans)
    if actual_occurrences != expected_occurrences:
        raise ValueError(str(actual_occurrences))

    selected_occurrence = _normalize_occurrence_index(occurrence_index, actual_occurrences)
    if selected_occurrence is None:
        new_text, replacement_count = pattern.subn(replacement_text, text, count=expected_occurrences)
    else:
        selected_match = matches[selected_occurrence - 1]
        start, end = selected_match.span()
        new_text = text[:start] + selected_match.expand(replacement_text) + text[end:]
        replacement_count = 1

    diff = _build_unified_patch_diff(text, new_text, fromfile=fromfile, tofile=tofile)
    diff_preview, diff_clipped = clip_text_value(diff, limit=diff_limit)
    return PatchPlan(
        mode="regex",
        old_text=text,
        new_text=new_text,
        occurrence_count=actual_occurrences,
        replacement_count=replacement_count,
        match_spans=match_spans,
        selected_occurrence=selected_occurrence,
        diff=diff,
        diff_preview=diff_preview,
        diff_clipped=diff_clipped,
        old_sha256=_text_sha256(text, encoding=encoding),
        new_sha256=_text_sha256(new_text, encoding=encoding),
    )


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
            "The target text was not found. This tool uses exact character-for-character matching (not regex). "
            "Read the smallest relevant slice first, then retry with an exact substring copied verbatim from the current file contents."
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
    regex: bool = False,
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
        hint = (
            " This tool requires an exact character-for-character match; it does not use regex."
            if not staged_only and not regex
            else ""
        )
        return f"Patch target text was not found in `{requested_path}`.{hint}"
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
    plan: PatchPlan | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    from .fs import _read_text_file

    measured_text = plan.new_text if plan is not None else _read_text_file(source_path, encoding=encoding)
    metadata: dict[str, Any] = {
        "path": str(path),
        "requested_path": requested_path,
        "bytes": len(measured_text.encode(encoding)),
        "target_text_bytes": len(target_text.encode(encoding)),
        "replacement_text_bytes": len(replacement_text.encode(encoding)),
        "occurrence_count": occurrence_count,
        "expected_occurrences": expected_occurrences,
        "source_path": str(source_path),
        "staged_only": staged_only,
        "changed": not dry_run,
        "target_text_preview": _build_patch_text_preview(target_text),
        "replacement_text_preview": _build_patch_text_preview(replacement_text),
    }
    if staging_path is not None:
        metadata["staging_path"] = str(staging_path)
    if session is not None:
        metadata["write_session_id"] = str(getattr(session, "write_session_id", "") or "").strip()
        metadata["write_session_status_block"] = status_block or ""
    if plan is not None:
        metadata.update(_build_patch_diff_metadata(plan, dry_run=dry_run))
    return metadata
