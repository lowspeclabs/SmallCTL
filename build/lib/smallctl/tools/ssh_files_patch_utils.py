from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from typing import Any


def preview_text(value: str, *, limit: int = 160) -> dict[str, Any]:
    text = str(value or "")
    clipped = len(text) > limit
    return {
        "preview": text[:limit],
        "chars": len(text),
        "truncated": clipped,
    }


def sha256_text(value: str, encoding: str) -> str:
    return hashlib.sha256(str(value).encode(encoding)).hexdigest()


def preview_match_context(content: str, start: int, end: int, *, limit: int = 240) -> dict[str, Any]:
    start_idx = max(0, start)
    end_idx = min(len(content), end)
    segment = content[start_idx:end_idx]
    if len(segment) <= limit:
        return {
            "preview": segment,
            "chars": len(segment),
            "truncated": False,
            "start": start_idx,
            "end": end_idx,
        }
    head = segment[: limit // 2]
    tail = segment[-(limit - len(head)) :]
    return {
        "preview": head + "\n...\n" + tail,
        "chars": len(segment),
        "truncated": True,
        "start": start_idx,
        "end": end_idx,
    }


def best_patch_match(content: str, target_text: str, *, limit: int = 240) -> dict[str, Any] | None:
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
            best_score = score
            best = {
                **preview_match_context(snippet, 0, len(snippet), limit=limit),
                "start_line": start_index + 1,
                "end_line": start_index + len(snippet_lines),
                "similarity": round(score, 3),
                "match_basis": "whitespace_normalized" if normalized_score > exact_score else "exact",
            }
    return best


def normalize_whitespace_with_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
    normalized_chars: list[str] = []
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            normalized_chars.append(" ")
            spans.append((index, end))
            index = end
            continue
        normalized_chars.append(char)
        spans.append((index, index + 1))
        index += 1
    return "".join(normalized_chars), spans


def find_whitespace_normalized_spans(
    content: str,
    needle: str,
) -> list[tuple[int, int]]:
    normalized_content, spans = normalize_whitespace_with_spans(content)
    normalized_needle, _needle_spans = normalize_whitespace_with_spans(needle)
    if not normalized_needle:
        return []

    matches: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = normalized_content.find(normalized_needle, cursor)
        if start == -1:
            break
        end = start + len(normalized_needle)
        original_start = spans[start][0]
        original_end = spans[end - 1][1]
        matches.append((original_start, original_end))
        cursor = end
    return matches


def find_whitespace_normalized_bounded_regions(
    content: str,
    *,
    start_text: str,
    end_text: str,
    include_bounds: bool = True,
) -> list[tuple[int, int]]:
    normalized_content, spans = normalize_whitespace_with_spans(content)
    normalized_start, _start_spans = normalize_whitespace_with_spans(start_text)
    normalized_end, _end_spans = normalize_whitespace_with_spans(end_text)
    if not normalized_start or not normalized_end:
        return []

    regions: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start_norm = normalized_content.find(normalized_start, cursor)
        if start_norm == -1:
            break
        end_start_norm = normalized_content.find(normalized_end, start_norm + len(normalized_start))
        if end_start_norm == -1:
            break
        start_norm_end = start_norm + len(normalized_start)
        end_norm_end = end_start_norm + len(normalized_end)
        if include_bounds:
            original_start = spans[start_norm][0]
            original_end = spans[end_norm_end - 1][1]
        else:
            original_start = spans[start_norm_end - 1][1]
            original_end = spans[end_start_norm][0]
        regions.append((original_start, original_end))
        cursor = end_norm_end
    return regions


def apply_exact_patch_content(
    content: str,
    *,
    target_text: str,
    replacement_text: str,
    expected_occurrences: int = 1,
    whitespace_normalized: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    match_spans = (
        find_whitespace_normalized_spans(content, target_text)
        if whitespace_normalized
        else []
    )
    actual = len(match_spans) if whitespace_normalized else content.count(target_text)
    metadata = {
        "actual_occurrences": actual,
        "expected_occurrences": expected_occurrences,
        "target_text_preview": preview_text(target_text),
        "replacement_text_preview": preview_text(replacement_text),
        "match_mode": "whitespace_normalized" if whitespace_normalized else "exact",
    }
    if actual == 0:
        metadata["error_kind"] = "patch_target_not_found"
        metadata["ambiguity_hint"] = (
            "Read the remote file and copy the exact target text, including whitespace."
            if not whitespace_normalized
            else "No whitespace-normalized patch target matched. Read the remote file and run a dry-run first."
        )
        metadata["best_match"] = best_patch_match(content, target_text)
        return False, content, metadata
    if actual != expected_occurrences:
        metadata["error_kind"] = "patch_occurrence_mismatch"
        metadata["ambiguity_hint"] = "Use a more specific exact target or set expected_occurrences to the intended count."
        return False, content, metadata
    if not whitespace_normalized:
        return True, content.replace(target_text, replacement_text, expected_occurrences), metadata

    updated = content
    for start, end in reversed(match_spans):
        updated = updated[:start] + replacement_text + updated[end:]
    metadata["matched_region_previews"] = [
        preview_match_context(content, start, end) for start, end in match_spans[:3]
    ]
    return True, updated, metadata


def find_bounded_regions(
    content: str,
    *,
    start_text: str,
    end_text: str,
    include_bounds: bool = True,
) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = content.find(start_text, cursor)
        if start == -1:
            break
        end_start = content.find(end_text, start + len(start_text))
        if end_start == -1:
            break
        end = end_start + len(end_text)
        if include_bounds:
            regions.append((start, end))
        else:
            regions.append((start + len(start_text), end_start))
        cursor = end
    return regions


def apply_replace_between_content(
    content: str,
    *,
    start_text: str,
    end_text: str,
    replacement_text: str,
    include_bounds: bool = False,
    expected_occurrences: int = 1,
    whitespace_normalized: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    regions = (
        find_whitespace_normalized_bounded_regions(
            content,
            start_text=start_text,
            end_text=end_text,
            include_bounds=include_bounds,
        )
        if whitespace_normalized
        else find_bounded_regions(
            content,
            start_text=start_text,
            end_text=end_text,
            include_bounds=include_bounds,
        )
    )
    actual = len(regions)
    metadata = {
        "actual_occurrences": actual,
        "expected_occurrences": expected_occurrences,
        "start_text_preview": preview_text(start_text),
        "end_text_preview": preview_text(end_text),
        "replacement_text_preview": preview_text(replacement_text),
        "include_bounds": include_bounds,
        "match_mode": "whitespace_normalized" if whitespace_normalized else "exact",
    }
    if actual == 0:
        metadata["error_kind"] = "bounded_region_not_found"
        if whitespace_normalized:
            norm_content, _ = normalize_whitespace_with_spans(content)
            norm_start, _ = normalize_whitespace_with_spans(start_text)
            norm_end, _ = normalize_whitespace_with_spans(end_text)
            start_found = norm_start in norm_content if norm_start else False
            end_found = norm_end in norm_content if norm_end else False
        else:
            start_found = start_text in content if start_text else False
            end_found = end_text in content if end_text else False
        metadata["start_text_found"] = start_found
        metadata["end_text_found"] = end_found
        if not start_found:
            metadata["start_text_best_match"] = best_patch_match(content, start_text)
        if not end_found:
            metadata["end_text_best_match"] = best_patch_match(content, end_text)
        if start_found and not end_found:
            metadata["ambiguity_hint"] = (
                "start_text was found but end_text was not found after it. "
                "Check the exact text between them and verify end_text exists in the file."
            )
        elif not start_found and end_found:
            metadata["ambiguity_hint"] = (
                "end_text was found but start_text was not found before it. "
                "Check the exact text before end_text and verify start_text exists in the file."
            )
        elif not start_found and not end_found:
            metadata["ambiguity_hint"] = (
                "Neither start_text nor end_text were found. "
                "Read the remote file and copy the exact bounds, including whitespace."
            )
        else:
            metadata["ambiguity_hint"] = (
                "Both start_text and end_text exist but not in the expected order. "
                "Verify the bounds appear in order in the file."
            )
        return False, content, metadata
    if actual != expected_occurrences:
        metadata["error_kind"] = "patch_occurrence_mismatch"
        metadata["ambiguity_hint"] = "Use more specific bounds or set expected_occurrences to the intended count."
        return False, content, metadata

    updated = content
    for start, end in reversed(regions):
        updated = updated[:start] + replacement_text + updated[end:]
    metadata["matched_region_previews"] = [
        preview_match_context(content, start, end) for start, end in regions[:3]
    ]
    return True, updated, metadata
