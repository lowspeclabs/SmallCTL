from __future__ import annotations

import base64
from difflib import SequenceMatcher
import hashlib
import json
import re
import shlex
from typing import Any

from ..state import LoopState
from .common import fail, ok
from . import network


SSH_FILE_MUTATING_TOOLS = {"ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}
REMOTE_MUTATION_VERIFICATION_KEY = "_remote_mutation_requires_verification"

# When the base64-encoded JSON payload exceeds this size, switch from passing it
# as a command-line argument (which is bounded by ARG_MAX / MAX_ARG_STRLEN) to
# piping it through the SSH process's stdin.
_MAX_ARGV_PAYLOAD_SIZE = 128 * 1024


def _preview_text(value: str, *, limit: int = 160) -> dict[str, Any]:
    text = str(value or "")
    clipped = len(text) > limit
    return {
        "preview": text[:limit],
        "chars": len(text),
        "truncated": clipped,
    }


def _sha256_text(value: str, encoding: str) -> str:
    return hashlib.sha256(str(value).encode(encoding)).hexdigest()


def _preview_match_context(content: str, start: int, end: int, *, limit: int = 240) -> dict[str, Any]:
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


def _best_patch_match(content: str, target_text: str, *, limit: int = 240) -> dict[str, Any] | None:
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
                **_preview_match_context(snippet, 0, len(snippet), limit=limit),
                "start_line": start_index + 1,
                "end_line": start_index + len(snippet_lines),
                "similarity": round(score, 3),
                "match_basis": "whitespace_normalized" if normalized_score > exact_score else "exact",
            }
    return best


def _normalize_whitespace_with_spans(text: str) -> tuple[str, list[tuple[int, int]]]:
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


def _find_whitespace_normalized_spans(
    content: str,
    needle: str,
) -> list[tuple[int, int]]:
    normalized_content, spans = _normalize_whitespace_with_spans(content)
    normalized_needle, _needle_spans = _normalize_whitespace_with_spans(needle)
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


def _find_whitespace_normalized_bounded_regions(
    content: str,
    *,
    start_text: str,
    end_text: str,
    include_bounds: bool = True,
) -> list[tuple[int, int]]:
    normalized_content, spans = _normalize_whitespace_with_spans(content)
    normalized_start, _start_spans = _normalize_whitespace_with_spans(start_text)
    normalized_end, _end_spans = _normalize_whitespace_with_spans(end_text)
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
        _find_whitespace_normalized_spans(content, target_text)
        if whitespace_normalized
        else []
    )
    actual = len(match_spans) if whitespace_normalized else content.count(target_text)
    metadata = {
        "actual_occurrences": actual,
        "expected_occurrences": expected_occurrences,
        "target_text_preview": _preview_text(target_text),
        "replacement_text_preview": _preview_text(replacement_text),
        "match_mode": "whitespace_normalized" if whitespace_normalized else "exact",
    }
    if actual == 0:
        metadata["error_kind"] = "patch_target_not_found"
        metadata["ambiguity_hint"] = (
            "Read the remote file and copy the exact target text, including whitespace."
            if not whitespace_normalized
            else "No whitespace-normalized patch target matched. Read the remote file and run a dry-run first."
        )
        metadata["best_match"] = _best_patch_match(content, target_text)
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
        _preview_match_context(content, start, end) for start, end in match_spans[:3]
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
    include_bounds: bool = True,
    expected_occurrences: int = 1,
    whitespace_normalized: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    regions = (
        _find_whitespace_normalized_bounded_regions(
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
        "start_text_preview": _preview_text(start_text),
        "end_text_preview": _preview_text(end_text),
        "replacement_text_preview": _preview_text(replacement_text),
        "include_bounds": include_bounds,
        "match_mode": "whitespace_normalized" if whitespace_normalized else "exact",
    }
    if actual == 0:
        metadata["error_kind"] = "bounded_region_not_found"
        metadata["ambiguity_hint"] = (
            "Read the remote file and verify both start_text and end_text exist in order."
            if not whitespace_normalized
            else "No whitespace-normalized bounded region matched. Read the remote file and run a dry-run first."
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
        _preview_match_context(content, start, end) for start, end in regions[:3]
    ]
    return True, updated, metadata


_REMOTE_HELPER_SOURCE = r"""
import base64, difflib, hashlib, json, os, pathlib, shutil, signal, sys, tempfile, time, traceback

def emit(payload):
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()

def preview(value, limit=160):
    text = str(value or "")
    return {"preview": text[:limit], "chars": len(text), "truncated": len(text) > limit}

def preview_match_context(content, start, end, limit=240):
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

def best_patch_match(content, target_text, limit=240):
    if not content or not target_text:
        return None
    content_lines = content.splitlines()
    if not content_lines:
        return None
    target_lines = target_text.splitlines() or [target_text]
    window_sizes = sorted({max(1, len(target_lines) - 1), max(1, len(target_lines)), len(target_lines) + 1})
    normalized_target = " ".join(target_text.split())
    best = None
    best_score = -1.0
    for window_size in window_sizes:
        for start_index in range(0, max(1, len(content_lines) - window_size + 1)):
            snippet_lines = content_lines[start_index : start_index + window_size]
            if not snippet_lines:
                continue
            snippet = "\n".join(snippet_lines)
            exact_score = difflib.SequenceMatcher(None, target_text, snippet).ratio()
            normalized_snippet = " ".join(snippet.split())
            normalized_score = (
                difflib.SequenceMatcher(None, normalized_target, normalized_snippet).ratio()
                if normalized_target and normalized_snippet
                else 0.0
            )
            score = max(exact_score, normalized_score)
            if score <= best_score:
                continue
            best_score = score
            best = preview_match_context(snippet, 0, len(snippet), limit=limit)
            best.update({
                "start_line": start_index + 1,
                "end_line": start_index + len(snippet_lines),
                "similarity": round(score, 3),
                "match_basis": "whitespace_normalized" if normalized_score > exact_score else "exact",
            })
    return best

def normalize_whitespace_with_spans(text):
    normalized_chars = []
    spans = []
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

def find_whitespace_normalized_spans(content, needle):
    normalized_content, spans = normalize_whitespace_with_spans(content)
    normalized_needle, _needle_spans = normalize_whitespace_with_spans(needle)
    if not normalized_needle:
        return []
    matches = []
    cursor = 0
    while True:
        start = normalized_content.find(normalized_needle, cursor)
        if start == -1:
            break
        end = start + len(normalized_needle)
        matches.append((spans[start][0], spans[end - 1][1]))
        cursor = end
    return matches

def find_whitespace_normalized_bounded_regions(content, start_text, end_text, include_bounds):
    normalized_content, spans = normalize_whitespace_with_spans(content)
    normalized_start, _start_spans = normalize_whitespace_with_spans(start_text)
    normalized_end, _end_spans = normalize_whitespace_with_spans(end_text)
    if not normalized_start or not normalized_end:
        return []
    regions = []
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

def read_file(payload):
    path = pathlib.Path(payload["path"])
    encoding = payload.get("encoding") or "utf-8"
    max_bytes = int(payload.get("max_bytes") or 262144)
    truncate = bool(payload.get("truncate", True))
    if not path.exists():
        return {"ok": False, "error_kind": "file_not_found", "path": str(path), "message": "Remote file not found."}
    if not path.is_file():
        return {"ok": False, "error_kind": "not_a_regular_file", "path": str(path), "message": "Remote path is not a regular file."}
    try:
        data = path.read_bytes()
    except PermissionError:
        return {"ok": False, "error_kind": "permission_denied", "path": str(path), "message": "Permission denied reading remote file."}
    except IsADirectoryError:
        return {"ok": False, "error_kind": "is_a_directory", "path": str(path), "message": "Remote path is a directory."}
    except OSError as exc:
        return {"ok": False, "error_kind": "read_error", "path": str(path), "message": str(exc)}
    truncated = len(data) > max_bytes
    if truncated and not truncate:
        return {
            "ok": False,
            "error_kind": "max_size_exceeded",
            "path": str(path),
            "bytes": len(data),
            "max_bytes": max_bytes,
            "message": "Remote file exceeded max_bytes and truncation was disabled.",
        }
    visible = data[:max_bytes] if truncated else data
    try:
        content = visible.decode(encoding)
    except UnicodeDecodeError as exc:
        return {"ok": False, "error_kind": "decode_failure", "path": str(path), "encoding": encoding, "message": str(exc)}
    return {
        "ok": True,
        "path": str(path),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "content": content,
        "truncated": truncated,
        "encoding": encoding,
    }

def atomic_write(path, data, payload):
    mode = payload.get("mode") or "overwrite"
    create_parent_dirs = bool(payload.get("create_parent_dirs"))
    backup = bool(payload.get("backup"))
    expected_sha256 = str(payload.get("expected_sha256") or "").strip()
    path = pathlib.Path(path)
    parent = path.parent
    if create_parent_dirs:
        parent.mkdir(parents=True, exist_ok=True)
    if not parent.exists():
        return {"ok": False, "error_kind": "parent_not_found", "path": str(path), "message": "Remote parent directory does not exist."}
    existed = path.exists()
    if mode == "create" and existed:
        return {"ok": False, "error_kind": "file_exists", "path": str(path), "message": "Remote file already exists."}
    if mode not in {"overwrite", "create", "append"}:
        return {"ok": False, "error_kind": "unsupported_write_mode", "path": str(path), "mode": mode, "message": "Unsupported remote file write mode."}
    old_data = b""
    old_sha = None
    old_stat = None
    if existed:
        try:
            old_data = path.read_bytes()
            old_sha = sha256_bytes(old_data)
            old_stat = path.stat()
        except PermissionError:
            return {"ok": False, "error_kind": "permission_denied", "path": str(path), "message": "Permission denied reading current remote file."}
    if expected_sha256 and old_sha != expected_sha256:
        return {"ok": False, "error_kind": "expected_sha256_mismatch", "path": str(path), "old_sha256": old_sha, "expected_sha256": expected_sha256, "message": "Current remote file hash did not match expected_sha256."}
    if mode == "append":
        data = old_data + data
    backup_path = None
    if backup and existed:
        backup_path = str(path) + ".bak." + time.strftime("%Y%m%d%H%M%S")
        shutil.copy2(path, backup_path)
    fd, tmp_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if old_stat is not None:
            os.chmod(tmp_name, old_stat.st_mode & 0o7777)
            try:
                os.chown(tmp_name, old_stat.st_uid, old_stat.st_gid)
            except PermissionError:
                pass
        os.replace(tmp_name, path)
    except PermissionError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        return {"ok": False, "error_kind": "permission_denied", "path": str(path), "message": "Permission denied writing remote file."}
    except Exception as exc:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        return {"ok": False, "error_kind": "remote_write_failed", "path": str(path), "message": str(exc)}
    readback = path.read_bytes()
    new_sha = sha256_bytes(data)
    readback_sha = sha256_bytes(readback)
    if readback_sha != new_sha:
        return {"ok": False, "error_kind": "readback_mismatch", "path": str(path), "new_sha256": new_sha, "readback_sha256": readback_sha, "message": "Remote readback hash did not match intended content."}
    return {
        "ok": True,
        "path": str(path),
        "bytes_written": len(data),
        "old_sha256": old_sha,
        "new_sha256": new_sha,
        "backup_path": backup_path,
        "changed": old_data != data,
        "readback_sha256": readback_sha,
    }

def apply_patch_content(content, payload):
    target = payload.get("target_text") or ""
    replacement = payload.get("replacement_text") or ""
    expected = int(payload.get("expected_occurrences") or 1)
    whitespace_normalized = bool(payload.get("whitespace_normalized"))
    match_spans = find_whitespace_normalized_spans(content, target) if whitespace_normalized else []
    actual = len(match_spans) if whitespace_normalized else content.count(target)
    meta = {
        "actual_occurrences": actual,
        "expected_occurrences": expected,
        "target_text_preview": preview(target),
        "replacement_text_preview": preview(replacement),
        "match_mode": "whitespace_normalized" if whitespace_normalized else "exact",
    }
    if actual == 0:
        meta.update({
            "ok": False,
            "error_kind": "patch_target_not_found",
            "message": "Remote patch target text was not found.",
            "ambiguity_hint": (
                "Read the remote file and copy the exact target text."
                if not whitespace_normalized
                else "No whitespace-normalized patch target matched. Read the remote file and run a dry-run first."
            ),
            "best_match": best_patch_match(content, target),
        })
        return content, meta
    if actual != expected:
        meta.update({"ok": False, "error_kind": "patch_occurrence_mismatch", "message": "Remote patch target occurrence count did not match expected_occurrences.", "ambiguity_hint": "Use a more specific target or update expected_occurrences."})
        return content, meta
    if whitespace_normalized:
        updated = content
        for start, end in reversed(match_spans):
            updated = updated[:start] + replacement + updated[end:]
        meta["matched_region_previews"] = [preview_match_context(content, start, end) for start, end in match_spans[:3]]
        meta["ok"] = True
        return updated, meta
    meta["ok"] = True
    return content.replace(target, replacement, expected), meta

def bounded_regions(content, start_text, end_text, include_bounds):
    regions = []
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

def apply_replace_between_content(content, payload):
    start_text = payload.get("start_text") or ""
    end_text = payload.get("end_text") or ""
    replacement = payload.get("replacement_text") or ""
    include_bounds = bool(payload.get("include_bounds", True))
    expected = int(payload.get("expected_occurrences") or 1)
    whitespace_normalized = bool(payload.get("whitespace_normalized"))
    regions = (
        find_whitespace_normalized_bounded_regions(content, start_text, end_text, include_bounds)
        if whitespace_normalized
        else bounded_regions(content, start_text, end_text, include_bounds)
    )
    actual = len(regions)
    meta = {
        "actual_occurrences": actual,
        "expected_occurrences": expected,
        "start_text_preview": preview(start_text),
        "end_text_preview": preview(end_text),
        "replacement_text_preview": preview(replacement),
        "include_bounds": include_bounds,
        "match_mode": "whitespace_normalized" if whitespace_normalized else "exact",
    }
    if actual == 0:
        meta.update({
            "ok": False,
            "error_kind": "bounded_region_not_found",
            "message": "Remote bounded region was not found.",
            "ambiguity_hint": (
                "Verify both start_text and end_text exist in order."
                if not whitespace_normalized
                else "No whitespace-normalized bounded region matched. Read the remote file and run a dry-run first."
            ),
        })
        return content, meta
    if actual != expected:
        meta.update({"ok": False, "error_kind": "patch_occurrence_mismatch", "message": "Remote bounded region count did not match expected_occurrences.", "ambiguity_hint": "Use more specific bounds or update expected_occurrences."})
        return content, meta
    updated = content
    for start, end in reversed(regions):
        updated = updated[:start] + replacement + updated[end:]
    meta["matched_region_previews"] = [preview_match_context(content, start, end) for start, end in regions[:3]]
    meta["ok"] = True
    return updated, meta

def main():
    try:
        if hasattr(signal, "alarm"):
            signal.alarm(60)
        if len(sys.argv) > 1 and sys.argv[1] == "--stdin":
            raw = sys.stdin.read()
        else:
            raw = sys.argv[1]
        payload = json.loads(base64.b64decode(raw).decode("utf-8"))
        action = payload.get("action")
        if action == "read":
            emit(read_file(payload))
            return
        if action == "write":
            encoding = payload.get("encoding") or "utf-8"
            content = payload.get("content") or ""
            emit(atomic_write(payload["path"], content.encode(encoding), payload))
            return
        read_result = read_file({**payload, "max_bytes": 1 << 60})
        if not read_result.get("ok"):
            read_result["error_kind"] = "remote_read_failed"
            emit(read_result)
            return
        encoding = payload.get("encoding") or "utf-8"
        content = read_result.get("content") or ""
        old_sha = read_result.get("sha256")
        expected_sha256 = str(payload.get("expected_sha256") or "").strip()
        if expected_sha256 and old_sha != expected_sha256:
            emit({
                "ok": False,
                "error_kind": "expected_sha256_mismatch",
                "path": payload.get("path"),
                "old_sha256": old_sha,
                "expected_sha256": expected_sha256,
                "message": "Current remote file hash did not match expected_sha256.",
            })
            return
        if action == "patch":
            updated, meta = apply_patch_content(content, payload)
        elif action == "replace_between":
            updated, meta = apply_replace_between_content(content, payload)
        else:
            emit({"ok": False, "error_kind": "unsupported_action", "message": "Unsupported SSH file helper action."})
            return
        if not meta.get("ok"):
            meta.update({"path": payload.get("path"), "old_sha256": old_sha})
            emit(meta)
            return
        replacement = payload.get("replacement_text") or ""
        verification = {
            "replacement_occurrences": updated.count(replacement) if replacement else 0,
        }
        if action == "patch":
            verification["target_occurrences_after"] = updated.count(payload.get("target_text") or "")
        dry_run = bool(payload.get("dry_run"))
        if dry_run:
            emit({
                **{k: v for k, v in meta.items() if k != "ok"},
                "ok": True,
                "path": payload.get("path"),
                "old_sha256": old_sha,
                "planned_new_sha256": sha256_bytes(updated.encode(encoding)),
                "changed": updated != content,
                "dry_run": True,
                "verification": verification,
            })
            return
        write_result = atomic_write(payload["path"], updated.encode(encoding), {**payload, "mode": "overwrite", "expected_sha256": old_sha})
        if not write_result.get("ok"):
            write_result["error_kind"] = write_result.get("error_kind") or "remote_write_failed"
            write_result.update({k: v for k, v in meta.items() if k != "ok"})
            emit(write_result)
            return
        verification["readback_sha256_matches"] = write_result.get("readback_sha256") == write_result.get("new_sha256")
        write_result.update({k: v for k, v in meta.items() if k != "ok"})
        write_result["old_sha256"] = old_sha
        write_result["verification"] = verification
        emit(write_result)
    except Exception as exc:
        emit({"ok": False, "error_kind": "remote_helper_exception", "message": str(exc), "traceback": traceback.format_exc(limit=4)})
    finally:
        if hasattr(signal, "alarm"):
            signal.alarm(0)

main()
"""


def _build_remote_command(payload: dict[str, Any]) -> tuple[str, str | None]:
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=True).encode("utf-8")).decode("ascii")
    if len(encoded) > _MAX_ARGV_PAYLOAD_SIZE:
        command = "python3 -c " + shlex.quote(_REMOTE_HELPER_SOURCE) + " --stdin"
        return command, encoded
    return "python3 -c " + shlex.quote(_REMOTE_HELPER_SOURCE) + " " + shlex.quote(encoded), None


def _normalize_remote_connection(
    *,
    target: str | None,
    host: str | None,
    user: str | None,
    port: int,
    identity_file: str | None,
    password: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        normalized = network.normalize_ssh_arguments(
            {
                "target": target,
                "host": host,
                "user": user,
                "port": port,
                "identity_file": identity_file,
                "password": password,
            }
        )
    except ValueError as exc:
        return None, {"reason": "invalid_ssh_target", "message": str(exc)}
    return normalized, None


async def _run_remote_file_action(
    *,
    action: str,
    path: str,
    payload: dict[str, Any],
    target: str | None,
    host: str | None,
    user: str | None,
    port: int,
    identity_file: str | None,
    password: str | None,
    timeout_sec: int,
    state: LoopState | None,
    harness: Any,
) -> dict[str, Any]:
    connection, error = _normalize_remote_connection(
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
    )
    if error is not None or connection is None:
        return fail(error.get("message", "Invalid SSH target."), metadata=error or {})

    helper_payload = {
        **payload,
        "action": action,
        "path": path,
    }
    command, stdin_payload = _build_remote_command(helper_payload)
    result = await network.run_ssh_command(
        host=str(connection.get("host") or ""),
        user=connection.get("user"),
        port=int(connection.get("port") or 22),
        identity_file=connection.get("identity_file"),
        password=connection.get("password"),
        command=command,
        stdin_data=stdin_payload,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )
    if not result.get("success"):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        output = metadata.get("output") if isinstance(metadata.get("output"), dict) else {}
        stderr = str(output.get("stderr") or result.get("error") or "")
        reason = "remote_python_missing" if "python3" in stderr.lower() and "not found" in stderr.lower() else "remote_helper_failed"
        return fail(
            result.get("error") or "Remote SSH file helper failed.",
            metadata={
                "path": path,
                "host": connection.get("host"),
                "reason": reason,
                "recovery_hint": "Install python3 on the remote host or use explicit ssh_exec." if reason == "remote_python_missing" else "Inspect remote SSH helper output.",
                "ssh_result": result,
            },
        )

    output = result.get("output") if isinstance(result.get("output"), dict) else {}
    stdout = str(output.get("stdout") or "").strip()
    if not stdout:
        return fail(
            "Remote SSH file helper produced no JSON output.",
            metadata={"path": path, "host": connection.get("host"), "error_kind": "remote_helper_no_output"},
        )
    try:
        data = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return fail(
            f"Remote SSH file helper produced invalid JSON: {exc}",
            metadata={"path": path, "host": connection.get("host"), "error_kind": "remote_helper_invalid_json", "stdout": stdout[-1000:]},
        )
    if not isinstance(data, dict):
        return fail(
            "Remote SSH file helper returned a non-object JSON payload.",
            metadata={"path": path, "host": connection.get("host"), "error_kind": "remote_helper_invalid_json"},
        )

    data.setdefault("path", path)
    data["host"] = connection.get("host")
    if connection.get("user"):
        data["user"] = connection.get("user")
    data["tool_generated_remote_command"] = True
    if not data.get("ok"):
        message = str(data.get("message") or data.get("error") or "Remote SSH file operation failed.")
        data.pop("ok", None)
        return fail(message, metadata=data)

    data.pop("ok", None)
    metadata = {k: v for k, v in data.items() if k != "content"}
    if action == "read":
        metadata.setdefault("complete_file", not data.get("truncated", False))
        content = data.get("content", "")
        if isinstance(content, str):
            metadata.setdefault("total_lines", content.count("\n") + (1 if content and not content.endswith("\n") else 0))
    return ok(data, metadata=metadata)


def _clear_remote_mutation_requirement(state: LoopState | None, *, path: str, host: str) -> None:
    if state is None:
        return
    requirement = state.scratchpad.get(REMOTE_MUTATION_VERIFICATION_KEY)
    if not isinstance(requirement, dict):
        return
    guessed_paths = [str(item) for item in requirement.get("guessed_paths", []) if str(item).strip()]
    requirement_host = str(requirement.get("host") or "").strip().lower()
    if requirement_host and host and requirement_host != host.strip().lower():
        return
    if guessed_paths and path not in guessed_paths:
        return
    state.scratchpad.pop(REMOTE_MUTATION_VERIFICATION_KEY, None)


def _sha_from_artifact(artifact: Any) -> str:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("sha256", "new_sha256", "readback_sha256", "old_sha256"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def _artifact_path_for_precondition(artifact: Any) -> str:
    metadata = getattr(artifact, "metadata", {})
    if isinstance(metadata, dict):
        path = str(metadata.get("path") or "").strip()
        if path:
            return path
    return str(getattr(artifact, "source", "") or "").strip()


def _resolve_expected_sha_precondition(
    *,
    path: str,
    expected_sha256: str | None,
    source_artifact_id: str | None,
    state: LoopState | None,
) -> tuple[str | None, dict[str, Any] | None]:
    explicit = str(expected_sha256 or "").strip()
    artifact_id = str(source_artifact_id or "").strip()
    if not artifact_id:
        return (explicit or None), None
    if state is None:
        return None, {
            "reason": "source_artifact_unavailable",
            "message": "source_artifact_id requires harness state with artifact metadata.",
        }
    artifacts = getattr(state, "artifacts", {})
    if not isinstance(artifacts, dict):
        return None, {
            "reason": "source_artifact_unavailable",
            "message": "source_artifact_id requires an artifact registry in state.",
        }
    artifact = artifacts.get(artifact_id)
    if artifact is None:
        return None, {
            "reason": "source_artifact_missing",
            "message": f"Artifact `{artifact_id}` was not found in state.",
        }
    artifact_sha = _sha_from_artifact(artifact)
    if not artifact_sha:
        return None, {
            "reason": "source_artifact_missing_sha",
            "message": f"Artifact `{artifact_id}` does not carry a usable sha256 precondition.",
        }
    artifact_path = _artifact_path_for_precondition(artifact)
    if artifact_path and str(artifact_path).strip() != str(path).strip():
        return None, {
            "reason": "source_artifact_path_mismatch",
            "message": (
                f"Artifact `{artifact_id}` describes `{artifact_path}`, which does not match target path `{path}`."
            ),
        }
    if explicit and explicit != artifact_sha:
        return None, {
            "reason": "expected_sha256_mismatch",
            "message": "expected_sha256 did not match the hash referenced by source_artifact_id.",
        }
    return artifact_sha, {
        "resolved_expected_sha256": artifact_sha,
        "source_artifact_id": artifact_id,
    }


async def ssh_file_read(
    path: str,
    target: str | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    encoding: str = "utf-8",
    max_bytes: int = 262144,
    truncate: bool = True,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    return await _run_remote_file_action(
        action="read",
        path=path,
        payload={"encoding": encoding, "max_bytes": max_bytes, "truncate": truncate},
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )


async def ssh_file_write(
    path: str,
    content: str,
    target: str | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    encoding: str = "utf-8",
    mode: str = "overwrite",
    create_parent_dirs: bool = False,
    backup: bool = True,
    expected_sha256: str | None = None,
    source_artifact_id: str | None = None,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    intended_sha = _sha256_text(content, encoding)
    resolved_expected_sha256, precondition_metadata = _resolve_expected_sha_precondition(
        path=path,
        expected_sha256=expected_sha256,
        source_artifact_id=source_artifact_id,
        state=state,
    )
    if precondition_metadata is not None and "resolved_expected_sha256" not in precondition_metadata:
        return fail(precondition_metadata["message"], metadata=precondition_metadata)
    result = await _run_remote_file_action(
        action="write",
        path=path,
        payload={
            "content": content,
            "encoding": encoding,
            "mode": mode,
            "create_parent_dirs": create_parent_dirs,
            "backup": backup,
            "expected_sha256": resolved_expected_sha256,
        },
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )
    if result.get("success"):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if precondition_metadata is not None:
            metadata.update(precondition_metadata)
            result["metadata"] = metadata
        if metadata.get("readback_sha256") != intended_sha:
            return fail(
                "Remote readback hash did not match intended content.",
                metadata={**metadata, "error_kind": "readback_mismatch", "intended_sha256": intended_sha},
            )
        _clear_remote_mutation_requirement(state, path=path, host=str(metadata.get("host") or ""))
    return result


async def ssh_file_patch(
    path: str,
    target_text: str,
    replacement_text: str,
    target: str | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    encoding: str = "utf-8",
    expected_occurrences: int = 1,
    backup: bool = True,
    expected_sha256: str | None = None,
    source_artifact_id: str | None = None,
    whitespace_normalized: bool = False,
    dry_run: bool = False,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    resolved_expected_sha256, precondition_metadata = _resolve_expected_sha_precondition(
        path=path,
        expected_sha256=expected_sha256,
        source_artifact_id=source_artifact_id,
        state=state,
    )
    if precondition_metadata is not None and "resolved_expected_sha256" not in precondition_metadata:
        return fail(precondition_metadata["message"], metadata=precondition_metadata)
    if whitespace_normalized and not (resolved_expected_sha256 or str(source_artifact_id or "").strip()):
        return fail(
            "whitespace_normalized mode requires `expected_sha256` or `source_artifact_id`. Prefer `dry_run=True` first to preview the matched region and planned hash.",
            metadata={"reason": "precondition_required_for_whitespace_normalized"},
        )
    result = await _run_remote_file_action(
        action="patch",
        path=path,
        payload={
            "target_text": target_text,
            "replacement_text": replacement_text,
            "encoding": encoding,
            "expected_occurrences": expected_occurrences,
            "backup": backup,
            "expected_sha256": resolved_expected_sha256,
            "whitespace_normalized": whitespace_normalized,
            "dry_run": dry_run,
        },
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )
    if result.get("success"):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if precondition_metadata is not None:
            metadata.update(precondition_metadata)
            result["metadata"] = metadata
        if dry_run:
            return result
        _clear_remote_mutation_requirement(state, path=path, host=str(metadata.get("host") or ""))
    return result


async def ssh_file_replace_between(
    path: str,
    start_text: str,
    end_text: str,
    replacement_text: str,
    target: str | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    encoding: str = "utf-8",
    include_bounds: bool = True,
    expected_occurrences: int = 1,
    backup: bool = True,
    expected_sha256: str | None = None,
    source_artifact_id: str | None = None,
    whitespace_normalized: bool = False,
    dry_run: bool = False,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    resolved_expected_sha256, precondition_metadata = _resolve_expected_sha_precondition(
        path=path,
        expected_sha256=expected_sha256,
        source_artifact_id=source_artifact_id,
        state=state,
    )
    if precondition_metadata is not None and "resolved_expected_sha256" not in precondition_metadata:
        return fail(precondition_metadata["message"], metadata=precondition_metadata)
    if whitespace_normalized and not (resolved_expected_sha256 or str(source_artifact_id or "").strip()):
        return fail(
            "whitespace_normalized mode requires `expected_sha256` or `source_artifact_id`. Prefer `dry_run=True` first to preview the matched region and planned hash.",
            metadata={"reason": "precondition_required_for_whitespace_normalized"},
        )
    result = await _run_remote_file_action(
        action="replace_between",
        path=path,
        payload={
            "start_text": start_text,
            "end_text": end_text,
            "replacement_text": replacement_text,
            "encoding": encoding,
            "include_bounds": include_bounds,
            "expected_occurrences": expected_occurrences,
            "backup": backup,
            "expected_sha256": resolved_expected_sha256,
            "whitespace_normalized": whitespace_normalized,
            "dry_run": dry_run,
        },
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )
    if result.get("success"):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if precondition_metadata is not None:
            metadata.update(precondition_metadata)
            result["metadata"] = metadata
        if dry_run:
            return result
        _clear_remote_mutation_requirement(state, path=path, host=str(metadata.get("host") or ""))
    return result
