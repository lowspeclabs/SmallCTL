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
        return {
            "ok": True,
            "path": str(path),
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "content": "",
            "truncated": truncated,
            "encoding": encoding,
            "binary": True,
            "decode_error": str(exc),
            "message": "Remote file is not valid text for the requested encoding; returning metadata only.",
        }
    return {
        "ok": True,
        "path": str(path),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "content": content,
        "truncated": truncated,
        "encoding": encoding,
    }

def list_dir(payload):
    path = pathlib.Path(payload["path"])
    if not path.exists():
        return {"ok": False, "error_kind": "file_not_found", "path": str(path), "message": "Remote directory not found."}
    if not path.is_dir():
        return {"ok": False, "error_kind": "not_a_directory", "path": str(path), "message": "Remote path is not a directory."}
    try:
        entries = []
        total = 0
        for entry in path.iterdir():
            total += 1
            if len(entries) >= 200:
                continue
            if entry.is_symlink():
                entry_type = "symlink"
            elif entry.is_dir():
                entry_type = "dir"
            elif entry.is_file():
                entry_type = "file"
            else:
                entry_type = "other"
            item = {"name": entry.name, "type": entry_type}
            if entry_type == "file":
                try:
                    item["size"] = entry.stat().st_size
                except OSError:
                    pass
            if entry_type == "symlink":
                try:
                    item["target"] = os.readlink(entry)
                except OSError:
                    pass
            entries.append(item)
    except PermissionError:
        return {"ok": False, "error_kind": "permission_denied", "path": str(path), "message": "Permission denied listing remote directory."}
    except OSError as exc:
        return {"ok": False, "error_kind": "list_error", "path": str(path), "message": str(exc)}
    return {
        "ok": True,
        "path": str(path),
        "entries": entries,
        "count": len(entries),
        "truncated": total > len(entries),
        "total_items": total,
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
        if whitespace_normalized:
            norm_content, _ = normalize_whitespace_with_spans(content)
            norm_start, _ = normalize_whitespace_with_spans(start_text)
            norm_end, _ = normalize_whitespace_with_spans(end_text)
            start_found = norm_start in norm_content if norm_start else False
            end_found = norm_end in norm_content if norm_end else False
        else:
            start_found = start_text in content if start_text else False
            end_found = end_text in content if end_text else False
        if start_found and not end_found:
            ambiguity_hint = (
                "start_text was found but end_text was not found after it. "
                "Check the exact text between them and verify end_text exists in the file."
            )
        elif not start_found and end_found:
            ambiguity_hint = (
                "end_text was found but start_text was not found before it. "
                "Check the exact text before end_text and verify start_text exists in the file."
            )
        elif not start_found and not end_found:
            ambiguity_hint = (
                "Neither start_text nor end_text were found. "
                "Read the remote file and copy the exact bounds, including whitespace."
            )
        else:
            ambiguity_hint = (
                "Both start_text and end_text exist but not in the expected order. "
                "Verify the bounds appear in order in the file."
            )
        meta.update({
            "ok": False,
            "error_kind": "bounded_region_not_found",
            "message": "Remote bounded region was not found.",
            "ambiguity_hint": ambiguity_hint,
            "start_text_found": start_found,
            "end_text_found": end_found,
        })
        if not start_found:
            meta["start_text_best_match"] = best_patch_match(content, start_text)
        if not end_found:
            meta["end_text_best_match"] = best_patch_match(content, end_text)
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
        if action == "list_dir":
            emit(list_dir(payload))
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
