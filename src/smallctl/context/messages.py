from __future__ import annotations

import re

from ..models.tool_result import ToolEnvelope
from ..state import ArtifactRecord
from .rendering import render_shell_failure, render_shell_output


_LISTING_PREVIEW_ENTRY_LIMIT = 50
_ARTIFACT_PREVIEW_CHAR_LIMIT = 4000
_SHELL_EXEC_INLINE_CHAR_LIMIT = 1600


def next_step_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
    request_text: str | None = None,
) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    result_metadata = result.metadata if result and isinstance(result.metadata, dict) else {}
    intent = metadata.get("intent", "").lower()
    tool_name = (
        metadata.get("_original_tool_name", "")
        or artifact.kind
        or artifact.tool_name
        or ""
    ).lower()

    if result_metadata.get("source_artifact_id"):
        summary_hint = _artifact_summary_exit_hint(
            artifact,
            result=result,
            request_text=request_text,
        )
        if summary_hint:
            return summary_hint
        if result_metadata.get("truncated"):
            return _artifact_read_continuation_hint(artifact, result_metadata=result_metadata)
        return ""

    summary_hint = _artifact_summary_exit_hint(
        artifact,
        result=result,
        request_text=request_text,
    )
    if summary_hint:
        return summary_hint

    if "listing" in intent or tool_name == "dir_list":
        return _dir_list_followup_hint(artifact, result=result)
    if intent == "dir_tree" or tool_name == "dir_tree":
        return _dir_tree_followup_hint(artifact, result=result)
    if "search" in intent or "grep" in intent or tool_name in {"grep", "find_files"}:
        return _search_followup_hint(artifact, result=result)
    if tool_name == "file_read":
        return _file_read_followup_hint(artifact, result=result)
    if tool_name == "yaml_read":
        return _yaml_read_followup_hint(artifact, result=result)

    if _listing_preview_is_incomplete(artifact, result=result):
        return _artifact_read_hint(artifact, lead="To inspect the full result,")

    return ""


def _artifact_read_continuation_hint(
    artifact: ArtifactRecord,
    *,
    result_metadata: dict[str, object],
) -> str:
    line_start = result_metadata.get("line_start")
    line_end = result_metadata.get("line_end")
    total_lines = result_metadata.get("total_lines")

    if isinstance(line_end, int) and isinstance(total_lines, int) and line_end < total_lines:
        next_start = line_end + 1
        range_note = ""
        if isinstance(line_start, int):
            range_note = f" You have lines {line_start}-{line_end} of {total_lines}."
        else:
            range_note = f" You have lines 1-{line_end} of {total_lines}."
        return (
            f"To inspect more of this artifact, continue at `start_line={next_start}`."
            f"{range_note} Call `artifact_read(artifact_id='{artifact.artifact_id}', start_line={next_start})`."
        )

    return _artifact_read_hint(
        artifact,
        lead="To inspect more of this artifact,",
        tail="use `artifact_read` with a later `start_line` or a narrower `end_line`.",
    )


def _request_prefers_summary_exit(request_text: str | None) -> bool:
    text = re.sub(r"\s+", " ", str(request_text or "").strip().lower())
    if not text:
        return False
    asks_for_summary = any(keyword in text for keyword in ("table", "summary", "summarize", "report", "overview", "present"))
    asks_about_listing = any(keyword in text for keyword in ("list", "listing", "files", "directories", "artifact", "results", "output", "current env"))
    return asks_for_summary and asks_about_listing


def _artifact_summary_exit_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
    request_text: str | None = None,
) -> str:
    if not _request_prefers_summary_exit(request_text):
        return ""

    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    result_metadata = result.metadata if result and isinstance(result.metadata, dict) else {}
    if result_metadata.get("truncated"):
        return ""

    tool_name = (
        metadata.get("_original_tool_name", "")
        or artifact.kind
        or artifact.tool_name
        or ""
    ).lower()
    if tool_name not in {"dir_list", "dir_tree", "artifact_read", "artifact_print", "grep", "find_files"}:
        return ""

    return (
        "You already have enough evidence to produce the requested table or summary. "
        "Synthesize the answer now instead of rereading or printing the same artifact again."
    )


def format_compact_tool_message(
    artifact: ArtifactRecord,
    result: ToolEnvelope,
    *,
    request_text: str | None = None,
    inline_full_file: bool = True,
    full_file_preview_chars: int | None = None,
) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    mutation_message = _format_ssh_file_mutation_message(artifact, result, metadata=metadata)
    if mutation_message:
        return mutation_message
    if artifact.kind in {"shell_exec", "ssh_exec"} and not metadata.get("source_artifact_id"):
        return _format_shell_exec_message(artifact, result, request_text=request_text)

    msg = f"Tool output captured as Artifact {artifact.artifact_id} ({artifact.size_bytes} bytes)."
    complete_file = bool(metadata.get("complete_file"))
    total_lines = metadata.get("total_lines")
    file_text = ""
    if artifact.kind in {"file_read", "ssh_file_read"} and complete_file:
        if artifact.kind == "ssh_file_read" and isinstance(result.output, dict):
            file_text = str(result.output.get("content") or "").rstrip()
        elif isinstance(result.output, str):
            file_text = result.output.rstrip()

    if file_text:
        line_label = f"{total_lines} lines" if isinstance(total_lines, int) else "the full file"
        if inline_full_file:
            msg = (
                f"{msg}\n\nFull file captured ({line_label}).\n\nfull_file_content:\n"
                f"{file_text}"
            )
        else:
            preview_limit = full_file_preview_chars or 600
            preview = file_text
            if len(preview) > preview_limit:
                preview = f"{preview[:preview_limit].rstrip()}..."
            msg = f"{msg}\n\nFull file captured ({line_label}).\n\nPreview:\n{preview}"
    elif metadata.get("source_artifact_id") and isinstance(result.output, str):
        preview = _preview_result_text(result.output)
        if preview:
            if _request_has_full_artifact_intent(request_text) and not metadata.get("truncated"):
                msg = f"{msg}\n\nFull artifact captured."
            msg = f"{msg}\n\nPreview:\n{preview}"
    else:
        preview = _preview_text(artifact)
        if preview:
            msg = f"{msg}\n\nPreview:\n{preview}"
    hint = next_step_hint(artifact, result=result, request_text=request_text)
    if hint:
        msg = f"{msg}\n\n{hint}"
    return msg


def _format_ssh_file_mutation_message(
    artifact: ArtifactRecord,
    result: ToolEnvelope,
    *,
    metadata: dict[str, object],
) -> str:
    tool_name = str(artifact.kind or artifact.tool_name or "").strip()
    if not result.success or tool_name not in {"ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}:
        return ""

    path = str(metadata.get("path") or artifact.source or "").strip()
    host = str(metadata.get("host") or "").strip()
    user = str(metadata.get("user") or "").strip()
    target = path
    if host:
        prefix = f"{user}@{host}" if user else host
        target = f"{prefix}:{path}" if path else prefix

    if tool_name == "ssh_file_write":
        lines = [f"Remote file written: {target or 'remote file'}"]
    elif tool_name == "ssh_file_patch":
        lines = [f"Remote file patched: {target or 'remote file'}"]
    else:
        lines = [f"Remote file region replaced: {target or 'remote file'}"]

    changed = metadata.get("changed")
    if isinstance(changed, bool):
        lines.append(f"changed: {'yes' if changed else 'no'}")
    bytes_written = metadata.get("bytes_written")
    if bytes_written not in (None, ""):
        lines.append(f"bytes_written: {bytes_written}")

    for key in ("actual_occurrences", "expected_occurrences"):
        value = metadata.get(key)
        if value not in (None, ""):
            lines.append(f"{key}: {value}")

    old_sha = str(metadata.get("old_sha256") or "").strip()
    new_sha = str(metadata.get("new_sha256") or "").strip()
    if old_sha:
        lines.append(f"old_sha256: {old_sha}")
    if new_sha:
        lines.append(f"new_sha256: {new_sha}")

    readback_sha = str(metadata.get("readback_sha256") or "").strip()
    verification = metadata.get("verification") if isinstance(metadata.get("verification"), dict) else {}
    readback_verified = (
        bool(verification.get("readback_sha256_matches"))
        or bool(new_sha and readback_sha and new_sha == readback_sha)
    )
    if readback_sha or verification:
        lines.append(f"readback verified: {'yes' if readback_verified else 'no'}")

    backup_path = str(metadata.get("backup_path") or "").strip()
    if backup_path and backup_path.lower() != "none":
        lines.append(f"backup_path: {backup_path}")

    for preview_key in ("target_text_preview", "start_text_preview", "end_text_preview", "replacement_text_preview"):
        preview_payload = metadata.get(preview_key)
        if not isinstance(preview_payload, dict):
            continue
        preview = str(preview_payload.get("preview") or "").strip()
        if preview:
            if len(preview) > 160:
                preview = f"{preview[:160].rstrip()}..."
            lines.append(f"{preview_key}: {preview}")

    return "\n".join(lines)


def _format_shell_exec_message(
    artifact: ArtifactRecord,
    result: ToolEnvelope,
    *,
    request_text: str | None = None,
) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    output = result.output if isinstance(result.output, dict) else {}
    if not result.success:
        failure_output = metadata.get("output") if isinstance(metadata.get("output"), dict) else output
        transcript = render_shell_failure(
            error=result.error,
            output=failure_output if isinstance(failure_output, dict) else None,
            preview_limit=None,
            strip_whitespace=False,
        )
        msg = transcript or "Shell command failed."
    else:
        transcript = render_shell_output(
            output,
            preview_limit=None,
            strip_whitespace=False,
        )
        msg = transcript or "ok"

    truncated = (
        bool(metadata.get("truncated"))
        or len(transcript) > _SHELL_EXEC_INLINE_CHAR_LIMIT
        or transcript.endswith("... output truncated")
    )
    if truncated:
        preview = render_shell_output(
            output,
            preview_limit=_SHELL_EXEC_INLINE_CHAR_LIMIT,
            strip_whitespace=False,
        ) if result.success else render_shell_failure(
            error=result.error,
            output=failure_output if isinstance(failure_output, dict) else None,
            preview_limit=_SHELL_EXEC_INLINE_CHAR_LIMIT,
            strip_whitespace=False,
        )
        footer = f"Output truncated; full transcript stored in hidden Artifact {artifact.artifact_id}."
        msg = preview or msg
        if _request_has_full_artifact_intent(request_text):
            footer = (
                f"{footer} Call `artifact_read(artifact_id='{artifact.artifact_id}')` "
                "if you need the full transcript."
            )
        msg = f"{msg}\n\n{footer}"
    return msg


def format_reused_artifact_message(artifact: ArtifactRecord, *, tool_name: str | None = None) -> str:
    summary = artifact.summary or artifact.source or artifact.kind or "cached artifact"
    normalized_tool = str(tool_name or "").strip().lower()
    if normalized_tool == "artifact_print":
        return (
            f"Reused Artifact {artifact.artifact_id}: {summary}. "
            "This evidence is already visible in context, so do not print it again. "
            "Synthesize the answer from it or call `task_complete(message='...')` if you are finished."
        )
    if artifact.kind == "file_read":
        path = str(artifact.source or artifact.metadata.get("path") or artifact.content_path or "").strip()
        path_note = f" for `{path}`" if path else ""
        return (
            f"Reused Artifact {artifact.artifact_id}: {summary}{path_note}. "
            "You already have the full file evidence in context. "
            "Do not call `file_read` again on the same path. "
            "Work from this artifact, patch the file, or use `artifact_read` if you need paging."
        )
    return f"Reused Artifact {artifact.artifact_id}: {summary}"


def _artifact_read_hint(
    artifact: ArtifactRecord,
    *,
    lead: str,
    tail: str = "",
) -> str:
    hint = f"{lead} call `artifact_read(artifact_id='{artifact.artifact_id}')`"
    if tail:
        hint = f"{hint} {tail}"
    return hint


def _artifact_print_hint(
    artifact: ArtifactRecord,
    *,
    lead: str,
    tail: str = "",
) -> str:
    hint = f"{lead} call `artifact_print(artifact_id='{artifact.artifact_id}')`"
    if tail:
        hint = f"{hint} {tail}"
    return hint


def _preview_text(artifact: ArtifactRecord) -> str:
    preview = (artifact.preview_text or "").strip()
    if not preview:
        return ""
    if len(preview) > _ARTIFACT_PREVIEW_CHAR_LIMIT:
        preview = f"{preview[:_ARTIFACT_PREVIEW_CHAR_LIMIT].rstrip()}..."
    return preview


def _preview_result_text(text: str) -> str:
    preview = text.strip()
    if not preview:
        return ""
    if len(preview) > _ARTIFACT_PREVIEW_CHAR_LIMIT:
        preview = f"{preview[:_ARTIFACT_PREVIEW_CHAR_LIMIT].rstrip()}..."
    return preview


def _request_has_full_artifact_intent(request_text: str | None) -> bool:
    if not request_text:
        return False

    text = re.sub(r"\s+", " ", request_text.strip().lower())
    if not text:
        return False

    direct_phrases = (
        "read the entire artifact",
        "read the whole artifact",
        "read the full artifact",
        "read the complete artifact",
        "show me the entire artifact",
        "show me the whole artifact",
        "show me the full artifact",
        "show me the complete artifact",
        "entire artifact",
        "whole artifact",
        "full artifact",
        "complete artifact",
        "entire output",
        "whole output",
        "full output",
        "complete output",
        "entire file",
        "whole file",
        "full file",
        "complete file",
        "entire log",
        "whole log",
        "full log",
        "complete log",
        "entire scan",
        "whole scan",
        "full scan",
        "complete scan",
        "entire listing",
        "whole listing",
        "full listing",
        "complete listing",
        "all of it",
        "all of the output",
        "all output",
        "everything",
    )
    if any(phrase in text for phrase in direct_phrases):
        return True

    if re.search(
        r"\b(read|show|give|display|print|see|inspect|view|dump)\b(?:\W+\w+){0,4}\W+\b(all|everything|the whole thing|all of it)\b",
        text,
    ):
        return True

    return bool(
        re.search(
            r"\b(full|entire|whole|complete|all)\b(?:\W+\w+){0,4}\W+\b(artifact|file|output|log|scan|listing|result|results|content|contents)\b",
            text,
        )
    )


def _dir_list_followup_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> str:
    if not _listing_preview_is_incomplete(artifact, result=result):
        return ""
    return _artifact_read_hint(
        artifact,
        lead="To continue the directory listing in the next chunk,",
        tail="instead of rerunning another listing command.",
    )


def render_dir_list_tree(
    items: list[object],
    *,
    max_depth: int = 2,
    max_children: int = 8,
) -> str:
    lines: list[str] = []
    _append_dir_tree_lines(
        lines,
        items,
        depth=0,
        max_depth=max_depth,
        max_children=max_children,
    )
    return "\n".join(lines).strip()


def render_dir_list_result(
    items: list[object],
    *,
    metadata: dict[str, object] | None = None,
    max_depth: int = 2,
    max_children: int = 8,
    max_items: int | None = None,
) -> str:
    lines: list[str] = []
    listing_metadata = metadata if isinstance(metadata, dict) else {}

    path = listing_metadata.get("path")
    total_items = listing_metadata.get("total_items")
    count = total_items if isinstance(total_items, int) and total_items >= 0 else listing_metadata.get("count")
    if not isinstance(count, int):
        count = len(items)

    if isinstance(path, str) and path.strip():
        if count >= 0:
            lines.append(f"{path.strip()} ({count} items)")
        else:
            lines.append(path.strip())
    elif count >= 0:
        lines.append(f"{count} items")

    rendered_items = items if max_items is None else items[:max_items]
    tree_preview = render_dir_list_tree(
        rendered_items,
        max_depth=max_depth,
        max_children=max_children,
    )
    if tree_preview:
        lines.append(tree_preview)

    if max_items is not None:
        remaining = len(items) - len(rendered_items)
        if remaining > 0:
            lines.append(f"... {remaining} more items")

    return "\n".join(lines).strip() or "directory listed"


def _dir_tree_followup_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> str:
    if not _listing_preview_is_incomplete(artifact, result=result):
        return ""
    return (
        f"{_artifact_read_hint(artifact, lead='To continue this stored tree in the next chunk,', tail='before rerunning `dir_tree` with a larger `max_depth` or `max_entries`.')}"
        " If you also need to analyze it yourself, use `artifact_read` to keep paging forward."
    )


def _search_followup_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> str:
    if not _listing_preview_is_incomplete(artifact, result=result):
        return ""

    return _artifact_read_hint(
        artifact,
        lead="To continue through more results in the next chunk,",
        tail="instead of rerun the search with a narrower query/path or a larger `max_results`.",
    )


def _file_read_followup_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    complete_file = bool(metadata.get("complete_file"))
    total_lines = metadata.get("total_lines")
    if complete_file:
        line_label = f"{total_lines} lines" if isinstance(total_lines, int) else "the full file"
        path = str(metadata.get("path") or artifact.source or artifact.content_path or "").strip()
        path_note = f" from `{path}`" if path else ""
        return (
            f"You now have {line_label}{path_note}. Do not call `file_read` on the same path again. "
            f"Next, patch the file, run focused verification, or call `task_complete`. "
            f"If you need line-level detail later, use `artifact_read(artifact_id='{artifact.artifact_id}')`."
        )
    if total_lines is None and result and isinstance(result.output, str):
        total_lines = len(result.output.splitlines())
    line_label = f"{total_lines} lines" if isinstance(total_lines, int) else "this excerpt"
    path = str(metadata.get("path") or artifact.source or artifact.content_path or "").strip()
    path_hint = f"file_read(path='{path}')" if path else "file_read(path='...')"
    return (
        f"This artifact only contains the excerpt already read. ({line_label}) "
        f"To continue reading, call `artifact_read(artifact_id='{artifact.artifact_id}')` "
        f"or use `start_line`/`end_line` to page forward instead of rerunning {path_hint}."
    )


def _yaml_read_followup_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    total_keys = metadata.get("total_keys")
    if isinstance(total_keys, int):
        if total_keys <= 50:
            return ""
    elif result and isinstance(result.output, dict) and len(result.output) <= 50:
        return ""
    return _artifact_read_hint(
        artifact,
        lead="To continue the structured data in the next chunk,",
        tail="before rerunning the original tool.",
    )


def _dir_list_preview_is_incomplete(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    total_items = metadata.get("total_items")
    if total_items is None:
        total_items = metadata.get("count")
    if total_items is None and result and result.output:
        if isinstance(result.output, list):
            total_items = len(result.output)

    if total_items is not None and total_items > _LISTING_PREVIEW_ENTRY_LIMIT:
        return True
    if result and isinstance(result.output, list) and _dir_list_tree_has_truncation(result.output):
        return True
    return False


def _dir_list_tree_has_truncation(items: list[object]) -> bool:
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("children_truncated"):
            return True
        children = item.get("children")
        if isinstance(children, list) and _dir_list_tree_has_truncation(children):
            return True
    return False


def _listing_preview_is_incomplete(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    if metadata.get("truncated"):
        return True
    return _dir_list_preview_is_incomplete(artifact, result=result)


def _append_dir_tree_lines(
    lines: list[str],
    items: list[object],
    *,
    depth: int,
    max_depth: int,
    max_children: int,
) -> None:
    indent = "  " * depth
    preview_items = items[:max_children]
    for item in preview_items:
        _append_dir_tree_line(
            lines,
            item,
            depth=depth,
            max_depth=max_depth,
            max_children=max_children,
        )
    if len(items) > len(preview_items):
        lines.append(f"{indent}... {len(items) - len(preview_items)} more items")


def _append_dir_tree_line(
    lines: list[str],
    item: object,
    *,
    depth: int,
    max_depth: int,
    max_children: int,
) -> None:
    indent = "  " * depth
    if not isinstance(item, dict):
        text = str(item or "").strip()
        if text:
            lines.append(f"{indent}{text}")
        return

    name = str(item.get("name") or item.get("path") or "").strip()
    if not name:
        return

    parts = [name]
    item_type = str(item.get("type") or "").strip()
    if item_type:
        parts.append(f"[{item_type}]")

    size = item.get("size")
    if isinstance(size, int) and size >= 0:
        parts.append(f"({size} bytes)")

    children = item.get("children")
    children_count = item.get("children_count")
    if item_type == "dir" and isinstance(children_count, int) and children_count >= 0:
        parts.append(f"({children_count} children)")

    lines.append(f"{indent}{' '.join(parts).strip()}")

    if depth >= max_depth:
        if isinstance(children_count, int) and children_count > 0 and isinstance(children, list):
            lines.append(f"{indent}  ... more nested items")
        return

    if isinstance(children, list) and children:
        _append_dir_tree_lines(
            lines,
            children,
            depth=depth + 1,
            max_depth=max_depth,
            max_children=max_children,
        )
        if item.get("children_truncated"):
            lines.append(f"{indent}  ... more nested items")
