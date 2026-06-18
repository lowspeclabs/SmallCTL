from __future__ import annotations

from ..models.tool_result import ToolEnvelope
from ..state import ArtifactRecord
from .messages_compact_helpers import (
    FILE_READ_PREVIEW_LINE_LIMIT as _FILE_READ_PREVIEW_LINE_LIMIT,
    file_read_status_header as _file_read_status_header,
    format_local_file_mutation_message as _format_local_file_mutation_message,
    format_shell_exec_message as _format_shell_exec_message,
    format_ssh_file_mutation_message as _format_ssh_file_mutation_message,
    request_has_full_artifact_intent as _request_has_full_artifact_intent,
)
from .messages_dir_rendering import (
    LISTING_PREVIEW_ENTRY_LIMIT as _LISTING_PREVIEW_ENTRY_LIMIT,
    append_dir_tree_line as _append_dir_tree_line,
    append_dir_tree_lines as _append_dir_tree_lines,
    dir_list_preview_is_incomplete as _dir_list_preview_is_incomplete,
    dir_list_tree_has_truncation as _dir_list_tree_has_truncation,
    listing_preview_is_incomplete as _listing_preview_is_incomplete,
    render_dir_list_result,
    render_dir_list_tree,
)
from .messages_next_steps import (
    WRITE_OUTPUT_KEYWORDS as _WRITE_OUTPUT_KEYWORDS,
    artifact_print_hint as _artifact_print_hint,
    artifact_read_continuation_hint as _artifact_read_continuation_hint,
    artifact_read_hint as _artifact_read_hint,
    artifact_summary_exit_hint as _artifact_summary_exit_hint,
    dir_list_followup_hint as _dir_list_followup_hint,
    dir_tree_followup_hint as _dir_tree_followup_hint,
    file_read_followup_hint as _file_read_followup_hint,
    next_step_hint,
    request_prefers_summary_exit as _request_prefers_summary_exit,
    request_requires_saved_output as _request_requires_saved_output,
    search_followup_hint as _search_followup_hint,
    yaml_read_followup_hint as _yaml_read_followup_hint,
)
from .messages_reused_artifacts import (
    REUSED_ARTIFACT_INLINE_CHAR_LIMIT as _REUSED_ARTIFACT_INLINE_CHAR_LIMIT,
    REUSED_ARTIFACT_INLINE_LINE_LIMIT as _REUSED_ARTIFACT_INLINE_LINE_LIMIT,
    format_reused_artifact_message,
    format_reused_file_read_message as _format_reused_file_read_message,
    small_complete_artifact_content as _small_complete_artifact_content,
)


_ARTIFACT_PREVIEW_CHAR_LIMIT = 4000


def format_compact_tool_message(
    artifact: ArtifactRecord,
    result: ToolEnvelope,
    *,
    request_text: str | None = None,
    inline_full_file: bool = True,
    full_file_preview_chars: int | None = None,
) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    local_mutation_message = _format_local_file_mutation_message(artifact, result, metadata=metadata)
    if local_mutation_message:
        return local_mutation_message
    mutation_message = _format_ssh_file_mutation_message(artifact, result, metadata=metadata)
    if mutation_message:
        return mutation_message
    if artifact.kind in {"shell_exec", "ssh_exec"} and not metadata.get("source_artifact_id"):
        return _format_shell_exec_message(artifact, result, request_text=request_text)

    msg = f"Tool output captured as Artifact {artifact.artifact_id} ({artifact.size_bytes} bytes)."
    file_status_header = _file_read_status_header(
        artifact,
        metadata=metadata,
        inline_full_file=inline_full_file,
        full_file_preview_chars=full_file_preview_chars,
        result=result,
    )
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
        file_lines = file_text.splitlines()
        if inline_full_file:
            if len(file_lines) <= 500:
                msg = (
                    f"{msg}{file_status_header}\n\nFull file captured ({line_label}).\n\nfull_file_content:\n"
                    f"{file_text}"
                )
            else:
                truncated_text = "\n".join(file_lines[:490])
                file_status_header = _file_read_status_header(
                    artifact,
                    metadata=metadata,
                    inline_full_file=inline_full_file,
                    full_file_preview_chars=full_file_preview_chars,
                    result=result,
                    force_display_preview_truncated=True,
                )
                msg = (
                    f"{msg}{file_status_header}\n\nFile captured ({line_label}). Showing first 490 lines.\n\n"
                    f"file_content_preview:\n{truncated_text}\n\n"
                    f"... display preview truncated at 490/{len(file_lines)} lines. "
                    f"Use `artifact_read(artifact_id='{artifact.artifact_id}')` for the rest."
                )
        else:
            display_preview_truncated = False
            if full_file_preview_chars is not None:
                preview = file_text
                if len(preview) > full_file_preview_chars:
                    preview = f"{preview[:full_file_preview_chars].rstrip()}..."
                    display_preview_truncated = True
                preview_note = "Preview"
            else:
                display_preview_truncated = len(file_lines) > _FILE_READ_PREVIEW_LINE_LIMIT
                preview = "\n".join(file_lines[:_FILE_READ_PREVIEW_LINE_LIMIT])
                preview_note = (
                    f"Preview (first {_FILE_READ_PREVIEW_LINE_LIMIT} of {len(file_lines)} lines)"
                    if display_preview_truncated
                    else "Preview"
                )
            if display_preview_truncated and "display_preview_truncated=true" not in file_status_header:
                file_status_header = _file_read_status_header(
                    artifact,
                    metadata=metadata,
                    inline_full_file=inline_full_file,
                    full_file_preview_chars=full_file_preview_chars,
                    result=result,
                    force_display_preview_truncated=True,
                )
            msg = f"{msg}{file_status_header}\n\nFull file captured ({line_label}).\n\n{preview_note}:\n{preview}"
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
