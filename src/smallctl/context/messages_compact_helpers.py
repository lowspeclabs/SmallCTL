from __future__ import annotations

import re

from ..models.tool_result import ToolEnvelope
from ..state import ArtifactRecord
from .rendering import render_shell_failure, render_shell_output

FILE_READ_PREVIEW_LINE_LIMIT = 300
SHELL_EXEC_INLINE_CHAR_LIMIT = 1600


def request_has_full_artifact_intent(request_text: str | None) -> bool:
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


def file_read_status_header(
    artifact: ArtifactRecord,
    *,
    metadata: dict[str, object],
    inline_full_file: bool,
    full_file_preview_chars: int | None,
    result: ToolEnvelope,
    force_display_preview_truncated: bool = False,
) -> str:
    tool_name = str(artifact.kind or artifact.tool_name or "").strip()
    if tool_name not in {"file_read", "ssh_file_read"}:
        return ""
    path = str(metadata.get("path") or artifact.source or "").strip()
    source_path = str(metadata.get("source_path") or "").strip()
    read_from_staging = bool(metadata.get("read_from_staging") or metadata.get("staged_only"))
    complete_file = bool(metadata.get("complete_file"))
    file_content_truncated = bool(metadata.get("truncated"))
    total_lines = metadata.get("total_lines")
    line_start = metadata.get("line_start")
    line_end = metadata.get("line_end")
    display_preview_truncated = force_display_preview_truncated
    if not display_preview_truncated and not inline_full_file:
        output = result.output
        if tool_name == "ssh_file_read" and isinstance(output, dict):
            text = str(output.get("content") or "")
        else:
            text = output if isinstance(output, str) else ""
        if full_file_preview_chars is not None:
            display_preview_truncated = bool(text and len(text.rstrip()) > full_file_preview_chars)
        else:
            display_preview_truncated = bool(text and len(text.rstrip().splitlines()) > FILE_READ_PREVIEW_LINE_LIMIT)
    lines = [
        "",
        "",
        "FILE READ STATUS:",
        f"path={path or '(unknown)'}",
    ]
    if source_path and source_path != path:
        lines.append(f"source_path={source_path}")
    if read_from_staging:
        session_id = str(metadata.get("write_session_id") or "").strip()
        detail = f"true; write_session_id={session_id}" if session_id else "true"
        lines.append(f"read_from_active_write_session_staging={detail}")
    lines.extend(
        [
            f"complete_file={'true' if complete_file else 'false'}",
            f"display_preview_truncated={'true' if display_preview_truncated else 'false'}",
            f"file_content_truncated={'true' if file_content_truncated else 'false'}",
            f"artifact_id={artifact.artifact_id}",
        ]
    )
    if isinstance(line_start, int) or isinstance(line_end, int) or isinstance(total_lines, int):
        lines.append(f"lines={line_start}-{line_end} of {total_lines}")
    if display_preview_truncated and not file_content_truncated:
        lines.append(
            "NOTE: The file itself was not truncated; only this chat preview was shortened. "
            "Use artifact_read on this artifact to inspect more content."
        )
    return "\n".join(lines)


def format_local_file_mutation_message(
    artifact: ArtifactRecord,
    result: ToolEnvelope,
    *,
    metadata: dict[str, object],
) -> str:
    tool_name = str(artifact.kind or artifact.tool_name or "").strip()
    if not result.success or tool_name not in {"file_write", "file_append", "file_patch", "ast_patch", "file_delete"}:
        return ""

    path = str(metadata.get("path") or artifact.source or "").strip()
    if tool_name == "file_write":
        lines = [f"Local file written: {path or 'local file'}"]
    elif tool_name == "file_append":
        lines = [f"Local file appended: {path or 'local file'}"]
    elif tool_name in {"file_patch", "ast_patch"}:
        lines = [f"Local file patched: {path or 'local file'}"]
    else:
        lines = [f"Local file deleted: {path or 'local file'}"]

    changed = metadata.get("changed")
    if isinstance(changed, bool):
        lines.append(f"changed: {'yes' if changed else 'no'}")

    bytes_written = metadata.get("bytes_written")
    if bytes_written in (None, ""):
        bytes_written = metadata.get("bytes")
    if bytes_written not in (None, ""):
        lines.append(f"bytes_written: {bytes_written}")

    for key in ("actual_occurrences", "expected_occurrences"):
        value = metadata.get(key)
        if value not in (None, ""):
            lines.append(f"{key}: {value}")

    staging_path = str(metadata.get("staging_path") or "").strip()
    if staging_path:
        lines.append(f"staging_path: {staging_path}")

    session_id = str(metadata.get("write_session_id") or "").strip()
    if session_id:
        lines.append(f"write_session_id: {session_id}")
    if bool(metadata.get("staged_only")):
        lines.append("persisted_to_target: no; staged_only=true")
    elif tool_name != "file_delete":
        lines.append("persisted_to_target: yes")

    return "\n".join(lines)


def format_ssh_file_mutation_message(
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
    readback_verified = bool(verification.get("readback_sha256_matches")) or bool(new_sha and readback_sha and new_sha == readback_sha)
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


def format_shell_exec_message(
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
        if artifact.kind == "ssh_exec" and bool(metadata.get("ssh_transport_succeeded")):
            failure_mode = str(metadata.get("failure_mode") or metadata.get("ssh_error_class") or "").strip()
            if failure_mode == "remote_exit_nonzero":
                msg = f"SSH reached the remote host; remote command exited non-zero.\n\n{msg}"
    else:
        transcript = render_shell_output(
            output,
            preview_limit=None,
            strip_whitespace=False,
        )
        msg = transcript or "ok"

    truncated = (
        bool(metadata.get("truncated"))
        or len(transcript) > SHELL_EXEC_INLINE_CHAR_LIMIT
        or transcript.endswith("... output truncated")
    )
    if truncated:
        preview = render_shell_output(
            output,
            preview_limit=SHELL_EXEC_INLINE_CHAR_LIMIT,
            strip_whitespace=False,
        ) if result.success else render_shell_failure(
            error=result.error,
            output=failure_output if isinstance(failure_output, dict) else None,
            preview_limit=SHELL_EXEC_INLINE_CHAR_LIMIT,
            strip_whitespace=False,
        )
        footer = f"Output truncated; full transcript stored in hidden Artifact {artifact.artifact_id}."
        msg = preview or msg
        if request_has_full_artifact_intent(request_text):
            footer = (
                f"{footer} Call `artifact_read(artifact_id='{artifact.artifact_id}')` "
                "if you need the full transcript."
            )
        msg = f"{msg}\n\n{footer}"
    return msg


def collapse_repeated_shell_failures(messages: list[Any]) -> list[Any]:
    """Collapse 3+ shell_exec failures with identical first-200-char content into a summary line.

    Preserves the last 2 occurrences so the model still sees the most recent evidence,
    but replaces older identical failures with a compact token-saving summary.
    """
    if len(messages) < 3:
        return messages

    # Find shell_exec tool-message indices and group by content signature
    shell_indices: list[int] = []
    for idx, msg in enumerate(messages):
        role = getattr(msg, "role", None)
        name = getattr(msg, "name", None)
        if role == "tool" and name == "shell_exec":
            shell_indices.append(idx)

    if len(shell_indices) < 3:
        return messages

    signatures: dict[str, list[int]] = {}
    for idx in shell_indices:
        content = str(getattr(messages[idx], "content", "") or "").strip()
        # Only consider failure-looking content
        lowered = content.lower()
        if not any(marker in lowered for marker in ("failed", "error:", "exited with code", "timed out")):
            continue
        sig = content[:200]
        signatures.setdefault(sig, []).append(idx)

    # Determine which indices to replace (all but last 2 in each large group)
    to_replace: set[int] = set()
    for sig, indices in signatures.items():
        if len(indices) >= 3:
            to_replace.update(indices[:-2])

    if not to_replace:
        return messages

    result: list[Any] = []
    summary_text = f"[{len(to_replace)} repeated shell_exec failures with identical error collapsed to save tokens. Last 2 occurrences preserved.]"
    summary_added = False
    for idx, msg in enumerate(messages):
        if idx in to_replace:
            if not summary_added:
                from ..models.conversation import ConversationMessage
                result.append(
                    ConversationMessage(
                        role="tool",
                        name="shell_exec",
                        content=summary_text,
                    )
                )
                summary_added = True
            continue
        result.append(msg)

    return result
