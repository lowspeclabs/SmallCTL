from __future__ import annotations

from pathlib import Path

from ..state import ArtifactRecord

REUSED_ARTIFACT_INLINE_CHAR_LIMIT = 24000
REUSED_ARTIFACT_INLINE_LINE_LIMIT = 500


def format_reused_artifact_message(artifact: ArtifactRecord, *, tool_name: str | None = None) -> str:
    summary = artifact.summary or artifact.source or artifact.kind or "cached artifact"
    normalized_tool = str(tool_name or "").strip().lower()
    if artifact.kind in {"file_read", "ssh_file_read"}:
        return format_reused_file_read_message(artifact, summary=summary)
    inline_content = small_complete_artifact_content(artifact)
    if inline_content:
        path = str(artifact.source or artifact.metadata.get("path") or "").strip()
        path_note = f" ({path})" if path else ""
        return (
            f"Reused Artifact {artifact.artifact_id}: {summary}{path_note}.\n"
            "Full cached content is visible below. Use it to answer now; do not call "
            "`artifact_read`, `artifact_print`, or reread the same file unless the user asks for fresh state.\n\n"
            f"```text\n{inline_content}\n```"
        )
    if normalized_tool == "artifact_print":
        return (
            f"Reused Artifact {artifact.artifact_id}: {summary}. "
            "This evidence is already visible in context, so do not print it again. "
            "Synthesize the answer from it or call `task_complete(message='...')` if you are finished."
        )
    return f"Reused Artifact {artifact.artifact_id}: {summary}"


def format_reused_file_read_message(artifact: ArtifactRecord, *, summary: str) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    path = str(artifact.source or metadata.get("path") or artifact.content_path or "").strip()
    path_note = f" for `{path}`" if path else ""
    complete_file = bool(metadata.get("complete_file"))
    file_content_truncated = bool(metadata.get("truncated"))
    total_lines = metadata.get("total_lines")
    line_start = metadata.get("line_start")
    line_end = metadata.get("line_end")
    lines = [
        f"Reused Artifact {artifact.artifact_id}: {summary}{path_note}.",
        "FILE READ CACHE STATUS:",
        f"complete_file={'true' if complete_file else 'false'}",
        f"file_content_truncated={'true' if file_content_truncated else 'false'}",
    ]
    if isinstance(line_start, int) or isinstance(line_end, int) or isinstance(total_lines, int):
        lines.append(f"lines={line_start}-{line_end} of {total_lines}")
    if complete_file and not file_content_truncated:
        lines.append(
            "The full file was already captured; any prompt preview shortening is transcript compaction, "
            "not missing file content. Patch, verify, or move on instead of rereading the same path."
        )
    else:
        lines.append("Use artifact_read or start_line/end_line only if you need lines not already read.")
    return "\n".join(lines)


def small_complete_artifact_content(artifact: ArtifactRecord) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    if bool(metadata.get("truncated")):
        return ""
    complete = bool(metadata.get("complete_file")) or str(artifact.summary or "").lower().find("full file") >= 0
    if artifact.kind == "ssh_file_read":
        complete = complete or bool(metadata.get("path"))
    if not complete and artifact.kind not in {"file_read", "ssh_file_read"}:
        return ""

    text = str(artifact.inline_content or "")
    if not text and artifact.content_path:
        try:
            text = Path(artifact.content_path).read_text(encoding="utf-8")
        except OSError:
            text = ""
    text = text.rstrip("\n")
    if not text:
        return ""
    line_count = len(text.splitlines())
    if len(text) > REUSED_ARTIFACT_INLINE_CHAR_LIMIT or line_count > REUSED_ARTIFACT_INLINE_LINE_LIMIT:
        return ""
    return text
