from __future__ import annotations

import re

from ..models.tool_result import ToolEnvelope
from ..state import ArtifactRecord
from .messages_dir_rendering import listing_preview_is_incomplete

WRITE_OUTPUT_KEYWORDS = (
    "save",
    "write",
    "store",
    "export",
    "persist",
    "record",
)


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
        summary_hint = artifact_summary_exit_hint(
            artifact,
            result=result,
            request_text=request_text,
        )
        if summary_hint:
            return summary_hint
        if result_metadata.get("truncated"):
            return artifact_read_continuation_hint(artifact, result_metadata=result_metadata)
        return ""

    summary_hint = artifact_summary_exit_hint(
        artifact,
        result=result,
        request_text=request_text,
    )
    if summary_hint:
        return summary_hint

    if "listing" in intent or tool_name == "dir_list":
        return dir_list_followup_hint(artifact, result=result)
    if intent == "dir_tree" or tool_name == "dir_tree":
        return dir_tree_followup_hint(artifact, result=result)
    if "search" in intent or "grep" in intent or tool_name in {"grep", "find_files"}:
        return search_followup_hint(artifact, result=result)
    if tool_name == "file_read":
        return file_read_followup_hint(artifact, result=result)
    if tool_name == "yaml_read":
        return yaml_read_followup_hint(artifact, result=result)

    if listing_preview_is_incomplete(artifact, result=result):
        return artifact_read_hint(artifact, lead="To inspect the full result,")

    return ""


def artifact_read_continuation_hint(
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

    return artifact_read_hint(
        artifact,
        lead="To inspect more of this artifact,",
        tail="use `artifact_read` with a later `start_line` or a narrower `end_line`.",
    )


def request_prefers_summary_exit(request_text: str | None) -> bool:
    text = re.sub(r"\s+", " ", str(request_text or "").strip().lower())
    if not text:
        return False
    if request_requires_saved_output(text):
        return False
    asks_for_summary = any(keyword in text for keyword in ("table", "summary", "summarize", "report", "overview", "present"))
    asks_about_listing = any(
        keyword in text
        for keyword in ("list", "listing", "files", "directories", "artifact", "results", "output", "current env", "cron", "job")
    )
    return asks_for_summary and asks_about_listing


def request_requires_saved_output(text: str) -> bool:
    if not any(keyword in text for keyword in WRITE_OUTPUT_KEYWORDS):
        return False
    return bool(re.search(r"(?:^|\s)(?:\.{0,2}/)?[\w./-]+\.(?:txt|md|json|jsonl|csv|log|yaml|yml)(?:\s|$|[.,;:])", text))


def artifact_summary_exit_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
    request_text: str | None = None,
) -> str:
    if not request_prefers_summary_exit(request_text):
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


def artifact_read_hint(
    artifact: ArtifactRecord,
    *,
    lead: str,
    tail: str = "",
) -> str:
    hint = f"{lead} call `artifact_read(artifact_id='{artifact.artifact_id}')`"
    if tail:
        hint = f"{hint} {tail}"
    return hint


def artifact_print_hint(
    artifact: ArtifactRecord,
    *,
    lead: str,
    tail: str = "",
) -> str:
    hint = f"{lead} call `artifact_print(artifact_id='{artifact.artifact_id}')`"
    if tail:
        hint = f"{hint} {tail}"
    return hint


def dir_list_followup_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> str:
    if not listing_preview_is_incomplete(artifact, result=result):
        return ""
    return artifact_read_hint(
        artifact,
        lead="To continue the directory listing in the next chunk,",
        tail="instead of rerunning another listing command.",
    )


def dir_tree_followup_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> str:
    if not listing_preview_is_incomplete(artifact, result=result):
        return ""
    return (
        f"{artifact_read_hint(artifact, lead='To continue this stored tree in the next chunk,', tail='before rerunning `dir_tree` with a larger `max_depth` or `max_entries`.')}"
        " If you also need to analyze it yourself, use `artifact_read` to keep paging forward."
    )


def search_followup_hint(
    artifact: ArtifactRecord,
    *,
    result: ToolEnvelope | None = None,
) -> str:
    if not listing_preview_is_incomplete(artifact, result=result):
        return ""

    return artifact_read_hint(
        artifact,
        lead="To continue through more results in the next chunk,",
        tail="instead of rerun the search with a narrower query/path or a larger `max_results`.",
    )


def file_read_followup_hint(
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
        repair_cycle_id = str(metadata.get("system_repair_cycle_id") or "").strip()
        cycle_note = f" during repair cycle `{repair_cycle_id}`" if repair_cycle_id else " in this repair cycle"
        return (
            f"You now have {line_label}{path_note}. Do not call `file_read` on the same path again{cycle_note} "
            "unless a new repair-cycle gate explicitly requires a fresh disk snapshot. "
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


def yaml_read_followup_hint(
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
    return artifact_read_hint(
        artifact,
        lead="To continue the structured data in the next chunk,",
        tail="before rerunning the original tool.",
    )
