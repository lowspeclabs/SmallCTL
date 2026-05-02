from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import ok, fail

log = logging.getLogger("smallctl.tools.artifact")

DEFAULT_MAX_LINES = 500
DEFAULT_MAX_CHARS = 20_000
MAX_LINES_PER_READ = 1000
MAX_CHARS_PER_READ = 50_000
MAX_PRINT_CHARS = 100_000


MAX_PRINT_CHARS = 100_000
_ARTIFACT_TOKEN_RE = re.compile(r"\bA\d+\b", re.IGNORECASE)


async def artifact_print(
    harness: Any,
    *,
    artifact_id: str,
) -> dict[str, Any]:
    """
    Print the full contents of an artifact directly to the user's console/UI.
    This content is NOT seen by you directly to save context window space.
    Use this when the user needs to inspect raw output but you only need to confirm it was shown.
    """
    from ..models.events import UIEvent, UIEventType

    # Delegate to read but with high limits
    result = artifact_read(
        harness.state,
        artifact_id=artifact_id,
        start_line=1,
        end_line=20000,
        max_chars=MAX_PRINT_CHARS,
    )
    if not result.get("success"):
        return result

    content = result.get("output", "")
    msg = f"--- [ARTIFACT PRINT: {artifact_id}] ---\n{content}\n--- [END ARTIFACT PRINT] ---"

    # Emit UI event if possible
    # Note: Harness will handle the event propagation if we use _emit
    if hasattr(harness, "_emit") and harness.event_handler:
        event = UIEvent(
            event_type=UIEventType.ASSISTANT,
            content=msg,
            data={"artifact_id": artifact_id, "kind": "print"}
        )
        try:
            await harness._emit(harness.event_handler, event)
        except Exception:
            pass

    if hasattr(harness, "_runlog"):
        harness._runlog(
            "artifact_printed",
            f"Artifact {artifact_id} contents printed to user interface.",
            artifact_id=artifact_id,
            content_snippet=content[:200]
        )

    return ok(f"Artifact {artifact_id} contents printed to user interface. (Model context preserved)")


def artifact_grep(
    state: LoopState,
    *,
    artifact_id: str,
    query: str,
    case_insensitive: bool = True,
    max_results: int = 20,
) -> dict[str, Any]:
    """
    Search for a pattern within an artifact and return matching lines with context.
    Use this to find specific information within large artifacts (logs, command output)
    without reading the entire content.
    """
    artifact = _resolve_artifact_record(state, artifact_id)
    if not artifact:
        return fail(f"Artifact not found: {artifact_id}")

    try:
        content = ""
        if artifact.content_path:
            path = Path(artifact.content_path)
            if path.exists():
                content = path.read_text(encoding="utf-8")

        if not content and artifact.inline_content:
            content = artifact.inline_content

        if not content:
            return fail(f"Artifact {artifact_id} has no content to search.")

        lines = content.splitlines()
        matches = []
        pattern = query.lower() if case_insensitive else query

        for i, line in enumerate(lines):
            target = line.lower() if case_insensitive else line
            if pattern in target:
                matches.append({
                    "line": i + 1,
                    "content": line.strip()
                })
                if len(matches) >= max_results:
                    break

        canonical_artifact_id = str(getattr(artifact, "artifact_id", "") or artifact_id).strip() or artifact_id

        if not matches:
            return ok(f"No matches found for '{query}' in artifact {canonical_artifact_id}")

        output_lines = [f"Found {len(matches)} matches in {canonical_artifact_id}:"]
        for m in matches:
            output_lines.append(f"L{m['line']}: {m['content']}")

        metadata = {
            "artifact_id": canonical_artifact_id,
            "query": query,
            "match_count": len(matches),
            "total_lines": len(lines)
        }
        return ok("\n".join(output_lines), metadata=metadata)
    except Exception as exc:
        log.exception("Failed to grep artifact %s", artifact_id)
        return fail(f"Error searching artifact {artifact_id}: {exc}")


def artifact_read(
    state: LoopState,
    *,
    artifact_id: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_chars: int | None = None,
) -> dict[str, Any]:
    """
    Read stored artifact content by ID, paging large artifacts into bounded chunks.
    Reads up to 500 lines by default. Do not use tiny end_line values.

    Args:
        state: The current loop state (injected).
        artifact_id: The ID of the artifact to read (e.g., 'A0001').
        start_line: Optional 1-based inclusive starting line.
        end_line: Optional 1-based inclusive ending line. Omit this to read full chunks.
        max_chars: Optional character cap for the returned slice.
    """
    artifact = _resolve_artifact_record(state, artifact_id)

    if not artifact:
        # Provide a session-mismatch hint if the ID format is valid but belongs to another run
        hint = ""
        if artifact_id.startswith("A"):
            current_session = str(getattr(state, "thread_id", "") or "")
            if current_session:
                # Check if any artifact has a mismatched session_id to give a useful diagnostic
                for aid, rec in state.artifacts.items():
                    if getattr(rec, "session_id", "") and rec.session_id != current_session:
                        hint = " This artifact ID was created in a previous session. Re-execute the original tool call to regenerate the data in the current session."
                        break
                else:
                    if not state.artifacts:
                        hint = " No artifacts exist in the current session state. Re-execute the original tool call."
        return fail(f"Artifact {artifact_id} not found in state.{hint}")


    requested_max_chars = _coerce_positive_int(max_chars)
    if max_chars is not None and requested_max_chars is None:
        return fail("max_chars must be a positive integer.")
    requested_start_line = _coerce_positive_int(start_line) or 1
    if start_line is not None and requested_start_line < 1:
        return fail("start_line must be >= 1.")
    requested_end_line = _coerce_positive_int(end_line)
    if end_line is not None and requested_end_line is None:
        return fail("end_line must be a positive integer.")
    if requested_end_line is not None and requested_end_line < requested_start_line:
        return fail("end_line must be greater than or equal to start_line.")

    if not artifact.content_path:
        # Fallback to inline content if no file path exists
        if artifact.inline_content:
            return _render_artifact_slice(
                artifact=artifact,
                content=artifact.inline_content,
                start_line=requested_start_line,
                end_line=requested_end_line,
                max_chars=requested_max_chars,
            )
        return fail(f"Artifact {artifact_id} has no stored content.")

    try:
        path = Path(artifact.content_path)
        if not path.exists():
            # If the file is missing from disk but we have a record, try inline fallback
            if artifact.inline_content:
                return _render_artifact_slice(
                    artifact=artifact,
                    content=artifact.inline_content,
                    start_line=requested_start_line,
                    end_line=requested_end_line,
                    max_chars=requested_max_chars,
                )
            return fail(f"Artifact content file not found: {artifact.content_path}")

        content = path.read_text(encoding="utf-8")
        return _render_artifact_slice(
            artifact=artifact,
            content=content,
            start_line=requested_start_line,
            end_line=requested_end_line,
            max_chars=requested_max_chars,
        )
    except Exception as exc:
        log.exception("Failed to read artifact %s", artifact_id)
        return fail(f"Error reading artifact {artifact_id}: {exc}")


def _render_artifact_slice(
    *,
    artifact: Any,
    content: str,
    start_line: int,
    end_line: int | None,
    max_chars: int | None,
) -> dict[str, Any]:
    lines = content.splitlines()
    total_lines = max(len(lines), 1)

    effective_start = start_line
    default_end = effective_start + DEFAULT_MAX_LINES - 1
    if end_line is None:
        effective_end = default_end
    else:
        # Enforce a minimum chunk size to prevent inefficient micro-pagination loops
        min_end = effective_start + 99
        requested_end = min(end_line, effective_start + MAX_LINES_PER_READ - 1)
        effective_end = max(requested_end, min_end)

    clipped_end = min(effective_end, total_lines)

    char_limit = min(max_chars or DEFAULT_MAX_CHARS, MAX_PRINT_CHARS)

    if effective_start > total_lines:
        text = f"[EOF: Start line {effective_start} is past the end of the artifact. The artifact only has {total_lines} lines. Stop reading and synthesize the results.]"
    else:
        selected_lines = lines[effective_start - 1 : clipped_end]
        if not selected_lines and content:
            selected_lines = [content]

        text = "\n".join(selected_lines)
    stale_warning = _artifact_stale_warning(artifact)
    if stale_warning:
        text = f"{stale_warning}\n\n{text}" if text else stale_warning
    truncated_by_lines = effective_end < total_lines
    if len(text) > char_limit:
        text = text[:char_limit].rstrip()
        truncated_by_chars = True
    else:
        truncated_by_chars = False

    metadata = {
        "artifact_id": artifact.artifact_id,
        "path": artifact.source,
        "source_artifact_id": artifact.artifact_id,
        "line_start": effective_start,
        "line_end": clipped_end if effective_start <= total_lines else total_lines,
        "total_lines": total_lines,
        "truncated": truncated_by_lines or truncated_by_chars,
        "read_mode": "paged",
        "max_chars": char_limit,
    }
    if effective_start > total_lines:
        metadata["eof_overread"] = True
        metadata["requested_start_line"] = effective_start
        metadata["artifact_total_lines"] = total_lines
    artifact_metadata = artifact.metadata if isinstance(getattr(artifact, "metadata", None), dict) else {}
    if artifact_metadata.get("stale"):
        metadata["stale"] = True
        metadata["artifact_stale_reason"] = str(artifact_metadata.get("artifact_stale_reason") or "")
        metadata["authoritative_path"] = str(artifact_metadata.get("authoritative_path") or "")
    # Keep the tool contract raw; human-facing framing is added by the UI display layer.
    return ok(text, metadata=metadata)


def _artifact_stale_warning(artifact: Any) -> str:
    metadata = artifact.metadata if isinstance(getattr(artifact, "metadata", None), dict) else {}
    if not metadata.get("stale"):
        return ""
    authoritative_path = str(metadata.get("authoritative_path") or metadata.get("target_path") or "").strip()
    if authoritative_path:
        return (
            "WARNING: This artifact is stale because its write session was promoted. "
            f"Use `file_read(path='{authoritative_path}')` for the current authoritative file."
        )
    return "WARNING: This artifact is stale because its write session was promoted."


def _coerce_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_artifact_record(state: LoopState, artifact_id: str) -> Any | None:
    artifacts = getattr(state, "artifacts", None)
    if not isinstance(artifacts, dict) or not artifacts:
        return None

    for candidate in _artifact_lookup_candidates(artifact_id):
        artifact = artifacts.get(candidate)
        if artifact is not None:
            return artifact

    for candidate in _artifact_lookup_candidates(artifact_id):
        normalized = str(candidate or "").strip().upper()
        if not normalized.startswith("A"):
            continue
        try:
            numeric_val = int(normalized[1:])
        except ValueError:
            continue
        for aid, record in artifacts.items():
            if not isinstance(aid, str) or not aid.upper().startswith("A"):
                continue
            try:
                if int(aid[1:]) == numeric_val:
                    return record
            except ValueError:
                continue
    return None


def _artifact_lookup_candidates(artifact_id: str) -> tuple[str, ...]:
    raw = str(artifact_id or "").strip()
    if not raw:
        return ()

    candidates: list[str] = [raw]
    extracted = _extract_artifact_id_token(raw)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    upper_raw = raw.upper()
    if upper_raw not in candidates:
        candidates.append(upper_raw)
    if extracted:
        upper_extracted = extracted.upper()
        if upper_extracted not in candidates:
            candidates.append(upper_extracted)
    return tuple(candidates)


def _extract_artifact_id_token(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = _ARTIFACT_TOKEN_RE.search(text)
    if match is None:
        return None
    return match.group(0).upper()
