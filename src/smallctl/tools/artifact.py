from __future__ import annotations

import logging
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


def artifact_recall(
    state: LoopState,
    *,
    artifact_id: str,
) -> dict[str, Any]:
    """
    Retrieve the full text content of an artifact and bring it into your context window.
    Use this when you need to closely analyze the entire content of a previous result.
    """
    return artifact_read(
        state, 
        artifact_id=artifact_id, 
        start_line=1, 
        end_line=20000, 
        max_chars=MAX_PRINT_CHARS
    )


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
    result = artifact_recall(harness.state, artifact_id=artifact_id)
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
    artifact = state.artifacts.get(artifact_id)
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

        if not matches:
            return ok(f"No matches found for '{query}' in artifact {artifact_id}")

        output_lines = [f"Found {len(matches)} matches in {artifact_id}:"]
        for m in matches:
            output_lines.append(f"L{m['line']}: {m['content']}")

        metadata = {
            "artifact_id": artifact_id,
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

    Args:
        state: The current loop state (injected).
        artifact_id: The ID of the artifact to read (e.g., 'A0001').
        start_line: Optional 1-based inclusive starting line.
        end_line: Optional 1-based inclusive ending line.
        max_chars: Optional character cap for the returned slice.
    """
    artifact = state.artifacts.get(artifact_id)
    if not artifact:
        # Leniency: if ID looks like A003 or A3, try to match by numeric index
        if artifact_id.startswith("A"):
            try:
                numeric_val = int(artifact_id[1:])
                for aid, record in state.artifacts.items():
                    if aid.startswith("A"):
                        try:
                            if int(aid[1:]) == numeric_val:
                                artifact = record
                                break
                        except ValueError:
                            continue
            except ValueError:
                pass

    if not artifact:
        return fail(f"Artifact {artifact_id} not found in state.")

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
    effective_start = min(start_line, total_lines)
    default_end = min(effective_start + DEFAULT_MAX_LINES - 1, total_lines)
    effective_end = default_end if end_line is None else min(end_line, effective_start + MAX_LINES_PER_READ - 1, total_lines)
    char_limit = min(max_chars or DEFAULT_MAX_CHARS, MAX_PRINT_CHARS)
    selected_lines = lines[effective_start - 1 : effective_end]
    if not selected_lines and content:
        selected_lines = [content]
    text = "\n".join(selected_lines)
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
        "line_end": effective_end,
        "total_lines": total_lines,
        "truncated": truncated_by_lines or truncated_by_chars,
        "read_mode": "paged",
        "max_chars": char_limit,
    }
    # Keep the tool contract raw; human-facing framing is added by the UI display layer.
    return ok(text, metadata=metadata)


def _coerce_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
