from __future__ import annotations

from typing import Any

from ..normalization import dedupe_keep_tail
from ..state_schema import MemoryEntry
from .touched_symbols import (
    SYMBOL_CAPTURE_LIMIT,
    extract_symbol_candidates_from_file,
    extract_symbol_candidates_from_path,
    extract_symbol_candidates_from_text,
)


def _record_touched_symbols_from_mutation(
    service: Any,
    *,
    tool_name: str,
    result: Any,
    arguments: dict[str, Any] | None,
    artifact: Any,
    mutated_path: str,
) -> None:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    text_chunks: list[str] = []
    if isinstance(arguments, dict):
        for key in ("content", "replacement_text", "target_text", "patch", "diff"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                text_chunks.append(value)
        for key in ("section_name", "section_id", "next_section_name"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                text_chunks.append(f"def {value.strip()}(): pass")
    for key in ("content", "replacement_text", "target_text"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            text_chunks.append(value)
    touched_symbols = metadata.get("touched_symbols")
    if isinstance(touched_symbols, list):
        existing_symbols = [str(item).strip() for item in touched_symbols if str(item).strip()]
    else:
        existing_symbols = []
    for key in ("replacement_text_preview", "target_text_preview"):
        preview_payload = metadata.get(key)
        if isinstance(preview_payload, dict):
            preview = str(preview_payload.get("preview") or "").replace("\\n", "\n").strip()
            if preview:
                text_chunks.append(preview)

    candidates: list[str] = []
    for chunk in text_chunks:
        candidates.extend(extract_symbol_candidates_from_text(chunk))

    path_candidate = str(mutated_path or metadata.get("path") or "").strip()
    if not path_candidate and isinstance(arguments, dict):
        path_candidate = str(arguments.get("path") or "").strip()
    if not path_candidate and artifact is not None:
        path_candidate = str(getattr(artifact, "source", "") or "").strip()
    candidates.extend(extract_symbol_candidates_from_path(path_candidate))
    candidates.extend(extract_symbol_candidates_from_file(path_candidate, cwd=str(service.harness.state.cwd or "")))

    deduped = dedupe_keep_tail(existing_symbols + [token for token in candidates if token], limit=SYMBOL_CAPTURE_LIMIT)
    if not deduped:
        return
    existing = service.harness.state.scratchpad.get("_touched_symbols")
    existing_list = [str(item).strip() for item in existing] if isinstance(existing, list) else []
    merged = dedupe_keep_tail(existing_list + deduped, limit=SYMBOL_CAPTURE_LIMIT)
    if not merged:
        return
    service.harness.state.scratchpad["_touched_symbols"] = merged
    runlog = getattr(service.harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "coding_symbols_captured",
            "captured touched symbols from file mutation",
            tool_name=tool_name,
            path=path_candidate,
            symbol_count=len(deduped),
            symbols=deduped[:8],
        )
