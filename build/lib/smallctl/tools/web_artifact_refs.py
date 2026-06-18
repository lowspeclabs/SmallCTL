from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_WEB_FETCH_ARTIFACT_RESULT_ALIAS_RE = re.compile(r"^(A\d+)\s*[-:_#/]\s*(\d+)$", re.IGNORECASE)
_WEB_FETCH_ARTIFACT_ID_RE = re.compile(r"^(A\d+)$", re.IGNORECASE)
_WEB_SEARCH_RESULT_RANK_RE = re.compile(r"^\s*(\d+)\.\s+")
_WEB_SEARCH_FETCH_ID_RE = re.compile(r"web_fetch\(result_id='([^']+)'\)")
_WEB_SEARCH_RESULT_ID_LINE_RE = re.compile(r"^\s*Result ID:\s*(\S+)\s*$")
_WEB_SEARCH_FETCH_ID_LINE_RE = re.compile(r"^\s*Fetch ID:\s*(\S+)\s*$")
_WEB_SEARCH_URL_LINE_RE = re.compile(r"^\s*URL:\s*(\S+)\s*$")
_WEB_SEARCH_DOMAIN_LINE_RE = re.compile(r"^\s*Domain:\s*(\S+)\s*$")


def web_fetch_artifact_id_re() -> re.Pattern[str]:
    return _WEB_FETCH_ARTIFACT_ID_RE


def _load_result_index(state: Any) -> dict[str, dict[str, Any]]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    index = scratchpad.get("_web_result_index")
    if not isinstance(index, dict):
        index = {}
        scratchpad["_web_result_index"] = index
    return index


def _state_artifacts(state: Any) -> dict[str, Any]:
    artifacts = getattr(state, "artifacts", None)
    return artifacts if isinstance(artifacts, dict) else {}


def canonical_artifact_id(state: Any, candidate: str) -> str:
    normalized = str(candidate or "").strip()
    if not normalized:
        return normalized
    artifacts = _state_artifacts(state)
    if not artifacts:
        return normalized.upper()
    if normalized in artifacts:
        return normalized
    upper_candidate = normalized.upper()
    if upper_candidate in artifacts:
        return upper_candidate
    lowered = normalized.lower()
    for artifact_id in artifacts.keys():
        if isinstance(artifact_id, str) and artifact_id.lower() == lowered:
            return artifact_id
    return upper_candidate


def artifact_text(artifact: Any) -> str:
    preview = str(getattr(artifact, "preview_text", "") or "").strip()
    inline = str(getattr(artifact, "inline_content", "") or "").strip()
    path_value = str(getattr(artifact, "content_path", "") or "").strip()
    if path_value:
        try:
            text = Path(path_value).read_text(encoding="utf-8")
            if text.strip():
                return text
        except OSError:
            pass
    if inline:
        return inline
    return preview


def recover_web_search_entries_from_artifact(state: Any, artifact_id: str) -> list[dict[str, Any]]:
    artifact = _state_artifacts(state).get(canonical_artifact_id(state, artifact_id))
    if artifact is None:
        return []

    metadata = getattr(artifact, "metadata", None)
    metadata = metadata if isinstance(metadata, dict) else {}
    result_ids = [
        str(item).strip()
        for item in (metadata.get("web_result_ids") or [])
        if str(item).strip()
    ]
    fetch_ids = [
        str(item).strip()
        for item in (metadata.get("web_fetch_ids") or [])
        if str(item).strip()
    ]

    text = artifact_text(artifact)
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = str(raw_line or "")
        rank_match = _WEB_SEARCH_RESULT_RANK_RE.match(line)
        if rank_match is not None:
            if current:
                entries.append(current)
            current = {"rank": int(rank_match.group(1))}
            continue
        if current is None:
            continue
        fetch_match = _WEB_SEARCH_FETCH_ID_RE.search(line)
        if fetch_match is not None:
            current["fetch_id"] = fetch_match.group(1).strip()
        fetch_id_line = _WEB_SEARCH_FETCH_ID_LINE_RE.match(line)
        if fetch_id_line is not None:
            current["fetch_id"] = fetch_id_line.group(1).strip()
        result_id_match = _WEB_SEARCH_RESULT_ID_LINE_RE.match(line)
        if result_id_match is not None:
            current["result_id"] = result_id_match.group(1).strip()
        url_match = _WEB_SEARCH_URL_LINE_RE.match(line)
        if url_match is not None:
            url_value = url_match.group(1).strip()
            current["url"] = url_value
            current["canonical_url"] = url_value
        domain_match = _WEB_SEARCH_DOMAIN_LINE_RE.match(line)
        if domain_match is not None:
            current["domain"] = domain_match.group(1).strip().lower()
    if current:
        entries.append(current)

    if not entries and (result_ids or fetch_ids):
        index = _load_result_index(state)
        max_count = max(len(result_ids), len(fetch_ids))
        for idx in range(max_count):
            entry: dict[str, Any] = {"rank": idx + 1}
            result_id = result_ids[idx] if idx < len(result_ids) else ""
            fetch_id = fetch_ids[idx] if idx < len(fetch_ids) else ""
            indexed = (
                index.get(fetch_id)
                if fetch_id and isinstance(index.get(fetch_id), dict)
                else index.get(result_id)
                if result_id and isinstance(index.get(result_id), dict)
                else None
            )
            if isinstance(indexed, dict):
                entry.update(dict(indexed))
            if result_id:
                entry.setdefault("result_id", result_id)
            if fetch_id:
                entry.setdefault("fetch_id", fetch_id)
            entries.append(entry)

    for idx, entry in enumerate(entries):
        if idx < len(result_ids) and result_ids[idx]:
            entry.setdefault("result_id", result_ids[idx])
        if idx < len(fetch_ids) and fetch_ids[idx]:
            entry.setdefault("fetch_id", fetch_ids[idx])
    return entries


def resolve_search_result_from_artifact_reference(
    state: Any,
    token: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    normalized = str(token or "").strip()
    if not normalized:
        return None, {}

    artifact_id = ""
    rank = 1
    match = _WEB_FETCH_ARTIFACT_RESULT_ALIAS_RE.fullmatch(normalized)
    if match is not None:
        artifact_id = canonical_artifact_id(state, match.group(1))
        rank = int(match.group(2))
    elif _WEB_FETCH_ARTIFACT_ID_RE.fullmatch(normalized) is not None:
        artifact_id = canonical_artifact_id(state, normalized)
    if not artifact_id or rank <= 0:
        return None, {}

    entries = recover_web_search_entries_from_artifact(state, artifact_id)
    if rank > len(entries):
        return None, {}
    entry = dict(entries[rank - 1])
    if not entry:
        return None, {}
    metadata = {
        "argument_repair": "web_fetch_result_alias_to_search_result",
        "repair_field": "result_id",
        "original_result_id": normalized,
        "resolved_search_artifact_id": artifact_id,
        "resolved_search_result_rank": rank,
    }
    resolved_result_id = str(
        entry.get("canonical_result_id")
        or entry.get("result_id")
        or entry.get("fetch_id")
        or ""
    ).strip()
    if resolved_result_id:
        metadata["resolved_result_id"] = resolved_result_id
    return entry, metadata
