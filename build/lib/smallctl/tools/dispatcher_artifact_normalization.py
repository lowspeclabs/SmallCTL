from __future__ import annotations

import re
from typing import Any

from ..models.tool_result import ToolEnvelope

_ARTIFACT_TOKEN_RE = re.compile(r"\bA\d+\b", re.IGNORECASE)
_WRITE_SESSION_ARTIFACT_ID_RE = re.compile(
    r"^ws[-_A-Za-z0-9]+(?:__[^/\s]+)*__stage(?:\.[A-Za-z0-9]+)?$",
    re.IGNORECASE,
)
_RECENT_ARTIFACT_ALIASES = {
    "above",
    "above artifact",
    "above output",
    "current artifact",
    "current output",
    "last",
    "last artifact",
    "last output",
    "latest",
    "latest artifact",
    "latest output",
    "most recent",
    "most recent artifact",
    "most recent output",
    "previous artifact",
    "recent",
    "recent artifact",
    "recent output",
    "that artifact",
    "that output",
    "the artifact above",
    "the latest artifact",
    "the latest output",
    "the most recent artifact",
    "the most recent output",
    "the previous artifact",
}
_WEB_FETCH_ARTIFACT_RESULT_ALIAS_RE = re.compile(r"^(A\d+)\s*[-:_#/]\s*(\d+)$", re.IGNORECASE)
_WEB_FETCH_ORDINAL_RESULT_ALIAS_RE = re.compile(
    r"^(?:result|res|rank|item)?\s*[-#: ]?\s*(\d+)$",
    re.IGNORECASE,
)


def normalize_artifact_read_request(
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not isinstance(arguments, dict):
        return "artifact_read", arguments, {}

    repaired = dict(arguments)

    explicit_artifact_id = repaired.get("artifact_id")
    if isinstance(explicit_artifact_id, str):
        normalized_artifact_id, metadata = normalize_artifact_reference(
            explicit_artifact_id,
            field_name="artifact_id",
            state=state,
        )
        if normalized_artifact_id is not None:
            repaired["artifact_id"] = normalized_artifact_id
            return "artifact_read", repaired, metadata
        if looks_like_file_path(explicit_artifact_id):
            return "file_read", build_file_read_args(repaired, explicit_artifact_id), {
                "rewritten_from_tool": "artifact_read",
                "routing_reason": "artifact_id_to_file_read",
                "repair_field": "artifact_id",
            }

    for field_name in ("path", "id"):
        raw_value = repaired.get(field_name)
        if not isinstance(raw_value, str):
            continue

        normalized_artifact_id, metadata = normalize_artifact_reference(
            raw_value,
            field_name=field_name,
            state=state,
        )
        if normalized_artifact_id is not None:
            repaired.pop(field_name, None)
            repaired["artifact_id"] = normalized_artifact_id
            return "artifact_read", repaired, metadata

        if looks_like_file_path(raw_value):
            return "file_read", build_file_read_args(repaired, raw_value), {
                "rewritten_from_tool": "artifact_read",
                "routing_reason": "artifact_path_to_file_read",
                "repair_field": field_name,
            }

    recent_artifact_id = most_recent_artifact_id(state)
    if recent_artifact_id and artifact_read_implicitly_targets_recent(arguments):
        repaired["artifact_id"] = recent_artifact_id
        return "artifact_read", repaired, {
            "argument_repair": "artifact_read_recent_fallback",
            "resolved_artifact_id": recent_artifact_id,
        }

    return "artifact_read", arguments, {}


def normalize_web_fetch_request(
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(arguments, dict):
        return arguments, {}

    repaired = dict(arguments)
    metadata: dict[str, Any] = {}

    raw_fetch_id = repaired.get("fetch_id")
    normalized_fetch_id = str(raw_fetch_id).strip() if isinstance(raw_fetch_id, str) else ""
    raw_result_id = repaired.get("result_id")
    normalized_existing_result_id = str(raw_result_id).strip() if isinstance(raw_result_id, str) else ""
    if normalized_fetch_id and not normalized_existing_result_id:
        repaired["result_id"] = normalized_fetch_id
        repaired.pop("fetch_id", None)
        metadata.update(
            {
                "field_alias_repair": "web_fetch_fetch_id_to_result_id",
                "original_fetch_id": normalized_fetch_id,
            }
        )
    elif normalized_fetch_id and normalized_existing_result_id == normalized_fetch_id:
        repaired.pop("fetch_id", None)
        metadata.update(
            {
                "field_alias_repair": "web_fetch_fetch_id_to_result_id",
                "original_fetch_id": normalized_fetch_id,
            }
        )

    raw_result_id = repaired.get("result_id")
    if not isinstance(raw_result_id, str):
        return repaired, metadata

    normalized_result_id = str(raw_result_id).strip()
    if not normalized_result_id:
        return repaired, metadata

    known_results = web_result_index(state)
    if normalized_result_id in known_results:
        return repaired, metadata

    resolved_result_id, alias_metadata = resolve_web_fetch_result_alias(
        normalized_result_id,
        state=state,
    )
    if not resolved_result_id:
        return repaired, metadata

    repaired["result_id"] = resolved_result_id
    metadata.update(alias_metadata)
    return repaired, metadata


def normalize_artifact_reference(
    value: str,
    *,
    field_name: str,
    state: Any | None = None,
) -> tuple[str | None, dict[str, Any]]:
    canonical_key = canonical_artifact_key(value, state=state)
    if canonical_key is not None:
        return canonical_key, {
            "argument_repair": "artifact_read_alias_to_existing_key",
            "repair_field": field_name,
            "resolved_artifact_id": canonical_key,
        }

    normalized = normalize_recent_artifact_alias(value, state=state)
    if normalized is not None:
        return normalized, {
            "argument_repair": "artifact_read_recent_fallback",
            "repair_field": field_name,
            "resolved_artifact_id": normalized,
        }

    if looks_like_write_session_artifact_alias(value):
        preserved = str(value or "").strip()
        return preserved, {
            "argument_repair": "artifact_read_preserve_write_session_alias",
            "repair_field": field_name,
            "resolved_artifact_id": preserved,
        }

    extracted = extract_artifact_id_token(value)
    if extracted is None:
        return None, {}

    canonical = canonical_artifact_id(extracted, state=state)
    metadata: dict[str, Any] = {
        "argument_repair": "artifact_read_alias_to_artifact_id",
        "repair_field": field_name,
        "resolved_artifact_id": canonical,
    }
    return canonical, metadata


def normalize_recent_artifact_alias(value: str, *, state: Any | None = None) -> str | None:
    alias = " ".join(str(value or "").strip().lower().split())
    if alias not in _RECENT_ARTIFACT_ALIASES:
        return None
    return most_recent_artifact_id(state)


def extract_artifact_id_token(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None

    match = _ARTIFACT_TOKEN_RE.search(text)
    if match is None:
        return None
    return match.group(0).upper()


def canonical_artifact_id(candidate: str, *, state: Any | None = None) -> str:
    normalized = str(candidate or "").strip()
    if not normalized:
        return normalized

    artifacts = getattr(state, "artifacts", None)
    if not isinstance(artifacts, dict) or not artifacts:
        return normalized.upper() if normalized.upper().startswith("A") else normalized

    if normalized in artifacts:
        return normalized

    upper_candidate = normalized.upper()
    if upper_candidate in artifacts:
        return upper_candidate

    if not upper_candidate.startswith("A"):
        return normalized

    try:
        numeric_value = int(upper_candidate[1:])
    except ValueError:
        return upper_candidate

    for artifact_id in artifacts.keys():
        if not isinstance(artifact_id, str) or not artifact_id.upper().startswith("A"):
            continue
        try:
            if int(artifact_id[1:]) == numeric_value:
                return artifact_id
        except ValueError:
            continue
    return upper_candidate


def canonical_artifact_key(candidate: str, *, state: Any | None = None) -> str | None:
    normalized = str(candidate or "").strip()
    if not normalized:
        return None

    artifacts = getattr(state, "artifacts", None)
    if not isinstance(artifacts, dict) or not artifacts:
        return None

    if normalized in artifacts:
        return normalized

    lowered = normalized.lower()
    for artifact_id in artifacts.keys():
        if isinstance(artifact_id, str) and artifact_id.lower() == lowered:
            return artifact_id
    return None


def artifact_read_implicitly_targets_recent(arguments: dict[str, Any]) -> bool:
    if not isinstance(arguments, dict):
        return False

    non_empty_keys = {
        key
        for key, value in arguments.items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }
    if not non_empty_keys:
        return True
    return non_empty_keys <= {"start_line", "end_line", "max_chars"}


def most_recent_artifact_id(state: Any | None) -> str | None:
    if state is None:
        return None

    artifacts = getattr(state, "artifacts", None)
    if not isinstance(artifacts, dict) or not artifacts:
        return None

    retrieval_cache = getattr(state, "retrieval_cache", None)
    if isinstance(retrieval_cache, list):
        for artifact_id in reversed(retrieval_cache):
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                continue
            canonical = canonical_artifact_id(artifact_id, state=state)
            if canonical in artifacts:
                return canonical

    for artifact_id in reversed(list(artifacts.keys())):
        if isinstance(artifact_id, str) and artifact_id.strip():
            return artifact_id
    return None


def build_file_read_args(arguments: dict[str, Any], path_value: str) -> dict[str, Any]:
    repaired: dict[str, Any] = {"path": path_value}
    for key in ("start_line", "end_line"):
        if key in arguments:
            repaired[key] = arguments[key]
    if "max_bytes" in arguments:
        repaired["max_bytes"] = arguments["max_bytes"]
    elif "max_chars" in arguments:
        repaired["max_bytes"] = arguments["max_chars"]
    return repaired


def looks_like_file_path(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False

    if candidate.startswith(("./", "../", "/", "~")):
        return True
    if "\\" in candidate or "/" in candidate:
        return True
    if "." in candidate:
        return True
    return False


def looks_like_write_session_artifact_alias(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    return _WRITE_SESSION_ARTIFACT_ID_RE.fullmatch(candidate) is not None


def resolve_web_fetch_result_alias(
    value: str,
    *,
    state: Any | None = None,
) -> tuple[str | None, dict[str, Any]]:
    normalized = str(value or "").strip()
    if not normalized:
        return None, {}

    artifact_match = _WEB_FETCH_ARTIFACT_RESULT_ALIAS_RE.fullmatch(normalized)
    if artifact_match is not None:
        artifact_id = canonical_artifact_id(artifact_match.group(1), state=state)
        rank = int(artifact_match.group(2))
        resolved = web_result_id_for_rank(state, artifact_id=artifact_id, rank=rank)
        if resolved:
            return resolved, {
                "argument_repair": "web_fetch_result_alias_to_search_result",
                "repair_field": "result_id",
                "original_result_id": normalized,
                "resolved_result_id": resolved,
                "resolved_search_artifact_id": artifact_id,
                "resolved_search_result_rank": rank,
            }

    canonical_artifact = canonical_artifact_key(normalized, state=state)
    if canonical_artifact and canonical_artifact.upper().startswith("A"):
        resolved = web_result_id_for_rank(state, artifact_id=canonical_artifact, rank=1)
        if resolved:
            return resolved, {
                "argument_repair": "web_fetch_result_alias_to_search_result",
                "repair_field": "result_id",
                "original_result_id": normalized,
                "resolved_result_id": resolved,
                "resolved_search_artifact_id": canonical_artifact,
                "resolved_search_result_rank": 1,
            }

    ordinal_match = _WEB_FETCH_ORDINAL_RESULT_ALIAS_RE.fullmatch(normalized)
    if ordinal_match is not None:
        rank = int(ordinal_match.group(1))
        resolved = web_result_id_for_rank(state, artifact_id=None, rank=rank)
        if resolved:
            artifact_id = most_recent_web_search_artifact_id(state)
            metadata = {
                "argument_repair": "web_fetch_result_alias_to_search_result",
                "repair_field": "result_id",
                "original_result_id": normalized,
                "resolved_result_id": resolved,
                "resolved_search_result_rank": rank,
            }
            if artifact_id:
                metadata["resolved_search_artifact_id"] = artifact_id
            return resolved, metadata

    return None, {}


def web_result_id_for_rank(
    state: Any | None,
    *,
    artifact_id: str | None,
    rank: int,
) -> str | None:
    if rank <= 0:
        return None

    result_ids: list[str] = []
    if artifact_id:
        artifact_map = web_search_artifact_results(state)
        result_ids = list(artifact_map.get(artifact_id) or [])
        if not result_ids and artifact_id == most_recent_web_search_artifact_id(state):
            result_ids = list(last_web_search_result_ids(state))
    else:
        result_ids = list(last_web_search_result_ids(state))

    if rank > len(result_ids):
        return None

    candidate = str(result_ids[rank - 1] or "").strip()
    if not candidate:
        return None

    known_results = web_result_index(state)
    if known_results and candidate not in known_results:
        return None
    return candidate


def web_result_index(state: Any | None) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    index = scratchpad.get("_web_result_index")
    return index if isinstance(index, dict) else {}


def web_search_artifact_results(state: Any | None) -> dict[str, list[str]]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    mapping = scratchpad.get("_web_search_artifact_results")
    return mapping if isinstance(mapping, dict) else {}


def last_web_search_result_ids(state: Any | None) -> list[str]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return []
    result_ids = scratchpad.get("_web_last_search_result_ids")
    if not isinstance(result_ids, list):
        return []
    return [str(item).strip() for item in result_ids if str(item).strip()]


def most_recent_web_search_artifact_id(state: Any | None) -> str | None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    artifact_id = str(scratchpad.get("_web_last_search_artifact_id") or "").strip()
    return artifact_id or None
