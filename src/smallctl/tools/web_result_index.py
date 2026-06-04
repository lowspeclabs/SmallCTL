from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from .web_artifact_refs import web_fetch_artifact_id_re

_WEB_FETCH_ARTIFACT_ID_RE = web_fetch_artifact_id_re()


def next_fetch_id(state: Any) -> str:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    counter = scratchpad.get("_web_fetch_id_counter")
    try:
        next_counter = int(counter or 0) + 1
    except (TypeError, ValueError):
        next_counter = 1
    scratchpad["_web_fetch_id_counter"] = next_counter
    return f"r{next_counter}"


def assign_fetch_ids(state: Any, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        payload = dict(result)
        fetch_id = str(payload.get("fetch_id") or "").strip() or next_fetch_id(state)
        payload["fetch_id"] = fetch_id
        assigned.append(payload)
    return assigned


def load_result_index(state: Any) -> dict[str, dict[str, Any]]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    index = scratchpad.get("_web_result_index")
    if not isinstance(index, dict):
        index = {}
        scratchpad["_web_result_index"] = index
    return index


def load_fetch_artifact_index(state: Any) -> dict[str, str]:
    """Map of result_id/fetch_id -> artifact_id for previously fetched results."""
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    index = scratchpad.get("_web_fetch_artifact_index")
    if not isinstance(index, dict):
        index = {}
        scratchpad["_web_fetch_artifact_index"] = index
    return index


def record_fetch_artifact_mapping(
    state: Any, result_id: str, fetch_id: str | None, artifact_id: str
) -> None:
    index = load_fetch_artifact_index(state)
    if not artifact_id:
        return
    if result_id:
        index[result_id] = artifact_id
    if fetch_id and fetch_id != result_id:
        index[fetch_id] = artifact_id


def find_existing_fetch_artifact(state: Any, token: str) -> str | None:
    """Fallback scan of artifacts for a matching previously fetched result."""
    artifacts = getattr(state, "artifacts", None)
    if not isinstance(artifacts, dict):
        return None
    for artifact in artifacts.values():
        metadata = getattr(artifact, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("result_id") or "") == token:
            return str(getattr(artifact, "artifact_id", "") or "")
        if str(metadata.get("fetch_id") or "") == token:
            return str(getattr(artifact, "artifact_id", "") or "")
    return None


def update_result_index(state: Any, results: list[dict[str, Any]]) -> None:
    index = load_result_index(state)
    for result in results:
        result_id = str(result.get("result_id") or "").strip()
        fetch_id = str(result.get("fetch_id") or "").strip()
        payload = dict(result)
        if result_id:
            payload["canonical_result_id"] = result_id
            index[result_id] = dict(payload)
        if fetch_id:
            payload["canonical_result_id"] = result_id or str(payload.get("canonical_result_id") or "").strip()
            index[fetch_id] = dict(payload)


def normalized_result_url_candidates(state: Any) -> tuple[set[str], set[str]]:
    index = load_result_index(state)
    exact_urls: set[str] = set()
    domains: set[str] = set()
    for result in index.values():
        if not isinstance(result, dict):
            continue
        for key in ("url", "canonical_url"):
            candidate = str(result.get(key) or "").strip()
            if candidate:
                exact_urls.add(candidate)
                domain = urlparse(candidate).netloc.strip().lower()
                if domain:
                    domains.add(domain)
        domain = str(result.get("domain") or "").strip().lower()
        if domain:
            domains.add(domain)
    return exact_urls, domains


def preferred_known_results(state: Any) -> list[dict[str, Any]]:
    scratchpad = getattr(state, "scratchpad", None)
    index = load_result_index(state)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _append_result(result: Any) -> None:
        if not isinstance(result, dict):
            return
        identity = str(
            result.get("canonical_result_id")
            or result.get("result_id")
            or result.get("fetch_id")
            or result.get("canonical_url")
            or result.get("url")
            or ""
        ).strip().lower()
        if not identity or identity in seen:
            return
        seen.add(identity)
        ordered.append(result)

    if isinstance(scratchpad, dict):
        for key in scratchpad.get("_web_last_search_result_ids") or []:
            _append_result(index.get(str(key).strip()))
    for result in index.values():
        _append_result(result)
    return ordered


def suggested_result_for_url(state: Any, *, url: str) -> dict[str, Any] | None:
    candidate_domain = urlparse(str(url or "").strip()).netloc.strip().lower()
    if not candidate_domain:
        return None
    for result in preferred_known_results(state):
        domain = str(result.get("domain") or "").strip().lower()
        if not domain:
            domain = urlparse(str(result.get("canonical_url") or result.get("url") or "").strip()).netloc.strip().lower()
        if domain == candidate_domain:
            return result
    return None


def reject_invented_result_domain_url(state: Any, *, url: str | None, result_id: str | None) -> str | None:
    if result_id or not url:
        return None
    exact_urls, known_domains = normalized_result_url_candidates(state)
    if not exact_urls or not known_domains:
        return None
    normalized_url = str(url or "").strip()
    if not normalized_url or normalized_url in exact_urls:
        return None
    candidate_domain = urlparse(normalized_url).netloc.strip().lower()
    if not candidate_domain or candidate_domain not in known_domains:
        return None
    # Same-domain follow-up links are allowed. Public-web validation happens in the
    # fetch layer, and strict exact-match blocking prevented legitimate navigation.
    return None


def known_fetch_ids(state: Any, *, limit: int = 6) -> list[str]:
    fetch_ids: list[str] = []
    seen: set[str] = set()
    for result in preferred_known_results(state):
        fetch_id = str(result.get("fetch_id") or "").strip()
        if not fetch_id or fetch_id in seen:
            continue
        seen.add(fetch_id)
        fetch_ids.append(fetch_id)
        if len(fetch_ids) >= limit:
            break
    return fetch_ids


def unknown_result_id_error(state: Any, token: str) -> tuple[str, dict[str, Any]]:
    valid_fetch_ids = known_fetch_ids(state)
    reason = "web_fetch_result_id_not_found"
    metadata: dict[str, Any] = {
        "reason": reason,
        "requested_result_id": token,
        "valid_fetch_ids": list(valid_fetch_ids),
    }
    artifact_like = bool(_WEB_FETCH_ARTIFACT_ID_RE.fullmatch(str(token or "").strip()))
    if artifact_like:
        message = (
            f"Invalid result_id: {token}. Artifact IDs like `{token}` are for `artifact_read`, "
            "not direct `web_fetch` targets."
        )
    else:
        message = (
            f"Invalid result_id: {token}. It was not found in current session search results "
            "or preserved web-search artifacts."
        )
    if valid_fetch_ids:
        message += " Valid fetch IDs from recent searches: " + ", ".join(valid_fetch_ids) + "."
    else:
        message += " Run `web_search(...)` first, then use the returned fetch ID such as `r1`."
    return message, metadata
