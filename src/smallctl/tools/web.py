from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..search_server.app import SearchServerError, get_search_runtime
from ..search_server.config import SearchServerConfig
from ..search_server.models import WebFetchRequest, WebSearchRequest
from ..tool_output_formatting import structured_plain_text, summarize_structured_output
from .common import fail, ok

_MAX_TOOL_LIMIT = 10
_MAX_TOOL_FETCH_CHARS = 20000
_WEB_FETCH_ARTIFACT_RESULT_ALIAS_RE = re.compile(r"^(A\d+)\s*[-:_#/]\s*(\d+)$", re.IGNORECASE)
_WEB_FETCH_ARTIFACT_ID_RE = re.compile(r"^(A\d+)$", re.IGNORECASE)
_WEB_SEARCH_RESULT_RANK_RE = re.compile(r"^\s*(\d+)\.\s+")
_WEB_SEARCH_FETCH_ID_RE = re.compile(r"web_fetch\(result_id='([^']+)'\)")
_WEB_SEARCH_RESULT_ID_LINE_RE = re.compile(r"^\s*Result ID:\s*(\S+)\s*$")
_WEB_SEARCH_FETCH_ID_LINE_RE = re.compile(r"^\s*Fetch ID:\s*(\S+)\s*$")
_WEB_SEARCH_URL_LINE_RE = re.compile(r"^\s*URL:\s*(\S+)\s*$")
_WEB_SEARCH_DOMAIN_LINE_RE = re.compile(r"^\s*Domain:\s*(\S+)\s*$")


def _next_fetch_id(state: Any) -> str:
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


def _assign_fetch_ids(state: Any, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        payload = dict(result)
        fetch_id = str(payload.get("fetch_id") or "").strip() or _next_fetch_id(state)
        payload["fetch_id"] = fetch_id
        assigned.append(payload)
    return assigned


def _budget_state(state: Any) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        scratchpad = {}
        state.scratchpad = scratchpad
    budget = scratchpad.get("_web_budget")
    if not isinstance(budget, dict):
        budget = {"searches_used": 0, "fetches_used": 0, "total_fetched_chars": 0}
        scratchpad["_web_budget"] = budget
    return budget


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


def _update_result_index(state: Any, results: list[dict[str, Any]]) -> None:
    index = _load_result_index(state)
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


def _config_for_harness(harness: Any) -> SearchServerConfig:
    return SearchServerConfig.from_harness(harness)


def _ensure_budget(state: Any, *, config: SearchServerConfig, action: str, chars: int = 0) -> None:
    budget = _budget_state(state)
    if action == "search":
        if int(budget.get("searches_used", 0)) >= config.max_searches_per_run:
            raise SearchServerError("Web search budget exhausted for this run.")
        budget["searches_used"] = int(budget.get("searches_used", 0)) + 1
    elif action == "fetch":
        if int(budget.get("fetches_used", 0)) >= config.max_fetches_per_run:
            raise SearchServerError("Web fetch budget exhausted for this run.")
        budget["fetches_used"] = int(budget.get("fetches_used", 0)) + 1
    elif action == "fetch_chars":
        if int(budget.get("total_fetched_chars", 0)) + int(chars) > config.max_total_fetched_chars:
            raise SearchServerError("Web fetch character budget exhausted for this run.")
        budget["total_fetched_chars"] = int(budget.get("total_fetched_chars", 0)) + int(chars)


def _normalize_domains(domains: list[str] | None) -> list[str] | None:
    cleaned = [str(item).strip().lower() for item in (domains or []) if str(item).strip()]
    return cleaned or None


def _normalized_result_url_candidates(state: Any) -> tuple[set[str], set[str]]:
    index = _load_result_index(state)
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


def _preferred_known_results(state: Any) -> list[dict[str, Any]]:
    scratchpad = getattr(state, "scratchpad", None)
    index = _load_result_index(state)
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


def _suggested_result_for_url(state: Any, *, url: str) -> dict[str, Any] | None:
    candidate_domain = urlparse(str(url or "").strip()).netloc.strip().lower()
    if not candidate_domain:
        return None
    for result in _preferred_known_results(state):
        domain = str(result.get("domain") or "").strip().lower()
        if not domain:
            domain = urlparse(str(result.get("canonical_url") or result.get("url") or "").strip()).netloc.strip().lower()
        if domain == candidate_domain:
            return result
    return None


def _reject_invented_result_domain_url(state: Any, *, url: str | None, result_id: str | None) -> str | None:
    if result_id or not url:
        return None
    exact_urls, known_domains = _normalized_result_url_candidates(state)
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


def _state_artifacts(state: Any) -> dict[str, Any]:
    artifacts = getattr(state, "artifacts", None)
    return artifacts if isinstance(artifacts, dict) else {}


def _canonical_artifact_id(state: Any, candidate: str) -> str:
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


def _artifact_text(artifact: Any) -> str:
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


def _recover_web_search_entries_from_artifact(state: Any, artifact_id: str) -> list[dict[str, Any]]:
    artifact = _state_artifacts(state).get(_canonical_artifact_id(state, artifact_id))
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

    text = _artifact_text(artifact)
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


def _resolve_search_result_from_artifact_reference(
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
        artifact_id = _canonical_artifact_id(state, match.group(1))
        rank = int(match.group(2))
    elif _WEB_FETCH_ARTIFACT_ID_RE.fullmatch(normalized) is not None:
        artifact_id = _canonical_artifact_id(state, normalized)
    if not artifact_id or rank <= 0:
        return None, {}

    entries = _recover_web_search_entries_from_artifact(state, artifact_id)
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


def _known_fetch_ids(state: Any, *, limit: int = 6) -> list[str]:
    fetch_ids: list[str] = []
    seen: set[str] = set()
    for result in _preferred_known_results(state):
        fetch_id = str(result.get("fetch_id") or "").strip()
        if not fetch_id or fetch_id in seen:
            continue
        seen.add(fetch_id)
        fetch_ids.append(fetch_id)
        if len(fetch_ids) >= limit:
            break
    return fetch_ids


def _unknown_result_id_error(state: Any, token: str) -> tuple[str, dict[str, Any]]:
    valid_fetch_ids = _known_fetch_ids(state)
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


async def web_search(
    *,
    harness: Any,
    state: Any,
    query: str,
    domains: list[str] | None = None,
    recency_days: int | None = None,
    limit: int = 5,
    sort: str = "relevance",
) -> dict[str, Any]:
    config = _config_for_harness(harness)
    if not config.enabled:
        return fail("Web search is disabled.")
    try:
        normalized_limit = min(max(1, int(limit or config.default_limit)), min(config.max_limit, _MAX_TOOL_LIMIT))
        _ensure_budget(state, config=config, action="search")
        runtime = get_search_runtime(harness, config=config)
        request = WebSearchRequest(
            query=query,
            domains=_normalize_domains(domains),
            recency_days=recency_days,
            limit=normalized_limit,
            sort=sort,
        )
        response = await runtime.search(request, token=runtime.token)
        payload = response.to_dict()
        payload["results"] = _assign_fetch_ids(state, payload["results"])
        _update_result_index(state, payload["results"])
        return ok(
            payload,
            metadata={
                "provider": response.provider,
                "result_count": len(response.results),
                "recency_enforced": response.recency_enforced,
                "recency_support": response.recency_support,
            },
        )
    except SearchServerError as exc:
        return fail(str(exc), metadata=getattr(exc, "metadata", {}))
    except Exception as exc:
        return fail(str(exc))


async def web_fetch(
    *,
    harness: Any,
    state: Any,
    url: str | None = None,
    result_id: str | None = None,
    fetch_id: str | None = None,
    max_chars: int = 12000,
    extract_mode: str = "article",
) -> dict[str, Any]:
    config = _config_for_harness(harness)
    if not config.enabled:
        return fail("Web search is disabled.")
    requested_url = str(url or "").strip() or None
    requested_result_id = str(result_id or "").strip() or None
    requested_fetch_id = str(fetch_id or "").strip() or None
    requested_result_token = requested_result_id or requested_fetch_id
    selector_count = sum(1 for value in (requested_url, requested_result_id, requested_fetch_id) if value)
    if selector_count != 1:
        return fail("Provide exactly one of url, result_id, or fetch_id.")
    try:
        bounded_max_chars = min(max(1, int(max_chars or config.default_fetch_chars)), min(config.max_fetch_chars, _MAX_TOOL_FETCH_CHARS))
        invented_url_error = _reject_invented_result_domain_url(state, url=requested_url, result_id=requested_result_token)
        if invented_url_error:
            return fail(invented_url_error, metadata={"reason": "web_fetch_url_not_from_results"})
        _ensure_budget(state, config=config, action="fetch")
        runtime = get_search_runtime(harness, config=config)
        resolved_result_id = requested_result_token
        resolved_url = requested_url
        indexed_result = None
        alias_metadata: dict[str, Any] = {}
        if requested_result_token:
            indexed_result = _load_result_index(state).get(requested_result_token)
            if isinstance(indexed_result, dict):
                resolved_result_id = str(
                    indexed_result.get("canonical_result_id")
                    or indexed_result.get("result_id")
                    or requested_result_token
                ).strip() or requested_result_token
            else:
                indexed_result, alias_metadata = _resolve_search_result_from_artifact_reference(
                    state,
                    requested_result_token,
                )
                if isinstance(indexed_result, dict):
                    resolved_result_id = str(
                        indexed_result.get("canonical_result_id")
                        or indexed_result.get("result_id")
                        or indexed_result.get("fetch_id")
                        or requested_result_token
                    ).strip() or requested_result_token
            if not isinstance(indexed_result, dict):
                error_message, error_metadata = _unknown_result_id_error(state, requested_result_token)
                return fail(error_message, metadata=error_metadata)
            resolved_url = str(
                indexed_result.get("canonical_url")
                or indexed_result.get("url")
                or ""
            ).strip() or None
        request = WebFetchRequest(
            url=resolved_url,
            result_id=None if resolved_url else resolved_result_id,
            max_chars=bounded_max_chars,
            extract_mode=extract_mode,
        )
        response, full_text, citation = await runtime.fetch(request, token=runtime.token)
        charged_chars = min(len(full_text), bounded_max_chars)
        _ensure_budget(state, config=config, action="fetch_chars", chars=charged_chars)
        if requested_result_token:
            source = indexed_result if isinstance(indexed_result, dict) else _load_result_index(state).get(resolved_result_id or "")
            if source is not None:
                source_id = str(
                    source.get("canonical_result_id")
                    or source.get("result_id")
                    or ""
                ).strip()
                if source_id:
                    response.source_id = source_id
                    citation.source_id = source_id
                response.url = str(source.get("url") or response.url)
                response.canonical_url = str(source.get("canonical_url") or response.canonical_url)
                response.domain = str(source.get("domain") or response.domain)
        payload = response.to_dict()
        if requested_result_token:
            payload["result_id"] = str(resolved_result_id or "")
            if isinstance(indexed_result, dict):
                payload["resolved_via_url"] = bool(resolved_url)
                fetch_id = str(indexed_result.get("fetch_id") or "").strip()
                if fetch_id:
                    payload["fetch_id"] = fetch_id
            if requested_result_token != resolved_result_id:
                payload["requested_result_id"] = requested_result_token
            if alias_metadata:
                payload["resolution"] = dict(alias_metadata)
        payload["citation"] = citation.to_dict()
        payload["excerpt_only"] = True
        payload["body_available_via_artifact"] = False
        artifact_id = None
        if hasattr(harness, "artifact_store"):
            body_total_lines = len(full_text.splitlines()) or (1 if full_text else 0)
            excerpt = str(response.text_excerpt or response.untrusted_text or "")
            preview_text = structured_plain_text(payload) or excerpt or full_text[:bounded_max_chars]
            summary = (
                summarize_structured_output(tool_name="web_fetch", output=payload)
                or response.title
                or response.domain
                or response.url
                or "web fetch"
            )
            artifact = harness.artifact_store.persist_generated_text(
                kind="web_fetch",
                source=response.canonical_url or response.url,
                content=full_text,
                summary=summary,
                preview_text=preview_text,
                metadata={
                    "source_id": response.source_id,
                    "url": response.url,
                    "canonical_url": response.canonical_url,
                    "domain": response.domain,
                    "title": response.title,
                    "byline": response.byline,
                    "content_type": response.content_type,
                    "content_sha256": response.content_sha256,
                    "provider": citation.provider,
                    "extractor": citation.extractor,
                    "published_at": response.published_at,
                    "fetched_at": response.fetched_at,
                    "body_char_count": len(full_text),
                    "body_total_lines": body_total_lines,
                    "excerpt_char_count": len(excerpt),
                    "render_mode": "body_with_preview",
                    "untrusted": True,
                },
                tool_name="web_fetch",
                session_id=str(getattr(state, "thread_id", "") or ""),
            )
            state.artifacts[artifact.artifact_id] = artifact
            artifact_id = artifact.artifact_id
            response.artifact_id = artifact_id
            payload["artifact_id"] = artifact_id
            payload["body_artifact_id"] = artifact_id
            payload["body_available_via_artifact"] = True
        return ok(
            payload,
            metadata={
                "artifact_id": artifact_id,
                "content_sha256": response.content_sha256,
                "body_char_count": len(full_text),
                "body_total_lines": len(full_text.splitlines()) or (1 if full_text else 0),
                "excerpt_char_count": len(str(response.text_excerpt or response.untrusted_text or "")),
                "render_mode": "body_with_preview",
                "untrusted": True,
            },
        )
    except SearchServerError as exc:
        return fail(str(exc), metadata=getattr(exc, "metadata", {}))
    except Exception as exc:
        return fail(str(exc))
