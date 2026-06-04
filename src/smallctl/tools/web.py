from __future__ import annotations

import logging
from typing import Any

from ..search_server.app import SearchServerError, get_search_runtime
from .web_fetch_utils import resolve_fetch_selector as _resolve_fetch_selector
from ..search_server.config import SearchServerConfig
from ..search_server.models import WebFetchRequest, WebSearchRequest
from .common import fail, ok
from .web_artifact_refs import (
    resolve_search_result_from_artifact_reference as _resolve_search_result_from_artifact_reference,
)
from .web_budget import (
    budget_remaining as _budget_remaining,
    ensure_budget as _ensure_budget,
    mark_web_fetch_budget_exhausted as _mark_web_fetch_budget_exhausted,
)
from .web_fetch_artifacts import persist_fetch_artifact as _persist_fetch_artifact
from .web_result_index import (
    assign_fetch_ids as _assign_fetch_ids,
    find_existing_fetch_artifact as _find_existing_fetch_artifact,
    load_fetch_artifact_index as _load_fetch_artifact_index,
    load_result_index as _load_result_index,
    record_fetch_artifact_mapping as _record_fetch_artifact_mapping,
    reject_invented_result_domain_url as _reject_invented_result_domain_url,
    unknown_result_id_error as _unknown_result_id_error,
    update_result_index as _update_result_index,
)

_MAX_TOOL_LIMIT = 10
_MAX_TOOL_FETCH_CHARS = 20000
log = logging.getLogger("smallctl.tools.web")
def _config_for_harness(harness: Any) -> SearchServerConfig:
    return SearchServerConfig.from_harness(harness)


def _normalize_domains(domains: list[str] | None) -> list[str] | None:
    cleaned = [str(item).strip().lower() for item in (domains or []) if str(item).strip()]
    return cleaned or None


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
        payload["budget_remaining"] = _budget_remaining(state, config)
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
    requested_url, requested_result_id, requested_fetch_id, toleration_warnings = _resolve_fetch_selector(
        state, url=url, result_id=result_id, fetch_id=fetch_id
    )
    if requested_url is None and requested_result_id is None and requested_fetch_id is None:
        return fail("Provide exactly one of url, result_id, or fetch_id.")
    requested_result_token = requested_result_id or requested_fetch_id
    try:
        bounded_max_chars = min(max(1, int(max_chars or config.default_fetch_chars)), min(config.max_fetch_chars, _MAX_TOOL_FETCH_CHARS))
        invented_url_error = _reject_invented_result_domain_url(state, url=requested_url, result_id=requested_result_token)
        if invented_url_error:
            return fail(invented_url_error, metadata={"reason": "web_fetch_url_not_from_results"})
        if requested_result_token:
            existing_artifact_id = _load_fetch_artifact_index(state).get(requested_result_token)
            if not existing_artifact_id:
                existing_artifact_id = _find_existing_fetch_artifact(state, requested_result_token)
            if existing_artifact_id:
                return fail(
                    f"Result '{requested_result_token}' was already fetched in this session. "
                    f"The full content is available in artifact {existing_artifact_id}. "
                    f"Use artifact_read(artifact_id='{existing_artifact_id}') instead of repeating this fetch.",
                    metadata={
                        "reason": "web_fetch_duplicate_result_id",
                        "requested_result_id": requested_result_token,
                        "existing_artifact_id": existing_artifact_id,
                    },
                )
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
        if toleration_warnings:
            response.warnings.extend(toleration_warnings)
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
        payload["_bounded_max_chars"] = bounded_max_chars
        if requested_result_token:
            payload["result_id"] = str(resolved_result_id or "")
            if isinstance(indexed_result, dict):
                payload["resolved_via_url"] = bool(resolved_url)
                fid = str(indexed_result.get("fetch_id") or "").strip()
                if fid:
                    payload["fetch_id"] = fid
            if requested_result_token != resolved_result_id:
                payload["requested_result_id"] = requested_result_token
            if alias_metadata:
                payload["resolution"] = dict(alias_metadata)
        payload["citation"] = citation.to_dict()
        payload["excerpt_only"] = True
        payload["body_available_via_artifact"] = False
        artifact_id = _persist_fetch_artifact(harness, state, response, full_text, citation, resolved_result_id, payload)
        if artifact_id is not None:
            payload.pop("_bounded_max_chars", None)
            response.artifact_id = artifact_id
            payload["artifact_id"] = artifact_id
            payload["body_artifact_id"] = artifact_id
            payload["body_available_via_artifact"] = True
            if requested_result_token:
                _record_fetch_artifact_mapping(
                    state,
                    result_id=str(resolved_result_id or ""),
                    fetch_id=str(payload.get("fetch_id") or ""),
                    artifact_id=artifact_id,
                )
        payload["budget_remaining"] = _budget_remaining(state, config)
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
        _mark_web_fetch_budget_exhausted(state, str(exc))
        return fail(str(exc), metadata=getattr(exc, "metadata", {}))
    except Exception as exc:
        log.warning("web_fetch unexpected error: %s", exc, exc_info=True)
        return fail(str(exc), metadata={"error_type": type(exc).__name__})
