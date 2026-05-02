from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

try:  # pragma: no cover - optional dependency path
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from .citations import canonicalize_url, domain_from_url, make_result_id
from .config import SearchServerConfig
from .models import WebSearchRequest, WebSearchResult
from .provider_base import SearchProvider


@dataclass(slots=True)
class SearchProviderWarning:
    provider: str
    message: str


class BraveSearchProvider:
    name = "brave"

    def is_enabled(self, config: SearchServerConfig) -> bool:
        return bool(config.brave_api_key)

    def recency_support(self, config: SearchServerConfig) -> str:
        del config
        return "strict"

    async def search(self, request: WebSearchRequest, config: SearchServerConfig) -> list[WebSearchResult]:
        if httpx is None or not config.brave_api_key:
            return []
        params = {"q": request.query, "count": str(request.limit)}
        if request.recency_days:
            params["freshness"] = f"pd{request.recency_days}d"
        async with httpx.AsyncClient(timeout=float(config.timeout_seconds), trust_env=False) as client:
            response = await client.get(
                config.brave_api_endpoint,
                params=params,
                headers={"X-Subscription-Token": config.brave_api_key, "User-Agent": config.user_agent},
            )
        if response.status_code >= 400:
            return []
        payload = response.json()
        items = payload.get("web", {}).get("results", []) if isinstance(payload, dict) else []
        return _normalize_items(items, provider=self.name, request=request)


class SearxNGProvider:
    name = "searxng"

    def is_enabled(self, config: SearchServerConfig) -> bool:
        return bool(config.searxng_url)

    def recency_support(self, config: SearchServerConfig) -> str:
        return _normalize_recency_support(config.searxng_recency_support)

    async def search(self, request: WebSearchRequest, config: SearchServerConfig) -> list[WebSearchResult]:
        if httpx is None or not config.searxng_url:
            return []
        params = {"q": request.query, "format": "json"}
        async with httpx.AsyncClient(timeout=float(config.timeout_seconds), trust_env=False) as client:
            response = await client.get(config.searxng_url, params=params)
        if response.status_code >= 400:
            return []
        payload = response.json()
        items = payload.get("results", []) if isinstance(payload, dict) else []
        return _normalize_items(items, provider=self.name, request=request)


class DuckDuckGoProvider:
    name = "duckduckgo"

    def is_enabled(self, config: SearchServerConfig) -> bool:
        return True

    def recency_support(self, config: SearchServerConfig) -> str:
        del config
        return "none"

    async def search(self, request: WebSearchRequest, config: SearchServerConfig) -> list[WebSearchResult]:
        if httpx is None:
            return []
        params = {"q": request.query}
        async with httpx.AsyncClient(timeout=float(config.timeout_seconds), trust_env=False) as client:
            response = await client.get(config.duckduckgo_url, params=params, headers={"User-Agent": config.user_agent})
        if response.status_code >= 400:
            return []
        return _parse_duckduckgo_html(response.text, request=request)


def build_provider_chain(config: SearchServerConfig) -> list[SearchProvider]:
    registry: list[SearchProvider] = [BraveSearchProvider(), SearxNGProvider(), DuckDuckGoProvider()]
    order = [str(name).strip().lower() for name in config.providers if str(name).strip()]
    by_name = {provider.name: provider for provider in registry}
    selected: list[SearchProvider] = []
    for name in order:
        provider = by_name.get(name)
        if provider is not None:
            selected.append(provider)
    for provider in registry:
        if provider not in selected:
            selected.append(provider)
    return selected


def normalize_search_results(
    results: list[WebSearchResult],
    *,
    limit: int,
    domains: list[str] | None = None,
    sort: str = "relevance",
) -> list[WebSearchResult]:
    domain_filters = {domain.lower() for domain in (domains or []) if domain}
    deduped: dict[str, WebSearchResult] = {}
    ordered: list[WebSearchResult] = []
    for result in results:
        if domain_filters and not any(result.domain == domain or result.domain.endswith(f".{domain}") for domain in domain_filters):
            continue
        key = result.canonical_url or result.url
        if key in deduped:
            continue
        deduped[key] = result
        ordered.append(result)
    if sort == "recency":
        ordered.sort(key=lambda item: item.published_at or "", reverse=True)
    return ordered[:limit]


async def search_with_providers(
    request: WebSearchRequest,
    *,
    config: SearchServerConfig,
) -> tuple[list[WebSearchResult], list[str], str, str]:
    warnings: list[str] = []
    collected: list[WebSearchResult] = []
    selected_provider = ""
    selected_recency_support = "unknown"
    for provider in build_provider_chain(config):
        if not provider.is_enabled(config):
            warnings.append(f"{provider.name} is not configured.")
            continue
        try:
            items = await provider.search(request, config)
        except Exception as exc:
            warnings.append(f"{provider.name} failed: {exc}")
            continue
        if items:
            selected_provider = provider.name
            selected_recency_support = provider.recency_support(config)
            collected.extend(items)
            break
        warnings.append(f"{provider.name} returned no results.")
    normalized = normalize_search_results(collected, limit=request.limit, domains=request.domains, sort=request.sort)
    return normalized, warnings, selected_provider or "duckduckgo", selected_recency_support


def _normalize_recency_support(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"strict", "best_effort", "none"}:
        return normalized
    return "best_effort"


def _normalize_items(items: list[dict[str, Any]], *, provider: str, request: WebSearchRequest) -> list[WebSearchResult]:
    normalized: list[WebSearchResult] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        raw_url = str(item.get("url") or item.get("link") or "").strip()
        url = _normalize_result_url(raw_url)
        title = str(item.get("title") or item.get("name") or url).strip()
        snippet = str(item.get("description") or item.get("snippet") or item.get("body") or "").strip()
        published = str(item.get("published_at") or item.get("date") or item.get("published") or "").strip() or None
        canonical_url = canonicalize_url(url)
        normalized.append(
            WebSearchResult(
                result_id=make_result_id(provider=provider, canonical_url=canonical_url, rank=idx),
                title=title,
                url=url,
                canonical_url=canonical_url,
                domain=domain_from_url(url),
                snippet=snippet,
                published_at=published,
                provider=provider,
                rank=idx,
            )
        )
    return normalized


def _parse_duckduckgo_html(html_text: str, *, request: WebSearchRequest) -> list[WebSearchResult]:
    results: list[WebSearchResult] = []
    link_pattern = re.compile(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    snippet_pattern = re.compile(r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    snippets = snippet_pattern.findall(html_text)
    for idx, match in enumerate(link_pattern.findall(html_text), start=1):
        href, title_html = match
        title = re.sub(r"<[^>]+>", " ", title_html)
        title = re.sub(r"\s+", " ", title).strip()
        url = _normalize_result_url(href)
        snippet = re.sub(r"<[^>]+>", " ", snippets[idx - 1]) if idx - 1 < len(snippets) else ""
        snippet = re.sub(r"\s+", " ", snippet).strip()
        canonical_url = canonicalize_url(url)
        results.append(
            WebSearchResult(
                result_id=make_result_id(provider="duckduckgo", canonical_url=canonical_url, rank=idx),
                title=title or url,
                url=url,
                canonical_url=canonical_url,
                domain=domain_from_url(url),
                snippet=snippet,
                provider="duckduckgo",
                rank=idx,
            )
        )
        if len(results) >= request.limit:
            break
    return results


def _normalize_result_url(url: str) -> str:
    normalized = str(url or "").strip()
    if not normalized:
        return normalized
    if normalized.startswith("//"):
        normalized = f"https:{normalized}"

    parsed = urlsplit(normalized)
    if parsed.netloc.lower() == "duckduckgo.com" and parsed.path == "/l/":
        uddg_values = parse_qs(parsed.query).get("uddg", [])
        if uddg_values:
            redirected = unquote(str(uddg_values[0] or "").strip())
            if redirected:
                redirected = redirected if not redirected.startswith("//") else f"https:{redirected}"
                return redirected
    return normalized
