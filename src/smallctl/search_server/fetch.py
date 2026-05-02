from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin

try:  # pragma: no cover - optional dependency path
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from .citations import canonicalize_url, content_sha256, domain_from_url, make_source_id, now_iso
from .config import SearchServerConfig
from .extract import extract_page
from .models import CitationSource, WebFetchRequest, WebFetchResponse
from .security import validate_public_web_url, validate_redirect_target


@dataclass(slots=True)
class FetchedDocument:
    response: WebFetchResponse
    full_text: str
    citation: CitationSource
    extractor: str
    content_type: str
    status_code: int
    warnings: list[str]


async def fetch_document(
    request: WebFetchRequest,
    *,
    config: SearchServerConfig,
    url: str,
) -> FetchedDocument:
    if httpx is None:
        raise RuntimeError("Dependency missing: httpx")

    warnings: list[str] = []
    validated = validate_public_web_url(
        url,
        allowed_ports=config.allowed_ports,
        allow_private_targets=config.allow_private_network_targets,
    )
    current = validated.url
    response = None
    async with httpx.AsyncClient(
        timeout=float(config.timeout_seconds),
        follow_redirects=False,
        trust_env=False,
    ) as client:
        for redirect_count in range(config.max_redirects + 1):
            response = await client.get(
                current,
                headers={
                    "Accept": "text/html, text/plain;q=0.9, application/xhtml+xml;q=0.8",
                    "User-Agent": config.user_agent,
                },
            )
            if 300 <= response.status_code < 400 and response.headers.get("location"):
                if redirect_count >= config.max_redirects:
                    raise RuntimeError("Redirect limit exceeded.")
                target = urljoin(current, response.headers["location"])
                current = validate_redirect_target(
                    target,
                    allowed_ports=config.allowed_ports,
                    allow_private_targets=config.allow_private_network_targets,
                ).url
                continue
            break

    assert response is not None
    content_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} while fetching {current}")
    if content_type not in {"text/html", "text/plain", "application/xhtml+xml"}:
        raise RuntimeError(f"Unsupported content type: {content_type or 'unknown'}")

    body = response.text
    extracted = extract_page(
        body,
        max_chars=config.max_fetch_chars,
        mode=request.extract_mode,
        content_type=content_type,
    )
    full_text = extracted.full_text[: config.max_fetch_chars]
    excerpt = full_text[: max(1, min(request.max_chars, config.max_fetch_chars))]
    truncated = len(full_text) > len(excerpt)
    content_hash = content_sha256(full_text)
    canonical_url = canonicalize_url(str(response.url))
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_id = make_source_id(provider="fetch", canonical_url=canonical_url, fetched_at=fetched_at)
    response_payload = WebFetchResponse(
        source_id=source_id,
        url=str(response.url),
        canonical_url=canonical_url,
        domain=domain_from_url(str(response.url)),
        title=extracted.title or canonical_url,
        byline=extracted.byline,
        published_at=extracted.published_at,
        fetched_at=fetched_at,
        content_type=content_type,
        content_sha256=content_hash,
        text_excerpt=excerpt,
        char_start=1,
        char_end=len(excerpt),
        warnings=warnings + ([f"Excerpt truncated to {len(excerpt)} chars."] if truncated else []),
        untrusted_text=excerpt,
    )
    citation = CitationSource(
        source_id=source_id,
        url=str(response.url),
        canonical_url=canonical_url,
        domain=domain_from_url(str(response.url)),
        fetched_at=fetched_at,
        published_at=extracted.published_at,
        content_sha256=content_hash,
        extractor=extracted.extractor,
        provider="fetch",
        title=extracted.title or canonical_url,
    )
    return FetchedDocument(
        response=response_payload,
        full_text=full_text,
        citation=citation,
        extractor=extracted.extractor,
        content_type=content_type,
        status_code=response.status_code,
        warnings=response_payload.warnings,
    )
