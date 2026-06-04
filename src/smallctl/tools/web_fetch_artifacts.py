from __future__ import annotations

from typing import Any

from ..tool_output_formatting import structured_plain_text, summarize_structured_output


def persist_fetch_artifact(
    harness: Any,
    state: Any,
    response: Any,
    full_text: str,
    citation: Any,
    resolved_result_id: str | None,
    payload: dict[str, Any],
) -> str | None:
    if not hasattr(harness, "artifact_store"):
        return None
    bounded_max_chars = payload.get("_bounded_max_chars", 12000)
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
            "result_id": str(resolved_result_id or ""),
            "fetch_id": str(payload.get("fetch_id") or ""),
        },
        tool_name="web_fetch",
        session_id=str(getattr(state, "thread_id", "") or ""),
    )
    state.artifacts[artifact.artifact_id] = artifact
    return artifact.artifact_id
