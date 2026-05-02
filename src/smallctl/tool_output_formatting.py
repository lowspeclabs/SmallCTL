from __future__ import annotations

from typing import Any


def summarize_structured_output(*, tool_name: str, output: dict[str, Any]) -> str | None:
    web_search_summary = _summarize_web_search_output(output)
    if web_search_summary:
        return web_search_summary
    web_fetch_summary = _summarize_web_fetch_output(output)
    if web_fetch_summary:
        return web_fetch_summary
    return None


def structured_plain_text(output: dict[str, Any]) -> str | None:
    web_search_text = _render_web_search_output(output)
    if web_search_text:
        return web_search_text
    web_fetch_text = _render_web_fetch_output(output)
    if web_fetch_text:
        return web_fetch_text

    keys = set(output.keys())
    if keys <= {"status", "message"}:
        value = output.get("message")
    elif keys <= {"status", "question"}:
        value = output.get("question")
    else:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _summarize_web_search_output(output: dict[str, Any]) -> str | None:
    results = output.get("results")
    if not isinstance(results, list):
        return None

    query = _clean_text(output.get("query"))
    count = len(results)
    if count <= 0:
        if query:
            return f'0 results for "{_truncate(query, 80)}"'
        return "0 web results"

    result_fragments: list[str] = []
    for item in results[:2]:
        if not isinstance(item, dict):
            continue
        title = _clean_text(item.get("title"))
        domain = _clean_text(item.get("domain"))
        url = _clean_text(item.get("url"))
        label = title or domain or url
        if not label:
            continue
        if domain and title and domain.lower() not in title.lower():
            label = f"{title} ({domain})"
        result_fragments.append(label)

    head = f'{count} result{"s" if count != 1 else ""}'
    if query:
        head = f'{head} for "{_truncate(query, 60)}"'
    if result_fragments:
        return f"{head}: {'; '.join(result_fragments)}"
    return head


def _summarize_web_fetch_output(output: dict[str, Any]) -> str | None:
    if not _looks_like_web_fetch_output(output):
        return None

    title = _clean_text(output.get("title"))
    domain = _clean_text(output.get("domain"))
    excerpt = _clean_text(output.get("text_excerpt")) or _clean_text(output.get("untrusted_text"))
    label = title or domain or _clean_text(output.get("url"))
    if not label:
        return "web fetch"
    if domain and title and domain.lower() not in title.lower():
        label = f"{title} ({domain})"
    if excerpt:
        return f"{label}: {_truncate(excerpt, 120)}"
    return label


def _render_web_search_output(output: dict[str, Any]) -> str | None:
    results = output.get("results")
    if not isinstance(results, list):
        return None

    lines: list[str] = []
    query = _clean_text(output.get("query"))
    provider = _clean_text(output.get("provider"))
    recency_support = _clean_text(output.get("recency_support"))
    recency_enforced = output.get("recency_enforced")
    warnings = output.get("warnings")

    if query:
        lines.append(f"Query: {query}")
    if provider:
        lines.append(f"Provider: {provider}")
    if recency_support:
        lines.append(f"Recency support: {recency_support}")
    if isinstance(recency_enforced, bool):
        lines.append(f"Recency enforced: {'yes' if recency_enforced else 'no'}")

    if lines:
        lines.append("")

    if not results:
        lines.append("No results.")
    else:
        for index, item in enumerate(results[:5], start=1):
            if not isinstance(item, dict):
                continue
            title = _clean_text(item.get("title")) or _clean_text(item.get("url")) or f"Result {index}"
            fetch_id = _clean_text(item.get("fetch_id")) or _clean_text(item.get("result_id"))
            result_id = _clean_text(item.get("result_id"))
            url = _clean_text(item.get("url"))
            domain = _clean_text(item.get("domain"))
            snippet = _clean_text(item.get("snippet"))

            lines.append(f"{index}. {title}")
            if fetch_id:
                lines.append(f"   Fetch ID: {fetch_id}")
            if result_id:
                lines.append(f"   Result ID: {result_id}")
            if fetch_id:
                lines.append(f"   Use with: web_fetch(result_id='{fetch_id}')")
            if url:
                lines.append(f"   URL: {url}")
            if domain:
                lines.append(f"   Domain: {domain}")
            if snippet:
                lines.append(f"   Snippet: {_truncate(snippet, 280)}")
            if index < min(len(results), 5):
                lines.append("")

    warning_lines = _normalize_string_list(warnings)
    if warning_lines:
        lines.append("")
        lines.append("Warnings:")
        for warning in warning_lines[:3]:
            lines.append(f"- {warning}")

    rendered = "\n".join(lines).strip()
    return rendered or None


def _render_web_fetch_output(output: dict[str, Any]) -> str | None:
    if not _looks_like_web_fetch_output(output):
        return None

    lines: list[str] = []
    title = _clean_text(output.get("title"))
    url = _clean_text(output.get("url"))
    domain = _clean_text(output.get("domain"))
    published_at = _clean_text(output.get("published_at"))
    excerpt = _clean_text(output.get("text_excerpt")) or _clean_text(output.get("untrusted_text"))
    fetch_id = _clean_text(output.get("fetch_id")) or _clean_text(output.get("requested_result_id")) or _clean_text(output.get("result_id"))
    result_id = _clean_text(output.get("result_id"))
    requested_token = _clean_text(output.get("requested_result_id"))
    artifact_id = _clean_text(output.get("body_artifact_id")) or _clean_text(output.get("artifact_id"))
    warnings = _normalize_string_list(output.get("warnings"))

    if title:
        lines.append(f"Title: {title}")
    if fetch_id:
        lines.append(f"Fetch ID: {fetch_id}")
    if result_id and result_id != fetch_id:
        lines.append(f"Result ID: {result_id}")
    if requested_token and requested_token not in {fetch_id, result_id}:
        lines.append(f"Requested Token: {requested_token}")
    if url:
        lines.append(f"URL: {url}")
    if domain:
        lines.append(f"Domain: {domain}")
    if published_at:
        lines.append(f"Published: {published_at}")
    if artifact_id:
        lines.append(f"Full Body Artifact: {artifact_id}")
    if excerpt:
        if lines:
            lines.append("")
        lines.append("Excerpt:")
        lines.append(_truncate(excerpt, 1200))
        lines.append("")
        if artifact_id:
            lines.append(
                f"Note: inline web text is excerpt-only; use `artifact_read(artifact_id='{artifact_id}')` for the full fetched body."
            )
        else:
            lines.append("Note: inline web text is excerpt-only; use the artifact for the full fetched body.")

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for warning in warnings[:3]:
            lines.append(f"- {warning}")

    rendered = "\n".join(lines).strip()
    return rendered or None


def _looks_like_web_fetch_output(output: dict[str, Any]) -> bool:
    if "text_excerpt" in output or "untrusted_text" in output:
        return "url" in output or "source_id" in output or "canonical_url" in output
    return False


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text:
            result.append(text)
    return result


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(str(text or "").split()).strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"
