from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

try:  # pragma: no cover - optional dependency path
    import trafilatura
except Exception:  # pragma: no cover
    trafilatura = None


@dataclass(slots=True)
class ExtractedPage:
    title: str = ""
    byline: str | None = None
    published_at: str | None = None
    headings: list[str] = field(default_factory=list)
    full_text: str = ""
    extractor: str = "trafilatura" if trafilatura is not None else "html_parser"


class _BasicTextExtractor(HTMLParser):
    def __init__(self, *, ignore_boilerplate: bool = True) -> None:
        super().__init__()
        self.ignore_boilerplate = ignore_boilerplate
        self.title = ""
        self.byline = ""
        self.headings: list[str] = []
        self.parts: list[str] = []
        self._capture_title = False
        self._capture_heading = False
        self._heading_parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if self.ignore_boilerplate and tag in {"script", "style", "nav", "header", "footer", "aside"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "title":
            self._capture_title = True
        if tag in {"h1", "h2", "h3"}:
            self._capture_heading = True
            self._heading_parts = []

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            if self.ignore_boilerplate and tag in {"script", "style", "nav", "header", "footer", "aside"}:
                self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._capture_title = False
        if tag in {"h1", "h2", "h3"}:
            self._capture_heading = False
            heading = html.unescape(" ".join(self._heading_parts)).strip()
            if heading:
                self.headings.append(heading)
        if tag in {"p", "br", "li", "div", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth:
            return
        text = html.unescape(data).strip()
        if not text:
            return
        if self._capture_title:
            self.title += f" {text}"
        elif self._capture_heading:
            self._heading_parts.append(text)
        else:
            self.parts.append(text)


def extract_page(
    html_text: str,
    *,
    max_chars: int = 20000,
    mode: str = "article",
    content_type: str = "text/html",
) -> ExtractedPage:
    normalized_mode = str(mode or "article").strip().lower() or "article"
    normalized_content_type = str(content_type or "text/html").strip().lower()
    if normalized_content_type == "text/plain":
        return _extract_plain_text_page(html_text, max_chars=max_chars)
    if normalized_mode == "text":
        return _extract_visible_text_page(html_text, max_chars=max_chars)
    if trafilatura is not None:
        try:
            extracted = trafilatura.extract(
                html_text,
                include_comments=False,
                include_tables=False,
                include_images=False,
                favor_precision=True,
            )
            metadata = trafilatura.metadata.extract_metadata(html_text) if hasattr(trafilatura, "metadata") else None
            title = str(getattr(metadata, "title", "") or "").strip()
            byline = str(getattr(metadata, "author", "") or "").strip() or None
            published = str(getattr(metadata, "date", "") or "").strip() or None
            if extracted:
                full_text = _normalize_text(str(extracted), max_chars=max_chars)
                headings = _find_headings(html_text)
                if not title:
                    title = headings[0] if headings else ""
                return ExtractedPage(
                    title=title,
                    byline=byline,
                    published_at=published,
                    headings=headings,
                    full_text=full_text,
                    extractor="trafilatura",
                )
        except Exception:
            pass

    return _extract_article_fallback(html_text, max_chars=max_chars)


def _extract_article_fallback(html_text: str, *, max_chars: int) -> ExtractedPage:
    parser = _BasicTextExtractor(ignore_boilerplate=True)
    parser.feed(html_text)
    full_text = _normalize_text("\n".join(parser.parts), max_chars=max_chars)
    title = parser.title.strip() or (parser.headings[0] if parser.headings else "")
    byline = parser.byline.strip() or None
    return ExtractedPage(
        title=title,
        byline=byline,
        published_at=None,
        headings=parser.headings[:8],
        full_text=full_text,
        extractor="html_parser",
    )


def _extract_visible_text_page(html_text: str, *, max_chars: int) -> ExtractedPage:
    parser = _BasicTextExtractor(ignore_boilerplate=False)
    parser.feed(html_text)
    visible_parts: list[str] = []
    title = parser.title.strip()
    if title:
        visible_parts.append(title)
    visible_parts.extend(parser.headings[:8])
    visible_parts.extend(parser.parts)
    full_text = _normalize_text("\n".join(visible_parts), max_chars=max_chars)
    return ExtractedPage(
        title=title or (parser.headings[0] if parser.headings else ""),
        byline=parser.byline.strip() or None,
        published_at=None,
        headings=parser.headings[:8],
        full_text=full_text,
        extractor="html_text",
    )


def _extract_plain_text_page(text: str, *, max_chars: int) -> ExtractedPage:
    full_text = _normalize_text(text, max_chars=max_chars)
    first_line = next((line.strip() for line in full_text.splitlines() if line.strip()), "")
    return ExtractedPage(
        title=first_line,
        byline=None,
        published_at=None,
        headings=[],
        full_text=full_text,
        extractor="plain_text",
    )


def _normalize_text(text: str, *, max_chars: int) -> str:
    normalized = re.sub(r"\n{3,}", "\n\n", str(text or "").strip())
    return normalized[:max_chars].rstrip()


def _find_headings(html_text: str) -> list[str]:
    headings = re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html_text, flags=re.IGNORECASE | re.DOTALL)
    results: list[str] = []
    for heading in headings[:8]:
        cleaned = re.sub(r"<[^>]+>", " ", heading)
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            results.append(cleaned)
    return results
