from __future__ import annotations

from smallctl.search_server.extract import (
    _extract_article_fallback,
    _extract_visible_text_page,
)


def test_visible_text_mode_always_skips_invisible_tags() -> None:
    html = """
    <html>
      <head><title>Page Title</title></head>
      <body>
        <script>var x = 1;</script>
        <style>.red { color: red; }</style>
        <noscript>Enable JS</noscript>
        <iframe>frame content</iframe>
        <p>Visible paragraph.</p>
      </body>
    </html>
    """
    extracted = _extract_visible_text_page(html, max_chars=10000)
    text = extracted.full_text
    assert "var x = 1" not in text
    assert ".red" not in text
    assert "Enable JS" not in text
    assert "frame content" not in text
    assert "Visible paragraph." in text
    assert extracted.title.strip() == "Page Title"


def test_visible_text_mode_includes_layout_tags_when_no_boilerplate() -> None:
    html = """
    <html>
      <body>
        <nav>Navigation link</nav>
        <header>Header text</header>
        <aside>Sidebar</aside>
        <footer>Footer text</footer>
        <article>Article content</article>
      </body>
    </html>
    """
    extracted = _extract_visible_text_page(html, max_chars=10000)
    text = extracted.full_text
    assert "Navigation link" in text
    assert "Header text" in text
    assert "Sidebar" in text
    assert "Footer text" in text
    assert "Article content" in text


def test_article_fallback_skips_invisible_and_layout_tags() -> None:
    html = """
    <html>
      <head><title>Article Title</title></head>
      <body>
        <script>var y = 2;</script>
        <style>.blue { color: blue; }</style>
        <noscript>No JS</noscript>
        <iframe>inner frame</iframe>
        <nav>Nav item</nav>
        <header>Site header</header>
        <aside>Ad block</aside>
        <footer>Copyright 2024</footer>
        <article>
          <h1>Main Heading</h1>
          <p>Important content.</p>
        </article>
      </body>
    </html>
    """
    extracted = _extract_article_fallback(html, max_chars=10000)
    text = extracted.full_text
    assert "var y = 2" not in text
    assert ".blue" not in text
    assert "No JS" not in text
    assert "inner frame" not in text
    assert "Nav item" not in text
    assert "Site header" not in text
    assert "Ad block" not in text
    assert "Copyright 2024" not in text
    assert "Main Heading" in extracted.headings
    assert "Important content." in text


def test_invisible_tags_with_nested_content_are_fully_skipped() -> None:
    html = """
    <div>
      <script>
        function foo() {
          return "bar";
        }
        <div>nested inside script</div>
      </script>
      <p>Real paragraph.</p>
    </div>
    """
    extracted = _extract_visible_text_page(html, max_chars=10000)
    text = extracted.full_text
    assert "function foo" not in text
    assert "nested inside script" not in text
    assert "Real paragraph." in text
