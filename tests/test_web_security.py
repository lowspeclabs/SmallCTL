from __future__ import annotations

import socket

import pytest

from smallctl.search_server.config import SearchServerConfig
from smallctl.search_server.fetch import fetch_document
from smallctl.search_server.models import WebFetchRequest
from smallctl.search_server.security import ValidatedWebUrl, WebSecurityError, validate_public_web_url


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.0.1/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://localhost./",
        "http://2130706433/",
        "http://0x7f000001/",
        "http://0177.0.0.1/",
        "https://user:pass@example.com/",
        "ftp://example.com/",
    ],
)
def test_validate_public_web_url_blocks_unsafe_targets(url: str) -> None:
    with pytest.raises(WebSecurityError):
        validate_public_web_url(url)


def test_validate_public_web_url_rejects_disallowed_port() -> None:
    def fake_resolver(host: str, port: int, type=None):
        del host, port, type
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 444))]

    with pytest.raises(WebSecurityError, match="Port is not allowed"):
        validate_public_web_url(
            "https://example.com:444/",
            allowed_ports={80, 443},
            resolver=fake_resolver,
        )


def test_validate_public_web_url_rejects_invalid_port() -> None:
    with pytest.raises(WebSecurityError, match="Invalid port"):
        validate_public_web_url("https://example.com:99999/")


def test_validate_public_web_url_allows_explicit_private_ip_target() -> None:
    def fake_resolver(host: str, port: int, type=None):
        del host, port, type
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.63", 80))]

    validated = validate_public_web_url(
        "http://192.168.1.63/llm-explainer.html",
        allow_private_targets=("192.168.1.63",),
        resolver=fake_resolver,
    )

    assert validated.url == "http://192.168.1.63/llm-explainer.html"
    assert validated.resolved_addresses == ("192.168.1.63",)


def test_validate_public_web_url_allows_private_target_for_allowlisted_host() -> None:
    def fake_resolver(host: str, port: int, type=None):
        del port, type
        assert host == "docs.lab.example"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.10.25", 80))]

    validated = validate_public_web_url(
        "http://docs.lab.example/guide",
        allow_private_targets=("lab.example",),
        resolver=fake_resolver,
    )

    assert validated.host == "docs.lab.example"
    assert validated.resolved_addresses == ("192.168.10.25",)


def test_fetch_document_rejects_unsupported_content_type(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"content-type": "application/pdf"}
            self.text = "%PDF-1.7"
            self.url = "http://example.com/doc.pdf"

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self._response = FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None):
            del url, headers
            return self._response

    monkeypatch.setattr("smallctl.search_server.fetch.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(
        "smallctl.search_server.fetch.validate_public_web_url",
        lambda url, allowed_ports=None, allow_private_targets=None, resolver=None: ValidatedWebUrl(
            url=url,
            scheme="http",
            host="example.com",
            port=80,
            domain="example.com",
            resolved_addresses=("93.184.216.34",),
        ),
    )

    with pytest.raises(RuntimeError, match="Unsupported content type"):
        import asyncio

        asyncio.run(
            fetch_document(
                WebFetchRequest(url="http://example.com/doc.pdf", max_chars=120),
                config=SearchServerConfig(),
                url="http://example.com/doc.pdf",
            )
        )


def test_fetch_document_revalidates_redirect_targets_and_blocks_private_ip(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = 302
            self.headers = {"location": "http://127.0.0.1/"}
            self.text = ""
            self.url = "http://example.com/"

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self._response = FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None):
            del url, headers
            return self._response

    monkeypatch.setattr("smallctl.search_server.fetch.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(
        "smallctl.search_server.fetch.validate_public_web_url",
        lambda url, allowed_ports=None, allow_private_targets=None, resolver=None: ValidatedWebUrl(
            url=url,
            scheme="http",
            host="example.com",
            port=80,
            domain="example.com",
            resolved_addresses=("93.184.216.34",),
        ),
    )

    with pytest.raises(WebSecurityError):
        import asyncio

        asyncio.run(
            fetch_document(
                WebFetchRequest(url="http://example.com/", max_chars=120),
                config=SearchServerConfig(max_redirects=3),
                url="http://example.com/",
            )
        )


def test_fetch_document_extract_mode_changes_extracted_text(monkeypatch) -> None:
    html_body = """
    <html>
      <head><title>Example Title</title></head>
      <body>
        <nav>Navigation Links</nav>
        <article>
          <h1>Main Heading</h1>
          <p>Article body text.</p>
        </article>
      </body>
    </html>
    """

    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = 200
            self.headers = {"content-type": "text/html"}
            self.text = html_body
            self.url = "https://example.com/story"

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self._response = FakeResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None):
            del url, headers
            return self._response

    monkeypatch.setattr("smallctl.search_server.fetch.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("smallctl.search_server.extract.trafilatura", None)
    monkeypatch.setattr(
        "smallctl.search_server.fetch.validate_public_web_url",
        lambda url, allowed_ports=None, allow_private_targets=None, resolver=None: ValidatedWebUrl(
            url=url,
            scheme="https",
            host="example.com",
            port=443,
            domain="example.com",
            resolved_addresses=("93.184.216.34",),
        ),
    )

    import asyncio

    article = asyncio.run(
        fetch_document(
            WebFetchRequest(url="https://example.com/story", max_chars=200, extract_mode="article"),
            config=SearchServerConfig(),
            url="https://example.com/story",
        )
    )
    text = asyncio.run(
        fetch_document(
            WebFetchRequest(url="https://example.com/story", max_chars=200, extract_mode="text"),
            config=SearchServerConfig(),
            url="https://example.com/story",
        )
    )

    assert "Example Title" not in article.response.untrusted_text
    assert "Navigation Links" not in article.response.untrusted_text
    assert "Example Title" in text.response.untrusted_text
    assert "Main Heading" in text.response.untrusted_text
    assert "Navigation Links" in text.response.untrusted_text


def test_fetch_document_redirect_loop_fails_safely(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, url: str) -> None:
            self.status_code = 302
            self.headers = {"location": f"{url}?next=1"}
            self.text = ""
            self.url = url

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, headers=None):
            del headers
            return FakeResponse(url)

    monkeypatch.setattr("smallctl.search_server.fetch.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(
        "smallctl.search_server.fetch.validate_public_web_url",
        lambda url, allowed_ports=None, allow_private_targets=None, resolver=None: ValidatedWebUrl(
            url=url,
            scheme="https",
            host="example.com",
            port=443,
            domain="example.com",
            resolved_addresses=("93.184.216.34",),
        ),
    )
    monkeypatch.setattr(
        "smallctl.search_server.fetch.validate_redirect_target",
        lambda url, allowed_ports=None, allow_private_targets=None, resolver=None: ValidatedWebUrl(
            url=url,
            scheme="https",
            host="example.com",
            port=443,
            domain="example.com",
            resolved_addresses=("93.184.216.34",),
        ),
    )

    with pytest.raises(RuntimeError, match="Redirect limit exceeded"):
        import asyncio

        asyncio.run(
            fetch_document(
                WebFetchRequest(url="https://example.com/story", max_chars=120),
                config=SearchServerConfig(max_redirects=2),
                url="https://example.com/story",
            )
        )
