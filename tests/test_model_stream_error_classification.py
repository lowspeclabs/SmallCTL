from __future__ import annotations

from smallctl.graph.model_stream_fallback_support import _classify_model_call_error


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int, text: str = "") -> None:
        super().__init__(f"http {status_code}")
        self.response = _FakeResponse(status_code, text)


def test_http_400_is_classified_as_provider_error() -> None:
    error_type, details = _classify_model_call_error(_FakeHTTPError(400, "bad request"))

    assert error_type == "provider"
    assert details["status_code"] == 400
    assert details["body"] == "bad request"
    assert "retryable" not in details


def test_http_429_is_classified_as_provider_error() -> None:
    error_type, details = _classify_model_call_error(_FakeHTTPError(429, "rate limited"))

    assert error_type == "provider"
    assert details["status_code"] == 429
    assert details["body"] == "rate limited"
    assert details["retryable"] is True


def test_non_http_error_remains_stream_error() -> None:
    error_type, details = _classify_model_call_error(RuntimeError("broken stream"))

    assert error_type == "stream"
    assert details == {}
