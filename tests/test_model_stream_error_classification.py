from __future__ import annotations

from smallctl.graph.model_stream_resolution import (
    _chunk_error_failure_message,
    _chunk_error_failure_type,
)
from smallctl.graph.model_stream_fallback_support import _classify_model_call_error
from smallctl.graph.model_stream_loop import _parse_context_window_overflow


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


def test_context_budget_chunk_error_reports_prompt_budget_failure() -> None:
    details = {
        "type": "context_budget_exceeded",
        "provider_profile": "llamacpp",
        "over_budget_tokens": 58,
    }

    assert (
        _chunk_error_failure_message(details)
        == "llamacpp prompt exceeded the local context budget before request by 58 estimated tokens"
    )
    assert _chunk_error_failure_type(details) == "prompt_budget"


def test_context_overflow_details_parse_as_window_overflow() -> None:
    from types import SimpleNamespace

    from smallctl.client.transport_error_classification import (
        _llamacpp_context_overflow_chunk_error_details,
    )

    details = _llamacpp_context_overflow_chunk_error_details(
        SimpleNamespace(provider_profile="llamacpp"),
        payload=None,
        body_summary={"request_tokens": 16385, "context_limit": 16384},
        status_code=400,
        attempt=1,
    )

    assert _parse_context_window_overflow("llamacpp context window exceeded", details) == (16385, 16384)
