from __future__ import annotations

from smallctl.client import get_provider_adapter
from smallctl.client.provider_adapters import sanitize_messages_for_openrouter


def test_adapter_registry_returns_profile_specific_adapter() -> None:
    assert get_provider_adapter("lmstudio").name == "lmstudio"
    assert get_provider_adapter("openrouter").name == "openrouter"


def test_adapter_registry_falls_back_to_generic() -> None:
    adapter = get_provider_adapter("unknown-provider")
    assert adapter.name == "generic"
    assert adapter.stream_policy.supports_stream_options is True


def test_provider_policy_disables_stream_options_for_lmstudio_and_openrouter() -> None:
    assert get_provider_adapter("lmstudio").stream_policy.supports_stream_options is False
    assert get_provider_adapter("openrouter").stream_policy.supports_stream_options is False


def test_openrouter_sanitizer_rewrites_orphan_tool_messages() -> None:
    messages = [
        {"role": "tool", "name": "shell_exec", "tool_call_id": "call_1", "content": "ok"},
    ]
    sanitized = sanitize_messages_for_openrouter(messages)
    assert sanitized[0]["role"] == "user"
    assert "Orphan tool result" in str(sanitized[0]["content"])


class _FakeResponse:
    def __init__(self, status_code: int, text: str) -> None:
        self.status_code = status_code
        self.text = text


class _FakeExc:
    def __init__(self, status_code: int, text: str) -> None:
        self.response = _FakeResponse(status_code, text)


def test_retry_hint_for_stream_options_error() -> None:
    adapter = get_provider_adapter("generic")
    exc = _FakeExc(400, "unknown field: stream_options.include_usage")
    assert adapter.should_retry_without_stream_options(exc) is True


def test_no_retry_hint_for_unrelated_400_error() -> None:
    adapter = get_provider_adapter("generic")
    exc = _FakeExc(400, "invalid model name")
    assert adapter.should_retry_without_stream_options(exc) is False
