from types import SimpleNamespace

from smallctl.client.client_transport_helpers import request_first_token_timeout_sec
from smallctl.client.streaming import SSEStreamer


def test_first_token_timeout_is_not_capped_by_stream_read_timeout() -> None:
    streamer = SSEStreamer(
        provider_profile="generic",
        first_token_timeout_sec=300,
    )

    assert streamer._next_stream_read_timeout(
        chunk_count=0,
        tool_call_stream_active=False,
    ) == 300.0


def test_post_first_token_reads_keep_stream_read_timeout() -> None:
    streamer = SSEStreamer(
        provider_profile="generic",
        first_token_timeout_sec=300,
    )

    assert streamer._next_stream_read_timeout(
        chunk_count=1,
        tool_call_stream_active=False,
    ) == streamer.STREAM_READ_TIMEOUT_SEC


def test_write_heavy_adjusted_first_token_timeout_is_not_capped() -> None:
    client = SimpleNamespace(
        first_token_timeout_sec=300,
        provider_profile="generic",
        _WRITE_HEAVY_TOOL_NAMES={"file_write"},
        _WRITE_HEAVY_ARGUMENT_FIELDS=set(),
        _WRITE_HEAVY_ARGUMENT_TOOL_ALLOWLIST=set(),
        _READONLY_TOOL_NAME_DENYLIST=set(),
        WRITE_HEAVY_FIRST_TOKEN_TIMEOUT_MULTIPLIER=2.0,
    )
    tools = [{"function": {"name": "file_write", "parameters": {}}}]
    request_timeout = request_first_token_timeout_sec(client, tools)
    streamer = SSEStreamer(
        provider_profile="generic",
        first_token_timeout_sec=request_timeout,
    )

    assert request_timeout == 600.0
    assert streamer._next_stream_read_timeout(
        chunk_count=0,
        tool_call_stream_active=False,
    ) == 600.0
