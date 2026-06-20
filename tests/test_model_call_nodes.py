from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from smallctl.graph.model_call_nodes import (
    _assistant_text_is_degenerate_loop,
    _strip_trailing_unclosed_markdown_fence,
    model_call,
)
from smallctl.graph.model_stream_fallback import StreamProcessingResult
from smallctl.graph.state import GraphRunState
from smallctl.models.events import UIEventType
from smallctl.state import LoopState


def test_assistant_text_is_degenerate_loop_empty() -> None:
    assert _assistant_text_is_degenerate_loop("", "dagu") is True
    assert _assistant_text_is_degenerate_loop("   ", "dagu") is True


def test_assistant_text_is_degenerate_loop_pure_repetition() -> None:
    text = "dagu " * 8
    assert _assistant_text_is_degenerate_loop(text, "dagu") is True


def test_assistant_text_is_degenerate_loop_substantive_with_repeated_word() -> None:
    # Substantive prose that legitimately repeats the loop phrase should not be discarded.
    text = (
        "The Dagu docker container has been successfully installed on the remote host "
        "(192.168.1.89) using the command `docker run -d --name dagu -p 8080:8080 "
        "-v ~/.dagu:/var/lib/dagu ghcr.io/dagucloud/dagu:latest dagu start-all`. "
        "A subsequent `docker ps -a` confirmed that the container is running."
    )
    assert _assistant_text_is_degenerate_loop(text, "dagu") is False


def test_assistant_text_is_degenerate_loop_substantive_table_with_repeated_word() -> None:
    # Structured analysis tables may repeat a term many times without being a loop.
    text = (
        "## Analysis\n\n"
        "| # | Issue | Detail |\n"
        "|---|---|---|\n"
        "| 1 | retries hardcoded | `retries=0` is set unconditionally |\n"
        "| 2 | retries env ignored | `VIKUNJA_RETRIES` is never read |\n"
        "| 3 | retries field unused | `Config.retries` exists but is not wired |\n"
        "| 4 | retries CLI orphan | `--retries` exists but default is ignored |\n"
        "| 5 | retries retry path | The retry loop uses a local retries variable |\n"
        "| 6 | retries fallback | Should fall back to env retries |\n"
        "| 7 | retries override | CLI should override env retries |\n"
        "| 8 | retries validation | Validate retries is non-negative |\n"
    )
    assert _assistant_text_is_degenerate_loop(text, "retries") is False


def test_assistant_text_is_degenerate_loop_real_prose_with_repeated_word() -> None:
    # Prose that happens to repeat a term several times is not a loop.
    text = (
        "The retry configuration is broken. retries retries retries retries "
        "retries retries We should fix the retries handling by reading the env var."
    )
    assert _assistant_text_is_degenerate_loop(text, "retries") is False


def test_assistant_text_is_degenerate_loop_short_phrase_skipped() -> None:
    # Very short repeated phrases are not reliable loop markers.
    assert _assistant_text_is_degenerate_loop("a a a a a a a", "a") is False


def test_strip_trailing_unclosed_markdown_fence_removes_partial_block() -> None:
    text = "Summary text.\n\n```json\n{\"name\": \"task_complete\","
    assert _strip_trailing_unclosed_markdown_fence(text) == "Summary text."


def test_strip_trailing_unclosed_markdown_fence_preserves_closed_block() -> None:
    text = "Summary text.\n\n```json\n{\"name\": \"task_complete\"}\n```"
    assert _strip_trailing_unclosed_markdown_fence(text) == text


def test_model_call_preserves_substantive_assistant_text_after_degenerate_loop() -> None:
    emitted: list[object] = []
    recorded_messages: list[tuple[str, list[dict]]] = []

    async def _emit(_handler: object, event: object) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="test-model"),
        registry=SimpleNamespace(names=lambda: {"task_complete"}),
        _emit=_emit,
        _apply_usage=lambda _usage: None,
        _runlog=lambda *args, **kwargs: None,
        _record_assistant_message=lambda assistant_text, tool_calls, **kwargs: recorded_messages.append(
            (assistant_text, tool_calls)
        ),
        _log_conversation_state=lambda *_args, **_kwargs: None,
    )
    deps = SimpleNamespace(harness=harness, event_handler=object())
    graph_state = GraphRunState(loop_state=harness.state, thread_id="session123", run_mode="execute")

    assistant_text = (
        "The Dagu docker container has been successfully installed on the remote host. "
        "A subsequent `docker ps -a` confirmed that the container is running.\n\n"
        '```json\n{"name": "task_complete", "arguments": {"message": "Successfully installed the dagu docker'
    )

    result = StreamProcessingResult(
        stream=SimpleNamespace(
            assistant_text=assistant_text,
            thinking_text="",
            tool_calls=[],
        ),
        timeline=[],
        usage={},
        duration=0.5,
        ttft=0.1,
        halted=True,
        halt_reason="model_output_degenerate_loop",
        halt_details={"repeated_phrase": "dagu", "repeat_count": 6, "buffer_chars": 406},
    )

    with patch("smallctl.graph.model_call_nodes.process_model_stream", AsyncMock(return_value=result)):
        asyncio.run(
            model_call(
                graph_state,
                deps,
                messages=[{"role": "user", "content": "install dagu"}],
                tools=[],
            )
        )

    # The inline tool JSON is stripped, and the substantive prose is preserved.
    expected_preserved = (
        "The Dagu docker container has been successfully installed on the remote host. "
        "A subsequent `docker ps -a` confirmed that the container is running."
    )
    assert graph_state.last_assistant_text == expected_preserved
    assert recorded_messages[-1][0] == expected_preserved

    replace_events = [
        event
        for event in emitted
        if getattr(event, "event_type", None) == UIEventType.ASSISTANT
        and getattr(event, "data", {}).get("kind") == "replace"
    ]
    assert replace_events
    assert replace_events[-1].content == expected_preserved


def test_model_call_uses_placeholder_for_pure_degenerate_loop() -> None:
    emitted: list[object] = []
    recorded_messages: list[tuple[str, list[dict]]] = []

    async def _emit(_handler: object, event: object) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="test-model"),
        registry=SimpleNamespace(names=lambda: {"task_complete"}),
        _emit=_emit,
        _apply_usage=lambda _usage: None,
        _runlog=lambda *args, **kwargs: None,
        _record_assistant_message=lambda assistant_text, tool_calls, **kwargs: recorded_messages.append(
            (assistant_text, tool_calls)
        ),
        _log_conversation_state=lambda *_args, **_kwargs: None,
    )
    deps = SimpleNamespace(harness=harness, event_handler=object())
    graph_state = GraphRunState(loop_state=harness.state, thread_id="session123", run_mode="execute")

    result = StreamProcessingResult(
        stream=SimpleNamespace(
            assistant_text="dagu dagu dagu dagu dagu dagu dagu",
            thinking_text="",
            tool_calls=[],
        ),
        timeline=[],
        usage={},
        duration=0.5,
        ttft=0.1,
        halted=True,
        halt_reason="model_output_degenerate_loop",
        halt_details={"repeated_phrase": "dagu", "repeat_count": 6, "buffer_chars": 100},
    )

    with patch("smallctl.graph.model_call_nodes.process_model_stream", AsyncMock(return_value=result)):
        asyncio.run(
            model_call(
                graph_state,
                deps,
                messages=[{"role": "user", "content": "install dagu"}],
                tools=[],
            )
        )

    placeholder = "[Previous assistant output was halted because it entered a repetition loop.]"
    assert graph_state.last_assistant_text == placeholder
    assert recorded_messages[-1][0] == placeholder
