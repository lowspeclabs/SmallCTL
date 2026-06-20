from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import smallctl.graph.model_stream as model_stream_module
import smallctl.graph.model_stream_loop as model_stream_loop_module
from smallctl.client import OpenAICompatClient, StreamResult
from smallctl.graph.model_stream_loop import (
    run_model_stream_loop,
    _trim_degenerate_suffix,
    _detect_degenerate_repetition,
    _DEGENERATE_REPETITION_WINDOW_CHARS,
)
from smallctl.graph.model_stream import process_model_stream
from smallctl.graph.model_stream_resolution import resolve_model_stream_result
from smallctl.graph.state import GraphRunState
from smallctl.state import LoopState
from smallctl.write_session_fsm import new_write_session


class _Harness:
    def __init__(self, state: LoopState) -> None:
        self.state = state
        self.reasoning_mode = "off"
        self.thinking_start_tag = "<think>"
        self.thinking_end_tag = "</think>"
        self.thinking_visibility = False
        self.runlog_events = []

    def _runlog(self, *args, **kwargs) -> None:
        self.runlog_events.append((args, kwargs))

    async def _emit(self, *args, **kwargs) -> None:
        del args, kwargs

    def _failure(self, message: str, *, error_type: str = "stream", details=None):
        del details
        return {
            "status": "failed",
            "message": message,
            "error": {
                "message": message,
                "type": error_type,
            },
        }


class _FallbackClient:
    model = "qwen3.5:4b"

    def __init__(self) -> None:
        self.calls = []

    async def stream_chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        yield {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "```python\nprint('recovered')\n```"
                        }
                    }
                ]
            },
        }


class _WeakFallbackClient:
    model = "qwen3.5:9b"

    def __init__(self) -> None:
        self.calls = []

    async def stream_chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        yield {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "```python\n#"
                        }
                    }
                ]
            },
        }


class _WeakHtmlFallbackClient:
    model = "qwen3.5:9b"

    def __init__(self) -> None:
        self.calls = []

    async def stream_chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        yield {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "```html\n<!DOCTYPE"
                        }
                    }
                ]
            },
        }


class _RemoteFallbackClient:
    model = "qwen3.5:9b"

    def __init__(self) -> None:
        self.calls = []

    async def stream_chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        yield {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "```html\n<html><body><h1>Recovered</h1></body></html>\n```"
                        }
                    }
                ]
            },
        }


class _ReasoningOnlyClient:
    model = "qwen3.5:9b"

    def __init__(self) -> None:
        self.calls = []

    async def stream_chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        if len(self.calls) > 1:
            yield {
                "type": "chunk",
                "data": {
                    "choices": [
                        {
                            "delta": {
                                "content": "Done."
                            }
                        }
                    ]
                },
            }
            return
        for _ in range(3):
            yield {
                "type": "chunk",
                "data": {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "still thinking"
                            }
                        }
                    ]
                },
            }


class _AlwaysReasoningOnlyClient:
    model = "qwen3.5:9b"

    def __init__(self) -> None:
        self.calls = []

    async def stream_chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        for _ in range(3):
            yield {
                "type": "chunk",
                "data": {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": "still thinking"
                            }
                        }
                    ]
                },
            }


class _ProgressingReasoningClient:
    model = "qwen3.5:9b"

    def __init__(self) -> None:
        self.calls = []

    async def stream_chat(self, messages, tools):
        self.calls.append({"messages": messages, "tools": tools})
        for text in [
            "I read the file and found a concrete paddle collision bounds issue. ",
            "The paddle rows are drawn with range(paddle_h), so the end row is exclusive. ",
            "Using a less-than-or-equal comparison includes one extra row below each paddle. ",
            "That means collisions can register when the ball is visually outside the paddle. ",
            "I can now answer with the exact bug and the suggested patch. ",
        ]:
            yield {
                "type": "chunk",
                "data": {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": text,
                            }
                        }
                    ]
                },
            }
        yield {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "The paddle collision check should use an exclusive lower bound."
                        }
                    }
                ]
            },
        }

class _LfmProgressingReasoningClient(_ProgressingReasoningClient):
    model = "lfm2.5-8b-a1b"



def _incident_native_tool_call_chunks(*, wrapped: bool = True) -> list[dict[str, object]]:
    raw_chunks: list[dict[str, object]] = [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "reasoning_content": "Gathering low-level host facts.\n",
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_diskstats",
                                "type": "function",
                                "function": {
                                    "name": "ssh_exec",
                                    "arguments": (
                                        '{"host":"192.168.1.89","command":"cat /proc/diskstats | head -20"}'
                                    ),
                                },
                            },
                            {
                                "index": 1,
                                "id": "call_free",
                                "type": "function",
                                "function": {
                                    "name": "ssh_exec",
                                    "arguments": '{"host":"192.168.1.89","command":"free -m"}',
                                },
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 2,
                                "id": "call_dmesg",
                                "type": "function",
                                "function": {
                                    "name": "ssh_exec",
                                    "arguments": (
                                        '{"host":"192.168.1.89","command":"dmesg 2>/dev/null | tail -30 | '
                                        'grep -iE \\"error|fail|overrun|timeout|I/O\\" || echo \\"dmesg not '
                                        'available or no errors\\""}'
                                    ),
                                },
                            },
                            {
                                "index": 3,
                                "id": "call_ss",
                                "type": "function",
                                "function": {
                                    "name": "ssh_exec",
                                    "arguments": (
                                        '{"host":"192.168.1.89","command":"ss -tuln 2>/dev/null || '
                                        'netstat -tuln 2>/dev/null || echo \\"network tools not available\\""}'
                                    ),
                                },
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {},
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    if not wrapped:
        return raw_chunks
    return [{"type": "chunk", "data": chunk} for chunk in raw_chunks]


def _build_state_with_write_session() -> LoopState:
    state = LoopState(cwd="/tmp")
    session = new_write_session(
        session_id="ws_auto_resume",
        target_path="./target.py",
        intent="replace_file",
    )
    session.write_current_section = "body"
    session.write_sections_completed = ["imports"]
    state.write_session = session
    return state


def test_chunk_reasoning_text_falls_back_to_reasoning_when_content_field_is_not_text() -> None:
    event = {
        "type": "chunk",
        "data": {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": {"unexpected": "shape"},
                        "reasoning": "fallback reasoning",
                    }
                }
            ]
        },
    }

    assert model_stream_loop_module._chunk_reasoning_text(event) == "fallback reasoning"


def test_reasoning_progress_rejects_numbered_boilerplate() -> None:
    fragments = [
        (
            "Still analyzing possible collision bounds parser fallback retry guard threshold "
            f"without action attempt {index}"
        )
        for index in range(6)
    ]

    assessment = model_stream_loop_module._assess_reasoning_fragments_progress(fragments)

    assert assessment.progress is False
    assert assessment.fragment_count == 6
    assert assessment.unique_ratio == 1.0
    assert assessment.distinct_word_count >= 12
    assert assessment.novel_word_count == 0


def test_reasoning_progress_accepts_developing_analysis() -> None:
    fragments = [
        "I read the file and found a concrete paddle collision bounds issue. ",
        "The paddle rows are drawn with range(paddle_h), so the end row is exclusive. ",
        "Using a less-than-or-equal comparison includes one extra row below each paddle. ",
        "That means collisions can register when the ball is visually outside the paddle. ",
        "I can now answer with the exact bug and the suggested patch. ",
    ]

    assessment = model_stream_loop_module._assess_reasoning_fragments_progress(fragments)

    assert assessment.progress is True
    assert assessment.fragment_count == 5
    assert assessment.novel_word_count >= 4


def test_reasoning_only_stream_retries_with_action_nudge(monkeypatch) -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    harness.client = _ReasoningOnlyClient()
    harness._cancel_requested = False
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    deps = SimpleNamespace(event_handler=None, harness=harness)
    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]

    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_MAX_CHUNKS", 2)
    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_MAX_SECONDS", 9999.0)

    result = asyncio.run(
        run_model_stream_loop(
            graph_state,
            deps,
            harness=harness,
            messages=[{"role": "user", "content": "what is the weather?"}],
            tools=tools,
            echo_to_stdout=False,
            start_tag="<think>",
            end_tag="</think>",
            start_time=time.perf_counter(),
        )
    )

    assert graph_state.final_result is None
    assert len(harness.client.calls) == 2
    assert "too long in reasoning" in harness.client.calls[1]["messages"][-1]["content"]
    assert result["stream_completed_cleanly"] is True


def test_reasoning_only_stream_halts_after_escalation_nudge_exhausted(monkeypatch) -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    harness.client = _AlwaysReasoningOnlyClient()
    harness._cancel_requested = False
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    deps = SimpleNamespace(event_handler=None, harness=harness)
    tools = [
        {"type": "function", "function": {"name": "file_patch", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "escalate_to_bigger_model", "parameters": {"type": "object"}}},
    ]

    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_MAX_CHUNKS", 2)
    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_MAX_SECONDS", 9999.0)

    result = asyncio.run(
        run_model_stream_loop(
            graph_state,
            deps,
            harness=harness,
            messages=[{"role": "user", "content": "fix pong.py"}],
            tools=tools,
            echo_to_stdout=False,
            start_tag="<think>",
            end_tag="</think>",
            start_time=time.perf_counter(),
        )
    )

    assert len(harness.client.calls) == 2
    assert "call escalate_to_bigger_model" in harness.client.calls[1]["messages"][-1]["content"]
    assert result["stream_completed_cleanly"] is False
    assert result["stream_ended_without_done"] is True
    assert result["stream_ended_without_done_details"]["reason"] == "reasoning_only_stream_stall"
    assert result["stream_ended_without_done_details"]["retrying"] is False


def test_progressing_reasoning_only_stream_gets_hard_budget_before_retry(monkeypatch) -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    harness.client = _ProgressingReasoningClient()
    harness._cancel_requested = False
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    deps = SimpleNamespace(event_handler=None, harness=harness)
    tools = [{"type": "function", "function": {"name": "file_patch", "parameters": {"type": "object"}}}]

    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_TOOL_MAX_CHUNKS", 4)
    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_MAX_CHUNKS", 8)
    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_TOOL_MAX_SECONDS", 9999.0)
    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_MAX_SECONDS", 9999.0)

    result = asyncio.run(
        run_model_stream_loop(
            graph_state,
            deps,
            harness=harness,
            messages=[{"role": "user", "content": "fix pong.py"}],
            tools=tools,
            echo_to_stdout=False,
            start_tag="<think>",
            end_tag="</think>",
            start_time=time.perf_counter(),
        )
    )

    assert len(harness.client.calls) == 1
    assert result["stream_completed_cleanly"] is True
    assert any(event[0][0] == "reasoning_only_stream_progress_defer" for event in harness.runlog_events)

def test_lfm_progressing_reasoning_only_stream_retries_without_hard_budget(monkeypatch) -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    harness.client = _LfmProgressingReasoningClient()
    harness._cancel_requested = False
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    deps = SimpleNamespace(event_handler=None, harness=harness)
    tools = [{"type": "function", "function": {"name": "ssh_exec", "parameters": {"type": "object"}}}]

    monkeypatch.setattr(model_stream_loop_module, "_LFM25_REASONING_ONLY_TOOL_MAX_CHUNKS", 4)
    monkeypatch.setattr(model_stream_loop_module, "_LFM25_REASONING_ONLY_TOOL_MAX_SECONDS", 9999.0)
    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_MAX_CHUNKS", 8)
    monkeypatch.setattr(model_stream_loop_module, "_REASONING_ONLY_MAX_SECONDS", 9999.0)

    result = asyncio.run(
        run_model_stream_loop(
            graph_state,
            deps,
            harness=harness,
            messages=[{"role": "user", "content": "install pihole"}],
            tools=tools,
            echo_to_stdout=False,
            start_tag="<think>",
            end_tag="</think>",
            start_time=time.perf_counter(),
        )
    )

    assert len(harness.client.calls) == 2
    assert any(event[0][0] == "reasoning_only_stream_retry" for event in harness.runlog_events)
    assert not any(event[0][0] == "reasoning_only_stream_progress_defer" for event in harness.runlog_events)
    assert result["stream_completed_cleanly"] is False
    assert result["stream_ended_without_done_details"]["reason"] == "reasoning_only_stream_stall"


def test_stream_chunk_error_schedules_one_auto_resume_for_recoverable_write_session() -> None:
    state = _build_state_with_write_session()
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=[],
            salvage_partial_stream=None,
            last_chunk_error_details={"reason": "tool_call_continuation_timeout"},
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.halted is True
    assert result.halt_reason == "stream_chunk_error_auto_resume"
    assert graph_state.final_result is None
    assert state.scratchpad.get("_stream_chunk_error_auto_resume_signature")
    assert any(
        message.metadata.get("recovery_kind") == "stream_chunk_error_auto_resume"
        for message in state.recent_messages
    )


def test_stream_chunk_error_auto_resume_only_runs_once_per_write_session_signature() -> None:
    state = _build_state_with_write_session()
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")

    async def _run_once():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=[],
            salvage_partial_stream=None,
            last_chunk_error_details={"reason": "tool_call_continuation_timeout"},
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    first = asyncio.run(_run_once())
    second = asyncio.run(_run_once())

    assert first is not None and first.halted is True
    assert second is not None
    assert graph_state.final_result is not None
    assert graph_state.final_result["message"] == "Upstream chunk error after retries"
    assert graph_state.error is not None
    assert graph_state.error["type"] == "stream"


def test_provider_400_chunk_error_finalizes_with_provider_message() -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=[],
            salvage_partial_stream=None,
            last_chunk_error_details={
                "status_code": 400,
                "provider_profile": "openrouter",
                "upstream_provider": "Together",
                "provider_error": "Input validation error",
            },
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert graph_state.final_result is not None
    assert graph_state.final_result["message"] == (
        "openrouter/Together input validation failed after retries (HTTP 400: Input validation error)"
    )
    assert graph_state.error is not None
    assert graph_state.error["type"] == "provider"


def test_process_model_stream_preserves_preset_provider_failure(monkeypatch) -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    provider_failure = harness._failure("LM Studio rejected the request", error_type="provider")
    resolve_called = False

    async def _fake_run_model_stream_loop(*args, **kwargs):
        del args, kwargs
        graph_state.final_result = provider_failure
        graph_state.error = provider_failure["error"]
        return {
            "chunks": [],
            "salvage_partial_stream": None,
            "last_chunk_error_details": None,
            "stream_ended_without_done": False,
            "stream_ended_without_done_details": {},
            "trigger_early_4b_fallback": False,
            "stream_completed_cleanly": False,
            "first_token_time": None,
        }

    async def _fake_resolve_model_stream_result(*args, **kwargs):
        del args, kwargs
        nonlocal resolve_called
        resolve_called = True
        raise AssertionError("resolve_model_stream_result should not run after final_result is preset")

    monkeypatch.setattr(model_stream_module, "run_model_stream_loop", _fake_run_model_stream_loop)
    monkeypatch.setattr(model_stream_module, "resolve_model_stream_result", _fake_resolve_model_stream_result)

    result = asyncio.run(
        process_model_stream(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            messages=[],
            tools=[],
        )
    )

    assert result is not None
    assert result.chunks == []
    assert resolve_called is False
    assert graph_state.final_result == provider_failure
    assert graph_state.error is not None
    assert graph_state.error["type"] == "provider"


def test_resolve_model_stream_result_preserves_existing_final_result() -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    provider_failure = harness._failure("LM Studio rejected the request", error_type="provider")
    graph_state.final_result = provider_failure
    graph_state.error = provider_failure["error"]

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=[],
            salvage_partial_stream=None,
            last_chunk_error_details=None,
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.chunks == []
    assert graph_state.final_result == provider_failure
    assert graph_state.error is not None
    assert graph_state.error["type"] == "provider"


def test_resolve_model_stream_result_reports_reasoning_only_stall() -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    details = {
        "reason": "reasoning_only_stream_stall",
        "retrying": False,
        "attempt": 2,
        "reasoning_only_chunks": 604,
    }

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=[],
            salvage_partial_stream=None,
            last_chunk_error_details=None,
            stream_ended_without_done=True,
            stream_ended_without_done_details=details,
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.chunks == []
    assert graph_state.final_result is not None
    assert graph_state.final_result["message"].startswith("Model stream halted after repeated reasoning-only")
    assert graph_state.error is not None
    assert graph_state.error["type"] == "model_stream_stall"


def test_resolve_model_stream_result_salvages_partial_tool_call_from_degenerate_loop() -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    # Simulate a stream that produced prose plus the start of an inline tool
    # call JSON block, then degenerated while escaping a string. The collected
    # chunks contain the prose; the partial prefix carries the JSON payload.
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": "I will start by exploring the remote host.\n\n"
                        }
                    }
                ]
            },
        }
    ]
    partial_text = (
        '```json\n{"name": "ssh_exec", "arguments": '
        '{"command": "ls -la /opt/notes", "host": "192.168.1.89", "user": "root"}}\n```'
    )
    details = {
        "reason": "model_output_degenerate_loop",
        "repeated_phrase": '\": "',
        "repeat_count": 6,
        "buffer_chars": 449,
        "partial_assistant_text": partial_text,
    }

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=chunks,
            salvage_partial_stream=None,
            last_chunk_error_details=None,
            stream_ended_without_done=True,
            stream_ended_without_done_details=details,
            partial_assistant_text="",
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[{"role": "user", "content": "repair notes.lab.local"}],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.halt_reason == "model_output_degenerate_loop"
    assert result.stream.assistant_text.startswith("I will start by exploring")
    assert "ssh_exec" in result.stream.assistant_text
    # The partial inline JSON block is appended to assistant_text so that the
    # downstream tool-call parser can recover the action on the next step.
    assert "ls -la /opt/notes" in result.stream.assistant_text


def test_stalled_file_write_stream_uses_no_tools_fallback_without_session() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_target_paths"] = ["temp/logwatch.py"]
    state.run_brief.original_task = "update temp/logwatch.py"
    harness = _Harness(state)
    harness.client = _FallbackClient()
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    partial_stream = StreamResult(
        tool_calls=[
            {
                "id": "call-empty",
                "type": "function",
                "function": {"name": "file_write", "arguments": ""},
            }
        ]
    )

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=[],
            salvage_partial_stream=partial_stream,
            last_chunk_error_details={"reason": "tool_call_continuation_timeout"},
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[{"role": "user", "content": "update temp/logwatch.py"}],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.stream.tool_calls
    tool_call = result.stream.tool_calls[0]
    assert tool_call["function"]["name"] == "file_write"
    assert "print('recovered')" in tool_call["function"]["arguments"]
    assert harness.client.calls[0]["tools"] == []
    assert harness.client.calls[0]["messages"][-1]["role"] == "user"
    assert state.scratchpad["_last_text_write_fallback"]["target_path"] == "temp/logwatch.py"


def test_stalled_file_write_fallback_rejects_tiny_code_fragment() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_target_paths"] = ["temp/artifact_retention.py"]
    state.run_brief.original_task = "write temp/artifact_retention.py"
    harness = _Harness(state)
    harness.client = _WeakFallbackClient()
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    partial_stream = StreamResult(
        tool_calls=[
            {
                "id": "call-empty",
                "type": "function",
                "function": {"name": "file_write", "arguments": ""},
            }
        ]
    )

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=[],
            salvage_partial_stream=partial_stream,
            last_chunk_error_details={"reason": "tool_call_continuation_timeout"},
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[{"role": "user", "content": "write temp/artifact_retention.py"}],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.stream.tool_calls == partial_stream.tool_calls
    assert harness.client.calls[0]["tools"] == []
    assert state.scratchpad["_last_text_write_fallback"]["status"] == "failed"
    assert any(event[0][0] == "stream_text_write_fallback_rejected" for event in harness.runlog_events)


def test_stalled_html_file_write_fallback_rejects_truncated_doctype() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_target_paths"] = ["temp/rogue-grid-defense.html"]
    state.run_brief.original_task = "write temp/rogue-grid-defense.html"
    harness = _Harness(state)
    harness.client = _WeakHtmlFallbackClient()
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    partial_stream = StreamResult(
        tool_calls=[
            {
                "id": "call-empty-html",
                "type": "function",
                "function": {"name": "file_write", "arguments": ""},
            }
        ]
    )

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=[],
            salvage_partial_stream=partial_stream,
            last_chunk_error_details={"reason": "tool_call_continuation_timeout"},
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[{"role": "user", "content": "write temp/rogue-grid-defense.html"}],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.stream.tool_calls == partial_stream.tool_calls
    assert harness.client.calls[0]["tools"] == []
    assert state.scratchpad["_last_text_write_fallback"]["status"] == "failed"
    assert any(event[0][0] == "stream_text_write_fallback_rejected" for event in harness.runlog_events)


def test_stalled_ssh_file_write_stream_uses_remote_write_fallback() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "update the remote landing page"
    harness = _Harness(state)
    harness.client = _RemoteFallbackClient()
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    partial_stream = StreamResult(
        tool_calls=[
            {
                "id": "call-ssh-write",
                "type": "function",
                "function": {
                    "name": "ssh_file_write",
                    "arguments": "{\"host\":\"192.168.1.10\",\"path\":\"/var/www/html/index.html\"",
                },
            }
        ]
    )
    state.scratchpad["_last_incomplete_tool_call"] = {
        "details": {"reason": "tool_call_continuation_timeout", "provider_profile": "lmstudio", "message": "timed out"},
        "tool_call_diagnostics": [
            {
                "tool_name": "ssh_file_write",
                "required_fields": ["path", "content"],
                "present_fields": ["host", "path"],
                "missing_required_fields": ["content"],
                "arguments": {"host": "192.168.1.10", "path": "/var/www/html/index.html"},
                "raw_arguments_preview": "{\"host\":\"192.168.1.10\",\"path\":\"/var/www/html/index.html\"",
            }
        ],
    }

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=[],
            salvage_partial_stream=partial_stream,
            last_chunk_error_details={"reason": "tool_call_continuation_timeout"},
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=False,
            echo_to_stdout=False,
            messages=[{"role": "user", "content": "update the remote landing page"}],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.stream.tool_calls
    tool_call = result.stream.tool_calls[0]
    assert tool_call["function"]["name"] == "ssh_file_write"
    assert "/var/www/html/index.html" in tool_call["function"]["arguments"]
    assert "Recovered" in tool_call["function"]["arguments"]
    assert harness.client.calls[0]["tools"] == []
    assert harness.client.calls[0]["messages"][-1]["role"] == "user"
    assert state.scratchpad["_last_remote_write_fallback"]["target_path"] == "/var/www/html/index.html"


def test_collect_stream_accepts_raw_provider_native_tool_call_chunks() -> None:
    stream = OpenAICompatClient.collect_stream(_incident_native_tool_call_chunks(wrapped=False))

    assert [call["id"] for call in stream.tool_calls] == [
        "call_diskstats",
        "call_free",
        "call_dmesg",
        "call_ss",
    ]
    assert all(call["function"]["name"] == "ssh_exec" for call in stream.tool_calls)
    assert "cat /proc/diskstats" in stream.tool_calls[0]["function"]["arguments"]
    assert "network tools not available" in stream.tool_calls[3]["function"]["arguments"]


def test_clean_stream_with_multiple_native_tool_calls_survives_resolution() -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=_incident_native_tool_call_chunks(),
            salvage_partial_stream=None,
            last_chunk_error_details=None,
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=True,
            echo_to_stdout=False,
            messages=[{"role": "user", "content": "triage 192.168.1.89"}],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.halted is False
    assert len(result.stream.tool_calls) == 4
    assert [call["id"] for call in result.stream.tool_calls] == [
        "call_diskstats",
        "call_free",
        "call_dmesg",
        "call_ss",
    ]
    assert "_last_tool_call_aggregation_failure" not in state.scratchpad


def test_tool_calls_finish_without_collected_calls_schedules_recovery_nudge() -> None:
    state = LoopState(cwd="/tmp")
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        }
    ]

    async def _run():
        return await resolve_model_stream_result(
            graph_state,
            SimpleNamespace(event_handler=None, harness=harness),
            harness=harness,
            chunks=chunks,
            salvage_partial_stream=None,
            last_chunk_error_details=None,
            stream_ended_without_done=False,
            stream_ended_without_done_details={},
            trigger_early_4b_fallback=False,
            stream_completed_cleanly=True,
            echo_to_stdout=False,
            messages=[{"role": "user", "content": "triage 192.168.1.89"}],
            start_time=time.perf_counter(),
            first_token_time=None,
        )

    result = asyncio.run(_run())

    assert result is not None
    assert result.halted is True
    assert result.halt_reason == "tool_call_aggregation_failure"
    assert state.scratchpad["_last_tool_call_aggregation_failure"]["saw_tool_calls_finish"] is True
    assert any(
        message.metadata.get("recovery_kind") == "tool_call_aggregation_failure"
        for message in state.recent_messages
    )


def test_trim_degenerate_suffix_removes_repetitive_window() -> None:
    buffer = (
        'I will start exploring.\n\n```json\n'
        '{"name": "ssh_exec", "arguments": {"command": "ls"}}\n'
        '```\n" : " : " : " : " : " : "'
    )
    window = '" : " : " : " : " : " : "'
    trimmed = _trim_degenerate_suffix(buffer, window, '\": "')
    assert '" : "' not in trimmed
    assert '{"name": "ssh_exec"' in trimmed


def test_trim_degenerate_suffix_falls_back_to_phrase_trim() -> None:
    buffer = "some text then dagu dagu dagu dagu"
    trimmed = _trim_degenerate_suffix(buffer, "", "dagu")
    assert len(trimmed) < len(buffer)
    assert "some text then" in trimmed


def test_trim_degenerate_suffix_returns_original_when_no_window_or_phrase() -> None:
    buffer = "plain text without repetition"
    assert _trim_degenerate_suffix(buffer, "", "") == buffer


def test_detect_degenerate_repetition_ignores_short_markdown_bullets() -> None:
    """Legitimate markdown lists can contain repeated backtick-to-bullet transitions.

    Gemma-4-e4b produced a structured container summary where each container name
    was wrapped in backticks and followed by bullet details. The transition
    `` `\n    * `` repeated many times and was misclassified as a degenerate loop.
    Such short, formatting-only patterns should not trigger the guard.
    """
    buffer = (
        "The task was to connect to `root@192.168.1.89` and list all active and exited "
        "Docker containers. I have successfully executed `docker ps -a` on the host, and "
        "the output provides the list of containers:\n\n"
        "`eloquent_williamson`\n"
        "    * ID: 316eae6be6a5\n"
        "    * Image: vikunja/vikunja:latest\n"
        "`dagu`\n"
        "    * ID: 5122b89d0ff4\n"
        "    * Image: ghcr.io/dagucloud/dagu:latest\n"
        "`third`\n"
        "    * ID: 111111111111\n"
        "    * Image: example/third:latest\n"
        "`fourth`\n"
        "    * ID: 222222222222\n"
        "    * Image: example/fourth:latest\n"
    )
    assert _detect_degenerate_repetition(buffer) is None


def test_detect_degenerate_repetition_still_catches_semantic_loops() -> None:
    """Phrases that carry semantic content should still be detected when repeated."""
    buffer = "I understand. " * 50
    result = _detect_degenerate_repetition(buffer)
    assert result is not None
    phrase, count, window = result
    assert "understand" in phrase
    assert count >= 6
    assert len(window) <= _DEGENERATE_REPETITION_WINDOW_CHARS


def test_detect_degenerate_repetition_density_ignores_spread_words() -> None:
    """Verify that common terms spread out in normal text do not trigger loop detection."""
    # "compose" appears 6 times but spread out across 400 characters
    buffer = (
        "We need to check the docker-compose.yml file first. Let's see if the compose file "
        "contains options for notes. `/opt/notes/docker-compose.yml` is the expected path, "
        "but it might be that compose is located in `/opt/ghost/docker-compose.yml` or "
        "another location entirely. We should look for any compose stack in `/opt/qwen-compose`."
        " Finally, if the compose configurations are not there, we will verify general settings."
    )
    padding = (
        "This is a block of non-repetitive text that we are using to fill up the window "
        "so that the degenerate repetition loop detector does not exit early. It contains "
        "many unique words and structures, ensuring that there are no loops or repetitive "
        "patterns within this initial part of the buffer. Let's make it even longer to ensure "
        "that the total buffer length is well over 400 characters. This is a very standard way "
        "to test these features and ensure there is no early exit due to length constraints. "
        "We want to make sure it functions properly under all circumstances."
    )
    buffer = padding + "\n" + buffer
    assert _detect_degenerate_repetition(buffer) is None


def test_detect_degenerate_repetition_density_catches_true_repetitions() -> None:
    """Verify that back-to-back repetitions of the same phrase are still detected."""
    buffer = "dagu dagu dagu dagu dagu dagu"
    padding = (
        "This is a block of non-repetitive text that we are using to fill up the window "
        "so that the degenerate repetition loop detector does not exit early. It contains "
        "many unique words and structures, ensuring that there are no loops or repetitive "
        "patterns within this initial part of the buffer. Let's make it even longer to ensure "
        "that the total buffer length is well over 400 characters. This is a very standard way "
        "to test these features and ensure there is no early exit due to length constraints. "
        "We want to make sure it functions properly under all circumstances."
    )
    buffer = padding + "\n" + buffer
    result = _detect_degenerate_repetition(buffer)
    assert result is not None
    phrase, count, window = result
    assert phrase.strip() == "dagu"
    assert count >= 6


def test_overflow_with_passed_verdict_auto_completes() -> None:
    from smallctl.graph.lifecycle_prompt import _overflow_with_passed_verdict_to_success
    from smallctl.state import LoopState

    state = LoopState(cwd="/tmp")
    state.last_verifier_verdict = {"verdict": "pass", "tool_name": "ssh_exec"}
    harness = _Harness(state)
    harness._runlog = lambda *args, **kwargs: harness.runlog_events.append((args, kwargs))
    graph_state = GraphRunState(loop_state=state, thread_id="test", run_mode="loop")
    graph_state.last_assistant_text = "Remote deployment verified."

    result = _overflow_with_passed_verdict_to_success(
        graph_state, harness, RuntimeError("PROMPT BUDGET OVERFLOW")
    )

    assert result is True
    assert graph_state.final_result is not None
    assert graph_state.final_result["status"] == "completed"
    assert "verified" in graph_state.final_result["message"]["message"].lower()


def test_overflow_without_passed_verdict_remains_failure() -> None:
    from smallctl.graph.lifecycle_prompt import _overflow_with_passed_verdict_to_success
    from smallctl.state import LoopState

    state = LoopState(cwd="/tmp")
    state.last_verifier_verdict = {"verdict": "fail", "tool_name": "ssh_exec"}
    harness = _Harness(state)
    graph_state = GraphRunState(loop_state=state, thread_id="test", run_mode="loop")

    result = _overflow_with_passed_verdict_to_success(
        graph_state, harness, RuntimeError("PROMPT BUDGET OVERFLOW")
    )

    assert result is False
    assert graph_state.final_result is None
