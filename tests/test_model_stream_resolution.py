from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import smallctl.graph.model_stream as model_stream_module
from smallctl.client import StreamResult
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

    def _runlog(self, *args, **kwargs) -> None:
        del args, kwargs

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
    assert state.scratchpad["_last_text_write_fallback"]["target_path"] == "temp/logwatch.py"


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
    assert state.scratchpad["_last_remote_write_fallback"]["target_path"] == "/var/www/html/index.html"
