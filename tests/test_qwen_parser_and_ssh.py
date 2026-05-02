from __future__ import annotations

import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from smallctl.client.client import OpenAICompatClient
from smallctl.graph.model_call_nodes import _conversation_tool_calls_from_pending, model_call
from smallctl.graph.model_stream_fallback import StreamProcessingResult
from smallctl.graph.model_stream_loop_rendering import StreamTagState, flush_model_stream_buffer, handle_model_stream_chunk
from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.graph.tool_call_parser import parse_tool_calls
from smallctl.models.events import UIEventType
from smallctl.state import ArtifactRecord, LoopState, WriteSession
from smallctl.tools.base import ToolSpec
from smallctl.tools.dispatcher import ToolDispatcher, normalize_tool_request
from smallctl.tools.registry import ToolRegistry
from smallctl.tools import network


class _FakeStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeProc:
    def __init__(self, *, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.stdout = _FakeStream([stdout, b""])
        self.stderr = _FakeStream([stderr, b""])
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        return None


class _Registry:
    @staticmethod
    def names() -> set[str]:
        return {"task_complete"}


def test_qwen_distilled_wrappers_preserve_task_complete_and_plan_reasoning() -> None:
    model_name = "qwen3.5-4b-claude-4.6-opus-reasoning-distilled"
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "Field reasoning.\n",
                            "content": (
                                "<analysis>Inspect the gathered evidence.</analysis>\n"
                                "<plan>1. Call task_complete with the final answer.</plan>\n"
                                "<execution><tool_call>"
                                '{"name":"task_complete","arguments":{"message":"Task complete."}}'
                                "</tool_call>"
                            ),
                        }
                    }
                ]
            },
        }
    ]

    stream = OpenAICompatClient.collect_stream(chunks, reasoning_mode="auto")

    assert "Field reasoning." in stream.thinking_text
    assert "Inspect the gathered evidence." in stream.thinking_text
    assert "Call task_complete with the final answer." in stream.thinking_text
    assert "<analysis>" not in stream.assistant_text
    assert "<plan>" not in stream.assistant_text
    assert "<execution>" not in stream.assistant_text
    assert "<tool_call>" in stream.assistant_text

    harness = SimpleNamespace(
        registry=_Registry(),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model=model_name),
        _runlog=lambda *args, **kwargs: None,
    )
    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="execute"),
        deps=SimpleNamespace(harness=harness),
        model_name=model_name,
    )

    assert len(parse_result.pending_tool_calls) == 1
    assert parse_result.pending_tool_calls[0].tool_name == "task_complete"
    assert parse_result.pending_tool_calls[0].args["message"] == "Task complete."
    assert parse_result.final_assistant_text == ""


def test_recovered_inline_tool_calls_are_materialized_for_conversation_history() -> None:
    pending = PendingToolCall(
        tool_name="task_complete",
        args={"message": "Task complete."},
        raw_arguments='{"message": "Task complete."}',
    )

    tool_calls = _conversation_tool_calls_from_pending(
        [pending],
        thread_id="session123",
        step_count=2,
    )

    assert pending.tool_call_id == "call_inline_session123_2_0"
    assert tool_calls == [
        {
            "id": "call_inline_session123_2_0",
            "type": "function",
            "function": {
                "name": "task_complete",
                "arguments": '{"message": "Task complete."}',
            },
        }
    ]


def test_qwen_response_wrapper_is_unwrapped_into_assistant_text() -> None:
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                "<analysis>Reason through the greeting.</analysis>\n"
                                "<response>Task complete: Greeting acknowledged.</response>"
                            ),
                        }
                    }
                ]
            },
        }
    ]

    stream = OpenAICompatClient.collect_stream(chunks, reasoning_mode="auto")

    assert stream.assistant_text.strip() == "Task complete: Greeting acknowledged."
    assert "Reason through the greeting." in stream.thinking_text
    assert "<response>" not in stream.assistant_text


def test_collect_stream_tags_mode_keeps_field_reasoning() -> None:
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": "Thinking Process: hello.\n",
                            "content": "",
                        }
                    }
                ]
            },
        }
    ]

    stream = OpenAICompatClient.collect_stream(chunks, reasoning_mode="tags")

    assert stream.assistant_text == ""
    assert "Thinking Process: hello." in stream.thinking_text


def test_parse_tool_calls_recovers_qwen_response_wrapper_from_reasoning_trace() -> None:
    stream = SimpleNamespace(
        assistant_text="",
        thinking_text=(
            '<analysis>The user is simply saying "hello".</analysis>\n'
            "<response>Task complete: Greeting acknowledged.</response>\n"
        ),
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=_Registry(),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="qwen3.5-4b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="chat"),
        deps=SimpleNamespace(harness=harness),
        model_name="qwen3.5-4b",
    )

    assert parse_result.pending_tool_calls == []
    assert parse_result.final_assistant_text == "Task complete: Greeting acknowledged."


def test_parse_tool_calls_recovers_inline_tool_call_from_thinking_text() -> None:
    stream = SimpleNamespace(
        assistant_text="Docker is installed; uninstalling next.",
        thinking_text=(
            "<tool_call>\n"
            "<function=ssh_exec>\n"
            "<parameter=command>\n"
            "apt-get remove -y docker.io\n"
            "</parameter>\n"
            "<parameter=host>\n"
            "192.168.1.63\n"
            "</parameter>\n"
            "<parameter=password>\n"
            "@S02v1735\n"
            "</parameter>\n"
            "<parameter=user>\n"
            "root\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        ),
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=SimpleNamespace(names=lambda: {"ssh_exec", "task_complete"}),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="qwen/qwen3.5-9b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="execute"),
        deps=SimpleNamespace(harness=harness),
        model_name="qwen/qwen3.5-9b",
    )

    assert len(parse_result.pending_tool_calls) == 1
    assert parse_result.pending_tool_calls[0].tool_name == "ssh_exec"
    assert parse_result.pending_tool_calls[0].args["command"] == "apt-get remove -y docker.io"
    assert parse_result.pending_tool_calls[0].args["host"] == "192.168.1.63"
    assert parse_result.pending_tool_calls[0].args["user"] == "root"
    assert parse_result.final_thinking_text == ""


def test_parse_tool_calls_recovers_orphan_parameter_block_from_thinking_text() -> None:
    stream = SimpleNamespace(
        assistant_text="Docker is confirmed installed.",
        thinking_text=(
            "</parameter>\n"
            "<parameter=command>\n"
            "docker ps -a\n"
            "</parameter>\n"
            "<parameter=host>\n"
            "192.168.1.63\n"
            "</parameter>\n"
            "<parameter=password>\n"
            "@S02v1735\n"
            "</parameter>\n"
            "<parameter=user>\n"
            "root\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        ),
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=SimpleNamespace(names=lambda: {"ssh_exec", "task_complete"}),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="qwen/qwen3.5-9b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="execute"),
        deps=SimpleNamespace(harness=harness),
        model_name="qwen/qwen3.5-9b",
    )

    assert len(parse_result.pending_tool_calls) == 1
    recovered = parse_result.pending_tool_calls[0]
    assert recovered.tool_name == "ssh_exec"
    assert recovered.args["command"] == "docker ps -a"
    assert recovered.args["host"] == "192.168.1.63"
    assert recovered.args["password"] == "@S02v1735"
    assert recovered.args["user"] == "root"
    assert parse_result.final_thinking_text == ""


def test_parse_tool_calls_strips_qwen_tool_payload_noise_from_visible_thinking() -> None:
    stream = SimpleNamespace(
        assistant_text="",
        thinking_text=(
            "Docker is not installed. Now I need to install it on the remote host 192.168.1.63. "
            "I'll install Docker using the standard Docker installation script.\n"
            "</parameter>\n"
            "<parameter=command>\n"
            "curl -fsSL https://get.docker.com -o get-docker.sh && chmod +x get-docker.sh && ./get-docker.sh\n"
            "</parameter>\n"
            "<parameter=host>\n"
            "192.168.1.63\n"
            "</parameter>\n"
            "<parameter=password>\n"
            "@S02v1735\n"
            "</parameter>\n"
            "<parameter=user>\n"
            "root\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
        ),
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=SimpleNamespace(names=lambda: {"ssh_exec", "task_complete"}),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="qwen/qwen3.5-9b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="execute"),
        deps=SimpleNamespace(harness=harness),
        model_name="qwen/qwen3.5-9b",
    )

    assert len(parse_result.pending_tool_calls) == 1
    assert parse_result.pending_tool_calls[0].tool_name == "ssh_exec"
    assert parse_result.final_thinking_text == (
        "Docker is not installed. Now I need to install it on the remote host 192.168.1.63. "
        "I'll install Docker using the standard Docker installation script."
    )
    assert "get-docker.sh" not in parse_result.final_thinking_text
    assert "@S02v1735" not in parse_result.final_thinking_text


def test_parse_tool_calls_skips_reasoning_fallback_when_tool_protocol_is_present() -> None:
    stream = SimpleNamespace(
        assistant_text="",
        thinking_text=(
            "<tool_call>\n"
            "<parameter=password>\n"
            "@S0v1735\n"
            "</parameter>\n"
            "</tool_call>\n"
        ),
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=SimpleNamespace(names=lambda: {"ssh_exec", "task_complete"}),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="qwen/qwen3.5-9b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="execute"),
        deps=SimpleNamespace(harness=harness),
        model_name="qwen/qwen3.5-9b",
    )

    assert parse_result.pending_tool_calls == []
    assert parse_result.final_assistant_text == ""


def test_parse_tool_calls_recovers_markdown_wrapped_task_complete_function_syntax() -> None:
    stream = SimpleNamespace(
        assistant_text="**task_complete(message='HTTP GET confirmed the page is served remotely.')**",
        thinking_text="",
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=SimpleNamespace(names=lambda: {"ssh_exec", "task_complete"}),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="wrench-9b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="execute"),
        deps=SimpleNamespace(harness=harness),
        model_name="wrench-9b",
    )

    assert len(parse_result.pending_tool_calls) == 1
    recovered = parse_result.pending_tool_calls[0]
    assert recovered.tool_name == "task_complete"
    assert recovered.args["message"] == "HTTP GET confirmed the page is served remotely."
    assert parse_result.final_assistant_text == ""


def test_parse_tool_calls_preserves_qwen3_14b_response_after_task_complete() -> None:
    stream = SimpleNamespace(
        assistant_text=(
            '<tool_call>{"name":"task_complete","arguments":{"message":"Task complete."}}</tool_call>'
        ),
        thinking_text=(
            "<analysis>The user asked why the greeting was missed.</analysis>\n"
            "<response>Hello! I notice you're pointing out I didn't greet you. "
            "Let me know how I can assist.</response>\n"
        ),
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=_Registry(),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="qwen3-14b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="chat"),
        deps=SimpleNamespace(harness=harness),
        model_name="qwen3-14b",
    )

    assert len(parse_result.pending_tool_calls) == 1
    assert parse_result.pending_tool_calls[0].tool_name == "task_complete"
    assert parse_result.final_assistant_text == (
        "Hello! I notice you're pointing out I didn't greet you. Let me know how I can assist."
    )


def test_parse_tool_calls_strips_orphan_tool_protocol_residue() -> None:
    stream = SimpleNamespace(
        assistant_text='<function=dir_list>{"path":"."}</function></tool_call>',
        thinking_text="",
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=SimpleNamespace(names=lambda: {"dir_list", "task_complete"}),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="qwen/qwen3.5-9b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="chat"),
        deps=SimpleNamespace(harness=harness),
        model_name="qwen/qwen3.5-9b",
    )

    assert len(parse_result.pending_tool_calls) == 1
    assert parse_result.pending_tool_calls[0].tool_name == "dir_list"
    assert parse_result.pending_tool_calls[0].args["path"] == "."
    assert parse_result.final_assistant_text == ""


def test_parse_tool_calls_recovers_reasoning_only_final_output_marker() -> None:
    stream = SimpleNamespace(
        assistant_text="",
        thinking_text=(
            "Thinking Process:\n"
            "1. Analyze the request.\n"
            'Final Output: "Hello!"\n'
            "Actually, that is the safest reply.\n"
        ),
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=_Registry(),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="wrench-9b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="chat"),
        deps=SimpleNamespace(harness=harness),
        model_name="wrench-9b",
    )

    assert parse_result.pending_tool_calls == []
    assert parse_result.final_assistant_text == "Hello!"


def test_parse_tool_calls_reasoning_only_trace_uses_safe_non_silent_fallback() -> None:
    stream = SimpleNamespace(
        assistant_text="",
        thinking_text=(
            "Thinking Process:\n"
            "1. Analyze the request.\n"
            "2. Check constraints.\n"
            "3. Formulate response.\n"
        ),
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=_Registry(),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="wrench-9b"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="chat"),
        deps=SimpleNamespace(harness=harness),
        model_name="wrench-9b",
    )

    assert parse_result.pending_tool_calls == []
    assert parse_result.final_assistant_text == "The model returned reasoning text but no final answer."
    assert "Analyze the request" not in parse_result.final_assistant_text


def test_streaming_qwen_wrappers_hide_analysis_and_keep_response_visible() -> None:
    events: list[object] = []
    printed: list[str] = []

    async def _emit(_handler: object, event: object) -> None:
        events.append(event)

    harness = SimpleNamespace(
        thinking_visibility=True,
        state=SimpleNamespace(planning_mode_enabled=False),
        _emit=_emit,
        _runlog=lambda *args, **kwargs: None,
        _stream_print=lambda text: printed.append(text),
    )
    deps = SimpleNamespace(event_handler=object())
    state = StreamTagState()
    first_token_time = None
    chunks: list[dict[str, object]] = []

    stream_events = [
        {"data": {"choices": [{"delta": {"reasoning_content": "Reason through the greeting.\n"}}]}},
        {"data": {"choices": [{"delta": {"content": "<"}}]}},
        {"data": {"choices": [{"delta": {"content": "analysis"}}]}},
        {"data": {"choices": [{"delta": {"content": ">"}}]}},
        {"data": {"choices": [{"delta": {"content": "Reason through the greeting.\n"}}]}},
        {"data": {"choices": [{"delta": {"content": "</"}}]}},
        {"data": {"choices": [{"delta": {"content": "analysis"}}]}},
        {"data": {"choices": [{"delta": {"content": ">"}}]}},
        {"data": {"choices": [{"delta": {"content": "\n<"}}]}},
        {"data": {"choices": [{"delta": {"content": "response"}}]}},
        {"data": {"choices": [{"delta": {"content": ">"}}]}},
        {"data": {"choices": [{"delta": {"content": "Hello!"}}]}},
        {"data": {"choices": [{"delta": {"content": "</"}}]}},
        {"data": {"choices": [{"delta": {"content": "response"}}]}},
        {"data": {"choices": [{"delta": {"content": ">"}}]}},
    ]

    for event in stream_events:
        state, first_token_time = asyncio.run(
            handle_model_stream_chunk(
                harness=harness,
                deps=deps,
                event=event,
                start_tag="<think>",
                end_tag="</think>",
                echo_to_stdout=True,
                chunks=chunks,
                stream_state=state,
                first_token_time=first_token_time,
            )
        )

    asyncio.run(
        flush_model_stream_buffer(
            harness=harness,
            deps=deps,
            stream_state=state,
            start_tag="<think>",
            end_tag="</think>",
            echo_to_stdout=True,
        )
    )

    assistant_text = "".join(
        event.content
        for event in events
        if getattr(event, "event_type", None) == UIEventType.ASSISTANT
    )
    thinking_text = "".join(
        event.content
        for event in events
        if getattr(event, "event_type", None) == UIEventType.THINKING
    )

    assert assistant_text == "\nHello!"
    assert thinking_text == "Reason through the greeting.\n"
    assert "<analysis>" not in assistant_text
    assert "<response>" not in assistant_text
    assert printed == ["\n", "Hello!"]


def test_model_call_emits_thinking_replace_after_qwen_tool_recovery() -> None:
    emitted: list[object] = []
    recorded_messages: list[tuple[str, list[dict[str, object]]]] = []

    async def _emit(_handler: object, event: object) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="qwen/qwen3.5-9b"),
        registry=SimpleNamespace(names=lambda: {"ssh_exec", "task_complete"}),
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
            assistant_text="",
            thinking_text=(
                "Docker is not installed.\n"
                "</parameter>\n"
                "<parameter=command>\n"
                "curl -fsSL https://get.docker.com -o get-docker.sh && chmod +x get-docker.sh && ./get-docker.sh\n"
                "</parameter>\n"
                "<parameter=host>\n"
                "192.168.1.63\n"
                "</parameter>\n"
                "<parameter=password>\n"
                "@S02v1735\n"
                "</parameter>\n"
                "<parameter=user>\n"
                "root\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>\n"
            ),
            tool_calls=[],
        ),
        timeline=[],
        usage={},
        duration=0.5,
        ttft=0.1,
    )

    with patch("smallctl.graph.model_call_nodes.process_model_stream", AsyncMock(return_value=result)):
        asyncio.run(
            model_call(
                graph_state,
                deps,
                messages=[{"role": "user", "content": "install docker"}],
                tools=[],
            )
        )

    assert graph_state.last_thinking_text == "Docker is not installed."
    assert len(graph_state.pending_tool_calls) == 1
    replace_events = [
        event
        for event in emitted
        if getattr(event, "event_type", None) == UIEventType.THINKING
        and getattr(event, "data", {}).get("kind") == "replace"
    ]
    assert replace_events
    assert replace_events[-1].content == "Docker is not installed."
    assert recorded_messages[-1][1][0]["function"]["name"] == "ssh_exec"


def test_model_stream_batches_adjacent_assistant_chunks() -> None:
    emitted: list[object] = []

    async def _emit(_handler: object, event: object) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        _emit=_emit,
        _runlog=lambda *args, **kwargs: None,
        _stream_print=lambda _text: None,
        state=SimpleNamespace(planning_mode_enabled=False),
        thinking_visibility=True,
    )
    deps = SimpleNamespace(event_handler=object())
    state = StreamTagState()
    chunks: list[dict[str, object]] = []
    first_token_time = None

    for content in ("Hel", "lo", "!"):
        state, first_token_time = asyncio.run(
            handle_model_stream_chunk(
                harness=harness,
                deps=deps,
                event={"data": {"choices": [{"delta": {"content": content}}]}},
                start_tag="<think>",
                end_tag="</think>",
                echo_to_stdout=False,
                chunks=chunks,
                stream_state=state,
                first_token_time=first_token_time,
            )
        )

    asyncio.run(
        flush_model_stream_buffer(
            harness=harness,
            deps=deps,
            stream_state=state,
            start_tag="<think>",
            end_tag="</think>",
            echo_to_stdout=False,
        )
    )

    assistant_events = [
        event for event in emitted if getattr(event, "event_type", None) == UIEventType.ASSISTANT
    ]

    assert len(assistant_events) == 1
    assert assistant_events[0].content == "Hello!"


def test_ssh_exec_distinguishes_remote_exit_from_transport_failure() -> None:
    state = LoopState(cwd="/tmp")

    with patch.object(
        network,
        "create_process",
        AsyncMock(return_value=_FakeProc(returncode=1)),
    ):
        result = asyncio.run(
            network.ssh_exec(
                host="192.168.1.63",
                user="root",
                password="secret",
                command="which guacamole || which apache-guacamole",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is False
    assert result["error"] == "Remote SSH command exited with code 1"
    assert result["metadata"]["failure_kind"] == "remote_command"
    assert result["metadata"]["ssh_transport_succeeded"] is True
    assert result["metadata"]["ssh_auth_mode"] == "password"
    assert result["metadata"]["ssh_auth_transport"] == "sshpass_env"
    assert result["metadata"]["ssh_password_provided"] is True
    assert result["metadata"]["output"]["exit_code"] == 1
    assert not any("username/password" in hint.lower() for hint in result["metadata"]["hints"])


def test_ssh_exec_retries_when_accept_new_is_rejected() -> None:
    state = LoopState(cwd="/tmp")
    create_process = AsyncMock(
        side_effect=[
            _FakeProc(
                returncode=255,
                stderr=b"command-line line 0: keyword StrictHostKeyChecking extra arguments at end of line\n",
            ),
            _FakeProc(
                returncode=0,
                stdout=b"ii  guacamole 1.5.0\n",
            ),
        ]
    )

    with patch.object(network, "create_process", create_process):
        result = asyncio.run(
            network.ssh_exec(
                host="192.168.1.63",
                user="root",
                password="secret",
                command="dpkg -l | grep guacamole",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is True
    assert result["output"]["stdout"] == "ii  guacamole 1.5.0\n"
    assert result["metadata"]["ssh_auth_mode"] == "password"
    assert result["metadata"]["ssh_auth_transport"] == "sshpass_env"
    assert result["metadata"]["ssh_option_retry"] == "strict_host_key_checking_no"
    assert result["metadata"]["ssh_option_retry_reason"] == "accept_new_incompatible"
    assert result["metadata"]["ssh_strict_host_key_checking"] == "no"
    assert create_process.await_count == 2
    first_command = create_process.await_args_list[0].kwargs["command"]
    second_command = create_process.await_args_list[1].kwargs["command"]
    assert "StrictHostKeyChecking=accept-new" in first_command
    assert "StrictHostKeyChecking=no" in second_command


def test_ssh_exec_recovers_missing_user_from_task_context() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = (
        'ssh root@192.168.1.63 with username root password "@S02v1735" '
        "is apache guacamole installed?"
    )

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.63",
            "command": "which guac && guac --version || echo 'Guacamole not installed'",
            "password": "@S02v1735",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["user"] == "root"
    assert metadata["recovered_ssh_user"] == "root"
    assert metadata["ssh_auth_mode"] == "password"
    assert metadata["ssh_auth_transport"] == "sshpass_env"
    assert metadata["ssh_password_origin"] == "explicit"
    assert metadata["ssh_password_recovered"] is False


def test_ssh_exec_recovers_missing_password_from_task_context() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = (
        'ssh into 192.168.1.63 with username "root" and password "@S02v1735", '
        "then install nginx"
    )

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.63",
            "user": "root",
            "command": "whoami && pwd",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["password"] == "@S02v1735"
    assert metadata["recovered_ssh_password"] is True
    assert metadata["recovered_ssh_password_source"] == "task_context"
    assert metadata["ssh_auth_mode"] == "password"
    assert metadata["ssh_auth_transport"] == "sshpass_env"
    assert metadata["ssh_password_origin"] == "task_context"
    assert metadata["ssh_password_recovered"] is True


def test_ssh_exec_recovers_missing_password_from_prior_authenticated_ssh_exec() -> None:
    state = LoopState(cwd=".")
    state.tool_execution_records["ssh-1"] = {
        "tool_name": "ssh_exec",
        "args": {
            "host": "192.168.1.63",
            "user": "root",
            "password": "@S02v1735",
            "command": "apt-get update && apt-get install -y nginx",
        },
        "result": {
            "success": True,
            "output": {"stdout": "", "stderr": "", "exit_code": 0},
            "error": None,
            "metadata": {},
        },
    }

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "target": "root@192.168.1.63",
            "command": "whoami && pwd",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["password"] == "@S02v1735"
    assert metadata["recovered_ssh_password"] is True
    assert metadata["recovered_ssh_password_source"] == "prior_ssh_exec"
    assert metadata["ssh_password_origin"] == "prior_ssh_exec"
    assert metadata["ssh_password_recovered"] is True


def test_ssh_exec_recovers_missing_user_and_password_from_session_memory() -> None:
    state = LoopState(cwd=".")
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {
            "host": "192.168.1.63",
            "user": "root",
            "password": "@S02v1735",
        }
    }

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.63",
            "command": "docker ps",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["user"] == "root"
    assert args["password"] == "@S02v1735"
    assert metadata["recovered_ssh_user"] == "root"
    assert metadata["recovered_ssh_user_source"] == "session_memory"
    assert metadata["recovered_ssh_password"] is True
    assert metadata["recovered_ssh_password_source"] == "session_memory"
    assert metadata["ssh_password_origin"] == "session_memory"
    assert metadata["ssh_password_recovered"] is True


def test_ssh_exec_recovers_connection_probe_command_from_task_context() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = 'ssh into root@192.168.1.63 password "@S02v1735"'

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.63",
            "password": "@S02v1735",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["user"] == "root"
    assert args["command"] == "whoami"
    assert metadata["recovered_ssh_user"] == "root"
    assert metadata["recovered_ssh_command"] == "whoami"
    assert metadata["ssh_password_origin"] == "explicit"
    assert metadata["ssh_password_recovered"] is False


def test_normalize_tool_request_blocks_unparseable_raw_shell_ssh() -> None:
    state = LoopState(cwd=".")

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(
            get=lambda name: SimpleNamespace(
                phase_allowed=lambda phase: True,
                profile_allowed=lambda profiles: True,
            )
            if name == "ssh_exec"
            else None
        ),
        "shell_exec",
        {"command": "scp ./local.txt root@192.168.1.63:/tmp/local.txt"},
        phase="execute",
        state=state,
    )

    assert intercepted is not None
    assert tool_name == "shell_exec"
    assert args["command"].startswith("scp ")
    assert intercepted.metadata["reason"] == "raw_ssh_shell_blocked"
    assert "Use canonical `ssh_exec`" in intercepted.error
    assert metadata == {}


def test_normalize_tool_request_blocks_repeated_ssh_auth_failure_without_new_password() -> None:
    state = LoopState(cwd=".")
    state.scratchpad["_ssh_auth_recovery_state"] = {
        "root@192.168.1.63": {
            "host": "192.168.1.63",
            "user": "root",
            "failure_count": 1,
            "password_provided": True,
            "password_fingerprint": hashlib.sha256(b"secret").hexdigest()[:16],
            "last_command": "whoami",
            "last_error": "Permission denied (publickey,password).",
        }
    }

    tool_name, _args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.63",
            "user": "root",
            "password": "secret",
            "command": "whoami",
        },
        phase="execute",
        state=state,
    )

    assert tool_name == "ssh_exec"
    assert intercepted is not None
    assert intercepted.metadata["reason"] == "ssh_auth_recovery_required"
    assert intercepted.metadata["next_required_action"]["tool_names"] == ["ssh_exec", "ask_human", "task_fail"]
    assert metadata["ssh_auth_recovery_required"] is True


def test_normalize_tool_request_allows_ssh_auth_retry_with_corrected_password() -> None:
    state = LoopState(cwd=".")
    state.scratchpad["_ssh_auth_recovery_state"] = {
        "root@192.168.1.63": {
            "host": "192.168.1.63",
            "user": "root",
            "failure_count": 1,
            "password_provided": True,
            "password_fingerprint": hashlib.sha256(b"secret").hexdigest()[:16],
            "last_command": "whoami",
            "last_error": "Permission denied (publickey,password).",
        }
    }

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.63",
            "user": "root",
            "password": "new-secret",
            "command": "whoami",
        },
        phase="execute",
        state=state,
    )

    assert tool_name == "ssh_exec"
    assert intercepted is None
    assert args["password"] == "new-secret"
    assert metadata["ssh_auth_recovery_branch"] == "retry_with_password"


def test_normalize_tool_request_repairs_small_model_tool_aliases() -> None:
    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "use_shell_exec",
        {"command": "pwd"},
        phase="execute",
        state=LoopState(cwd="."),
    )

    assert intercepted is None
    assert tool_name == "shell_exec"
    assert args == {"command": "pwd"}
    assert metadata["repaired_tool_alias_from"] == "use_shell_exec"
    assert metadata["repaired_tool_alias_to"] == "shell_exec"


def test_normalize_tool_request_backfills_active_write_session_path() -> None:
    state = LoopState(cwd=".")
    state.write_session = WriteSession(
        write_session_id="ws_f4be97",
        write_target_path="./temp/service_supervisor.py",
        status="open",
    )

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "file_write",
        {
            "content": "def supervise():\n    return True\n",
            "write_session_id": "ws_f4be97",
            "section_name": "supervisor_core",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "file_write"
    assert args["path"] == "./temp/service_supervisor.py"
    assert args["write_session_id"] == "ws_f4be97"
    assert metadata["argument_repair"] == "active_write_session_path"
    assert metadata["repaired_write_session_path"] is True


def test_normalize_tool_request_does_not_backfill_mismatched_write_session_path() -> None:
    state = LoopState(cwd=".")
    state.write_session = WriteSession(
        write_session_id="ws_active",
        write_target_path="./temp/service_supervisor.py",
        status="open",
    )

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "file_write",
        {
            "content": "def other():\n    return True\n",
            "write_session_id": "ws_other",
            "section_name": "other_core",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "file_write"
    assert "path" not in args
    assert "argument_repair" not in metadata


def test_tool_dispatcher_repairs_write_session_path_before_schema_validation() -> None:
    async def _handler(**kwargs):
        return {
            "success": True,
            "output": kwargs["path"],
            "error": None,
            "metadata": {"path": kwargs["path"]},
        }

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="file_write",
            description="test file_write",
            schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "write_session_id": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            handler=_handler,
        )
    )
    state = LoopState(cwd=".")
    state.write_session = WriteSession(
        write_session_id="ws_f4be97",
        write_target_path="./temp/service_supervisor.py",
        status="open",
    )

    result = asyncio.run(
        ToolDispatcher(registry, state=state, phase="execute").dispatch(
            "file_write",
            {
                "content": "def supervise():\n    return True\n",
                "write_session_id": "ws_f4be97",
            },
        )
    )

    assert result.success is True
    assert result.output == "./temp/service_supervisor.py"
    assert result.metadata["argument_repair"] == "active_write_session_path"


def test_normalize_artifact_read_preserves_write_session_stage_alias() -> None:
    state = LoopState(cwd=".")

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "artifact_read",
        {"artifact_id": "ws_7b8cd9__circuit_breaker__stage.py"},
        phase="repair",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "artifact_read"
    assert args["artifact_id"] == "ws_7b8cd9__circuit_breaker__stage.py"
    assert metadata["argument_repair"] == "artifact_read_preserve_write_session_alias"


def test_normalize_artifact_read_resolves_existing_non_a_artifact_key() -> None:
    state = LoopState(cwd=".")
    state.artifacts["ws_7b8cd9__circuit_breaker__stage.py"] = ArtifactRecord(
        artifact_id="ws_7b8cd9__stage",
        kind="file",
        source="/tmp/.smallctl/write_sessions/ws_7b8cd9__circuit_breaker__stage.py",
        created_at="2026-04-20T00:00:00+00:00",
        size_bytes=12,
        summary="stage artifact",
        tool_name="file_write",
    )

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "artifact_read",
        {"artifact_id": "ws_7b8cd9__circuit_breaker__stage.py"},
        phase="repair",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "artifact_read"
    assert args["artifact_id"] == "ws_7b8cd9__circuit_breaker__stage.py"
    assert metadata["argument_repair"] == "artifact_read_alias_to_existing_key"


def test_remote_task_guard_blocks_local_shell_exec_for_remote_context() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = (
        'ssh root@192.168.1.63 with username root password "@S02v1735" '
        "is apache guacamole installed?"
    )

    def _get(name: str):
        if name == "ssh_exec":
            return SimpleNamespace(phase_allowed=lambda phase: True)
        return None

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=_get),
        "shell_exec",
        {"command": "docker ps"},
        phase="execute",
        state=state,
    )

    assert tool_name == "shell_exec"
    assert args == {"command": "docker ps"}
    assert metadata == {}
    assert intercepted is not None
    assert intercepted.success is False
    assert intercepted.error == "This is a remote task. Use `ssh_exec`, not local `shell_exec`."
    assert intercepted.metadata["reason"] == "remote_task_requires_ssh_exec"


def test_remote_scope_guard_blocks_shell_exec_on_remote_nginx_paths_after_resteer() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    state.task_mode = "debug_inspect"
    state.active_intent = "general_task"
    state.scratchpad["_last_task_handoff"] = {
        "task_mode": "remote_execute",
        "effective_task": "ssh into root@192.168.1.63 and split the remote explainer pages",
    }
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }

    def _get(name: str):
        if name == "ssh_exec":
            return SimpleNamespace(phase_allowed=lambda phase: True)
        return None

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=_get),
        "shell_exec",
        {"command": "ls /etc/nginx/sites-available"},
        phase="execute",
        state=state,
    )

    assert tool_name == "shell_exec"
    assert args == {"command": "ls /etc/nginx/sites-available"}
    assert metadata == {}
    assert intercepted is not None
    assert intercepted.success is False
    assert intercepted.metadata["reason"] == "remote_path_requires_ssh_exec"


def test_remote_scope_guard_does_not_recommend_unavailable_ssh_exec() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    state.task_mode = "analysis"
    state.active_tool_profiles = ["core"]
    state.scratchpad["_last_task_handoff"] = {
        "task_mode": "remote_execute",
        "effective_task": "ssh into root@192.168.1.63 and publish the remote site",
    }
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "shell_exec",
        {"command": "ls /var/www/html"},
        phase="execute",
        state=state,
    )

    assert tool_name == "shell_exec"
    assert args == {"command": "ls /var/www/html"}
    assert metadata == {}
    assert intercepted is not None
    assert intercepted.success is False
    assert intercepted.metadata["reason"] == "remote_path_requires_ssh_exec"
    assert "suggested_tool" not in intercepted.metadata
    assert "not currently available" in intercepted.error


def test_remote_scope_guard_blocks_local_nginx_probe_after_resteer() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    state.task_mode = "analysis"
    state.active_intent = "general_task"
    state.scratchpad["_last_task_handoff"] = {
        "task_mode": "remote_execute",
        "effective_task": "ssh into root@192.168.1.63 and configure the nginx site",
    }
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }

    def _get(name: str):
        if name == "ssh_exec":
            return SimpleNamespace(phase_allowed=lambda phase: True)
        return None

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=_get),
        "shell_exec",
        {"command": "which nginx"},
        phase="execute",
        state=state,
    )

    assert tool_name == "shell_exec"
    assert args == {"command": "which nginx"}
    assert metadata == {}
    assert intercepted is not None
    assert intercepted.success is False
    assert intercepted.metadata["reason"] == "remote_task_requires_ssh_exec"


def test_remote_scope_guard_blocks_dir_list_for_remote_web_root_after_resteer() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    state.task_mode = "analysis"
    state.active_intent = "general_task"
    state.scratchpad["_last_task_handoff"] = {
        "task_mode": "remote_execute",
        "effective_task": "ssh into root@192.168.1.63 and publish the split explainer pages",
    }
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "dir_list",
        {"path": "/var/www/html"},
        phase="execute",
        state=state,
    )

    assert tool_name == "dir_list"
    assert args == {"path": "/var/www/html"}
    assert metadata == {}
    assert intercepted is not None
    assert intercepted.success is False
    assert intercepted.metadata["reason"] == "remote_path_requires_ssh_exec"


def test_normalize_ssh_arguments_prefers_target_user_at_host() -> None:
    normalized = network.normalize_ssh_arguments(
        {
            "target": "root@192.168.1.63",
            "command": "whoami",
            "password": "@S02v1735",
        }
    )

    assert normalized["host"] == "192.168.1.63"
    assert normalized["user"] == "root"
    assert "target" not in normalized


def test_normalize_ssh_arguments_supports_username_alias() -> None:
    normalized = network.normalize_ssh_arguments(
        {
            "host": "192.168.1.63",
            "username": "root",
            "command": "whoami",
        }
    )

    assert normalized["host"] == "192.168.1.63"
    assert normalized["user"] == "root"


def test_normalize_ssh_arguments_requires_host_or_target() -> None:
    try:
        network.normalize_ssh_arguments({"command": "whoami"})
    except ValueError as exc:
        assert str(exc) == "SSH target requires either `target` or `host`."
    else:
        raise AssertionError("expected ValueError for missing SSH target")


def test_parse_ssh_exec_args_from_shell_command_recovers_connection_probe() -> None:
    normalized = network.parse_ssh_exec_args_from_shell_command("ssh root@192.168.1.63")

    assert normalized == {
        "host": "192.168.1.63",
        "user": "root",
        "command": "whoami",
    }
