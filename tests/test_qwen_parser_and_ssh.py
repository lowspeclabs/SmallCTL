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
from smallctl.harness.memory import MemoryService
from smallctl.graph.tool_call_parser import parse_tool_calls
from smallctl.graph.tool_inline_parsing import _extract_inline_tool_calls
from smallctl.models.events import UIEventType
from smallctl.models.conversation import ConversationMessage
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState, WriteSession
from smallctl.tools.base import ToolSpec
from smallctl.tools.dispatcher import ToolDispatcher, normalize_tool_request
from smallctl.tools.registry import ToolRegistry
from smallctl.tools import network
from smallctl.tools.network_ssh_helpers import ssh_semantic_failure


class _FakeStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _size: int) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        return b""


class _FakeStdin:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    def __init__(self, *, returncode: int, stdout: bytes = b"", stderr: bytes = b"", stdin: _FakeStdin | None = None) -> None:
        self.stdout = _FakeStream([stdout, b""])
        self.stderr = _FakeStream([stderr, b""])
        self.returncode = returncode
        self.stdin = stdin

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        return None


class _Registry:
    @staticmethod
    def names() -> set[str]:
        return {"task_complete"}


def test_parse_tool_calls_strips_thinking_tags_from_gemma_assistant_text() -> None:
    raw_assistant_text = (
        "<think>\n"
        'The user said "hi". This is a greeting. I should respond politely and ask how I can help.\n'
        "</think>Hello! How can I help you today?\n"
        '{"name": "task_complete", "arguments": {"message": "Greeted the user."}}'
    )
    stream = SimpleNamespace(
        assistant_text=raw_assistant_text,
        thinking_text="",
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=_Registry(),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="google/gemma-4-31b-it"),
        _runlog=lambda *args, **kwargs: None,
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="chat"),
        deps=SimpleNamespace(harness=harness),
        model_name="google/gemma-4-31b-it",
    )

    assert "<think>" not in parse_result.final_assistant_text
    assert "</think>" not in parse_result.final_assistant_text
    assert parse_result.final_assistant_text.startswith("Hello!")
    assert len(parse_result.pending_tool_calls) == 1
    assert parse_result.pending_tool_calls[0].tool_name == "task_complete"


def test_ssh_semantic_failure_detects_pipelined_dnf_error_exit_zero() -> None:
    output = {
        "stdout": "dnf search: error: unrecognized arguments: --allrepo\n",
        "stderr": "",
        "exit_code": 0,
    }

    reason = ssh_semantic_failure("dnf search --allrepo pihole 2>&1 | head -50", output)

    assert "unrecognized arguments" in reason


def test_ssh_semantic_failure_ignores_plain_diagnostic_probe() -> None:
    output = {
        "stdout": "bash: line 1: which: command not found\n/bin/dnf\n",
        "stderr": "",
        "exit_code": 0,
    }

    assert ssh_semantic_failure("which apt apt-get yum dnf 2>&1; ls /bin/dnf", output) == ""


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


def test_inline_json_tool_call_preserves_stripped_top_level_session_id_metadata() -> None:
    cleaned, calls = _extract_inline_tool_calls(
        '{"session_id":"2d3ac862","name":"task_complete","arguments":{"message":"ok"}}'
    )

    assert cleaned == ""
    assert len(calls) == 1
    assert calls[0].tool_name == "task_complete"
    assert calls[0].args == {"message": "ok"}
    assert calls[0].parser_metadata["inline_json_extra_fields"] == {
        "session_id": "2d3ac862"
    }


def test_plain_json_without_tool_name_is_not_removed_by_inline_parser() -> None:
    text = '{"session_id":"2d3ac862","status":"debug"}'

    cleaned, calls = _extract_inline_tool_calls(text)

    assert cleaned == text
    assert calls == []


def test_native_ssh_exec_malformed_arguments_preserve_parse_diagnostic() -> None:
    raw_arguments = '{"command": "cd /opt/qwen-compose-medium && docker compose up -d"{}'

    pending = PendingToolCall.from_payload(
        {
            "id": "call_step4",
            "type": "function",
            "function": {"name": "ssh_exec", "arguments": raw_arguments},
        }
    )

    assert pending is not None
    assert pending.tool_name == "ssh_exec"
    assert pending.args == {}
    assert pending.raw_arguments == raw_arguments
    assert pending.parser_metadata["arguments_empty"] is False
    assert pending.parser_metadata["raw_arguments_preview"] == raw_arguments
    assert pending.parser_metadata["arguments_parse_error"]["kind"] == "malformed_json_arguments"


def test_native_tool_empty_arguments_are_not_marked_malformed() -> None:
    pending = PendingToolCall.from_payload(
        {
            "id": "call_empty",
            "type": "function",
            "function": {"name": "ssh_exec", "arguments": ""},
        }
    )

    assert pending is not None
    assert pending.args == {}
    assert pending.parser_metadata["arguments_empty"] is True
    assert "arguments_parse_error" not in pending.parser_metadata


def test_native_tool_empty_json_object_arguments_are_not_marked_malformed() -> None:
    pending = PendingToolCall.from_payload(
        {
            "id": "call_empty_object",
            "type": "function",
            "function": {"name": "ssh_exec", "arguments": "{}"},
        }
    )

    assert pending is not None
    assert pending.args == {}
    assert pending.parser_metadata["raw_arguments_preview"] == "{}"
    assert "arguments_empty" not in pending.parser_metadata
    assert "arguments_parse_error" not in pending.parser_metadata


def test_parse_tool_calls_logs_stripped_inline_json_metadata() -> None:
    runlog_events = []
    stream = SimpleNamespace(
        assistant_text='{"session_id":"2d3ac862","name":"task_complete","arguments":{"message":"ok"}}',
        thinking_text="",
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=_Registry(),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="gemma-4-e4b-it"),
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="chat"),
        deps=SimpleNamespace(harness=harness),
        model_name="gemma-4-e4b-it",
    )

    assert len(parse_result.pending_tool_calls) == 1
    assert runlog_events == [
        (
            "inline_tool_metadata_stripped",
            "inline tool JSON contained non-argument top-level fields",
            {
                "tool_name": "task_complete",
                "extra_fields": {"session_id": "2d3ac862"},
            },
        )
    ]

def test_lfm_plan_json_is_stripped_and_next_action_tool_recovered() -> None:
    stream = SimpleNamespace(
        assistant_text=(
            '{\n'
            '  "plan": "Check remote state, then continue.",\n'
            '  "next_actions": [\n'
            '    {"tool_name": "ssh_exec", "arguments": {"host": "192.168.1.161", "user": "root", "command": "test -d /opt/pi-hole && echo present"}}\n'
            '  ],\n'
            '  "status_required": "low",\n'
            '  "next_step": null\n'
            '}'
        ),
        thinking_text="",
        tool_calls=[],
    )
    registry = SimpleNamespace(names=lambda: {"ssh_exec", "task_complete", "task_fail"})
    harness = SimpleNamespace(
        registry=registry,
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="lfm2.5-8b-a1b"),
        _runlog=lambda *args, **kwargs: None,
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="chat"),
        deps=SimpleNamespace(harness=harness),
        model_name="lfm2.5-8b-a1b",
    )

    assert parse_result.final_assistant_text == ""
    assert "next_actions" not in parse_result.cleaned_text
    assert len(parse_result.pending_tool_calls) == 1
    pending = parse_result.pending_tool_calls[0]
    assert pending.tool_name == "ssh_exec"
    assert pending.args["command"] == "test -d /opt/pi-hole && echo present"
    assert pending.parser_metadata["lfm_plan_json_recovered"] is True


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


def test_gemma_channel_marker_does_not_make_prior_prose_thinking() -> None:
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                "I will patch the function and then run the verifier."
                                "<channel|>"
                            ),
                        }
                    }
                ]
            },
        }
    ]

    stream = OpenAICompatClient.collect_stream(chunks, reasoning_mode="auto")

    assert stream.assistant_text == "I will patch the function and then run the verifier."
    assert stream.thinking_text == ""


def test_gemma_thought_tags_are_extracted_as_thinking() -> None:
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                "<thought>Inspect the previous failure.</thought>"
                                "I will patch the function."
                            ),
                        }
                    }
                ]
            },
        }
    ]

    stream = OpenAICompatClient.collect_stream(chunks, reasoning_mode="auto")

    assert stream.assistant_text == "I will patch the function."
    assert stream.thinking_text == "Inspect the previous failure."


def test_gemma_unclosed_thought_ends_at_channel_marker() -> None:
    chunks = [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                "<thought>Inspect the previous failure."
                                "<channel|>I will patch the function."
                            ),
                        }
                    }
                ]
            },
        }
    ]

    stream = OpenAICompatClient.collect_stream(chunks, reasoning_mode="auto")

    assert stream.assistant_text == "I will patch the function."
    assert stream.thinking_text == "Inspect the previous failure."


def test_gemma_thought_aliases_survive_stream_chunk_boundaries_in_timeline() -> None:
    chunks = [
        {
            "type": "chunk",
            "data": {"choices": [{"delta": {"content": "<tho"}}]},
        },
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {"delta": {"content": "ught>Think it through.</thought>Done.<channel|>"}}
                ]
            },
        },
    ]

    timeline = OpenAICompatClient.collect_timeline(chunks, reasoning_mode="auto")

    assert [(entry.kind, entry.content) for entry in timeline] == [
        ("thinking", "Think it through."),
        ("assistant", "Done."),
    ]


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
    runlog: list[tuple[str, str, dict[str, object]]] = []

    async def _emit(_handler: object, event: object) -> None:
        events.append(event)

    harness = SimpleNamespace(
        thinking_visibility=True,
        state=SimpleNamespace(planning_mode_enabled=False),
        _emit=_emit,
        _runlog=lambda event, message, **kwargs: runlog.append((event, message, kwargs)),
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
    thinking_tokens = [
        kwargs["token"]
        for event, message, kwargs in runlog
        if event == "model_token" and message == "thinking token"
    ]
    assert thinking_tokens == ["Reason through the greeting.\n"]


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
    assert result["metadata"]["failure_mode"] == "remote_exit_nonzero"
    assert result["metadata"]["ssh_error_class"] == "remote_exit_nonzero"
    assert result["metadata"]["ssh_transport_succeeded"] is True
    assert result["metadata"]["ssh_auth_mode"] == "password"
    assert result["metadata"]["ssh_auth_transport"] == "sshpass_env"
    assert result["metadata"]["ssh_password_provided"] is True
    assert result["metadata"]["output"]["exit_code"] == 1
    assert not any("username/password" in hint.lower() for hint in result["metadata"]["hints"])


def test_ssh_exec_diagnostic_not_found_returns_success() -> None:
    state = LoopState(cwd="/tmp")

    with patch.object(
        network,
        "create_process",
        AsyncMock(return_value=_FakeProc(returncode=1, stderr=b"Unit fog-pxe.service could not be found.\n")),
    ):
        result = asyncio.run(
            network.ssh_exec(
                host="192.168.1.63",
                user="root",
                password="secret",
                command="systemctl status fog-pxe || service fog-pxe status",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is True
    assert result["output"]["exit_code"] == 1


def test_ssh_exec_diagnostic_empty_output_still_fails() -> None:
    """Implicit empty-output probes (e.g. which, grep) still return fail from network.py;
    the verifier handles them separately."""
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
                command="which nonexistent-binary",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is False


def test_ssh_remote_nonzero_memory_label_is_not_unknown_failure() -> None:
    state = LoopState(cwd="/tmp")
    state.thread_id = "thread-ssh-memory"
    state.current_phase = "execute"
    state.active_intent = "requested_ssh_exec"
    state.intent_tags = ["ssh_exec", "phase_execute"]
    state.task_mode = "remote_execute"
    state.run_brief.original_task = "ssh into 192.168.1.63 and inspect a service"
    store = SimpleNamespace(upsert=lambda memory: None)
    harness = SimpleNamespace(
        state=state,
        warm_memory_store=store,
        cold_memory_store=store,
    )
    result = ToolEnvelope(
        success=False,
        error="Remote SSH command exited with code 1",
        metadata={
            "arguments": {"host": "192.168.1.63", "user": "root", "command": "systemctl status missing"},
            "failure_mode": "remote_exit_nonzero",
            "ssh_error_class": "remote_exit_nonzero",
            "ssh_transport_succeeded": True,
        },
    )

    memory = MemoryService(harness).record_experience(tool_name="ssh_exec", result=result)

    assert memory.failure_mode == "remote_exit_nonzero"
    assert "SSH reached the remote host" in memory.notes
    assert "remote command exited non-zero" in memory.notes


def test_ssh_exec_writes_stdin_data_to_remote_command() -> None:
    state = LoopState(cwd="/tmp")
    fake_stdin = _FakeStdin()
    create_process = AsyncMock(return_value=_FakeProc(returncode=0, stdout=b"done\n", stdin=fake_stdin))

    with patch.object(network, "create_process", create_process):
        result = asyncio.run(
            network.ssh_exec(
                host="192.168.1.63",
                user="root",
                password="secret",
                command="cd /root/fogproject/bin && ./installfog.sh",
                stdin_data="n\n\n2\nN\nY\n",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is True
    assert fake_stdin.writes == [b"n\n\n2\nN\nY\n"]
    assert fake_stdin.closed is True
    assert create_process.await_args.kwargs["stdin"] is asyncio.subprocess.PIPE


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


def test_ssh_exec_blocks_likely_foreground_service_command_before_launch() -> None:
    state = LoopState(cwd="/tmp")
    create_process = AsyncMock()

    with patch.object(network, "create_process", create_process):
        result = asyncio.run(
            network.run_ssh_command(
                host="192.168.1.89",
                user="root",
                command="caddy run /etc/caddy/Caddyfile",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "long_running_foreground_command"
    assert result["metadata"]["host"] == "192.168.1.89"
    assert result["metadata"]["foreground_detection"] == "service_foreground_subcommand"
    assert "separate health check" in result["error"]
    create_process.assert_not_awaited()


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


def test_ssh_exec_blocks_harness_tool_name_as_remote_shell_command() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = 'ssh root@192.168.1.89 with password "Temp@Pass" and fix service files'

    tool_name, args, intercepted, _metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.89",
            "user": "root",
            "password": "Temp@Pass",
            "command": (
                "ssh_file_write host='192.168.1.89' path='/lib/systemd/system/mysql.service' "
                "content='bad idea'"
            ),
        },
        phase="execute",
        state=state,
    )

    assert tool_name == "ssh_exec"
    assert args["command"].startswith("ssh_file_write")
    assert intercepted is not None
    assert intercepted.success is False
    assert intercepted.metadata["reason"] == "harness_tool_as_remote_shell_command"
    assert intercepted.metadata["suggested_tool"] == "ssh_file_write"
    assert "Call `ssh_file_write` directly" in (intercepted.error or "")


def test_ssh_exec_blocks_nested_raw_ssh_command() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = 'ssh root@192.168.1.89 with password "Temp@Pass" and run whoami'

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.89",
            "user": "root",
            "password": "Temp@Pass",
            "command": "ssh root@192.168.1.89 -o PreferredAuthentications=password whoami",
        },
        phase="execute",
        state=state,
    )

    assert tool_name == "ssh_exec"
    assert args["command"].startswith("ssh root@192.168.1.89")
    assert intercepted is not None
    assert intercepted.success is False
    assert intercepted.metadata["reason"] == "nested_raw_ssh_in_ssh_exec"
    assert "already opens the SSH connection" in (intercepted.error or "")
    assert metadata == {}
    assert "_ssh_auth_recovery_state" not in state.scratchpad


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


def test_normalize_tool_request_pins_confirmed_ssh_password_over_stale_explicit_password() -> None:
    state = LoopState(cwd=".")
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.89": {
            "host": "192.168.1.89",
            "user": "root",
            "password": "confirmed-password",
            "confirmed": True,
        }
    }

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_file_read",
        {
            "host": "192.168.1.89",
            "user": "root",
            "password": "stale-memory-password",
            "path": "/var/www/turbquant.html",
        },
        phase="execute",
        state=state,
    )

    assert tool_name == "ssh_file_read"
    assert intercepted is None
    assert args["password"] == "confirmed-password"
    assert metadata["pinned_ssh_password"] is True
    assert metadata["pinned_ssh_password_overrode_mismatch"] is True
    assert metadata["pinned_ssh_password_overrode_origin"] == "explicit"


def test_normalize_tool_request_allows_fresh_user_ssh_password_rotation_candidate() -> None:
    state = LoopState(cwd=".")
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.89": {
            "host": "192.168.1.89",
            "user": "root",
            "password": "old-confirmed-password",
            "confirmed": True,
        }
    }
    state.recent_messages.append(
        ConversationMessage(
            role="user",
            content='ssh root@192.168.1.89 with password "fresh-user-password" and read /var/www/turbquant.html',
        )
    )

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_file_read",
        {
            "host": "192.168.1.89",
            "user": "root",
            "password": "fresh-user-password",
            "path": "/var/www/turbquant.html",
        },
        phase="execute",
        state=state,
    )

    assert tool_name == "ssh_file_read"
    assert intercepted is None
    assert args["password"] == "fresh-user-password"
    assert metadata["ssh_credential_rotation_candidate"] is True
    assert "pinned_ssh_password_overrode_mismatch" not in metadata


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


def test_normalize_tool_request_drops_write_session_none_sentinels() -> None:
    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "file_write",
        {
            "path": "./temp/leader_election_sim.py",
            "content": "print('debug')\n",
            "write_session_id": "None",
            "next_section_name": "None",
            "replace_strategy": "overwrite",
            "section_name": "full_file",
        },
        phase="execute",
        state=LoopState(cwd="."),
    )

    assert intercepted is None
    assert tool_name == "file_write"
    assert args["path"] == "./temp/leader_election_sim.py"
    assert args["content"] == "print('debug')\n"
    assert args["replace_strategy"] == "overwrite"
    assert args["section_name"] == "full_file"
    assert "write_session_id" not in args
    assert "next_section_name" not in args
    assert metadata["optional_none_sentinel_removed"] == ["next_section_name", "write_session_id"]


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


def test_normalize_ssh_arguments_accepts_redundant_user_with_target() -> None:
    normalized = network.normalize_ssh_arguments(
        {
            "target": "root@192.168.1.63",
            "user": "root",
            "command": "whoami",
        }
    )

    assert normalized["host"] == "192.168.1.63"
    assert normalized["user"] == "root"


def test_normalize_ssh_arguments_accepts_redundant_host_with_target_user() -> None:
    normalized = network.normalize_ssh_arguments(
        {
            "target": "root@192.168.1.63",
            "host": "192.168.1.63",
            "user": "root",
            "command": "whoami",
        }
    )

    assert normalized["host"] == "192.168.1.63"
    assert normalized["user"] == "root"
    assert "target" not in normalized


def test_normalize_ssh_arguments_rejects_conflicting_target_host() -> None:
    try:
        network.normalize_ssh_arguments(
            {
                "target": "root@192.168.1.63",
                "host": "192.168.1.64",
                "command": "whoami",
            }
        )
    except ValueError as exc:
        assert str(exc) == "Conflicting SSH targets provided via `target` and `host`."
    else:
        raise AssertionError("expected ValueError for conflicting SSH target host")


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


def test_normalize_tool_request_repairs_nested_ssh_exec_arguments() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = 'ssh root@192.168.1.89 with password "Temp@Pass" and cleanup nginx'

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "arguments": {"arg": "#!/bin/bash\necho hello"},
            "command": "#!/bin/bash\necho hello",
            "name": "nginx_cleanup_script",
        },
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["command"] == "#!/bin/bash\necho hello"
    assert args["host"] == "192.168.1.89"
    assert args["user"] == "root"
    assert args["password"] == "Temp@Pass"
    assert "name" not in args
    assert "arguments" not in args
    assert metadata.get("repaired_ssh_exec_nested_args") is True
    assert metadata.get("repaired_ssh_exec_hallucinated_name") is True


def test_normalize_tool_request_recovers_host_from_task_context() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = 'ssh root@192.168.1.99 with password "Secret123" and install docker'

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {"command": "whoami"},
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["host"] == "192.168.1.99"
    assert args["user"] == "root"
    assert args["password"] == "Secret123"
    assert metadata.get("recovered_ssh_host") == "192.168.1.99"


def test_normalize_tool_request_recovers_host_from_at_host_task_context() -> None:
    state = LoopState(cwd=".")
    state.run_brief.original_task = "deploy to admin@10.0.0.5 via ssh"

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {"command": "uptime"},
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "ssh_exec"
    assert args["host"] == "10.0.0.5"
    assert args["user"] == "admin"


def test_store_verifier_verdict_shows_missing_host_placeholder() -> None:
    from smallctl.harness.tool_result_verification import _store_verifier_verdict
    from smallctl.models.tool_result import ToolEnvelope

    state = LoopState(cwd=".")
    result = ToolEnvelope(
        success=False,
        error="SSH target requires either `target` or `host`.",
        metadata={"command": "rm -rf /etc/nginx", "reason": "invalid_ssh_target"},
    )
    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={"command": "rm -rf /etc/nginx"},
    )

    assert verdict is not None
    assert verdict["target"] == "(missing host) :: rm -rf /etc/nginx"
    assert verdict["command"] == "rm -rf /etc/nginx"


def test_build_repair_recovery_message_includes_ssh_schema_hint() -> None:
    from smallctl.graph.tool_execution_recovery_helpers import _build_repair_recovery_message
    from smallctl.graph.state import ToolExecutionRecord
    from smallctl.models.tool_result import ToolEnvelope

    harness = SimpleNamespace(
        state=LoopState(cwd="."),
    )
    harness.state.stagnation_counters = {"repeat_command": 2}
    harness.state.repair_cycle_id = "rc-test"
    record = ToolExecutionRecord(
        operation_id="op1",
        tool_name="ssh_exec",
        args={"arguments": {"arg": "whoami"}, "command": "whoami", "name": "bad"},
        tool_call_id="call_1",
        result=ToolEnvelope(
            success=False,
            error="SSH target requires either `target` or `host`.",
            metadata={"reason": "invalid_ssh_target"},
        ),
    )
    message = _build_repair_recovery_message(harness, record)

    assert "Repair loop stalled" in message
    assert "ssh_exec" in message
    assert "Required ssh_exec fields" in message
    assert "target='root@192.168.1.89'" in message


def test_build_repair_recovery_message_no_schema_hint_when_args_valid() -> None:
    from smallctl.graph.tool_execution_recovery_helpers import _build_repair_recovery_message
    from smallctl.graph.state import ToolExecutionRecord
    from smallctl.models.tool_result import ToolEnvelope

    harness = SimpleNamespace(
        state=LoopState(cwd="."),
    )
    harness.state.stagnation_counters = {"repeat_command": 2}
    record = ToolExecutionRecord(
        operation_id="op1",
        tool_name="ssh_exec",
        args={"target": "root@192.168.1.89", "command": "whoami"},
        tool_call_id="call_1",
        result=ToolEnvelope(
            success=False,
            error="Permission denied",
            metadata={},
        ),
    )
    message = _build_repair_recovery_message(harness, record)

    assert "Repair loop stalled" in message
    assert "Required ssh_exec fields" not in message


def test_build_repair_recovery_message_detects_interactive_ssh_script() -> None:
    from smallctl.graph.tool_execution_recovery_helpers import _build_repair_recovery_message
    from smallctl.graph.state import ToolExecutionRecord
    from smallctl.models.tool_result import ToolEnvelope

    harness = SimpleNamespace(
        state=LoopState(cwd="."),
    )
    harness.state.stagnation_counters = {"repeat_command": 2}
    record = ToolExecutionRecord(
        operation_id="op1",
        tool_name="ssh_exec",
        args={"target": "root@192.168.1.89", "command": "cd /root/fogproject/bin && echo \"y\" | ./installfog.sh"},
        tool_call_id="call_1",
        result=ToolEnvelope(
            success=False,
            error="Remote SSH command exited with code 1",
            metadata={
                "output": {
                    "stdout": "Should the installer try to disable the local firewall for you now? (y/N)\n"
                    "Are you sure you wish to continue (Y/N)\n"
                    "Sorry, answer not recognized",
                    "stderr": "",
                    "exit_code": 1,
                },
            },
        ),
    )
    message = _build_repair_recovery_message(harness, record)

    assert "interactive script prompt" in message
    assert "stdin_data" in message
    assert "Do not retry a single" in message


def test_normalize_tool_request_repairs_patch_argument_aliases() -> None:
    state = LoopState(cwd=".")

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "file_patch",
        {"path": "foo.py", "source": "old", "dest": "new"},
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert tool_name == "file_patch"
    assert args["target_text"] == "old"
    assert args["replacement_text"] == "new"
    assert "source" not in args
    assert "dest" not in args
    assert metadata["argument_alias_repair"] == {"source": "target_text", "dest": "replacement_text"}


def test_normalize_tool_request_does_not_clobber_canonical_patch_args() -> None:
    state = LoopState(cwd=".")

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "file_patch",
        {"path": "foo.py", "target_text": "old", "replacement_text": "new"},
        phase="execute",
        state=state,
    )

    assert intercepted is None
    assert args["target_text"] == "old"
    assert args["replacement_text"] == "new"
    assert "argument_alias_repair" not in metadata


def test_parse_tool_calls_recovers_raw_parameter_tags_for_patch_tools() -> None:
    stream = SimpleNamespace(
        assistant_text=(
            "<tool_call>\n"
            "<function=file_patch>\n"
            "<path>foo.py</path>\n"
            "<source>old</source>\n"
            "<dest>new</dest>\n"
            "</function>\n"
            "</tool_call>\n"
        ),
        thinking_text="",
        tool_calls=[],
    )
    harness = SimpleNamespace(
        registry=SimpleNamespace(names=lambda: {"file_patch", "task_complete"}),
        state=LoopState(cwd="."),
        client=SimpleNamespace(model="gemma-4-4b-it"),
        _runlog=lambda *args, **kwargs: None,
    )

    parse_result = parse_tool_calls(
        stream,
        timeline=[],
        graph_state=SimpleNamespace(run_mode="execute"),
        deps=SimpleNamespace(harness=harness),
        model_name="gemma-4-4b-it",
    )

    assert len(parse_result.pending_tool_calls) == 1
    assert parse_result.pending_tool_calls[0].tool_name == "file_patch"
    assert parse_result.pending_tool_calls[0].args["path"] == "foo.py"
    assert parse_result.pending_tool_calls[0].args["target_text"] == "old"
    assert parse_result.pending_tool_calls[0].args["replacement_text"] == "new"


def test_tool_dispatcher_coerce_args_returns_dropped_keys() -> None:
    from smallctl.tools.dispatcher import ToolDispatcher

    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "target_text": {"type": "string"},
        },
        "required": ["path", "target_text"],
        "additionalProperties": False,
    }
    args, dropped = ToolDispatcher._coerce_args(schema, {"path": "a", "target_text": "b", "source": "c"})

    assert args == {"path": "a", "target_text": "b"}
    assert dropped == ["source"]


def test_tool_dispatcher_validation_includes_dropped_keys_when_required_missing() -> None:
    from smallctl.tools.dispatcher import ToolDispatcher

    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "target_text": {"type": "string"},
        },
        "required": ["path", "target_text"],
        "additionalProperties": False,
    }
    error = ToolDispatcher._validate_args(schema, {"path": "a"})
    assert error is not None
    assert "target_text" in error


def test_remote_scope_guard_allows_shell_exec_after_repeated_ssh_auth_failure() -> None:
    state = LoopState(cwd="/home/stephen/Scripts/Harness-Redo")
    state.task_mode = "remote_execute"
    state.active_intent = "requested_ssh_exec"
    state.scratchpad["_last_task_handoff"] = {
        "task_mode": "remote_execute",
        "effective_task": "ssh into root@192.168.1.63 and fix the remote service",
    }
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    state.tool_execution_records = {
        "rec-1": {
            "tool_name": "ssh_exec",
            "result": {
                "success": False,
                "error": "Permission denied (publickey,password).",
                "metadata": {
                    "output": {
                        "stdout": "",
                        "stderr": "Permission denied (publickey,password).",
                        "exit_code": 255,
                    }
                },
            },
        }
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
    assert intercepted is None


def test_ssh_exec_retries_on_host_key_verification_failure() -> None:
    state = LoopState(cwd="/tmp")
    create_process = AsyncMock(
        side_effect=[
            _FakeProc(
                returncode=255,
                stderr=(
                    b"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\r\n"
                    b"@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\r\n"
                    b"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\r\n"
                    b"Host key verification failed.\n"
                ),
            ),
            _FakeProc(
                returncode=0,
                stdout=b"pihole\n",
            ),
        ]
    )

    with patch.object(network, "create_process", create_process):
        result = asyncio.run(
            network.ssh_exec(
                host="192.168.1.161",
                user="root",
                password="Temp@Pass",
                command="hostname",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is True
    assert result["output"]["stdout"] == "pihole\n"
    assert result["metadata"]["ssh_option_retry"] == "strict_host_key_checking_no"
    assert result["metadata"]["ssh_option_retry_reason"] == "host_key_verification"
    assert create_process.await_count == 2


def test_ssh_exec_host_key_circuit_breaker_blocks_third_attempt() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_ssh_auth_recovery_state"] = {
        "root@192.168.1.161": {
            "host": "192.168.1.161",
            "user": "root",
            "last_error_class": "host_key_verification",
            "consecutive_count": 2,
            "last_command": "hostname",
            "last_error": "Host key verification failed.",
        }
    }

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.161",
            "user": "root",
            "password": "Temp@Pass",
            "command": "hostname",
        },
        phase="execute",
        state=state,
    )

    assert tool_name == "ssh_exec"
    assert intercepted is not None
    assert intercepted.metadata["reason"] == "ssh_host_key_recovery_required"
    assert metadata["ssh_host_key_recovery_required"] is True


def test_ssh_exec_records_host_key_failure_for_circuit_breaker() -> None:
    state = LoopState(cwd="/tmp")

    def _make_proc(**kwargs):
        return _FakeProc(
            returncode=255,
            stderr=(
                b"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\r\n"
                b"@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @\r\n"
                b"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\r\n"
                b"Host key verification failed.\n"
            ),
        )

    create_process = AsyncMock(side_effect=_make_proc)

    with patch.object(network, "create_process", create_process):
        result = asyncio.run(
            network.ssh_exec(
                host="192.168.1.161",
                user="root",
                password="Temp@Pass",
                command="hostname",
                state=state,
                harness=None,
            )
        )

    assert result["success"] is False
    assert result["metadata"]["ssh_error_class"] == "host_key_verification"
    recovery_state = state.scratchpad.get("_ssh_auth_recovery_state", {})
    record = recovery_state.get("root@192.168.1.161", {})
    assert record.get("last_error_class") == "host_key_verification"
    assert record.get("consecutive_count") == 1
    assert record.get("host_key_retry_done") is True


def _ssh_registry():
    return SimpleNamespace(
        get=lambda name: SimpleNamespace(
            phase_allowed=lambda phase: True,
            profile_allowed=lambda profiles: True,
        )
        if name == "ssh_exec"
        else None
    )


def test_normalize_tool_request_allows_ssh_keygen_known_hosts_removal() -> None:
    for command in (
        "ssh-keygen -R 192.168.1.161 -f ~/.ssh/known_hosts",
        "ssh-keygen -f ~/.ssh/known_hosts -R 192.168.1.161",
    ):
        state = LoopState(cwd=".")
        tool_name, args, intercepted, _metadata = normalize_tool_request(
            _ssh_registry(),
            "shell_exec",
            {"command": command},
            phase="execute",
            state=state,
        )

        assert intercepted is None, f"{command} should not be intercepted"
        assert tool_name == "shell_exec"
        assert args["command"] == command


def test_normalize_tool_request_blocks_non_removal_ssh_keygen() -> None:
    state = LoopState(cwd=".")
    tool_name, args, intercepted, _metadata = normalize_tool_request(
        _ssh_registry(),
        "shell_exec",
        {"command": "ssh-keygen -t rsa -f /tmp/key"},
        phase="execute",
        state=state,
    )

    assert intercepted is not None
    assert intercepted.metadata["reason"] == "raw_ssh_shell_blocked"
    assert tool_name == "shell_exec"


def test_normalize_tool_request_blocks_or_rewrites_raw_ssh_family() -> None:
    for command, expected_tool, reason in (
        ("ssh root@192.168.1.161 whoami", "ssh_exec", None),
        ("sshpass -p secret ssh root@192.168.1.161 whoami", "ssh_exec", None),
        ("sftp root@192.168.1.161:/tmp/file", "shell_exec", "raw_ssh_shell_blocked"),
    ):
        state = LoopState(cwd=".")
        tool_name, args, intercepted, _metadata = normalize_tool_request(
            _ssh_registry(),
            "shell_exec",
            {"command": command},
            phase="execute",
            state=state,
        )

        assert tool_name == expected_tool, command
        if reason:
            assert intercepted is not None, command
            assert intercepted.metadata["reason"] == reason
        else:
            assert intercepted is None, command
            assert args.get("host") == "192.168.1.161"


def test_normalize_tool_request_host_key_circuit_breaker_suggests_ssh_keygen() -> None:
    state = LoopState(cwd=".")
    state.scratchpad["_ssh_auth_recovery_state"] = {
        "root@192.168.1.161": {
            "host": "192.168.1.161",
            "user": "root",
            "last_error_class": "host_key_verification",
            "consecutive_count": 2,
            "last_command": "hostname",
            "last_error": "Host key verification failed.",
        }
    }

    tool_name, args, intercepted, metadata = normalize_tool_request(
        SimpleNamespace(get=lambda _name: None),
        "ssh_exec",
        {
            "host": "192.168.1.161",
            "user": "root",
            "password": "Temp@Pass",
            "command": "hostname",
        },
        phase="execute",
        state=state,
    )

    assert tool_name == "ssh_exec"
    assert intercepted is not None
    assert intercepted.metadata["reason"] == "ssh_host_key_recovery_required"
    assert "ssh-keygen -R 192.168.1.161" in intercepted.metadata.get("suggested_command", "")
    assert "shell_exec" in intercepted.metadata["next_required_action"]["tool_names"]
    assert metadata["ssh_host_key_recovery_required"] is True


def test_record_ssh_failure_resets_consecutive_count_on_error_class_change() -> None:
    state = LoopState(cwd="/tmp")
    network._record_ssh_failure(state, "host", "user", "cmd1", "auth_permission_denied", "err")
    network._record_ssh_failure(state, "host", "user", "cmd2", "host_key_verification", "err")

    record = state.scratchpad["_ssh_auth_recovery_state"]["user@host"]
    assert record["consecutive_count"] == 1
    assert record["last_error_class"] == "host_key_verification"

    network._record_ssh_failure(state, "host", "user", "cmd3", "host_key_verification", "err")
    record = state.scratchpad["_ssh_auth_recovery_state"]["user@host"]
    assert record["consecutive_count"] == 2


def test_gemma_call_tag_self_closing_tool_call_is_parsed() -> None:
    text = '<call:task_complete message="Hello! How can I help you today?" />'
    cleaned, calls = _extract_inline_tool_calls(text)
    assert cleaned == ""
    assert len(calls) == 1
    assert calls[0].tool_name == "task_complete"
    assert calls[0].args == {"message": "Hello! How can I help you today?"}


def test_gemma_call_tag_with_multiple_attributes() -> None:
    text = '<call:ssh_exec host="192.168.1.161" user="root" command="hostname" />'
    cleaned, calls = _extract_inline_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].tool_name == "ssh_exec"
    assert calls[0].args == {
        "host": "192.168.1.161",
        "user": "root",
        "command": "hostname",
    }


def test_gemma_direct_xml_terminal_tag_is_parsed() -> None:
    text = '<task_complete message="Hello! I am ready to assist you." />'
    cleaned, calls = _extract_inline_tool_calls(
        text,
        allowed_raw_function_names={"task_complete", "task_fail"},
    )
    assert cleaned == ""
    assert len(calls) == 1
    assert calls[0].tool_name == "task_complete"
    assert calls[0].args == {"message": "Hello! I am ready to assist you."}


def test_gemma_direct_xml_tag_ignores_unknown_tool_names() -> None:
    text = '<unknown_tool message="Hello" />'
    cleaned, calls = _extract_inline_tool_calls(
        text,
        allowed_raw_function_names={"task_complete"},
    )
    assert cleaned == text
    assert calls == []


def test_gemma_direct_xml_tag_parsed_from_mixed_assistant_text() -> None:
    text = (
        "Hello! How can I help you today?\n\n"
        '<task_complete message="Hello! I am ready to assist you." />'
    )
    cleaned, calls = _extract_inline_tool_calls(
        text,
        allowed_raw_function_names={"task_complete", "task_fail"},
    )
    assert cleaned.strip() == "Hello! How can I help you today?"
    assert len(calls) == 1
    assert calls[0].tool_name == "task_complete"
    assert calls[0].args == {"message": "Hello! I am ready to assist you."}


def test_gemma_direct_xml_task_fail_tag_is_parsed() -> None:
    text = '<task_fail message="I cannot help with that." />'
    cleaned, calls = _extract_inline_tool_calls(
        text,
        allowed_raw_function_names={"task_complete", "task_fail"},
    )
    assert cleaned == ""
    assert len(calls) == 1
    assert calls[0].tool_name == "task_fail"
    assert calls[0].args == {"message": "I cannot help with that."}


def test_streaming_gemma_channel_marker_ends_thinking_block() -> None:
    events: list[object] = []
    runlog: list[tuple[str, str, dict[str, object]]] = []

    async def _emit(_handler: object, event: object) -> None:
        events.append(event)

    harness = SimpleNamespace(
        thinking_visibility=True,
        state=SimpleNamespace(planning_mode_enabled=False),
        _emit=_emit,
        _runlog=lambda event, message, **kwargs: runlog.append((event, message, kwargs)),
        _stream_print=lambda _text: None,
    )
    deps = SimpleNamespace(event_handler=object())
    state = StreamTagState()
    first_token_time = None
    chunks: list[dict[str, object]] = []

    # Split <think> across chunks, matching real Gemma streams.
    stream_events = [
        {"data": {"choices": [{"delta": {"content": "<think"}}]}},
        {"data": {"choices": [{"delta": {"content": ">"}}]}},
        {"data": {"choices": [{"delta": {"content": "\n"}}]}},
        {"data": {"choices": [{"delta": {"content": "<channel|>"}}]}},
        {"data": {"choices": [{"delta": {"content": "Hello!"}}]}},
    ]

    for event in stream_events:
        state, first_token_time = asyncio.run(
            handle_model_stream_chunk(
                harness=harness,
                deps=deps,
                event=event,
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

    assert assistant_text == "Hello!"
    assert thinking_text == "\n"
    assert "<channel|>" not in assistant_text
