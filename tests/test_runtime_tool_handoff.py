from __future__ import annotations

import asyncio
import json
from pathlib import Path

from smallctl.graph.runtime_auto import AutoGraphRuntime
from smallctl.graph.runtime import ChatGraphRuntime, LoopGraphRuntime
from smallctl.harness import Harness


def _tool_call_stream(*, tool_name: str, args: dict[str, object], tool_call_id: str) -> list[dict[str, object]]:
    return [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "reasoning_content": f"Calling {tool_name}.\n",
                        },
                        "finish_reason": None,
                    }
                ]
            },
        },
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
        },
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "type": "function",
                                    "function": {
                                        "arguments": json.dumps(args),
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
        },
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
        },
        {"type": "done"},
    ]


def test_loop_runtime_continues_after_successful_tool_dispatch(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "visible.txt").write_text("hello\n", encoding="utf-8")

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    stream_sequences = [
        _tool_call_stream(
            tool_name="dir_list",
            args={"path": str(tmp_path)},
            tool_call_id="tool-1",
        ),
        _tool_call_stream(
            tool_name="task_complete",
            args={"message": "Listed the directory."},
            tool_call_id="tool-2",
        ),
    ]

    async def fake_stream_chat(*, messages, tools):
        del messages, tools
        if not stream_sequences:
            raise AssertionError("unexpected extra model call")
        for event in stream_sequences.pop(0):
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = LoopGraphRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("use ls to list the files you can see in this dir"),
            timeout=5,
        )
    )

    assert result["status"] == "completed"
    assert "Listed the directory." in json.dumps(result)
    assert stream_sequences == []


def test_run_task_with_events_defers_task_boundary_reset_to_graph_lifecycle(monkeypatch) -> None:
    reset_calls: list[tuple[str, str]] = []
    runtime_calls: list[str] = []

    class _StubRuntime:
        async def run(self, task: str) -> dict[str, object]:
            runtime_calls.append(task)
            return {"status": "completed", "assistant": task}

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        lambda harness, event_handler=None: _StubRuntime(),
    )

    harness = type("HarnessStub", (), {})()
    harness.event_handler = None
    harness._reset_task_boundary_state = lambda **kwargs: reset_calls.append(
        (str(kwargs.get("reason") or ""), str(kwargs.get("new_task") or ""))
    )

    result = asyncio.run(Harness.run_task_with_events(harness, "start with 1"))

    assert result["status"] == "completed"
    assert runtime_calls == ["start with 1"]
    assert reset_calls == []


def test_loop_runtime_uses_task_complete_message_as_assistant_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    async def fake_stream_chat(*, messages, tools):
        del messages, tools
        for event in _tool_call_stream(
            tool_name="task_complete",
            args={"message": "Jacksonville is 85F today."},
            tool_call_id="tool-terminal-answer",
        ):
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = LoopGraphRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("do a websearch and tell me today's weather"),
            timeout=5,
        )
    )

    assert result["status"] == "completed"
    assert result["assistant"] == "Jacksonville is 85F today."


def test_auto_runtime_capability_query_routes_without_mode_model_fallback(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    stream_calls: list[tuple[list[dict[str, object]], list[dict[str, object]]]] = []

    async def fake_stream_chat(*, messages, tools):
        stream_calls.append((messages, tools))
        for event in _tool_call_stream(
            tool_name="task_complete",
            args={"message": "Available tools can be inspected from the current runtime."},
            tool_call_id="tool-capability-1",
        ):
            yield event

    runlog: list[tuple[str, dict[str, object]]] = []
    harness.client.stream_chat = fake_stream_chat
    harness._runlog = lambda event, _message, **data: runlog.append((event, data))
    runtime = AutoGraphRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("what tools do you have access to?"),
            timeout=5,
        )
    )

    assert result["status"] == "completed"
    assert "Available tools can be inspected" in json.dumps(result)
    assert len(stream_calls) == 1
    assert any(
        event == "mode_decision" and data.get("mode") == "loop" and data.get("intent") == "capability_query"
        for event, data in runlog
    )


def test_chat_runtime_capability_query_keeps_real_tools_when_forced_to_chat(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    stream_calls: list[tuple[list[dict[str, object]], list[dict[str, object]]]] = []

    async def fake_stream_chat(*, messages, tools):
        stream_calls.append((messages, tools))
        for event in _tool_call_stream(
            tool_name="task_complete",
            args={"message": "Chat-mode capability inspection completed."},
            tool_call_id="tool-chat-capability-1",
        ):
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = ChatGraphRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("what mode are you in?"),
            timeout=5,
        )
    )

    assert result["status"] == "chat_completed"
    assert "Chat-mode capability inspection completed." in json.dumps(result)
    assert len(stream_calls) == 1
    _, tools = stream_calls[0]
    tool_names = [
        str(tool["function"]["name"])
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ]
    assert "file_read" in tool_names
    assert harness.state.scratchpad["_chat_runtime_intent"] == "capability_query"
    assert "_chat_tools_suppressed_reason" not in harness.state.scratchpad

    system_prompt = next(
        str(message.get("content") or "")
        for message in stream_calls[0][0]
        if isinstance(message, dict) and message.get("role") == "system"
    )
    assert "Available tools on this turn:" in system_prompt
    assert "file_read" in system_prompt
