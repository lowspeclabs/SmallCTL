from __future__ import annotations

import asyncio
import json
from pathlib import Path

from smallctl.graph.runtime import LoopGraphRuntime
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
