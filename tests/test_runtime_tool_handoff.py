from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from smallctl.graph.runtime_auto import AutoGraphRuntime
from smallctl.graph.runtime import ChatGraphRuntime, LoopGraphRuntime
from smallctl.harness import Harness
from smallctl.state import ExecutionPlan


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


def _assistant_text_stream(text: str) -> list[dict[str, object]]:
    return [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "role": "assistant",
                            "content": text,
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
                        "finish_reason": "stop",
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


def test_run_task_with_events_uses_staged_runtime_for_approved_plan(monkeypatch) -> None:
    loop_calls: list[str] = []
    staged_calls: list[str] = []

    class _LoopRuntime:
        async def run(self, task: str) -> dict[str, object]:
            loop_calls.append(task)
            return {"status": "loop"}

    class _StagedRuntime:
        async def run(self, task: str) -> dict[str, object]:
            staged_calls.append(task)
            return {"status": "staged"}

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        lambda harness, event_handler=None: _LoopRuntime(),
    )
    monkeypatch.setattr(
        "smallctl.graph.runtime_staged.StagedExecutionRuntime.from_harness",
        lambda harness, event_handler=None: _StagedRuntime(),
    )

    plan = ExecutionPlan(plan_id="plan-1", goal="patch the thing")
    plan.approved = True
    harness = type("HarnessStub", (), {})()
    harness.event_handler = None
    harness.config = SimpleNamespace(staged_execution_enabled=True)
    harness.state = SimpleNamespace(
        active_plan=plan,
        draft_plan=None,
        planning_mode_enabled=False,
        pending_interrupt=None,
    )

    result = asyncio.run(Harness.run_task_with_events(harness, "continue"))

    assert result["status"] == "staged"
    assert staged_calls == ["continue"]
    assert loop_calls == []


def test_resume_task_with_events_uses_staged_runtime_for_blocked_step(monkeypatch) -> None:
    loop_resumes: list[str] = []
    staged_resumes: list[str] = []

    class _LoopRuntime:
        async def resume(self, human_input: str) -> dict[str, object]:
            loop_resumes.append(human_input)
            return {"status": "loop"}

    class _StagedRuntime:
        async def resume(self, human_input: str) -> dict[str, object]:
            staged_resumes.append(human_input)
            return {"status": "staged"}

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        lambda harness, event_handler=None: _LoopRuntime(),
    )
    monkeypatch.setattr(
        "smallctl.graph.runtime_staged.StagedExecutionRuntime.from_harness",
        lambda harness, event_handler=None: _StagedRuntime(),
    )

    plan = ExecutionPlan(plan_id="plan-1", goal="patch the thing")
    plan.approved = True
    interrupt = {
        "kind": "staged_step_blocked",
        "question": "Step S1 is blocked",
        "response_mode": "revise/skip/retry",
    }
    harness = type("HarnessStub", (), {})()
    harness.event_handler = None
    harness.config = SimpleNamespace(staged_execution_enabled=True)
    harness.state = SimpleNamespace(
        active_plan=plan,
        draft_plan=None,
        plan_execution_mode=True,
        pending_interrupt=interrupt,
    )
    harness.get_pending_interrupt = lambda: interrupt

    result = asyncio.run(Harness.resume_task_with_events(harness, "retry"))

    assert result["status"] == "staged"
    assert staged_resumes == ["retry"]
    assert loop_resumes == []


def test_auto_runtime_resumes_staged_interrupt_with_staged_runtime(monkeypatch) -> None:
    loop_resumes: list[str] = []
    staged_resumes: list[str] = []

    class _LoopRuntime:
        async def resume(self, human_input: str) -> dict[str, object]:
            loop_resumes.append(human_input)
            return {"status": "loop"}

    class _StagedRuntime:
        async def resume(self, human_input: str) -> dict[str, object]:
            staged_resumes.append(human_input)
            return {"status": "staged"}

    monkeypatch.setattr(
        "smallctl.graph.runtime.LoopGraphRuntime.from_harness",
        lambda harness, event_handler=None: _LoopRuntime(),
    )
    monkeypatch.setattr(
        "smallctl.graph.runtime_staged.StagedExecutionRuntime.from_harness",
        lambda harness, event_handler=None: _StagedRuntime(),
    )

    plan = ExecutionPlan(plan_id="plan-1", goal="patch the thing")
    plan.approved = True
    interrupt = {
        "kind": "staged_step_blocked",
        "question": "Step S1 is blocked",
        "response_mode": "revise/skip/retry",
    }
    harness = SimpleNamespace(
        config=SimpleNamespace(staged_execution_enabled=True, run_mode="auto"),
        state=SimpleNamespace(
            active_plan=plan,
            draft_plan=None,
            plan_execution_mode=True,
            pending_interrupt=interrupt,
        ),
        has_pending_interrupt=lambda: True,
        get_pending_interrupt=lambda: interrupt,
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(AutoGraphRuntime.from_harness(harness).run("retry"))

    assert result["status"] == "staged"
    assert staged_resumes == ["retry"]
    assert loop_resumes == []


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


def test_loop_runtime_promotes_terminal_prose_task_complete_after_tool_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "visible.txt").write_text("hello\n", encoding="utf-8")

    harness = Harness(
        endpoint="http://example.test/v1",
        model="Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled",
        provider_profile="llamacpp",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    final_answer = (
        "## Final Summary\n\n"
        "- visible.txt exists in the directory.\n\n"
        "**Task Complete**"
    )
    stream_sequences = [
        _tool_call_stream(
            tool_name="dir_list",
            args={"path": str(tmp_path)},
            tool_call_id="tool-list-before-terminal-prose",
        ),
        _assistant_text_stream(final_answer),
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
            runtime.run("list files and summarize the findings"),
            timeout=5,
        )
    )

    assert result["status"] == "completed"
    assert result["assistant"] == final_answer
    assert stream_sequences == []
    assert harness.state.scratchpad["_terminal_prose_task_complete_autopromoted"]["recovery_kind"] == (
        "terminal_prose_task_complete"
    )


def test_loop_runtime_promotes_raw_task_complete_json_after_tool_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "visible.txt").write_text("hello\n", encoding="utf-8")

    harness = Harness(
        endpoint="http://example.test/v1",
        model="Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled",
        provider_profile="llamacpp",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    final_answer = (
        "visible.txt exists.\n\n"
        "```json\n"
        "{\"task_complete\": \"Listed the directory and found visible.txt.\"}\n"
        "```"
    )
    stream_sequences = [
        _tool_call_stream(
            tool_name="dir_list",
            args={"path": str(tmp_path)},
            tool_call_id="tool-list-before-raw-terminal-json",
        ),
        _assistant_text_stream(final_answer),
    ]
    observed_prompts: list[list[dict[str, object]]] = []

    async def fake_stream_chat(*, messages, tools):
        del tools
        observed_prompts.append(messages)
        if not stream_sequences:
            raise AssertionError("unexpected extra model call")
        for event in stream_sequences.pop(0):
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = LoopGraphRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("list files and summarize the findings"),
            timeout=5,
        )
    )

    assert result["status"] == "completed"
    assert result["assistant"] == final_answer
    assert stream_sequences == []
    assert harness.state.scratchpad["_terminal_prose_task_complete_autopromoted"]["message_preview"] == (
        "Listed the directory and found visible.txt."
    )
    assert not any(
        message.get("metadata", {}).get("recovery_kind") == "missing_task_complete"
        for prompt in observed_prompts
        for message in prompt
    )


def test_loop_runtime_promotes_readonly_answer_after_missing_complete_nudge(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pong.py").write_text("print('pong')\n", encoding="utf-8")

    harness = Harness(
        endpoint="http://example.test/v1",
        model="Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled",
        provider_profile="llamacpp",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    first_answer = (
        "I read `pong.py` and found several improvements:\n\n"
        "1. Extract the game loop into a helper so replay does not duplicate control flow.\n"
        "2. Separate rendering from state updates so movement and scoring can be tested.\n"
        "3. Add pause-state rendering so the user gets visible feedback."
    )
    second_answer = (
        "Improvements for `pong.py`:\n\n"
        "1. Extract the duplicated play loop into a `play_round()` helper.\n"
        "2. Separate game-state updates from curses rendering for easier tests.\n"
        "3. Add a visible paused overlay and clarify the right-paddle control text."
    )
    stream_sequences = [
        _tool_call_stream(
            tool_name="file_read",
            args={"path": str(tmp_path / "pong.py")},
            tool_call_id="tool-read-before-readonly-answer",
        ),
        _assistant_text_stream(first_answer),
        _assistant_text_stream(second_answer),
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
            runtime.run("read ./pong.py and list improvements you would make to that file"),
            timeout=5,
        )
    )

    assert result["status"] == "completed"
    assert result["assistant"] == second_answer
    assert stream_sequences == []
    assert harness.state.scratchpad["_terminal_prose_task_complete_autopromoted"]["message_preview"] == second_answer


def test_loop_runtime_does_not_promote_plain_nonterminal_summary(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled",
        provider_profile="llamacpp",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    stream_sequences = [
        _assistant_text_stream("I found the directory has several files and can inspect them next."),
        _tool_call_stream(
            tool_name="task_complete",
            args={"message": "Finished after the recovery nudge."},
            tool_call_id="tool-complete-after-nudge",
        ),
    ]
    observed_prompts: list[list[dict[str, object]]] = []

    async def fake_stream_chat(*, messages, tools):
        del tools
        observed_prompts.append(messages)
        if not stream_sequences:
            raise AssertionError("unexpected extra model call")
        for event in stream_sequences.pop(0):
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = LoopGraphRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("inspect the directory"),
            timeout=5,
        )
    )

    assert result["status"] == "completed"
    assert stream_sequences == []
    assert "_terminal_prose_task_complete_autopromoted" not in harness.state.scratchpad
    assert any(
        message.get("metadata", {}).get("recovery_kind") == "missing_task_complete"
        for message in observed_prompts[-1]
    )


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
        event == "mode_decision"
        and data.get("selected_mode") == "loop"
        and data.get("intent") == "capability_query"
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


def test_auto_chat_smalltalk_sends_no_tools_to_qwen35(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="qwen3.5:4b",
        provider_profile="lmstudio",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    stream_calls: list[tuple[list[dict[str, object]], list[dict[str, object]]]] = []

    async def fake_stream_chat(*, messages, tools):
        stream_calls.append((messages, tools))
        yield {
            "type": "chunk",
            "data": {"choices": [{"delta": {"content": "Hello! How can I help?"}}]},
        }
        yield {"type": "done"}

    harness.client.stream_chat = fake_stream_chat

    result = asyncio.run(
        asyncio.wait_for(
            AutoGraphRuntime.from_harness(harness).run("hello"),
            timeout=5,
        )
    )

    assert result["status"] == "chat_completed"
    assert result["assistant"] == "Hello! How can I help?"
    assert len(stream_calls) == 1
    assert [
        tool["function"]["name"]
        for tool in stream_calls[0][1]
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    ] == ["task_complete", "task_fail"]
    assert harness.state.scratchpad["_chat_runtime_intent"] == "smalltalk"
    assert harness.state.scratchpad["_chat_tools_suppressed_reason"] == "smalltalk_terminal_only"


def test_chat_runtime_promotes_fenced_task_complete_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="qwen3.5:4b",
        provider_profile="lmstudio",
        phase="explore",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )

    async def fake_stream_chat(*, messages, tools):
        yield {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                "```json\n"
                                '{"task_complete": {"message": "Task marked complete from chat JSON."}}\n'
                                "```"
                            )
                        }
                    }
                ]
            },
        }
        yield {"type": "done"}

    harness.client.stream_chat = fake_stream_chat

    result = asyncio.run(
        asyncio.wait_for(
            ChatGraphRuntime.from_harness(harness).run("good job"),
            timeout=5,
        )
    )

    assert result["status"] == "chat_completed"
    assert result["message"] == "Task marked complete from chat JSON."
    assert (
        harness.state.scratchpad["_terminal_json_task_complete_autopromoted"]["recovery_kind"]
        == "terminal_json_task_complete"
    )
