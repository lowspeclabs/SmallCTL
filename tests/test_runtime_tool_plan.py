from __future__ import annotations

import asyncio
import json
from pathlib import Path

from smallctl.graph.runtime_tool_plan import ToolPlanRuntime
from smallctl.harness import Harness


def _content_stream(text: str, *, usage: dict[str, object] | None = None) -> list[dict[str, object]]:
    return [
        {
            "type": "chunk",
            "data": {
                **({"usage": usage} if usage else {}),
                "choices": [
                    {
                        "delta": {"role": "assistant", "content": text},
                        "finish_reason": None,
                    }
                ]
            },
        }
    ]


def _tool_call_stream(
    tool_call_id: str,
    tool_name: str,
    args: dict[str, object],
    *,
    usage: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "type": "chunk",
            "data": {
                **({"usage": usage} if usage else {}),
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
                                        "arguments": json.dumps(args),
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
        }
    ]


def test_tool_plan_runtime_executes_plan_and_injects_observations(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("def dispatch_tools():\n    return 'ok'\n", encoding="utf-8")

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        tool_plan_max_repair_attempts=0,
    )
    planner_response = json.dumps(
        {
            "mode": "tool_plan",
            "objective": "locate dispatch seam",
            "steps": [
                {
                    "id": "E1",
                    "tool": "file_read",
                    "args": {"path": "src/app.py"},
                    "reason": "read the target file",
                }
            ],
        }
    )
    captured_prompts: list[list[dict[str, object]]] = []
    captured_tool_counts: list[int] = []
    responses = [
        _content_stream(planner_response, usage={"prompt_tokens": 20, "completion_tokens": 10}),
        _tool_call_stream(
            "solver-complete",
            "task_complete",
            {"message": "The dispatch seam is in src/app.py."},
            usage={"total_tokens": 45},
        ),
    ]

    async def fake_stream_chat(*, messages, tools):
        captured_prompts.append(list(messages))
        captured_tool_counts.append(len(tools))
        if not responses:
            raise AssertionError("unexpected extra model call")
        for event in responses.pop(0):
            yield event

    harness.client.stream_chat = fake_stream_chat

    result = asyncio.run(asyncio.wait_for(ToolPlanRuntime.from_harness(harness).run("find dispatch seam"), timeout=10))

    assert result["status"] == "completed"
    assert captured_tool_counts[0] == 0
    assert captured_tool_counts[1] > 0
    solver_prompt_text = "\n".join(str(message.get("content") or "") for message in captured_prompts[1])
    assert "TOOL PLAN OBSERVATIONS" in solver_prompt_text
    assert "E1 file_read src/app.py" in solver_prompt_text
    assert not any(
        message.get("role") == "tool" and message.get("name") == "file_read"
        for message in captured_prompts[1]
    )
    assert any(
        str(record.get("tool_name") or "") == "file_read"
        for record in harness.state.tool_execution_records.values()
    )
    assert any(
        record.get("hidden_from_prompt") is True
        for record in harness.state.tool_execution_records.values()
        if str(record.get("tool_name") or "") == "file_read"
    )
    metrics = harness.state.scratchpad["_recovery_metrics"]
    assert metrics["tool_plan_invocations"] == 1
    assert metrics["tool_plan_steps_requested"] == 1
    assert metrics["tool_plan_steps_executed"] == 1
    assert metrics["tool_plan_step_failures"] == 0
    assert metrics["tool_plan_observation_tokens"] > 0
    assert metrics["tool_plan_planner_tokens"] == 30
    assert metrics["tool_plan_solver_tokens"] == 45
    assert metrics["tool_plan_total_tokens"] == 75
    active = harness.state.subtask_ledger.active()
    assert active is not None
    assert any(item.startswith("ToolPlan observations:") for item in active.evidence)
    assert harness.state.scratchpad["_tool_plan_evidence_ids"] == ["TP-E0-E1"]
    assert any(record.kind == "tool_plan_observation" for record in harness.state.reasoning_graph.evidence_records)


def test_tool_plan_runtime_uses_rewoo_planner_and_solver_frames(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("def dispatch_tools():\n    return 'ok'\n", encoding="utf-8")

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        tool_plan_max_repair_attempts=0,
        rewoo_planner_frame_enabled=True,
        rewoo_solver_frame_enabled=True,
    )
    harness.state.working_memory.failures = ["Earlier read used the wrong directory"]
    harness.state.working_memory.open_questions = ["Which file owns dispatch_tools?"]
    planner_response = json.dumps(
        {
            "mode": "tool_plan",
            "objective": "locate dispatch seam",
            "steps": [
                {
                    "id": "E1",
                    "tool": "file_read",
                    "args": {"path": "src/app.py"},
                    "reason": "read the target file",
                }
            ],
        }
    )
    captured_prompts: list[list[dict[str, object]]] = []
    responses = [
        _content_stream(planner_response),
        _tool_call_stream("solver-complete", "task_complete", {"message": "The dispatch seam is in src/app.py."}),
    ]

    async def fake_stream_chat(*, messages, tools):
        del tools
        captured_prompts.append(list(messages))
        if not responses:
            raise AssertionError("unexpected extra model call")
        for event in responses.pop(0):
            yield event

    harness.client.stream_chat = fake_stream_chat

    result = asyncio.run(asyncio.wait_for(ToolPlanRuntime.from_harness(harness).run("find dispatch seam"), timeout=10))

    assert result["status"] == "completed"
    planner_prompt_text = "\n".join(str(message.get("content") or "") for message in captured_prompts[0])
    assert "REWOO PLAN STATE" in planner_prompt_text
    assert "Earlier read used the wrong directory" in planner_prompt_text
    assert "Which file owns dispatch_tools?" in planner_prompt_text
    assert "TOOL PLAN OBSERVATIONS" not in planner_prompt_text
    solver_prompt_text = "\n".join(str(message.get("content") or "") for message in captured_prompts[1])
    assert "REWOO EVIDENCE" in solver_prompt_text
    assert "TP-E0-E1" in solver_prompt_text
    assert "TOOL PLAN OBSERVATIONS" not in solver_prompt_text


def test_tool_plan_runtime_invalid_plan_falls_back_to_loop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        tool_plan_max_repair_attempts=0,
    )
    captured_prompts: list[list[dict[str, object]]] = []
    responses = [
        _content_stream("not json"),
        _tool_call_stream("fallback-complete", "task_complete", {"message": "Fallback loop response."}),
    ]

    async def fake_stream_chat(*, messages, tools):
        del tools
        captured_prompts.append(list(messages))
        if not responses:
            raise AssertionError("unexpected extra model call")
        for event in responses.pop(0):
            yield event

    harness.client.stream_chat = fake_stream_chat

    result = asyncio.run(asyncio.wait_for(ToolPlanRuntime.from_harness(harness).run("find dispatch seam"), timeout=10))

    assert result["status"] == "completed"
    fallback_prompt_text = "\n".join(str(message.get("content") or "") for message in captured_prompts[1])
    assert "ToolPlan planner did not return valid bounded JSON" in fallback_prompt_text
    metrics = harness.state.scratchpad["_recovery_metrics"]
    assert metrics["tool_plan_parse_failures"] == 1
    assert metrics["tool_plan_fallback_count"] == 1
    assert harness.state.failure_events[-1].failure_class == "tool_plan_invalid"
    assert harness.state.reflexion_memory[-1].failure_class == "tool_plan_invalid"


def test_tool_plan_runtime_unsafe_plan_is_not_dispatched(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        tool_plan_max_repair_attempts=0,
    )
    planner_response = json.dumps(
        {
            "mode": "tool_plan",
            "objective": "unsafe read",
            "steps": [
                {
                    "id": "E1",
                    "tool": "file_read",
                    "args": {"path": "/etc/passwd"},
                    "reason": "absolute paths are not allowed",
                }
            ],
        }
    )
    captured_prompts: list[list[dict[str, object]]] = []
    responses = [
        _content_stream(planner_response),
        _tool_call_stream("fallback-complete", "task_complete", {"message": "Fallback after unsafe ToolPlan."}),
    ]

    async def fake_stream_chat(*, messages, tools):
        del tools
        captured_prompts.append(list(messages))
        if not responses:
            raise AssertionError("unexpected extra model call")
        for event in responses.pop(0):
            yield event

    harness.client.stream_chat = fake_stream_chat

    result = asyncio.run(asyncio.wait_for(ToolPlanRuntime.from_harness(harness).run("read unsafe file"), timeout=10))

    assert result["status"] == "completed"
    assert not any(
        str(record.get("tool_name") or "") == "file_read"
        for record in harness.state.tool_execution_records.values()
    )
    fallback_prompt_text = "\n".join(str(message.get("content") or "") for message in captured_prompts[1])
    assert "ToolPlan rejected unsafe evidence steps" in fallback_prompt_text
    metrics = harness.state.scratchpad["_recovery_metrics"]
    assert metrics["tool_plan_unsafe_steps_blocked"] == 1
    assert metrics["tool_plan_fallback_count"] == 1
    assert harness.state.failure_events[-1].failure_class == "tool_plan_unsafe"


def test_tool_plan_runtime_repairs_invalid_planner_output_once(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "src" / "app.py"
    source.parent.mkdir()
    source.write_text("def dispatch_tools():\n    return 'ok'\n", encoding="utf-8")
    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
        tool_plan_max_repair_attempts=1,
    )
    repaired_plan = json.dumps(
        {
            "mode": "tool_plan",
            "objective": "locate dispatch seam",
            "steps": [
                {
                    "id": "E1",
                    "tool": "file_read",
                    "args": {"path": "src/app.py"},
                    "reason": "read the target file",
                }
            ],
        }
    )
    captured_prompts: list[list[dict[str, object]]] = []
    responses = [
        _content_stream("not json"),
        _content_stream(repaired_plan),
        _tool_call_stream("solver-complete", "task_complete", {"message": "The dispatch seam is in src/app.py."}),
    ]

    async def fake_stream_chat(*, messages, tools):
        del tools
        captured_prompts.append(list(messages))
        if not responses:
            raise AssertionError("unexpected extra model call")
        for event in responses.pop(0):
            yield event

    harness.client.stream_chat = fake_stream_chat

    result = asyncio.run(asyncio.wait_for(ToolPlanRuntime.from_harness(harness).run("find dispatch seam"), timeout=10))

    assert result["status"] == "completed"
    assert "Repair previous invalid ToolPlan output" in str(captured_prompts[1][0]["content"])
    metrics = harness.state.scratchpad["_recovery_metrics"]
    assert metrics["tool_plan_parse_failures"] == 1
    assert metrics["tool_plan_steps_executed"] == 1
