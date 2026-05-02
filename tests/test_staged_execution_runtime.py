from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from smallctl.graph.runtime_staged import StagedExecutionRuntime
from smallctl.harness import Harness
from smallctl.state import ExecutionPlan, LoopState, PlanStep, StepEvidenceArtifact, StepVerifierSpec


def _step_complete_stream(tool_call_id: str, message: str = "step done") -> list[dict[str, object]]:
    return [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {"role": "assistant", "reasoning_content": "Completing step.\n"},
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
                                        "name": "step_complete",
                                        "arguments": json.dumps({"message": message}),
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


def test_three_step_plan_runs_in_dependency_order(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )
    harness.config.staged_execution_enabled = True
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="three step goal",
        approved=True,
        steps=[
            PlanStep(step_id="S1", title="first", task="do first"),
            PlanStep(step_id="S2", title="second", task="do second", depends_on=["S1"]),
            PlanStep(step_id="S3", title="third", task="do third", depends_on=["S2"]),
        ],
    )

    call_count = 0
    stream_sequences = [
        _step_complete_stream("tc-1", "s1 done"),
        _step_complete_stream("tc-2", "s2 done"),
        _step_complete_stream("tc-3", "s3 done"),
    ]

    async def fake_stream_chat(*, messages, tools):
        del messages, tools
        nonlocal call_count
        call_count += 1
        if call_count > len(stream_sequences):
            raise AssertionError(f"unexpected extra model call #{call_count}")
        for event in stream_sequences[call_count - 1]:
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = StagedExecutionRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("execute staged plan"),
            timeout=10,
        )
    )

    assert result["status"] == "complete"
    assert result["plan_id"] == "plan-1"
    assert call_count == 3
    assert harness.state.step_evidence["S1"].summary == "s1 done"
    assert harness.state.step_evidence["S2"].summary == "s2 done"
    assert harness.state.step_evidence["S3"].summary == "s3 done"


def test_completion_gate_blocks_premature_advancement(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )
    harness.config.staged_execution_enabled = True
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="first",
                verifiers=[
                    StepVerifierSpec(kind="file_exists", args={"path": "missing.txt"}, required=True),
                ],
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _step_complete_stream("tc-1", "attempt 1"),
        _step_complete_stream("tc-2", "attempt 2"),
        _step_complete_stream("tc-3", "attempt 3"),
    ]

    async def fake_stream_chat(*, messages, tools):
        del messages, tools
        nonlocal call_count
        call_count += 1
        if call_count > len(stream_sequences):
            raise AssertionError(f"unexpected extra model call #{call_count}")
        for event in stream_sequences[call_count - 1]:
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = StagedExecutionRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("execute staged plan"),
            timeout=10,
        )
    )

    # Verification fails each time; after max_retries (default 3) the step blocks
    assert result["status"] == "needs_human"
    step = harness.state.active_plan.find_step("S1")
    assert step.status == "blocked"
    assert step.retry_count == 3
    assert harness.state.pending_interrupt["kind"] == "staged_step_blocked"


def test_dependent_step_sees_prior_evidence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )
    harness.config.staged_execution_enabled = True
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        approved=True,
        steps=[
            PlanStep(step_id="S1", title="first"),
            PlanStep(step_id="S2", title="second", depends_on=["S1"]),
        ],
    )

    call_count = 0
    stream_sequences = [
        _step_complete_stream("tc-1", "s1 done"),
        _step_complete_stream("tc-2", "s2 done"),
    ]
    captured_prompts: list[list[dict[str, object]]] = []

    async def fake_stream_chat(*, messages, tools):
        del tools
        nonlocal call_count
        call_count += 1
        captured_prompts.append(list(messages))
        for event in stream_sequences[call_count - 1]:
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = StagedExecutionRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("execute staged plan"),
            timeout=10,
        )
    )

    assert result["status"] == "complete"
    # Second step prompt should include evidence from S1
    second_prompt = "\n".join(
        str(m.get("content", "")) for m in captured_prompts[1]
    )
    assert "s1 done" in second_prompt or "S1" in second_prompt


def test_interrupt_resume_preserves_active_step(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )
    harness.config.staged_execution_enabled = True
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="first",
                max_retries=5,
                verifiers=[
                    StepVerifierSpec(kind="file_exists", args={"path": "missing.txt"}, required=True),
                ],
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _step_complete_stream("tc-1", "attempt 1"),
        _step_complete_stream("tc-2", "attempt 2"),
        _step_complete_stream("tc-3", "attempt 3"),
        _step_complete_stream("tc-4", "attempt 4"),
        _step_complete_stream("tc-5", "attempt 5"),
    ]

    async def fake_stream_chat(*, messages, tools):
        del messages, tools
        nonlocal call_count
        call_count += 1
        if call_count > len(stream_sequences):
            raise AssertionError(f"unexpected extra model call #{call_count}")
        for event in stream_sequences[call_count - 1]:
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = StagedExecutionRuntime.from_harness(harness)

    # Step will fail verification because file doesn't exist, but with max_retries=5
    # it stays pending after each failure until all retries are exhausted.
    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("execute staged plan"),
            timeout=10,
        )
    )

    # After 5 failed attempts, step should be blocked (retry_count == max_retries)
    assert result["status"] == "needs_human"
    step = harness.state.active_plan.find_step("S1")
    assert step.status == "blocked"
    assert step.retry_count == 5
    assert harness.state.active_step_id == ""
    assert harness.state.active_step_run_id == ""

    # Test that resume("retry") resets the blocked step to pending
    runtime._apply_blocked_step_resume_choice("retry")
    assert step.status == "pending"
    assert harness.state.pending_interrupt is None


def test_finalization_only_after_all_required_steps_complete(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="wrench-9b",
        provider_profile="lmstudio",
        phase="execute",
        api_key="test-key",
        runtime_context_probe=False,
        graph_checkpointer="memory",
    )
    harness.config.staged_execution_enabled = True
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="goal",
        approved=True,
        steps=[
            PlanStep(step_id="S1", title="first"),
            PlanStep(step_id="S2", title="second", depends_on=["S1"]),
            PlanStep(step_id="S3", title="third", depends_on=["S2"]),
        ],
    )

    call_count = 0
    stream_sequences = [
        _step_complete_stream("tc-1", "s1 done"),
        _step_complete_stream("tc-2", "s2 done"),
        _step_complete_stream("tc-3", "s3 done"),
    ]

    async def fake_stream_chat(*, messages, tools):
        del messages, tools
        nonlocal call_count
        call_count += 1
        if call_count > len(stream_sequences):
            raise AssertionError(f"unexpected extra model call #{call_count}")
        for event in stream_sequences[call_count - 1]:
            yield event

    harness.client.stream_chat = fake_stream_chat
    runtime = StagedExecutionRuntime.from_harness(harness)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("execute staged plan"),
            timeout=10,
        )
    )

    assert result["status"] == "complete"
    assert result["plan_id"] == "plan-1"
    assert all(
        s.status in {"completed", "skipped"}
        for s in harness.state.active_plan.iter_steps()
    )
    assert harness.state.plan_execution_mode is False
