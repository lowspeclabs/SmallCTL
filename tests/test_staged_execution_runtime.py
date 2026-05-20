from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from smallctl.graph.runtime_staged import StagedExecutionRuntime
from smallctl.graph.test_time_scaling import FileSnapshotGuard
from smallctl.harness import Harness
from smallctl.models.events import UIEvent, UIEventType
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ExecutionPlan, LoopState, PlanStep, StepEvidenceArtifact, StepOutputSpec, StepVerifierSpec


def _step_complete_stream(tool_call_id: str, message: str = "step done") -> list[dict[str, object]]:
    return _tool_call_stream("step_complete", {"message": message}, tool_call_id=tool_call_id)


def _tool_call_stream(
    tool_name: str,
    arguments: dict[str, object],
    *,
    tool_call_id: str,
    assistant_text: str = "",
) -> list[dict[str, object]]:
    first_delta: dict[str, object] = {"role": "assistant", "reasoning_content": f"Calling {tool_name}.\n"}
    if assistant_text:
        first_delta["content"] = assistant_text
    return [
        {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": first_delta,
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
                                        "arguments": json.dumps(arguments),
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


def test_file_snapshot_guard_removes_new_empty_parent_directories(tmp_path: Path) -> None:
    guard = FileSnapshotGuard.capture(cwd=tmp_path, paths=["generated/nested/answer.py"])

    target = tmp_path / "generated" / "nested" / "answer.py"
    target.parent.mkdir(parents=True)
    target.write_text("branch attempt\n", encoding="utf-8")

    guard.restore()

    assert not target.exists()
    assert not (tmp_path / "generated" / "nested").exists()
    assert not (tmp_path / "generated").exists()


def test_file_snapshot_guard_keeps_existing_parent_directories(tmp_path: Path) -> None:
    existing = tmp_path / "generated"
    existing.mkdir()
    guard = FileSnapshotGuard.capture(cwd=tmp_path, paths=["generated/answer.py"])

    target = existing / "answer.py"
    target.write_text("branch attempt\n", encoding="utf-8")

    guard.restore()

    assert existing.exists()
    assert not target.exists()


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
    events: list[UIEvent] = []
    runtime = StagedExecutionRuntime.from_harness(harness, event_handler=events.append)

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


def test_test_time_scaling_selects_allowed_proposal_and_dispatches_it(tmp_path: Path, monkeypatch) -> None:
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
    harness.config.test_time_scaling_enabled = True
    harness.config.test_time_scaling_max_candidates = 2
    harness.config.test_time_scaling_min_candidates = 2
    harness.config.test_time_scaling_parallel_max = 2
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="scaled goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="hard step",
                task="finish the hard step",
                difficulty="hard",
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _tool_call_stream(
            "shell_exec",
            {"command": "pytest"},
            tool_call_id="tc-bad",
            assistant_text="I will try a command.",
        ),
        _step_complete_stream("tc-good", "scaled winner"),
    ]
    captured_prompts: list[list[dict[str, object]]] = []

    async def fake_stream_chat(*, messages, tools):
        del tools
        nonlocal call_count
        call_count += 1
        captured_prompts.append(list(messages))
        if call_count > len(stream_sequences):
            raise AssertionError(f"unexpected extra model call #{call_count}")
        for event in stream_sequences[call_count - 1]:
            yield event

    harness.client.stream_chat = fake_stream_chat
    events: list[UIEvent] = []
    runtime = StagedExecutionRuntime.from_harness(harness, event_handler=events.append)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("execute staged plan"),
            timeout=10,
        )
    )

    assert result["status"] == "complete"
    assert call_count == 2
    assert harness.state.step_evidence["S1"].summary == "scaled winner"
    assert harness.state.scratchpad["_test_time_scaling_last"]["selected_candidate"] == 2
    metrics = harness.state.scratchpad["_recovery_metrics"]
    assert metrics["test_time_scaling_attempts"] == 1
    assert metrics["test_time_scaling_candidates"] == 2
    assert metrics["test_time_scaling_parallel_proposal_batches"] == 1
    assert metrics["test_time_scaling_clean_selection"] == 1
    assert metrics["test_time_scaling_last"]["selected_candidate"] == 2
    assert len(harness.state.transcript_messages) == 2
    assert "scaled winner" in str(harness.state.transcript_messages)
    assert "pytest" not in str(harness.state.transcript_messages)
    assert any("TEST-TIME SCALING CANDIDATE" in str(message.get("content", "")) for message in captured_prompts[0])
    scaling_events = [
        event for event in events
        if event.event_type == UIEventType.SYSTEM and event.data.get("kind") == "test_time_scaling"
    ]
    assert [event.data["phase"] for event in scaling_events] == ["proposal_start", "proposal_selected"]
    assert scaling_events[-1].data["selected_candidate"] == 2


def test_sequential_branch_scaling_restores_loser_file_and_commits_winner(tmp_path: Path, monkeypatch) -> None:
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
    harness.config.test_time_scaling_enabled = True
    harness.config.test_time_scaling_policy = "sequential_branch"
    harness.config.test_time_scaling_max_candidates = 2
    harness.config.test_time_scaling_min_candidates = 2
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="scaled branch goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="write valid python",
                task="write answer.py",
                difficulty="hard",
                tool_allowlist=["file_write"],
                outputs_expected=[StepOutputSpec(kind="file", ref="answer.py", required=True)],
                verifiers=[StepVerifierSpec(kind="syntax_ok", args={"path": "answer.py"}, required=True)],
            ),
        ],
    )
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("dirty before", encoding="utf-8")

    call_count = 0
    stream_sequences = [
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def broken(:\n    pass\n"},
            tool_call_id="tc-bad-write",
        ),
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def answer():\n    return 42\n"},
            tool_call_id="tc-good-write",
        ),
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
    assert call_count == 2
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "def answer():\n    return 42\n"
    assert unrelated.read_text(encoding="utf-8") == "dirty before"
    evidence = harness.state.step_evidence["S1"]
    assert evidence.step_run_id.endswith("-cand2")
    assert any(result["kind"] == "syntax_ok" and result["passed"] is True for result in evidence.verifier_results)
    assert "tc-bad-write" not in str(harness.state.tool_execution_records)
    assert "broken" not in str(harness.state.transcript_messages)
    metrics = harness.state.scratchpad["_recovery_metrics"]["test_time_scaling_last"]
    assert metrics["policy"] == "sequential_branch"
    assert metrics["selected_candidate"] == 2
    assert metrics["failed"] is False


def test_sequential_branch_scaling_restores_loser_undeclared_file(tmp_path: Path, monkeypatch) -> None:
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
    harness.config.test_time_scaling_enabled = True
    harness.config.test_time_scaling_policy = "sequential_branch"
    harness.config.test_time_scaling_max_candidates = 2
    harness.config.test_time_scaling_min_candidates = 2
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="scaled branch undeclared file goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="write valid python",
                task="write answer.py",
                difficulty="hard",
                tool_allowlist=["file_write"],
                outputs_expected=[StepOutputSpec(kind="file", ref="answer.py", required=True)],
                verifiers=[StepVerifierSpec(kind="syntax_ok", args={"path": "answer.py"}, required=True)],
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _tool_call_stream(
            "file_write",
            {"path": "side.txt", "content": "loser side effect\n"},
            tool_call_id="tc-side-write",
        ),
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def answer():\n    return 42\n"},
            tool_call_id="tc-good-write",
        ),
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
    assert not (tmp_path / "side.txt").exists()
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "def answer():\n    return 42\n"
    assert "tc-side-write" not in str(harness.state.tool_execution_records)


def test_sequential_branch_scaling_parallelizes_read_only_candidates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ready.txt").write_text("ready\n", encoding="utf-8")

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
    harness.config.test_time_scaling_enabled = True
    harness.config.test_time_scaling_policy = "sequential_branch"
    harness.config.test_time_scaling_max_candidates = 2
    harness.config.test_time_scaling_min_candidates = 2
    harness.config.test_time_scaling_parallel_max = 2
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="scaled read-only branch goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="verify existing file",
                task="read ready.txt and finish",
                difficulty="hard",
                tool_allowlist=["file_read"],
                verifiers=[StepVerifierSpec(kind="file_exists", args={"path": "ready.txt"}, required=True)],
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _tool_call_stream("file_read", {"path": "ready.txt"}, tool_call_id="tc-read-slow"),
        _tool_call_stream("file_read", {"path": "ready.txt"}, tool_call_id="tc-read-fast"),
    ]

    async def fake_stream_chat(*, messages, tools):
        del messages, tools
        nonlocal call_count
        call_count += 1
        if call_count > len(stream_sequences):
            raise AssertionError(f"unexpected extra model call #{call_count}")
        for event in stream_sequences[call_count - 1]:
            yield event

    active_dispatches = 0
    max_active_dispatches = 0

    async def fake_dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        del tool_name
        nonlocal active_dispatches, max_active_dispatches
        active_dispatches += 1
        max_active_dispatches = max(max_active_dispatches, active_dispatches)
        try:
            delay = 0.08 if str(args.get("path") or "") == "ready.txt" and active_dispatches == 1 else 0.01
            await asyncio.sleep(delay)
            return ToolEnvelope(success=True, output="ready", metadata={"tool_name": "file_read"})
        finally:
            active_dispatches -= 1

    harness.client.stream_chat = fake_stream_chat
    harness._dispatch_tool_call = fake_dispatch_tool_call
    events: list[UIEvent] = []
    runtime = StagedExecutionRuntime.from_harness(harness, event_handler=events.append)

    result = asyncio.run(
        asyncio.wait_for(
            runtime.run("execute staged plan"),
            timeout=10,
        )
    )

    assert result["status"] == "complete"
    assert call_count == 2
    assert max_active_dispatches == 2
    evidence = harness.state.step_evidence["S1"]
    assert evidence.step_run_id.endswith("-cand2")
    metrics = harness.state.scratchpad["_recovery_metrics"]
    assert metrics["test_time_scaling_parallel_read_only_branch_batches"] == 1
    assert metrics["test_time_scaling_last"]["selected_candidate"] == 2
    assert metrics["test_time_scaling_last"]["read_only_branch_parallel_count"] == 1
    assert "tc-read-slow" not in str(harness.state.tool_execution_records)
    assert "tc-read-fast" in str(harness.state.tool_execution_records)
    scaling_events = [
        event for event in events
        if event.event_type == UIEventType.SYSTEM and event.data.get("kind") == "test_time_scaling"
    ]
    assert [event.data["phase"] for event in scaling_events] == ["branch_start", "branch_selected"]
    assert scaling_events[-1].data["selected_candidate"] == 2
    assert not any(event.event_type == UIEventType.TOOL_RESULT for event in events)


def test_sequential_branch_scaling_all_fail_falls_back_to_normal_retry(tmp_path: Path, monkeypatch) -> None:
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
    harness.config.test_time_scaling_enabled = True
    harness.config.test_time_scaling_policy = "sequential_branch"
    harness.config.test_time_scaling_max_candidates = 2
    harness.config.test_time_scaling_min_candidates = 2
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="scaled branch fallback goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="write valid python",
                task="write answer.py",
                difficulty="hard",
                tool_allowlist=["file_write"],
                outputs_expected=[StepOutputSpec(kind="file", ref="answer.py", required=True)],
                verifiers=[StepVerifierSpec(kind="syntax_ok", args={"path": "answer.py"}, required=True)],
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def broken(:\n    pass\n"},
            tool_call_id="tc-bad-write-1",
        ),
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def still_broken(:\n    pass\n"},
            tool_call_id="tc-bad-write-2",
        ),
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def answer():\n    return 42\n"},
            tool_call_id="tc-normal-write",
        ),
        _step_complete_stream("tc-normal-complete", "normal retry winner"),
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
    assert call_count == 4
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "def answer():\n    return 42\n"
    metrics = harness.state.scratchpad["_recovery_metrics"]["test_time_scaling_last"]
    assert metrics["failed"] is True
    assert metrics["all_failed_action"] == "fallback_normal_retry"
    assert "still_broken" not in str(harness.state.transcript_messages)


def test_sequential_branch_scaling_skips_mutating_shell_candidate(tmp_path: Path, monkeypatch) -> None:
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
    harness.config.test_time_scaling_enabled = True
    harness.config.test_time_scaling_policy = "sequential_branch"
    harness.config.test_time_scaling_max_candidates = 2
    harness.config.test_time_scaling_min_candidates = 2
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="scaled branch shell safety goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="write valid python",
                task="write answer.py",
                difficulty="hard",
                tool_allowlist=["shell_exec", "file_write"],
                outputs_expected=[StepOutputSpec(kind="file", ref="answer.py", required=True)],
                verifiers=[StepVerifierSpec(kind="syntax_ok", args={"path": "answer.py"}, required=True)],
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _tool_call_stream(
            "shell_exec",
            {"command": "touch shell-side-effect.txt"},
            tool_call_id="tc-shell-touch",
        ),
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def answer():\n    return 42\n"},
            tool_call_id="tc-good-write",
        ),
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
    assert not (tmp_path / "shell-side-effect.txt").exists()
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "def answer():\n    return 42\n"
    assert "tc-shell-touch" not in str(harness.state.tool_execution_records)
    metrics = harness.state.scratchpad["_recovery_metrics"]["test_time_scaling_last"]
    assert metrics["selected_candidate"] == 2


def test_sequential_branch_scaling_skips_remote_mutation_candidate(tmp_path: Path, monkeypatch) -> None:
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
    harness.config.test_time_scaling_enabled = True
    harness.config.test_time_scaling_policy = "sequential_branch"
    harness.config.test_time_scaling_max_candidates = 2
    harness.config.test_time_scaling_min_candidates = 2
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="scaled branch remote mutation safety goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="write valid python",
                task="write answer.py",
                difficulty="hard",
                tool_allowlist=["ssh_file_write", "file_write"],
                outputs_expected=[StepOutputSpec(kind="file", ref="answer.py", required=True)],
                verifiers=[StepVerifierSpec(kind="syntax_ok", args={"path": "answer.py"}, required=True)],
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _tool_call_stream(
            "ssh_file_write",
            {"target": "root@example.test", "path": "/tmp/answer.py", "content": "remote side effect"},
            tool_call_id="tc-remote-write",
        ),
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def answer():\n    return 42\n"},
            tool_call_id="tc-good-write",
        ),
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
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "def answer():\n    return 42\n"
    assert "tc-remote-write" not in str(harness.state.tool_execution_records)
    metrics = harness.state.scratchpad["_recovery_metrics"]["test_time_scaling_last"]
    assert metrics["selected_candidate"] == 2


def test_sequential_branch_scaling_cleans_failed_candidate_parent_dirs(tmp_path: Path, monkeypatch) -> None:
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
    harness.config.test_time_scaling_enabled = True
    harness.config.test_time_scaling_policy = "sequential_branch"
    harness.config.test_time_scaling_max_candidates = 2
    harness.config.test_time_scaling_min_candidates = 2
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="scaled branch cleanup goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="write valid python",
                task="write answer.py",
                difficulty="hard",
                tool_allowlist=["file_write"],
                outputs_expected=[StepOutputSpec(kind="file", ref="answer.py", required=True)],
                verifiers=[StepVerifierSpec(kind="syntax_ok", args={"path": "answer.py"}, required=True)],
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _tool_call_stream(
            "file_write",
            {"path": "scratch/candidate/answer.py", "content": "def wrong_location():\n    return 0\n"},
            tool_call_id="tc-wrong-location",
        ),
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def answer():\n    return 42\n"},
            tool_call_id="tc-good-write",
        ),
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
    assert call_count == 2
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == "def answer():\n    return 42\n"
    assert not (tmp_path / "scratch").exists()
    metrics = harness.state.scratchpad["_recovery_metrics"]["test_time_scaling_last"]
    assert metrics["selected_candidate"] == 2


def test_sequential_branch_scaling_all_fail_can_fail_step(tmp_path: Path, monkeypatch) -> None:
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
    harness.config.test_time_scaling_enabled = True
    harness.config.test_time_scaling_policy = "sequential_branch"
    harness.config.test_time_scaling_all_fail_action = "fail_step"
    harness.config.test_time_scaling_max_candidates = 2
    harness.config.test_time_scaling_min_candidates = 2
    harness.state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="scaled branch fail-step goal",
        approved=True,
        steps=[
            PlanStep(
                step_id="S1",
                title="write valid python",
                task="write answer.py",
                difficulty="hard",
                max_retries=1,
                tool_allowlist=["file_write"],
                outputs_expected=[StepOutputSpec(kind="file", ref="answer.py", required=True)],
                verifiers=[StepVerifierSpec(kind="syntax_ok", args={"path": "answer.py"}, required=True)],
            ),
        ],
    )

    call_count = 0
    stream_sequences = [
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def broken(:\n    pass\n"},
            tool_call_id="tc-bad-write-1",
        ),
        _tool_call_stream(
            "file_write",
            {"path": "answer.py", "content": "def still_broken(:\n    pass\n"},
            tool_call_id="tc-bad-write-2",
        ),
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

    assert result["status"] == "needs_human"
    assert call_count == 2
    step = harness.state.active_plan.find_step("S1")
    assert step.status == "blocked"
    assert step.retry_count == 1
    assert harness.state.pending_interrupt["kind"] == "staged_step_blocked"
    metrics = harness.state.scratchpad["_recovery_metrics"]["test_time_scaling_last"]
    assert metrics["failed"] is True
    assert metrics["all_failed_action"] == "fail_step"
    assert not (tmp_path / "answer.py").exists()


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
