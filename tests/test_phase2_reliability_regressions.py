"""Regression tests for Phase 2 reliability and error-handling improvements.

Covers bugs 9 through 16 from bugs-found.md.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import ANY, AsyncMock, Mock

import pytest

from smallctl.client import OpenAICompatClient
from smallctl.client.client_transport_openrouter_preflight import _preflight_openrouter_auth
from smallctl.graph import runtime_base
from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.model_stream import _run_nonstream_model_call
from smallctl.graph.runtime import ChatGraphRuntime
from smallctl.graph.runtime_tool_plan import ToolPlanRuntime
from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.graph.tool_execution_nodes import dispatch_tools
from smallctl.graph.tool_execution_node_guards import _synthetic_blocked_records
from smallctl.harness.approvals import ApprovalService
from smallctl.state import LoopState
from smallctl.tools.data import yaml_read
from smallctl.tools.search import find_files, grep


class _FakeHarness:
    def __init__(self, state: LoopState | None = None) -> None:
        self.state = state or LoopState(cwd=str(Path.cwd()))
        self.allow_interactive_shell_approval = True
        self.shell_approval_session_default = False
        self.event_handler = "dummy"
        self.log = Mock()
        self._runlog = Mock()

    async def _emit(self, *args, **kwargs) -> None:
        del args, kwargs


@pytest.mark.asyncio
async def test_approval_service_shell_approval_times_out() -> None:
    """Bug 9: shell approval prompt should return False on timeout."""
    harness = _FakeHarness()
    service = ApprovalService(harness)

    result = await service.request_shell_approval(
        command="rm -rf /",
        cwd="/tmp",
        timeout_sec=0,
    )

    assert result is False
    harness._runlog.assert_any_call(
        "approval_timeout",
        "Shell approval request timed out",
        command="rm -rf /",
        approval_id=ANY,
        timeout_sec=0,
    )
    assert not service._shell_approval_waiters


@pytest.mark.asyncio
async def test_approval_service_sudo_password_times_out() -> None:
    """Bug 9: sudo password prompt should return None on timeout."""
    harness = _FakeHarness()
    service = ApprovalService(harness)

    result = await service.request_sudo_password(
        command="sudo apt update",
        prompt_text="[sudo] password:",
        timeout_sec=0,
    )

    assert result is None
    harness._runlog.assert_any_call(
        "sudo_password_timeout",
        "Sudo password prompt timed out",
        command="sudo apt update",
        prompt_id=ANY,
        timeout_sec=0,
    )
    assert not service._sudo_password_waiters


@pytest.mark.asyncio
async def test_openrouter_auth_preflight_passes_finite_timeout(monkeypatch) -> None:
    """Bug 10: OpenRouter auth preflight must supply a finite request timeout."""
    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        provider_profile="openrouter",
        api_key="key",
    )

    captured: dict[str, object] = {}

    class _FakeAsyncClient:
        async def get(self, url, headers, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout

            class _Response:
                status_code = 200
                text = '{"data":{"total_credits":1}}'

            return _Response()

    await _preflight_openrouter_auth(client, _FakeAsyncClient())

    assert captured["timeout"] == 10.0
    assert "credits" in str(captured["url"])


@pytest.mark.asyncio
async def test_content_tools_resolve_relative_paths_against_cwd(
    tmp_path: Path, monkeypatch
) -> None:
    """Bug 11: grep/find_files/yaml_read must resolve relative paths against harness cwd."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    other_dir = tmp_path / "other"
    other_dir.mkdir()

    (work_dir / "file.yaml").write_text("answer: 42\n", encoding="utf-8")
    (work_dir / "sub").mkdir()
    (work_dir / "sub" / "foo.txt").write_text("hello\n", encoding="utf-8")

    monkeypatch.chdir(other_dir)

    yaml_result = await yaml_read("file.yaml", cwd=str(work_dir))
    assert yaml_result["success"] is True
    assert yaml_result["output"] == {"answer": 42}

    find_result = await find_files("*.txt", path=".", cwd=str(work_dir))
    assert find_result["success"] is True
    assert find_result["output"] == [str(work_dir / "sub" / "foo.txt")]

    grep_result = await grep("hello", path=".", cwd=str(work_dir))
    assert grep_result["success"] is True
    assert len(grep_result["output"]) == 1
    assert grep_result["output"][0]["path"] == str(work_dir / "sub" / "foo.txt")


def test_chat_runtime_has_interrupt_resume_edge() -> None:
    """Bug 12: ChatGraphRuntime must resume after interrupt_for_human."""
    static_edges = set(ChatGraphRuntime.GRAPH_SPEC.static_edges)
    assert ("interrupt_for_human", "prepare_step") in static_edges
    assert ("persist_tool_results", "apply_chat_tool_outcomes") in static_edges


@pytest.mark.asyncio
async def test_dispatch_node_timeout_re_raises_without_cancel_flag() -> None:
    """Bug 13: CancelledError should be re-raised when no explicit cancel is requested."""
    state = LoopState(cwd=str(Path.cwd()))
    state.active_tool_profiles = ["core"]

    class _Harness:
        def __init__(self) -> None:
            self.state = state
            self._runlog = lambda *args, **kwargs: None
            self._emit = AsyncMock()
            self._cancel_requested = False
            self.registry = None
            self.log = Mock()
            self._active_dispatch_task = None
            self._active_ui_tool_context = None

        async def _dispatch_tool_call(self, tool_name: str, args: dict) -> None:
            raise asyncio.CancelledError("node timeout")

    harness = _Harness()
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args={"path": "foo.txt"},
            tool_call_id="call-1",
            source="model",
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    with pytest.raises(asyncio.CancelledError):
        await dispatch_tools(graph_state, deps)


@pytest.mark.asyncio
async def test_dispatch_node_cancel_flag_becomes_cancelled_result() -> None:
    """Bug 13: explicit cancel flag should still produce a cancelled result."""
    state = LoopState(cwd=str(Path.cwd()))
    state.active_tool_profiles = ["core"]

    class _Harness:
        def __init__(self) -> None:
            self.state = state
            self._runlog = lambda *args, **kwargs: None
            self._emit = AsyncMock()
            self._cancel_requested = True
            self._cancel_source = "user"
            self.registry = None
            self.log = Mock()
            self._active_dispatch_task = None
            self._active_ui_tool_context = None

        async def _dispatch_tool_call(self, tool_name: str, args: dict) -> None:
            raise asyncio.CancelledError("user cancel")

    harness = _Harness()
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args={"path": "foo.txt"},
            tool_call_id="call-1",
            source="model",
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    await dispatch_tools(graph_state, deps)

    assert graph_state.final_result == {"status": "cancelled", "reason": "cancel_requested"}
    assert len(graph_state.last_tool_results) == 1
    assert graph_state.last_tool_results[0].result.status == "cancelled"


@pytest.mark.asyncio
async def test_synthetic_blocked_records_preserve_tool_call_ids() -> None:
    """Bug 14: helper produces records with original tool_call_ids and blocked status."""
    state = LoopState(cwd=str(Path.cwd()))
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_write",
            args={"path": "foo.txt"},
            tool_call_id="call-1",
            source="model",
        ),
        PendingToolCall(
            tool_name="file_read",
            args={"path": "foo.txt"},
            tool_call_id="call-2",
            source="model",
        ),
    ]

    records = _synthetic_blocked_records(graph_state, reason="blocked")

    assert len(records) == 2
    assert {r.tool_call_id for r in records} == {"call-1", "call-2"}
    for record in records:
        assert record.result.status == "blocked"
        assert record.result.metadata.get("reason") == "pre_dispatch_guard_blocked"


@pytest.mark.asyncio
async def test_remote_task_guard_produces_synthetic_blocked_records(
    monkeypatch,
) -> None:
    """Bug 14: remote-task guard must emit matching tool messages for blocked calls."""
    state = LoopState(cwd=str(Path.cwd()))
    state.active_tool_profiles = ["core"]

    from smallctl.models.tool_result import ToolEnvelope
    from smallctl.tools import dispatcher_normalization_flow

    def _fake_guard_remote_file_tool_request(tool_name, arguments, *, state, ssh_available):
        return ToolEnvelope.make_error(
            tool_name,
            "remote task requires ssh_exec",
            reason="remote_task_requires_ssh_exec",
        )

    monkeypatch.setattr(
        dispatcher_normalization_flow,
        "_guard_remote_file_tool_request",
        _fake_guard_remote_file_tool_request,
    )

    class _Harness:
        def __init__(self) -> None:
            self.state = state
            self._runlog = lambda *args, **kwargs: None
            self._emit = AsyncMock()
            self.registry = object()
            self.log = Mock()

    harness = _Harness()
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args={"path": "/etc/hosts"},
            tool_call_id="remote-call-1",
            source="model",
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    await dispatch_tools(graph_state, deps)

    assert len(graph_state.last_tool_results) == 1
    record = graph_state.last_tool_results[0]
    assert record.tool_call_id == "remote-call-1"
    assert record.result.status == "blocked"
    assert record.result.metadata.get("reason") == "pre_dispatch_guard_blocked"


@pytest.mark.asyncio
async def test_patch_contract_violation_produces_synthetic_blocked_records(
    monkeypatch,
) -> None:
    """Bug 14: patch-existing stage violation must emit matching tool messages."""
    state = LoopState(cwd=str(Path.cwd()))
    state.active_tool_profiles = ["core"]

    from smallctl.graph import tool_execution_nodes as ten

    def _fake_contract_violation(*args, **kwargs):
        return (
            "Ambiguous write after staged read",
            {"target_path": "foo.py", "write_session_id": "ws-1"},
        )

    monkeypatch.setattr(
        ten,
        "_detect_patch_existing_stage_read_contract_violation",
        _fake_contract_violation,
    )

    class _Registry:
        def names(self):
            return {"file_write"}

    class _Harness:
        def __init__(self) -> None:
            self.state = state
            self._runlog = lambda *args, **kwargs: None
            self._emit = AsyncMock()
            self.registry = _Registry()
            self.log = Mock()
            self._active_dispatch_task = None
            self._active_ui_tool_context = None

        async def _dispatch_tool_call(self, tool_name: str, args: dict) -> None:
            raise AssertionError("dispatch should not be reached")

    harness = _Harness()
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_write",
            args={"path": "foo.py", "content": "x"},
            tool_call_id="patch-call-1",
            source="model",
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    await dispatch_tools(graph_state, deps)

    assert len(graph_state.last_tool_results) == 1
    record = graph_state.last_tool_results[0]
    assert record.tool_call_id == "patch-call-1"
    assert record.result.status == "blocked"
    assert record.result.metadata.get("reason") == "pre_dispatch_guard_blocked"


@pytest.mark.asyncio
async def test_nonstream_fallback_reports_chunk_error() -> None:
    """Bug 15: non-stream fallback should not report clean completion on chunk_error."""
    state = LoopState(cwd=str(Path.cwd()))

    class _ErrorClient:
        model = "test"

        async def stream_chat(self, *, messages, tools, force_nonstream):
            yield {"type": "chunk_error", "error": "backend failure", "details": {"reason": "backend_stream_failure"}}

    harness = _FakeHarness(state)
    harness.client = _ErrorClient()
    harness.thinking_start_tag = "<think>"
    harness.thinking_end_tag = "</think>"
    harness.thinking_visibility = False
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    result = await _run_nonstream_model_call(
        graph_state,
        deps,
        harness=harness,
        messages=[],
        tools=[],
        echo_to_stdout=False,
        start_tag="<think>",
        end_tag="</think>",
        start_time=0.0,
    )

    assert result["stream_completed_cleanly"] is False
    assert result["last_chunk_error_details"] is not None
    assert result["last_chunk_error_details"]["error"] == "backend failure"


@pytest.mark.asyncio
async def test_nonstream_fallback_reports_exception() -> None:
    """Bug 15: non-stream fallback should not report clean completion on exception."""
    state = LoopState(cwd=str(Path.cwd()))

    class _ExceptionClient:
        model = "test"

        async def stream_chat(self, *, messages, tools, force_nonstream):
            raise RuntimeError("connection refused")
            yield {}  # makes it an async generator; unreachable

    harness = _FakeHarness(state)
    harness.client = _ExceptionClient()
    harness.thinking_start_tag = "<think>"
    harness.thinking_end_tag = "</think>"
    harness.thinking_visibility = False
    graph_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    result = await _run_nonstream_model_call(
        graph_state,
        deps,
        harness=harness,
        messages=[],
        tools=[],
        echo_to_stdout=False,
        start_tag="<think>",
        end_tag="</think>",
        start_time=0.0,
    )

    assert result["stream_completed_cleanly"] is False
    assert result["last_chunk_error_details"] is not None
    assert result["last_chunk_error_details"]["exception_type"] == "RuntimeError"
    assert "connection refused" in result["last_chunk_error_details"]["error"]


@pytest.mark.asyncio
async def test_tool_plan_runtime_calls_after_run_hook(
    tmp_path: Path, monkeypatch
) -> None:
    """Bug 16: ToolPlanRuntime.run must invoke _after_run trajectory recording."""
    monkeypatch.chdir(tmp_path)
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()

    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "find dispatch seam"
    state.scratchpad["_tool_plan"] = {
        "steps": [
            {"id": "E1", "tool": "file_read", "args": {"path": "src/app.py"}}
        ]
    }
    state.scratchpad["_tool_plan_observations_text"] = "read src/app.py"
    state.scratchpad["_recovery_metrics"] = {
        "tool_plan_invocations": 1,
        "tool_plan_planner_tokens": 10,
        "tool_plan_solver_tokens": 20,
        "tool_plan_total_tokens": 30,
        "tool_plan_steps_requested": 1,
        "tool_plan_steps_executed": 1,
        "tool_plan_step_failures": 0,
    }

    harness = _FakeHarness(state)
    harness.conversation_id = "conv-1"

    runtime = ToolPlanRuntime.from_harness(harness)

    async def _fake_base_run(self, task: str) -> dict[str, object]:
        return {
            "status": "completed",
            "reason": "done",
            "latency_metrics": {"tool_execution_duration_sec": 1.23},
        }

    monkeypatch.setattr(runtime_base.CompiledGraphRuntimeBase, "run", _fake_base_run)
    monkeypatch.setattr(
        "smallctl.graph.runtime_tool_plan.TrajectoryRecorder.__init__",
        lambda self, base_dir=".smallctl/traces": None,
    )
    recorded: dict[str, object] = {}

    def _fake_record_tool_plan_trajectory(self, harness, result):
        recorded["harness"] = harness
        recorded["result"] = result
        recorded["task"] = harness.state.run_brief.original_task
        return None

    monkeypatch.setattr(
        "smallctl.graph.runtime_tool_plan.TrajectoryRecorder.record_tool_plan_trajectory",
        _fake_record_tool_plan_trajectory,
    )

    result = await runtime.run("find dispatch seam")

    assert result["status"] == "completed"
    assert recorded["harness"] is harness
    assert recorded["result"] is result
    assert recorded["task"] == "find dispatch seam"


@pytest.mark.asyncio
async def test_tool_plan_runtime_after_run_records_trajectory_file(
    tmp_path: Path, monkeypatch
) -> None:
    """Bug 16: _after_run writes a trajectory file when run succeeds."""
    monkeypatch.chdir(tmp_path)
    trace_dir = tmp_path / "traces"

    state = LoopState(cwd=str(tmp_path))
    state.run_brief.original_task = "find dispatch seam"
    state.scratchpad["_tool_plan"] = {
        "steps": [
            {"id": "E1", "tool": "file_read", "args": {"path": "src/app.py"}}
        ]
    }
    state.scratchpad["_tool_plan_observations_text"] = "read src/app.py"
    state.scratchpad["_recovery_metrics"] = {}

    harness = _FakeHarness(state)
    harness.conversation_id = "conv-1"

    runtime = ToolPlanRuntime.from_harness(harness)

    async def _fake_base_run(self, task: str) -> dict[str, object]:
        return {
            "status": "completed",
            "reason": "done",
            "latency_metrics": {"tool_execution_duration_sec": 1.0},
        }

    monkeypatch.setattr(runtime_base.CompiledGraphRuntimeBase, "run", _fake_base_run)

    from smallctl.harness.trajectory_recorder import TrajectoryRecorder

    recorder = TrajectoryRecorder(base_dir=str(trace_dir))
    monkeypatch.setattr(
        "smallctl.graph.runtime_tool_plan.TrajectoryRecorder",
        lambda: recorder,
    )

    await runtime.run("find dispatch seam")

    trace_files = list(trace_dir.glob("*.jsonl"))
    assert len(trace_files) == 1
    content = json.loads(trace_files[0].read_text(encoding="utf-8"))
    assert content["task"] == "find dispatch seam"
    assert content["success"] is True
