from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from smallctl.graph import model_stream
from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.lifecycle_prompt import (
    prepare_indexer_prompt,
    prepare_prompt,
    select_loop_tools,
)
from smallctl.graph.nodes import LoopRoute, interpret_model_output
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_outcomes import (
    _reset_no_tool_nudges_after_successful_dispatch,
    apply_tool_outcomes,
)
from smallctl.harness.prompt_builder import prompt_budget_overflow_error
from smallctl.harness.runtime_facade import (
    HarnessRunAlreadyActiveError,
    resume_task_with_events,
    run_auto_with_events,
    run_task_with_events,
)
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.register import build_registry
from smallctl.ui.harness_bridge import HarnessBridge, HarnessRunBusyError


# --- H16: resume path enters the facade run guard -----------------------------


def _facade_harness_stub() -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(
            task_received_at="",
            touch=lambda: None,
            active_plan=None,
            draft_plan=None,
            planning_mode_enabled=False,
        ),
        config=SimpleNamespace(staged_execution_enabled=False),
        get_pending_interrupt=lambda: None,
        event_handler=None,
        _cancel_requested=True,
        log=logging.getLogger("test.gap_fixes.h16"),
    )


def test_h16_concurrent_resume_refused_while_run_active(monkeypatch) -> None:
    release = threading.Event()
    started: list[str] = []

    class _StubRuntime:
        async def run(self, task: str) -> dict[str, object]:
            started.append(task)
            while not release.is_set():
                await asyncio.sleep(0.005)
            return {"status": "completed", "task": task}

    monkeypatch.setattr(
        "smallctl.graph.runtime.AutoGraphRuntime.from_harness",
        lambda harness, event_handler=None: _StubRuntime(),
    )
    harness = _facade_harness_stub()

    async def _run() -> None:
        first = asyncio.create_task(run_auto_with_events(harness, "first"))
        loops = 0
        while not started:
            loops += 1
            assert loops < 1000
            await asyncio.sleep(0.005)

        with pytest.raises(HarnessRunAlreadyActiveError):
            await resume_task_with_events(harness, "yes")

        release.set()
        assert (await first)["status"] == "completed"
        assert harness._run_guard_in_flight is False

    asyncio.run(_run())

    assert started == ["first"]


def test_h16_interrupt_redirect_inside_run_does_not_trip_guard(monkeypatch) -> None:
    resumed: list[str] = []

    class _StubRuntime:
        async def resume(self, human_input: str) -> dict[str, object]:
            resumed.append(human_input)
            return {"status": "resumed", "choice": human_input}

    monkeypatch.setattr(
        "smallctl.graph.runtime.PlanningGraphRuntime.from_harness",
        lambda harness, event_handler=None: _StubRuntime(),
    )
    harness = _facade_harness_stub()
    harness.get_pending_interrupt = lambda: {
        "kind": "plan_execute_approval",
        "question": "Plan ready. Execute it now?",
        "response_mode": "yes/no/revise",
    }

    result = asyncio.run(run_task_with_events(harness, "yes"))

    assert result == {"status": "resumed", "choice": "yes"}
    assert resumed == ["yes"]
    assert harness._run_guard_in_flight is False


# --- L31: prompt-budget slimming state resets across prompts/tasks ------------


class _BudgetHarness:
    def __init__(self, state: LoopState, *, limit: int, per_tool_tokens: int) -> None:
        self.state = state
        self.strategy_prompt = ""
        self._indexer = False
        self.log = logging.getLogger("test.gap_fixes.l31")
        self.runlog_events: list[tuple[tuple, dict]] = []
        self._limit = limit
        self._per_tool_tokens = per_tool_tokens

    def _runlog(self, *args, **kwargs) -> None:
        self.runlog_events.append((args, kwargs))

    async def _emit(self, event_handler, event) -> None:
        return None

    def _failure(self, message: str, error_type: str = "", details: dict | None = None) -> dict:
        return {"status": "failed", "error": message, "error_type": error_type}

    async def _build_prompt_messages(self, system_prompt: str, event_handler=None) -> list[dict]:
        tool_count = system_prompt.count("tool:")
        estimated = 10 + tool_count * self._per_tool_tokens
        if estimated > self._limit:
            raise prompt_budget_overflow_error(
                estimated_tokens=estimated,
                limit=self._limit,
                section_tokens={"system_prompt": estimated, "transcript": 5},
            )
        return [{"role": "system", "content": system_prompt}]


def _tool_schema(name: str, description: str) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": {}}},
    }


def _patch_exposure(monkeypatch) -> None:
    exposure = {
        "names": ["file_read", "task_complete", "web_search"],
        "schemas": [
            _tool_schema("file_read", "read a file " * 40),
            _tool_schema("task_complete", "complete the task " * 40),
            _tool_schema("web_search", "search the web " * 40),
        ],
    }
    monkeypatch.setattr(
        "smallctl.graph.lifecycle_prompt.resolve_turn_tool_exposure",
        lambda harness, mode: exposure,
    )
    monkeypatch.setattr(
        "smallctl.graph.lifecycle_prompt.build_system_prompt",
        lambda state, phase, available_tool_names=None, **kwargs: "sys " + " ".join(
            f"tool:{name}" for name in (available_tool_names or [])
        ),
    )


def test_l31_slimming_state_resets_for_next_prompt_and_task(monkeypatch, tmp_path) -> None:
    _patch_exposure(monkeypatch)
    state = LoopState(cwd=str(tmp_path))
    harness = _BudgetHarness(state, limit=30, per_tool_tokens=10)
    deps = GraphRuntimeDeps(harness=harness)

    first = asyncio.run(prepare_prompt(GraphRunState(loop_state=state, thread_id="t1", run_mode="loop"), deps))
    assert first is not None
    assert state.scratchpad["_prompt_budget_tool_slimming_active"] is True
    slimmed = select_loop_tools(GraphRunState(loop_state=state, thread_id="t1", run_mode="loop"), deps)
    assert [schema["function"]["name"] for schema in slimmed] == ["task_complete"]

    harness._limit = 100
    second_state = GraphRunState(loop_state=state, thread_id="t1", run_mode="loop")
    second = asyncio.run(prepare_prompt(second_state, deps))

    assert second is not None
    assert "_prompt_budget_tool_slimming_active" not in state.scratchpad
    assert "_prompt_budget_slimmed_tool_schemas" not in state.scratchpad
    restored = select_loop_tools(second_state, deps)
    assert {schema["function"]["name"] for schema in restored} == {
        "file_read",
        "task_complete",
        "web_search",
    }
    assert any(
        args and args[0] == "prompt_budget_tool_slimming_cleared" for args, _ in harness.runlog_events
    )

    state.scratchpad["_prompt_budget_tool_slimming_active"] = True
    state.scratchpad["_prompt_budget_slimmed_tool_schemas"] = [_tool_schema("task_complete", "x")]
    next_task = asyncio.run(
        prepare_prompt(GraphRunState(loop_state=state, thread_id="t2", run_mode="loop"), deps)
    )

    assert next_task is not None
    assert "_prompt_budget_tool_slimming_active" not in state.scratchpad
    assert "_prompt_budget_slimmed_tool_schemas" not in state.scratchpad


def test_l31_indexer_prompt_clears_stale_slimming_state(monkeypatch, tmp_path) -> None:
    _patch_exposure(monkeypatch)
    state = LoopState(cwd=str(tmp_path))
    harness = _BudgetHarness(state, limit=100, per_tool_tokens=10)
    state.scratchpad["_prompt_budget_tool_slimming_active"] = True
    state.scratchpad["_prompt_budget_slimmed_tool_schemas"] = [_tool_schema("task_complete", "x")]
    deps = GraphRuntimeDeps(harness=harness)

    messages = asyncio.run(
        prepare_indexer_prompt(GraphRunState(loop_state=state, thread_id="t3", run_mode="indexer"), deps)
    )

    assert messages is not None
    assert "_prompt_budget_tool_slimming_active" not in state.scratchpad
    assert "_prompt_budget_slimmed_tool_schemas" not in state.scratchpad


# --- M1: no-tool nudge counter resets only on successful dispatch --------------


def _m1_harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=SimpleNamespace(min_exploration_steps=0),
        summarizer=None,
        _extract_planning_request=lambda task: None,
        _record_experience=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
        _emit=lambda *args, **kwargs: None,
        _failure=lambda error, error_type="runtime", details=None: {
            "error": error,
            "error_type": error_type,
            "details": details or {},
        },
    )


def _m1_record(tool_name: str, *, success: bool) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        operation_id=f"op-{tool_name}",
        tool_name=tool_name,
        args={},
        tool_call_id=f"call-{tool_name}",
        result=ToolEnvelope(
            success=success,
            output={"status": "ok"} if success else None,
            error=None if success else "dispatch failed",
        ),
    )


def test_m1_failed_dispatch_preserves_counter_and_success_resets_it() -> None:
    state = LoopState(cwd=".")
    state.active_tool_profiles = ["core"]
    state.artifacts = {}
    harness = _m1_harness(state)
    state.scratchpad["_no_tool_nudges"] = 3

    asyncio.run(
        apply_tool_outcomes(
            GraphRunState(
                loop_state=state,
                thread_id="m1-fail",
                run_mode="loop",
                last_tool_results=[_m1_record("file_read", success=False)],
            ),
            GraphRuntimeDeps(harness=harness, event_handler=None),
        )
    )
    assert int(state.scratchpad.get("_no_tool_nudges", 0)) == 3

    asyncio.run(
        apply_tool_outcomes(
            GraphRunState(
                loop_state=state,
                thread_id="m1-success",
                run_mode="loop",
                last_tool_results=[_m1_record("file_read", success=True)],
            ),
            GraphRuntimeDeps(harness=harness, event_handler=None),
        )
    )
    assert int(state.scratchpad.get("_no_tool_nudges", 0)) == 0


def test_m1_mixed_batch_with_any_success_resets_counter() -> None:
    state = LoopState(cwd=".")
    harness = _m1_harness(state)
    state.scratchpad["_no_tool_nudges"] = 2

    _reset_no_tool_nudges_after_successful_dispatch(
        harness,
        [_m1_record("file_read", success=False), _m1_record("dir_list", success=True)],
    )
    assert int(state.scratchpad.get("_no_tool_nudges", 0)) == 0

    state.scratchpad["_no_tool_nudges"] = 2
    _reset_no_tool_nudges_after_successful_dispatch(
        harness,
        [_m1_record("file_read", success=False), _m1_record("dir_list", success=False)],
    )
    assert int(state.scratchpad.get("_no_tool_nudges", 0)) == 2


def test_m1_dispatch_routing_alone_does_not_reset_counter() -> None:
    state = LoopState(cwd=".")
    state.active_tool_profiles = ["core"]
    state.artifacts = {}
    harness = _m1_harness(state)
    state.scratchpad["_no_tool_nudges"] = 2
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="m1-route",
        run_mode="loop",
        pending_tool_calls=[PendingToolCall(tool_name="file_read", args={"path": "README.md"})],
    )

    route = asyncio.run(interpret_model_output(graph_state, GraphRuntimeDeps(harness=harness, event_handler=None)))

    assert route == LoopRoute.DISPATCH_TOOLS
    assert int(state.scratchpad.get("_no_tool_nudges", 0)) == 2


# --- M4: file_append parity with file_write ------------------------------------


def _m4_dispatcher(state: LoopState) -> ToolDispatcher:
    registry = build_registry(SimpleNamespace(state=state, log=logging.getLogger("test.gap_fixes.m4")))
    return ToolDispatcher(registry, state=state, phase="execute")


def test_m4_file_append_schema_at_file_write_parity(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    registry = build_registry(SimpleNamespace(state=state, log=logging.getLogger("test.gap_fixes.m4")))
    file_write = registry.get("file_write")
    file_append = registry.get("file_append")

    assert file_append.allowed_modes == file_write.allowed_modes
    assert file_append.profiles == file_write.profiles
    assert file_append.risk == file_write.risk
    assert file_append.category == file_write.category
    assert file_append.schema["required"] == file_write.schema["required"]
    assert file_append.schema["additionalProperties"] is False
    assert set(file_append.schema["properties"]) == set(file_write.schema["properties"])
    for field in (
        "path",
        "write_session_id",
        "section_name",
        "section_id",
        "section_role",
        "next_section_name",
        "expected_followup_verifier",
    ):
        assert file_append.schema["properties"][field] == file_write.schema["properties"][field]
    assert (
        file_append.schema["properties"]["replace_strategy"]["enum"]
        == file_write.schema["properties"]["replace_strategy"]["enum"]
        == ["append", "overwrite"]
    )


def test_m4_file_append_overwrite_replaces_instead_of_appending(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _m4_dispatcher(state)
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")

    result = asyncio.run(
        dispatcher.dispatch(
            "file_append",
            {"path": "notes.txt", "content": "replacement\n", "replace_strategy": "overwrite"},
        )
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "replacement\n"


def test_m4_file_append_default_still_appends(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _m4_dispatcher(state)
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")

    result = asyncio.run(
        dispatcher.dispatch("file_append", {"path": "notes.txt", "content": "extra\n"})
    )

    assert result.success is True
    assert target.read_text(encoding="utf-8") == "original\nextra\n"


# --- M16: shutdown with a cancellation-suppressing run wedges safely -----------


class _CancellationSuppressingHarness:
    def __init__(self, *, suppress_for_sec: float) -> None:
        self.release_after = suppress_for_sec
        self.started = threading.Event()
        self.exited = threading.Event()
        self.run_calls: list[str] = []
        self.cancel_sources: list[str] = []

    async def run_auto_with_events(self, task: str, event_handler) -> dict[str, object]:
        self.run_calls.append(task)
        self.started.set()
        if len(self.run_calls) > 1:
            return {"status": "ok", "task": task}
        deadline = time.monotonic() + self.release_after
        try:
            while time.monotonic() < deadline:
                await asyncio.sleep(0.02)
        except asyncio.CancelledError:
            while time.monotonic() < deadline:
                try:
                    await asyncio.sleep(0.02)
                except asyncio.CancelledError:
                    pass
            raise
        finally:
            self.exited.set()
        return {"status": "ok", "task": task}

    async def resume_task_with_events(self, human_input: str, event_handler) -> dict[str, object]:
        return {"status": "resumed", "choice": human_input}

    def cancel(self, source: str = "manual") -> None:
        self.cancel_sources.append(source)

    async def teardown(self) -> None:
        await asyncio.sleep(0)


def test_m16_shutdown_bounded_wedged_bridge_refuses_new_runs(caplog) -> None:
    harness = _CancellationSuppressingHarness(suppress_for_sec=7.0)
    bridge = HarnessBridge(
        harness=harness,
        post_ui_event=lambda event: None,
        thread_name="smallctl-test-wedged",
        shutdown_timeout_sec=0.2,
    )

    async def _run() -> None:
        first = asyncio.create_task(bridge.run_auto("first"))
        loops = 0
        while not harness.started.is_set():
            loops += 1
            assert loops < 1000
            await asyncio.sleep(0.005)

        with caplog.at_level(logging.WARNING, logger="smallctl.ui.harness_bridge"):
            started_at = time.monotonic()
            await bridge.shutdown()
            shutdown_elapsed = time.monotonic() - started_at

        assert shutdown_elapsed < 12.0
        thread = bridge._thread
        assert thread is not None and thread.is_alive()
        assert any("STILL ALIVE" in record.getMessage() for record in caplog.records)

        with pytest.raises(HarnessRunBusyError):
            await bridge.run_auto("second")
        assert harness.run_calls == ["first"]

        deadline = time.monotonic() + 10.0
        while thread.is_alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert not thread.is_alive()

        third = await bridge.run_auto("third")
        assert third["status"] == "ok"
        await bridge.shutdown()

        with pytest.raises(asyncio.CancelledError):
            await first

    asyncio.run(_run())


# --- L8: model_stream timing uses the monotonic clock ---------------------------


def test_l8_model_stream_uses_only_monotonic_clock(monkeypatch) -> None:
    assert "perf_counter" not in inspect.getsource(model_stream)

    monkeypatch.setattr(time, "monotonic", lambda: 1_000.0)
    monkeypatch.setattr(time, "perf_counter", lambda: 900_000_000.0)

    captured: dict[str, float] = {}

    async def _fake_loop(graph_state, deps, **kwargs):
        captured["start_time"] = kwargs["start_time"]
        graph_state.final_result = {"status": "completed"}
        return {"chunks": []}

    monkeypatch.setattr(model_stream, "run_model_stream_loop", _fake_loop)

    state = LoopState(cwd=".")
    harness = SimpleNamespace(
        state=state,
        event_handler=None,
        thinking_start_tag="<think>",
        thinking_end_tag="</think>",
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)
    graph_state = GraphRunState(loop_state=state, thread_id="l8", run_mode="loop")

    asyncio.run(model_stream.process_model_stream(graph_state, deps, messages=[], tools=[]))

    assert captured["start_time"] == 1_000.0


def test_l8_nonstream_first_token_uses_monotonic_clock(monkeypatch) -> None:
    monkeypatch.setattr(time, "monotonic", lambda: 1_000.0)
    monkeypatch.setattr(time, "perf_counter", lambda: 900_000_000.0)

    async def _fake_handle(**kwargs):
        return kwargs["stream_state"], kwargs["first_token_time"]

    async def _fake_flush(**kwargs):
        return None

    monkeypatch.setattr(model_stream, "handle_model_stream_chunk", _fake_handle)
    monkeypatch.setattr(model_stream, "flush_model_stream_buffer", _fake_flush)

    class _Client:
        async def stream_chat(self, **kwargs):
            yield {"type": "chunk", "data": {"choices": [{"delta": {"content": "hi"}}]}}
            yield {"type": "done"}

    harness = SimpleNamespace(
        client=_Client(),
        log=logging.getLogger("test.gap_fixes.l8"),
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(
        model_stream._run_nonstream_model_call(
            GraphRunState(loop_state=LoopState(cwd="."), thread_id="l8-ns", run_mode="loop"),
            GraphRuntimeDeps(harness=harness, event_handler=None),
            harness=harness,
            messages=[],
            tools=[],
            echo_to_stdout=False,
            start_tag="<think>",
            end_tag="</think>",
            start_time=time.monotonic(),
        )
    )

    assert result["first_token_time"] == 1_000.0
