import asyncio
from pathlib import Path
from types import SimpleNamespace

from smallctl.guards import GuardConfig, check_guards
from smallctl.graph.recovery_context import build_goal_recap
from smallctl.graph.nodes import LoopRoute, interpret_model_output
from smallctl.graph.state import PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_loop_guards import _detect_repeated_tool_loop, _record_tool_attempt
from smallctl.graph.progress_guard import (
    _update_progress_tracking,
    _check_progress_stagnation,
    _turn_has_actionable_progress,
)
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord


class _FakeState:
    def __init__(self) -> None:
        self.scratchpad: dict[str, object] = {"_model_name": "qwen2.5-coder-7b-instruct", "_model_is_small": True}
        self.strategy: dict[str, object] | None = None
        self.current_phase = "explore"
        self.recent_messages: list[object] = []
        self.working_memory = SimpleNamespace(known_facts=[])
        self.planning_mode_enabled = False
        self.run_brief = SimpleNamespace(original_task="")

    def append_message(self, message: object) -> None:
        self.recent_messages.append(message)


class _FakeHarness:
    def __init__(self) -> None:
        self.client = SimpleNamespace(model="qwen2.5-coder-7b-instruct")
        self.state = _FakeState()
        self.log = SimpleNamespace()
        self.summarizer = None
        self.summarizer_client = None
        self.config = SimpleNamespace(min_exploration_steps=1)

    async def _emit(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def _runlog(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def _failure(self, error: str, *, error_type: str, details: dict[str, object] | None = None) -> dict[str, object]:
        return {"error": error, "error_type": error_type, "details": details or {}}

    def _extract_planning_request(self, task: str) -> None:
        del task
        return None


def test_small_model_empty_turn_gets_blank_message_nudge() -> None:
    async def _run() -> tuple[object, object]:
        harness = _FakeHarness()
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[],
            last_assistant_text="",
            last_thinking_text="",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, route

    harness, route = asyncio.run(_run())

    assert route == LoopRoute.NEXT_STEP
    assert harness.state.scratchpad["_blank_message_nudges"] == 1
    assert harness.state.recent_messages
    assert harness.state.recent_messages[-1].role == "user"
    assert "The assistant turn was empty." in harness.state.recent_messages[-1].content


def test_stream_halt_without_done_gets_goal_recap_nudge_for_any_model() -> None:
    async def _run() -> tuple[object, object]:
        harness = _FakeHarness()
        harness.client.model = "gpt-4.1"
        harness.state.scratchpad["_model_name"] = "gpt-4.1"
        harness.state.scratchpad["_last_stream_halted_without_done"] = True
        harness.state.run_brief.original_task = "Run nmap on localhost and report open ports"
        harness.state.run_brief.current_phase_objective = "explore: wait for scan output"
        harness.state.working_memory.current_goal = "Run nmap on localhost and report open ports"
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[],
            last_assistant_text="",
            last_thinking_text="",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, route

    harness, route = asyncio.run(_run())

    assert route == LoopRoute.NEXT_STEP
    assert harness.state.recent_messages
    message = harness.state.recent_messages[-1]
    assert message.role == "user"
    assert message.metadata["recovery_kind"] == "model_halt"
    assert "Goal recap:" in message.content
    assert "Run nmap on localhost and report open ports" in message.content
    assert "explore: wait for scan output" in message.content


def test_goal_recap_omits_stale_task_boundary_goal() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = "Read the latest harness log"
    harness.state.working_memory.current_goal = "hello"
    harness.state.scratchpad["_task_boundary_previous_task"] = "hello"

    recap = build_goal_recap(harness)

    assert recap == "Goal recap: Original task: Read the latest harness log"
    assert "Current goal" not in recap


def test_multiphase_discovery_uses_state_strategy_when_scratchpad_missing() -> None:
    async def _run() -> tuple[object, object, object]:
        harness = _FakeHarness()
        harness.state.strategy = {"thought_architecture": "multi_phase_discovery"}
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[
                PendingToolCall(
                    tool_name="file_write",
                    args={"path": "temp/dependency_resolver.py", "content": "print('hello')\n"},
                )
            ],
            last_assistant_text="",
            last_thinking_text="",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, graph_state, route

    harness, graph_state, route = asyncio.run(_run())

    assert route == LoopRoute.NEXT_STEP
    assert graph_state.pending_tool_calls == []
    assert harness.state.recent_messages
    assert harness.state.recent_messages[-1].role == "user"
    assert "DISCOVERY phase" in harness.state.recent_messages[-1].content


def test_check_guards_uses_the_updated_repeated_action_threshold() -> None:
    state = _FakeState()
    state.step_count = 0
    state.token_usage = 0
    state.recent_errors = []
    state.stagnation_counters = {}
    state.tool_history = ["artifact_print|A0002"] * 5

    assert check_guards(state, GuardConfig()) is None

    state.tool_history.append("artifact_print|A0002")

    assert check_guards(state, GuardConfig()) == (
        "Guard tripped: repeated tool call loop "
        "(artifact_print repeated 6 times with identical args and outcome)"
    )


def test_check_guards_sub4b_repeated_action_adds_directive_hint() -> None:
    state = _FakeState()
    state.scratchpad["_model_name"] = "gemma-2b"
    state.step_count = 0
    state.token_usage = 0
    state.recent_errors = []
    state.stagnation_counters = {}
    state.tool_history = ["file_read|{}|success"] * 6

    guard_error = check_guards(state, GuardConfig())

    assert guard_error is not None
    assert "Guard tripped: repeated tool call loop" in guard_error
    assert "Directive Hint:" in guard_error
    assert "`file_patch`, `ast_patch`, `shell_exec`, or `task_complete`" in guard_error


def test_sub4b_repeated_file_read_loop_includes_directive_hint() -> None:
    harness = _FakeHarness()
    harness.client.model = "qwen3.5:4b"
    harness.state.scratchpad["_model_name"] = "qwen3.5:4b"
    pending = PendingToolCall(tool_name="file_read", args={"path": "temp/logwatch.py"})

    _record_tool_attempt(harness, pending)
    _record_tool_attempt(harness, pending)

    repeat_error = _detect_repeated_tool_loop(harness, pending)

    assert repeat_error is not None
    assert "Guard tripped: repeated tool call loop" in repeat_error
    assert "Directive Hint:" in repeat_error
    assert "`file_patch` or `ast_patch`" in repeat_error
    assert "`shell_exec`" in repeat_error


def test_repeated_artifact_grep_trips_after_three_identical_calls_with_intervening_tools() -> None:
    harness = _FakeHarness()
    harness.client.model = "qwen3.5:4b"
    harness.state.scratchpad["_model_name"] = "qwen3.5:4b"
    pending = PendingToolCall(tool_name="artifact_grep", args={"artifact_id": "A0013", "query": "*"})
    other = PendingToolCall(tool_name="shell_exec", args={"command": "echo ok"})

    _record_tool_attempt(harness, pending)
    _record_tool_attempt(harness, other)
    _record_tool_attempt(harness, pending)

    repeat_error = _detect_repeated_tool_loop(harness, pending)

    assert repeat_error is not None
    assert "artifact_grep repeated 3 times with identical arguments" in repeat_error
    assert "different source/query" in repeat_error


def test_repeated_artifact_read_of_ssh_write_confirmation_stops_early() -> None:
    harness = _FakeHarness()
    harness.state.artifacts = {
        "A0007": ArtifactRecord(
            artifact_id="A0007",
            kind="ssh_file_write",
            source="/var/www/html/index.html",
            created_at="2026-04-30T00:00:00+00:00",
            size_bytes=256,
            summary="remote file written",
            tool_name="ssh_file_write",
            metadata={"path": "/var/www/html/index.html", "changed": True},
        )
    }
    pending = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0007"})

    _record_tool_attempt(harness, pending)
    repeat_error = _detect_repeated_tool_loop(harness, pending)

    assert repeat_error is not None
    assert "successful SSH file mutation artifact already contains the write confirmation" in repeat_error


def test_repeated_artifact_read_past_eof_stops_early(tmp_path: Path) -> None:
    harness = _FakeHarness()
    content_path = tmp_path / "A0011.txt"
    content_path.write_text("one\ntwo\n", encoding="utf-8")
    harness.state.artifacts = {
        "A0011": ArtifactRecord(
            artifact_id="A0011",
            kind="file_read",
            source="/tmp/example.py",
            created_at="2026-04-30T00:00:00+00:00",
            size_bytes=12,
            summary="example.py full file",
            tool_name="file_read",
            content_path=str(content_path),
            metadata={"total_lines": 2},
        )
    }
    pending = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0011", "start_line": 20})

    _record_tool_attempt(harness, pending)
    repeat_error = _detect_repeated_tool_loop(harness, pending)

    assert repeat_error is not None
    assert "artifact_read EOF overread" in repeat_error


def test_artifact_read_new_ranges_still_count_as_progress() -> None:
    harness = _FakeHarness()
    prior = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0012", "start_line": 1, "end_line": 50})
    pending = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0012", "start_line": 51, "end_line": 100})

    _record_tool_attempt(harness, prior)
    repeat_error = _detect_repeated_tool_loop(harness, pending)

    assert repeat_error is None


def _make_graph_state(*, tool_results: list[ToolExecutionRecord] | None = None, assistant_text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        last_tool_results=tool_results or [],
        last_assistant_text=assistant_text,
        last_thinking_text="",
    )


def _make_record(tool_name: str, args: dict, *, success: bool = True, metadata: dict | None = None, changed: bool | None = None) -> ToolExecutionRecord:
    meta = dict(metadata or {})
    if changed is not None:
        meta["changed"] = changed
    return ToolExecutionRecord(
        operation_id=f"op:{tool_name}",
        tool_name=tool_name,
        args=args,
        tool_call_id=None,
        result=ToolEnvelope(success=success, metadata=meta),
    )


def test_repeated_same_artifact_range_is_no_progress() -> None:
    harness = _FakeHarness()
    harness.state.artifacts = {
        "A0001": ArtifactRecord(
            artifact_id="A0001",
            kind="file_read",
            source="/tmp/test.py",
            created_at="2026-04-30T00:00:00+00:00",
            size_bytes=100,
            summary="test file",
            tool_name="file_read",
            metadata={"total_lines": 10},
        )
    }

    # Turn 1: first read of range 1-50 -> progress
    pending1 = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0001", "start_line": 1, "end_line": 50})
    _record_tool_attempt(harness, pending1)
    graph_state1 = _make_graph_state(tool_results=[_make_record("artifact_read", {"artifact_id": "A0001", "start_line": 1, "end_line": 50})])
    _update_progress_tracking(harness, graph_state1)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0

    # Turn 2: same range again -> no progress
    pending2 = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0001", "start_line": 1, "end_line": 50})
    _record_tool_attempt(harness, pending2)
    graph_state2 = _make_graph_state(tool_results=[_make_record("artifact_read", {"artifact_id": "A0001", "start_line": 1, "end_line": 50})])
    _update_progress_tracking(harness, graph_state2)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1

    # Turn 3: same range again -> no progress, counter reaches 2
    pending3 = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0001", "start_line": 1, "end_line": 50})
    _record_tool_attempt(harness, pending3)
    graph_state3 = _make_graph_state(tool_results=[_make_record("artifact_read", {"artifact_id": "A0001", "start_line": 1, "end_line": 50})])
    _update_progress_tracking(harness, graph_state3)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 2


def test_repeated_eof_overread_is_no_progress() -> None:
    harness = _FakeHarness()
    harness.state.artifacts = {
        "A0002": ArtifactRecord(
            artifact_id="A0002",
            kind="file_read",
            source="/tmp/test.py",
            created_at="2026-04-30T00:00:00+00:00",
            size_bytes=20,
            summary="short file",
            tool_name="file_read",
            metadata={"total_lines": 2},
        )
    }

    # Turn 1: read past EOF -> no progress
    pending1 = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0002", "start_line": 20})
    _record_tool_attempt(harness, pending1)
    graph_state1 = _make_graph_state(tool_results=[_make_record("artifact_read", {"artifact_id": "A0002", "start_line": 20})])
    _update_progress_tracking(harness, graph_state1)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1

    # Turn 2: read past EOF again -> no progress
    pending2 = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0002", "start_line": 25})
    _record_tool_attempt(harness, pending2)
    graph_state2 = _make_graph_state(tool_results=[_make_record("artifact_read", {"artifact_id": "A0002", "start_line": 25})])
    _update_progress_tracking(harness, graph_state2)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 2


def test_repeated_assistant_analysis_no_tools_is_no_progress() -> None:
    harness = _FakeHarness()
    text = "Let me analyze the current state before proceeding."

    # Turn 1: new analysis text -> progress
    graph_state1 = _make_graph_state(assistant_text=text)
    _update_progress_tracking(harness, graph_state1)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0

    # Record the assistant message so repeat detection works
    harness.state.append_message(SimpleNamespace(role="assistant", content=text))

    # Turn 2: same analysis text, no tools -> no progress
    graph_state2 = _make_graph_state(assistant_text=text)
    _update_progress_tracking(harness, graph_state2)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1

    # Turn 3: same analysis text again -> no progress
    graph_state3 = _make_graph_state(assistant_text=text)
    _update_progress_tracking(harness, graph_state3)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 2


def test_new_artifact_range_is_progress() -> None:
    harness = _FakeHarness()
    harness.state.artifacts = {
        "A0003": ArtifactRecord(
            artifact_id="A0003",
            kind="file_read",
            source="/tmp/test.py",
            created_at="2026-04-30T00:00:00+00:00",
            size_bytes=500,
            summary="test file",
            tool_name="file_read",
            metadata={"total_lines": 100},
        )
    }

    # Turn 1: read range 1-50 -> progress
    pending1 = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0003", "start_line": 1, "end_line": 50})
    _record_tool_attempt(harness, pending1)
    graph_state1 = _make_graph_state(tool_results=[_make_record("artifact_read", {"artifact_id": "A0003", "start_line": 1, "end_line": 50})])
    _update_progress_tracking(harness, graph_state1)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0

    # Turn 2: read range 51-100 -> progress (new range)
    pending2 = PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0003", "start_line": 51, "end_line": 100})
    _record_tool_attempt(harness, pending2)
    graph_state2 = _make_graph_state(tool_results=[_make_record("artifact_read", {"artifact_id": "A0003", "start_line": 51, "end_line": 100})])
    _update_progress_tracking(harness, graph_state2)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0


def test_three_no_progress_cycles_inject_nudge() -> None:
    harness = _FakeHarness()
    harness.state.current_phase = "author"  # default thresholds
    harness.state.stagnation_counters = {"no_actionable_progress": 3}
    graph_state = _make_graph_state()

    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is None
    assert harness.state.recent_messages
    last_msg = harness.state.recent_messages[-1]
    assert last_msg.role == "user"
    assert "no actionable progress" in last_msg.content.lower()
    assert last_msg.metadata.get("recovery_kind") == "no_actionable_progress"


def test_five_no_progress_cycles_trip_guard() -> None:
    harness = _FakeHarness()
    harness.state.current_phase = "author"  # default thresholds
    harness.state.stagnation_counters = {"no_actionable_progress": 5}
    graph_state = _make_graph_state()

    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is not None
    assert "Progress stagnation guard tripped" in guard
    assert "no actionable progress made in 5 steps" in guard


def test_explore_phase_uses_higher_stagnation_thresholds() -> None:
    harness = _FakeHarness()
    harness.state.current_phase = "explore"
    harness.state.stagnation_counters = {"no_actionable_progress": 4}
    graph_state = _make_graph_state()

    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is None
    assert not harness.state.recent_messages

    harness.state.stagnation_counters = {"no_actionable_progress": 5}
    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is None
    assert harness.state.recent_messages
    last_msg = harness.state.recent_messages[-1]
    assert last_msg.metadata.get("recovery_kind") == "no_actionable_progress"

    harness.state.recent_messages.clear()
    harness.state.stagnation_counters = {"no_actionable_progress": 7}
    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is not None
    assert "Progress stagnation guard tripped" in guard
    assert "no actionable progress made in 7 steps" in guard


def test_successful_mutation_with_changed_resets_counter() -> None:
    harness = _FakeHarness()
    harness.state.stagnation_counters = {"no_actionable_progress": 2}

    graph_state = _make_graph_state(
        tool_results=[_make_record("ssh_file_write", {"path": "/tmp/x", "content": "hi"}, changed=True)],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0


def test_mutation_without_changed_does_not_count_as_progress() -> None:
    harness = _FakeHarness()

    graph_state = _make_graph_state(
        tool_results=[_make_record("ssh_file_write", {"path": "/tmp/x", "content": "hi"}, changed=False)],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_new_verifier_verdict_counts_as_progress() -> None:
    harness = _FakeHarness()
    harness.state.scratchpad["_progress_prior_verdict"] = "fail"

    graph_state = _make_graph_state(
        tool_results=[_make_record("shell_exec", {"command": "pytest"}, metadata={"verdict": "pass"})],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0
    assert harness.state.scratchpad.get("_progress_prior_verdict") == "pass"


def test_same_verifier_verdict_does_not_count_as_progress() -> None:
    harness = _FakeHarness()
    harness.state.scratchpad["_progress_prior_verdict"] = "fail"

    graph_state = _make_graph_state(
        tool_results=[_make_record("shell_exec", {"command": "pytest"}, metadata={"verdict": "fail"})],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_task_complete_counts_as_progress() -> None:
    harness = _FakeHarness()
    harness.state.stagnation_counters = {"no_actionable_progress": 2}

    graph_state = _make_graph_state(
        tool_results=[_make_record("task_complete", {"message": "done"})],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0


def test_continue_resets_no_actionable_progress_counter() -> None:
    harness = _FakeHarness()
    harness.state.stagnation_counters = {"no_actionable_progress": 3}
    harness.state.scratchpad["_progress_read_history"] = [
        {"tool_name": "artifact_read", "artifact_id": "A0001"}
    ]
    harness.state.scratchpad["_progress_prior_verdict"] = "fail"
    harness.state.scratchpad["_progress_prior_plan_step"] = "step_1"

    # Simulate the reset that initialize_loop_run / resume_loop_run perform
    # for continue-like tasks.
    harness.state.stagnation_counters.pop("no_actionable_progress", None)
    harness.state.scratchpad.pop("_progress_read_history", None)
    harness.state.scratchpad.pop("_progress_prior_verdict", None)
    harness.state.scratchpad.pop("_progress_prior_plan_step", None)

    assert "no_actionable_progress" not in harness.state.stagnation_counters
    assert "_progress_read_history" not in harness.state.scratchpad
    assert "_progress_prior_verdict" not in harness.state.scratchpad
    assert "_progress_prior_plan_step" not in harness.state.scratchpad


def test_full_file_read_of_new_artifact_counts_as_progress() -> None:
    harness = _FakeHarness()
    # No prior history — first full-file read should count as progress.
    graph_state = _make_graph_state(
        tool_results=[_make_record("artifact_read", {"artifact_id": "A0003"})],
    )
    _update_progress_tracking(harness, graph_state)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0


def test_repeated_full_file_read_of_same_artifact_is_no_progress() -> None:
    harness = _FakeHarness()
    # Seed history with a prior full-file read of the same artifact.
    harness.state.scratchpad["_progress_read_history"] = [
        {"tool_name": "artifact_read", "artifact_id": "A0003"}
    ]
    graph_state = _make_graph_state(
        tool_results=[_make_record("artifact_read", {"artifact_id": "A0003"})],
    )
    _update_progress_tracking(harness, graph_state)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_full_file_read_after_range_read_counts_as_progress() -> None:
    harness = _FakeHarness()
    # Prior range read of the artifact.
    harness.state.scratchpad["_progress_read_history"] = [
        {"tool_name": "artifact_read", "artifact_id": "A0003", "start_line": 1, "end_line": 50}
    ]
    # Full-file read is a different "range" → should count as progress.
    graph_state = _make_graph_state(
        tool_results=[_make_record("artifact_read", {"artifact_id": "A0003"})],
    )
    _update_progress_tracking(harness, graph_state)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0


def test_range_read_after_full_file_read_counts_as_progress() -> None:
    harness = _FakeHarness()
    # Prior full-file read of the artifact.
    harness.state.scratchpad["_progress_read_history"] = [
        {"tool_name": "artifact_read", "artifact_id": "A0003"}
    ]
    # Range read is a different "range" → should count as progress.
    graph_state = _make_graph_state(
        tool_results=[_make_record("artifact_read", {"artifact_id": "A0003", "start_line": 1, "end_line": 50})],
    )
    _update_progress_tracking(harness, graph_state)
    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0
