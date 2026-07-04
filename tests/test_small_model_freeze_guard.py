import asyncio
from pathlib import Path
from types import SimpleNamespace

from smallctl.guards import GuardConfig, check_guards
from smallctl.graph.recovery_context import build_concise_goal_hint, build_goal_recap
from smallctl.graph.chat_progress import should_pause_repeated_tool_loop, build_repeated_tool_loop_interrupt_payload
from smallctl.graph.chat_progress import looks_like_freeze_or_hang as chat_progress_freeze_guard
from smallctl.graph.node_support import looks_like_freeze_or_hang as node_support_freeze_guard
from smallctl.graph.nodes import LoopRoute, interpret_model_output
from smallctl.graph.state import PendingToolCall, ToolExecutionRecord
from smallctl.graph.tool_loop_guards import _detect_repeated_tool_loop, _record_tool_attempt
from smallctl.graph.progress_guard import (
    _update_progress_tracking,
    _check_progress_stagnation,
    _build_progress_stagnation_nudge,
    _next_unread_artifact_line,
)
from smallctl.harness.tool_visibility import filter_tools_for_runtime_state
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState


class _FakeState:
    def __init__(self) -> None:
        self.scratchpad: dict[str, object] = {"_model_name": "qwen2.5-coder-7b-instruct", "_model_is_small": True}
        self.strategy: dict[str, object] | None = None
        self.current_phase = "explore"
        self.recent_messages: list[object] = []
        self.working_memory = SimpleNamespace(known_facts=[])
        self.planning_mode_enabled = False
        self.run_brief = SimpleNamespace(original_task="")
        self.stagnation_counters: dict[str, int] = {}
        self.artifacts: dict[str, object] = {}
        self.last_verifier_verdict: dict[str, object] | None = None
        self.active_tool_profiles: list[str] = ["core"]

    def current_verifier_verdict(self) -> dict[str, object] | None:
        verdict = self.last_verifier_verdict
        if isinstance(verdict, dict) and verdict:
            return verdict
        scratch_verdict = self.scratchpad.get("_last_verifier_verdict")
        return scratch_verdict if isinstance(scratch_verdict, dict) and scratch_verdict else None

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


def test_small_model_empty_turn_gets_non_actionable_prose_nudge() -> None:
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
    assert harness.state.scratchpad["_non_actionable_prose_counts"] == {"": 1}
    assert harness.state.recent_messages
    assert harness.state.recent_messages[-1].role == "user"
    assert harness.state.recent_messages[-1].metadata["recovery_kind"] == "non_actionable_prose"
    assert "did not include a tool call" in harness.state.recent_messages[-1].content


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


def test_exhausted_stream_halt_does_not_finalize_as_no_tool_calls() -> None:
    async def _run() -> tuple[object, object, object]:
        harness = _FakeHarness()
        harness.client.model = "gpt-4.1"
        harness.state.scratchpad["_model_name"] = "gpt-4.1"
        harness.state.scratchpad["_last_stream_halted_without_done"] = True
        harness.state.scratchpad["_last_stream_halt_reason"] = "reasoning_only_stream_stall"
        harness.state.scratchpad["_last_stream_halt_details"] = {"attempt": 3}
        harness.state.scratchpad["_small_model_continue_nudges"] = 2
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[],
            last_assistant_text="Remote file not found.",
            last_thinking_text="I should continue repairing the remote script.",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, graph_state, route

    _harness, graph_state, route = asyncio.run(_run())

    assert route == LoopRoute.FINALIZE
    assert graph_state.final_result["error_type"] == "model_stream_stall"
    assert graph_state.final_result["details"]["halt_reason"] == "reasoning_only_stream_stall"
    assert graph_state.final_result.get("reason") != "no_tool_calls"


def test_gemma_stream_halt_gets_extra_bounded_autocontinue() -> None:
    async def _run() -> tuple[object, object, object]:
        harness = _FakeHarness()
        harness.client.model = "Gemma 4 e2b"
        harness.state.scratchpad["_model_name"] = "Gemma 4 e2b"
        harness.state.scratchpad["_last_stream_halted_without_done"] = True
        harness.state.scratchpad["_last_stream_halt_reason"] = "reasoning_only_stream_stall"
        harness.state.scratchpad["_last_stream_halt_details"] = {"attempt": 3}
        harness.state.scratchpad["_small_model_continue_nudges"] = 2
        harness.state.run_brief.original_task = "Fix the remote backup script"
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[],
            last_assistant_text="Remote file not found.",
            last_thinking_text="I can continue from the last verifier failure.",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, graph_state, route

    harness, graph_state, route = asyncio.run(_run())

    assert route == LoopRoute.NEXT_STEP
    assert graph_state.final_result is None
    assert harness.state.scratchpad["_small_model_continue_nudges"] == 3
    message = harness.state.recent_messages[-1]
    assert message.metadata["recovery_mode"] == "gemma_stream_autocontinue"
    assert message.metadata["max_retry_count"] == 5
    assert "Gemma stream auto-continue" in message.content
    assert "<|channel>thought" in message.content
    assert "Close the reasoning block immediately" in message.content


def test_gemma_stream_halt_tracks_stalls_and_disables_thinking() -> None:
    async def _run() -> tuple[object, object, object]:
        harness = _FakeHarness()
        harness.client.model = "Gemma 4 12b"
        harness.state.scratchpad["_model_name"] = "Gemma 4 12b"
        harness.state.scratchpad["_last_stream_halted_without_done"] = True
        harness.state.scratchpad["_last_stream_halt_reason"] = "reasoning_only_stream_stall"
        harness.state.scratchpad["_last_stream_halt_details"] = {"attempt": 3}
        harness.state.scratchpad["_small_model_continue_nudges"] = 1
        harness.state.scratchpad["_gemma_reasoning_only_stall_count"] = 1
        harness.reasoning_mode = "tags"
        harness.state.run_brief.original_task = "Create a single-file HTML game"
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[],
            last_assistant_text="",
            last_thinking_text="Still reasoning without action.",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, graph_state, route

    harness, graph_state, route = asyncio.run(_run())

    assert route == LoopRoute.NEXT_STEP
    assert harness.state.scratchpad["_gemma_reasoning_only_stall_count"] == 2
    assert harness.reasoning_mode == "off"
    assert harness.state.scratchpad["_thinking_tags_disabled"] is True
    message = harness.state.recent_messages[-1]
    assert "Thinking markers have been disabled for this turn" in message.content


def test_gemma_4_exact_small_it_preserves_reasoning_mode_on_stall() -> None:
    async def _run() -> tuple[object, object, object]:
        harness = _FakeHarness()
        harness.client.model = "gemma-4-e2b-it"
        harness.state.scratchpad["_model_name"] = "gemma-4-e2b-it"
        harness.state.scratchpad["_last_stream_halted_without_done"] = True
        harness.state.scratchpad["_last_stream_halt_reason"] = "reasoning_only_stream_stall"
        harness.state.scratchpad["_last_stream_halt_details"] = {"attempt": 3}
        harness.state.scratchpad["_small_model_continue_nudges"] = 1
        harness.state.scratchpad["_gemma_reasoning_only_stall_count"] = 1
        harness.reasoning_mode = "field"
        harness.state.run_brief.original_task = "Create a single-file HTML game"
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[],
            last_assistant_text="",
            last_thinking_text="Still reasoning without action.",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, graph_state, route

    harness, _graph_state, route = asyncio.run(_run())

    assert route == LoopRoute.NEXT_STEP
    assert harness.state.scratchpad["_gemma_reasoning_only_stall_count"] == 2
    assert harness.reasoning_mode == "field"


def test_gemma_stream_halt_still_stops_after_autocontinue_budget() -> None:
    async def _run() -> tuple[object, object, object]:
        harness = _FakeHarness()
        harness.client.model = "Gemma 4 e2b"
        harness.state.scratchpad["_model_name"] = "Gemma 4 e2b"
        harness.state.scratchpad["_last_stream_halted_without_done"] = True
        harness.state.scratchpad["_last_stream_halt_reason"] = "reasoning_only_stream_stall"
        harness.state.scratchpad["_last_stream_halt_details"] = {"attempt": 6}
        harness.state.scratchpad["_small_model_continue_nudges"] = 5
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[],
            last_assistant_text="Remote file not found.",
            last_thinking_text="Still reasoning without action.",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, graph_state, route

    _harness, graph_state, route = asyncio.run(_run())

    assert route == LoopRoute.FINALIZE
    assert graph_state.final_result["error_type"] == "model_stream_stall"
    assert graph_state.final_result["details"]["halt_reason"] == "reasoning_only_stream_stall"


def test_declared_file_read_without_tool_call_is_synthesized() -> None:
    async def _run() -> tuple[object, object, object]:
        harness = _FakeHarness()
        harness.state.run_brief.original_task = "read ./temp/pong.py, what is the first bug you see in that script?"
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="chat",
            pending_tool_calls=[],
            last_assistant_text="I need to read the file ./temp/pong.py to identify the first bug. Let me call file_read to read this file.",
            last_thinking_text="",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, graph_state, route

    harness, graph_state, route = asyncio.run(_run())

    assert route == LoopRoute.DISPATCH_TOOLS
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_read"
    assert pending.args == {"path": "./temp/pong.py"}
    assert pending.source == "system"
    assert harness.state.scratchpad["_declared_file_read_synthesized"]["path"] == "./temp/pong.py"


def test_goal_recap_omits_stale_task_boundary_goal() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = "Read the latest harness log"
    harness.state.working_memory.current_goal = "hello"
    harness.state.scratchpad["_task_boundary_previous_task"] = "hello"

    recap = build_goal_recap(harness)

    assert recap == "Goal recap: Original task: Read the latest harness log"
    assert "Current goal" not in recap


def test_goal_recap_omits_redundant_phase_focus() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = "Debug the remote backup job"
    harness.state.run_brief.current_phase_objective = "execute: Debug the remote backup job"

    recap = build_goal_recap(harness)

    assert "Original task: Debug the remote backup job" in recap
    assert "Phase focus" not in recap


def test_concise_goal_hint_trims_long_task() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = (
        "You are debugging a broken Linux backup job on a remote host.\n\n"
        "Remote IP: root@192.168.1.64\nPassword: secret\n\n"
        "The backup is supposed to archive /root/source into /root/backups using backup.sh.\n"
        "Tasks:\n1. Inspect the files.\n2. Identify all reasons the backup would fail."
    )

    hint = build_concise_goal_hint(harness, max_chars=80)

    assert hint.startswith("Task: ")
    assert len(hint) <= 80 + len("Task: ")
    assert "..." in hint or len(hint) <= 80 + len("Task: ")
    assert "Remote IP" not in hint


def test_gemma_stream_halt_uses_concise_goal_hint() -> None:
    async def _run() -> tuple[object, object]:
        harness = _FakeHarness()
        harness.client.model = "Gemma 4 12b"
        harness.state.scratchpad["_model_name"] = "Gemma 4 12b"
        harness.state.scratchpad["_last_stream_halted_without_done"] = True
        harness.state.scratchpad["_last_stream_halt_reason"] = "reasoning_only_stream_stall"
        harness.state.scratchpad["_last_stream_halt_details"] = {"attempt": 3}
        harness.state.scratchpad["_small_model_continue_nudges"] = 1
        harness.state.run_brief.original_task = (
            "You are debugging a broken Linux backup job on a remote host.\n\n"
            "Remote IP: root@192.168.1.64\nPassword: secret\n\n"
            "The backup is supposed to archive /root/source into /root/backups using backup.sh."
        )
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[],
            last_assistant_text="",
            last_thinking_text="Still reasoning without action.",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return harness, route

    harness, route = asyncio.run(_run())

    assert route == LoopRoute.NEXT_STEP
    message = harness.state.recent_messages[-1]
    assert message.metadata["recovery_mode"] == "gemma_stream_autocontinue"
    assert "Gemma stream auto-continue" in message.content
    # The full task body should not be duplicated inside the recovery nudge.
    assert "Remote IP" not in message.content
    assert "Password" not in message.content
    # A concise one-line hint is allowed.
    assert "Task:" in message.content


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
    assert harness.state.recent_messages[-1].role == "system"
    assert "DISCOVERY phase" in harness.state.recent_messages[-1].content


def test_check_guards_ignores_recovery_nudge_recent_errors() -> None:
    state = LoopState()
    state.recent_errors = [
        "Blocked file-write repair after a long-running remote SSH command timed out. Continue remotely.",
        "ssh_exec: docker inspect failed",
        "ssh_exec: docker inspect failed again",
        "VERIFIER LOOP HARD STOP: required action class change",
        "ssh_exec: curl failed",
    ]

    assert check_guards(state, GuardConfig(max_consecutive_errors=4)) is None

    state.recent_errors.append("ssh_exec: final verifier failed")
    guard_error = check_guards(state, GuardConfig(max_consecutive_errors=4))

    assert guard_error is not None
    assert "max_consecutive_errors" in guard_error
    assert "Blocked file-write repair" not in guard_error
    assert "VERIFIER LOOP HARD STOP" not in guard_error


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

    assert check_guards(state, GuardConfig()) is None

    state.tool_history.append("file_read|{}|success")

    guard_error = check_guards(state, GuardConfig())

    assert guard_error is not None
    assert "Guard tripped: repeated tool call loop" in guard_error
    assert "file_read repeated 7 times" in guard_error
    assert "Directive Hint:" in guard_error
    assert "`file_patch`, `ast_patch`, `shell_exec`, or `task_complete`" in guard_error


def test_sub4b_repeated_file_read_loop_trips_at_seven_with_directive_hint() -> None:
    harness = _FakeHarness()
    harness.client.model = "qwen3.5:4b"
    harness.state.scratchpad["_model_name"] = "qwen3.5:4b"
    pending = PendingToolCall(tool_name="file_read", args={"path": "temp/logwatch.py"})

    _record_tool_attempt(harness, pending)

    repeat_error = _detect_repeated_tool_loop(harness, pending)

    assert repeat_error is not None
    assert "Guard tripped: repeated full file_read loop" in repeat_error
    assert "repeated full file_read loop" in repeat_error
    assert "Directive Hint:" in repeat_error
    assert "`file_patch` or `ast_patch`" in repeat_error
    assert "`shell_exec`" in repeat_error


def test_file_read_repeat_guard_canonicalizes_relative_and_absolute_paths(tmp_path: Path) -> None:
    harness = _FakeHarness()
    harness.state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "logwatch.py"
    target.parent.mkdir()
    target.write_text("print('ok')\n", encoding="utf-8")
    relative = PendingToolCall(tool_name="file_read", args={"path": "./temp/logwatch.py"})
    absolute = PendingToolCall(tool_name="file_read", args={"path": str(target)})

    _record_tool_attempt(harness, relative)

    repeat_error = _detect_repeated_tool_loop(harness, absolute)

    assert repeat_error is not None
    assert "repeated full file_read loop" in repeat_error


def test_file_read_repeat_guard_allows_same_path_after_verifier(tmp_path: Path) -> None:
    harness = _FakeHarness()
    harness.state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "logwatch.py"
    target.parent.mkdir()
    target.write_text("print('ok')\n", encoding="utf-8")
    read = PendingToolCall(tool_name="file_read", args={"path": "./temp/logwatch.py"})
    verifier = PendingToolCall(tool_name="shell_exec", args={"command": f"python3 {target}"})

    _record_tool_attempt(harness, read)
    _record_tool_attempt(harness, verifier)

    assert _detect_repeated_tool_loop(harness, read) is None


def test_ssh_file_read_not_found_blocks_same_missing_path_retry() -> None:
    harness = _FakeHarness()
    harness.state.scratchpad["_deterministic_read_failures"] = [
        {
            "tool_name": "ssh_file_read",
            "host": "192.168.1.89",
            "user": "root",
            "path": "/var/lib/caddy/Caddyfile",
        }
    ]
    pending = PendingToolCall(
        tool_name="ssh_file_read",
        args={"host": "192.168.1.89", "user": "root", "path": "/var/lib/caddy/Caddyfile"},
    )

    repeat_error = _detect_repeated_tool_loop(harness, pending)

    assert repeat_error is not None
    assert "deterministic missing remote file read" in repeat_error
    assert "ssh_file_read" in repeat_error


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


def test_repeated_loop_status_pauses_with_verifier_guidance() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = "Fix temp/env_sanitizer.py"
    harness.state.last_verifier_verdict = {"verdict": "fail", "command": "python3 temp/env_sanitizer.py"}
    harness.state.scratchpad["_last_verifier_stale_after_mutation"] = {
        "reason": "file_changed_after_verifier",
        "tool_name": "file_patch",
        "paths": ["temp/env_sanitizer.py"],
    }
    pending = PendingToolCall(tool_name="loop_status", args={})
    graph_state = SimpleNamespace(thread_id="thread-1")

    assert should_pause_repeated_tool_loop(harness, pending) is True

    payload = build_repeated_tool_loop_interrupt_payload(
        harness=harness,
        graph_state=graph_state,
        pending=pending,
        repeat_error="Guard tripped: repeated tool call loop",
    )

    assert payload["guard"] == "repeated_tool_loop"
    assert "Repeated loop_status detected" in payload["guidance"]
    assert "last verifier verdict is stale" in payload["guidance"]
    assert "rerun the focused verifier" in payload["guidance"]


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


def _make_record(
    tool_name: str,
    args: dict,
    *,
    success: bool = True,
    metadata: dict | None = None,
    changed: bool | None = None,
    output: object | None = None,
    error: str | None = None,
) -> ToolExecutionRecord:
    meta = dict(metadata or {})
    if changed is not None:
        meta["changed"] = changed
    return ToolExecutionRecord(
        operation_id=f"op:{tool_name}",
        tool_name=tool_name,
        args=args,
        tool_call_id=None,
        result=ToolEnvelope(success=success, metadata=meta, output=output, error=error),
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


def test_file_line_read_after_complete_read_and_failed_verifier_is_no_progress() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.scratchpad["_last_failed_verifier"] = {
        "tool_name": "shell_exec",
        "command": "cd /repo && python3 ./temp/text_chunker.py",
        "summary": ["FAILED (failures=4)"],
        "raw_output": "FAILED (failures=4)",
    }
    harness.state.scratchpad["_progress_read_history"] = [
        {
            "tool_name": "file_read",
            "path": "./temp/text_chunker.py",
            "complete_file": True,
            "file_content_truncated": False,
        }
    ]

    graph_state = _make_graph_state(
        tool_results=[
            _make_record(
                "file_read",
                {"path": "./temp/text_chunker.py", "start_line": 100, "end_line": 145},
                metadata={
                    "path": "./temp/text_chunker.py",
                    "complete_file": False,
                    "truncated": False,
                    "line_start": 100,
                    "line_end": 145,
                    "total_lines": 289,
                },
            )
        ]
    )

    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_grep_does_not_reset_progress_for_mutation_task() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"

    graph_state = _make_graph_state(
        tool_results=[_make_record("grep", {"path": "./temp/game.html", "pattern": "startGame"})]
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_fix_html_task_requires_file_mutation() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = (
        "rca and fix startup bug in /repo/temp/chronoshift-labyrinth.html; use file_patch only"
    )

    graph_state = _make_graph_state(
        tool_results=[_make_record("artifact_grep", {"artifact_id": "A0006", "query": "startGame|onclick"})]
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_mutation_task_file_read_budget_exhaustion_is_no_progress() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    path = "./temp/chronoshift-labyrinth.html"
    harness.state.scratchpad["_progress_read_history"] = [
        {"tool_name": "file_read", "path": path, "start_line": index * 100 + 1, "end_line": (index + 1) * 100}
        for index in range(5)
    ]

    graph_state = _make_graph_state(
        tool_results=[
            _make_record(
                "file_read",
                {"path": path, "start_line": 501, "end_line": 650},
                metadata={"path": path, "line_start": 501, "line_end": 650, "total_lines": 2226},
            )
        ]
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_consecutive_read_only_turns_tracked_for_mutation_task() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = "fix the blackscreen in temp/chronoshift-labyrinth.html"

    for i in range(3):
        graph_state = _make_graph_state(
            tool_results=[_make_record("file_read", {"path": f"temp/file{i}.html"})]
        )
        _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters["consecutive_read_only_turns"] == 3


def test_consecutive_read_only_turns_reset_on_mutation() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = "fix the blackscreen in temp/chronoshift-labyrinth.html"

    graph_state1 = _make_graph_state(
        tool_results=[_make_record("file_read", {"path": "temp/chronoshift-labyrinth.html"})]
    )
    _update_progress_tracking(harness, graph_state1)
    assert harness.state.stagnation_counters["consecutive_read_only_turns"] == 1
    assert not harness.state.scratchpad.get("_read_only_loop_gate_active")

    graph_state2 = _make_graph_state(
        tool_results=[_make_record("file_patch", {"path": "temp/chronoshift-labyrinth.html"}, changed=True)]
    )
    _update_progress_tracking(harness, graph_state2)
    assert harness.state.stagnation_counters["consecutive_read_only_turns"] == 0
    assert not harness.state.scratchpad.get("_read_only_loop_gate_active")


def test_read_only_loop_gate_activates_after_threshold() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = "fix the blackscreen in temp/chronoshift-labyrinth.html"
    harness.state.current_phase = "execute"

    for i in range(6):
        graph_state = _make_graph_state(
            tool_results=[_make_record("file_read", {"path": f"temp/file{i}.html"})]
        )
        _update_progress_tracking(harness, graph_state)
        if i < 5:
            assert not harness.state.scratchpad.get("_read_only_loop_gate_active")

    assert harness.state.scratchpad.get("_read_only_loop_gate_active")
    assert harness.state.stagnation_counters["consecutive_read_only_turns"] == 6


def test_read_only_loop_guard_trips_before_stagnation_for_mutation_task() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = "fix the blackscreen in temp/chronoshift-labyrinth.html"
    harness.state.current_phase = "execute"

    # Six novel file reads: each counts as actionable progress, so stagnation counter stays low,
    # but consecutive_read_only_turns reaches the read-only loop threshold.
    for i in range(6):
        graph_state = _make_graph_state(
            tool_results=[_make_record("file_read", {"path": f"temp/file{i}.html"})]
        )
        _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0
    guard = _check_progress_stagnation(harness, _make_graph_state())
    assert guard is not None
    assert "Read-only loop guard tripped" in guard


def test_consecutive_read_only_turns_not_tracked_for_non_mutation_task() -> None:
    harness = _FakeHarness()
    harness.state.run_brief.original_task = "explain how temp/chronoshift-labyrinth.html works"

    for i in range(6):
        graph_state = _make_graph_state(
            tool_results=[_make_record("file_read", {"path": f"temp/file{i}.html"})]
        )
        _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("consecutive_read_only_turns", 0) == 0
    assert not harness.state.scratchpad.get("_read_only_loop_gate_active")


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


def test_freeze_guard_ignores_current_recorded_assistant_message() -> None:
    harness = _FakeHarness()
    text = "I need to inspect the current state before choosing the next tool."
    harness.state.recent_messages = [
        SimpleNamespace(role="user", content="continue"),
        SimpleNamespace(role="assistant", content=text),
    ]

    assert chat_progress_freeze_guard(harness, text) is False
    assert node_support_freeze_guard(harness, text) is False


def test_freeze_guard_detects_repeated_assistant_message_in_history() -> None:
    harness = _FakeHarness()
    text = "I need to inspect the current state before choosing the next tool."
    harness.state.recent_messages = [
        SimpleNamespace(role="assistant", content=text),
        SimpleNamespace(role="user", content="continue"),
        SimpleNamespace(role="assistant", content=text),
    ]

    assert chat_progress_freeze_guard(harness, text) is True
    assert node_support_freeze_guard(harness, text) is True


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


def test_artifact_read_effective_metadata_tracks_actual_covered_lines() -> None:
    harness = _FakeHarness()

    graph_state1 = _make_graph_state(
        tool_results=[
            _make_record(
                "artifact_read",
                {"artifact_id": "A0100", "start_line": 1, "end_line": 50},
                metadata={
                    "artifact_id": "A0100",
                    "line_start": 1,
                    "line_end": 100,
                    "total_lines": 200,
                    "truncated": True,
                },
            )
        ]
    )
    _update_progress_tracking(harness, graph_state1)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0
    assert _next_unread_artifact_line(harness, "A0100") == 101

    graph_state2 = _make_graph_state(
        tool_results=[
            _make_record(
                "artifact_read",
                {"artifact_id": "A0100", "start_line": 51, "end_line": 100},
                metadata={
                    "artifact_id": "A0100",
                    "line_start": 51,
                    "line_end": 100,
                    "total_lines": 200,
                    "truncated": True,
                },
            )
        ]
    )
    _update_progress_tracking(harness, graph_state2)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1
    assert _next_unread_artifact_line(harness, "A0100") == 101


def test_artifact_read_fully_covered_overlap_is_no_progress() -> None:
    harness = _FakeHarness()

    graph_state1 = _make_graph_state(
        tool_results=[
            _make_record(
                "artifact_read",
                {"artifact_id": "A0101", "start_line": 1, "end_line": 200},
                metadata={
                    "artifact_id": "A0101",
                    "line_start": 1,
                    "line_end": 200,
                    "total_lines": 400,
                    "truncated": True,
                },
            )
        ]
    )
    _update_progress_tracking(harness, graph_state1)

    graph_state2 = _make_graph_state(
        tool_results=[
            _make_record(
                "artifact_read",
                {"artifact_id": "A0101", "start_line": 50, "end_line": 150},
                metadata={
                    "artifact_id": "A0101",
                    "line_start": 50,
                    "line_end": 150,
                    "total_lines": 400,
                    "truncated": True,
                },
            )
        ]
    )
    _update_progress_tracking(harness, graph_state2)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1
    assert _next_unread_artifact_line(harness, "A0101") == 201


def test_artifact_read_next_uncovered_span_progresses_to_complete() -> None:
    harness = _FakeHarness()

    graph_state1 = _make_graph_state(
        tool_results=[
            _make_record(
                "artifact_read",
                {"artifact_id": "A0102", "start_line": 1, "end_line": 300},
                metadata={
                    "artifact_id": "A0102",
                    "line_start": 1,
                    "line_end": 300,
                    "total_lines": 400,
                    "truncated": True,
                },
            )
        ]
    )
    _update_progress_tracking(harness, graph_state1)
    assert _next_unread_artifact_line(harness, "A0102") == 301

    graph_state2 = _make_graph_state(
        tool_results=[
            _make_record(
                "artifact_read",
                {"artifact_id": "A0102", "start_line": 301, "end_line": 400},
                metadata={
                    "artifact_id": "A0102",
                    "line_start": 301,
                    "line_end": 400,
                    "total_lines": 400,
                    "truncated": False,
                },
            )
        ]
    )
    _update_progress_tracking(harness, graph_state2)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0
    assert _next_unread_artifact_line(harness, "A0102") is None
    assert harness.state.scratchpad["_artifact_read_coverage"]["A0102"]["complete"] is True


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


def test_no_progress_nudge_pins_last_failed_verifier_summary() -> None:
    harness = _FakeHarness()
    harness.state.current_phase = "author"
    harness.state.run_brief.original_task = "Fix ./temp/ssh_known_hosts_checker.py and run unit tests"
    stderr = """
ERROR: test_hashed_host_placeholder_handling
  File "/home/stephen/Scripts/Harness-Redo/temp/ssh_known_hosts_checker.py", line 211, in test_hashed_host_placeholder_handling
AttributeError: 'KnownHostsChecker' object has no attribute 'checker'
FAIL: test_malformed_line_skipping
  File "/home/stephen/Scripts/Harness-Redo/temp/ssh_known_hosts_checker.py", line 204, in test_malformed_line_skipping
AssertionError: 0 != 2
FAILED (failures=2, errors=2)
"""
    failed_verifier = _make_graph_state(
        tool_results=[
            _make_record(
                "shell_exec",
                {"command": "python3 -m unittest ./temp/ssh_known_hosts_checker.py -v"},
                success=False,
                output={"stderr": stderr, "stdout": ""},
            )
        ],
    )

    _update_progress_tracking(harness, failed_verifier)
    harness.state.stagnation_counters["no_actionable_progress"] = 3
    guard = _check_progress_stagnation(harness, _make_graph_state())

    assert guard is None
    message = harness.state.recent_messages[-1]
    assert "Last verifier failed: `python3 -m unittest ./temp/ssh_known_hosts_checker.py -v`" in message.content
    assert "AttributeError: 'KnownHostsChecker' object has no attribute 'checker'" in message.content
    assert "AssertionError: 0 != 2" in message.content
    assert "Do not reread unchanged evidence" in message.content


def test_seven_no_progress_cycles_trip_guard() -> None:
    harness = _FakeHarness()
    harness.state.current_phase = "author"  # default thresholds
    harness.state.stagnation_counters = {"no_actionable_progress": 7}
    graph_state = _make_graph_state()

    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is not None
    assert "Progress stagnation guard tripped" in guard
    assert "no actionable progress made in 7 steps" in guard


def test_explore_phase_uses_higher_stagnation_thresholds() -> None:
    harness = _FakeHarness()
    harness.state.current_phase = "explore"
    harness.state.stagnation_counters = {"no_actionable_progress": 2}
    graph_state = _make_graph_state()

    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is None
    assert not harness.state.recent_messages

    harness.state.stagnation_counters = {"no_actionable_progress": 3}
    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is None
    assert harness.state.recent_messages
    last_msg = harness.state.recent_messages[-1]
    assert last_msg.metadata.get("recovery_kind") == "no_actionable_progress"

    harness.state.recent_messages.clear()
    harness.state.stagnation_counters = {"no_actionable_progress": 6}
    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is not None
    assert "Progress stagnation guard tripped" in guard
    assert "no actionable progress made in 6 steps" in guard


def test_remote_scope_uses_remote_aware_stagnation_thresholds() -> None:
    harness = _FakeHarness()
    harness.state.current_phase = "author"
    harness.state.active_intent = "requested_ssh_exec"
    harness.state.stagnation_counters = {"no_actionable_progress": 6}
    graph_state = _make_graph_state()

    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is None
    assert not harness.state.recent_messages

    harness.state.stagnation_counters = {"no_actionable_progress": 7}
    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is None
    assert harness.state.recent_messages
    last_msg = harness.state.recent_messages[-1]
    assert last_msg.metadata.get("cycle_count") == 7

    harness.state.recent_messages.clear()
    harness.state.stagnation_counters = {"no_actionable_progress": 10}
    guard = _check_progress_stagnation(harness, graph_state)

    assert guard is not None
    assert "no actionable progress made in 10 steps" in guard


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


def test_novel_failed_patch_target_mismatch_counts_as_bounded_repair_progress() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.working_memory.current_goal = "patch ./temp/patch_dependency_sim.py"
    harness.state.stagnation_counters = {"no_actionable_progress": 2}

    graph_state = _make_graph_state(
        tool_results=[
            _make_record(
                "file_patch",
                {
                    "path": "./temp/patch_dependency_sim.py",
                    "target_text": "old failing assertion",
                    "replacement_text": "new passing assertion",
                },
                success=False,
                metadata={"path": "./temp/patch_dependency_sim.py", "error_kind": "patch_target_not_found"},
                error="Patch target text was not found.",
            )
        ],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0


def test_repeated_failed_patch_target_mismatch_is_no_progress() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.working_memory.current_goal = "patch ./temp/patch_dependency_sim.py"
    record = _make_record(
        "file_patch",
        {
            "path": "./temp/patch_dependency_sim.py",
            "target_text": "old failing assertion",
            "replacement_text": "new passing assertion",
        },
        success=False,
        metadata={"path": "./temp/patch_dependency_sim.py", "error_kind": "patch_target_not_found"},
        error="Patch target text was not found.",
    )

    _update_progress_tracking(harness, _make_graph_state(tool_results=[record]))
    _update_progress_tracking(harness, _make_graph_state(tool_results=[record]))

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_second_patch_target_not_found_suppresses_file_patch() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.working_memory.current_goal = "patch ./temp/patch_dependency_sim.py"
    record = _make_record(
        "file_patch",
        {
            "path": "./temp/patch_dependency_sim.py",
            "target_text": "old failing assertion",
            "replacement_text": "new passing assertion",
        },
        success=False,
        metadata={"path": "./temp/patch_dependency_sim.py", "error_kind": "patch_target_not_found"},
        error="Patch target text was not found.",
    )

    _update_progress_tracking(harness, _make_graph_state(tool_results=[record]))
    assert harness.state.scratchpad.get("_repeated_tool_loop_suppressed_tool") is None

    _update_progress_tracking(harness, _make_graph_state(tool_results=[record]))

    assert harness.state.scratchpad["_repeated_tool_loop_suppressed_tool"] == "file_patch"
    assert harness.state.scratchpad["_last_file_patch_suppression"]["count"] == 2
    tools = [
        {"type": "function", "function": {"name": "file_patch"}},
        {"type": "function", "function": {"name": "file_read"}},
    ]
    visible_names = {
        tool["function"]["name"]
        for tool in filter_tools_for_runtime_state(tools, state=harness.state, mode="loop")
    }
    assert "file_patch" not in visible_names
    assert "file_read" in visible_names


def test_failed_patch_repair_progress_is_budgeted_per_target() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.working_memory.current_goal = "patch ./temp/patch_dependency_sim.py"

    for index in range(4):
        record = _make_record(
            "file_patch",
            {
                "path": "./temp/patch_dependency_sim.py",
                "target_text": f"old block {index}",
                "replacement_text": f"new block {index}",
            },
            success=False,
            metadata={"path": "./temp/patch_dependency_sim.py", "error_kind": "patch_target_not_found"},
            error="Patch target text was not found.",
        )
        _update_progress_tracking(harness, _make_graph_state(tool_results=[record]))

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_patch_task_memory_update_does_not_count_as_actionable_progress() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.working_memory.current_goal = "patch /var/www/9-b-model.html without overwriting"

    graph_state = _make_graph_state(
        tool_results=[
            _make_record(
                "memory_update",
                {"section": "next_actions", "content": "Next action: apply ssh_file_patch"},
            )
        ],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_patch_task_artifact_grep_does_not_count_as_actionable_progress() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.working_memory.current_goal = "patch /var/www/9-b-model.html without overwriting"

    graph_state = _make_graph_state(
        tool_results=[_make_record("artifact_grep", {"artifact_id": "A0006", "query": "<button|</div>|</p>"})],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_patch_task_stagnation_nudge_names_remote_patch_tools() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.working_memory.current_goal = "patch /var/www/9-b-model.html without overwriting"

    message = _build_progress_stagnation_nudge(harness)

    assert "memory notes, artifact searches, and repeated reads are not progress" in message
    assert "`ssh_file_patch`" in message
    assert "`ssh_file_replace_between`" in message


def test_stagnation_nudge_clarifies_complete_file_read_preview_truncation() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.working_memory.current_goal = "patch ./temp/restart_backoff.py"
    harness.state.scratchpad["_progress_read_history"] = [
        {
            "tool_name": "file_read",
            "path": "./temp/restart_backoff.py",
            "complete_file": True,
            "file_content_truncated": False,
            "total_lines": 202,
        }
    ]

    message = _build_progress_stagnation_nudge(harness)

    assert "fully read `./temp/restart_backoff.py`" in message
    assert "chat display compaction" in message
    assert "not evidence that the file content is missing" in message


def test_stagnation_nudge_clarifies_staged_file_read_is_not_empty_target() -> None:
    harness = _FakeHarness()
    harness.state.active_intent = "requested_file_patch"
    harness.state.working_memory.current_goal = "patch ./temp/patch_dependency_sim.py"
    harness.state.scratchpad["_progress_read_history"] = [
        {
            "tool_name": "file_read",
            "path": "./temp/patch_dependency_sim.py",
            "source_path": "/repo/.smallctl/write_sessions/ws_abc123__patch_dependency_sim__stage.py",
            "read_from_staging": True,
            "complete_file": True,
            "file_content_truncated": False,
            "total_lines": 0,
        }
    ]

    message = _build_progress_stagnation_nudge(harness)

    assert "active write-session staged copy" in message
    assert "not proof that the authoritative target file is empty" in message
    assert "chat display compaction" not in message


def test_new_verifier_verdict_counts_as_progress() -> None:
    harness = _FakeHarness()
    harness.state.scratchpad["_progress_prior_verdict"] = "fail"

    graph_state = _make_graph_state(
        tool_results=[_make_record("shell_exec", {"command": "pytest"}, metadata={"verdict": "pass"})],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0
    assert harness.state.scratchpad.get("_progress_prior_verdict") == "pass"


def test_shell_exec_state_verifier_verdict_counts_as_progress() -> None:
    harness = _FakeHarness()
    harness.state.scratchpad["_progress_prior_verdict"] = "fail"
    harness.state.last_verifier_verdict = {
        "tool": "shell_exec",
        "command": "pytest -q",
        "verdict": "pass",
    }

    graph_state = _make_graph_state(
        tool_results=[_make_record("shell_exec", {"command": "pytest -q"})],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0
    assert harness.state.scratchpad.get("_progress_prior_verdict") == "pass"


def test_successful_non_read_shell_exec_counts_as_progress() -> None:
    harness = _FakeHarness()
    harness.state.stagnation_counters = {"no_actionable_progress": 2}

    graph_state = _make_graph_state(
        tool_results=[_make_record("shell_exec", {"command": "npm install"})],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 0


def test_same_verifier_verdict_does_not_count_as_progress() -> None:
    harness = _FakeHarness()
    harness.state.scratchpad["_progress_prior_verdict"] = "fail"

    graph_state = _make_graph_state(
        tool_results=[_make_record("shell_exec", {"command": "pytest"}, metadata={"verdict": "fail"})],
    )
    _update_progress_tracking(harness, graph_state)

    assert harness.state.stagnation_counters.get("no_actionable_progress", 0) == 1


def test_mutation_after_verifier_marks_verdict_stale() -> None:
    harness = _FakeHarness()
    harness.state.last_verifier_verdict = {"verdict": "fail", "command": "pytest"}

    graph_state = _make_graph_state(
        tool_results=[_make_record("file_patch", {"path": "src/app.py"}, changed=True)],
    )
    _update_progress_tracking(harness, graph_state)

    stale = harness.state.scratchpad.get("_last_verifier_stale_after_mutation")
    assert isinstance(stale, dict)
    assert stale["tool_name"] == "file_patch"
    assert stale["paths"] == ["src/app.py"]


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


def test_repeated_tool_loop_after_nudge_is_blocked_even_with_interleaving() -> None:
    """If a model was already nudged for repeating a specific tool call,
    repeating that exact call again should be blocked immediately,
    even if other tools were called in between."""
    harness = _FakeHarness()
    args = {"command": "curl -s https://example.com", "host": "192.168.1.1", "user": "root", "password": "secret"}
    pending = PendingToolCall(tool_name="ssh_exec", args=args)

    # Simulate that a generic loop nudge was already sent for this exact call.
    import json
    from smallctl.state import json_safe_value
    nudge_key = f"generic_loop:ssh_exec:{json.dumps(json_safe_value(args), sort_keys=True)}"
    harness.state.scratchpad["_generic_loop_nudged"] = nudge_key

    repeat_error = _detect_repeated_tool_loop(harness, pending)

    assert repeat_error is not None
    assert "after prior nudge" in repeat_error
