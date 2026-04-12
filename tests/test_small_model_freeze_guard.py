import asyncio
from types import SimpleNamespace

from smallctl.guards import GuardConfig, check_guards
from smallctl.graph.recovery_context import build_goal_recap
from smallctl.graph.nodes import LoopRoute, interpret_model_output
from smallctl.graph.state import PendingToolCall


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
