import asyncio
from types import SimpleNamespace

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


def test_small_model_empty_turn_gets_continue_nudge() -> None:
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
    assert harness.state.scratchpad["_small_model_continue_nudges"] == 1
    assert harness.state.recent_messages
    assert harness.state.recent_messages[-1].role == "user"
    assert "Continue from the last concrete step" in harness.state.recent_messages[-1].content


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
