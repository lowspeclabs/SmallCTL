from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.interpret_nodes import (
    _looks_like_non_actionable_prose,
    _non_actionable_turn_signature,
    interpret_model_output,
)
from smallctl.graph.routing import LoopRoute
from smallctl.graph.state import GraphRunState
from smallctl.state import LoopState


def _make_registry(*tool_names: str):
    names = list(tool_names)
    return SimpleNamespace(
        names=lambda: set(names),
        get=lambda _name: SimpleNamespace(schema={}),
        export_openai_tools=lambda **kwargs: [],
    )


def _make_harness(*, registry=None, task: str = "inspect the workspace"):
    state = LoopState(cwd=str(Path.cwd()))
    state.active_tool_profiles = ["core"]
    state.artifacts = {}
    state.run_brief.original_task = task
    state.working_memory.current_goal = task
    return SimpleNamespace(
        state=state,
        registry=registry,
        config=SimpleNamespace(min_exploration_steps=0),
        summarizer=None,
        summarizer_client=None,
        _runlog=lambda *args, **kwargs: None,
        _emit=lambda *args, **kwargs: None,
        _extract_planning_request=lambda _task: None,
        _record_experience=lambda *args, **kwargs: None,
        _failure=lambda error, error_type="runtime", details=None: {
            "error": error,
            "error_type": error_type,
            "details": details or {},
        },
    )


def test_looks_like_non_actionable_prose() -> None:
    assert _looks_like_non_actionable_prose("") is True
    assert _looks_like_non_actionable_prose("   \n  ") is True
    assert _looks_like_non_actionable_prose("I'll call file_read.") is True
    assert _looks_like_non_actionable_prose("I will check the logs next.") is True
    assert _looks_like_non_actionable_prose("Let me inspect that.") is True
    assert _looks_like_non_actionable_prose("I need to verify the output.") is True
    assert _looks_like_non_actionable_prose("Next I will run the tests.") is True
    assert _looks_like_non_actionable_prose("I'm going to patch the file.") is True
    assert _looks_like_non_actionable_prose("Going to read the config now.") is True
    assert _looks_like_non_actionable_prose("Gonna run the installer.") is True

    assert _looks_like_non_actionable_prose("The file contains a syntax error on line 42.") is False
    assert _looks_like_non_actionable_prose("Done.") is False


def test_turn_signature_is_stable() -> None:
    assert _non_actionable_turn_signature("I'll call file_read.") == "i'll call file_read."
    assert _non_actionable_turn_signature("  I'll   CALL file_read.  ") == "i'll call file_read."


def test_forward_looking_phrase_triggers_non_actionable_prose_nudge() -> None:
    harness = _make_harness()
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-forward-looking",
        run_mode="loop",
    )
    graph_state.last_assistant_text = "I need to check the logs before deciding what to do."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert harness.state.recent_messages
    nudge = harness.state.recent_messages[-1]
    assert nudge.metadata["is_recovery_nudge"] is True
    assert nudge.metadata["recovery_kind"] == "non_actionable_prose"
    assert nudge.metadata["count"] == 1
    assert "Emit ONE concrete next action" in nudge.content
    assert harness.state.scratchpad["_non_actionable_prose_counts"]["i need to check the logs before deciding what to do."] == 1


def test_repeated_same_signature_escalates_at_third_detection() -> None:
    harness = _make_harness()
    text = "I'll check the logs."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    for expected_count in (1, 2, 3):
        graph_state = GraphRunState(
            loop_state=harness.state,
            thread_id=f"thread-repeat-{expected_count}",
            run_mode="loop",
        )
        graph_state.last_assistant_text = text
        route = asyncio.run(interpret_model_output(graph_state, deps))
        assert route == LoopRoute.NEXT_STEP

    nudge = harness.state.recent_messages[-1]
    assert nudge.metadata["recovery_kind"] == "non_actionable_prose"
    assert nudge.metadata["count"] == 3
    assert "stuck repeating intent" in nudge.content
    assert "task_fail" in nudge.content


def test_repeated_detection_suggests_escalate_when_available() -> None:
    harness = _make_harness(registry=_make_registry("file_read", "escalate_to_bigger_model"))
    text = "I'll check the logs."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    for _ in range(3):
        graph_state = GraphRunState(
            loop_state=harness.state,
            thread_id="thread-escalate",
            run_mode="loop",
        )
        graph_state.last_assistant_text = text
        route = asyncio.run(interpret_model_output(graph_state, deps))
        assert route == LoopRoute.NEXT_STEP

    nudge = harness.state.recent_messages[-1]
    assert nudge.metadata["has_escalate"] is True
    assert "escalate_to_bigger_model" in nudge.content
    assert "task_fail" in nudge.content


def test_different_signature_resets_counter() -> None:
    harness = _make_harness()
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    for text in ("I'll check the logs.", "I'll check the logs.", "I will check the logs."):
        graph_state = GraphRunState(
            loop_state=harness.state,
            thread_id="thread-different-signature",
            run_mode="loop",
        )
        graph_state.last_assistant_text = text
        route = asyncio.run(interpret_model_output(graph_state, deps))
        assert route == LoopRoute.NEXT_STEP

    counts = harness.state.scratchpad["_non_actionable_prose_counts"]
    assert counts["i'll check the logs."] == 2
    assert counts["i will check the logs."] == 1


def test_chat_mode_is_not_nudged_for_non_actionable_prose() -> None:
    harness = _make_harness()
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-chat",
        run_mode="chat",
    )
    graph_state.last_assistant_text = "I'll check that for you."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    _route = asyncio.run(interpret_model_output(graph_state, deps))

    assert "_non_actionable_prose_counts" not in harness.state.scratchpad
    assert all(
        message.metadata.get("recovery_kind") != "non_actionable_prose"
        for message in harness.state.recent_messages
    )


def test_planning_mode_is_not_nudged_for_non_actionable_prose() -> None:
    harness = _make_harness()
    harness.state.planning_mode_enabled = True
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-planning",
        run_mode="loop",
    )
    graph_state.last_assistant_text = "I'll draft the plan next."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert harness.state.recent_messages[-1].metadata.get("recovery_kind") != "non_actionable_prose"


def test_readonly_complete_answer_is_not_nudged() -> None:
    harness = _make_harness(task="read the docs")
    harness.state.working_memory.known_facts = ["docs are concise"]
    harness.state.recent_messages.append(SimpleNamespace(role="tool", content="docs content"))
    text = (
        "The docs explain the project purpose and how to get started:\n"
        "- `pip install -e .[dev]`\n"
        "- `pytest` to verify\n"
        "It is ready for use."
    )
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-readonly-complete",
        run_mode="loop",
    )
    graph_state.last_assistant_text = text
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.FINALIZE
    assert "_non_actionable_prose_counts" not in harness.state.scratchpad


def test_action_stall_fires_first_then_non_actionable_prose_on_repeat() -> None:
    harness = _make_harness()
    text = "I'll call file_read to inspect the file."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    # Turn 1: explicit action keyword triggers action_stall.
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-action-then-prose",
        run_mode="loop",
    )
    graph_state.last_assistant_text = text
    route = asyncio.run(interpret_model_output(graph_state, deps))
    assert route == LoopRoute.NEXT_STEP
    assert harness.state.recent_messages[-1].metadata["recovery_kind"] == "action_stall"

    # Turn 2: same forward-looking text, action_stall cap reached -> non_actionable_prose.
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-action-then-prose",
        run_mode="loop",
    )
    graph_state.last_assistant_text = text
    route = asyncio.run(interpret_model_output(graph_state, deps))
    assert route == LoopRoute.NEXT_STEP
    assert harness.state.recent_messages[-1].metadata["recovery_kind"] == "non_actionable_prose"


def test_reasoning_fallback_empty_text_still_gets_blank_message() -> None:
    harness = _make_harness()
    harness.state.scratchpad["_assistant_text_from_reasoning_fallback"] = True
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-reasoning-fallback",
        run_mode="loop",
    )
    graph_state.last_assistant_text = "I'll call file_read."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert harness.state.recent_messages[-1].metadata["recovery_kind"] == "blank_message"
    assert "_non_actionable_prose_counts" not in harness.state.scratchpad


def test_action_stall_nudge_includes_gemma_example_for_small_gemma_4_it() -> None:
    harness = _make_harness(task="ssh root@192.168.1.89 and list docker containers")
    harness.state.scratchpad["_model_name"] = "gemma-4-e2b-it"
    text = "I will use the ssh_exec tool to connect to the remote host."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-gemma-stall",
        run_mode="loop",
    )
    graph_state.last_assistant_text = text
    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    nudge = harness.state.recent_messages[-1]
    assert nudge.metadata["recovery_kind"] == "action_stall"
    assert "GEMMA 4 e2b/e4b EXAMPLE" in nudge.content
    assert '"name":"ssh_exec"' in nudge.content


def test_action_stall_nudge_generic_for_other_models() -> None:
    harness = _make_harness(task="ssh root@192.168.1.89 and list docker containers")
    harness.state.scratchpad["_model_name"] = "qwen3:32b"
    text = "I will use the ssh_exec tool to connect to the remote host."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-generic-stall",
        run_mode="loop",
    )
    graph_state.last_assistant_text = text
    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    nudge = harness.state.recent_messages[-1]
    assert nudge.metadata["recovery_kind"] == "action_stall"
    assert "GEMMA 4 e2b/e4b EXAMPLE" not in nudge.content
