from __future__ import annotations

import logging
from typing import Any
from types import SimpleNamespace

import pytest

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.interpret_nodes import interpret_model_output
from smallctl.graph.lifecycle_tool_validation import _validate_pending_tool_calls
from smallctl.graph.routing import LoopRoute
from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.graph.tool_execution_nodes import dispatch_tools
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.base import ToolSpec, build_tool_schema
from smallctl.tools.registry import ToolRegistry
from smallctl.tools.tool_call_repair import repair_tool_call_args, validate_tool_args


def _spec(name: str, properties: dict[str, Any], required: list[str] | None = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        schema=build_tool_schema(properties=properties, required=required or []),
        handler=lambda **kwargs: kwargs,
    )


def test_valid_args_are_unchanged() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}}, ["path"])
    args = {"path": "src/app.py"}

    result = repair_tool_call_args(spec, args)

    assert result.valid_initially is True
    assert result.repaired is False
    assert result.args is args


def test_optional_null_is_omitted_only_after_validation_fails() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])

    result = repair_tool_call_args(spec, {"path": "a.py", "start_line": None})

    assert result.valid_initially is False
    assert result.valid_after_repair is True
    assert result.args == {"path": "a.py"}
    assert [action.kind for action in result.actions] == ["null_optional_to_omit"]


def test_required_null_is_not_omitted() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}}, ["path"])

    result = repair_tool_call_args(spec, {"path": None})

    assert result.valid_after_repair is False
    assert result.repaired is False
    assert result.args == {"path": None}
    assert result.actions == []


def test_stringified_array_parses_only_for_array_field() -> None:
    spec = _spec("index_write_import", {"symbols": {"type": "array", "items": {"type": "string"}}}, ["symbols"])

    result = repair_tool_call_args(spec, {"symbols": '["A", "B"]'})

    assert result.valid_after_repair is True
    assert result.args == {"symbols": ["A", "B"]}
    assert result.actions[0].kind == "json_string_to_array"


def test_bare_string_wraps_only_for_allowlisted_array_field() -> None:
    allowed = _spec("index_write_import", {"symbols": {"type": "array", "items": {"type": "string"}}}, ["symbols"])
    blocked = _spec("other_tool", {"symbols": {"type": "array", "items": {"type": "string"}}}, ["symbols"])

    assert repair_tool_call_args(allowed, {"symbols": "Thing"}).args == {"symbols": ["Thing"]}
    assert repair_tool_call_args(blocked, {"symbols": "Thing"}).valid_after_repair is False


def test_markdown_path_unwraps_only_degenerate_path_links() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}}, ["path"])

    result = repair_tool_call_args(spec, {"path": "[src/app.py](src/app.py)"})

    assert result.valid_after_repair is True
    assert result.args == {"path": "src/app.py"}
    assert result.actions[0].kind == "markdown_link_to_path"


def test_markdown_links_in_content_fields_are_not_touched() -> None:
    spec = _spec("file_write", {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"])
    args = {"path": "a.md", "content": "[label](target)"}

    result = repair_tool_call_args(spec, args)

    assert result.valid_initially is True
    assert result.args is args
    assert result.args["content"] == "[label](target)"


def test_wrapper_unwraps_only_when_nested_args_validate() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}}, ["path"])

    result = repair_tool_call_args(spec, {"args": {"path": "a.py"}})

    assert result.valid_after_repair is True
    assert result.args == {"path": "a.py"}
    assert result.actions[0].kind == "wrong_object_wrapper_unwrap"


def test_extra_fields_are_stripped_only_when_known_fields_validate() -> None:
    spec = _spec("shell_exec", {"command": {"type": "string"}, "timeout_sec": {"type": "integer"}}, ["command"])

    repaired = repair_tool_call_args(spec, {"command": "pwd", "foo": "bar"})
    unrepaired = repair_tool_call_args(spec, {"timeout_sec": 1, "foo": "bar"})

    assert repaired.valid_after_repair is True
    assert repaired.args == {"command": "pwd"}
    assert repaired.stripped_extra_fields == ["foo"]
    assert unrepaired.valid_after_repair is False
    assert unrepaired.args == {"timeout_sec": 1, "foo": "bar"}


def test_nested_array_object_validation_reports_paths() -> None:
    schema = build_tool_schema(
        properties={
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            }
        },
        required=["items"],
    )

    issues = validate_tool_args(schema, {"items": [{"name": 3}]})

    assert [(issue.path, issue.kind) for issue in issues] == [(("items", 0, "name"), "type")]


def test_paired_range_defaults_missing_side() -> None:
    spec = _spec(
        "file_read",
        {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
        ["path"],
    )

    result = repair_tool_call_args(spec, {"path": "a.py", "end_line": 10})

    assert result.valid_after_repair is True
    assert result.args == {"path": "a.py", "start_line": 1, "end_line": 10}
    assert result.actions[0].kind == "missing_paired_range_default"


class _State:
    def __init__(self) -> None:
        self.scratchpad: dict[str, Any] = {}
        self.messages: list[Any] = []
        self.write_session = None
        self.cwd = None
        self.step_count = 0
        self.recent_errors: list[str] = []
        self.strategy = {}
        self.active_tool_profiles = ["core"]
        self.run_brief = SimpleNamespace(original_task="read a file")
        self.current_phase = "execute"

    def append_message(self, message: Any) -> None:
        self.messages.append(message)


class _Harness:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.state = _State()
        self.config = SimpleNamespace(
            schema_validation_max_repair_attempts=2,
            tool_call_repair_enabled=True,
            tool_call_repair_log_only=False,
            tool_call_repair_max_actions_per_call=4,
        )
        self.runlogs: list[tuple[str, str, dict[str, Any]]] = []

    def _runlog(self, event: str, message: str, **data: Any) -> None:
        self.runlogs.append((event, message, data))

    async def _emit(self, event_handler: Any, event: Any) -> None:
        return None

    def _failure(self, message: str, *, error_type: str, details: dict[str, Any]) -> dict[str, Any]:
        return {"error": message, "error_type": error_type, "details": details}


def _registry_with(spec: ToolSpec) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(spec)
    return registry


@pytest.mark.asyncio
async def test_lifecycle_validation_repairs_pending_call_before_dispatch() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _Harness(_registry_with(spec))
    pending = PendingToolCall(tool_name="file_read", args={"path": "[src/app.py](src/app.py)", "start_line": None}, tool_call_id="tc1")
    graph_state = SimpleNamespace(pending_tool_calls=[pending], last_assistant_text="")

    short_circuited = await _validate_pending_tool_calls(harness, graph_state, SimpleNamespace(event_handler=None))

    assert short_circuited is False
    assert pending.args == {"path": "src/app.py"}
    assert pending.parser_metadata["tool_call_repaired"] is True
    assert pending.parser_metadata["tool_call_repair_kinds"] == ["null_optional_to_omit", "markdown_link_to_path"]
    assert harness.state.messages[-1].metadata["recovery_kind"] == "tool_call_repair"
    assert harness.state.scratchpad["_recovery_metrics"]["tool_call_repairs_total"] == 1
    assert "tool_call_repair_applied" in {event for event, _message, _data in harness.runlogs}


@pytest.mark.asyncio
async def test_lifecycle_validation_system_repair_does_not_inject_model_hint() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _Harness(_registry_with(spec))
    pending = PendingToolCall(
        tool_name="file_read",
        args={"path": "a.py", "start_line": None},
        tool_call_id="tc1",
        source="system",
    )
    graph_state = SimpleNamespace(pending_tool_calls=[pending], last_assistant_text="")

    short_circuited = await _validate_pending_tool_calls(harness, graph_state, SimpleNamespace(event_handler=None))

    assert short_circuited is False
    assert pending.args == {"path": "a.py"}
    assert harness.state.messages == []
    assert harness.state.scratchpad["_recovery_metrics"]["tool_call_repairs_total"] == 1


@pytest.mark.asyncio
async def test_lifecycle_validation_tracks_next_call_improvement() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _Harness(_registry_with(spec))
    first = PendingToolCall(tool_name="file_read", args={"path": "a.py", "start_line": None}, tool_call_id="tc1")

    await _validate_pending_tool_calls(harness, SimpleNamespace(pending_tool_calls=[first], last_assistant_text=""), SimpleNamespace(event_handler=None))
    harness.state.step_count = 1
    second = PendingToolCall(tool_name="file_read", args={"path": "a.py"}, tool_call_id="tc2")
    await _validate_pending_tool_calls(harness, SimpleNamespace(pending_tool_calls=[second], last_assistant_text=""), SimpleNamespace(event_handler=None))

    metrics = harness.state.scratchpad["_recovery_metrics"]
    assert metrics["tool_call_repair_next_call_improved_total"] == 1
    assert "_last_tool_call_repair_hint" not in harness.state.scratchpad
    assert "tool_call_repair_next_call_improved" in {event for event, _message, _data in harness.runlogs}


@pytest.mark.asyncio
async def test_lifecycle_validation_tracks_next_call_repeated() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _Harness(_registry_with(spec))

    await _validate_pending_tool_calls(
        harness,
        SimpleNamespace(pending_tool_calls=[PendingToolCall(tool_name="file_read", args={"path": "a.py", "start_line": None})], last_assistant_text=""),
        SimpleNamespace(event_handler=None),
    )
    harness.state.step_count = 1
    await _validate_pending_tool_calls(
        harness,
        SimpleNamespace(pending_tool_calls=[PendingToolCall(tool_name="file_read", args={"path": "b.py", "start_line": None})], last_assistant_text=""),
        SimpleNamespace(event_handler=None),
    )

    metrics = harness.state.scratchpad["_recovery_metrics"]
    assert metrics["tool_call_repair_next_call_repeated_total"] == 1
    assert "tool_call_repair_next_call_repeated" in {event for event, _message, _data in harness.runlogs}


@pytest.mark.asyncio
async def test_lifecycle_validation_invalid_unrepairable_call_uses_schema_nudge() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _Harness(_registry_with(spec))
    pending = PendingToolCall(tool_name="file_read", args={"path": "a.py", "start_line": "not-an-int"}, tool_call_id="tc1")
    graph_state = SimpleNamespace(pending_tool_calls=[pending], last_assistant_text="")

    short_circuited = await _validate_pending_tool_calls(harness, graph_state, SimpleNamespace(event_handler=None))

    assert short_circuited is True
    assert graph_state.pending_tool_calls == []
    assert harness.state.recent_errors == ["Field start_line expected integer but got string."]
    assert harness.state.messages[-1].metadata["recovery_kind"] == "schema_validation"
    assert "schema_validation_repair_decision" in {event for event, _message, _data in harness.runlogs}


@pytest.mark.asyncio
async def test_lifecycle_validation_respects_repair_disabled() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _Harness(_registry_with(spec))
    harness.config.tool_call_repair_enabled = False
    pending = PendingToolCall(tool_name="file_read", args={"path": "a.py", "start_line": None})

    short_circuited = await _validate_pending_tool_calls(harness, SimpleNamespace(pending_tool_calls=[pending], last_assistant_text=""), SimpleNamespace(event_handler=None))

    assert short_circuited is False
    assert pending.args == {"path": "a.py", "start_line": None}
    assert pending.parser_metadata == {}
    assert "_recovery_metrics" not in harness.state.scratchpad


@pytest.mark.asyncio
async def test_lifecycle_validation_log_only_does_not_mutate_or_hint() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _Harness(_registry_with(spec))
    harness.config.tool_call_repair_log_only = True
    pending = PendingToolCall(tool_name="file_read", args={"path": "a.py", "start_line": None})

    short_circuited = await _validate_pending_tool_calls(harness, SimpleNamespace(pending_tool_calls=[pending], last_assistant_text=""), SimpleNamespace(event_handler=None))

    assert short_circuited is False
    assert pending.args == {"path": "a.py", "start_line": None}
    assert pending.parser_metadata == {}
    assert harness.state.messages == []
    assert "tool_call_repair_log_only" in {event for event, _message, _data in harness.runlogs}


@pytest.mark.asyncio
async def test_lifecycle_validation_max_actions_blocks_repair_mutation() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _Harness(_registry_with(spec))
    harness.config.tool_call_repair_max_actions_per_call = 0
    pending = PendingToolCall(tool_name="file_read", args={"path": "a.py", "start_line": None})

    graph_state = SimpleNamespace(pending_tool_calls=[pending], last_assistant_text="")
    short_circuited = await _validate_pending_tool_calls(harness, graph_state, SimpleNamespace(event_handler=None))

    assert short_circuited is True
    assert pending.args == {"path": "a.py", "start_line": None}
    assert pending.parser_metadata == {}
    assert graph_state.pending_tool_calls == []
    assert harness.state.messages[-1].metadata["recovery_kind"] == "schema_validation"
    assert "tool_call_repair_failed" in {event for event, _message, _data in harness.runlogs}


class _InterpretHarness(_Harness):
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.state = LoopState(cwd="/tmp")
        self.state.current_phase = "execute"
        self.state.active_tool_profiles = ["core"]
        self.config = SimpleNamespace(
            schema_validation_max_repair_attempts=2,
            tool_call_repair_enabled=True,
            tool_call_repair_log_only=False,
            tool_call_repair_max_actions_per_call=4,
            min_exploration_steps=0,
        )
        self.runlogs: list[tuple[str, str, dict[str, Any]]] = []
        self.summarizer = None
        self.summarizer_client = None
        self.log = logging.getLogger("test")

    def _extract_planning_request(self, task: str) -> Any:
        return None

    def _record_experience(self, **kwargs: Any) -> None:
        return None


async def _noop_emit(*args: Any, **kwargs: Any) -> None:
    return None


def _make_interpret_harness(registry: ToolRegistry) -> _InterpretHarness:
    harness = _InterpretHarness(registry)
    harness._emit = _noop_emit
    return harness


@pytest.mark.asyncio
async def test_interpret_model_output_repairs_malformed_pending_call() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _make_interpret_harness(_registry_with(spec))
    harness.registry.export_openai_tools = lambda **kwargs: [{"type": "function", "function": {"name": name}} for name in harness.registry.names()]
    pending = PendingToolCall(
        tool_name="file_read",
        args={"path": "[src/app.py](src/app.py)", "start_line": None},
        tool_call_id="tc1",
        source="model",
    )
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-repair",
        run_mode="loop",
        pending_tool_calls=[pending],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = await interpret_model_output(graph_state, deps)

    assert route == LoopRoute.DISPATCH_TOOLS
    assert pending.args == {"path": "src/app.py"}
    assert pending.parser_metadata.get("tool_call_repaired") is True
    assert harness.state.recent_messages[-1].metadata["recovery_kind"] == "tool_call_repair"


@pytest.mark.asyncio
async def test_interpret_model_output_defers_failed_repair_when_valid_sibling_exists() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    harness = _make_interpret_harness(_registry_with(spec))
    harness.registry.export_openai_tools = lambda **kwargs: [{"type": "function", "function": {"name": name}} for name in harness.registry.names()]
    valid = PendingToolCall(tool_name="file_read", args={"path": "a.py"}, tool_call_id="tc1", source="model")
    invalid = PendingToolCall(tool_name="file_read", args={"path": "a.py", "start_line": "not-an-int"}, tool_call_id="tc2", source="model")
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-defer",
        run_mode="loop",
        pending_tool_calls=[valid, invalid],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = await interpret_model_output(graph_state, deps)

    assert route == LoopRoute.DISPATCH_TOOLS
    assert [p.tool_name for p in graph_state.pending_tool_calls] == ["file_read"]
    assert isinstance(harness.state.scratchpad.get("_deferred_schema_validation_repair_messages"), list)


@pytest.mark.asyncio
async def test_dispatch_tools_defense_in_depth_repairs_system_call() -> None:
    spec = _spec("file_read", {"path": {"type": "string"}, "start_line": {"type": "integer"}}, ["path"])
    calls: list[tuple[str, dict[str, Any]]] = []

    class _MockHarness(_InterpretHarness):
        async def _dispatch_tool_call(self, tool_name: str, args: dict[str, Any]) -> Any:
            calls.append((tool_name, args))
            return ToolEnvelope(success=True, output="ok")

    harness = _MockHarness(_registry_with(spec))
    harness._emit = _noop_emit
    harness.registry.export_openai_tools = lambda **kwargs: [{"type": "function", "function": {"name": name}} for name in harness.registry.names()]
    pending = PendingToolCall(
        tool_name="file_read",
        args={"path": "a.py", "start_line": None},
        tool_call_id="tc1",
        source="system",
    )
    graph_state = GraphRunState(
        loop_state=harness.state,
        thread_id="thread-system-repair",
        run_mode="loop",
        pending_tool_calls=[pending],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    await dispatch_tools(graph_state, deps)

    assert calls == [("file_read", {"path": "a.py"})]
    assert pending.parser_metadata.get("tool_call_repaired") is True
