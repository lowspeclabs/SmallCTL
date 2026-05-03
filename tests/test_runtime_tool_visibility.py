from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.harness import Harness
from smallctl.graph.lifecycle_prompt import select_loop_tools, select_planning_tools
from smallctl.harness.tool_visibility import resolve_turn_tool_exposure
from smallctl.harness.tool_dispatch import chat_mode_tools
from smallctl.harness.memory import assess_write_task_complexity
from smallctl.harness.run_mode import ModeDecisionService
from smallctl.models.conversation import ConversationMessage
from smallctl.state import LoopState
from smallctl.state_schema import ExecutionPlan
from smallctl.state_schema import WriteSession
from smallctl.tools.register import build_registry


def _tool_schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {},
        },
    }


def _tool_names(tools: list[dict[str, object]]) -> list[str]:
    return [
        str(entry["function"]["name"])
        for entry in tools
        if isinstance(entry, dict) and isinstance(entry.get("function"), dict)
    ]


def _real_registry_harness(tmp_path, *, task: str, model: str = "qwen2.5-4b-instruct"):
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    harness = SimpleNamespace(
        client=SimpleNamespace(model=model),
        state=state,
        _current_user_task=lambda: task,
        _runlog=lambda *args, **kwargs: None,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    harness.registry = build_registry(harness, registry_profiles={"core"})
    return harness


def _multi_script_build_prompt() -> str:
    return """
1. Build a self-contained Python script at `./temp/file_deduper.py` that recursively scans a local directory, groups files by content hash, reports duplicate sets with total reclaimable space, and includes built-in `unittest` cases for exact duplicates, unique files, empty files, nested directories, and missing-path handling.

2. Build a self-contained Python script at `./temp/job_retry_engine.py` that processes an embedded list of jobs with retry policies, exponential backoff, and final status tracking, prints a concise execution summary, and includes built-in `unittest` cases for immediate success, success after retries, permanent failure, max-retry exhaustion, and stable processing order.

3. Build a self-contained Python script at `./temp/rolling_log_buffer.py` that implements an in-memory fixed-size rolling log buffer with append, filter, and tail operations, supports severity levels, and includes built-in `unittest` cases for overwrite-on-capacity, severity filtering, empty-buffer reads, exact-capacity behavior, and tail length bounds.

4. Build a self-contained Python script at `./temp/dependency_resolver.py` that resolves execution order for tasks with directed dependencies using topological sorting, detects cycles with useful error output, and includes built-in `unittest` cases for linear chains, branching graphs, disconnected tasks, cycle detection, and unknown dependency references.

5. Build a self-contained Python script at `./temp/session_store.py` that implements an in-memory session store with create/get/update/delete/expire operations and TTL-based cleanup, prints a compact state report, and includes built-in `unittest` cases for create-and-read, TTL expiration, update extending life, deleting missing sessions, and multiple-session isolation.
""".strip()


def test_select_loop_tools_hides_runtime_gated_tools_without_index_or_write_session(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("dir_list"),
                _tool_schema("artifact_read"),
                _tool_schema("index_query_symbol"),
                _tool_schema("index_get_definition"),
                _tool_schema("plan_export"),
                _tool_schema("plan_step_update"),
                _tool_schema("finalize_write_session"),
                _tool_schema("process_kill"),
            ]
        ),
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    tools = select_loop_tools(SimpleNamespace(), SimpleNamespace(harness=harness))

    assert _tool_names(tools) == ["dir_list"]

    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert exposure["schemas"] == tools
    assert exposure["names"] == ["dir_list"]


def test_select_loop_tools_exposes_runtime_gated_tools_when_ready(tmp_path) -> None:
    smallctl_dir = tmp_path / ".smallctl"
    smallctl_dir.mkdir()
    (smallctl_dir / "index.db").write_text("", encoding="utf-8")

    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.artifacts = {"A0001": object()}
    state.active_plan = ExecutionPlan(
        plan_id="plan-1234",
        goal="ship feature",
        steps=[],
    )
    state.background_processes = {"123": {"pid": 123, "status": "running"}}
    state.write_session = WriteSession(
        write_session_id="ws_ready",
        write_target_path="src/app.py",
        write_session_mode="chunked_author",
        write_sections_completed=["imports"],
        status="open",
    )
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("dir_list"),
                _tool_schema("artifact_read"),
                _tool_schema("index_query_symbol"),
                _tool_schema("index_get_definition"),
                _tool_schema("plan_export"),
                _tool_schema("plan_step_update"),
                _tool_schema("finalize_write_session"),
                _tool_schema("process_kill"),
            ]
        ),
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    tools = select_loop_tools(SimpleNamespace(), SimpleNamespace(harness=harness))

    assert _tool_names(tools) == [
        "dir_list",
        "artifact_read",
        "index_query_symbol",
        "index_get_definition",
        "plan_export",
        "plan_step_update",
        "finalize_write_session",
        "process_kill",
    ]

    exposure = resolve_turn_tool_exposure(harness, "loop")
    assert exposure["schemas"] == tools
    assert exposure["names"] == _tool_names(tools)


def test_chat_mode_tools_hides_index_queries_until_an_index_exists(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: "inspect this log and tell me what failed",
        _runlog=lambda *args, **kwargs: None,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("artifact_read"),
                _tool_schema("index_query_symbol"),
                _tool_schema("index_get_references"),
            ],
            get=lambda name: None,
        ),
    )

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == ["file_read"]
    exposure = resolve_turn_tool_exposure(harness, "chat")
    assert exposure["schemas"] == tools
    assert exposure["names"] == ["file_read"]


def test_complex_write_chat_exposes_write_tools_for_four_b_models(tmp_path) -> None:
    task = (
        "Build temp/logwatch.py as a Python script with tests, debug mode, "
        "validation for malformed input, dependency ordering logic, and edge cases."
    )
    harness = _real_registry_harness(tmp_path, task=task)

    tools = chat_mode_tools(harness)
    names = set(_tool_names(tools))

    assert {"file_write", "file_patch", "ast_patch"} <= names
    exposure = resolve_turn_tool_exposure(harness, "chat")
    assert {"file_write", "file_patch", "ast_patch"} <= set(exposure["names"])


def test_complex_write_scorer_forces_multi_script_build_prompts(tmp_path) -> None:
    task = _multi_script_build_prompt()

    analysis = assess_write_task_complexity(task, cwd=str(tmp_path))

    assert analysis["force_chunk_mode"] is True
    assert analysis["force_chunk_mode_targets"] == [
        "./temp/file_deduper.py",
        "./temp/job_retry_engine.py",
        "./temp/rolling_log_buffer.py",
        "./temp/dependency_resolver.py",
        "./temp/session_store.py",
    ]
    assert "task requests multiple code targets" in analysis["reasons"]
    assert "script task combines multiple behavior requirements" in analysis["reasons"]


def test_complex_write_scorer_forces_requirement_heavy_script_without_path() -> None:
    task = (
        "Build a self-contained Python script that recursively scans a local directory, "
        "groups files by content hash, reports duplicate sets with total reclaimable space, "
        "and includes built-in unittest cases for exact duplicates, unique files, empty files, "
        "nested directories, and missing-path handling."
    )

    analysis = assess_write_task_complexity(task)

    assert analysis["force_chunk_mode"] is True
    assert analysis["force_chunk_mode_targets"] == []
    assert "task explicitly asks to build a script" in analysis["reasons"]
    assert "script task includes multiple test cases" in analysis["reasons"]


def test_complex_write_heuristic_routes_four_b_models_to_chat() -> None:
    async def _unexpected_stream_chat(*, messages, tools):
        del messages, tools
        raise AssertionError("complex write heuristic should decide before model fallback")
        if False:
            yield {}

    task = (
        "Build temp/logwatch.py as a Python script with tests, debug mode, "
        "validation for malformed input, dependency ordering logic, and edge cases."
    )
    harness = SimpleNamespace(
        client=SimpleNamespace(model="qwen2.5-4b-instruct", stream_chat=_unexpected_stream_chat),
        state=SimpleNamespace(
            planning_mode_enabled=False,
            active_plan=None,
            recent_messages=[],
            cwd="/tmp",
        ),
        _emit=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
    )

    assert asyncio.run(ModeDecisionService(harness).decide(task)) == "chat"


def test_complex_write_heuristic_routes_multi_script_prompt_to_chat(tmp_path) -> None:
    async def _unexpected_stream_chat(*, messages, tools):
        del messages, tools
        raise AssertionError("complex write heuristic should decide before model fallback")
        if False:
            yield {}

    task = _multi_script_build_prompt()
    harness = SimpleNamespace(
        client=SimpleNamespace(model="qwen2.5-4b-instruct", stream_chat=_unexpected_stream_chat),
        state=SimpleNamespace(
            planning_mode_enabled=False,
            active_plan=None,
            recent_messages=[],
            cwd=str(tmp_path),
        ),
        _emit=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
    )

    assert asyncio.run(ModeDecisionService(harness).decide(task)) == "chat"


def test_chat_mode_keeps_write_tools_hidden_for_capability_queries(tmp_path) -> None:
    harness = _real_registry_harness(tmp_path, task="what tools do you have access to?")

    names = set(_tool_names(chat_mode_tools(harness)))

    assert "file_write" not in names
    assert "file_patch" not in names
    assert "ast_patch" not in names


def test_author_write_chat_exposes_write_tools_even_without_chunked_complexity(tmp_path) -> None:
    harness = _real_registry_harness(
        tmp_path,
        task="create a detailed report of your findings",
        model="gemma-4-e2b-it",
    )

    names = set(_tool_names(chat_mode_tools(harness)))

    assert {"file_write", "file_patch", "ast_patch"} <= names


def test_chat_mode_keeps_write_tools_for_active_write_session_followup(tmp_path) -> None:
    harness = _real_registry_harness(tmp_path, task="continue")
    harness.state.write_session = WriteSession(
        write_session_id="ws_active",
        write_target_path="temp/logwatch.py",
        write_session_mode="chunked_author",
        status="open",
    )

    names = set(_tool_names(chat_mode_tools(harness)))

    assert {"file_write", "file_patch", "ast_patch"} <= names


def test_chat_mode_tools_expose_artifact_tools_when_artifacts_exist(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    state.artifacts = {"A0001": object()}
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: "inspect the most recent artifact for failures",
        _runlog=lambda *args, **kwargs: None,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("artifact_read"),
                _tool_schema("artifact_grep"),
            ],
            get=lambda name: None,
        ),
    )

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == ["file_read", "artifact_read", "artifact_grep"]


def test_chat_mode_tools_expose_real_tools_for_capability_queries(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: "what tools do you have access to?",
        _runlog=lambda *args, **kwargs: None,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("shell_exec"),
            ],
            get=lambda name: None,
        ),
    )

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == ["file_read"]
    assert state.scratchpad["_chat_runtime_intent"] == "capability_query"
    assert "_chat_tools_suppressed_reason" not in state.scratchpad


def test_chat_mode_tools_expose_real_tools_for_mode_queries_too(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: "what mode are you in?",
        _runlog=lambda *args, **kwargs: None,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("shell_exec"),
            ],
            get=lambda name: None,
        ),
    )

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == ["file_read"]
    assert state.scratchpad["_chat_runtime_intent"] == "capability_query"
    assert "_chat_tools_suppressed_reason" not in state.scratchpad


def test_chat_mode_tools_keep_terminal_tools_for_non_lookup_chat(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: "hello",
        _runlog=lambda *args, **kwargs: None,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("task_complete"),
                _tool_schema("task_fail"),
            ],
            get=lambda name: None,
        ),
    )

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == ["task_complete", "task_fail"]
    assert state.scratchpad["_chat_tools_exposed"] is True
    assert state.scratchpad["_chat_tools_suppressed_reason"] == "non_lookup_chat_terminal_only"
    exposure = resolve_turn_tool_exposure(harness, "chat")
    assert exposure["schemas"] == tools
    assert exposure["names"] == ["task_complete", "task_fail"]


def test_chat_mode_tools_logs_structured_diagnostic_when_selection_fails(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    runlog: list[tuple[str, dict[str, object]]] = []

    def _raise_export(**kwargs):
        del kwargs
        raise RuntimeError("boom")

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: "what tools do you have access to?",
        _runlog=lambda event, _message, **data: runlog.append((event, data)),
        registry=SimpleNamespace(
            export_openai_tools=_raise_export,
            get=lambda name: None,
        ),
    )

    tools = chat_mode_tools(harness)

    assert tools == []
    assert state.scratchpad["_chat_tools_suppressed_reason"] == "chat_tool_selection_error"
    diagnostic = next(data for event, data in runlog if event == "chat_tool_selection_error")
    assert diagnostic["exception_type"] == "RuntimeError"
    assert diagnostic["phase"] == "tool_export"
    assert diagnostic["mode"] == "chat"
    assert diagnostic["task_excerpt"] == "what tools do you have access to?"
    assert diagnostic["tool_profiles"] == ["core"]
    assert diagnostic["runtime_intent"] == "capability_query"


def test_chat_mode_tools_keep_tools_for_affirmative_followup_after_confirmation(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    task = "read ./portainer_cli.py and recommend updates to that script"
    state.run_brief.original_task = task
    state.working_memory.current_goal = task
    state.recent_messages = [
        ConversationMessage(
            role="assistant",
            content="Would you like me to create an updated version incorporating these recommendations?",
        ),
        ConversationMessage(role="user", content="yes"),
    ]
    state.scratchpad["_last_task_handoff"] = {
        "raw_task": task,
        "effective_task": task,
        "current_goal": task,
    }

    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _initial_phase="explore",
        _configured_planning_mode=False,
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: Harness._current_user_task(harness),
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("artifact_read"),
            ],
            get=lambda name: None,
        ),
    )

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == ["file_read"]


def test_chat_mode_tools_keep_execution_tools_for_pubkey_auth_request(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: "use pubkey auth",
        _runlog=lambda *args, **kwargs: None,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("shell_exec"),
                _tool_schema("ssh_exec"),
            ],
            get=lambda name: None,
        ),
    )

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == ["file_read", "shell_exec"]
    assert state.scratchpad["_chat_tools_exposed"] is True
    assert "_chat_tools_suppressed_reason" not in state.scratchpad


def test_mode_decision_routes_affirmative_remote_followup_to_loop(tmp_path) -> None:
    async def _unexpected_stream_chat(*, messages, tools):
        del messages, tools
        raise AssertionError("affirmative remote continuation should decide before model fallback")
        if False:
            yield {}

    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    prior = "ssh into root@192.168.1.63 and fix the nginx site routing"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.recent_messages = [
        ConversationMessage(
            role="assistant",
            content="I found the remote fixes. Would you like me to apply these fixes now?",
        )
    ]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it", stream_chat=_unexpected_stream_chat),
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _emit=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
        _initial_phase="execute",
        _configured_planning_mode=False,
    )

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, "yes")

    assert resolved.startswith("Continue remote task over SSH on root@192.168.1.63.")
    assert asyncio.run(ModeDecisionService(harness).decide("yes")) == "loop"


def test_mode_decision_routes_approval_style_remote_followup_to_loop(tmp_path) -> None:
    async def _unexpected_stream_chat(*, messages, tools):
        del messages, tools
        raise AssertionError("approval-style remote continuation should decide before model fallback")
        if False:
            yield {}

    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    prior = "ssh into root@192.168.1.63 and fix the nginx site routing"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.recent_messages = [
        ConversationMessage(
            role="assistant",
            content="I can apply the remote fixes next: update the nginx config and restart the service.",
        )
    ]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it", stream_chat=_unexpected_stream_chat),
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _emit=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
        _initial_phase="execute",
        _configured_planning_mode=False,
    )

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    raw = "1) approved, proceed with implementation"
    resolved = Harness._resolve_followup_task(harness, raw)

    assert resolved.startswith("Continue remote task over SSH on root@192.168.1.63.")
    assert asyncio.run(ModeDecisionService(harness).decide(raw)) == "loop"


def test_mode_decision_recovers_remote_plan_approval_when_interrupt_context_is_missing(tmp_path) -> None:
    async def _unexpected_stream_chat(*, messages, tools):
        del messages, tools
        raise AssertionError("remote plan approval should decide before model fallback")
        if False:
            yield {}

    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    prior = (
        'ssh root@192.168.1.89 with password "@S02v1735" and edit '
        "/var/www/demo-site/index.html with dynamic animation and cards"
    )
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.active_plan = ExecutionPlan(
        plan_id="plan-remote",
        goal=prior,
        status="awaiting_approval",
    )
    state.recent_messages = [
        ConversationMessage(
            role="user",
            content="yes",
            metadata={"resumed_from_interrupt": True, "interrupt_kind": "plan_execute_approval"},
        )
    ]
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it", stream_chat=_unexpected_stream_chat),
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _emit=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
        _initial_phase="execute",
        _configured_planning_mode=False,
    )

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    resolved = Harness._resolve_followup_task(harness, "yes")

    assert resolved.startswith("Continue remote task over SSH on root@192.168.1.89.")
    assert asyncio.run(ModeDecisionService(harness).decide("yes")) == "loop"


def test_mode_decision_routes_remote_site_mutation_followup_to_loop(tmp_path) -> None:
    async def _unexpected_stream_chat(*, messages, tools):
        del messages, tools
        raise AssertionError("remote site mutation follow-up should decide before model fallback")
        if False:
            yield {}

    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    prior = (
        'ssh root@192.168.1.89 go to /var/www/demo-site and update index.html '
        'to have a minimal design google theme, make all cards animated password is "@S02v1735"'
    )
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.89": {"host": "192.168.1.89", "user": "root", "confirmed": True}
    }
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it", stream_chat=_unexpected_stream_chat),
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _emit=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
        _initial_phase="execute",
        _configured_planning_mode=False,
    )

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    raw = "remove the google branding"
    resolved = Harness._resolve_followup_task(harness, raw)

    assert resolved.startswith("Continue remote task over SSH on root@192.168.1.89.")
    assert asyncio.run(ModeDecisionService(harness).decide(raw)) == "loop"


def test_chat_mode_tools_keep_ssh_for_affirmative_remote_followup_after_confirmation(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core", "network"]
    prior = "ssh into root@192.168.1.63 and fix the nginx site routing"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.recent_messages = [
        ConversationMessage(
            role="assistant",
            content="I found the remote fixes. Would you like me to apply these fixes now?",
        ),
        ConversationMessage(role="user", content="yes"),
    ]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _initial_phase="execute",
        _configured_planning_mode=False,
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: Harness._current_user_task(harness),
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("ssh_exec"),
                _tool_schema("ssh_file_read"),
                _tool_schema("ssh_file_write"),
                _tool_schema("ssh_file_patch"),
                _tool_schema("ssh_file_replace_between"),
                _tool_schema("task_complete"),
                _tool_schema("task_fail"),
            ],
            get=lambda name: None,
        ),
    )

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == [
        "file_read",
        "ssh_exec",
        "ssh_file_read",
        "ssh_file_write",
        "ssh_file_patch",
        "ssh_file_replace_between",
        "task_complete",
        "task_fail",
    ]
    assert state.scratchpad["_chat_tools_exposed"] is True
    assert "_chat_tools_suppressed_reason" not in state.scratchpad


def test_chat_mode_tools_keep_ssh_for_remote_site_mutation_followup(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core", "network"]
    prior = (
        'ssh root@192.168.1.89 go to /var/www/demo-site and update index.html '
        'to have a minimal design google theme, make all cards animated password is "@S02v1735"'
    )
    raw = "remove the google branding"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.recent_messages = [ConversationMessage(role="user", content=raw)]
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.89": {"host": "192.168.1.89", "user": "root", "confirmed": True}
    }
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _initial_phase="execute",
        _configured_planning_mode=False,
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: Harness._current_user_task(harness),
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("ssh_exec"),
                _tool_schema("ssh_file_read"),
                _tool_schema("ssh_file_write"),
                _tool_schema("ssh_file_patch"),
                _tool_schema("ssh_file_replace_between"),
                _tool_schema("task_complete"),
                _tool_schema("task_fail"),
            ],
            get=lambda name: None,
        ),
    )

    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == [
        "file_read",
        "ssh_exec",
        "ssh_file_read",
        "ssh_file_write",
        "ssh_file_patch",
        "ssh_file_replace_between",
        "task_complete",
        "task_fail",
    ]
    assert state.scratchpad["_chat_tools_exposed"] is True
    assert "_chat_tools_suppressed_reason" not in state.scratchpad


def test_chat_mode_tools_hide_typed_ssh_file_tools_for_local_execute_tasks(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core", "network"]
    harness = SimpleNamespace(
        client=SimpleNamespace(model="gemma-4-e2b-it"),
        state=state,
        _current_user_task=lambda: "run a local shell command to inspect this repo",
        _runlog=lambda *args, **kwargs: None,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("ssh_exec"),
                _tool_schema("ssh_file_read"),
                _tool_schema("ssh_file_write"),
                _tool_schema("ssh_file_patch"),
                _tool_schema("ssh_file_replace_between"),
                _tool_schema("shell_exec"),
            ],
            get=lambda name: None,
        ),
    )

    tools = chat_mode_tools(harness)

    assert _tool_names(tools) == ["file_read"]


def test_select_planning_tools_and_exposure_share_the_same_filtered_names(tmp_path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.current_phase = "execute"
    state.active_tool_profiles = ["core"]
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(
            export_openai_tools=lambda **kwargs: [
                _tool_schema("file_read"),
                _tool_schema("plan_export"),
                _tool_schema("plan_step_update"),
            ]
        ),
    )

    tools = select_planning_tools(SimpleNamespace(), SimpleNamespace(harness=harness))

    assert _tool_names(tools) == ["file_read"]
    exposure = resolve_turn_tool_exposure(harness, "planning")
    assert exposure["schemas"] == tools
    assert exposure["names"] == ["file_read"]
