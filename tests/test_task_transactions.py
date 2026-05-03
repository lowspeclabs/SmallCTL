from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.context.frame_compiler import PromptStateFrameCompiler
from smallctl.context.subtasks import ChildRunRequest, SubtaskRunner
from smallctl.graph.progress_guard import _build_progress_stagnation_nudge
from smallctl.graph.state import PendingToolCall
from smallctl.graph.tool_loop_guards import _format_repeated_tool_loop_message
from smallctl.harness import Harness
from smallctl.harness.run_mode import ModeDecisionService
from smallctl.harness.task_transactions import FollowupSignals, classify_followup_transaction
from smallctl.harness.tool_dispatch import chat_mode_tools, dispatch_tool_call
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState
from smallctl.state_schema import WriteSession


def _make_harness(state: LoopState) -> SimpleNamespace:
    harness = SimpleNamespace(
        state=state,
        memory=SimpleNamespace(prime_write_policy=lambda _task: None),
        _initial_phase="explore",
        _configured_planning_mode=False,
        _active_task_scope=None,
        _task_sequence=0,
        _runlog=lambda *args, **kwargs: None,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    return harness


def _tool_schema(name: str) -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {},
        },
    }


def test_classifies_unrelated_task_as_new_task() -> None:
    tx = classify_followup_transaction(
        raw_task="Now build a CLI for this.",
        effective_task="Now build a CLI for this.",
        previous_task="Patch temp/log_report.py",
        signals=FollowupSignals(has_prior_task=True),
    )

    assert tx.turn_type == "NEW_TASK"
    assert tx.reset_policy.keep_prior_result is False


def test_classifies_continue_with_handoff_as_iteration() -> None:
    tx = classify_followup_transaction(
        raw_task="continue",
        effective_task="Patch temp/log_report.py",
        previous_task="Patch temp/log_report.py",
        signals=FollowupSignals(has_prior_task=True, contextual_reference=True),
        allowed_paths=["temp/log_report.py"],
    )

    assert tx.turn_type == "ITERATION"
    assert tx.allowed_paths == ["temp/log_report.py"]
    assert tx.reset_policy.force_fresh_plan is True


def test_classifies_corrective_tool_resteer_as_correction() -> None:
    tx = classify_followup_transaction(
        raw_task="no, use ast_patch instead",
        effective_task="Continue current task: patch temp/log_report.py. User correction: no, use ast_patch instead",
        previous_task="patch temp/log_report.py",
        signals=FollowupSignals(has_prior_task=True, corrective_resteer=True),
        failure_summary="file_patch: exact text did not match",
    )

    assert tx.turn_type == "CORRECTION"
    assert tx.failure_summary.startswith("file_patch")
    assert tx.reset_policy.preserve_guard_context is True


def test_classifies_remote_deployment_clarification() -> None:
    tx = classify_followup_transaction(
        raw_task="it does not have to be nginx",
        effective_task="Continue remote task over SSH. User follow-up: it does not have to be nginx",
        previous_task="Install nginx on root@192.168.1.63",
        signals=FollowupSignals(has_prior_task=True, remote_clarification=True),
    )

    assert tx.turn_type == "CLARIFICATION"
    assert tx.reset_policy.force_fresh_plan is False


def test_classifies_retry_after_guard_error() -> None:
    tx = classify_followup_transaction(
        raw_task="try again, but use SSH this time",
        effective_task="try again, but use SSH this time",
        previous_task="deploy the page",
        signals=FollowupSignals(
            has_prior_task=True,
            guard_failure_context=True,
            retry_language=True,
        ),
    )

    assert tx.turn_type == "RETRY"
    assert tx.reset_policy.preserve_guard_context is True


def test_classifies_explicit_conflicting_target_as_new_task() -> None:
    tx = classify_followup_transaction(
        raw_task="Instead patch temp/new_parser.py",
        effective_task="Instead patch temp/new_parser.py",
        previous_task="Patch temp/log_report.py",
        signals=FollowupSignals(
            has_prior_task=True,
            explicit_conflicting_target=True,
        ),
        allowed_paths=["temp/new_parser.py"],
    )

    assert tx.turn_type == "NEW_TASK"
    assert tx.previous_task_relevance == "low"


def test_transaction_payload_is_clipped_and_json_safe() -> None:
    tx = classify_followup_transaction(
        raw_task="continue",
        effective_task="x" * 600,
        previous_task="prior",
        signals=FollowupSignals(has_prior_task=True, contextual_reference=True),
        allowed_paths=["temp/app.py", {"path": "odd"}],
        allowed_artifacts=["A1"],
        failure_summary="failure " * 80,
    )
    payload = tx.to_dict()

    assert len(payload["user_goal"]) <= 320
    assert len(payload["failure_summary"]) <= 220
    assert payload["allowed_paths"][1].startswith("{")
    assert payload["reset_policy"]["force_fresh_plan"] is True


def test_task_boundary_stores_transaction_and_drops_stale_plan_for_iteration() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Patch temp/log_report.py"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.working_memory.plan = ["rewrite the whole module"]
    state.working_memory.next_actions = ["rerun old verifier"]
    state.scratchpad["_last_task_text"] = prior
    state.scratchpad["_last_task_handoff"] = {
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/log_report.py"],
        "last_good_artifact_ids": ["A1002"],
    }
    state.scratchpad["_tool_attempt_history"] = [{"tool_name": "file_read"}]
    harness = _make_harness(state)

    Harness._maybe_reset_for_new_task(
        harness,
        "Continue current task: Patch temp/log_report.py. User follow-up: also add JSON output",
        raw_task="also add JSON output",
    )

    tx = state.scratchpad["_task_transaction"]
    assert tx["turn_type"] == "ITERATION"
    assert "temp/log_report.py" in tx["allowed_paths"]
    assert tx["allowed_artifacts"] == ["A1002"]
    assert state.working_memory.plan == []
    assert state.working_memory.next_actions == []
    assert "_tool_attempt_history" not in state.scratchpad


def test_task_boundary_does_not_keep_transaction_for_unrelated_new_task() -> None:
    state = LoopState(cwd="/tmp")
    prior = "Patch temp/log_report.py"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.working_memory.plan = ["old plan"]
    state.scratchpad["_last_task_text"] = prior
    state.scratchpad["_last_task_handoff"] = {
        "effective_task": prior,
        "current_goal": prior,
        "target_paths": ["temp/log_report.py"],
    }
    harness = _make_harness(state)

    Harness._maybe_reset_for_new_task(
        harness,
        "Build a new tool at temp/new_cli.py",
        raw_task="Build a new tool at temp/new_cli.py",
    )

    assert "_task_transaction" not in state.scratchpad
    assert state.working_memory.plan == []


def test_resolve_remote_clarification_stores_transaction_and_mode_stays_chat() -> None:
    state = LoopState(cwd="/tmp")
    prior = "ssh into 192.168.1.63 and install a task tracker docker container"
    state.run_brief.original_task = prior
    state.working_memory.current_goal = prior
    state.current_phase = "execute"
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
    }
    harness = _make_harness(state)
    harness.client = SimpleNamespace(model="gemma-4-e2b-it")
    harness._current_user_task = lambda: "the image does not have to be called exactly task tracker"
    harness.registry = SimpleNamespace(
        export_openai_tools=lambda **kwargs: [
            _tool_schema("ssh_exec"),
            _tool_schema("ssh_file_write"),
            _tool_schema("task_complete"),
            _tool_schema("task_fail"),
        ],
        get=lambda _name: None,
    )
    Harness._store_task_handoff(harness, raw_task=prior, effective_task=prior)

    raw = "the image does not have to be called exactly task tracker"
    resolved = Harness._resolve_followup_task(harness, raw)
    mode = asyncio.run(ModeDecisionService(harness).decide(raw))
    names = {entry["function"]["name"] for entry in chat_mode_tools(harness)}

    assert resolved.startswith("Continue remote task over SSH")
    assert state.scratchpad["_task_transaction"]["turn_type"] == "CLARIFICATION"
    assert mode == "chat"
    assert "ssh_exec" not in names
    assert names <= {"task_complete", "task_fail"}


def test_prompt_frame_renders_run_boundary_from_transaction() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_transaction"] = {
        "status": "active",
        "turn_type": "ITERATION",
        "user_goal": "Add JSON output to the existing log parser.",
        "success_condition": "Existing script supports JSON output and focused tests pass.",
        "allowed_paths": ["temp/log_report.py"],
        "allowed_artifacts": ["A1002"],
        "ignored_context": ["old plan steps", "old verifier commands"],
    }

    frame = PromptStateFrameCompiler().compile(state=state)
    content = frame.spine.working_memory_text

    assert "Run boundary:" in content
    assert "Current turn type: ITERATION." in content
    assert "temp/log_report.py" in content
    assert "old plan steps" in content
    assert "_tool_attempt_history" not in content


def test_prompt_frame_does_not_render_boundary_for_fresh_session() -> None:
    state = LoopState(cwd="/tmp")

    frame = PromptStateFrameCompiler().compile(state=state)

    assert "Run boundary:" not in frame.spine.working_memory_text


def test_patch_first_blocks_iteration_file_write_to_existing_file(tmp_path) -> None:
    target = tmp_path / "app.py"
    target.write_text("print('old')\n", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    state.scratchpad["_task_transaction"] = {
        "turn_type": "ITERATION",
        "allowed_paths": ["app.py"],
    }
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: {"file_write", "file_patch"}),
        dispatcher=SimpleNamespace(dispatch=lambda _name, _args: None),
        _current_user_task=lambda: "add one line to app.py",
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(dispatch_tool_call(harness, "file_write", {"path": "app.py", "content": "new\n"}))

    assert result.success is False
    assert result.status == "recoverable"
    assert result.metadata["suggested_tool"] == "file_patch"
    assert state.scratchpad["_patch_first_blocked_write"]["path"] == "app.py"


def test_patch_first_allows_new_file_creation(tmp_path) -> None:
    async def _dispatch(tool_name: str, args: dict) -> ToolEnvelope:
        return ToolEnvelope(success=True, output={"tool": tool_name, "args": args})

    state = LoopState(cwd=str(tmp_path))
    state.scratchpad["_task_transaction"] = {"turn_type": "ITERATION"}
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: {"file_write", "file_patch"}),
        dispatcher=SimpleNamespace(dispatch=_dispatch),
        _current_user_task=lambda: "add a helper file",
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(dispatch_tool_call(harness, "file_write", {"path": "new.py", "content": ""}))

    assert result.success is True


def test_patch_first_allows_active_write_session_chunking(tmp_path) -> None:
    async def _dispatch(tool_name: str, args: dict) -> ToolEnvelope:
        return ToolEnvelope(success=True, output={"tool": tool_name, "args": args})

    target = tmp_path / "app.py"
    target.write_text("old\n", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    state.write_session = WriteSession(
        write_session_id="ws1",
        write_target_path="app.py",
        write_session_mode="chunked_author",
        status="open",
    )
    state.scratchpad["_task_transaction"] = {"turn_type": "ITERATION", "allowed_paths": ["app.py"]}
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: {"file_write", "file_patch"}),
        dispatcher=SimpleNamespace(dispatch=_dispatch),
        _current_user_task=lambda: "continue writing app.py",
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(
        dispatch_tool_call(
            harness,
            "file_write",
            {"path": "app.py", "write_session_id": "ws1", "content": "chunk\n"},
        )
    )

    assert result.success is True


def test_patch_first_blocks_known_remote_file_write() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_transaction"] = {
        "turn_type": "CORRECTION",
        "allowed_paths": ["/var/www/html/index.html"],
    }
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: {"ssh_file_write", "ssh_file_patch"}),
        dispatcher=SimpleNamespace(dispatch=lambda _name, _args: None),
        _current_user_task=lambda: "fix the footer",
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(
        dispatch_tool_call(
            harness,
            "ssh_file_write",
            {"path": "/var/www/html/index.html", "content": "<html></html>"},
        )
    )

    assert result.success is False
    assert result.metadata["suggested_tool"] == "ssh_file_patch"


def test_patch_first_allows_explicit_rewrite(tmp_path) -> None:
    async def _dispatch(tool_name: str, args: dict) -> ToolEnvelope:
        return ToolEnvelope(success=True, metadata={"tool_name": tool_name})

    target = tmp_path / "app.py"
    target.write_text("old\n", encoding="utf-8")
    state = LoopState(cwd=str(tmp_path))
    state.scratchpad["_task_transaction"] = {"turn_type": "ITERATION", "allowed_paths": ["app.py"]}
    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: {"file_write", "file_patch"}),
        dispatcher=SimpleNamespace(dispatch=_dispatch),
        _current_user_task=lambda: "rewrite the whole file",
        _runlog=lambda *args, **kwargs: None,
    )

    result = asyncio.run(dispatch_tool_call(harness, "file_write", {"path": "app.py", "content": "new\n"}))

    assert result.success is True


def test_subtask_child_state_uses_transaction_scope() -> None:
    parent = LoopState(cwd="/tmp")
    parent.recent_messages.append(SimpleNamespace(role="user", content="old broad transcript", metadata={}))
    parent.artifacts["A1002"] = ArtifactRecord(
        artifact_id="A1002",
        kind="file_read",
        source="temp/log_report.py",
        created_at="2026-05-01T00:00:00+00:00",
        size_bytes=10,
        summary="log parser",
        tool_name="file_read",
    )
    parent.artifacts["A9999"] = ArtifactRecord(
        artifact_id="A9999",
        kind="file_read",
        source="unrelated.py",
        created_at="2026-05-01T00:00:00+00:00",
        size_bytes=10,
        summary="unrelated",
        tool_name="file_read",
    )
    parent.scratchpad["_task_transaction"] = {
        "turn_type": "CORRECTION",
        "user_goal": "Patch temp/log_report.py with ast_patch",
        "success_condition": "Focused tests pass",
        "allowed_paths": ["temp/log_report.py"],
        "allowed_artifacts": ["A1002"],
        "failure_summary": "file_patch missed exact text",
        "verification_hint": "Run pytest tests/test_log_report.py",
    }

    child = SubtaskRunner().create_child_state(
        parent_state=parent,
        request=ChildRunRequest(brief="old brief", phase="execute"),
    )

    assert child.run_brief.original_task == "Patch temp/log_report.py with ast_patch"
    assert "Focused tests pass" in child.run_brief.acceptance_criteria
    assert child.working_memory.failures == ["file_patch missed exact text"]
    assert child.working_memory.next_actions == ["Run pytest tests/test_log_report.py"]
    assert set(child.artifacts) == {"A1002"}
    assert child.recent_messages == []


def test_subtask_iteration_child_state_gets_delta_and_paths() -> None:
    parent = LoopState(cwd="/tmp")
    parent.scratchpad["_task_transaction"] = {
        "turn_type": "ITERATION",
        "user_goal": "Add JSON output to temp/log_report.py",
        "success_condition": "JSON flag works",
        "allowed_paths": ["temp/log_report.py"],
    }

    child = SubtaskRunner().create_child_state(
        parent_state=parent,
        request=ChildRunRequest(brief="broad old brief", phase="execute"),
    )

    assert child.run_brief.original_task == "Add JSON output to temp/log_report.py"
    assert "allowed_paths=temp/log_report.py" in child.run_brief.constraints
    assert "JSON flag works" in child.run_brief.acceptance_criteria


def test_subtask_new_task_does_not_inherit_prior_parent_plan() -> None:
    parent = LoopState(cwd="/tmp")
    parent.working_memory.plan = ["old parent plan"]
    parent.scratchpad["_task_transaction"] = {
        "turn_type": "NEW_TASK",
        "user_goal": "Build temp/new_cli.py",
        "success_condition": "New CLI runs",
    }

    child = SubtaskRunner().create_child_state(
        parent_state=parent,
        request=ChildRunRequest(brief="old brief", phase="execute"),
    )

    assert child.run_brief.original_task == "Build temp/new_cli.py"
    assert child.working_memory.plan == []
    assert "old parent plan" not in child.run_brief.original_task


def test_progress_stagnation_nudge_includes_transaction_context() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "old goal"
    state.scratchpad["_task_transaction"] = {
        "turn_type": "ITERATION",
        "success_condition": "Focused tests pass.",
    }
    state.scratchpad["_tool_attempt_history"] = [{"tool_name": "file_read"}]
    harness = SimpleNamespace(state=state)

    nudge = _build_progress_stagnation_nudge(harness)

    assert "Current turn type: ITERATION." in nudge
    assert "Current success condition: Focused tests pass." in nudge
    assert "Last stalled action: file_read." in nudge
    assert "Choose exactly one:" in nudge


def test_repeated_tool_message_mentions_transaction_and_repeated_action() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_task_transaction"] = {
        "turn_type": "CORRECTION",
        "success_condition": "Corrected patch is verified.",
    }
    harness = SimpleNamespace(state=state, client=SimpleNamespace(model="qwen2.5-4b-instruct"))
    pending = PendingToolCall(tool_name="file_read", args={"path": "app.py"})

    message = _format_repeated_tool_loop_message(harness, pending, "Guard tripped")

    assert "Current turn type: CORRECTION." in message
    assert "Corrected patch is verified." in message
    assert "Last repeated action: `file_read`." in message
    assert "A. Explain the blocker and stop." in message
