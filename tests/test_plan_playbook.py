from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.nodes import (
    LoopRoute,
    _apply_small_model_authoring_budget,
    _matching_write_session_for_pending,
    dispatch_tools,
    interpret_chat_output,
)
from smallctl.graph.tool_call_parser import (
    _detect_repeated_tool_loop,
    _repair_active_write_session_args,
    allow_repeated_tool_call_once,
)
from smallctl.graph.tool_call_parser import _build_schema_repair_message
from smallctl.graph.runtime import LoopGraphRuntime, ChatGraphRuntime
from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.harness.tool_results import _store_verifier_verdict
from smallctl.prompts import build_system_prompt
from smallctl.state import ArtifactRecord, ExecutionPlan, LoopState, PlanStep, WriteSession
from smallctl.graph.tool_outcomes import _maybe_emit_repair_recovery_nudge, apply_tool_outcomes
from smallctl.graph.tool_outcomes import _shell_workspace_relative_retry_hint
from smallctl.graph.state import ToolExecutionRecord
from smallctl.harness import Harness
from smallctl.models.tool_result import ToolEnvelope
from smallctl.config import resolve_config
from smallctl.ui.display import StatusState
from smallctl.ui.statusbar import StatusBar
from smallctl.tools import fs
from smallctl.tools import shell, network
from smallctl.tools import control, planning
from smallctl.context.artifacts import ArtifactStore


def _make_state() -> LoopState:
    state = LoopState(cwd=str(Path.cwd()))
    state.active_tool_profiles = ["core"]
    state.artifacts = {}
    return state


def _make_open_write_session(
    target: Path,
    *,
    session_id: str = "ws-1",
    intent: str = "replace_file",
) -> SimpleNamespace:
    return SimpleNamespace(
        write_session_id=session_id,
        status="open",
        write_session_mode="chunked_author",
        write_target_path=str(target),
        write_session_intent=intent,
        write_staging_path="",
        write_original_snapshot_path="",
        write_last_attempt_snapshot_path="",
        write_last_staged_hash="",
        write_sections_completed=[],
        write_section_ranges={},
        write_current_section="",
        write_next_section="",
        write_failed_local_patches=0,
        write_pending_finalize=False,
        write_last_verifier={},
        write_target_existed_at_start=False,
        write_first_chunk_at=0.0,
    )


def test_plan_set_creates_a_playbook_artifact(tmp_path: Path) -> None:
    state = _make_state()
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path, "run-1"),
        log=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    result = asyncio.run(
        planning.plan_set(
            goal="Create a small CLI script",
            summary="Break the work into bounded stages.",
            inputs=["A target directory", "The Python runtime"],
            outputs=["A working CLI script", "A short verification test"],
            constraints=["Keep the implementation small", "Avoid shell-heavy reasoning"],
            acceptance_criteria=["The script runs", "The test passes"],
            implementation_plan=["Write the skeleton", "Fill in the logic", "Verify the result"],
            claim_refs=["C1"],
            steps=[
                {"step_id": "P1", "title": "Create file skeleton", "claim_refs": ["C1"]},
                {"step_id": "P2", "title": "Implement functions"},
                {"step_id": "P3", "title": "Debug and verify"},
            ],
            state=state,
            harness=harness,
        )
    )

    assert result["success"] is True
    assert state.plan_artifact_id
    assert state.plan_artifact_id in state.artifacts
    playbook_artifact = state.artifacts[state.plan_artifact_id]
    assert playbook_artifact.kind == "plan_playbook"
    playbook_text = Path(playbook_artifact.content_path).read_text(encoding="utf-8")
    assert "Spec Contract" in playbook_text
    assert "Acceptance Criteria" in playbook_text
    assert "Implementation Order" in playbook_text
    assert "Claim References" in playbook_text
    assert "claims: C1" in playbook_text
    assert result["output"]["artifact_id"] == state.plan_artifact_id


def test_system_prompt_surfaces_playbook_guidance() -> None:
    state = _make_state()
    state.run_brief.original_task = "Write a script"
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="Write a script",
        status="draft",
        claim_refs=["C1"],
        steps=[PlanStep(step_id="P1", title="Create file skeleton", claim_refs=["C1"])],
    )
    state.plan_resolved = True
    state.plan_artifact_id = "A0007"

    prompt = build_system_prompt(state, "execute")

    assert "PLAN PLAYBOOK" in prompt
    assert "A0007" in prompt
    assert "file skeleton" in prompt
    assert "acceptance criteria" in prompt.lower()
    assert "C1" in prompt


def test_task_complete_is_blocked_until_acceptance_is_met() -> None:
    state = _make_state()
    state.run_brief.acceptance_criteria = ["The script runs", "The test passes"]
    state.acceptance_ledger = {"The script runs": "done", "The test passes": "pending"}

    blocked = asyncio.run(control.task_complete("done", state=state))

    assert blocked["success"] is False
    assert blocked["error"]
    assert "pending_acceptance_criteria" in blocked["metadata"]

    state.acceptance_ledger["The test passes"] = "passed"
    allowed = asyncio.run(control.task_complete("done", state=state))

    assert allowed["success"] is True
    assert allowed["output"]["status"] == "complete"


def test_continue_like_followup_accepts_common_typos() -> None:
    harness = Harness.__new__(Harness)

    assert harness._is_continue_like_followup("cntinue")
    assert harness._is_continue_like_followup("cotinue please")


def test_verifier_pass_updates_acceptance_ledger() -> None:
    state = _make_state()
    state.run_brief.acceptance_criteria = ["The script runs"]
    state.acceptance_ledger = {"The script runs": "pending"}

    _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=SimpleNamespace(
            success=True,
            status=None,
            output={"stdout": "ok", "stderr": "", "exit_code": 0},
            error=None,
            metadata={"command": "pytest -q"},
        ),
        arguments={"command": "pytest -q"},
    )

    assert state.acceptance_ledger["The script runs"] == "passed"
    assert state.scratchpad["_contract_phase"] == "execute"


def test_verifier_failure_preserves_failed_shell_output_metadata() -> None:
    state = _make_state()

    verdict = _store_verifier_verdict(
        state,
        tool_name="shell_exec",
        result=SimpleNamespace(
            success=False,
            status=None,
            output=None,
            error="bash: line 1: docker: command not found",
            metadata={
                "command": "ssh root@host \"docker ps\"",
                "output": {
                    "stdout": "",
                    "stderr": "bash: line 1: docker: command not found",
                    "exit_code": 127,
                },
            },
        ),
        arguments={"command": "ssh root@host \"docker ps\""},
    )

    assert verdict is not None
    assert verdict["exit_code"] == 127
    assert verdict["key_stderr"] == "bash: line 1: docker: command not found"
    assert verdict["verdict"] == "fail"


def test_contract_phase_derives_author_for_write_task_without_verifier() -> None:
    state = _make_state()
    state.run_brief.original_task = "Create a Python script in `./temp/dependency_resolver.py`"
    state.working_memory.current_goal = state.run_brief.original_task
    state.scratchpad["_task_target_paths"] = ["./temp/dependency_resolver.py"]

    assert state.contract_phase() == "author"


def test_contract_phase_stays_explore_for_non_authoring_task() -> None:
    state = _make_state()
    state.run_brief.original_task = "Read the latest harness log and summarize the error"
    state.working_memory.current_goal = state.run_brief.original_task

    assert state.contract_phase() == "explore"


def test_repair_cycle_requires_read_before_patch(tmp_path: Path) -> None:
    state = _make_state()
    state.repair_cycle_id = "repair-1"

    target = tmp_path / "example.txt"
    target.write_text("original\n", encoding="utf-8")

    blocked = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="original",
            replacement_text="patched",
            cwd=str(tmp_path),
            state=state,
        )
    )
    assert blocked["success"] is False
    assert "reading the target file before patching" in blocked["error"]
    assert blocked["metadata"]["error_kind"] == "repair_cycle_read_required"

    read_back = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))
    assert read_back["success"] is True

    allowed = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="original",
            replacement_text="patched",
            cwd=str(tmp_path),
            state=state,
        )
    )
    assert allowed["success"] is True
    assert allowed["metadata"]["occurrence_count"] == 1
    assert str(target.resolve()) in state.files_changed_this_cycle
    assert target.read_text(encoding="utf-8") == "patched\n"


def test_file_patch_exact_match_updates_target_file(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("keep\nold value\nkeep\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="old value",
            replacement_text="new value",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["staged_only"] is False
    assert result["metadata"]["requested_path"] == str(target)
    assert result["metadata"]["occurrence_count"] == 1
    assert result["metadata"]["expected_occurrences"] == 1
    assert target.read_text(encoding="utf-8") == "keep\nnew value\nkeep\n"


def test_file_patch_zero_occurrences_fails_without_mutating_file(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="missing",
            replacement_text="replacement",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "patch_target_not_found"
    assert result["metadata"]["actual_occurrences"] == 0
    assert "not found" in result["metadata"]["ambiguity_hint"]
    assert result["metadata"]["target_text_preview"]["preview"] == "missing"
    assert target.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_file_patch_zero_occurrences_in_active_write_session_reports_staged_copy_context(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "session.txt"
    session = _make_open_write_session(target)
    state.write_session = session

    seeded = asyncio.run(
        fs.file_write(
            path=str(target),
            content="alpha beta gamma\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
            section_name="body",
        )
    )
    assert seeded["success"] is True

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="missing",
            replacement_text="replacement",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert "active staged copy" in result["error"]
    assert result["metadata"]["error_kind"] == "patch_target_not_found"
    assert result["metadata"]["staged_only"] is True
    assert result["metadata"]["requested_path"] == str(target)
    assert result["metadata"]["write_session_id"] == session.write_session_id
    assert result["metadata"]["source_path"] == session.write_staging_path


def test_file_patch_occurrence_mismatch_in_active_write_session_reports_staged_copy_context(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "session.txt"
    session = _make_open_write_session(target)
    state.write_session = session

    seeded = asyncio.run(
        fs.file_write(
            path=str(target),
            content="beta beta gamma\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
            section_name="body",
        )
    )
    assert seeded["success"] is True

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="beta",
            replacement_text="gamma",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert "active staged copy" in result["error"]
    assert result["metadata"]["error_kind"] == "patch_occurrence_mismatch"
    assert result["metadata"]["staged_only"] is True
    assert result["metadata"]["requested_path"] == str(target)
    assert result["metadata"]["actual_occurrences"] == 2
    assert result["metadata"]["expected_occurrences"] == 1
    assert result["metadata"]["source_path"] == session.write_staging_path


def test_file_patch_multiple_occurrences_fails_when_expected_single_match(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("beta beta beta\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="beta",
            replacement_text="gamma",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "patch_occurrence_mismatch"
    assert result["metadata"]["actual_occurrences"] == 3
    assert result["metadata"]["expected_occurrences"] == 1
    assert "matched 3 times" in result["metadata"]["ambiguity_hint"]
    assert result["metadata"]["target_text_preview"]["preview"] == "beta"
    assert target.read_text(encoding="utf-8") == "beta beta beta\n"


def test_file_patch_can_replace_multiple_exact_matches_when_requested(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("beta beta beta\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="beta",
            replacement_text="gamma",
            expected_occurrences=3,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["occurrence_count"] == 3
    assert result["metadata"]["expected_occurrences"] == 3
    assert target.read_text(encoding="utf-8") == "gamma gamma gamma\n"


def test_file_patch_uses_active_write_session_staging_and_leaves_target_untouched(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "session.txt"
    session = _make_open_write_session(target)
    state.write_session = session

    seeded = asyncio.run(
        fs.file_write(
            path=str(target),
            content="alpha beta gamma\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
            section_name="body",
        )
    )
    assert seeded["success"] is True

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="beta",
            replacement_text="gamma",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["staged_only"] is True
    assert result["metadata"]["requested_path"] == str(target)
    assert result["metadata"]["write_session_id"] == session.write_session_id
    assert result["metadata"]["occurrence_count"] == 1
    assert target.exists() is False
    assert Path(session.write_staging_path).read_text(encoding="utf-8") == "alpha gamma gamma\n"


def test_file_patch_rejects_session_id_mismatch(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "session.txt"
    state.write_session = _make_open_write_session(target, session_id="ws-correct")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="alpha",
            replacement_text="beta",
            cwd=str(tmp_path),
            state=state,
            write_session_id="ws-wrong",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "session_id_mismatch"
    assert "Session ID mismatch" in result["error"]


def test_file_write_rejects_direct_staging_path_mutation(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "generated_script.py"
    session = _make_open_write_session(target)
    stage = fs._session_stage_path(session.write_session_id, target, str(tmp_path))
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("", encoding="utf-8")
    session.write_staging_path = str(stage)
    state.write_session = session

    result = asyncio.run(
        fs.file_write(
            path=str(stage),
            content="print('hello')\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
            section_name="body",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "write_session_staging_path_used_as_target"
    assert result["metadata"]["target_path"] == str(target)
    assert result["metadata"]["staging_path"] == str(stage)
    assert result["metadata"]["next_required_tool"]["tool_name"] == "file_write"
    assert result["metadata"]["next_required_tool"]["required_arguments"]["path"] == str(target)
    assert "read/verify only" in result["error"]
    assert stage.read_text(encoding="utf-8") == ""


def test_file_patch_rejects_direct_staging_path_mutation_without_session_id(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "generated_script.py"
    session = _make_open_write_session(target)
    state.write_session = session

    seeded = asyncio.run(
        fs.file_write(
            path=str(target),
            content="print('alpha')\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
            section_name="body",
        )
    )
    assert seeded["success"] is True

    stage = Path(session.write_staging_path)
    before = stage.read_text(encoding="utf-8")
    result = asyncio.run(
        fs.file_patch(
            path=str(stage),
            target_text="alpha",
            replacement_text="beta",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "write_session_staging_path_used_as_target"
    assert result["metadata"]["target_path"] == str(target)
    assert result["metadata"]["staging_path"] == str(stage)
    assert result["metadata"]["next_required_tool"]["tool_name"] == "file_patch"
    assert result["metadata"]["next_required_tool"]["required_arguments"]["path"] == str(target)
    assert stage.read_text(encoding="utf-8") == before


def test_file_write_patch_existing_auto_strategy_requires_explicit_choice(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "session.txt"
    target.write_text("alpha\n", encoding="utf-8")
    session = _make_open_write_session(target, intent="patch_existing")
    state.write_session = session

    result = asyncio.run(
        fs.file_write(
            path=str(target),
            content="beta\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
            section_name="body",
        )
    )

    assert result["success"] is False
    assert "file_patch" in result["error"]
    assert "replace_strategy='overwrite'" in result["error"]
    assert "replace_strategy='append'" in result["error"]
    assert "file_read(path='" in result["error"]
    assert "Do not assume earlier chunks were lost" in result["error"]
    assert result["metadata"]["error_kind"] == "patch_existing_requires_explicit_replace_strategy"
    assert result["metadata"]["staged_only"] is True
    assert result["metadata"]["write_session_id"] == session.write_session_id


def test_pending_tool_call_normalizes_file_patch_session_id_alias() -> None:
    pending = PendingToolCall.from_payload(
        {
            "function": {
                "name": "file_patch",
                "arguments": json.dumps(
                    {
                        "path": "src/example.py",
                        "session_id": "ws-123",
                        "target_text": "old",
                        "replacement_text": "new",
                    }
                ),
            }
        }
    )

    assert pending is not None
    assert pending.args["write_session_id"] == "ws-123"
    assert "session_id" not in pending.args


def test_matching_write_session_for_pending_supports_file_patch() -> None:
    state = _make_state()
    state.write_session = _make_open_write_session(Path("src/example.py"))
    harness = SimpleNamespace(state=state)
    pending = PendingToolCall(
        tool_name="file_patch",
        args={
            "path": "src/example.py",
            "target_text": "old",
            "replacement_text": "new",
        },
    )

    matched = _matching_write_session_for_pending(harness, pending)

    assert matched is state.write_session


def test_repair_active_write_session_args_rebinds_mismatched_session_id_for_active_target() -> None:
    state = _make_state()
    target = Path("temp/task_queue.py")
    session = _make_open_write_session(target, session_id="ws-active")
    session.write_next_section = "imports"
    state.write_session = session
    harness = SimpleNamespace(state=state)
    pending = PendingToolCall(
        tool_name="file_append",
        args={
            "path": str(target),
            "content": "import heapq\n",
            "write_session_id": "ws-model-guess",
        },
    )

    repaired = _repair_active_write_session_args(harness, pending)

    assert repaired is True
    assert pending.args["write_session_id"] == "ws-active"
    assert pending.args["section_name"] == "imports"


def test_file_read_marks_cap_limited_reads_as_partial(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "large.txt"
    target.write_text("a" * 150_000, encoding="utf-8")

    result = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))

    assert result["success"] is True
    assert result["metadata"]["complete_file"] is False
    assert result["metadata"]["truncated"] is True
    assert result["metadata"]["bytes"] == 100_000
    assert len(result["output"]) == 100_000


def test_file_read_rehydrates_missing_staging_path_for_active_write_session(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "logwatch.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('target')\n", encoding="utf-8")

    session = _make_open_write_session(target, intent="patch_existing")
    stage_path = fs._session_stage_path(session.write_session_id, target, str(tmp_path))
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text("print('staged')\n", encoding="utf-8")
    session.write_staging_path = ""
    state.write_session = session

    result = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))

    assert result["success"] is True
    assert result["output"] == "print('staged')"
    assert result["metadata"]["read_from_staging"] is True
    assert result["metadata"]["source_path"] == str(stage_path)
    assert state.write_session.write_staging_path == str(stage_path)


def test_repeated_file_read_triggers_a_recovery_nudge(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.current_phase = "execute"
    state.scratchpad["_contract_phase"] = "execute"
    target = tmp_path / "temp" / "packet_log_analyzer.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('hello')\n", encoding="utf-8")
    fingerprint = json.dumps(
        {"tool_name": "file_read", "args": {"path": str(target)}},
        sort_keys=True,
        ensure_ascii=True,
    )
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
    ]

    messages: list[SimpleNamespace] = []

    harness = SimpleNamespace(
        state=state,
        log=SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None),
        _runlog=lambda *args, **kwargs: None,
        _emit=lambda *args, **kwargs: None,
    )
    state.append_message = lambda message: messages.append(message)

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
        pending_tool_calls=[PendingToolCall(tool_name="file_read", args={"path": str(target)})],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert graph_state.pending_tool_calls == []
    assert messages
    assert messages[-1].metadata["recovery_kind"] == "file_read"
    assert "already read" in messages[-1].content


def test_repeated_file_read_is_rerouted_to_artifact_read_when_full_artifact_exists(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "dependency_resolver.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('hello')\n" * 80, encoding="utf-8")
    fingerprint = json.dumps(
        {"tool_name": "file_read", "args": {"path": str(target)}},
        sort_keys=True,
        ensure_ascii=True,
    )
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
    ]
    state.artifacts = {
        "A0009": ArtifactRecord(
            artifact_id="A0009",
            kind="file_read",
            source=str(target),
            created_at="2026-04-03T00:00:00+00:00",
            size_bytes=target.stat().st_size,
            summary="dependency_resolver.py full file",
            tool_name="file_read",
            metadata={"path": str(target), "complete_file": True, "total_lines": 80},
        )
    }

    dispatched: list[tuple[str, dict[str, object]]] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="ok")

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"artifact_read"}

        @staticmethod
        def get(tool_name: str):
            if tool_name != "artifact_read":
                return None
            return SimpleNamespace(schema={"required": ["artifact_id"]})

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        log=logging.getLogger("test.plan.reroute"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
        pending_tool_calls=[PendingToolCall(tool_name="file_read", args={"path": str(target)})],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == [("artifact_read", {"artifact_id": "A0009", "start_line": 1})]
    assert graph_state.final_result is None
    assert graph_state.last_tool_results
    assert graph_state.last_tool_results[0].tool_name == "artifact_read"


def test_dispatch_tools_reroutes_declared_read_first_file_write_to_file_read(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "logwatch.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('staged')\n", encoding="utf-8")
    state.write_session = _make_open_write_session(target, session_id="ws-read", intent="patch_existing")

    dispatched: list[tuple[str, dict[str, object]]] = []
    runlog_events: list[tuple[str, str, dict[str, object]]] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output=target.read_text(encoding="utf-8"))

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"file_read", "file_write"}

        @staticmethod
        def get(tool_name: str):
            if tool_name == "file_read":
                return SimpleNamespace(schema={"required": ["path"]})
            if tool_name == "file_write":
                return SimpleNamespace(schema={"required": ["path", "content"]})
            return None

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        log=logging.getLogger("test.plan.read_first_write"),
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-read-first-write",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_write",
                args={
                    "path": str(target),
                    "content": "replacement",
                    "write_session_id": "ws-read",
                },
            )
        ],
        last_assistant_text="I see the staged content is incomplete. Let me read exactly what we have so far and then finish it properly.",
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == [("file_read", {"path": str(target)})]
    assert graph_state.last_tool_results
    assert graph_state.last_tool_results[0].tool_name == "file_read"
    assert state.recent_messages[-1].metadata["recovery_kind"] == "declared_read_before_write"
    assert runlog_events
    event, message, data = runlog_events[-1]
    assert event == "intent_mismatch_detected_redirection_activated"
    assert message == "intent mismatch detected redirection activated"
    assert data["original_tool_name"] == "file_write"
    assert data["redirected_tool_name"] == "file_read"
    assert data["mismatch_kind"] == "declared_read_before_write"
    assert data["reason"]["matched_phrase"] == "let me read"


def test_dispatch_tools_reroutes_declared_read_first_file_patch_to_file_read(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "logwatch.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('staged')\n", encoding="utf-8")
    state.write_session = _make_open_write_session(target, session_id="ws-patch", intent="patch_existing")

    dispatched: list[tuple[str, dict[str, object]]] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output=target.read_text(encoding="utf-8"))

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"file_read", "file_patch"}

        @staticmethod
        def get(tool_name: str):
            if tool_name == "file_read":
                return SimpleNamespace(schema={"required": ["path"]})
            if tool_name == "file_patch":
                return SimpleNamespace(schema={"required": ["path", "target_text", "replacement_text"]})
            return None

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        log=logging.getLogger("test.plan.read_first_patch"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-read-first-patch",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_patch",
                args={
                    "path": str(target),
                    "target_text": "print",
                    "replacement_text": "echo",
                    "write_session_id": "ws-patch",
                },
            )
        ],
        last_assistant_text="The staged copy looks suspect. I'll read the current staged content first, then decide on a narrow patch.",
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == [("file_read", {"path": str(target)})]
    assert graph_state.last_tool_results
    assert graph_state.last_tool_results[0].tool_name == "file_read"
    assert state.recent_messages[-1].metadata["recovery_kind"] == "declared_read_before_write"


def test_dispatch_tools_reroutes_shell_exec_ssh_to_ssh_exec() -> None:
    state = _make_state()

    dispatched: list[tuple[str, dict[str, object]]] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="ok")

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"shell_exec", "ssh_exec"}

        @staticmethod
        def get(tool_name: str):
            if tool_name == "shell_exec":
                return SimpleNamespace(schema={"required": ["command"]})
            if tool_name == "ssh_exec":
                return SimpleNamespace(schema={"required": ["host", "command"]})
            return None

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        dispatcher=SimpleNamespace(phase="explore"),
        log=logging.getLogger("test.plan.ssh_route"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-ssh",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="shell_exec",
                args={"command": "ssh root@192.168.1.63 'hostname'"},
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == [
        (
            "ssh_exec",
            {
                "host": "192.168.1.63",
                "user": "root",
                "command": "hostname",
            },
        )
    ]
    assert graph_state.final_result is None
    assert graph_state.last_tool_results
    assert graph_state.last_tool_results[0].tool_name == "ssh_exec"


def test_dispatch_tools_blocks_local_shell_exec_for_remote_task_and_nudges() -> None:
    state = _make_state()
    state.run_brief.original_task = (
        'ssh root@192.168.1.63 with username root password "@S02v1735" '
        "check whether docker is installed"
    )

    dispatched: list[tuple[str, dict[str, object]]] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="ok")

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"shell_exec", "ssh_exec"}

        @staticmethod
        def get(tool_name: str):
            if tool_name == "shell_exec":
                return SimpleNamespace(schema={"required": ["command"]})
            if tool_name == "ssh_exec":
                return SimpleNamespace(schema={"required": ["host", "command"]}, phase_allowed=lambda phase: True)
            return None

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        dispatcher=SimpleNamespace(phase="explore"),
        log=logging.getLogger("test.plan.remote_guard"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-remote-guard",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="shell_exec",
                args={"command": "docker ps"},
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == []
    assert graph_state.last_tool_results == []
    assert state.recent_messages
    assert state.recent_messages[-1].content == "This is a remote task. Use `ssh_exec`, not local `shell_exec`."
    assert state.recent_messages[-1].metadata["recovery_mode"] == "remote_task_guard"


def test_repeated_dir_list_loop_pauses_for_resume_instead_of_failing() -> None:
    state = _make_state()
    state.run_brief.original_task = "Inspect the repository structure."
    state.working_memory.current_goal = state.run_brief.original_task
    fingerprint = json.dumps(
        {"tool_name": "dir_list", "args": {"path": "."}},
        sort_keys=True,
        ensure_ascii=True,
    )
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "dir_list", "fingerprint": fingerprint},
        {"tool_name": "dir_list", "fingerprint": fingerprint},
    ]

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: {"dir_list"}, get=lambda _name: None),
        log=logging.getLogger("test.plan.dir_list_interrupt"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda error, **kwargs: {"error": error, **kwargs},
        _dispatch_tool_call=None,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
        pending_tool_calls=[PendingToolCall(tool_name="dir_list", args={"path": "."})],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert graph_state.final_result is None
    assert graph_state.interrupt_payload is not None
    assert state.pending_interrupt["kind"] == "repeated_tool_loop_resume"
    assert state.pending_interrupt["tool_name"] == "dir_list"
    assert "continue" in state.pending_interrupt["question"].lower()
    assert "50" in state.pending_interrupt["guidance"]
    assert "targeted next step" in state.pending_interrupt["guidance"]
    assert graph_state.pending_tool_calls == []


def test_repeated_artifact_read_for_table_task_gets_summary_exit_nudge() -> None:
    state = _make_state()
    state.run_brief.original_task = "List the files you can see in the current env and present a table summary."
    state.working_memory.current_goal = state.run_brief.original_task
    fingerprint = json.dumps(
        {"tool_name": "artifact_read", "args": {"artifact_id": "A0003"}},
        sort_keys=True,
        ensure_ascii=True,
    )
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "artifact_read", "fingerprint": fingerprint},
        {"tool_name": "artifact_read", "fingerprint": fingerprint},
    ]

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    harness = SimpleNamespace(
        state=state,
        registry=SimpleNamespace(names=lambda: {"artifact_read"}, get=lambda _name: None),
        log=logging.getLogger("test.plan.artifact_summary_exit"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda error, **kwargs: {"error": error, **kwargs},
        _dispatch_tool_call=None,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
        pending_tool_calls=[PendingToolCall(tool_name="artifact_read", args={"artifact_id": "A0003"})],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert graph_state.final_result is None
    assert graph_state.pending_tool_calls == []
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["recovery_kind"] == "artifact_summary_exit"
    assert "requested table or summary" in state.recent_messages[-1].content


def test_shell_and_ssh_are_blocked_before_authoring_artifact_exists(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.run_brief.original_task = "Create a Python script in `./temp/app.py`"
    state.working_memory.current_goal = state.run_brief.original_task
    state.scratchpad["_task_target_paths"] = ["./temp/app.py"]
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="Write a script",
        status="approved",
        approved=True,
        steps=[PlanStep(step_id="P1", title="Create file skeleton")],
    )
    state.draft_plan = state.active_plan

    shell_blocked = asyncio.run(shell.shell_exec(command="pwd", state=state, harness=None))
    assert shell_blocked["success"] is False
    assert shell_blocked["metadata"]["reason"] == "authoring_target_missing"

    ssh_blocked = asyncio.run(
        network.ssh_exec(
            host="example.com",
            command="pwd",
            state=state,
            harness=None,
        )
    )
    assert ssh_blocked["success"] is False
    assert ssh_blocked["metadata"]["reason"] == "authoring_target_missing"

    target = tmp_path / "app.py"
    write_result = asyncio.run(fs.file_write(path=str(target), content="print('ok')\n", cwd=str(tmp_path), state=state))
    assert write_result["success"] is True

    shell_allowed = asyncio.run(shell.shell_exec(command="pwd", state=state, harness=None))
    assert shell_allowed["success"] is True


def test_shell_task_is_not_blocked_when_contract_flow_is_inactive(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.run_brief.original_task = "Run pwd and report the output"
    state.working_memory.current_goal = state.run_brief.original_task

    shell_allowed = asyncio.run(shell.shell_exec(command="pwd", state=state, harness=None))

    assert shell_allowed["success"] is True


def test_shell_usage_error_prompts_for_missing_required_arguments(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.current_phase = "execute"

    script = tmp_path / "requires_input.py"
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--input', required=True)\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )

    result = asyncio.run(shell.shell_exec(command=f"python3 {script}", state=state, harness=None))

    assert result["success"] is False
    assert result["status"] == "needs_human"
    assert result["metadata"]["reason"] == "missing_required_arguments"
    assert "--input" in result["metadata"]["question"]
    assert "missing required arguments" in result["metadata"]["question"].lower()


def test_shell_exec_missing_root_temp_path_gets_workspace_relative_hint(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.current_phase = "execute"
    (tmp_path / "temp").mkdir()
    (tmp_path / "temp" / "test-calc.py").write_text("print(1)\n", encoding="utf-8")

    class _FakeProc:
        returncode = 2

        async def communicate(self) -> tuple[bytes, bytes]:
            return (
                b"",
                b"python3: can't open file '/temp/test-calc.py': [Errno 2] No such file or directory\n",
            )

    with patch.object(shell, "_create_process", AsyncMock(return_value=_FakeProc())):
        result = asyncio.run(shell.shell_exec(command="python3 /temp/test-calc.py", state=state, harness=None))

    assert result["success"] is False
    assert "./temp/test-calc.py" in result["error"]


def test_shell_workspace_relative_retry_hint_targets_root_temp_paths() -> None:
    state = _make_state()
    harness = SimpleNamespace(state=state)
    pending = PendingToolCall(tool_name="shell_exec", args={"command": "python3 /temp/test-calc.py"})

    hint = _shell_workspace_relative_retry_hint(harness, pending)

    assert hint is not None
    assert "./temp/test-calc.py" in hint


def test_shell_exec_emits_progress_heartbeats_before_timeout(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.current_phase = "execute"
    state.scratchpad["_contract_phase"] = "execute"
    state.run_brief.original_task = "Run a long shell command"
    state.working_memory.current_goal = state.run_brief.original_task

    events: list[object] = []

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    class _FakeStream:
        def __init__(self, chunks: list[bytes], *, delay: float = 0.0) -> None:
            self._chunks = list(chunks)
            self._delay = delay

        async def read(self, _size: int) -> bytes:
            if self._chunks:
                if self._delay:
                    await asyncio.sleep(self._delay)
                return self._chunks.pop(0)
            return b""

    class _FakeProc:
        def __init__(self) -> None:
            self.stdout = _FakeStream([b"starting\n", b""])
            self.stderr = _FakeStream([b""])
            self.returncode: int | None = None
            self._done = asyncio.Event()

        async def wait(self) -> int | None:
            await self._done.wait()
            return self.returncode

        def kill(self) -> None:
            self.returncode = 124
            self._done.set()

    proc = _FakeProc()
    fake_create_process = AsyncMock(return_value=proc)

    async def _capture_event(*args, **kwargs) -> None:
        del kwargs
        if len(args) >= 2:
            events.append(args[1])

    harness = SimpleNamespace(
        event_handler=object(),
        _emit=_capture_event,
        _active_processes=set(),
    )

    with patch.object(shell, "_create_process", fake_create_process):
        result = asyncio.run(shell.shell_exec(command="python -c 'import time; time.sleep(5)'", state=state, timeout_sec=2, harness=harness))

    assert result["success"] is False
    assert result["metadata"]["command"] == "python -c 'import time; time.sleep(5)'"
    assert result["metadata"]["progress_updates"]
    assert any(
        getattr(getattr(event, "event_type", None), "value", None) == "shell_stream"
        and "still running" in getattr(event, "content", "")
        for event in events
    )


def test_shell_exec_tracks_background_shell_jobs(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)

    class _FakeProc:
        def __init__(self) -> None:
            self.pid = 4242
            self.returncode: int | None = None

    proc = _FakeProc()
    harness = SimpleNamespace(
        _active_processes=set(),
        state=state,
    )

    with patch.object(shell, "_create_process", AsyncMock(return_value=proc)):
        started = asyncio.run(
            shell.shell_exec(
                command="python -c 'import time; time.sleep(5)'",
                background=True,
                state=state,
                harness=harness,
            )
        )

    assert started["success"] is True
    job_id = started["output"]["job_id"]
    assert job_id in state.background_processes
    harness._active_processes.add(proc)

    running = asyncio.run(shell.shell_exec(job_id=job_id, state=state, harness=harness))
    assert running["success"] is True
    assert running["output"]["status"] == "running"

    proc.returncode = 0
    completed = asyncio.run(shell.shell_exec(job_id=job_id, state=state, harness=harness))
    assert completed["success"] is True
    assert completed["output"]["status"] == "completed"
    assert completed["output"]["exit_code"] == 0


def test_repeated_shell_exec_job_poll_is_not_treated_as_a_loop() -> None:
    state = _make_state()
    harness = SimpleNamespace(state=state)
    state.scratchpad["_tool_attempt_history"] = [
        {
            "tool_name": "shell_exec",
            "fingerprint": json.dumps(
                {"tool_name": "shell_exec", "args": {"job_id": "4242"}},
                sort_keys=True,
            ),
        }
    ]

    pending = PendingToolCall(tool_name="shell_exec", args={"job_id": "4242"})

    assert _detect_repeated_tool_loop(harness, pending) is None


def test_repeated_shell_exec_command_still_trips_after_the_updated_streak_limit() -> None:
    state = _make_state()
    harness = SimpleNamespace(state=state)
    fingerprint = json.dumps(
        {"tool_name": "shell_exec", "args": {"command": "python -V"}},
        sort_keys=True,
    )
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "shell_exec", "fingerprint": fingerprint},
        {"tool_name": "shell_exec", "fingerprint": fingerprint},
        {"tool_name": "shell_exec", "fingerprint": fingerprint},
        {"tool_name": "shell_exec", "fingerprint": fingerprint},
        {"tool_name": "shell_exec", "fingerprint": fingerprint},
    ]

    pending = PendingToolCall(tool_name="shell_exec", args={"command": "python -V"})

    assert _detect_repeated_tool_loop(harness, pending) == (
        "Guard tripped: repeated tool call loop (shell_exec repeated with identical arguments)"
    )


def test_repeated_artifact_print_trips_after_one_repeat() -> None:
    state = _make_state()
    harness = SimpleNamespace(state=state)
    fingerprint = json.dumps(
        {"tool_name": "artifact_print", "args": {"artifact_id": "A0002"}},
        sort_keys=True,
    )
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "artifact_print", "fingerprint": fingerprint}
    ]

    pending = PendingToolCall(tool_name="artifact_print", args={"artifact_id": "A0002"})

    assert _detect_repeated_tool_loop(harness, pending) == (
        "Guard tripped: repeated tool call loop (artifact_print repeated with identical arguments)"
    )


def test_one_shot_repeat_guard_allows_scheduled_recovery_file_read_once() -> None:
    state = _make_state()
    harness = SimpleNamespace(state=state)
    fingerprint = json.dumps(
        {"tool_name": "file_read", "args": {"path": "src/app.py"}},
        sort_keys=True,
    )
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
    ]
    pending = PendingToolCall(tool_name="file_read", args={"path": "src/app.py"})

    allow_repeated_tool_call_once(harness, "file_read", {"path": "src/app.py"})

    assert _detect_repeated_tool_loop(harness, pending) is None
    assert "_repeat_guard_one_shot_fingerprints" not in state.scratchpad
    assert _detect_repeated_tool_loop(harness, pending) == (
        "Guard tripped: repeated tool call loop (file_read repeated with identical arguments)"
    )


def test_authoring_budget_trims_multi_call_turn_for_small_model() -> None:
    state = _make_state()
    state.current_phase = "execute"
    state.scratchpad["_model_name"] = "qwen2.5-coder-7b-instruct"
    state.scratchpad["_model_is_small"] = True
    state.scratchpad["_contract_phase"] = "author"
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    graph_state = SimpleNamespace(
        pending_tool_calls=[
            SimpleNamespace(tool_name="file_read", args={"path": "src/app.py"}),
            SimpleNamespace(tool_name="file_write", args={"path": "src/app.py", "content": "x"}),
        ],
        last_tool_results=[],
    )

    applied = _apply_small_model_authoring_budget(harness, graph_state)

    assert applied is True
    assert len(graph_state.pending_tool_calls) == 1
    assert graph_state.pending_tool_calls[0].tool_name == "file_read"
    assert state.scratchpad["_authoring_action_budget_nudges"] == 1
    assert state.recent_messages[-1].role == "system"
    assert "one concrete action at a time" in state.recent_messages[-1].content


def test_loop_status_surfaces_acceptance_progress() -> None:
    state = _make_state()
    state.current_phase = "repair"
    state.run_brief.acceptance_criteria = ["The script runs"]
    state.acceptance_ledger = {"The script runs": "done"}
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "pytest",
        "command": "pytest -q",
        "exit_code": 0,
        "key_stdout": "1 passed",
        "key_stderr": "",
        "verdict": "pass",
        "acceptance_delta": {"status": "satisfied", "notes": ["execution succeeded"]},
    }
    state.repair_cycle_id = "repair-1"
    state.stagnation_counters = {"repeat_patch": 1}
    state.files_changed_this_cycle = ["src/app.py"]

    status = asyncio.run(control.loop_status(state))

    assert status["success"] is True
    payload = status["output"]
    assert payload["contract_phase"] == "repair"
    assert payload["acceptance_ready"] is True
    assert payload["pending_acceptance_criteria"] == []
    assert payload["last_verifier_verdict"]["verdict"] == "pass"
    assert payload["system_repair_cycle_id"] == "repair-1"
    assert "repair_cycle_id" not in payload


def test_contract_flow_status_text_includes_verdict_and_acceptance() -> None:
    state = _make_state()
    state.current_phase = "execute"
    state.run_brief.acceptance_criteria = ["The script runs", "The test passes"]
    state.acceptance_ledger = {"The script runs": "passed", "The test passes": "pending"}
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "pytest",
        "command": "pytest -q",
        "exit_code": 0,
        "verdict": "pass",
    }
    harness = SimpleNamespace(
        state=state,
        context_policy=SimpleNamespace(max_prompt_tokens=4096),
        server_context_limit=2048,
        guards=SimpleNamespace(max_tokens=1024),
    )

    status = StatusState.from_harness(
        harness,
        {"model": "qwen3.5:4b", "phase": "execute", "contract_flow_ui": True},
    )

    assert status.contract_flow_ui is True
    assert status.contract_phase == "verify"
    assert status.acceptance_progress == "1/2"
    assert status.latest_verdict == "pass | pytest | exit 0"

    bar = object.__new__(StatusBar)
    bar.__dict__.update(
        {
            "_model": status.model,
            "_phase": status.phase,
            "_step": status.step,
            "_mode": status.mode,
            "_plan": status.plan,
            "_active_step": status.active_step,
            "_activity": status.activity,
            "_contract_flow_ui": status.contract_flow_ui,
            "_contract_phase": status.contract_phase,
            "_acceptance_progress": status.acceptance_progress,
            "_latest_verdict": status.latest_verdict,
            "_token_usage": status.token_usage,
            "_token_total": status.token_total,
            "_token_limit": status.token_limit,
            "_api_errors": status.api_errors,
        }
    )

    text = StatusBar._build_status_text(bar)
    assert "contract: verify" in text
    assert "acceptance: 1/2" in text
    assert "verdict: pass | pytest | exit 0" in text


def test_system_prompt_surfaces_repair_focus() -> None:
    state = _make_state()
    state.current_phase = "execute"
    state.repair_cycle_id = "repair-9"
    state.last_failure_class = "syntax"
    state.files_changed_this_cycle = ["/tmp/example.py"]
    state.stagnation_counters = {"no_progress": 2, "repeat_patch": 1}

    prompt = build_system_prompt(state, "execute")

    assert "REPAIR FOCUS" in prompt
    assert "failure class: syntax" in prompt
    assert "system repair cycle: repair-9" in prompt
    assert "Never copy a system repair cycle ID into `write_session_id`." in prompt
    assert "files changed this cycle: /tmp/example.py" in prompt
    assert "stagnation counters: no_progress=2, repeat_patch=1" in prompt


def test_system_prompt_surfaces_file_patch_guidance() -> None:
    state = _make_state()
    prompt = build_system_prompt(state, "execute")

    assert "Use `file_patch` for small exact edits inside an existing file or active staged session." in prompt
    assert "Use `file_write` or `file_append` for new files, large sections, or chunked authoring." in prompt
    assert "When resuming an active session, prefer `file_write` for chunk continuation" in prompt
    assert "If you need a narrow exact repair inside the staged copy, prefer `file_patch` instead." in prompt
    assert "If prior chunks are no longer visible because tool previews were compacted or truncated" in prompt
    assert "Do not assume earlier chunks were lost" in prompt


def test_system_prompt_surfaces_small_model_tool_routing_card() -> None:
    state = _make_state()
    state.scratchpad["_model_name"] = "qwen3.5:4b"

    prompt = build_system_prompt(state, "execute", available_tool_names=["shell_exec", "ssh_exec"])

    assert "never invent aliases like `use_shell_exec`" in prompt
    assert "Remote host/IP/user/password mentioned means `ssh_exec`." in prompt
    assert "`shell_exec` is local-only." in prompt


def test_system_prompt_write_recovery_surfaces_stage_read_before_overwrite() -> None:
    state = _make_state()
    target = Path("./temp/logwatch.py")
    state.write_session = _make_open_write_session(target)

    prompt = build_system_prompt(state, "execute")

    assert "artifact_read(artifact_id='ws-1__stage')" in prompt
    assert "file_read(path='" in prompt
    assert "temp/logwatch.py" in prompt
    assert "Do not assume the chunks were lost or rewrite the whole file from memory" in prompt


def test_system_prompt_surfaces_general_verifier_context() -> None:
    state = _make_state()
    state.current_phase = "execute"
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "ssh root@host \"docker ps\"",
        "command": "ssh root@host \"docker ps\"",
        "exit_code": 127,
        "key_stdout": "",
        "key_stderr": "bash: line 1: docker: command not found",
        "verdict": "fail",
        "acceptance_delta": {"status": "blocked", "notes": ["bash: line 1: docker: command not found"]},
    }

    prompt = build_system_prompt(state, "execute")

    assert "LATEST VERIFIER" in prompt
    assert "docker ps" in prompt
    assert "docker: command not found" in prompt
    assert "Do not repeat `task_complete`" in prompt


def test_task_complete_error_surfaces_latest_verifier_summary() -> None:
    state = _make_state()
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "ssh root@host \"docker ps\"",
        "command": "ssh root@host \"docker ps\"",
        "exit_code": 127,
        "key_stdout": "",
        "key_stderr": "bash: line 1: docker: command not found",
        "verdict": "fail",
        "acceptance_delta": {"status": "blocked", "notes": ["bash: line 1: docker: command not found"]},
    }

    blocked = asyncio.run(control.task_complete("done", state=state))

    assert blocked["success"] is False
    assert "Latest verifier:" in blocked["error"]
    assert "docker ps" in blocked["error"]
    assert "docker: command not found" in blocked["error"]


def test_repair_recovery_nudge_triggers_on_repeated_shell_failures() -> None:
    state = _make_state()
    state.current_phase = "execute"
    state.repair_cycle_id = "repair-1"
    state.last_failure_class = "syntax"
    state.stagnation_counters = {"no_progress": 2, "repeat_command": 1}
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    record = ToolExecutionRecord(
        operation_id="op-1",
        tool_name="shell_exec",
        args={"command": "python broken.py"},
        tool_call_id="tool-1",
        result=ToolEnvelope(
            success=False,
            error="SyntaxError: invalid syntax",
            metadata={"command": "python broken.py"},
        ),
    )
    deps = SimpleNamespace(harness=harness, event_handler=None)

    nudged = _maybe_emit_repair_recovery_nudge(harness, record, deps)

    assert nudged is True
    assert harness.state.recent_messages[-1].role == "user"
    assert "Repair loop stalled" in harness.state.recent_messages[-1].content
    assert "system repair cycle repair-1" in harness.state.recent_messages[-1].content
    assert "Do not repeat the same command blindly" in harness.state.recent_messages[-1].content


def test_repair_recovery_nudge_triggers_on_repeated_file_patch_failures() -> None:
    state = _make_state()
    state.current_phase = "execute"
    state.repair_cycle_id = "repair-2"
    state.last_failure_class = "patch"
    state.stagnation_counters = {"no_progress": 2, "repeat_patch": 3}
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    record = ToolExecutionRecord(
        operation_id="op-2",
        tool_name="file_patch",
        args={
            "path": "src/example.py",
            "target_text": "return False",
            "replacement_text": "return True",
        },
        tool_call_id="tool-2",
        result=ToolEnvelope(
            success=False,
            error="patch_occurrence_mismatch: target text occurred 3 times",
            metadata={
                "path": "src/example.py",
                "error_kind": "patch_occurrence_mismatch",
                "actual_occurrences": 3,
                "expected_occurrences": 1,
                "ambiguity_hint": "The target text matched 3 times. Read a smaller slice and make `target_text` more specific before retrying.",
            },
        ),
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    nudged = _maybe_emit_repair_recovery_nudge(harness, record, deps)

    assert nudged is True
    assert harness.state.recent_messages[-1].role == "user"
    assert "Repair loop stalled" in harness.state.recent_messages[-1].content
    assert "src/example.py" in harness.state.recent_messages[-1].content
    assert "matched 3 times" in harness.state.recent_messages[-1].content
    assert "Do not broad-rewrite the file" in harness.state.recent_messages[-1].content


def test_schema_repair_message_for_file_patch_mentions_required_fields() -> None:
    state = _make_state()
    pending = PendingToolCall(
        tool_name="file_patch",
        args={"path": "src/example.py"},
        tool_call_id="tool-3",
    )

    message = _build_schema_repair_message(
        SimpleNamespace(state=state),
        pending,
        ["target_text", "replacement_text"],
    )

    assert "file_patch" in message
    assert "target_text, replacement_text" in message
    assert "exact target text and replacement text" in message


def test_schema_repair_message_for_file_patch_mentions_active_staged_copy() -> None:
    state = _make_state()
    target = Path("src/example.py")
    state.write_session = _make_open_write_session(target)
    pending = PendingToolCall(
        tool_name="file_patch",
        args={"path": "src/example.py"},
        tool_call_id="tool-4",
    )

    message = _build_schema_repair_message(
        SimpleNamespace(state=state),
        pending,
        ["target_text", "replacement_text"],
    )

    assert "active staged copy" in message
    assert "canonical destination" in message


def test_failed_task_complete_due_to_verifier_injects_recovery_nudge() -> None:
    state = _make_state()
    state.current_phase = "execute"
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=lambda *args, **kwargs: None,
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="task_complete",
            args={"message": "done"},
            tool_call_id="tool-1",
            result=ToolEnvelope(
                success=False,
                error="Cannot complete the task while the latest verifier verdict is still failing. Latest verifier: check=ssh root@host \"docker ps\" | details=bash: line 1: docker: command not found.",
                metadata={
                    "last_verifier_verdict": {
                        "tool": "shell_exec",
                        "target": "ssh root@host \"docker ps\"",
                        "command": "ssh root@host \"docker ps\"",
                        "exit_code": 127,
                        "key_stdout": "",
                        "key_stderr": "bash: line 1: docker: command not found",
                        "verdict": "fail",
                        "acceptance_delta": {
                            "status": "blocked",
                            "notes": ["bash: line 1: docker: command not found"],
                        },
                    }
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["recovery_kind"] == "task_complete_verifier_retry"
    assert "Do not repeat `task_complete` yet." in state.recent_messages[-1].content
    assert "loop_status" in state.recent_messages[-1].content


def test_patch_existing_first_choice_failure_injects_stage_read_nudge(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "logwatch.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-1-stage.py"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("print('staged')\n", encoding="utf-8")

    session = _make_open_write_session(target, intent="patch_existing")
    session.write_staging_path = str(stage)
    state.write_session = session

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_write",
            args={
                "path": "./logwatch.py",
                "content": "replacement body",
                "write_session_id": "ws-1",
                "section_name": "imports",
            },
            tool_call_id="tool-1",
            result=ToolEnvelope(
                success=False,
                error="Patch-existing write sessions need an explicit first-chunk choice.",
                metadata={
                    "path": str(target),
                    "staging_path": str(stage),
                    "write_session_id": "ws-1",
                    "write_session_intent": "patch_existing",
                    "replace_strategy": "auto",
                    "staged_only": True,
                    "error_kind": "patch_existing_requires_explicit_replace_strategy",
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert state.recent_messages
    message = state.recent_messages[-1]
    assert message.metadata["recovery_kind"] == "patch_existing_first_choice"
    assert "file_read(path='" in message.content
    assert "artifact_read(artifact_id='ws-1__stage')" in message.content
    assert "Do not assume earlier chunks were lost" in message.content


def test_patch_existing_first_choice_failure_autocontinues_with_file_read(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "logwatch.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-1-stage.py"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("print('staged')\n", encoding="utf-8")

    session = _make_open_write_session(target, intent="patch_existing")
    session.write_staging_path = str(stage)
    state.write_session = session

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_write",
            args={
                "path": "./logwatch.py",
                "content": "replacement body",
                "write_session_id": "ws-1",
                "section_name": "imports",
            },
            tool_call_id="tool-1",
            result=ToolEnvelope(
                success=False,
                error="Patch-existing write sessions need an explicit first-chunk choice.",
                metadata={
                    "path": str(target),
                    "staging_path": str(stage),
                    "write_session_id": "ws-1",
                    "write_session_intent": "patch_existing",
                    "replace_strategy": "auto",
                    "staged_only": True,
                    "error_kind": "patch_existing_requires_explicit_replace_strategy",
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_read"
    assert pending.args == {"path": str(target)}
    assert state.scratchpad["_repeat_guard_one_shot_fingerprints"]
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind") == "patch_existing_stage_read_autocontinue"
        for message in state.recent_messages
    )
    assert any(event == "patch_existing_stage_read_autocontinue" for event, _message, _data in runlog_events)


def test_patch_existing_first_choice_failure_stops_autocontinue_after_repeat(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "logwatch.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-1-stage.py"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("print('staged')\n", encoding="utf-8")
    state.scratchpad["_patch_existing_stage_read_autocontinue_counts"] = {f"ws-1|{target}": 1}

    session = _make_open_write_session(target, intent="patch_existing")
    session.write_staging_path = str(stage)
    state.write_session = session

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_write",
            args={
                "path": "./logwatch.py",
                "content": "replacement body",
                "write_session_id": "ws-1",
                "section_name": "imports",
            },
            tool_call_id="tool-1",
            result=ToolEnvelope(
                success=False,
                error="Patch-existing write sessions need an explicit first-chunk choice.",
                metadata={
                    "path": str(target),
                    "staging_path": str(stage),
                    "write_session_id": "ws-1",
                    "write_session_intent": "patch_existing",
                    "replace_strategy": "auto",
                    "staged_only": True,
                    "error_kind": "patch_existing_requires_explicit_replace_strategy",
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert graph_state.pending_tool_calls == []
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind") == "patch_existing_stage_read_circuit_breaker"
        for message in state.recent_messages
    )
    assert any(event == "patch_existing_stage_read_circuit_breaker" for event, _message, _data in runlog_events)


def test_dispatch_tools_blocks_ambiguous_write_after_patch_existing_stage_read(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "task_queue.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('staged')\n", encoding="utf-8")
    session = _make_open_write_session(target, session_id="ws-stage", intent="patch_existing")
    session.write_staging_path = str(tmp_path / ".smallctl" / "write_sessions" / "ws-stage.py")
    state.write_session = session
    state.scratchpad["_patch_existing_stage_read_contract"] = {
        "session_id": "ws-stage",
        "target_path": str(target),
        "staging_path": session.write_staging_path,
    }

    dispatched: list[tuple[str, dict[str, object]]] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="ok")

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"file_write"}

        @staticmethod
        def get(tool_name: str):
            if tool_name == "file_write":
                return SimpleNamespace(schema={"required": ["path", "content"]})
            return None

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        log=logging.getLogger("test.plan.patch_existing_contract_block"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-stage-contract-block",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_write",
                args={
                    "path": str(target),
                    "content": "replacement body",
                    "write_session_id": "ws-stage",
                    "section_name": "imports",
                },
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == []
    assert graph_state.pending_tool_calls == []
    assert state.recent_messages
    message = state.recent_messages[-1]
    assert message.metadata["recovery_kind"] == "schema_validation"
    assert "must choose one explicit repair shape" in message.content
    assert "replace_strategy='overwrite'" in message.content
    assert state.scratchpad["_patch_existing_stage_read_contract"]["session_id"] == "ws-stage"


def test_dispatch_tools_allows_explicit_overwrite_after_patch_existing_stage_read(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "task_queue.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('staged')\n", encoding="utf-8")
    session = _make_open_write_session(target, session_id="ws-stage", intent="patch_existing")
    session.write_staging_path = str(tmp_path / ".smallctl" / "write_sessions" / "ws-stage.py")
    state.write_session = session
    state.scratchpad["_patch_existing_stage_read_contract"] = {
        "session_id": "ws-stage",
        "target_path": str(target),
        "staging_path": session.write_staging_path,
    }

    dispatched: list[tuple[str, dict[str, object]]] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="ok")

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"file_write"}

        @staticmethod
        def get(tool_name: str):
            if tool_name == "file_write":
                return SimpleNamespace(schema={"required": ["path", "content"]})
            return None

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        log=logging.getLogger("test.plan.patch_existing_contract_overwrite"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-stage-contract-overwrite",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_write",
                args={
                    "path": str(target),
                    "content": "replacement body",
                    "write_session_id": "ws-stage",
                    "section_name": "imports",
                    "replace_strategy": "overwrite",
                },
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == [
        (
            "file_write",
            {
                "path": str(target),
                "content": "replacement body",
                "write_session_id": "ws-stage",
                "section_name": "imports",
                "replace_strategy": "overwrite",
            },
        )
    ]
    assert "_patch_existing_stage_read_contract" not in state.scratchpad


def test_dispatch_tools_allows_file_patch_after_patch_existing_stage_read(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "task_queue.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('staged')\n", encoding="utf-8")
    session = _make_open_write_session(target, session_id="ws-stage", intent="patch_existing")
    session.write_staging_path = str(tmp_path / ".smallctl" / "write_sessions" / "ws-stage.py")
    state.write_session = session
    state.scratchpad["_patch_existing_stage_read_contract"] = {
        "session_id": "ws-stage",
        "target_path": str(target),
        "staging_path": session.write_staging_path,
    }

    dispatched: list[tuple[str, dict[str, object]]] = []

    async def _dispatch_tool_call(tool_name: str, args: dict[str, object]) -> ToolEnvelope:
        dispatched.append((tool_name, args))
        return ToolEnvelope(success=True, output="ok")

    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    class _Registry:
        @staticmethod
        def names() -> set[str]:
            return {"file_patch"}

        @staticmethod
        def get(tool_name: str):
            if tool_name == "file_patch":
                return SimpleNamespace(schema={"required": ["path", "target_text", "replacement_text"]})
            return None

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        log=logging.getLogger("test.plan.patch_existing_contract_patch"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-stage-contract-patch",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_patch",
                args={
                    "path": str(target),
                    "target_text": "print('staged')\n",
                    "replacement_text": "print('fixed')\n",
                    "write_session_id": "ws-stage",
                },
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == [
        (
            "file_patch",
            {
                "path": str(target),
                "target_text": "print('staged')\n",
                "replacement_text": "print('fixed')\n",
                "write_session_id": "ws-stage",
            },
        )
    ]
    assert "_patch_existing_stage_read_contract" not in state.scratchpad


def test_patch_existing_first_choice_failure_recovers_missing_write_session(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "logwatch.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-1-stage.py"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("print('staged')\n", encoding="utf-8")

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_write",
            args={
                "path": "./logwatch.py",
                "content": "replacement body",
                "write_session_id": "ws-1",
                "section_name": "imports",
            },
            tool_call_id="tool-1",
            result=ToolEnvelope(
                success=False,
                error="Patch-existing write sessions need an explicit first-chunk choice.",
                metadata={
                    "path": str(target),
                    "staging_path": str(stage),
                    "write_session_id": "ws-1",
                    "write_session_intent": "patch_existing",
                    "replace_strategy": "auto",
                    "staged_only": True,
                    "error_kind": "patch_existing_requires_explicit_replace_strategy",
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_read"
    assert pending.args == {"path": str(target)}
    assert isinstance(state.write_session, WriteSession)
    assert state.write_session.write_session_id == "ws-1"
    assert state.write_session.write_target_path == str(target)
    assert state.write_session.write_staging_path == str(stage)
    assert any(event == "patch_existing_stage_read_autocontinue" for event, _message, _data in runlog_events)


def test_patch_existing_first_choice_failure_rebinds_mismatched_write_session(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "logwatch.py"
    wrong_stage = tmp_path / ".smallctl" / "write_sessions" / "ws-other-stage.py"
    wrong_stage.parent.mkdir(parents=True, exist_ok=True)
    wrong_stage.write_text("print('wrong staged')\n", encoding="utf-8")
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-1-stage.py"
    stage.write_text("print('staged')\n", encoding="utf-8")
    state.write_session = WriteSession(
        write_session_id="ws-other",
        write_target_path=str(target),
        write_session_intent="patch_existing",
        write_staging_path=str(wrong_stage),
        status="open",
    )

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-1",
            tool_name="file_write",
            args={
                "path": "./logwatch.py",
                "content": "replacement body",
                "write_session_id": "ws-1",
                "section_name": "imports",
            },
            tool_call_id="tool-1",
            result=ToolEnvelope(
                success=False,
                error="Patch-existing write sessions need an explicit first-chunk choice.",
                metadata={
                    "path": str(target),
                    "staging_path": str(stage),
                    "write_session_id": "ws-1",
                    "write_session_intent": "patch_existing",
                    "replace_strategy": "auto",
                    "staged_only": True,
                    "error_kind": "patch_existing_requires_explicit_replace_strategy",
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert len(graph_state.pending_tool_calls) == 1
    assert graph_state.pending_tool_calls[0].tool_name == "file_read"
    assert state.write_session is not None
    assert state.write_session.write_session_id == "ws-1"
    assert state.write_session.write_staging_path == str(stage)


def test_staging_path_mutation_failure_injects_target_path_redirect_nudge(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "script.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-1-stage.py"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("print('staged')\n", encoding="utf-8")

    session = _make_open_write_session(target)
    session.write_staging_path = str(stage)
    state.write_session = session

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-redirect",
            tool_name="file_patch",
            args={
                "path": str(stage),
                "target_text": "staged",
                "replacement_text": "target",
            },
            tool_call_id="tool-redirect",
            result=ToolEnvelope(
                success=False,
                error="staging path used as target",
                metadata={
                    "error_kind": "write_session_staging_path_used_as_target",
                    "write_session_id": "ws-1",
                    "target_path": str(target),
                    "staging_path": str(stage),
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind") == "write_session_target_path_redirect"
        for message in state.recent_messages
    )
    assert any(event == "write_session_target_path_redirect_nudge" for event, _message, _data in runlog_events)


def test_file_patch_mismatch_autocontinues_with_file_read(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("print('hello')\n", encoding="utf-8")

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-2",
            tool_name="file_patch",
            args={
                "path": str(target),
                "target_text": "return False",
                "replacement_text": "return True",
            },
            tool_call_id="tool-2",
            result=ToolEnvelope(
                success=False,
                error="Patch target text was not found.",
                metadata={
                    "path": str(target),
                    "requested_path": str(target),
                    "error_kind": "patch_target_not_found",
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_read"
    assert pending.args == {"path": str(target)}
    assert state.scratchpad["_repeat_guard_one_shot_fingerprints"]
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind") == "file_patch_read_autocontinue"
        for message in state.recent_messages
    )
    assert any(event == "file_patch_read_autocontinue" for event, _message, _data in runlog_events)


def test_task_complete_verifier_failure_autocontinues_with_loop_status() -> None:
    state = _make_state()
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "pytest",
        "command": "pytest -q",
        "exit_code": 1,
        "key_stdout": "",
        "key_stderr": "1 failed",
        "verdict": "fail",
        "acceptance_delta": {"status": "blocked", "notes": ["1 failed"]},
    }

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-3",
            tool_name="task_complete",
            args={"message": "done"},
            tool_call_id="tool-3",
            result=ToolEnvelope(
                success=False,
                error="Latest verifier verdict is still failing.",
                metadata={"last_verifier_verdict": state.last_verifier_verdict},
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "loop_status"
    assert pending.args == {}
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind") == "task_complete_verifier_loop_status_autocontinue"
        for message in state.recent_messages
    )
    assert any(event == "task_complete_verifier_loop_status_autocontinue" for event, _message, _data in runlog_events)


def test_task_complete_repair_phase_block_autocontinues_with_loop_status() -> None:
    state = _make_state()

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-repair",
            tool_name="task_complete",
            args={"message": "done"},
            tool_call_id="tool-repair",
            result=ToolEnvelope(
                success=False,
                error="Tool 'task_complete' is not allowed in phase 'repair'",
                metadata={"phase": "repair"},
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "loop_status"
    assert pending.args == {}
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind") == "task_complete_repair_loop_status_autocontinue"
        for message in state.recent_messages
    )
    assert any(event == "task_complete_repair_loop_status_autocontinue" for event, _message, _data in runlog_events)


def test_interpret_chat_output_blocks_plain_completion_while_verifier_fails() -> None:
    state = _make_state()
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "pytest",
        "command": "pytest -q",
        "exit_code": 1,
        "key_stdout": "",
        "key_stderr": "1 failed",
        "verdict": "fail",
    }
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="chat")
    graph_state.last_assistant_text = "All 16 tests pass and the task is complete."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_chat_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert graph_state.final_result is None
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["recovery_kind"] == "verifier_guard"
    assert "latest verifier is still failing" in state.recent_messages[-1].content.lower()


def test_interpret_chat_output_blocks_plain_completion_with_open_write_session() -> None:
    state = _make_state()
    state.write_session = WriteSession(
        write_session_id="ws-1",
        write_target_path="./temp/task_queue.py",
        write_session_intent="patch_existing",
        write_pending_finalize=True,
        status="open",
    )
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="chat")
    graph_state.last_assistant_text = "Implementation is finished."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_chat_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert graph_state.final_result is None
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["recovery_kind"] == "write_session_guard"
    assert "write session `ws-1`" in state.recent_messages[-1].content.lower()


def test_route_after_apply_dispatches_pending_recovery_tool_calls() -> None:
    payload = {
        "final_result": None,
        "interrupt_payload": None,
        "pending_tool_calls": [{"tool_name": "file_read", "args": {"path": "./temp/logwatch.py"}}],
    }

    assert LoopGraphRuntime._route_after_apply(payload) == "dispatch_tools"
    assert ChatGraphRuntime._route_after_chat_apply(payload) == "dispatch_tools"


def test_runtime_graph_specs_include_dispatch_edge_after_apply() -> None:
    loop_targets = LoopGraphRuntime.GRAPH_SPEC.edge_map["apply_tool_outcomes"][1]
    chat_targets = ChatGraphRuntime.GRAPH_SPEC.edge_map["apply_chat_tool_outcomes"][1]

    assert loop_targets["dispatch_tools"] == "dispatch_tools"
    assert chat_targets["dispatch_tools"] == "dispatch_tools"


def test_contract_flow_ui_flag_parses_from_cli_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMALLCTL_CONTRACT_FLOW_UI", "true")
    env_config = resolve_config({})
    assert env_config.contract_flow_ui is True

    cli_config = resolve_config({"contract_flow_ui": False})
    assert cli_config.contract_flow_ui is False


def test_first_token_timeout_parses_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMALLCTL_FIRST_TOKEN_TIMEOUT_SEC", "17")

    config = resolve_config({})

    assert config.first_token_timeout_sec == 17


def test_backend_supervisor_fields_parse_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SMALLCTL_HEALTHCHECK_URL", "http://localhost:1234/v1/models")
    monkeypatch.setenv("SMALLCTL_RESTART_COMMAND", "systemctl restart lmstudio")
    monkeypatch.setenv("SMALLCTL_BACKEND_UNLOAD_COMMAND", "echo unloading")
    monkeypatch.setenv("SMALLCTL_STARTUP_GRACE_PERIOD_SEC", "33")
    monkeypatch.setenv("SMALLCTL_MAX_RESTARTS_PER_HOUR", "4")

    config = resolve_config({})

    assert config.healthcheck_url == "http://localhost:1234/v1/models"
    assert config.restart_command == "systemctl restart lmstudio"
    assert config.backend_unload_command == "echo unloading"
    assert config.startup_grace_period_sec == 33
    assert config.max_restarts_per_hour == 4

    cli_config = resolve_config({"backend_unload_command": "echo cli unloading"})
    assert cli_config.backend_unload_command == "echo cli unloading"
