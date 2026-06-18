from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.autocontinue import _DURABLE_AUTOCONTINUE_KEY, drain_durable_autocontinue
from smallctl.graph.lifecycle_nodes import prepare_loop_step, resume_loop_run
from smallctl.graph.nodes import (
    LoopRoute,
    _apply_small_model_authoring_budget,
    _matching_write_session_for_pending,
    dispatch_tools,
    interpret_chat_output,
    interpret_model_output,
    interpret_planning_output,
)
from smallctl.graph.tool_call_parser import (
    _detect_repeated_tool_loop,
    _repair_empty_target_file_patch_to_file_write,
    _repair_active_write_session_args,
    allow_repeated_tool_call_once,
    _detect_oversize_write_payload,
)
from smallctl.graph.tool_call_parser import _build_schema_repair_message
from smallctl.graph.tool_execution_node_guards import _long_running_remote_timeout_write_guard
from smallctl.graph.tool_loop_guards import _repeat_loop_limits
from smallctl.graph.runtime import LoopGraphRuntime, ChatGraphRuntime
from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.harness.tool_results import _store_verifier_verdict
from smallctl.prompts import build_system_prompt
from smallctl.state import ArtifactRecord, ExecutionPlan, LoopState, PlanStep, WriteSession
from smallctl.graph.tool_outcomes import _maybe_emit_repair_recovery_nudge, apply_tool_outcomes
from smallctl.graph.tool_outcomes import _shell_workspace_relative_retry_hint
from smallctl.graph.state import ToolExecutionRecord
from smallctl.guards import GuardConfig, check_guards
from smallctl.harness import Harness, HarnessConfig
from smallctl.harness.tool_visibility import hidden_tool_reason
from smallctl.models.tool_result import ToolEnvelope
from smallctl.config import resolve_config
from smallctl.recovery_schema import Subtask, SubtaskLedger
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


def test_recovery_nudge_metadata_literals_include_recovery_kind() -> None:
    missing: list[str] = []
    for path in Path("src/smallctl").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"metadata\s*=\s*\{(.*?)\}", text, re.DOTALL):
            body = match.group(1)
            if '"is_recovery_nudge": True' in body and '"recovery_kind"' not in body:
                line = text.count("\n", 0, match.start()) + 1
                missing.append(f"{path}:{line}")

    assert missing == []


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
                {
                    "step_id": "P1",
                    "title": "Create file skeleton",
                    "claim_refs": ["C1"],
                    "difficulty": "hard",
                    "tool_allowlist": ["file_write"],
                },
                {"step_id": "P2", "title": "Implement functions"},
                {"step_id": "P3", "title": "Debug and verify"},
            ],
            state=state,
            harness=harness,
        )
    )

    assert result["success"] is True
    assert state.plan_artifact_id
    assert state.draft_plan.steps[0].difficulty == "hard"
    assert state.draft_plan.steps[0].tool_allowlist == ["file_write"]
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


def test_plan_set_accepts_task_only_step_payloads(tmp_path: Path) -> None:
    state = _make_state()
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path, "run-1"),
        log=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    result = asyncio.run(
        planning.plan_set(
            goal="Create a script and verify it",
            inputs=["User request"],
            outputs=["temp/example.py"],
            constraints=["Use stdlib only"],
            acceptance_criteria=["Script runs successfully"],
            implementation_plan=["Write", "Run"],
            steps=[
                {"task": "Write script skeleton", "tool_allowlist": ["file_write"]},
                {"task": "Run verifier", "tool_allowlist": ["shell_exec"]},
            ],
            state=state,
            harness=harness,
        )
    )

    assert result["success"] is True
    assert state.draft_plan is not None
    assert [step.title for step in state.draft_plan.steps] == [
        "Write script skeleton",
        "Run verifier",
    ]
    assert [step.step_id for step in state.draft_plan.steps] == ["P1", "P2"]


def test_plan_set_rejects_incomplete_plan_before_approval(tmp_path: Path) -> None:
    state = _make_state()
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path, "run-1"),
        log=SimpleNamespace(warning=lambda *args, **kwargs: None),
    )

    result = asyncio.run(
        planning.plan_set(
            goal="Create a small CLI script",
            summary="I will explore, then propose a plan.",
            state=state,
            harness=harness,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "incomplete_plan"
    assert set(result["metadata"]["missing_fields"]) == {
        "outputs",
        "acceptance_criteria",
        "implementation_plan",
        "steps",
    }
    assert state.draft_plan is None
    assert state.active_plan is None


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


def test_system_prompt_web_research_prefers_result_id_and_real_answer() -> None:
    state = _make_state()
    state.active_tool_profiles = ["core", "network_read"]
    state.run_brief.original_task = "what is the weather in jacksonville today?"

    prompt = build_system_prompt(
        state,
        "execute",
        available_tool_names=["web_search", "web_fetch", "task_complete"],
    )

    assert "prefer the exact `web_fetch(result_id='...')` form shown in the result list" in prompt
    assert "do not finish with only 'found N results'" in prompt


def test_system_prompt_guides_phase_contract_update_for_phased_coding() -> None:
    state = _make_state()
    state.scratchpad["_model_name"] = "qwen3:4b"
    state.run_brief.original_task = "read temp/tetris-spec.md and implement phase 3"

    prompt = build_system_prompt(
        state,
        "execute",
        available_tool_names=["file_read", "phase_contract_update", "task_complete"],
    )

    assert "PHASED CODING CONTRACT" in prompt
    assert "call `phase_contract_update`" in prompt
    assert "Promotion requires a behavioral verifier" in prompt
    assert "py_compile" in prompt


def test_system_prompt_omits_phase_contract_guidance_for_simple_task() -> None:
    state = _make_state()
    state.run_brief.original_task = "write a small hello world script"

    prompt = build_system_prompt(
        state,
        "execute",
        available_tool_names=["file_write", "task_complete"],
    )

    assert "PHASED CODING CONTRACT" not in prompt


def test_loop_status_surfaces_subtask_ledger() -> None:
    state = _make_state()
    state.subtask_ledger = SubtaskLedger(
        task_id="task-1",
        active_subtask_id="S2",
        subtasks=[
            Subtask(
                subtask_id="S1",
                title="Read file",
                goal="Understand the current implementation",
                status="done",
                evidence=["file_read succeeded for temp/example.py"],
            ),
            Subtask(
                subtask_id="S2",
                title="Run verifier",
                goal="Verify the script",
                status="active",
                acceptance=["Verifier passes"],
                blockers=["No verifier has run yet"],
                next_action="Run python3 temp/example.py",
                attempts=1,
            ),
        ],
    )

    result = asyncio.run(control.loop_status(state))

    assert result["success"] is True
    ledger = result["output"]["subtask_ledger"]
    assert ledger["active_subtask_id"] == "S2"
    assert ledger["active_subtask"]["title"] == "Run verifier"
    assert ledger["done_subtask_ids"] == ["S1"]
    assert ledger["pending_subtask_ids"] == ["S2"]


def test_system_prompt_guards_header_only_tables_and_inline_tool_xml() -> None:
    state = _make_state()
    state.run_brief.original_task = "Check docker state on a remote host"

    prompt = build_system_prompt(
        state,
        "execute",
        available_tool_names=["ssh_exec", "task_complete"],
    )

    assert "only column headers and no data rows" in prompt
    assert "never infer rows with blank fields" in prompt
    assert "same turn that you formulate the answer" in prompt
    assert "never XML or angle-bracket markup inside thinking" in prompt


def test_task_complete_is_blocked_until_acceptance_is_met() -> None:
    state = _make_state()
    state.run_brief.acceptance_criteria = ["The script runs", "The test passes"]
    state.acceptance_ledger = {"The script runs": "done", "The test passes": "pending"}

    blocked = asyncio.run(control.task_complete("done", state=state, harness=None))

    assert blocked["success"] is False
    assert blocked["error"]
    assert "pending_acceptance_criteria" in blocked["metadata"]

    state.acceptance_ledger["The test passes"] = "passed"
    allowed = asyncio.run(control.task_complete("done", state=state, harness=None))

    assert allowed["success"] is True
    assert allowed["output"]["status"] == "complete"


def test_task_complete_blocks_weather_lookup_meta_summary_without_actual_answer() -> None:
    state = _make_state()
    state.run_brief.original_task = "do a web search, what is the weather in jax fl 4/26/26?"

    blocked = asyncio.run(
        control.task_complete(
            "Web search completed. Found 5 results for weather in Jacksonville FL on 4/26/26, including weather.com and timeanddate.com.",
            state=state,
            harness=None,
        )
    )

    assert blocked["success"] is False
    assert "user asked for the weather" in blocked["error"]
    assert blocked["metadata"]["reason"] == "lookup_answer_missing"
    assert blocked["metadata"]["lookup_kind"] == "weather"


def test_task_complete_allows_weather_lookup_with_explicit_unverified_answer() -> None:
    state = _make_state()
    state.run_brief.original_task = "what is the weather in jacksonville today?"

    allowed = asyncio.run(
        control.task_complete(
            "I could not verify the exact current weather from the fetched evidence, but the top returned sources were weather.com and weather.gov.",
            state=state,
            harness=None,
        )
    )

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

    # Auto-read enhancement: when file exists and is readable, patch is allowed
    # and the file is silently recorded as read.
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
    # Verify the file was auto-recorded as read
    assert str(target.resolve()) in state.scratchpad.get("_repair_cycle_reads", [])

    read_back = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))
    assert read_back["success"] is True
    assert allowed["metadata"]["occurrence_count"] == 1
    assert str(target.resolve()) in state.files_changed_this_cycle
    assert target.read_text(encoding="utf-8") == "patched\n"


def test_repair_cycle_accepts_existing_successful_file_read_for_same_path(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.repair_cycle_id = "repair-1"

    target = tmp_path / "example.txt"
    target.write_text("original\n", encoding="utf-8")
    state.tool_execution_records["op-read"] = {
        "tool_name": "file_read",
        "args": {"path": str(target)},
        "result": {
            "success": True,
            "metadata": {"path": str(target), "complete_file": True},
        },
    }

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
    assert target.read_text(encoding="utf-8") == "patched\n"


def test_repair_cycle_accepts_read_only_shell_exec_for_same_path(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.repair_cycle_id = "repair-1"

    target = tmp_path / "example.txt"
    target.write_text("original\n", encoding="utf-8")
    state.tool_execution_records["op-shell"] = {
        "tool_name": "shell_exec",
        "args": {"command": f"cat {target}"},
        "result": {
            "success": True,
            "output": "original\n",
            "metadata": {"command": f"cat {target}"},
        },
    }

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
    assert target.read_text(encoding="utf-8") == "patched\n"


def test_repair_cycle_rejects_mutating_shell_exec_for_same_path(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.repair_cycle_id = "repair-1"

    target = tmp_path / "example.txt"
    target.write_text("original\n", encoding="utf-8")
    state.tool_execution_records["op-shell"] = {
        "tool_name": "shell_exec",
        "args": {"command": f"sed -i 's/original/patched/' {target}"},
        "result": {
            "success": True,
            "output": "",
            "metadata": {"command": f"sed -i 's/original/patched/' {target}"},
        },
    }

    blocked = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="patched",
            replacement_text="final",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert blocked["success"] is False
    assert blocked["metadata"]["error_kind"] == "repair_cycle_read_required"


def test_repair_cycle_requires_new_read_after_failed_file_patch_on_same_path(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    state.repair_cycle_id = "repair-1"

    target = tmp_path / "example.txt"
    target.write_text("original\n", encoding="utf-8")
    state.tool_execution_records["op-read"] = {
        "tool_name": "file_read",
        "args": {"path": str(target)},
        "result": {
            "success": True,
            "metadata": {"path": str(target), "complete_file": True},
        },
    }
    state.tool_execution_records["op-patch-failed"] = {
        "tool_name": "file_patch",
        "args": {
            "path": str(target),
            "target_text": "missing",
            "replacement_text": "patched",
        },
        "result": {
            "success": False,
            "metadata": {"path": str(target), "error_kind": "patch_target_not_found"},
        },
    }

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
    assert blocked["metadata"]["error_kind"] == "repair_cycle_read_required"


def test_cached_file_read_satisfies_active_repair_cycle_read_gate(tmp_path: Path) -> None:
    from smallctl.harness.artifact_tracking import file_read_cache_key
    from smallctl.harness.tool_dispatch import _reuse_cached_file_read

    state = _make_state()
    state.cwd = str(tmp_path)
    state.repair_cycle_id = "repair-1"

    target = tmp_path / "example.txt"
    target.write_text("original\n", encoding="utf-8")
    artifact = ArtifactRecord(
        artifact_id="art-read",
        kind="file_read",
        source=str(target),
        created_at="2026-05-21T00:00:00+00:00",
        size_bytes=9,
        summary="file_read success",
        tool_name="file_read",
        metadata={"path": str(target), "complete_file": True},
    )
    state.artifacts[artifact.artifact_id] = artifact
    cache_key = file_read_cache_key(state.cwd, {"path": str(target)})
    assert cache_key
    state.scratchpad["file_read_cache"] = {cache_key: artifact.artifact_id}

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )

    cached = _reuse_cached_file_read(harness, {"path": str(target)})

    assert cached is not None
    assert cached.success is True
    assert str(target.resolve()) in state.scratchpad["_repair_cycle_reads"]


def test_repair_cycle_missing_file_read_unblocks_create_write(tmp_path: Path) -> None:
    state = _make_state()
    state.repair_cycle_id = "repair-1"

    target = tmp_path / "new.txt"

    blocked = asyncio.run(
        fs.file_write(
            path=str(target),
            content="hello\n",
            cwd=str(tmp_path),
            state=state,
        )
    )
    assert blocked["success"] is False
    assert blocked["metadata"]["error_kind"] == "repair_cycle_read_required"

    missing = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))
    assert missing["success"] is False
    assert missing["metadata"]["read_result"] == "missing"

    allowed = asyncio.run(
        fs.file_write(
            path=str(target),
            content="hello\n",
            cwd=str(tmp_path),
            state=state,
        )
    )
    assert allowed["success"] is True
    assert target.read_text(encoding="utf-8") == "hello\n"


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
    assert result["metadata"]["dry_run"] is False
    assert result["metadata"]["old_sha256"] != result["metadata"]["new_sha256"]
    assert "old value" in result["metadata"]["diff"]
    assert "new value" in result["metadata"]["diff"]
    assert target.read_text(encoding="utf-8") == "keep\nnew value\nkeep\n"


def test_file_patch_dry_run_returns_diff_without_mutating_file(tmp_path: Path) -> None:
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
            dry_run=True,
        )
    )

    assert result["success"] is True
    assert "Dry run" in result["output"]
    assert result["metadata"]["dry_run"] is True
    assert result["metadata"]["occurrence_count"] == 1
    assert result["metadata"]["replacement_count"] == 1
    assert "old value" in result["metadata"]["diff"]
    assert "new value" in result["metadata"]["diff_preview"]
    diff_lines = result["metadata"]["diff"].splitlines()
    assert diff_lines[0].startswith("--- ")
    assert diff_lines[1].startswith("+++ ")
    assert diff_lines[2].startswith("@@ ")
    assert target.read_text(encoding="utf-8") == "keep\nold value\nkeep\n"


def test_file_patch_dry_run_active_write_session_does_not_advance_stage_or_session(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "session.txt"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-dry-stage.txt"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("alpha beta gamma\n", encoding="utf-8")
    session = _make_open_write_session(target, session_id="ws-dry", intent="patch_existing")
    session.write_staging_path = str(stage)
    session.write_next_section = "patch"
    state.write_session = session

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="beta",
            replacement_text="delta",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
            dry_run=True,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["staged_only"] is True
    assert result["metadata"]["dry_run"] is True
    assert stage.read_text(encoding="utf-8") == "alpha beta gamma\n"
    assert session.write_sections_completed == []
    assert session.write_next_section == "patch"
    assert session.write_pending_finalize is False


def test_file_patch_empty_target_text_suggests_anchor_or_ast_patch(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.py"
    target.write_text("def run():\n    return False\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="",
            replacement_text="return True",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert "non-empty exact anchor" in result["error"]
    assert "`ast_patch`" in result["error"]
    assert result["metadata"]["error_kind"] == "patch_target_empty"
    assert result["metadata"]["suggested_tools"] == ["ast_patch"]
    assert "function" in result["metadata"]["recovery_hint"]


def test_file_patch_empty_target_text_on_empty_write_session_requires_file_write(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "replica_sync_checker.py"
    session = _make_open_write_session(target, session_id="ws-empty")
    session.write_next_section = "imports"
    state.write_session = session

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="",
            replacement_text="#!/usr/bin/env python3\nprint('ready')\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
        )
    )

    assert result["success"] is False
    assert "Use `file_write`" in result["error"]
    assert result["metadata"]["error_kind"] == "patch_target_empty_for_new_file"
    assert result["metadata"]["suggested_tools"] == ["file_write"]
    assert result["metadata"]["next_required_tool"] == {
        "tool_name": "file_write",
        "required_fields": ["path", "content", "section_name", "replace_strategy"],
        "required_arguments": {
            "path": str(target),
            "content": "#!/usr/bin/env python3\nprint('ready')\n",
            "replace_strategy": "overwrite",
            "section_name": "imports",
        },
        "reason": "empty_or_new_file_requires_file_write_not_file_patch",
    }


def test_file_patch_zero_occurrences_fails_without_mutating_file(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="alpah",
            replacement_text="replacement",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "patch_target_not_found"
    assert result["metadata"]["actual_occurrences"] == 0
    assert "not found" in result["metadata"]["ambiguity_hint"]
    assert result["metadata"]["target_text_preview"]["preview"] == "alpah"
    assert result["metadata"]["best_match"]["preview"] == "alpha"
    assert result["metadata"]["best_match"]["start_line"] == 1
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


def test_file_patch_occurrence_index_replaces_only_selected_match(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("beta beta beta\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="beta",
            replacement_text="gamma",
            expected_occurrences=3,
            occurrence_index=2,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["occurrence_count"] == 1
    assert result["metadata"]["actual_occurrences"] == 3
    assert result["metadata"]["selected_occurrence"] == 2
    assert target.read_text(encoding="utf-8") == "beta gamma beta\n"


def test_file_patch_occurrence_index_out_of_range_fails_without_mutating_file(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("beta beta\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="beta",
            replacement_text="gamma",
            expected_occurrences=2,
            occurrence_index=3,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "invalid_occurrence_index"
    assert result["metadata"]["occurrence_index"] == 3
    assert target.read_text(encoding="utf-8") == "beta beta\n"


def test_file_patch_regex_replacement_succeeds(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("version = 12\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text=r"version = (\d+)",
            replacement_text=r"version = \g<1>0",
            regex=True,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["patch_mode"] == "regex"
    assert result["metadata"]["occurrence_count"] == 1
    assert target.read_text(encoding="utf-8") == "version = 120\n"


def test_file_patch_regex_flags_are_applied(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("HEADER\nvalue\nfooter\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text=r"header.*footer",
            replacement_text="body",
            regex=True,
            case_insensitive=True,
            dotall=True,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["patch_mode"] == "regex"
    assert target.read_text(encoding="utf-8") == "body\n"


def test_file_patch_invalid_regex_fails_with_error_kind(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("alpha\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="[",
            replacement_text="beta",
            regex=True,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "invalid_regex"
    assert target.read_text(encoding="utf-8") == "alpha\n"


def test_file_patch_regex_broad_match_fails_on_expected_occurrence_mismatch(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.txt"
    target.write_text("one two three\n", encoding="utf-8")

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text=r"\w+",
            replacement_text="word",
            regex=True,
            expected_occurrences=1,
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "patch_occurrence_mismatch"
    assert result["metadata"]["actual_occurrences"] == 3
    assert target.read_text(encoding="utf-8") == "one two three\n"


def test_write_text_file_atomic_replace_failure_preserves_original(tmp_path: Path) -> None:
    target = tmp_path / "atomic.txt"
    target.write_text("original\n", encoding="utf-8")

    with patch("smallctl.tools.fs_write_sessions.Path.replace", side_effect=OSError("replace failed")):
        with pytest.raises(OSError):
            fs._write_text_file(target, "patched\n", encoding="utf-8")

    assert target.read_text(encoding="utf-8") == "original\n"


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


def test_patch_existing_file_patch_completes_first_section_for_finalization(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "task_queue.py"
    target.write_text("import logging\n", encoding="utf-8")
    session = _make_open_write_session(target, intent="patch_existing")
    session.write_next_section = "imports"
    state.write_session = session

    result = asyncio.run(
        fs.file_patch(
            path=str(target),
            target_text="import logging",
            replacement_text="import heapq\nimport logging",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["staged_only"] is True
    assert result["metadata"]["section_added"] is True
    assert result["metadata"]["write_session_final_chunk"] is True
    assert result["metadata"]["write_sections_completed"] == ["imports"]
    assert result["metadata"]["write_next_section"] == ""
    assert session.write_sections_completed == ["imports"]
    assert session.write_next_section == ""
    assert session.write_pending_finalize is True
    assert hidden_tool_reason("finalize_write_session", state=state, mode="chat") is None
    assert target.read_text(encoding="utf-8") == "import logging\n"
    assert Path(session.write_staging_path).read_text(encoding="utf-8") == "import heapq\nimport logging\n"


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
    assert "target path is still the canonical destination" in result["error"]
    assert "No sections are committed yet" in result["error"]
    assert result["metadata"]["error_kind"] == "patch_existing_requires_explicit_replace_strategy"
    assert result["metadata"]["staged_only"] is True
    assert result["metadata"]["write_session_id"] == session.write_session_id


def test_file_write_patch_existing_first_choice_rejects_append_strategy(tmp_path: Path) -> None:
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
            replace_strategy="append",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "patch_existing_requires_explicit_replace_strategy"
    assert result["metadata"]["replace_strategy"] == "append"
    assert "No sections are committed yet" in result["error"]


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
    # Path-based implicit resolution means the harness no longer rewrites the
    # model-supplied session ID; the tool layer matches by target path.
    assert pending.args["write_session_id"] == "ws-model-guess"
    assert pending.args["section_name"] == "imports"


def test_repair_active_write_session_args_backfills_missing_session_id_for_active_target() -> None:
    state = _make_state()
    target = Path("temp/task_queue.py")
    session = _make_open_write_session(target, session_id="ws-active")
    session.write_next_section = "helpers"
    state.write_session = session
    harness = SimpleNamespace(state=state)
    pending = PendingToolCall(
        tool_name="file_write",
        args={
            "path": str(target),
            "content": "def helper():\n    return 1\n",
        },
    )

    repaired = _repair_active_write_session_args(harness, pending)

    assert repaired is True
    assert "write_session_id" not in pending.args
    assert pending.args["section_name"] == "helpers"


def test_oversize_write_guidance_recommends_chunking_with_path_and_section() -> None:
    state = _make_state()
    target = Path("temp/task_queue.py")
    session = _make_open_write_session(target, session_id="ws-active")
    state.write_session = session

    class _FakeClient:
        model = "qwen3-4b"

    class _FakeConfig:
        small_model_hard_write_chars = 50

    harness = SimpleNamespace(state=state, client=_FakeClient(), config=_FakeConfig())
    pending = PendingToolCall(
        tool_name="file_write",
        args={
            "path": str(target),
            "content": "x" * 100,
        },
    )

    result = _detect_oversize_write_payload(harness, pending)

    assert result is not None
    message, meta = result
    assert meta["reason"] == "session_context_missing"
    assert "section_name" in message.lower()
    assert "path" in message.lower()
    assert "write_session_id" not in message.lower()


def test_empty_target_file_patch_repaired_to_file_write_for_empty_write_session() -> None:
    state = _make_state()
    target = Path("temp/replica_sync_checker.py")
    session = _make_open_write_session(target, session_id="ws-active")
    session.write_next_section = "imports"
    state.write_session = session
    events: list[tuple[str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, _message, **data: events.append((event, data)),
    )
    pending = PendingToolCall(
        tool_name="file_patch",
        args={
            "path": str(target),
            "target_text": "",
            "replacement_text": "import unittest\n",
            "write_session_id": "ws-active",
        },
    )

    repaired = _repair_empty_target_file_patch_to_file_write(harness, pending)

    assert repaired is True
    assert pending.tool_name == "file_write"
    assert pending.args == {
        "path": str(target),
        "content": "import unittest\n",
        "replace_strategy": "overwrite",
        "write_session_id": "ws-active",
        "section_name": "imports",
    }
    assert pending.parser_metadata["auto_repaired_from_tool"] == "file_patch"
    assert events[0][0] == "tool_call_auto_repaired"


def test_empty_target_file_patch_repaired_to_file_write_for_zero_byte_file(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "empty.py"
    target.write_text("", encoding="utf-8")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    pending = PendingToolCall(
        tool_name="file_patch",
        args={
            "path": str(target),
            "target_text": "",
            "replacement_text": "print('created')\n",
        },
    )

    repaired = _repair_empty_target_file_patch_to_file_write(harness, pending)

    assert repaired is True
    assert pending.tool_name == "file_write"
    assert pending.args == {
        "path": str(target),
        "content": "print('created')\n",
        "replace_strategy": "overwrite",
    }


def test_empty_target_file_patch_not_repaired_for_existing_file_edit(tmp_path: Path) -> None:
    state = _make_state()
    target = tmp_path / "example.py"
    target.write_text("def run():\n    return False\n", encoding="utf-8")
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    pending = PendingToolCall(
        tool_name="file_patch",
        args={
            "path": str(target),
            "target_text": "",
            "replacement_text": "return True",
        },
    )

    repaired = _repair_empty_target_file_patch_to_file_write(harness, pending)

    assert repaired is False
    assert pending.tool_name == "file_patch"
    assert pending.args["target_text"] == ""


def test_interpret_model_output_backfills_missing_write_session_metadata_for_active_target() -> None:
    state = _make_state()
    target = Path("temp/task_queue.py")
    session = _make_open_write_session(target, session_id="ws-active")
    session.write_next_section = "helpers"
    state.write_session = session

    harness = SimpleNamespace(
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
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-write-session-repair",
        run_mode="loop",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_write",
                args={
                    "path": str(target),
                    "content": "def helper():\n    return 1\n",
                },
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.DISPATCH_TOOLS
    pending = graph_state.pending_tool_calls[0]
    assert pending.args["write_session_id"] == "ws-active"
    assert pending.args["section_name"] == "helpers"


def test_interpret_model_output_repairs_empty_target_patch_before_dispatch() -> None:
    state = _make_state()
    target = Path("temp/replica_sync_checker.py")
    session = _make_open_write_session(target, session_id="ws-active")
    session.write_next_section = "imports"
    state.write_session = session
    runlog_events: list[tuple[str, dict[str, object]]] = []

    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(min_exploration_steps=0),
        summarizer=None,
        _extract_planning_request=lambda task: None,
        _record_experience=lambda *args, **kwargs: None,
        _runlog=lambda event, _message, **data: runlog_events.append((event, data)),
        _emit=lambda *args, **kwargs: None,
        _failure=lambda error, error_type="runtime", details=None: {
            "error": error,
            "error_type": error_type,
            "details": details or {},
        },
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-empty-patch-repair",
        run_mode="loop",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_patch",
                args={
                    "path": str(target),
                    "target_text": "",
                    "replacement_text": "import unittest\n",
                    "write_session_id": "ws-active",
                },
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.DISPATCH_TOOLS
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "file_write"
    assert pending.args == {
        "path": str(target),
        "content": "import unittest\n",
        "replace_strategy": "overwrite",
        "write_session_id": "ws-active",
        "section_name": "imports",
    }
    assert any(event == "tool_call_auto_repaired" for event, _data in runlog_events)


def test_prepare_loop_step_schema_repairs_argumentless_file_patch_before_dispatch() -> None:
    state = _make_state()
    emitted: list[object] = []
    runlog_events: list[tuple[str, dict[str, object]]] = []

    class _Registry:
        @staticmethod
        def get(tool_name: str):
            if tool_name == "file_patch":
                return SimpleNamespace(
                    schema={
                        "required": ["path", "target_text", "replacement_text"],
                        "properties": {
                            "path": {"type": "string"},
                            "target_text": {"type": "string"},
                            "replacement_text": {"type": "string"},
                        },
                    }
                )
            return None

    async def _emit(_handler, event) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        dispatcher=SimpleNamespace(phase="execute"),
        guards=SimpleNamespace(max_steps=35, max_tokens=None, max_consecutive_errors=5, max_repeated_actions=6),
        _cancel_requested=False,
        _runlog=lambda event, _message, **data: runlog_events.append((event, data)),
        _emit=_emit,
        _failure=lambda error, error_type="runtime", details=None: {
            "error": {"message": error, "type": error_type, "details": details or {}}
        },
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-compiled-schema-repair",
        run_mode="planning",
        pending_tool_calls=[
            PendingToolCall(tool_name="file_patch", args={}, tool_call_id="tool-empty-patch")
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(prepare_loop_step(graph_state, deps))

    assert graph_state.pending_tool_calls == []
    assert graph_state.final_result is None
    assert state.recent_messages
    repair = state.recent_messages[-1]
    assert repair.metadata["recovery_kind"] == "schema_validation"
    assert repair.metadata["tool_name"] == "file_patch"
    assert repair.metadata["required_fields"] == ["path", "target_text", "replacement_text"]
    assert "file_patch" in repair.content
    assert "path, target_text, replacement_text" in repair.content
    assert emitted
    assert getattr(emitted[-1], "event_type", None).value == "alert"
    assert any(event == "tool_call_repair" for event, _data in runlog_events)


def test_interpret_model_output_marks_action_stall_nudge_as_recovery_message() -> None:
    state = _make_state()
    state.run_brief.original_task = "find the vikunja docker compose on the remote host"
    harness = SimpleNamespace(
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
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-action-stall",
        run_mode="loop",
    )
    graph_state.last_assistant_text = "I'll search the remote host for the docker compose file next."
    graph_state.last_thinking_text = "I should use ssh_exec to find the compose file."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["is_recovery_nudge"] is True
    assert state.recent_messages[-1].metadata["recovery_kind"] == "action_stall"


def test_interpret_model_output_ignores_reasoning_fallback_text_for_action_stall() -> None:
    state = _make_state()
    state.run_brief.original_task = "find the vikunja docker compose on the remote host"
    state.scratchpad["_assistant_text_from_reasoning_fallback"] = True
    harness = SimpleNamespace(
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
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-reasoning-fallback",
        run_mode="loop",
    )
    graph_state.last_assistant_text = "I'll search the remote host for the docker compose file next."
    graph_state.last_thinking_text = "I should use ssh_exec to find the compose file."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["is_recovery_nudge"] is True
    assert state.recent_messages[-1].metadata["recovery_kind"] == "blank_message"
    assert all(message.metadata.get("recovery_kind") != "action_stall" for message in state.recent_messages)


def test_interpret_model_output_tags_phase_contract_block_nudges_as_recovery_messages() -> None:
    state = _make_state()
    state.strategy = {"thought_architecture": "staged_reasoning"}
    state.current_phase = "explore"
    harness = SimpleNamespace(
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
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-phase-contract-block",
        run_mode="loop",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="ssh_exec",
                args={"command": "find / -name docker-compose.yml"},
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert graph_state.pending_tool_calls == []
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["is_recovery_nudge"] is True
    assert state.recent_messages[-1].metadata["recovery_kind"] == "phase_contract_all_tools_blocked"


def test_interpret_model_output_tags_min_exploration_nudge_as_recovery_message() -> None:
    state = _make_state()
    state.strategy = {"thought_architecture": "staged_reasoning"}
    state.current_phase = "explore"
    state.step_count = 1
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(min_exploration_steps=3),
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
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-min-exploration",
        run_mode="loop",
        pending_tool_calls=[PendingToolCall(tool_name="task_complete", args={"message": "done"})],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["is_recovery_nudge"] is True
    assert state.recent_messages[-1].metadata["recovery_kind"] == "phase_contract_min_exploration_steps"


def test_interpret_model_output_tags_missing_task_complete_nudge_as_recovery_message() -> None:
    state = _make_state()
    harness = SimpleNamespace(
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
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-missing-task-complete",
        run_mode="loop",
    )
    graph_state.last_assistant_text = "The investigation is complete and the compose file has been identified."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["is_recovery_nudge"] is True
    assert state.recent_messages[-1].metadata["recovery_kind"] == "missing_task_complete"
    assert "in this same turn" in state.recent_messages[-1].content


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


def test_file_read_does_not_shadow_existing_target_with_fresh_empty_write_session(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "file_deduper.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('target')\n", encoding="utf-8")

    state.write_session = WriteSession(
        write_session_id="ws-fresh",
        write_target_path=str(target),
        write_session_intent="replace_file",
        status="open",
    )

    result = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))

    assert result["success"] is True
    assert result["output"] == "print('target')"
    assert result["metadata"]["read_from_staging"] is False
    assert state.write_session.write_staging_path == ""


def test_file_read_does_not_shadow_empty_target_with_fresh_empty_stage(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "patch_dependency_sim.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("", encoding="utf-8")

    session = _make_open_write_session(target, intent="replace_file")
    stage_path = fs._session_stage_path(session.write_session_id, target, str(tmp_path))
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text("", encoding="utf-8")
    session.write_staging_path = str(stage_path)
    state.write_session = session

    result = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))

    assert result["success"] is True
    assert result["output"] == ""
    assert result["metadata"]["read_from_staging"] is False
    assert result["metadata"]["source_path"] == str(target)


def test_file_read_missing_target_with_fresh_empty_stage_reports_missing(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "patch_dependency_sim.py"

    session = _make_open_write_session(target, intent="replace_file")
    stage_path = fs._session_stage_path(session.write_session_id, target, str(tmp_path))
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text("", encoding="utf-8")
    session.write_staging_path = str(stage_path)
    state.write_session = session

    result = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))

    assert result["success"] is False
    assert result["metadata"]["read_result"] == "missing"


def test_file_read_allows_empty_stage_after_write_session_progress(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "file_deduper.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('target')\n", encoding="utf-8")

    session = _make_open_write_session(target, intent="replace_file")
    stage_path = fs._session_stage_path(session.write_session_id, target, str(tmp_path))
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_text("", encoding="utf-8")
    session.write_staging_path = str(stage_path)
    session.write_sections_completed = ["body"]
    state.write_session = session

    result = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))

    assert result["success"] is True
    assert result["output"] == ""
    assert result["metadata"]["read_from_staging"] is True
    assert result["metadata"]["source_path"] == str(stage_path)


def test_chunk_mode_prearm_does_not_abandon_existing_target_from_path_symbols(tmp_path: Path) -> None:
    from smallctl.graph.tool_write_session_policy import _ensure_chunk_write_session

    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "file_deduper.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("def real_func():\n    return 1\n", encoding="utf-8")
    state.scratchpad["_force_chunk_mode_targets"] = [str(target)]
    state.run_brief = SimpleNamespace(original_task=f'fix "{target}"')

    harness = SimpleNamespace(
        state=state,
        client=SimpleNamespace(model="qwen3.5-4b"),
        config=SimpleNamespace(),
        _runlog=lambda *args, **kwargs: None,
    )

    session = _ensure_chunk_write_session(harness, str(target))

    assert session is not None
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "def real_func():\n    return 1\n"
    assert not target.with_suffix(target.suffix + ".abandoned").exists()

    result = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))
    assert result["success"] is True
    assert result["output"] == "def real_func():\n    return 1"
    assert result["metadata"]["read_from_staging"] is False


def test_extract_symbols_from_task_ignores_quoted_paths() -> None:
    from smallctl.graph.write_session_health import extract_symbols_from_task

    symbols = extract_symbols_from_task(
        'fix "/home/stephen/Scripts/Harness-Redo/temp/file_deduper.py" '
        "and update `extract_symbols_from_task` for WriteSession safety"
    )

    assert "extract_symbols_from_task" in symbols
    assert "WriteSession" in symbols
    assert "Harness" not in symbols
    assert "Redo" not in symbols
    assert "Scripts" not in symbols
    assert "file_deduper" not in symbols
    assert "stephen" not in symbols


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


def test_ssh_exec_as_root_strips_redundant_sudo() -> None:
    stripped, changed = network._strip_redundant_root_sudo("sudo apt update && sudo apt upgrade -y", "root")

    assert changed is True
    assert stripped == "apt update && apt upgrade -y"


def test_store_verifier_classifies_remote_installer_timeout() -> None:
    state = _make_state()
    result = ToolEnvelope(
        success=False,
        error="SSH command timed out after 60s",
        metadata={
            "failure_kind": "timeout",
            "ssh_error_class": "command_timeout",
            "output": {
                "stdout": " * Installation Started\n\n * Testing internet connection.................................",
                "stderr": "",
                "exit_code": None,
            },
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={
            "host": "192.168.1.89",
            "user": "root",
            "command": "cd /root/fogproject/bin && ./installfog.sh -y",
        },
    )

    assert verdict is not None
    assert verdict["failure_mode"] == "long_running_remote_command"
    assert state.last_failure_class == "long_running_remote_command"
    assert state.scratchpad["_last_long_running_remote_command_timeout"]["host"] == "192.168.1.89"


def test_docker_detached_run_timeout_with_container_id_is_not_long_running_failure() -> None:
    state = _make_state()
    result = ToolEnvelope(
        success=False,
        error="SSH command timed out after 60s",
        metadata={
            "failure_kind": "timeout",
            "ssh_error_class": "command_timeout",
            "output": {
                "stdout": "Unable to find image 'nginx:latest' locally\nStatus: Downloaded newer image for nginx:latest\n0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n",
                "stderr": "",
                "exit_code": None,
            },
        },
    )

    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=result,
        arguments={
            "host": "192.168.1.161",
            "user": "root",
            "command": "docker run -d --name qwen-nginx-easy -p 8088:80 nginx:latest",
        },
    )

    assert verdict is not None
    assert verdict["failure_mode"] != "long_running_remote_command"
    assert "_last_long_running_remote_command_timeout" not in state.scratchpad


def test_dispatch_blocks_file_write_after_remote_installer_timeout() -> None:
    state = _make_state()
    state.run_brief.original_task = "install fogserver on root@192.168.1.89"
    state.scratchpad["_last_long_running_remote_command_timeout"] = {
        "host": "192.168.1.89",
        "command": "cd /root/fogproject/bin && ./installfog.sh -y",
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

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        dispatcher=SimpleNamespace(phase="repair"),
        log=logging.getLogger("test.plan.remote_timeout_write_guard"),
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-remote-installer",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_write",
                args={"path": "foginstall.text", "content": "sudo apt update"},
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == []
    assert graph_state.last_tool_results == []
    assert graph_state.pending_tool_calls == []
    assert state.recent_messages[-1].metadata["recovery_kind"] == "long_running_remote_command"
    assert "larger `timeout_sec`" in state.recent_messages[-1].content
    assert any(event == "long_running_remote_timeout_write_guard" for event, _message, _data in runlog_events)


def test_dispatch_allows_required_report_write_after_remote_timeout() -> None:
    state = _make_state()
    state.run_brief.original_task = (
        "Create a Docker nginx service and create a short report at /tmp/qwen-docker-easy-report.txt"
    )
    state.scratchpad["_last_long_running_remote_command_timeout"] = {
        "host": "192.168.1.161",
        "command": "docker run -d --name qwen-nginx-easy -p 8088:80 nginx:latest",
    }
    pending = PendingToolCall(
        tool_name="ssh_file_write",
        args={"path": "/tmp/qwen-docker-easy-report.txt", "content": "nginx running"},
    )

    assert _long_running_remote_timeout_write_guard(state, pending) is None


def test_docker_report_chain_does_not_trip_guard_on_probe_failures() -> None:
    """Replay the ae327dc2 chain: successful docker run timeout, report write, then probe failures.

    With the fixes, the detached docker run timeout is not treated as a long-running
    installer failure, the required report write is allowed, and the subsequent
    docker-inspect syntax errors do not immediately trip max_consecutive_errors.
    """
    state = _make_state()
    state.run_brief.original_task = (
        "Create a Docker nginx service on 192.168.1.161 and create a short report "
        "at /tmp/qwen-docker-easy-report.txt"
    )

    # 1) docker run -d succeeds but the local SSH client times out.
    run_result = ToolEnvelope(
        success=False,
        error="SSH command timed out after 60s",
        metadata={
            "failure_kind": "timeout",
            "ssh_error_class": "command_timeout",
            "output": {
                "stdout": (
                    "Unable to find image 'nginx:latest' locally\n"
                    "Status: Downloaded newer image for nginx:latest\n"
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
                ),
                "stderr": "",
                "exit_code": None,
            },
        },
    )
    verdict = _store_verifier_verdict(
        state,
        tool_name="ssh_exec",
        result=run_result,
        arguments={
            "host": "192.168.1.161",
            "user": "root",
            "command": "docker run -d --name qwen-nginx-easy -p 8088:80 nginx:latest",
        },
    )
    assert verdict is not None
    assert verdict["failure_mode"] != "long_running_remote_command"
    assert "_last_long_running_remote_command_timeout" not in state.scratchpad

    # 2) Required report write is allowed because the task names the path explicitly.
    report_write = PendingToolCall(
        tool_name="ssh_file_write",
        args={"path": "/tmp/qwen-docker-easy-report.txt", "content": "nginx is running"},
    )
    assert _long_running_remote_timeout_write_guard(state, report_write) is None

    # 3) Model runs docker inspect with an invalid template (two probe failures).
    for _ in range(2):
        state.recent_errors.append(
            "ssh_exec: docker inspect --format '{{json .PortMappings}}' qwen-nginx-easy failed: "
            "Template parsing error: template: :1:7: executing \"\" at <.PortMappings>: map has no entry for key \"PortMappings\""
        )

    # 4) Guard does not trip on only the probe failures.
    assert check_guards(state, GuardConfig(max_consecutive_errors=5)) is None

    # 5) After enough additional countable errors, the guard should still trip.
    for i in range(3):
        state.recent_errors.append(f"ssh_exec: additional failure {i}")
    guard_error = check_guards(state, GuardConfig(max_consecutive_errors=5))
    assert guard_error is not None
    assert "max_consecutive_errors" in guard_error


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


def test_dispatch_tools_blocks_local_file_read_for_remote_absolute_path_and_nudges() -> None:
    state = _make_state()
    state.run_brief.original_task = "fix the remote page on root@192.168.1.63 over SSH"
    state.task_mode = "remote_execute"
    state.scratchpad["_session_ssh_targets"] = {
        "192.168.1.63": {"host": "192.168.1.63", "user": "root", "confirmed": True}
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
            return {"file_read", "ssh_exec", "ssh_file_read"}

        @staticmethod
        def get(tool_name: str):
            if tool_name == "file_read":
                return SimpleNamespace(schema={"required": ["path"]})
            if tool_name == "ssh_exec":
                return SimpleNamespace(schema={"required": ["host", "command"]}, phase_allowed=lambda phase: True)
            return None

    harness = SimpleNamespace(
        state=state,
        registry=_Registry(),
        dispatcher=SimpleNamespace(phase="explore"),
        log=logging.getLogger("test.plan.remote_file_guard"),
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
        _failure=lambda *args, **kwargs: {"error": "guard"},
        _dispatch_tool_call=_dispatch_tool_call,
        _active_dispatch_task=None,
    )

    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-remote-file-guard",
        run_mode="execute",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_read",
                args={"path": "/var/www/html/llm-explainer.html"},
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(dispatch_tools(graph_state, deps))

    assert dispatched == []
    assert graph_state.last_tool_results == []
    assert state.recent_messages
    assert (
        state.recent_messages[-1].content
        == "This path appears to be on the remote host. Use `ssh_file_read`, not local `file_read`."
    )
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


def test_resume_continue_after_repeated_tool_loop_reseeds_guard_and_adds_nudge() -> None:
    state = _make_state()
    state.pending_interrupt = {
        "kind": "repeated_tool_loop_resume",
        "tool_name": "dir_list",
        "arguments": {},
        "guidance": (
            "Repeated dir_list loop detected. The chat preview shows up to 50 directory items, "
            "so trust the visible listing if the target is present."
        ),
    }
    state.tool_history = [{"tool_name": "dir_list"}]
    emitted: list[object] = []

    async def _emit(_handler, event) -> None:
        emitted.append(event)

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _log_conversation_state=lambda *args, **kwargs: None,
        _is_continue_like_followup=lambda value: str(value or "").strip().lower() == "continue",
        _emit=_emit,
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="execute")
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    asyncio.run(resume_loop_run(graph_state, deps, human_input="continue"))

    assert state.pending_interrupt is None
    assert graph_state.pending_interrupt is None
    assert graph_state.interrupt_payload is None
    assert state.recent_messages[-2].role == "user"
    assert state.recent_messages[-2].content == "continue"
    assert state.recent_messages[-1].role == "system"
    assert "Do not call `dir_list` again with the same arguments" in state.recent_messages[-1].content
    assert "trust the visible listing" in state.recent_messages[-1].content
    # With the 3+ threshold fix, _dir_list_same_path_repeat_is_loop requires 3 total
    # calls. After resume there is only 1 prior call, so the generic loop guard fires.
    assert "Guard tripped: repeated tool call loop" in (
        _detect_repeated_tool_loop(harness, PendingToolCall(tool_name="dir_list", args={})) or ""
    )
    assert getattr(emitted[0], "content", "") == "continue"


def test_dir_list_repeat_loop_limits_are_50_percent_tighter_than_default_strict_tools() -> None:
    state = _make_state()
    harness = SimpleNamespace(state=state)

    dir_list_limits = _repeat_loop_limits(harness, PendingToolCall(tool_name="dir_list", args={"path": "."}))
    file_read_limits = _repeat_loop_limits(harness, PendingToolCall(tool_name="file_read", args={"path": "src/app.py"}))

    assert dir_list_limits == (2, 4, 2)
    assert file_read_limits == (7, 7, 3)


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


def test_shell_exec_blocks_unpromoted_write_session_target_with_resume_hint(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "circuit_breaker.py"
    state.write_session = WriteSession(
        write_session_id="ws-shell-guard",
        write_target_path=str(target),
        write_session_mode="chunked_author",
        write_sections_completed=["imports"],
        write_current_section="imports",
        write_next_section="tests_entrypoint",
        status="open",
    )

    result = asyncio.run(
        shell.shell_exec(
            command=f"python3 {target}",
            state=state,
            harness=None,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "write_session_unpromoted_target_path"
    assert result["metadata"]["next_required_tool"]["tool_name"] == "file_write"
    assert (
        result["metadata"]["next_required_tool"]["required_arguments"]["path"]
        == str(target)
    )
    assert (
        result["metadata"]["next_required_tool"]["required_arguments"]["section_name"]
        == "tests_entrypoint"
    )


def test_shell_exec_blocks_unpromoted_write_session_target_with_finalize_hint(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "circuit_breaker.py"
    state.write_session = WriteSession(
        write_session_id="ws-shell-finalize",
        write_target_path=str(target),
        write_session_mode="local_repair",
        write_sections_completed=["full_script"],
        write_current_section="full_script",
        write_next_section="",
        status="open",
    )

    result = asyncio.run(
        shell.shell_exec(
            command=f"python3 {target}",
            state=state,
            harness=None,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "write_session_unpromoted_target_path"
    assert result["metadata"]["next_required_tool"]["tool_name"] == "finalize_write_session"


def test_shell_exec_blocks_active_write_session_artifact_delete(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "temp" / "circuit_breaker.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws_8148b8__circuit_breaker__stage.py"
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("print('staged')\n", encoding="utf-8")
    state.write_session = WriteSession(
        write_session_id="ws_8148b8",
        write_target_path=str(target),
        write_session_mode="chunked_author",
        write_staging_path=str(stage),
        write_sections_completed=["full_script"],
        status="open",
    )

    result = asyncio.run(
        shell.shell_exec(
            command=f"rm -rf {tmp_path}/.smallctl/write_sessions/ws_8148b8*",
            state=state,
            harness=None,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "write_session_artifact_delete_blocked"
    assert result["metadata"]["error_kind"] == "write_session_artifact_delete_blocked"
    assert result["metadata"]["write_session_id"] == "ws_8148b8"
    assert stage.exists()


def test_shell_exec_blocks_recently_touched_directory_delete(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    package_dir = tmp_path / "temp" / "generated_pkg"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("broken export\n", encoding="utf-8")
    state.challenge_progress.task_category = "coding"
    state.challenge_progress.code_change_count = 1
    state.challenge_progress.last_code_change_paths = [str(init_file)]

    result = asyncio.run(
        shell.shell_exec(
            command=f"rm -rf {package_dir}",
            state=state,
            harness=None,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "workspace_destructive_delete_blocked"
    assert result["metadata"]["blocked_targets"][0]["reasons"] == ["contains_protected_working_set_path"]
    assert package_dir.exists()
    assert init_file.exists()


def test_shell_exec_allows_cache_only_delete(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    cache_dir = tmp_path / "temp" / "__pycache__"
    cache_dir.mkdir(parents=True)
    (cache_dir / "module.cpython-312.pyc").write_bytes(b"cache")

    result = asyncio.run(
        shell.shell_exec(
            command=f"rm -rf {cache_dir}",
            state=state,
            harness=None,
        )
    )

    assert result["success"] is True
    assert not cache_dir.exists()


def test_shell_exec_blocks_mixed_cache_and_protected_delete(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    cache_dir = tmp_path / "temp" / "__pycache__"
    package_dir = tmp_path / "temp" / "generated_pkg"
    cache_dir.mkdir(parents=True)
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("broken export\n", encoding="utf-8")
    state.challenge_progress.task_category = "coding"
    state.challenge_progress.code_change_count = 1
    state.challenge_progress.last_code_change_paths = [str(init_file)]

    result = asyncio.run(
        shell.shell_exec(
            command=f"rm -rf {cache_dir} {package_dir}",
            state=state,
            harness=None,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["reason"] == "workspace_destructive_delete_blocked"
    assert str(cache_dir.resolve()) in result["metadata"]["allowed_targets"]
    assert package_dir.exists()
    assert init_file.exists()


def test_shell_exec_allows_exact_user_requested_delete(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target_dir = tmp_path / "temp" / "obsolete_pkg"
    target_dir.mkdir(parents=True)
    (target_dir / "__init__.py").write_text("obsolete\n", encoding="utf-8")
    state.run_brief.original_task = "Delete obsolete_pkg and clean up the generated package."

    result = asyncio.run(
        shell.shell_exec(
            command=f"rm -rf {target_dir}",
            state=state,
            harness=None,
        )
    )

    assert result["success"] is True
    assert not target_dir.exists()


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

    assert "Guard tripped: repeated tool call loop (shell_exec repeated with identical arguments)" in (
        _detect_repeated_tool_loop(harness, pending) or ""
    )


def test_web_fetch_budget_exhaustion_blocks_followup_fetch_before_repeat() -> None:
    state = _make_state()
    state.scratchpad["_web_fetch_budget_exhausted"] = {
        "error": "Web fetch budget exhausted for this run.",
        "terminal": True,
    }
    harness = SimpleNamespace(state=state)
    pending = PendingToolCall(tool_name="web_fetch", args={"result_id": "r6"})

    message = _detect_repeated_tool_loop(harness, pending) or ""

    assert "Web fetch budget exhausted for this run" in message
    assert "Do not retry web_fetch in this run" in message


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

    assert "Guard tripped: repeated tool call loop (artifact_print repeated with identical arguments)" in (
        _detect_repeated_tool_loop(harness, pending) or ""
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
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
        {"tool_name": "file_read", "fingerprint": fingerprint},
    ]
    pending = PendingToolCall(tool_name="file_read", args={"path": "src/app.py"})

    allow_repeated_tool_call_once(harness, "file_read", {"path": "src/app.py"})

    assert _detect_repeated_tool_loop(harness, pending) is None
    assert "_repeat_guard_one_shot_fingerprints" not in state.scratchpad
    tripped_msg = _detect_repeated_tool_loop(harness, pending) or ""
    assert "Guard tripped" in tripped_msg
    assert "file_read" in tripped_msg


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
        HarnessConfig(endpoint="http://test/v1", model="qwen3.5:4b", phase="execute", contract_flow_ui=True),
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
            "_context_window": status.context_window,
            "_api_errors": status.api_errors,
            "_phase_transition": "",
            "_waiver_reason": "",
            "_blocker_summary": "",
            "_blocker_persistent": False,
            "_recovery_banner": "Recovery: verifier loop (2 rejects)",
        }
    )

    text = StatusBar._build_status_text(bar)
    assert "contract: verify" in text
    assert "acceptance: 1/2" in text
    assert "verdict: pass | pytest | exit 0" in text
    assert "Recovery: verifier loop (2 rejects)" in text

    bar.__dict__["_vertical"] = True
    vertical_text = StatusBar._build_status_text(bar)
    assert "[bold #93c5fd]Model[/]" in vertical_text
    assert "[bold #93c5fd]Run[/]" in vertical_text
    assert "[bold #93c5fd]Usage[/]" in vertical_text
    assert "\n" in vertical_text


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

    assert "Use `file_patch` for small exact edits when you know the exact target text." in prompt
    assert "Use `ast_patch` when the edit is easier to describe by function, class, import, call, argument, or dataclass-field structure." in prompt
    assert "Use `file_write` for new files, large sections, or chunked authoring." in prompt
    assert "When resuming an active session, prefer `file_write` for chunk continuation" in prompt
    assert "If you need a narrow repair inside the staged copy, prefer `file_patch` for exact text or `ast_patch` for structural edits." in prompt
    assert "The target path is the canonical destination; the staged copy is for read/verify context while the session is active." in prompt
    assert "If prior chunks are no longer visible because tool previews were compacted or truncated" in prompt
    assert "Do not assume earlier chunks were lost" in prompt


def test_system_prompt_surfaces_small_model_tool_routing_card() -> None:
    state = _make_state()
    state.scratchpad["_model_name"] = "qwen3.5:4b"

    prompt = build_system_prompt(state, "execute", available_tool_names=["shell_exec", "ssh_exec"])

    assert "never invent aliases like `use_shell_exec`" in prompt
    assert "Remote host/IP/user/password mentioned means `ssh_exec`." in prompt
    assert "`shell_exec` is local-only." in prompt
    assert "Use exactly this SSH shape" in prompt
    assert "Never send both `host` and `target`." in prompt
    assert "When connecting as `root`, do not prefix the remote command with `sudo`" in prompt
    assert 'SSH_EXEC EXAMPLE: `{"host":"192.168.1.63","user":"root","password":"...","command":"whoami"}`.' in prompt
    assert 'INVALID SSH_EXAMPLE: do not send `{"host":"192.168.1.63","target":"root@192.168.1.63",...}`.' in prompt
    assert "do not rely on retrieved historical notes alone" in prompt
    assert "do not try to satisfy a shell/SSH/file guard by storing the intended command in memory" in prompt


def test_system_prompt_write_recovery_surfaces_stage_read_before_overwrite() -> None:
    state = _make_state()
    target = Path("./temp/logwatch.py")
    state.write_session = _make_open_write_session(target)

    prompt = build_system_prompt(state, "execute")

    assert "Retrieved Artifact Snippets, previews, and compact summaries do NOT count as a full artifact read." in prompt
    assert "first read 100% of the current content" in prompt
    assert "artifact_read(artifact_id='ws-1__stage')" in prompt
    assert "covered 100% of the current staged content" in prompt
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

    blocked = asyncio.run(control.task_complete("done", state=state, harness=None))

    assert blocked["success"] is False
    assert "Latest verifier:" in blocked["error"]
    assert "docker ps" in blocked["error"]
    assert "docker: command not found" in blocked["error"]


def test_task_complete_allows_diagnostic_failure_report_with_failing_verifier() -> None:
    state = _make_state()
    state.run_brief.original_task = "rca why the fog pxe server failed to install on the remote host"
    state.last_verifier_verdict = {
        "tool": "ssh_exec",
        "target": "curl -Is https://192.168.1.89:8080/fog/",
        "command": "curl -Is https://192.168.1.89:8080/fog/",
        "exit_code": 7,
        "key_stdout": "curl: (7) Failed to connect to 192.168.1.89 port 8080",
        "key_stderr": "",
        "verdict": "fail",
    }
    state.run_brief.acceptance_criteria = ["Report the root cause or blocker"]

    result = asyncio.run(
        control.task_complete(
            "RCA: site verification failed; port 8080 is not responding to curl.",
            state=state,
            harness=None,
        )
    )

    assert result["success"] is True
    assert result["output"]["status"] == "complete"


def test_task_complete_still_blocks_non_diagnostic_failing_verifier() -> None:
    state = _make_state()
    state.run_brief.original_task = "fix the docker deployment"
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "docker ps",
        "command": "docker ps",
        "exit_code": 127,
        "key_stderr": "bash: line 1: docker: command not found",
        "verdict": "fail",
    }

    blocked = asyncio.run(control.task_complete("Docker is not working.", state=state, harness=None))

    assert blocked["success"] is False
    assert "latest verifier verdict is still failing" in blocked["error"]


def test_task_complete_missing_module_blocker_recommends_dependency_install(tmp_path: Path) -> None:
    target = tmp_path / "temp" / "snake.py"
    target.parent.mkdir(parents=True)
    target.write_text("import pygame\n", encoding="utf-8")
    venv_python = target.parent / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    state = _make_state()
    state.cwd = str(tmp_path)
    state.challenge_progress.task_category = "coding"
    state.challenge_progress.code_change_count = 1
    state.challenge_progress.last_code_change_paths = ["temp/snake.py"]
    state.challenge_progress.verified_after_last_change = False
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "cd temp && python3 -c \"import snake\"",
        "command": "cd temp && python3 -c \"import snake\"",
        "exit_code": 1,
        "key_stderr": "ModuleNotFoundError: No module named 'pygame'",
        "verdict": "fail",
        "failure_mode": "import",
    }

    blocked = asyncio.run(control.task_complete("done", state=state, harness=None))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "missing_runtime_dependency"
    action = blocked["metadata"]["next_required_action"]
    assert "pygame" in action["required_arguments"]["command"]
    assert str(venv_python) in action["required_arguments"]["command"]
    assert "missing Python module `pygame`" in "\n".join(action["notes"])


def test_task_complete_post_change_verification_block_without_dependency_does_not_crash(tmp_path: Path) -> None:
    target = tmp_path / "temp" / "tetris-main.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('ok')\n", encoding="utf-8")

    state = _make_state()
    state.cwd = str(tmp_path)
    state.challenge_progress.task_category = "coding"
    state.challenge_progress.code_change_count = 1
    state.challenge_progress.last_code_change_paths = ["temp/tetris-main.py"]
    state.challenge_progress.verified_after_last_change = False

    blocked = asyncio.run(control.task_complete("done", state=state, harness=None))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "post_change_verification_required"
    action = blocked["metadata"]["next_required_action"]
    assert action["tool_name"] == "shell_exec"
    assert action["required_arguments"]["command"] == "python3 -m py_compile temp/tetris-main.py"


def test_task_complete_blocks_phase_promotion_on_syntax_only_verifier() -> None:
    state = _make_state()
    state.run_brief.original_task = "implement phase 3 of the tetris game"
    state.challenge_progress.task_category = "coding"
    state.challenge_progress.code_change_count = 1
    state.challenge_progress.verified_after_last_change = True
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "python3 -m py_compile temp/tetris_main.py",
        "command": "python3 -m py_compile temp/tetris_main.py",
        "exit_code": 0,
        "key_stdout": "",
        "key_stderr": "",
        "verdict": "pass",
        "acceptance_delta": {"status": "satisfied", "notes": ["execution succeeded"]},
    }

    blocked = asyncio.run(control.task_complete("Phase 3 complete", state=state, harness=None))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "phase_promotion_behavioral_verifier_required"
    assert blocked["metadata"]["verifier_quality"] == {"score": 1, "label": "syntax"}
    assert blocked["metadata"]["required_verifier_quality"] == {"score": 3, "label": "behavioral"}
    notes = "\n".join(blocked["metadata"]["next_required_action"]["notes"])
    assert "behavioral smoke verifier" in notes


def test_task_complete_blocks_phase_promotion_on_import_only_verifier() -> None:
    state = _make_state()
    state.run_brief.original_task = "implement phase 2 of the tetris game"
    state.challenge_progress.task_category = "coding"
    state.challenge_progress.code_change_count = 1
    state.challenge_progress.verified_after_last_change = True
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "python3 -c \"import tetris_main\"",
        "command": "python3 -c \"import tetris_main\"",
        "exit_code": 0,
        "key_stdout": "",
        "key_stderr": "",
        "verdict": "pass",
        "acceptance_delta": {"status": "satisfied", "notes": ["execution succeeded"]},
    }

    blocked = asyncio.run(control.task_complete("Phase 2 complete", state=state, harness=None))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "phase_promotion_behavioral_verifier_required"
    assert blocked["metadata"]["verifier_quality"] == {"score": 2, "label": "import"}


def test_task_complete_blocks_phase_promotion_on_timeout_verifier_even_if_waived() -> None:
    state = _make_state()
    state.run_brief.original_task = "continue with phase 3 implementation"
    state.challenge_progress.task_category = "coding"
    state.challenge_progress.code_change_count = 1
    state.challenge_progress.verified_after_last_change = True
    state.acceptance_waived = True
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "cd temp && .venv/bin/python tetris_main.py",
        "command": "cd temp && .venv/bin/python tetris_main.py",
        "exit_code": None,
        "key_stdout": "",
        "key_stderr": "",
        "verdict": "fail",
        "failure_mode": "environment",
        "acceptance_delta": {"status": "blocked", "notes": ["Command timed out after 30s"]},
    }

    blocked = asyncio.run(control.task_complete("Phase 3 complete", state=state, harness=None))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "phase_promotion_verifier_not_passing"
    assert blocked["metadata"]["required_verifier_quality"] == {"score": 3, "label": "behavioral"}


def test_task_complete_allows_phase_promotion_with_behavioral_verifier() -> None:
    state = _make_state()
    state.run_brief.original_task = "implement phase 3 of the tetris game"
    state.challenge_progress.task_category = "coding"
    state.challenge_progress.code_change_count = 1
    state.challenge_progress.verified_after_last_change = True
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "python3 -c \"import tetris_main as t; e=t.GameEngine(); assert e.spawn_piece(); e.handle_key_event(t.pygame.K_LEFT)\"",
        "command": "python3 -c \"import tetris_main as t; e=t.GameEngine(); assert e.spawn_piece(); e.handle_key_event(t.pygame.K_LEFT)\"",
        "exit_code": 0,
        "key_stdout": "",
        "key_stderr": "",
        "verdict": "pass",
        "acceptance_delta": {"status": "satisfied", "notes": ["execution succeeded"]},
    }

    allowed = asyncio.run(control.task_complete("Phase 3 complete", state=state, harness=None))

    assert allowed["success"] is True
    assert allowed["output"]["status"] == "complete"


def test_task_complete_blocks_active_phase_contract_missing_symbol(tmp_path: Path) -> None:
    target = tmp_path / "temp" / "tetris_main.py"
    target.parent.mkdir(parents=True)
    target.write_text("class GameEngine:\n    pass\n", encoding="utf-8")

    state = _make_state()
    state.cwd = str(tmp_path)
    state.scratchpad["_phase_contract"] = {
        "version": 1,
        "active_phase": "phase_3",
        "phases": {
            "phase_3": {
                "title": "Game Loop",
                "status": "active",
                "expected_files": ["temp/tetris_main.py"],
                "required_symbols": ["GameEngine", "GamePiece.move"],
                "promotion": {"required_quality": "behavioral"},
            }
        },
    }
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "python3 -c \"import tetris_main as t; assert True\"",
        "command": "python3 -c \"import tetris_main as t; assert True\"",
        "exit_code": 0,
        "verdict": "pass",
    }

    blocked = asyncio.run(control.task_complete("Phase 3 complete", state=state, harness=None))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "phase_contract_missing_required_symbols"
    contract = blocked["metadata"]["phase_contract"]
    assert contract["missing_symbols"] == ["GamePiece.move"]


def test_task_complete_blocks_active_phase_contract_low_quality_verifier(tmp_path: Path) -> None:
    target = tmp_path / "temp" / "tetris_main.py"
    target.parent.mkdir(parents=True)
    target.write_text("class GamePiece:\n    def move(self):\n        pass\n", encoding="utf-8")

    state = _make_state()
    state.cwd = str(tmp_path)
    state.scratchpad["_phase_contract"] = {
        "version": 1,
        "active_phase": "phase_3",
        "phases": {
            "phase_3": {
                "title": "Game Loop",
                "status": "active",
                "expected_files": ["temp/tetris_main.py"],
                "required_symbols": ["GamePiece.move"],
                "checks": [
                    {
                        "id": "spawn_input_smoke",
                        "quality": "behavioral",
                        "command": "python3 -c \"import tetris_main as t; p=t.GamePiece(); assert hasattr(p, 'move'); p.move()\"",
                    }
                ],
                "promotion": {"required_quality": "behavioral"},
            }
        },
    }
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "python3 -m py_compile temp/tetris_main.py",
        "command": "python3 -m py_compile temp/tetris_main.py",
        "exit_code": 0,
        "verdict": "pass",
    }

    blocked = asyncio.run(control.task_complete("Phase 3 complete", state=state, harness=None))

    assert blocked["success"] is False
    assert blocked["metadata"]["reason"] == "phase_contract_verifier_quality_too_low"
    assert blocked["metadata"]["phase_contract"]["verifier_quality"] == {"score": 1, "label": "syntax"}
    action = blocked["metadata"]["next_required_action"]
    assert action["check_id"] == "spawn_input_smoke"
    assert "p.move()" in action["required_arguments"]["command"]


def test_task_complete_allows_active_phase_contract_behavioral_verifier(tmp_path: Path) -> None:
    target = tmp_path / "temp" / "tetris_main.py"
    target.parent.mkdir(parents=True)
    target.write_text("class GamePiece:\n    def move(self):\n        pass\n", encoding="utf-8")

    state = _make_state()
    state.cwd = str(tmp_path)
    state.scratchpad["_phase_contract"] = {
        "version": 1,
        "active_phase": "phase_3",
        "phases": {
            "phase_3": {
                "title": "Game Loop",
                "status": "active",
                "expected_files": ["temp/tetris_main.py"],
                "required_symbols": ["GamePiece.move"],
                "promotion": {"required_quality": "behavioral"},
            }
        },
    }
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "python3 -c \"import tetris_main as t; p=t.GamePiece(); assert hasattr(p, 'move'); p.move()\"",
        "command": "python3 -c \"import tetris_main as t; p=t.GamePiece(); assert hasattr(p, 'move'); p.move()\"",
        "exit_code": 0,
        "verdict": "pass",
    }

    allowed = asyncio.run(control.task_complete("Phase 3 complete", state=state, harness=None))

    assert allowed["success"] is True


def test_loop_status_exposes_phase_contract_status(tmp_path: Path) -> None:
    target = tmp_path / "temp" / "tetris_main.py"
    target.parent.mkdir(parents=True)
    target.write_text("class GameEngine:\n    pass\n", encoding="utf-8")

    state = _make_state()
    state.cwd = str(tmp_path)
    state.scratchpad["_phase_contract"] = {
        "version": 1,
        "active_phase": "phase_3",
        "phases": {
            "phase_3": {
                "title": "Game Loop",
                "status": "active",
                "expected_files": ["temp/tetris_main.py"],
                "required_symbols": ["GameEngine"],
                "checks": [
                    {
                        "id": "engine_smoke",
                        "quality": "behavioral",
                        "command": "python3 -c \"import tetris_main as t; assert t.GameEngine\"",
                    }
                ],
                "promotion": {"required_quality": "behavioral"},
            }
        },
    }

    status = asyncio.run(control.loop_status(state))

    assert status["success"] is True
    assert status["output"]["phase_contract"]["active_phase"] == "phase_3"
    assert status["output"]["phase_contract"]["status"] == "blocked"
    assert status["output"]["phase_contract"]["reason"] == "phase_contract_verifier_not_passing"
    assert status["output"]["phase_contract"]["suggested_verifier"]["id"] == "engine_smoke"


def test_loop_status_infers_phase_contract_from_spec(tmp_path: Path) -> None:
    temp = tmp_path / "temp"
    temp.mkdir(parents=True)
    (temp / "tetris-spec.md").write_text(
        "# Tetris Plan\n\n"
        "## Phase 1: Foundation\n"
        "## Phase 2: Core Mechanics\n"
        "## Phase 3: Game Loop\n"
        "Expected file: ./temp/tetris_main.py\n",
        encoding="utf-8",
    )
    (temp / "tetris_main.py").write_text("class GameEngine:\n    pass\n", encoding="utf-8")

    state = _make_state()
    state.cwd = str(tmp_path)
    state.run_brief.original_task = "read ./temp/tetris-spec.md and implement phase 3"

    status = asyncio.run(control.loop_status(state))

    contract = status["output"]["phase_contract"]
    assert contract["active_phase"] == "phase_3"
    assert contract["title"] == "Phase 3: Game Loop"
    assert contract["expected_files"] == ["temp/tetris_main.py"]
    assert contract["suggested_verifier"]["id"] == "tetris_main_behavior_smoke"
    assert "PYTHONPATH=temp" in contract["suggested_verifier"]["command"]


def test_phase_contract_update_persists_and_exposes_status(tmp_path: Path) -> None:
    target = tmp_path / "temp" / "tetris_main.py"
    target.parent.mkdir(parents=True)
    target.write_text("class GameEngine:\n    pass\n", encoding="utf-8")
    state = _make_state()
    state.cwd = str(tmp_path)
    contract = {
        "version": 1,
        "active_phase": "phase_3",
        "phases": {
            "phase_3": {
                "title": "Game Loop",
                "status": "active",
                "expected_files": ["temp/tetris_main.py"],
                "required_symbols": ["GameEngine"],
                "promotion": {"required_quality": "behavioral"},
            }
        },
    }

    result = asyncio.run(control.phase_contract_update(contract=contract, state=state, persist=True))

    assert result["success"] is True
    assert state.scratchpad["_phase_contract"]["active_phase"] == "phase_3"
    persisted = tmp_path / ".smallctl" / "phase_contract.json"
    assert persisted.exists()
    assert json.loads(persisted.read_text(encoding="utf-8"))["active_phase"] == "phase_3"
    assert result["output"]["phase_contract"]["active_phase"] == "phase_3"


def test_phase_contract_update_normalizes_model_shaped_contract(tmp_path: Path) -> None:
    target = tmp_path / "temp" / "tetris-main.py"
    test_file = tmp_path / "temp" / "test_phase3.py"
    target.parent.mkdir(parents=True)
    target.write_text("class GameState:\n    pass\n", encoding="utf-8")
    test_file.write_text("print('ok')\n", encoding="utf-8")
    state = _make_state()
    state.cwd = str(tmp_path)
    contract = {
        "version": 1,
        "active_phase": "Phase_3",
        "phases": {
            "Phase_3": {
                "name": "Tetris Game Loop with Drop Timer, Score, Level & States",
                "expected_files": ["temp/tetris-main.py", "temp/test_phase3.py"],
                "required_symbols": ["GameState"],
                "checks": ["python3 -m py_compile tetris-main.py", "python3 test_phase3.py"],
                "promotion": "all_checks_pass",
            }
        },
    }

    result = asyncio.run(control.phase_contract_update(contract=contract, state=state, persist=False))

    assert result["success"] is True
    normalized = state.scratchpad["_phase_contract"]["phases"]["Phase_3"]
    assert normalized["title"] == "Tetris Game Loop with Drop Timer, Score, Level & States"
    assert normalized["promotion"] == {"required_quality": "behavioral"}
    assert normalized["checks"][0]["quality"] == "syntax"
    assert normalized["checks"][0]["command"] == "python3 -m py_compile temp/tetris-main.py"
    assert normalized["checks"][1]["quality"] == "behavioral"
    assert result["output"]["phase_contract"]["suggested_verifier"]["command"] == "python3 temp/test_phase3.py"


def test_phase_contract_update_rejects_invalid_contract() -> None:
    state = _make_state()

    result = asyncio.run(control.phase_contract_update(contract={"active_phase": "phase_1"}, state=state, persist=False))

    assert result["success"] is False
    assert result["metadata"]["reason"] == "invalid_phase_contract"
    assert "phases" in result["error"]


def test_task_complete_surfaces_approval_required_verifier_blocker() -> None:
    state = _make_state()
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "cd /repo && timeout 3 python pong.py || true",
        "command": "cd /repo && timeout 3 python pong.py || true",
        "exit_code": None,
        "key_stdout": "",
        "key_stderr": "",
        "verdict": "needs_human",
        "failure_mode": "approval_denied",
        "approval_denied": True,
        "acceptance_delta": {"status": "pending", "notes": ["Shell execution denied by user."]},
    }

    blocked = asyncio.run(control.task_complete("done", state=state, harness=None))

    assert blocked["success"] is False
    assert "approved or rerun with approval" in blocked["error"]
    assert "Shell execution denied by user" in blocked["error"]
    assert blocked["metadata"]["approval_required"] is True


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
    assert "ast_patch" in harness.state.recent_messages[-1].content
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
    assert "file_patch" in message.content
    assert "No sections are committed yet" in message.content


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


def test_patch_existing_stage_read_count_clears_after_successful_target_patch(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "logwatch.txt"
    stage = tmp_path / ".smallctl" / "write_sessions" / "ws-1-stage.txt"
    target.write_text("original\n", encoding="utf-8")
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.write_text("patched\n", encoding="utf-8")
    state.scratchpad["_patch_existing_stage_read_autocontinue_counts"] = {f"ws-1|{target}": 1}

    session = _make_open_write_session(target, intent="patch_existing")
    session.write_staging_path = str(stage)
    state.write_session = session

    harness = SimpleNamespace(
        state=state,
        artifact_store=None,
        _runlog=lambda *args, **kwargs: None,
        log=logging.getLogger("test.plan.patch_existing_count_clear"),
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-1",
        run_mode="execute",
    )
    graph_state.last_tool_results = [
        ToolExecutionRecord(
            operation_id="op-success",
            tool_name="file_patch",
            args={
                "path": str(target),
                "target_text": "original",
                "replacement_text": "patched",
                "write_session_id": "ws-1",
            },
            tool_call_id="tool-success",
            result=ToolEnvelope(
                success=True,
                output="Patched 1 occurrence.",
                metadata={
                    "path": str(target),
                    "requested_path": str(target),
                    "write_session_id": "ws-1",
                    "staged_only": True,
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert "_patch_existing_stage_read_autocontinue_counts" not in state.scratchpad


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
    assert "requires one explicit same-target repair shape" in message.content
    assert "replace_strategy='overwrite'" in message.content
    assert "replace_strategy='append'" in message.content
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


def test_missing_first_write_session_recovered_and_original_payload_replayed(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "new_script.py"
    content = "def main():\n    return 42\n"

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(),
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
                "path": str(target),
                "content": content,
                "write_session_id": "3dde23b4",
            },
            tool_call_id="tool-1",
            result=ToolEnvelope(
                success=False,
                error="No active write session found for session ID `3dde23b4`.",
                metadata={
                    "path": str(target),
                    "write_session_id": "3dde23b4",
                    "error_kind": "missing_active_write_session",
                    "staged_only": True,
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert isinstance(state.write_session, WriteSession)
    assert state.write_session.write_session_id == "3dde23b4"
    assert state.write_session.write_target_path == str(target)
    assert graph_state.pending_tool_calls == []
    assert state.write_session.status == "complete"
    assert target.read_text(encoding="utf-8") == content
    assert any(
        event == "missing_first_write_session_recovered"
        for event, _message, _data in runlog_events
    )


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
    durable_queue = state.scratchpad.get(_DURABLE_AUTOCONTINUE_KEY)
    assert isinstance(durable_queue, list)
    assert durable_queue[-1]["recovery_kind"] == "file_patch_read_autocontinue"
    assert durable_queue[-1]["tool_name"] == "file_read"
    assert durable_queue[-1]["args"] == {"path": str(target)}
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind") == "file_patch_read_autocontinue"
        for message in state.recent_messages
    )
    assert any(event == "file_patch_read_autocontinue" for event, _message, _data in runlog_events)


def test_file_patch_mismatch_durable_autocontinue_drains_next_turn(tmp_path: Path) -> None:
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
            operation_id="op-durable",
            tool_name="file_patch",
            args={
                "path": str(target),
                "target_text": "return False",
                "replacement_text": "return True",
            },
            tool_call_id="tool-durable",
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
    assert any(event == "file_patch_read_autocontinue" for event, _message, _data in runlog_events)
    assert state.scratchpad.get(_DURABLE_AUTOCONTINUE_KEY)

    resumed_graph_state = GraphRunState(
        loop_state=LoopState.from_dict(state.to_dict()),
        thread_id="thread-1",
        run_mode="execute",
    )
    harness.state = resumed_graph_state.loop_state

    assert drain_durable_autocontinue(resumed_graph_state, harness) is True
    assert len(resumed_graph_state.pending_tool_calls) == 1
    assert resumed_graph_state.pending_tool_calls[0].tool_name == "file_read"
    assert resumed_graph_state.pending_tool_calls[0].args == {"path": str(target)}
    assert _DURABLE_AUTOCONTINUE_KEY not in harness.state.scratchpad
    assert any(event == "durable_autocontinue_recovered" for event, _message, _data in runlog_events)


def test_repeated_structural_file_patch_misses_nudge_toward_ast_patch(tmp_path: Path) -> None:
    state = _make_state()
    state.cwd = str(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("def run():\n    return False\n", encoding="utf-8")

    runlog_events: list[tuple[str, str, dict[str, object]]] = []
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda event, message, **data: runlog_events.append((event, message, data)),
        _emit=AsyncMock(),
        _failure=lambda *args, **kwargs: {"error": "guard"},
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    def _make_graph_state() -> GraphRunState:
        graph_state = GraphRunState(
            loop_state=state,
            thread_id="thread-ast-patch-recovery",
            run_mode="execute",
        )
        graph_state.last_tool_results = [
            ToolExecutionRecord(
                operation_id="op-ast-patch-recovery",
                tool_name="file_patch",
                args={
                    "path": str(target),
                    "target_text": "def run():\n    return False\n",
                    "replacement_text": "def run():\n    return True\n",
                },
                tool_call_id="tool-ast-patch-recovery",
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
        return graph_state

    first_graph_state = _make_graph_state()
    first_route = asyncio.run(apply_tool_outcomes(first_graph_state, deps))
    second_graph_state = _make_graph_state()
    second_route = asyncio.run(apply_tool_outcomes(second_graph_state, deps))

    assert first_route == "next_step"
    assert second_route == "next_step"
    assert len(second_graph_state.pending_tool_calls) == 1
    assert second_graph_state.pending_tool_calls[0].tool_name == "file_read"
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind") == "file_patch_ast_patch_nudge"
        for message in state.recent_messages
    )
    assert any(event == "file_patch_ast_patch_nudge" for event, _message, _data in runlog_events)


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


def test_task_complete_verifier_needs_human_does_not_autocontinue_with_loop_status() -> None:
    state = _make_state()
    state.last_verifier_verdict = {
        "tool": "shell_exec",
        "target": "cd /repo && timeout 3 python pong.py || true",
        "command": "cd /repo && timeout 3 python pong.py || true",
        "exit_code": None,
        "key_stdout": "",
        "key_stderr": "",
        "verdict": "needs_human",
        "failure_mode": "approval_denied",
        "approval_denied": True,
        "acceptance_delta": {"status": "pending", "notes": ["Shell execution denied by user."]},
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
            operation_id="op-approval",
            tool_name="task_complete",
            args={"message": "done"},
            tool_call_id="tool-approval",
            result=ToolEnvelope(
                success=False,
                error="Cannot complete the task until the latest verifier check is approved or rerun with approval.",
                metadata={"last_verifier_verdict": state.last_verifier_verdict, "approval_required": True},
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert graph_state.pending_tool_calls == []
    assert not any(
        getattr(message, "metadata", {}).get("recovery_kind") == "task_complete_verifier_loop_status_autocontinue"
        for message in state.recent_messages
    )
    assert not any(event == "task_complete_verifier_loop_status_autocontinue" for event, _message, _data in runlog_events)


def test_task_complete_remote_mutation_block_autocontinues_with_ssh_file_read() -> None:
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
            operation_id="op-remote-delete",
            tool_name="task_complete",
            args={"message": "done"},
            tool_call_id="tool-remote-delete",
            result=ToolEnvelope(
                success=False,
                error=(
                    "Cannot complete the task while a raw `ssh_exec` remote file deletion still needs meaningful "
                    "verification. Next required verifier: `ssh_file_read(host='192.168.1.89', user='root', "
                    "path='/var/www/demo-site')`."
                ),
                metadata={
                    "reason": "remote_mutation_requires_verification",
                    "next_required_action": {
                        "tool_names": ["ssh_file_read"],
                        "required_arguments": {
                            "host": "192.168.1.89",
                            "user": "root",
                            "path": "/var/www/demo-site",
                        },
                    },
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "ssh_file_read"
    assert pending.args == {
        "host": "192.168.1.89",
        "user": "root",
        "path": "/var/www/demo-site",
    }
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind")
        == "task_complete_remote_mutation_verifier_autocontinue"
        for message in state.recent_messages
    )
    assert any(
        event == "task_complete_remote_mutation_verifier_autocontinue"
        for event, _message, _data in runlog_events
    )


def test_task_complete_remote_glob_deletion_block_autocontinues_with_ssh_exec() -> None:
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
            operation_id="op-remote-glob-delete",
            tool_name="task_complete",
            args={"message": "done"},
            tool_call_id="tool-remote-glob-delete",
            result=ToolEnvelope(
                success=False,
                error="Cannot complete the task while a raw `ssh_exec` remote glob deletion still needs meaningful verification.",
                metadata={
                    "reason": "remote_mutation_requires_verification",
                    "next_required_action": {
                        "tool_names": ["ssh_exec"],
                        "required_arguments": {
                            "host": "192.168.1.89",
                            "user": "root",
                            "command": "find /var/www -mindepth 1 -maxdepth 1 -print -quit",
                        },
                    },
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == "next_step"
    assert len(graph_state.pending_tool_calls) == 1
    pending = graph_state.pending_tool_calls[0]
    assert pending.tool_name == "ssh_exec"
    assert pending.args == {
        "host": "192.168.1.89",
        "user": "root",
        "command": "find /var/www -mindepth 1 -maxdepth 1 -print -quit",
    }
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind")
        == "task_complete_remote_mutation_verifier_autocontinue"
        for message in state.recent_messages
    )
    assert any(
        event == "task_complete_remote_mutation_verifier_autocontinue"
        for event, _message, _data in runlog_events
    )


def test_auto_remote_mutation_verifier_result_finalizes_completion() -> None:
    state = _make_state()
    state.scratchpad["_task_complete_remote_mutation_verifier_pending_complete"] = {
        "tool_name": "ssh_file_read",
        "args": {
            "host": "192.168.1.89",
            "user": "root",
            "path": "/etc/systemd/system/docker.service",
        },
        "message": "Docker fully uninstalled.",
        "tool_call_id": "tool-complete",
        "operation_id": "op-complete",
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
            operation_id="op-verifier",
            tool_name="ssh_file_read",
            args={
                "host": "192.168.1.89",
                "user": "root",
                "path": "/etc/systemd/system/docker.service",
            },
            tool_call_id="tool-verifier",
            result=ToolEnvelope(
                success=False,
                error="Remote file not found",
                metadata={
                    "host": "192.168.1.89",
                    "path": "/etc/systemd/system/docker.service",
                    "error_kind": "file_not_found",
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == LoopRoute.FINALIZE
    assert state.scratchpad["_task_complete"] is True
    assert state.scratchpad["_task_complete_message"] == "Docker fully uninstalled."
    assert "_task_complete_remote_mutation_verifier_pending_complete" not in state.scratchpad
    assert graph_state.final_result["status"] == "completed"
    assert any(
        event == "task_complete_remote_mutation_verifier_autoaccepted"
        for event, _message, _data in runlog_events
    )


def test_auto_remote_mutation_verifier_does_not_finalize_failed_readback() -> None:
    state = _make_state()
    state.scratchpad["_task_complete_remote_mutation_verifier_pending_complete"] = {
        "tool_name": "ssh_file_read",
        "args": {
            "host": "192.168.1.89",
            "path": "/etc/systemd/system/docker.service",
        },
        "message": "Docker fully uninstalled.",
    }

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
            operation_id="op-verifier",
            tool_name="ssh_file_read",
            args={
                "host": "192.168.1.89",
                "path": "/etc/systemd/system/docker.service",
            },
            tool_call_id="tool-verifier",
            result=ToolEnvelope(
                success=False,
                error="Permission denied",
                metadata={
                    "host": "192.168.1.89",
                    "path": "/etc/systemd/system/docker.service",
                    "error_kind": "permission_denied",
                },
            ),
        )
    ]
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(apply_tool_outcomes(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert "_task_complete" not in state.scratchpad
    assert "_task_complete_remote_mutation_verifier_pending_complete" in state.scratchpad


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


def test_repair_stall_shell_failure_autocontinues_with_loop_status() -> None:
    state = _make_state()
    state.stagnation_counters["repeat_patch"] = 2
    state.last_failure_class = "test"

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
            operation_id="op-shell",
            tool_name="shell_exec",
            args={"command": "python3 ./temp/circuit_breaker.py -v"},
            tool_call_id="tool-shell",
            result=ToolEnvelope(
                success=False,
                error="FAILED (failures=1, errors=4)",
                metadata={},
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
    assert pending.source == "system"
    assert any(
        getattr(message, "metadata", {}).get("recovery_kind") == "repair_stall_loop_status_autocontinue"
        for message in state.recent_messages
    )
    assert any(event == "repair_stall_loop_status_autocontinue" for event, _message, _data in runlog_events)


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


def test_interpret_chat_output_nudges_on_action_like_prose_without_tool_call() -> None:
    state = _make_state()
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    graph_state = GraphRunState(loop_state=state, thread_id="thread-chat-action", run_mode="chat")
    graph_state.last_assistant_text = "I'll gather this information with a shell command next."
    graph_state.last_thinking_text = "I will execute a shell command to inspect the remote host."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_chat_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert graph_state.final_result is None
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["recovery_kind"] == "chat_action_stall"
    assert "do not repeat the analysis" in state.recent_messages[-1].content.lower()


def test_interpret_model_output_tags_hello_completion_mission_check_as_recovery_message() -> None:
    state = _make_state()
    state.run_brief.original_task = "Run docker ps on the remote host and list running containers."
    harness = SimpleNamespace(
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
    graph_state = GraphRunState(loop_state=state, thread_id="thread-hello-mission-check", run_mode="loop")
    graph_state.last_assistant_text = "Hello complete, the greeting task is finished."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["is_recovery_nudge"] is True
    assert state.recent_messages[-1].metadata["recovery_kind"] == "mission_check_hello_completion"


def test_interpret_planning_output_tags_plan_set_nudge_as_recovery_message() -> None:
    state = _make_state()
    state.planning_mode_enabled = True
    harness = SimpleNamespace(
        state=state,
        _emit=AsyncMock(),
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-planning-nudge",
        run_mode="planning",
    )
    graph_state.last_assistant_text = "I need a bit more structure before execution."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_planning_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert state.recent_messages
    assert state.recent_messages[-1].metadata["is_recovery_nudge"] is True
    assert state.recent_messages[-1].metadata["recovery_kind"] == "planning_mode_requires_plan_set"
    assert state.recent_messages[-1].metadata["planner_nudge"] is True


def test_interpret_chat_output_backfills_missing_write_session_metadata_for_active_target() -> None:
    state = _make_state()
    target = Path("temp/lease_scheduler.py")
    session = _make_open_write_session(target, session_id="ws-active")
    session.write_next_section = "imports"
    state.write_session = session

    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-chat-write-session-repair",
        run_mode="chat",
        pending_tool_calls=[
            PendingToolCall(
                tool_name="file_write",
                args={
                    "path": str(target),
                    "content": "print('saved')\n",
                    "section_name": "imports",
                    "next_section_name": "tests",
                },
            )
        ],
    )
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_chat_output(graph_state, deps))

    assert route == LoopRoute.DISPATCH_TOOLS
    pending = graph_state.pending_tool_calls[0]
    # Path-based implicit resolution means write_session_id is no longer
    # backfilled; the tool layer matches by target path.
    assert "write_session_id" not in pending.args
    assert pending.args["section_name"] == "imports"
    assert pending.args["next_section_name"] == "tests"


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


def test_chat_runtime_interpret_edge_can_continue_after_no_tool_nudge() -> None:
    router_name, targets = ChatGraphRuntime.GRAPH_SPEC.edge_map["interpret_chat_output"]

    assert router_name == "_route_after_interpret"
    assert targets["prepare_step"] == "prepare_step"


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
