from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.nodes import _apply_small_model_authoring_budget, dispatch_tools
from smallctl.graph.tool_call_parser import _detect_repeated_tool_loop
from smallctl.graph.state import GraphRunState, PendingToolCall
from smallctl.harness.tool_results import _store_verifier_verdict
from smallctl.prompts import build_system_prompt
from smallctl.state import ArtifactRecord, ExecutionPlan, LoopState, PlanStep
from smallctl.graph.tool_outcomes import _maybe_emit_repair_recovery_nudge, apply_tool_outcomes
from smallctl.graph.tool_outcomes import _shell_workspace_relative_retry_hint
from smallctl.graph.state import ToolExecutionRecord
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
            steps=[
                {"step_id": "P1", "title": "Create file skeleton"},
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
    assert result["output"]["artifact_id"] == state.plan_artifact_id


def test_system_prompt_surfaces_playbook_guidance() -> None:
    state = _make_state()
    state.run_brief.original_task = "Write a script"
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="Write a script",
        status="draft",
        steps=[PlanStep(step_id="P1", title="Create file skeleton")],
    )
    state.plan_resolved = True
    state.plan_artifact_id = "A0007"

    prompt = build_system_prompt(state, "execute")

    assert "PLAN PLAYBOOK" in prompt
    assert "A0007" in prompt
    assert "file skeleton" in prompt
    assert "acceptance criteria" in prompt.lower()


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

    blocked = asyncio.run(fs.file_write(path=str(target), content="patched\n", cwd=str(tmp_path), state=state))
    assert blocked["success"] is False
    assert "reading the target file before patching" in blocked["error"]

    read_back = asyncio.run(fs.file_read(path=str(target), cwd=str(tmp_path), state=state))
    assert read_back["success"] is True

    allowed = asyncio.run(fs.file_write(path=str(target), content="patched\n", cwd=str(tmp_path), state=state))
    assert allowed["success"] is True
    assert str(target.resolve()) in state.files_changed_this_cycle


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


def test_repeated_shell_exec_command_still_trips_the_loop_guard() -> None:
    state = _make_state()
    harness = SimpleNamespace(state=state)
    fingerprint = json.dumps(
        {"tool_name": "shell_exec", "args": {"command": "python -V"}},
        sort_keys=True,
    )
    state.scratchpad["_tool_attempt_history"] = [
        {"tool_name": "shell_exec", "fingerprint": fingerprint},
        {"tool_name": "shell_exec", "fingerprint": fingerprint},
    ]

    pending = PendingToolCall(tool_name="shell_exec", args={"command": "python -V"})

    assert _detect_repeated_tool_loop(harness, pending) == (
        "Guard tripped: repeated tool call loop (shell_exec repeated with identical arguments)"
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
