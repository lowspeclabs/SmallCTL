from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from smallctl.config import resolve_config
from smallctl.graph.interrupts import build_interrupt_payload
from smallctl.graph.state import GraphRunState, ToolExecutionRecord
from smallctl.graph.write_session_outcomes import _run_syntax_check
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.fs_mutations import file_delete
from smallctl.tools.search import find_files, grep
from smallctl.write_session_fsm import new_write_session


def test_file_delete_removes_target_when_aborting_active_write_session(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    stage = tmp_path / ".smallctl" / "write_sessions" / "stage.py"
    target.write_text("print('old')\n", encoding="utf-8")
    stage.parent.mkdir(parents=True)
    stage.write_text("print('staged')\n", encoding="utf-8")

    state = LoopState(cwd=str(tmp_path))
    state.write_session = new_write_session(
        session_id="ws-delete",
        target_path=str(target),
        intent="replace_file",
    )
    state.write_session.write_staging_path = str(stage)

    result = asyncio.run(file_delete(str(target), state=state))

    assert result["success"] is True
    assert result["output"] == "deleted"
    assert result["metadata"]["reason"] == "active_write_session_aborted_by_delete"
    assert not target.exists()
    assert not stage.exists()
    assert state.write_session is None


def test_build_interrupt_payload_includes_creation_timestamp() -> None:
    state = LoopState(cwd="/tmp/project")
    harness = SimpleNamespace(state=state)
    graph_state = GraphRunState(loop_state=state, thread_id="thread-1", run_mode="auto")
    record = ToolExecutionRecord(
        operation_id="op-1",
        tool_name="ask_human",
        args={},
        tool_call_id="call-1",
        result=ToolEnvelope(success=False, output={"question": "Continue?"}),
    )

    payload = build_interrupt_payload(
        harness=harness,
        graph_state=graph_state,
        record=record,
    )

    assert payload["kind"] == "ask_human"
    assert payload["question"] == "Continue?"
    assert isinstance(payload["created_at"], float)


def test_search_tools_return_failure_for_invalid_regex(tmp_path: Path) -> None:
    (tmp_path / "example.txt").write_text("content\n", encoding="utf-8")

    grep_result = asyncio.run(grep("[", path=str(tmp_path), regex=True))
    find_result = asyncio.run(find_files("[", path=str(tmp_path), regex=True))

    assert grep_result["success"] is False
    assert grep_result["metadata"]["error_kind"] == "invalid_regex"
    assert find_result["success"] is False
    assert find_result["metadata"]["error_kind"] == "invalid_regex"


def test_tool_plan_allow_git_env_config_is_normalized(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_TOOL_PLAN_ALLOW_GIT", "true")

    config = resolve_config({})

    assert config.tool_plan_allow_git is True


def test_write_session_syntax_check_does_not_execute_shell_metacharacters(tmp_path: Path) -> None:
    target = tmp_path / "safe; touch owned.py"
    target.write_text("value = 1\n", encoding="utf-8")
    harness = SimpleNamespace(
        state=SimpleNamespace(cwd=str(tmp_path)),
        log=SimpleNamespace(error=lambda *_args, **_kwargs: None),
        _runlog=lambda *_args, **_kwargs: None,
    )

    result = asyncio.run(_run_syntax_check(harness, str(target)))

    assert result is not None
    assert result["verdict"] == "pass"
    assert not (tmp_path / "owned.py").exists()


def test_write_session_json_syntax_check_reports_parse_failure(tmp_path: Path) -> None:
    target = tmp_path / "broken.json"
    target.write_text("{", encoding="utf-8")
    harness = SimpleNamespace(
        state=SimpleNamespace(cwd=str(tmp_path)),
        log=SimpleNamespace(error=lambda *_args, **_kwargs: None),
        _runlog=lambda *_args, **_kwargs: None,
    )

    result = asyncio.run(_run_syntax_check(harness, str(target)))

    assert result is not None
    assert result["verdict"] == "fail"
    assert result["exit_code"] == 1
