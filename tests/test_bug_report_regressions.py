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
from smallctl.tools.http import file_download
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


def test_grep_searches_file_path_directly(tmp_path: Path) -> None:
    target = tmp_path / "chronoshift-labyrinth.html"
    target.write_text("<canvas id='game'></canvas>\n<script>function startGame() {}</script>\n", encoding="utf-8")

    result = asyncio.run(grep("canvas", path=str(target), case_sensitive=True))

    assert result["success"] is True
    assert result["metadata"]["count"] == 1
    assert result["output"] == [{"path": str(target), "line": 1, "text": "<canvas id='game'></canvas>"}]


def test_find_files_matches_file_path_directly(tmp_path: Path) -> None:
    target = tmp_path / "chronoshift-labyrinth.html"
    target.write_text("content\n", encoding="utf-8")

    result = asyncio.run(find_files("chronoshift", path=str(target)))

    assert result["success"] is True
    assert result["metadata"]["count"] == 1
    assert result["output"] == [str(target)]


def test_tool_plan_allow_git_env_config_is_normalized(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_TOOL_PLAN_ALLOW_GIT", "true")

    config = resolve_config({})

    assert config.tool_plan_allow_git is True


def test_solver_refine_env_config_is_normalized(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_SOLVER_REFINE_ENABLED", "true")
    monkeypatch.setenv("SMALLCTL_SOLVER_REFINE_MAX_PASSES", "3")
    monkeypatch.setenv("SMALLCTL_SOLVER_REFINE_TOKEN_BUDGET", "900")

    config = resolve_config({})

    assert config.solver_refine_enabled is True
    assert config.solver_refine_max_passes == 3
    assert config.solver_refine_token_budget == 900


def test_runtime_policy_env_config_is_normalized(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SMALLCTL_NEEDS_HUMAN_TIMEOUT_SEC", "42")
    monkeypatch.setenv("SMALLCTL_CHUNK_MODE_NEW_FILE_ONLY", "false")
    monkeypatch.setenv("SMALLCTL_CHUNK_MODE_SUPPORTED_MODELS", "qwen3.5,test-model")

    config = resolve_config({})

    assert config.needs_human_timeout_sec == 42
    assert config.chunk_mode_new_file_only is False
    assert config.chunk_mode_supported_models == ["qwen3.5", "test-model"]


def test_file_download_rejects_workspace_escape_before_network(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))

    result = asyncio.run(
        file_download(
            url="http://127.0.0.1:9/should-not-fetch",
            output_path="../outside.txt",
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "workspace_path_traversal"
    assert not (tmp_path.parent / "outside.txt").exists()


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
