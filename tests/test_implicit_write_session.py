from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from smallctl.graph.state import GraphRunState, ToolExecutionRecord
from smallctl.graph.tool_execution_recovery import handle_failed_file_write_outcome
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState
from smallctl.tools.fs import file_write, promote_write_session_target
from smallctl.write_session_fsm import new_write_session


def _big_markdown() -> str:
    body = "\n\n".join(
        f"## Section {idx}\n\nParagraph of documentation for section {idx}. " * 4
        for idx in range(1, 40)
    )
    return f"# Usage Guide\n\n## Overview\n\n{body}\n"


def test_one_shot_overwrite_with_section_name_writes_target_directly(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "bookstack-python" / "agents.md"
    content = _big_markdown()

    result = asyncio.run(
        file_write(
            "bookstack-python/agents.md",
            content,
            cwd=str(tmp_path),
            state=state,
            section_name="agents.md content",
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True
    assert target.exists()
    assert target.read_text(encoding="utf-8") == content
    assert state.write_session is None
    assert result["metadata"].get("write_session_id") is None
    events = state.scratchpad.get("_write_session_events", [])
    assert not any(e["event"] == "implicit_session_created" for e in events)


def test_one_shot_overwrite_env_example_style_file_writes_target_directly(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "bookstack-python" / ".env.example"
    content = "\n".join(f"VAR_{idx}=value_{idx}" for idx in range(1, 120)) + "\n"

    result = asyncio.run(
        file_write(
            "bookstack-python/.env.example",
            content,
            cwd=str(tmp_path),
            state=state,
            section_name=".env.example content",
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True
    assert target.exists()
    assert target.read_text(encoding="utf-8") == content
    assert state.write_session is None


def test_overwrite_with_declared_next_section_still_creates_implicit_session(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "app.py"

    result = asyncio.run(
        file_write(
            str(target),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="main_logic",
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True
    assert state.write_session is not None
    assert result["metadata"]["write_session_id"] == state.write_session.write_session_id
    assert result["metadata"]["write_next_section"] == "main_logic"
    assert not target.exists()


def test_ungrounded_overwrite_section_does_not_infer_phantom_next_and_finalizes(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "agents.md"
    session = new_write_session(
        session_id="ws-doc",
        target_path=str(target),
        intent="replace_file",
    )
    session.suggested_sections = ["header", "overview", "details", "footer"]
    state.write_session = session
    content = _big_markdown()

    result = asyncio.run(
        file_write(
            str(target),
            content,
            cwd=str(tmp_path),
            state=state,
            section_name="agents.md content",
            replace_strategy="overwrite",
        )
    )

    assert result["success"] is True
    assert result["metadata"]["write_next_section"] == ""
    assert result["metadata"]["write_next_section_inferred"] is False
    assert result["metadata"]["write_session_final_chunk"] is True
    assert "Next section inferred" not in result["output"]
    assert state.write_session.write_next_section == ""

    promoted, _detail = promote_write_session_target(state.write_session, cwd=str(tmp_path))
    assert promoted is True
    assert target.read_text(encoding="utf-8") == content


def test_ungrounded_append_section_stays_open_without_phantom_next(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "agents.md"
    session = new_write_session(
        session_id="ws-doc",
        target_path=str(target),
        intent="replace_file",
    )
    session.suggested_sections = ["header", "overview", "details", "footer"]
    state.write_session = session

    result = asyncio.run(
        file_write(
            str(target),
            _big_markdown(),
            cwd=str(tmp_path),
            state=state,
            section_name="agents.md content",
        )
    )

    assert result["success"] is True
    assert result["metadata"]["write_next_section"] == ""
    assert result["metadata"]["write_next_section_inferred"] is False
    assert result["metadata"]["write_session_final_chunk"] is False
    assert "Next section inferred" not in result["output"]
    assert result["metadata"]["next_legal_operation"] == "finalize"
    assert not target.exists()


def test_grounded_section_still_infers_next_section(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="ws-grounded",
        target_path=str(target),
        intent="replace_file",
    )
    session.suggested_sections = ["imports", "main_logic"]
    state.write_session = session

    result = asyncio.run(
        file_write(
            str(target),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
        )
    )

    assert result["success"] is True
    assert result["metadata"]["write_next_section"] == "main_logic"
    assert result["metadata"]["write_next_section_inferred"] is True
    assert result["metadata"]["write_session_final_chunk"] is False


def test_chunked_success_message_reports_staged_not_written(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "app.py"

    result = asyncio.run(
        file_write(
            str(target),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="imports",
            next_section_name="main_logic",
        )
    )

    assert result["success"] is True
    assert "staged; not yet written to" in result["output"]
    assert f"Section `imports` written to `{target}`" not in result["output"]
    assert result["metadata"]["staged_only"] is True
    assert not target.exists()


def test_stalled_session_message_points_at_finalize(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="ws-stall",
        target_path=str(target),
        intent="replace_file",
    )
    session.suggested_sections = ["imports", "main_logic"]
    state.write_session = session

    result = asyncio.run(
        file_write(
            str(target),
            "import os\n",
            cwd=str(tmp_path),
            state=state,
            section_name="random_block",
        )
    )

    assert result["success"] is True
    assert "staged; not yet written to" in result["output"]
    assert "finalize_write_session" in result["output"]


def test_file_write_without_replace_strategy_still_rejected_by_schema_gate(tmp_path: Path) -> None:
    from smallctl.tools.register_filesystem import _file_write_with_strategy_gate

    state = LoopState(cwd=str(tmp_path))
    result = asyncio.run(
        _file_write_with_strategy_gate(
            path=str(tmp_path / "plain.py"),
            content="print('hi')\n",
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert "file_write requires `replace_strategy`" in result["error"]


def _overlap_record(target: Path, content: str) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        operation_id="op-overlap",
        tool_name="file_write",
        args={"path": str(target), "content": content},
        tool_call_id="tool-overlap",
        result=ToolEnvelope(
            success=False,
            error="LoopGuard: overlapping chunk append detected.",
            metadata={
                "path": str(target),
                "error_kind": "chunked_write_append_overlap_detected",
                "loop_guard_schedule_read": True,
                "loop_guard_score": 3,
                "loop_guard_escalation_level": 1,
                "section_name": "header",
            },
        ),
    )


def _recovery_harness(state: LoopState) -> SimpleNamespace:
    async def _emit(*args, **kwargs) -> None:
        del args, kwargs

    return SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
        _emit=_emit,
    )


def _stage_content(state: LoopState, target: Path, content: str, session_id: str = "ws-overlap") -> None:
    session = new_write_session(
        session_id=session_id,
        target_path=str(target),
        intent="replace_file",
    )
    staging = target.parent / ".smallctl" / "write_sessions" / f"{session_id}__{target.stem}__stage{target.suffix}"
    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.write_text(content, encoding="utf-8")
    session.write_staging_path = str(staging)
    state.write_session = session


def test_loop_guard_overlap_nudge_says_already_staged_when_content_matches(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "agents.md"
    content = _big_markdown()
    _stage_content(state, target, content)
    harness = _recovery_harness(state)
    deps = SimpleNamespace(event_handler=None)
    graph_state = GraphRunState(loop_state=state, thread_id="thread-overlap", run_mode="execute")

    asyncio.run(
        handle_failed_file_write_outcome(
            graph_state=graph_state,
            harness=harness,
            deps=deps,
            record=_overlap_record(target, content),
        )
    )

    assert graph_state.pending_tool_calls
    assert graph_state.pending_tool_calls[0].tool_name == "file_read"
    nudges = [
        message
        for message in state.recent_messages
        if message.metadata.get("recovery_kind") == "chunked_write_loop_guard_already_staged"
    ]
    assert len(nudges) == 1
    assert "already present in the staged copy" in nudges[0].content
    assert "finalize_write_session" in nudges[0].content
    assert "`header`" not in nudges[0].content


def test_loop_guard_overlap_nudge_skipped_when_content_not_staged(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "agents.md"
    _stage_content(state, target, "completely different staged body\n")
    harness = _recovery_harness(state)
    deps = SimpleNamespace(event_handler=None)
    graph_state = GraphRunState(loop_state=state, thread_id="thread-overlap-2", run_mode="execute")

    asyncio.run(
        handle_failed_file_write_outcome(
            graph_state=graph_state,
            harness=harness,
            deps=deps,
            record=_overlap_record(target, _big_markdown()),
        )
    )

    assert graph_state.pending_tool_calls
    assert graph_state.pending_tool_calls[0].tool_name == "file_read"
    nudges = [
        message
        for message in state.recent_messages
        if message.metadata.get("recovery_kind") == "chunked_write_loop_guard_already_staged"
    ]
    assert nudges == []
