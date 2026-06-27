from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from smallctl.context.artifacts import ArtifactStore
from smallctl.context.policy import ContextPolicy
from smallctl.harness.task_intent import extract_intent_state
from smallctl.harness.tool_dispatch_cache import maybe_reuse_identical_read_call
from smallctl.harness.tool_result_flow import record_result
from smallctl.harness.tool_visibility import (
    _filter_artifact_read_for_run,
    _filter_read_only_loop_tools,
)
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState
from smallctl.tools.artifact import artifact_grep, artifact_read
from smallctl.tools.fs_listing import file_read


def _make_service(tmp_path: Path, *, inline_limit: int = 325) -> SimpleNamespace:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-test"
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path / "artifacts", "run-test", session_id=state.thread_id),
        context_policy=ContextPolicy(
            tool_result_inline_token_limit=inline_limit,
            artifact_summarization_threshold=999999,
        ),
        summarizer_client=None,
        summarizer=None,
        client=SimpleNamespace(model="qwen3.5:4b"),
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: "inspect a local file",
    )
    return SimpleNamespace(harness=harness)


def test_grep_small_result_not_persisted(tmp_path: Path) -> None:
    service = _make_service(tmp_path)

    message = asyncio.run(
        record_result(
            service,
            tool_name="grep",
            tool_call_id="call-1",
            result=ToolEnvelope(
                success=True,
                output=[{"path": "foo.py", "line": 1, "text": "def main():"}],
                metadata={"path": "foo.py"},
            ),
            arguments={"pattern": "def main", "path": "foo.py"},
        )
    )

    assert service.harness.state.artifacts == {}
    assert "artifact_id" not in message.metadata


def test_artifact_grep_small_result_not_persisted(tmp_path: Path) -> None:
    service = _make_service(tmp_path)

    message = asyncio.run(
        record_result(
            service,
            tool_name="artifact_grep",
            tool_call_id="call-1",
            result=ToolEnvelope(
                success=True,
                output="Found 1 matches in A0001:\nL1: def main():",
                metadata={"artifact_id": "A0001", "query": "def main"},
            ),
            arguments={"artifact_id": "A0001", "query": "def main"},
        )
    )

    assert service.harness.state.artifacts == {}
    assert "artifact_id" not in message.metadata


def test_grep_large_result_persisted(tmp_path: Path) -> None:
    service = _make_service(tmp_path, inline_limit=50)
    large_output = [{"path": "foo.py", "line": i, "text": f"line {i}"} for i in range(100)]

    message = asyncio.run(
        record_result(
            service,
            tool_name="grep",
            tool_call_id="call-1",
            result=ToolEnvelope(
                success=True,
                output=large_output,
                metadata={"path": "foo.py"},
            ),
            arguments={"pattern": "line", "path": "foo.py"},
        )
    )

    assert len(service.harness.state.artifacts) == 1
    assert message.metadata["artifact_id"] == "A0001"


def test_artifact_grep_blocked_on_grep_artifact(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    content_path = tmp_path / "A0001.txt"
    content_path.write_text("L1: def start_game():\n", encoding="utf-8")
    state.artifacts["A0001"] = ArtifactRecord(
        artifact_id="A0001",
        kind="grep",
        source="foo.py",
        created_at="2026-06-26T00:00:00+00:00",
        size_bytes=32,
        summary="grep results",
        tool_name="grep",
        content_path=str(content_path),
    )

    result = artifact_grep(state, artifact_id="A0001", query="def game_loop")

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "artifact_kind_mismatch"
    assert "search-result artifact" in result["error"]


def test_artifact_grep_allowed_on_file_read_artifact(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    content_path = tmp_path / "A0001.txt"
    content_path.write_text("def start_game():\ndef game_loop():\n", encoding="utf-8")
    state.artifacts["A0001"] = ArtifactRecord(
        artifact_id="A0001",
        kind="file_read",
        source="foo.py",
        created_at="2026-06-26T00:00:00+00:00",
        size_bytes=42,
        summary="file_read results",
        tool_name="file_read",
        content_path=str(content_path),
    )

    result = artifact_grep(state, artifact_id="A0001", query="def game_loop")

    assert result["success"] is True
    assert "Found 1 matches" in result["output"]


def test_artifact_grep_deduplicated(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.tool_execution_records = {
        "rec-1": {
            "tool_name": "artifact_grep",
            "args": {"artifact_id": "A0001", "query": "start_game"},
            "result": {
                "success": True,
                "output": "Found 1 matches",
                "metadata": {"artifact_id": "A0001"},
            },
        }
    }
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)

    cached = maybe_reuse_identical_read_call(
        harness,
        tool_name="artifact_grep",
        args={"artifact_id": "A0001", "query": "start_game"},
    )

    assert cached is not None
    assert cached.success is True


def test_fix_html_task_promotes_to_requested_file_patch() -> None:
    harness = SimpleNamespace(
        provider_profile="lmstudio",
        state=SimpleNamespace(
            current_phase="explore",
            cwd="/tmp",
            working_memory=SimpleNamespace(failures=[], next_actions=[]),
        ),
        _looks_like_shell_request=lambda task: False,
    )

    primary, secondary, tags = extract_intent_state(
        harness, "read and fix the bug in temp/chronoshift-labyrinth.html"
    )

    assert primary == "requested_file_patch"
    assert "file_patch" in tags


def test_fix_py_task_promotes_to_requested_file_patch() -> None:
    harness = SimpleNamespace(
        provider_profile="lmstudio",
        state=SimpleNamespace(
            current_phase="explore",
            cwd="/tmp",
            working_memory=SimpleNamespace(failures=[], next_actions=[]),
        ),
        _looks_like_shell_request=lambda task: False,
    )

    primary, secondary, tags = extract_intent_state(
        harness, "patch the error handling in ./app.py"
    )

    assert primary == "requested_file_patch"


def test_read_only_loop_gate_hides_read_tools() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "fix the bug in temp/chronoshift-labyrinth.html"
    state.scratchpad["_read_only_loop_gate_active"] = True
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    schemas = [
        {"type": "function", "function": {"name": "file_read"}},
        {"type": "function", "function": {"name": "grep"}},
        {"type": "function", "function": {"name": "artifact_grep"}},
        {"type": "function", "function": {"name": "artifact_read"}},
        {"type": "function", "function": {"name": "file_patch"}},
        {"type": "function", "function": {"name": "shell_exec"}},
        {"type": "function", "function": {"name": "ask_human"}},
        {"type": "function", "function": {"name": "task_fail"}},
    ]

    filtered = _filter_read_only_loop_tools(harness, schemas, mode="loop")
    names = {s["function"]["name"] for s in filtered}

    assert "file_read" not in names
    assert "grep" not in names
    assert "artifact_grep" not in names
    assert "artifact_read" not in names
    assert "file_patch" in names
    assert "shell_exec" in names
    assert "ask_human" in names
    assert "task_fail" in names


def test_read_only_loop_gate_disabled_after_mutation() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "fix the bug in temp/chronoshift-labyrinth.html"
    state.files_changed_this_cycle = ["temp/chronoshift-labyrinth.html"]
    state.scratchpad["_read_only_loop_gate_active"] = True
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )
    schemas = [
        {"type": "function", "function": {"name": "file_read"}},
        {"type": "function", "function": {"name": "file_patch"}},
    ]

    filtered = _filter_read_only_loop_tools(harness, schemas, mode="loop")
    names = {s["function"]["name"] for s in filtered}

    assert "file_read" in names
    assert "file_patch" in names


def test_progress_guard_activates_read_only_loop_gate() -> None:
    from smallctl.graph.progress_guard import _update_progress_tracking
    from smallctl.graph.state import ToolExecutionRecord

    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "fix the blackscreen in temp/chronoshift-labyrinth.html"
    harness = SimpleNamespace(
        state=state,
        _runlog=lambda *args, **kwargs: None,
    )

    for i in range(6):
        record = ToolExecutionRecord(
            operation_id=f"op-{i}",
            tool_name="file_read",
            args={"path": f"temp/file{i}.html"},
            tool_call_id=None,
            result=ToolEnvelope(
                success=True,
                metadata={"path": f"temp/file{i}.html", "line_start": 1, "line_end": 10},
            ),
        )
        _update_progress_tracking(
            harness,
            SimpleNamespace(last_tool_results=[record], last_assistant_text="", last_thinking_text=""),
        )

    schemas = [
        {"type": "function", "function": {"name": "file_read"}},
        {"type": "function", "function": {"name": "grep"}},
        {"type": "function", "function": {"name": "artifact_read"}},
        {"type": "function", "function": {"name": "file_patch"}},
        {"type": "function", "function": {"name": "ask_human"}},
        {"type": "function", "function": {"name": "task_fail"}},
    ]

    filtered = _filter_read_only_loop_tools(harness, schemas, mode="loop")
    names = {s["function"]["name"] for s in filtered}

    assert "file_read" not in names
    assert "grep" not in names
    assert "artifact_read" not in names
    assert "file_patch" in names
    assert "ask_human" in names
    assert "task_fail" in names


def test_large_context_exposes_artifact_grep_and_print() -> None:
    state = LoopState(cwd="/tmp")
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(allow_artifact_read_large_context=False),
        server_context_limit=1_048_576,
        client=SimpleNamespace(model="deepseek-v4-flash", runtime_context_limit=None, context_limit=None),
        _runlog=lambda *args, **kwargs: None,
    )
    schemas = [
        {"type": "function", "function": {"name": "artifact_read"}},
        {"type": "function", "function": {"name": "artifact_grep"}},
        {"type": "function", "function": {"name": "artifact_print"}},
        {"type": "function", "function": {"name": "file_read"}},
    ]

    filtered = _filter_artifact_read_for_run(harness, schemas, mode="loop")
    names = {s["function"]["name"] for s in filtered}

    assert "artifact_read" not in names
    assert "artifact_grep" in names
    assert "artifact_print" in names
    assert "file_read" in names


def test_large_context_exposes_artifact_read_for_mutation_task() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "fix the bug in temp/chronoshift-labyrinth.html"
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(allow_artifact_read_large_context=False),
        server_context_limit=1_048_576,
        client=SimpleNamespace(model="deepseek-v4-flash", runtime_context_limit=None, context_limit=None),
        _runlog=lambda *args, **kwargs: None,
    )
    schemas = [
        {"type": "function", "function": {"name": "artifact_read"}},
        {"type": "function", "function": {"name": "artifact_grep"}},
        {"type": "function", "function": {"name": "artifact_print"}},
        {"type": "function", "function": {"name": "file_read"}},
    ]

    filtered = _filter_artifact_read_for_run(harness, schemas, mode="loop")
    names = {s["function"]["name"] for s in filtered}

    assert "artifact_read" in names
    assert "artifact_grep" in names
    assert "artifact_print" in names
    assert "file_read" in names


def test_remote_html_task_classifies_as_remote_execute() -> None:
    from smallctl.harness.task_classifier import classify_task_mode

    assert classify_task_mode(
        "ssh to 192.168.1.161 and fix /var/www/index.html"
    ) == "remote_execute"
    assert classify_task_mode(
        "patch style.css on the remote host 10.0.0.5"
    ) == "remote_execute"


def test_local_html_task_gets_local_mutation_anchor() -> None:
    from smallctl.context.frame_run_rendering import coding_anchor_lines

    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "fix ./temp/chronoshift-labyrinth.html blackscreen"
    state.current_phase = "execute"
    state.active_intent = "requested_file_patch"
    state.task_mode = "local_execute"
    state.scratchpad = {}

    anchors = coding_anchor_lines(state)
    assert "mode=local_mutation" in anchors
    assert "directive=read_once_then_patch" in anchors


def test_remote_html_task_does_not_get_local_mutation_anchor() -> None:
    from smallctl.context.frame_run_rendering import coding_anchor_lines

    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "ssh to 192.168.1.161 and fix /var/www/index.html"
    state.current_phase = "execute"
    state.active_intent = "requested_file_patch"
    state.task_mode = "remote_execute"
    state.scratchpad = {}

    anchors = coding_anchor_lines(state)
    assert "mode=local_mutation" not in anchors
    assert "directive=read_once_then_patch" not in anchors


def test_file_read_reversed_range_normalized(tmp_path: Path) -> None:
    path = tmp_path / "sample.html"
    path.write_text("line one\nline two\nline three\n", encoding="utf-8")

    result = asyncio.run(file_read(str(path), cwd=str(tmp_path), start_line=3, end_line=1))

    assert result["success"] is True
    assert result["output"] == "line one\nline two\nline three"
    assert result["metadata"]["line_start"] == 1
    assert result["metadata"]["line_end"] == 3


def test_artifact_read_reversed_range_normalized(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    content_path = tmp_path / "A0001.txt"
    content_path.write_text("line one\nline two\nline three\n", encoding="utf-8")
    state.artifacts["A0001"] = ArtifactRecord(
        artifact_id="A0001",
        kind="file_read",
        source="sample.html",
        created_at="2026-06-26T00:00:00+00:00",
        size_bytes=30,
        summary="sample file",
        tool_name="file_read",
        content_path=str(content_path),
    )

    result = artifact_read(state, artifact_id="A0001", start_line=3, end_line=1)

    assert result["success"] is True
    assert result["metadata"]["line_start"] == 1
    assert result["metadata"]["line_end"] == 3
