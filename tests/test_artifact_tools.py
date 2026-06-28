from __future__ import annotations

from pathlib import Path
import asyncio
import json
from types import SimpleNamespace

from smallctl.context.artifacts import ArtifactStore
from smallctl.context.assembler import PromptAssembler
from smallctl.context.messages import format_compact_tool_message
from smallctl.context.policy import ContextPolicy
from smallctl.graph.state import (
    GraphRunState,
    ToolExecutionRecord,
    inflate_graph_state,
    serialize_graph_state,
)
from smallctl.harness.tool_result_flow import record_result
from smallctl.harness.tool_results import ToolResultService
from smallctl.models.conversation import ConversationMessage
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, LoopState
from smallctl.tools.artifact import artifact_grep, artifact_read


def _state_with_artifact(tmp_path: Path, *, content: str) -> LoopState:
    state = LoopState(cwd=str(tmp_path))
    content_path = tmp_path / "A0001.txt"
    content_path.write_text(content, encoding="utf-8")
    state.artifacts["A0001"] = ArtifactRecord(
        artifact_id="A0001",
        kind="shell_exec",
        source="nmap -sn 192.168.1.0/24",
        created_at="2026-04-14T22:48:21+00:00",
        size_bytes=len(content.encode("utf-8")),
        summary="nmap host discovery",
        tool_name="shell_exec",
        content_path=str(content_path),
    )
    return state


def test_artifact_read_accepts_evidence_style_artifact_id(tmp_path: Path) -> None:
    state = _state_with_artifact(tmp_path, content="alpha\nbeta\n")

    result = artifact_read(state, artifact_id="E-A0001")

    assert result["success"] is True
    assert result["output"] == "alpha\nbeta"
    assert result["metadata"]["artifact_id"] == "A0001"
    assert result["metadata"]["source_artifact_id"] == "A0001"


def test_artifact_grep_accepts_evidence_style_artifact_id(tmp_path: Path) -> None:
    state = _state_with_artifact(tmp_path, content="host one up\nhost two down\n")

    result = artifact_grep(state, artifact_id="E-A0001", query="down")

    assert result["success"] is True
    assert "Found 1 matches in A0001" in result["output"]
    assert "L2: host two down" in result["output"]
    assert result["metadata"]["artifact_id"] == "A0001"


def test_artifact_read_marks_past_eof_requests_in_metadata(tmp_path: Path) -> None:
    state = _state_with_artifact(tmp_path, content="alpha\nbeta\n")

    result = artifact_read(state, artifact_id="A0001", start_line=51, end_line=100)

    assert result["success"] is True
    assert result["metadata"]["eof_overread"] is True
    assert result["metadata"]["requested_start_line"] == 51
    assert result["metadata"]["artifact_total_lines"] == 2
    assert "Stop reading and synthesize the results" in result["output"]


def test_artifact_read_warns_for_durably_stale_file_snapshot(tmp_path: Path) -> None:
    state = _state_with_artifact(tmp_path, content="old line\n")
    state.artifacts["A0001"].kind = "file_read"
    state.artifacts["A0001"].tool_name = "file_read"
    state.artifacts["A0001"].source = str(tmp_path / "example.py")
    state.scratchpad["_artifact_staleness"] = {
        "A0001": {
            "stale": True,
            "reason": "file_changed",
            "paths": [str(tmp_path / "example.py")],
        }
    }

    result = artifact_read(state, artifact_id="A0001")

    assert result["success"] is True
    assert result["metadata"]["stale"] is True
    assert result["metadata"]["authoritative_path"] == str(tmp_path / "example.py")
    assert "WARNING: This artifact is stale" in result["output"]
    assert "Use `file_read(path='" in result["output"]


def test_artifact_grep_warns_for_durably_stale_file_snapshot(tmp_path: Path) -> None:
    state = _state_with_artifact(tmp_path, content="old target\n")
    state.artifacts["A0001"].kind = "file_read"
    state.artifacts["A0001"].tool_name = "file_read"
    state.artifacts["A0001"].source = str(tmp_path / "example.py")
    state.scratchpad["_artifact_staleness"] = {
        "A0001": {
            "stale": True,
            "reason": "file_changed",
            "paths": [str(tmp_path / "example.py")],
        }
    }

    result = artifact_grep(state, artifact_id="A0001", query="target")

    assert result["success"] is True
    assert result["metadata"]["stale"] is True
    assert "WARNING: This artifact is stale" in result["output"]
    assert "Found 1 matches in A0001" in result["output"]


def test_shell_exec_failure_message_caps_long_error_text() -> None:
    long_error = "ERROR START\n" + ("x" * 5000) + "\nERROR END"
    artifact = ArtifactRecord(
        artifact_id="A9999",
        kind="shell_exec",
        source="pytest",
        created_at="2026-04-20T00:00:00+00:00",
        size_bytes=len(long_error.encode("utf-8")),
        summary="long pytest failure",
        tool_name="shell_exec",
    )
    result = ToolEnvelope(
        success=False,
        error=long_error,
        metadata={
            "output": {
                "stdout": "",
                "stderr": "stderr tail should not make the inline message huge" * 200,
                "exit_code": 1,
            }
        },
    )

    message = format_compact_tool_message(artifact, result)

    assert "ERROR START" in message
    assert "ERROR END" not in message
    assert "... output truncated" in message
    assert "hidden Artifact A9999" in message
    assert len(message) < 1900


def test_artifact_grep_infers_regex_for_regex_looking_query(tmp_path: Path) -> None:
    state = _state_with_artifact(tmp_path, content="host one up\nhost two down\nserver ready\n")

    result = artifact_grep(state, artifact_id="A0001", query="host|server")

    assert result["success"] is True
    assert "Found 3 matches" in result["output"]
    assert result["metadata"]["regex_inferred"] is True


def test_artifact_grep_regex_mode_matches_pattern(tmp_path: Path) -> None:
    state = _state_with_artifact(tmp_path, content="host one up\nhost two down\nserver ready\n")

    result = artifact_grep(state, artifact_id="A0001", query=r"host|server", regex=True)

    assert result["success"] is True
    assert "Found 3 matches" in result["output"]


def test_artifact_grep_literal_mode_allows_normal_queries(tmp_path: Path) -> None:
    state = _state_with_artifact(tmp_path, content="const x = (a + b) * 2;\n")

    result = artifact_grep(state, artifact_id="A0001", query="(a + b)")

    assert result["success"] is True
    assert "Found 1 matches" in result["output"]


def test_artifact_grep_rejects_code_search_on_directory_listing(tmp_path: Path) -> None:
    state = _state_with_artifact(tmp_path, content="restart_backoff.py [file] (6439 bytes)\n")
    state.artifacts["A0001"].kind = "dir_list"
    state.artifacts["A0001"].tool_name = "dir_list"
    state.artifacts["A0001"].source = str(tmp_path)
    state.artifacts["A0001"].metadata = {"tool_name": "dir_list", "path": str(tmp_path)}

    result = artifact_grep(state, artifact_id="A0001", query="def calculate_delay")

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "artifact_kind_mismatch"
    assert result["metadata"]["next_recommended_tool"] == "file_read"
    assert "contains a directory listing, not file contents" in result["error"]


def test_record_result_compacts_ssh_file_write_arguments_in_artifact_json(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-test"
    content = "body { color: #111; }\n" * 200
    result = ToolEnvelope(
        success=True,
        output={
            "path": "/var/www/html/site.css",
            "bytes_written": len(content.encode("utf-8")),
            "old_sha256": "old",
            "new_sha256": "new",
            "readback_sha256": "new",
            "changed": True,
        },
        metadata={
            "path": "/var/www/html/site.css",
            "host": "example.test",
            "bytes_written": len(content.encode("utf-8")),
            "old_sha256": "old",
            "new_sha256": "new",
            "readback_sha256": "new",
            "changed": True,
        },
    )
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path / "artifacts", "run-test", session_id=state.thread_id),
        context_policy=ContextPolicy(artifact_summarization_threshold=999999),
        summarizer_client=None,
        summarizer=None,
        client=SimpleNamespace(model="qwen3.5:4b"),
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: "update remote site css",
    )
    service = SimpleNamespace(harness=harness)

    asyncio.run(
        record_result(
            service,
            tool_name="ssh_file_write",
            tool_call_id="call-1",
            result=result,
            arguments={
                "target": "root@example.test",
                "path": "/var/www/html/site.css",
                "content": content,
                "password": "secret-password",
            },
        )
    )

    artifact = next(iter(state.artifacts.values()))
    payload = json.loads(Path(artifact.content_path).with_suffix(".json").read_text(encoding="utf-8"))
    arguments = payload["metadata"]["arguments"]
    serialized_arguments = json.dumps(arguments, ensure_ascii=True)
    assert "content_sha256" in arguments
    assert arguments["content_chars"] == len(content)
    assert "content" not in arguments
    assert content not in serialized_arguments
    assert "secret-password" not in serialized_arguments


def test_record_result_keeps_file_reads_inline_without_artifacts(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-test"
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path / "artifacts", "run-test", session_id=state.thread_id),
        context_policy=ContextPolicy(artifact_summarization_threshold=999999),
        summarizer_client=None,
        summarizer=None,
        client=SimpleNamespace(model="qwen3.5:4b"),
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: "inspect a local file",
    )
    service = SimpleNamespace(harness=harness)

    message = asyncio.run(
        record_result(
            service,
            tool_name="file_read",
            tool_call_id="call-1",
            result=ToolEnvelope(success=True, output="alpha\nbeta\n", metadata={"path": "notes.txt"}),
            arguments={"path": "notes.txt"},
        )
    )

    assert state.artifacts == {}
    assert state.retrieval_cache == []
    assert "FILE READ STATUS:" in message.content
    assert "complete_file=false" in message.content
    assert "alpha\nbeta\n" in message.content
    assert "artifact_id" not in message.metadata


def test_record_result_persists_budget_exceeding_file_read_artifact(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-test"
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path / "artifacts", "run-test", session_id=state.thread_id),
        context_policy=ContextPolicy(tool_result_inline_token_limit=64, artifact_summarization_threshold=999999),
        summarizer_client=None,
        summarizer=None,
        client=SimpleNamespace(model="qwen3.5:4b"),
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: "inspect a local file",
    )
    service = SimpleNamespace(harness=harness)
    output = "\n".join(f"line {i}: def calculate_delay_{i}(): pass" for i in range(80))

    message = asyncio.run(
        record_result(
            service,
            tool_name="file_read",
            tool_call_id="call-1",
            result=ToolEnvelope(
                success=True,
                output=output,
                metadata={
                    "path": "restart_backoff.py",
                    "complete_file": True,
                    "truncated": False,
                    "line_start": 1,
                    "line_end": 80,
                    "total_lines": 80,
                },
            ),
            arguments={"path": "restart_backoff.py"},
        )
    )

    assert list(state.artifacts) == ["A0001"]
    assert "FILE READ STATUS:" in message.content
    assert "display_preview_truncated=" in message.content
    assert "file_content_truncated=false" in message.content
    assert message.metadata["artifact_id"] == "A0001"
    assert message.metadata["complete_file"] is True


def test_file_read_artifact_pages_dict_content_as_file_text(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    content = "\n".join(f"line {idx}" for idx in range(1, 901))
    store = ArtifactStore(tmp_path / "artifacts", "run-test", session_id="thread-test")

    artifact = store.persist_tool_result(
        tool_name="file_read",
        result=ToolEnvelope(
            success=True,
            output={"content": content, "path": "large.html"},
            metadata={
                "path": "large.html",
                "complete_file": True,
                "truncated": False,
                "line_start": 1,
                "line_end": 900,
                "total_lines": 900,
            },
        ),
        session_id="thread-test",
        tool_call_id="call-1",
    )
    state.artifacts[artifact.artifact_id] = artifact

    page = artifact_read(state, artifact_id=artifact.artifact_id, start_line=601, end_line=900)

    assert Path(artifact.content_path).read_text(encoding="utf-8") == content
    assert page["success"] is True
    assert page["metadata"]["total_lines"] == 900
    assert "line 601" in page["output"]
    assert "line 900" in page["output"]
    assert "content:" not in page["output"]


def test_record_result_reused_file_read_preserves_completion_metadata(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-test"
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path / "artifacts", "run-test", session_id=state.thread_id),
        context_policy=ContextPolicy(tool_result_inline_token_limit=64, artifact_summarization_threshold=999999),
        summarizer_client=None,
        summarizer=None,
        client=SimpleNamespace(model="qwen3.5:4b"),
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: "inspect a local file",
    )
    service = SimpleNamespace(harness=harness)
    output = "\n".join(f"line {i}" for i in range(150))

    first = asyncio.run(
        record_result(
            service,
            tool_name="file_read",
            tool_call_id="call-1",
            result=ToolEnvelope(
                success=True,
                output=output,
                metadata={
                    "path": "text_chunker.py",
                    "complete_file": True,
                    "truncated": False,
                    "line_start": 1,
                    "line_end": 150,
                    "total_lines": 150,
                },
            ),
            arguments={"path": "text_chunker.py"},
        )
    )

    second = asyncio.run(
        record_result(
            service,
            tool_name="file_read",
            tool_call_id="call-2",
            result=ToolEnvelope(
                success=True,
                output={"status": "cached", "artifact_id": first.metadata["artifact_id"]},
                metadata={"cache_hit": True, "artifact_id": first.metadata["artifact_id"]},
            ),
            arguments={"path": "text_chunker.py"},
        )
    )

    assert second.metadata["cache_hit"] is True
    assert second.metadata["complete_file"] is True
    assert second.metadata["file_content_truncated"] is False
    assert second.metadata["line_start"] == 1
    assert second.metadata["line_end"] == 150
    assert second.metadata["total_lines"] == 150
    assert "Full cached content is visible below" not in second.content


def test_prompt_assembler_keeps_300_line_file_read_excerpt(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    content = "\n".join(f"line {idx}" for idx in range(1, 351))
    content_path = tmp_path / "A0001.txt"
    content_path.write_text(content, encoding="utf-8")
    state.artifacts["A0001"] = ArtifactRecord(
        artifact_id="A0001",
        kind="file_read",
        source=str(tmp_path / "large_file.py"),
        created_at="2026-05-23T00:00:00+00:00",
        size_bytes=len(content.encode("utf-8")),
        summary="large_file.py full file",
        tool_name="file_read",
        content_path=str(content_path),
        metadata={
            "path": str(tmp_path / "large_file.py"),
            "complete_file": True,
            "truncated": False,
            "line_start": 1,
            "line_end": 350,
            "total_lines": 350,
        },
    )
    message = ConversationMessage(
        role="tool",
        name="file_read",
        content=content,
        metadata={
            "artifact_id": "A0001",
            "complete_file": True,
            "file_content_truncated": False,
            "line_start": 1,
            "line_end": 350,
            "total_lines": 350,
        },
    )
    assembler = PromptAssembler(
        ContextPolicy(
            tool_result_inline_token_limit=64,
            transcript_token_limit=10000,
            file_read_preview_line_limit=300,
        )
    )

    compacted = assembler._compact_message_for_prompt(state, message, transcript_token_limit=10000)

    assert "up to 300 lines before prompt-preview truncation" in compacted.content
    assert "line 300" in compacted.content
    assert "line 301" not in compacted.content
    assert "prompt preview truncated at 300/350 lines" in compacted.content


def test_record_result_keeps_ssh_file_reads_inline_without_artifacts(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-test"
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path / "artifacts", "run-test", session_id=state.thread_id),
        context_policy=ContextPolicy(artifact_summarization_threshold=999999),
        summarizer_client=None,
        summarizer=None,
        client=SimpleNamespace(model="qwen3.5:4b"),
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: "inspect a remote file",
    )
    service = SimpleNamespace(harness=harness)

    output = {"content": "alpha\nbeta\n", "path": "/etc/example.conf", "host": "example.test"}
    message = asyncio.run(
        record_result(
            service,
            tool_name="ssh_file_read",
            tool_call_id="call-1",
            result=ToolEnvelope(
                success=True,
                output=output,
                metadata={"path": "/etc/example.conf", "host": "example.test"},
            ),
            arguments={"path": "/etc/example.conf", "host": "example.test"},
        )
    )

    assert state.artifacts == {}
    assert state.retrieval_cache == []
    assert "FILE READ STATUS:" in message.content
    assert "alpha\nbeta\n" in message.content
    assert "artifact_id" not in message.metadata


def test_large_file_read_artifact_survives_graph_state_serialization(tmp_path: Path) -> None:
    """Regression: graph state serialization compacts large tool outputs, so
    artifacts must be persisted before serialization. Otherwise the artifact
    ends up containing only a preview while its metadata claims it is a
    complete file.
    """
    state = LoopState(cwd=str(tmp_path))
    state.thread_id = "thread-test"
    harness = SimpleNamespace(
        state=state,
        artifact_store=ArtifactStore(tmp_path / "artifacts", "run-test", session_id=state.thread_id),
        context_policy=ContextPolicy(tool_result_inline_token_limit=64, artifact_summarization_threshold=999999),
        summarizer_client=None,
        summarizer=None,
        client=SimpleNamespace(model="qwen3.5:4b"),
        _runlog=lambda *args, **kwargs: None,
        _current_user_task=lambda: "inspect a local file",
    )
    harness.tool_results = ToolResultService(harness)
    service = SimpleNamespace(harness=harness)

    output = "\n".join(
        f"line {i}: some content here that makes each line reasonably long for byte counting purposes"
        for i in range(1, 2300)
    )
    result = ToolEnvelope(
        success=True,
        output=output,
        metadata={
            "path": str(tmp_path / "large.html"),
            "complete_file": True,
            "truncated": False,
            "line_start": 1,
            "line_end": 2300,
            "total_lines": 2300,
            "bytes": len(output.encode("utf-8")),
        },
    )

    # Simulate dispatch_tools: persist artifact early with full content
    harness.tool_results.persist_artifact_early(
        tool_name="file_read",
        result=result,
        tool_call_id="call-1",
    )
    early_artifact_id = result.metadata["artifact_id"]
    early_artifact = state.artifacts[early_artifact_id]
    assert Path(early_artifact.content_path).read_text(encoding="utf-8") == output

    # Simulate the record that dispatch_tools adds to last_tool_results
    graph_state = GraphRunState(loop_state=state, thread_id="thread-test", run_mode="loop")
    graph_state.last_tool_results.append(
        ToolExecutionRecord(
            operation_id="op1",
            tool_name="file_read",
            args={"path": str(tmp_path / "large.html")},
            tool_call_id="call-1",
            result=result,
        )
    )

    # Serialization between graph nodes compacts the result
    payload = serialize_graph_state(graph_state)
    graph_state2 = inflate_graph_state(payload)
    compacted_result = graph_state2.last_tool_results[0].result
    assert len(compacted_result.output) < len(output)
    assert "[output compacted" in compacted_result.output

    # persist_tool_results receives the compacted result but reuses the early artifact
    message = asyncio.run(
        record_result(
            service,
            tool_name="file_read",
            tool_call_id="call-1",
            result=compacted_result,
            arguments={"path": str(tmp_path / "large.html")},
        )
    )

    assert message.metadata["artifact_id"] == early_artifact_id
    artifact = state.artifacts[early_artifact_id]
    assert Path(artifact.content_path).read_text(encoding="utf-8") == output
    assert artifact.metadata["complete_file"] is True
    assert artifact.metadata["total_lines"] == 2300
    assert artifact.metadata["truncated"] is False

