from __future__ import annotations

from pathlib import Path
import asyncio
import json
from types import SimpleNamespace

from smallctl.context.artifacts import ArtifactStore
from smallctl.context.messages import format_compact_tool_message
from smallctl.context.policy import ContextPolicy
from smallctl.harness.tool_result_flow import record_result
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
