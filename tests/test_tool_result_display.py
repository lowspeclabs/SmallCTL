from __future__ import annotations

from pathlib import Path

from smallctl.graph.display import format_tool_result_display
from smallctl.context.artifacts import ArtifactStore
from smallctl.context.messages import format_compact_tool_message, format_reused_artifact_message
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord
from smallctl.tool_output_formatting import structured_plain_text


def test_format_tool_result_display_preserves_actionable_tail_for_long_errors() -> None:
    prefix = "Patch-existing write sessions need an explicit first-chunk choice. " * 8
    tail = (
        "If earlier chunks are not fully visible in local context, call "
        "`file_read(path='./temp/logwatch.py')` first; during an active write session that reads from the staged copy. "
        "Do not assume earlier chunks were lost or rewrite the whole staged file from memory."
    )
    result = ToolEnvelope(
        success=False,
        error=prefix + tail,
        metadata={},
    )

    rendered = format_tool_result_display(tool_name="file_write", result=result)

    assert "... error truncated" not in rendered
    assert "file_read(path='./temp/logwatch.py')" in rendered
    assert "Do not assume earlier chunks were lost" in rendered
    assert "\n...\n" in rendered


def test_format_tool_result_display_keeps_short_errors_unchanged() -> None:
    result = ToolEnvelope(
        success=False,
        error="Short failure.",
        metadata={},
    )

    rendered = format_tool_result_display(tool_name="file_write", result=result)

    assert rendered == "Short failure."


def test_format_tool_result_display_labels_ssh_remote_nonzero_as_reached_host() -> None:
    result = ToolEnvelope(
        success=False,
        error="Remote SSH command exited with code 2",
        metadata={
            "failure_mode": "remote_exit_nonzero",
            "ssh_error_class": "remote_exit_nonzero",
            "ssh_transport_succeeded": True,
        },
    )

    rendered = format_tool_result_display(tool_name="ssh_exec", result=result)

    assert rendered.startswith("SSH reached the remote host; remote command exited non-zero.")
    assert "Remote SSH command exited with code 2" in rendered


def test_format_tool_result_display_preserves_word_boundaries_in_tail_guidance() -> None:
    prefix = "Patch-existing write sessions need an explicit first-chunk choice. " * 8
    tail = (
        "Choose `replace_strategy='overwrite'` only after rereading the staged copy, "
        "and do not broad-rewrite from memory."
    )
    result = ToolEnvelope(
        success=False,
        error=prefix + tail,
        metadata={},
    )

    rendered = format_tool_result_display(tool_name="file_write", result=result)

    assert "replace_strategy='overwrite'" in rendered
    assert "broad-rewrite from memory" in rendered


def test_structured_web_search_output_includes_result_ids() -> None:
    rendered = structured_plain_text(
        {
            "query": "latest example article",
            "provider": "duckduckgo",
            "recency_support": "none",
            "recency_enforced": False,
            "results": [
                {
                    "result_id": "webres-test-1",
                    "fetch_id": "r1",
                    "title": "Example result",
                    "url": "https://example.com/article",
                    "domain": "example.com",
                    "snippet": "A short snippet",
                }
            ],
        }
    )

    assert rendered is not None
    assert "Fetch ID: r1" in rendered
    assert "Result ID: webres-test-1" in rendered
    assert "Use with: web_fetch(result_id='r1')" in rendered
    assert "URL: https://example.com/article" in rendered


def test_structured_web_fetch_output_includes_exact_artifact_read_hint() -> None:
    rendered = structured_plain_text(
        {
            "title": "Example result",
            "url": "https://example.com/article",
            "domain": "example.com",
            "text_excerpt": "Short excerpt",
            "body_artifact_id": "A0007",
        }
    )

    assert rendered is not None
    assert "artifact_read(artifact_id='A0007')" in rendered


def test_ssh_file_write_compact_message_is_actionable_confirmation() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0100",
        kind="ssh_file_write",
        source="/var/www/html/index.html",
        created_at="2026-04-30T00:00:00+00:00",
        size_bytes=1024,
        summary="index.html written",
        tool_name="ssh_file_write",
    )
    result = ToolEnvelope(
        success=True,
        output={"status": "ok"},
        metadata={
            "host": "192.168.1.63",
            "user": "root",
            "path": "/var/www/html/index.html",
            "bytes_written": 42,
            "old_sha256": "oldhash",
            "new_sha256": "newhash",
            "readback_sha256": "newhash",
            "changed": True,
            "arguments": {"content_preview": "<html>short preview</html>", "content_chars": 4096},
        },
    )

    rendered = format_compact_tool_message(artifact, result)

    assert "Remote file written: root@192.168.1.63:/var/www/html/index.html" in rendered
    assert "bytes_written: 42" in rendered
    assert "old_sha256: oldhash" in rendered
    assert "new_sha256: newhash" in rendered
    assert "readback verified: yes" in rendered
    assert "Tool output captured as Artifact" not in rendered
    assert "artifact_read" not in rendered


def test_file_write_compact_message_is_actionable_disk_confirmation() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0104",
        kind="file_write",
        source="/repo/temp/leader_election_sim.py",
        created_at="2026-05-22T00:00:00+00:00",
        size_bytes=924,
        summary="leader_election_sim.py written",
        tool_name="file_write",
    )
    result = ToolEnvelope(
        success=True,
        output="written",
        metadata={
            "path": "/repo/temp/leader_election_sim.py",
            "bytes": 817,
            "changed": True,
        },
    )

    rendered = format_compact_tool_message(artifact, result)

    assert "Local file written: /repo/temp/leader_election_sim.py" in rendered
    assert "bytes_written: 817" in rendered
    assert "changed: yes" in rendered
    assert "persisted_to_target: yes" in rendered
    assert "Tool output captured as Artifact" not in rendered


def test_write_session_compact_message_labels_staged_only_not_persisted() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0105",
        kind="file_write",
        source="/repo/temp/leader_election_sim.py",
        created_at="2026-05-22T00:00:00+00:00",
        size_bytes=924,
        summary="leader_election_sim.py staged",
        tool_name="file_write",
    )
    result = ToolEnvelope(
        success=True,
        output="section written",
        metadata={
            "path": "/repo/temp/leader_election_sim.py",
            "staging_path": "/repo/.smallctl/write_sessions/ws_1__leader_election_sim__stage.py",
            "write_session_id": "ws_1",
            "staged_only": True,
            "bytes": 817,
            "changed": True,
        },
    )

    rendered = format_compact_tool_message(artifact, result)

    assert "Local file written: /repo/temp/leader_election_sim.py" in rendered
    assert "staging_path: /repo/.smallctl/write_sessions/ws_1__leader_election_sim__stage.py" in rendered
    assert "write_session_id: ws_1" in rendered
    assert "persisted_to_target: no; staged_only=true" in rendered


def test_file_read_compact_message_distinguishes_preview_from_file_truncation() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0102",
        kind="file_read",
        source="/repo/temp/restart_backoff.py",
        created_at="2026-05-22T00:00:00+00:00",
        size_bytes=4096,
        summary="restart_backoff.py read",
        tool_name="file_read",
    )
    content = "def calculate_delay(attempt):\n    return attempt\n" + ("# filler\n" * 200)
    result = ToolEnvelope(
        success=True,
        output=content,
        metadata={
            "path": "/repo/temp/restart_backoff.py",
            "source_path": "/repo/temp/restart_backoff.py",
            "complete_file": True,
            "truncated": False,
            "line_start": 1,
            "line_end": 202,
            "total_lines": 202,
        },
    )

    rendered = format_compact_tool_message(
        artifact,
        result,
        inline_full_file=False,
        full_file_preview_chars=80,
    )

    assert "FILE READ STATUS:" in rendered
    assert "complete_file=true" in rendered
    assert "display_preview_truncated=true" in rendered
    assert "file_content_truncated=false" in rendered
    assert "The file itself was not truncated" in rendered
    assert "artifact_id=A0102" in rendered


def test_small_model_file_read_preview_defaults_to_300_lines_before_truncation() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0110",
        kind="file_read",
        source="/repo/temp/large_file.py",
        created_at="2026-05-23T00:00:00+00:00",
        size_bytes=4096,
        summary="large_file.py read",
        tool_name="file_read",
    )
    content = "\n".join(f"line {idx}" for idx in range(1, 351))
    result = ToolEnvelope(
        success=True,
        output=content,
        metadata={
            "path": "/repo/temp/large_file.py",
            "complete_file": True,
            "truncated": False,
            "line_start": 1,
            "line_end": 350,
            "total_lines": 350,
        },
    )

    rendered = format_compact_tool_message(artifact, result, inline_full_file=False)

    assert "Preview (first 300 of 350 lines):" in rendered
    assert "line 300" in rendered
    assert "line 301" not in rendered
    assert "file_content_truncated=false" in rendered


def test_reused_file_read_message_is_status_not_misleading_full_inline() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0111",
        kind="file_read",
        source="/repo/temp/text_chunker.py",
        created_at="2026-05-23T00:00:00+00:00",
        size_bytes=4096,
        summary="text_chunker.py full file",
        tool_name="file_read",
        inline_content="print('cached')\n",
        metadata={
            "path": "/repo/temp/text_chunker.py",
            "complete_file": True,
            "truncated": False,
            "line_start": 1,
            "line_end": 150,
            "total_lines": 150,
        },
    )

    rendered = format_reused_artifact_message(artifact, tool_name="file_read")

    assert "FILE READ CACHE STATUS:" in rendered
    assert "complete_file=true" in rendered
    assert "file_content_truncated=false" in rendered
    assert "lines=1-150 of 150" in rendered
    assert "Full cached content is visible below" not in rendered
    assert "```text" not in rendered


def test_file_read_compact_message_labels_active_write_session_staging() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0103",
        kind="file_read",
        source="/repo/temp/patch_dependency_sim.py",
        created_at="2026-05-22T00:00:00+00:00",
        size_bytes=32,
        summary="patch_dependency_sim.py read",
        tool_name="file_read",
    )
    result = ToolEnvelope(
        success=True,
        output="print('staged')\n",
        metadata={
            "path": "/repo/temp/patch_dependency_sim.py",
            "source_path": "/repo/.smallctl/write_sessions/ws_abc123__patch_dependency_sim__stage.py",
            "read_from_staging": True,
            "write_session_id": "ws_abc123",
            "complete_file": True,
            "truncated": False,
            "line_start": 1,
            "line_end": 1,
            "total_lines": 1,
        },
    )

    rendered = format_compact_tool_message(
        artifact,
        result,
        inline_full_file=True,
    )

    assert "source_path=/repo/.smallctl/write_sessions/ws_abc123__patch_dependency_sim__stage.py" in rendered
    assert "read_from_active_write_session_staging=true; write_session_id=ws_abc123" in rendered


def test_ssh_exec_compact_message_labels_remote_nonzero_as_reached_host() -> None:
    artifact = ArtifactRecord(
        artifact_id="A0101",
        kind="ssh_exec",
        source="ssh://root@192.168.1.63",
        created_at="2026-04-30T00:00:00+00:00",
        size_bytes=512,
        summary="ssh_exec failure",
        tool_name="ssh_exec",
    )
    result = ToolEnvelope(
        success=False,
        error="Remote SSH command exited with code 2",
        metadata={
            "failure_mode": "remote_exit_nonzero",
            "ssh_error_class": "remote_exit_nonzero",
            "ssh_transport_succeeded": True,
            "output": {"stdout": "", "stderr": "", "exit_code": 2},
        },
    )

    rendered = format_compact_tool_message(artifact, result)

    assert rendered.startswith("SSH reached the remote host; remote command exited non-zero.")
    assert "Remote SSH command exited with code 2" in rendered


def test_shell_exec_failure_summary_precedes_long_unittest_transcript() -> None:
    passing_lines = "\n".join(f"test_ok_{idx} (__main__.Suite.test_ok_{idx}) ... ok" for idx in range(40))
    stderr = (
        f"{passing_lines}\n"
        "test_print_verdict_allowed (__main__.TestIPAllowlist.test_print_verdict_allowed) ... ERROR\n"
        "\n"
        "======================================================================\n"
        "ERROR: test_print_verdict_allowed (__main__.TestIPAllowlist.test_print_verdict_allowed)\n"
        "----------------------------------------------------------------------\n"
        "Traceback (most recent call last):\n"
        "  File \"/tmp/ip_allowlist.py\", line 65, in print_verdict\n"
        "    print(f\"ALLOWED: {ip_str}\")\n"
        "AttributeError: 'list' object has no attribute 'write'\n"
        "\n"
        "----------------------------------------------------------------------\n"
        "Ran 16 tests in 0.003s\n"
        "\n"
        "FAILED (errors=3)\n"
    )
    artifact = ArtifactRecord(
        artifact_id="A0102",
        kind="shell_exec",
        source="python3 temp/ip_allowlist.py",
        created_at="2026-05-21T00:00:00+00:00",
        size_bytes=len(stderr.encode("utf-8")),
        summary="unittest failure",
        tool_name="shell_exec",
    )
    result = ToolEnvelope(
        success=False,
        error=stderr,
        metadata={"output": {"stdout": "", "stderr": stderr, "exit_code": 120}},
    )

    rendered = format_compact_tool_message(artifact, result)

    assert rendered.startswith("--- [FAILURE SUMMARY] ---")
    assert "Command failed with exit code 120." in rendered
    assert "FAILED (errors=3)" in rendered
    assert "ERROR: test_print_verdict_allowed" in rendered
    assert "AttributeError: 'list' object has no attribute 'write'" in rendered
    assert rendered.index("FAILED (errors=3)") < rendered.index("test_ok_0")


def test_format_tool_result_display_includes_next_required_action() -> None:
    result = ToolEnvelope(
        success=False,
        error="Tool blocked.",
        metadata={
            "next_required_action": {
                "tool_name": "ssh_exec",
                "required_arguments": {"command": "echo ok"},
                "notes": ["Run validator first."],
            }
        },
    )

    rendered = format_tool_result_display(tool_name="ssh_exec", result=result)

    assert "Recovery hint:" in rendered
    assert "Next required action:" in rendered
    assert '"tool_name": "ssh_exec"' in rendered
    assert "Run validator first." in rendered


def test_format_tool_result_display_includes_next_required_tool() -> None:
    result = ToolEnvelope(
        success=False,
        error="Tool blocked.",
        metadata={
            "next_required_tool": {
                "tool_name": "file_read",
                "required_arguments": {"path": "/etc/apt/sources.list"},
            }
        },
    )

    rendered = format_tool_result_display(tool_name="shell_exec", result=result)

    assert "Recovery hint:" in rendered
    assert "Next required tool:" in rendered
    assert '"tool_name": "file_read"' in rendered


def test_format_tool_result_display_skips_recovery_hint_when_none() -> None:
    result = ToolEnvelope(
        success=False,
        error="Simple failure.",
        metadata={},
    )

    rendered = format_tool_result_display(tool_name="shell_exec", result=result)

    assert "Recovery hint:" not in rendered
    assert rendered == "Simple failure."


def test_format_tool_result_display_includes_both_action_and_tool() -> None:
    result = ToolEnvelope(
        success=False,
        error="Blocked.",
        metadata={
            "next_required_action": "fresh clone or clean reset",
            "next_required_tool": {"tool_name": "git_status"},
        },
    )

    rendered = format_tool_result_display(tool_name="shell_exec", result=result)

    assert "Recovery hint:" in rendered
    assert "Next required action: fresh clone or clean reset" in rendered
    assert "Next required tool:" in rendered
    assert '"tool_name": "git_status"' in rendered


def test_artifact_store_redacts_sensitive_tool_output(tmp_path) -> None:
    store = ArtifactStore(tmp_path, run_id="run1")
    result = ToolEnvelope(
        success=True,
        output={"stdout": "TOKEN=secret-value\n", "stderr": "", "exit_code": 0},
        metadata={"arguments": {"command": "cat .env"}},
    )

    artifact = store.persist_tool_result(tool_name="shell_exec", result=result)

    content = Path(artifact.content_path).read_text(encoding="utf-8")
    assert "secret-value" not in content
    assert "[REDACTED]" in content
    assert artifact.preview_text is None or "secret-value" not in artifact.preview_text
