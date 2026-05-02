from __future__ import annotations

from smallctl.graph.display import format_tool_result_display
from smallctl.context.messages import format_compact_tool_message
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
