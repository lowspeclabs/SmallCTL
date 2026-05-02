from __future__ import annotations

from pathlib import Path

from smallctl.models.tool_result import ToolEnvelope
from smallctl.context.artifacts import ArtifactStore, artifact_storage_id


def test_artifact_storage_id_prefixes_session_id_when_distinct() -> None:
    assert artifact_storage_id(run_id="6caafbd0", session_id="thread-42") == "thread-42-6caafbd0"


def test_artifact_storage_id_avoids_duplicate_session_prefix() -> None:
    assert artifact_storage_id(run_id="thread-42-6caafbd0", session_id="thread-42") == "thread-42-6caafbd0"
    assert artifact_storage_id(run_id="thread-42", session_id="thread-42") == "thread-42"


def test_artifact_storage_id_sanitizes_session_id_for_directory_names() -> None:
    assert artifact_storage_id(run_id="6caafbd0", session_id="team/alpha child:01") == "team_alpha_child_01-6caafbd0"


def test_artifact_store_uses_session_prefixed_storage_dir(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "6caafbd0", session_id="thread-42")

    assert store.storage_id == "thread-42-6caafbd0"
    assert store.run_dir == tmp_path / "thread-42-6caafbd0"


def test_artifact_store_summarizes_write_session_file_write_with_handoff_state(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "run-1")

    artifact = store.persist_tool_result(
        tool_name="file_write",
        result=ToolEnvelope(
            success=True,
            output=(
                "Section `imports_and_classes` written to `./temp/dead_letter_queue.py`. "
                "Waiting for next section: `test_runner`.\n"
                "WRITE_SESSION_STATUS type=write_session id=ws_185620 mode=chunked_author "
                "next=test_runner staged_hash=15bd6f99fbc9 finalized=no"
            ),
            metadata={
                "path": "/tmp/dead_letter_queue.py",
                "write_session_id": "ws_185620",
                "section_name": "imports_and_classes",
                "write_next_section": "test_runner",
                "staged_only": True,
                "write_session_finalized": False,
            },
        ),
    )

    assert artifact.summary == (
        "dead_letter_queue.py section imports_and_classes written in Write Session ws_185620; "
        "next=test_runner; staged-only; not finalized"
    )


def test_artifact_store_summarizes_ast_patch(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "run-1")

    artifact = store.persist_tool_result(
        tool_name="ast_patch",
        result=ToolEnvelope(
            success=True,
            output="Structurally patched `src/app.py` with `replace_function`.",
            metadata={
                "path": "/tmp/src/app.py",
                "operation": "replace_function",
                "changed": True,
            },
        ),
    )

    assert artifact.summary == "app.py structurally patched"


def test_artifact_store_persists_dir_list_as_readable_listing_text(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "run-1")

    artifact = store.persist_tool_result(
        tool_name="dir_list",
        result=ToolEnvelope(
            success=True,
            output=[
                {
                    "name": "src",
                    "path": "/workspace/src",
                    "type": "dir",
                    "children_count": 2,
                    "children": [
                        {"name": "app.py", "path": "/workspace/src/app.py", "type": "file", "size": 123},
                        {"name": "lib.py", "path": "/workspace/src/lib.py", "type": "file", "size": 456},
                    ],
                },
                {"name": "README.md", "path": "/workspace/README.md", "type": "file", "size": 42},
            ],
            metadata={"path": "/workspace", "count": 2, "total_items": 2},
        ),
    )

    content = Path(artifact.content_path).read_text(encoding="utf-8")

    assert content.startswith("/workspace (2 items)")
    assert "src [dir] (2 children)" in content
    assert "README.md [file] (42 bytes)" in content
    assert '"name": "src"' not in content


def test_artifact_store_dir_list_preview_uses_listing_text(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "run-1")

    artifact = store.persist_tool_result(
        tool_name="dir_list",
        result=ToolEnvelope(
            success=True,
            output=[f"item-{index:02d}" for index in range(1, 5)],
            metadata={"path": "./temp", "count": 4, "total_items": 4},
        ),
    )

    assert artifact.preview_text == "./temp (4 items)\nitem-01\nitem-02\nitem-03\nitem-04"


def test_artifact_store_renders_web_search_results_as_readable_text(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "run-1")

    artifact = store.persist_tool_result(
        tool_name="web_search",
        result=ToolEnvelope(
            success=True,
            output={
                "query": "weather jacksonville fl current forecast",
                "provider": "duckduckgo",
                "recency_support": "none",
                "recency_enforced": False,
                "warnings": ["recency not enforced"],
                "results": [
                    {
                        "result_id": "webres-1",
                        "fetch_id": "r1",
                        "title": "Jacksonville, FL Weather Forecast | National Weather Service",
                        "url": "https://www.weather.gov/jax/",
                        "domain": "weather.gov",
                        "snippet": "Current conditions and local forecast for Jacksonville, Florida.",
                    },
                    {
                        "result_id": "webres-2",
                        "fetch_id": "r2",
                        "title": "Jacksonville Weather",
                        "url": "https://example.com/jax-weather",
                        "domain": "example.com",
                        "snippet": "Backup forecast coverage.",
                    },
                ],
            },
        ),
    )

    content = Path(artifact.content_path).read_text(encoding="utf-8")

    assert artifact.summary.startswith('2 results for "weather jacksonville fl current forecast":')
    assert "National Weather Service" in artifact.summary
    assert "Use with: web_fetch(result_id='r1')" in artifact.preview_text
    assert content.startswith("Query: weather jacksonville fl current forecast")
    assert "1. Jacksonville, FL Weather Forecast | National Weather Service" in content
    assert "Use with: web_fetch(result_id='r1')" in content
    assert "URL: https://www.weather.gov/jax/" in content
    assert "Snippet: Current conditions and local forecast" in content
    assert '"query": "weather jacksonville fl current forecast"' not in content


def test_persist_generated_text_supports_preview_override_without_changing_body(tmp_path) -> None:
    store = ArtifactStore(tmp_path, "run-1")

    artifact = store.persist_generated_text(
        kind="web_fetch",
        source="https://example.com/article",
        content="line one\nline two\nline three\n",
        summary="Example article",
        preview_text="Title: Example article\nURL: https://example.com/article\n\nExcerpt:\nline one",
        metadata={"render_mode": "body_with_preview"},
        tool_name="web_fetch",
    )

    content = Path(artifact.content_path).read_text(encoding="utf-8")

    assert content == "line one\nline two\nline three\n"
    assert artifact.preview_text.startswith("Title: Example article")
