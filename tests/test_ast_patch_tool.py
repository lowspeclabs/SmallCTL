from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from smallctl.state import LoopState
from smallctl.tools import fs


def _make_state(tmp_path: Path) -> LoopState:
    state = LoopState(cwd=str(tmp_path))
    state.active_tool_profiles = ["core"]
    state.artifacts = {}
    return state


def _make_open_write_session(
    target: Path,
    *,
    session_id: str = "ws-ast-1",
    intent: str = "replace_file",
) -> SimpleNamespace:
    return SimpleNamespace(
        write_session_id=session_id,
        status="open",
        write_session_mode="chunked_author",
        write_target_path=str(target),
        write_session_intent=intent,
        write_staging_path="",
        write_original_snapshot_path="",
        write_last_attempt_snapshot_path="",
        write_last_staged_hash="",
        write_sections_completed=[],
        write_section_ranges={},
        write_current_section="",
        write_next_section="",
        write_failed_local_patches=0,
        write_pending_finalize=False,
        write_last_verifier={},
        write_target_existed_at_start=False,
        write_first_chunk_at=0.0,
    )


def test_ast_patch_add_import_adds_missing_import(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("def build_path():\n    return Path('tmp')\n", encoding="utf-8")

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="add_import",
            target={"module": "pathlib", "name": "Path", "style": "from"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["changed"] is True
    assert target.read_text(encoding="utf-8").startswith("from pathlib import Path\n")


def test_ast_patch_add_import_is_idempotent(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    original = "from pathlib import Path\n\ndef build_path():\n    return Path('tmp')\n"
    target.write_text(original, encoding="utf-8")

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="add_import",
            target={"module": "pathlib", "name": "Path", "style": "from"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["changed"] is False
    assert target.read_text(encoding="utf-8") == original


def test_ast_patch_add_import_extends_existing_from_import(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text(
        "from pathlib import PurePath\n\n"
        "def build_path():\n"
        "    return Path('tmp')\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="add_import",
            target={"module": "pathlib", "name": "Path", "style": "from"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert "from pathlib import Path, PurePath\n" in target.read_text(encoding="utf-8")


def test_ast_patch_add_plain_import_is_idempotent(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    original = "import json\n\nvalue = json.dumps({'ok': True})\n"
    target.write_text(original, encoding="utf-8")

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="add_import",
            target={"module": "json", "style": "import"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["changed"] is False
    assert target.read_text(encoding="utf-8") == original


def test_ast_patch_replace_function_replaces_only_target_function(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text(
        "def keep_me():\n"
        "    return 'keep'\n\n"
        "def replace_me():\n"
        "    return 'old'\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="replace_function",
            target={"function": "replace_me"},
            payload={
                "source": "def replace_me():\n    return 'new'\n",
                "replace": "entire_node",
            },
            cwd=str(tmp_path),
            state=state,
        )
    )

    contents = target.read_text(encoding="utf-8")
    assert result["success"] is True
    assert "return 'keep'" in contents
    assert "return 'new'" in contents
    assert "return 'old'" not in contents


def test_ast_patch_insert_in_function_keeps_docstring_first(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text(
        "def run():\n"
        "    \"\"\"doc\"\"\"\n"
        "    return 1\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="insert_in_function",
            target={"function": "run", "position": "start"},
            payload={"statements": "value = 2"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == (
        "def run():\n"
        "    \"\"\"doc\"\"\"\n"
        "    value = 2\n"
        "    return 1\n"
    )


def test_ast_patch_insert_in_async_function_keeps_docstring_first(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text(
        "async def run():\n"
        "    \"\"\"doc\"\"\"\n"
        "    return 1\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="insert_in_function",
            target={"function": "run", "position": "start"},
            payload={"statements": "value = await fetch_value()"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == (
        "async def run():\n"
        "    \"\"\"doc\"\"\"\n"
        "    value = await fetch_value()\n"
        "    return 1\n"
    )


def test_ast_patch_update_call_keyword_updates_only_requested_scope(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text(
        "import subprocess\n\n"
        "def first():\n"
        "    return subprocess.run(['echo', 'one'])\n\n"
        "def second():\n"
        "    return subprocess.run(['echo', 'two'])\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="update_call_keyword",
            target={
                "scope_function": "second",
                "callee": "subprocess.run",
                "keyword": "timeout",
            },
            payload={"mode": "set", "value": "60"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    contents = target.read_text(encoding="utf-8")
    assert result["success"] is True
    assert "subprocess.run(['echo', 'one'])" in contents
    assert "subprocess.run(['echo', 'two'], timeout=60)" in contents


def test_ast_patch_update_call_keyword_removes_existing_keyword(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text(
        "import subprocess\n\n"
        "def run():\n"
        "    return subprocess.run(['echo', 'two'], timeout=60, check=True)\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="update_call_keyword",
            target={
                "scope_function": "run",
                "callee": "subprocess.run",
                "keyword": "timeout",
            },
            payload={"mode": "remove"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    contents = target.read_text(encoding="utf-8")
    assert result["success"] is True
    assert "timeout" not in contents
    assert "subprocess.run(['echo', 'two'], check=True)" in contents


def test_ast_patch_update_call_keyword_ambiguous_calls_require_occurrence(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text(
        "import subprocess\n\n"
        "def run():\n"
        "    subprocess.run(['echo', 'one'])\n"
        "    subprocess.run(['echo', 'two'])\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="update_call_keyword",
            target={
                "scope_function": "run",
                "callee": "subprocess.run",
                "keyword": "timeout",
            },
            payload={"mode": "set", "value": "60"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "ast_target_ambiguous"
    assert "subprocess.run@1" in result["metadata"]["candidate_node_names"]


def test_ast_patch_add_dataclass_field_inserts_before_methods(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text(
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class WriteSession:\n"
        "    \"\"\"Session state.\"\"\"\n"
        "    write_session_id: str\n\n"
        "    def is_open(self) -> bool:\n"
        "        return True\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="add_dataclass_field",
            target={"class": "WriteSession", "field": "ast_patch_count"},
            payload={"annotation": "int", "default": "0"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert target.read_text(encoding="utf-8") == (
        "from dataclasses import dataclass\n\n"
        "@dataclass\n"
        "class WriteSession:\n"
        "    \"\"\"Session state.\"\"\"\n"
        "    write_session_id: str\n"
        "    ast_patch_count: int = 0\n\n"
        "    def is_open(self) -> bool:\n"
        "        return True\n"
    )


def test_ast_patch_add_dataclass_field_ignore_existing_is_noop(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    original = (
        "class WriteSession:\n"
        "    ast_patch_count: int = 0\n"
    )
    target.write_text(original, encoding="utf-8")

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="add_dataclass_field",
            target={"class": "WriteSession", "field": "ast_patch_count"},
            payload={"annotation": "int", "default": "0", "if_exists": "ignore"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["changed"] is False
    assert target.read_text(encoding="utf-8") == original


def test_ast_patch_ambiguous_function_target_returns_error(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text(
        "class Alpha:\n"
        "    def run(self):\n"
        "        return 1\n\n"
        "class Beta:\n"
        "    def run(self):\n"
        "        return 2\n",
        encoding="utf-8",
    )

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="replace_function",
            target={"function": "run"},
            payload={"source": "def run(self):\n    return 3\n", "replace": "entire_node"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "ast_target_ambiguous"


def test_ast_patch_invalid_replacement_returns_replacement_parse_failed(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("def run():\n    return 1\n", encoding="utf-8")

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="replace_function",
            target={"function": "run"},
            payload={"source": "def run(:\n    return 2\n", "replace": "entire_node"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "replacement_parse_failed"


def test_ast_patch_invalid_source_returns_parse_failed(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("def run(:\n    return 1\n", encoding="utf-8")

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="insert_in_function",
            target={"function": "run", "position": "start"},
            payload={"statements": "value = 2"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "parse_failed"


def test_ast_patch_unsupported_language_returns_clear_failure(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.js"
    target.write_text("function run() { return 1; }\n", encoding="utf-8")

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="javascript",
            operation="insert_in_function",
            target={"function": "run", "position": "start"},
            payload={"statements": "const value = 2;"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "unsupported_language"
    assert result["metadata"]["supported_languages"] == ["python"]


def test_ast_patch_dry_run_does_not_mutate_file(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    original = "def run():\n    return 1\n"
    target.write_text(original, encoding="utf-8")

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="insert_in_function",
            target={"function": "run", "position": "start"},
            payload={"statements": "value = 2"},
            cwd=str(tmp_path),
            state=state,
            dry_run=True,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["dry_run"] is True
    assert result["metadata"]["changed"] is True
    assert target.read_text(encoding="utf-8") == original
    assert state.files_changed_this_cycle == []


def test_ast_patch_active_write_session_updates_staged_copy_only(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "session.py"
    session = _make_open_write_session(target)
    state.write_session = session

    seeded = asyncio.run(
        fs.file_write(
            path=str(target),
            content="def run():\n    return 1\n",
            cwd=str(tmp_path),
            state=state,
            write_session_id=session.write_session_id,
            section_name="body",
        )
    )
    assert seeded["success"] is True

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="insert_in_function",
            target={"function": "run", "position": "start"},
            payload={"statements": "value = 2"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert result["metadata"]["staged_only"] is True
    assert result["metadata"]["write_session_id"] == session.write_session_id
    assert target.exists() is False
    assert Path(session.write_staging_path).read_text(encoding="utf-8") == (
        "def run():\n"
        "    value = 2\n"
        "    return 1\n"
    )


def test_ast_patch_direct_mutation_records_file_change(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    target = tmp_path / "example.py"
    target.write_text("def run():\n    return 1\n", encoding="utf-8")

    result = asyncio.run(
        fs.ast_patch(
            path=str(target),
            language="python",
            operation="insert_in_function",
            target={"function": "run", "position": "start"},
            payload={"statements": "value = 2"},
            cwd=str(tmp_path),
            state=state,
        )
    )

    assert result["success"] is True
    assert str(target.resolve()) in state.files_changed_this_cycle
