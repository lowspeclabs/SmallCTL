from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import smallctl.graph.progress_guard_constants as progress_guard_constants
import smallctl.harness.task_intent as task_intent
import smallctl.harness.tool_visibility as tool_visibility
import smallctl.memory_namespace as memory_namespace
from smallctl.phases import _PHASE_CONTRACTS
from smallctl.state import LoopState
from smallctl.tools.dispatcher import ToolDispatcher
from smallctl.tools.fs import file_append as file_append_handler
from smallctl.tools.fs_sessions import _normalize_replace_strategy
from smallctl.tools.register import build_registry
from smallctl.write_session_fsm import new_write_session

_REPO_ROOT = Path(__file__).resolve().parents[1]

# M4: these names had prompt/gate references but no registered handler. They
# must stay purged from every owned name-list.
_DEAD_TOOL_NAMES = ("bash_exec", "long_context_lookup", "summarize_report")

_OWNED_SOURCE_FILES = (
    "src/smallctl/memory_namespace.py",
    "src/smallctl/graph/tool_loop_guards.py",
    "src/smallctl/graph/tool_loop_guards_support.py",
    "src/smallctl/graph/tool_loop_guard_progress.py",
    "src/smallctl/harness/tool_visibility.py",
    "src/smallctl/harness/task_intent.py",
    "src/smallctl/graph/progress_guard_constants.py",
    "src/smallctl/tools/fs_mutations.py",
    "src/smallctl/tools/register_filesystem.py",
    "src/smallctl/tools/fs_sessions.py",
)

# Names that appear in owned name-lists for intent routing/legacy alias
# matching only; they are not dispatchable registry tools by design.
_LEGACY_INTENT_ALIASES = {
    "read_file",
    "write_file",
    "search",
    "artifact_list",
    "ssh_session_send_and_read",
}

# Backticked tokens in prompts.py that are argument names, literals, or prose,
# not tool references.
_PROMPTS_NON_TOOL_TOKENS = {
    "main",
    "content_preview",
    "section_name",
    "start_line",
    "sudo",
}

_PROMPTS_CALL_REFERENCE_RE = re.compile(r"`([a-z][a-z0-9_]+)\(")
_PROMPTS_VERB_REFERENCE_RE = re.compile(
    r"(?:call|calls|called|use|using|via|prefer|invoke|run|emit|retry|try|with)"
    r"\s+`([a-z][a-z0-9_]+)`"
)


def _build_registry(state: LoopState):
    provider = SimpleNamespace(state=state, log=logging.getLogger("test.phase3"))
    return build_registry(provider)


def _build_dispatcher(state: LoopState) -> ToolDispatcher:
    registry = _build_registry(state)
    return ToolDispatcher(registry, state=state, phase="execute")


def _registry_names() -> set[str]:
    state = LoopState(cwd="/tmp")
    return set(_build_registry(state).names())


def _owned_name_lists() -> dict[str, set[str]]:
    lists: dict[str, set[str]] = {
        "memory_namespace._LOCAL_SHELL_TOOL_NAMES": set(memory_namespace._LOCAL_SHELL_TOOL_NAMES),
        "memory_namespace._REMOTE_TOOL_NAMES": set(memory_namespace._REMOTE_TOOL_NAMES),
        "memory_namespace._CODING_TOOL_NAMES": set(memory_namespace._CODING_TOOL_NAMES),
        "memory_namespace._READ_TOOL_NAMES": set(memory_namespace._READ_TOOL_NAMES),
        "memory_namespace._PLANNING_TOOL_NAMES": set(memory_namespace._PLANNING_TOOL_NAMES),
        "memory_namespace._TERMINAL_TOOL_NAMES": set(memory_namespace._TERMINAL_TOOL_NAMES),
        "tool_visibility._INDEX_QUERY_TOOL_NAMES": set(tool_visibility._INDEX_QUERY_TOOL_NAMES),
        "tool_visibility._ARTIFACT_TOOL_NAMES": set(tool_visibility._ARTIFACT_TOOL_NAMES),
        "tool_visibility._PLAN_DEPENDENT_TOOL_NAMES": set(tool_visibility._PLAN_DEPENDENT_TOOL_NAMES),
        "tool_visibility._BACKGROUND_JOB_TOOL_NAMES": set(tool_visibility._BACKGROUND_JOB_TOOL_NAMES),
        "tool_visibility._LOCAL_CODING_SSH_TOOLS": set(tool_visibility._LOCAL_CODING_SSH_TOOLS),
        "tool_visibility._RETRYABLE_HIDDEN_TOOL_NAMES": set(tool_visibility._RETRYABLE_HIDDEN_TOOL_NAMES),
        "tool_visibility._READ_ONLY_LOOP_HIDDEN_TOOLS": set(tool_visibility._READ_ONLY_LOOP_HIDDEN_TOOLS),
        "tool_visibility._REMOTE_ONLY_CONTINUE_ESSENTIAL_TOOLS": set(tool_visibility._REMOTE_ONLY_CONTINUE_ESSENTIAL_TOOLS),
        "task_intent._READONLY_INTENT_TOOLS": set(task_intent._READONLY_INTENT_TOOLS),
        "task_intent._EXECUTION_INTENT_TOOLS": set(task_intent._EXECUTION_INTENT_TOOLS),
        "task_intent._MUTATION_INTENT_TOOLS": set(task_intent._MUTATION_INTENT_TOOLS),
        "progress_guard_constants._MUTATION_TOOLS": set(progress_guard_constants._MUTATION_TOOLS),
        "progress_guard_constants._READ_TOOLS": set(progress_guard_constants._READ_TOOLS),
        "progress_guard_constants._PATCH_META_TOOLS": set(progress_guard_constants._PATCH_META_TOOLS),
        "progress_guard_constants._READ_ONLY_LOOP_TOOLS": set(progress_guard_constants._READ_ONLY_LOOP_TOOLS),
    }
    requested = {
        str(intent).removeprefix("requested_")
        for intent in memory_namespace._INTENT_NAMESPACE_OVERRIDES
        if str(intent).startswith("requested_")
    }
    lists["memory_namespace._INTENT_NAMESPACE_OVERRIDES"] = requested
    return lists


# --- M4: file_append is registered and performs real appends -----------------


def test_file_append_registered_with_valid_schema(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    registry = _build_registry(state)
    spec = registry.get("file_append")

    assert spec is not None
    assert spec.category == "filesystem"
    assert spec.risk == "high"
    schema = spec.schema
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["path", "content"]
    properties = schema["properties"]
    assert properties["replace_strategy"]["enum"] == ["append", "overwrite"]
    for field in (
        "path",
        "content",
        "write_session_id",
        "section_name",
        "section_id",
        "section_role",
        "next_section_name",
        "expected_followup_verifier",
    ):
        assert field in properties


def test_file_append_exposed_like_file_write(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    registry = _build_registry(state)
    file_write_spec = registry.get("file_write")
    file_append_spec = registry.get("file_append")

    assert file_append_spec.allowed_modes == file_write_spec.allowed_modes
    assert file_append_spec.profiles == file_write_spec.profiles
    assert file_append_spec.risk == file_write_spec.risk
    exported = {
        entry["function"]["name"]
        for entry in registry.export_openai_tools(mode="loop")
    }
    assert "file_append" in exported
    assert "file_write" in exported


def test_file_append_executes_real_append_in_workspace(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _build_dispatcher(state)

    first = asyncio.run(
        dispatcher.dispatch("file_append", {"path": "notes.txt", "content": "line1\n"})
    )
    second = asyncio.run(
        dispatcher.dispatch("file_append", {"path": "notes.txt", "content": "line2\n"})
    )

    assert first.success is True
    assert second.success is True
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "line1\nline2\n"
    assert first.metadata["changed"] is True


# --- M4: dead tool names stay purged ------------------------------------------


def test_dead_tool_names_absent_from_owned_sources() -> None:
    dead_pattern = re.compile(r"\b(?:" + "|".join(_DEAD_TOOL_NAMES) + r")\b")
    for relative in _OWNED_SOURCE_FILES:
        source = (_REPO_ROOT / relative).read_text(encoding="utf-8")
        matches = dead_pattern.findall(source)
        assert not matches, f"{relative} still references dead tool names: {matches}"


def test_dead_tool_names_absent_from_full_source_tree() -> None:
    dead_pattern = re.compile(r"\b(?:" + "|".join(_DEAD_TOOL_NAMES) + r")\b")
    tree_root = _REPO_ROOT / "src" / "smallctl"
    sources = sorted(tree_root.rglob("*.py"))
    assert sources, "expected the full smallctl source tree to be scanned"
    offenders: dict[str, list[str]] = {}
    for path in sources:
        matches = dead_pattern.findall(path.read_text(encoding="utf-8"))
        if matches:
            offenders[str(path.relative_to(_REPO_ROOT))] = sorted(set(matches))
    assert not offenders, f"dead tool names found in source tree: {offenders}"


def test_dead_tool_names_absent_from_owned_name_lists() -> None:
    for label, names in _owned_name_lists().items():
        intersection = names & set(_DEAD_TOOL_NAMES)
        assert not intersection, f"{label} still lists dead tool names: {intersection}"


def test_dead_tool_names_not_registered(tmp_path: Path) -> None:
    registered = _registry_names()
    for dead in _DEAD_TOOL_NAMES:
        assert dead not in registered


# --- M4: registry cross-check (drift guard) ------------------------------------


def test_phase_contract_blocked_tools_exist_in_registry() -> None:
    registered = _registry_names()
    for phase, contract in _PHASE_CONTRACTS.items():
        for tool_name in contract.blocked_tools:
            assert tool_name in registered, (
                f"phase '{phase}' blocks unregistered tool '{tool_name}'"
            )


def test_owned_name_lists_exist_in_registry() -> None:
    registered = _registry_names()
    allowed = registered | _LEGACY_INTENT_ALIASES
    for label, names in _owned_name_lists().items():
        missing = names - allowed
        assert not missing, f"{label} references unregistered tools: {missing}"


def test_prompts_tool_references_exist_in_registry() -> None:
    source = (_REPO_ROOT / "src" / "smallctl" / "prompts.py").read_text(encoding="utf-8")
    referenced = set(_PROMPTS_CALL_REFERENCE_RE.findall(source))
    referenced |= set(_PROMPTS_VERB_REFERENCE_RE.findall(source))
    referenced -= _PROMPTS_NON_TOOL_TOKENS
    registered = _registry_names()
    missing = referenced - registered
    assert not missing, f"prompts.py references unregistered tools: {sorted(missing)}"


# --- L9: replace_strategy enum/omission enforcement ----------------------------


def test_l9_standalone_omission_rejects_with_typed_error(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _build_dispatcher(state)

    result = asyncio.run(
        dispatcher.dispatch("file_write", {"path": "solo.txt", "content": "x = 1\n"})
    )

    assert result.success is False
    assert result.metadata["error_kind"] == "replace_strategy_required_standalone"
    assert "replace_strategy" in str(result.error)
    assert not (tmp_path / "solo.txt").exists()


def test_l9_non_enum_values_reject_for_file_write(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _build_dispatcher(state)

    for bad_value in ("replace", "rewrite", "auto"):
        result = asyncio.run(
            dispatcher.dispatch(
                "file_write",
                {"path": "solo.txt", "content": "x = 1\n", "replace_strategy": bad_value},
            )
        )
        assert result.success is False, bad_value
        assert "replace_strategy" in str(result.error)
        assert not (tmp_path / "solo.txt").exists(), bad_value


def test_l9_non_enum_values_reject_for_file_append_handler(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    target = tmp_path / "notes.txt"
    target.write_text("original\n", encoding="utf-8")

    result = asyncio.run(
        file_append_handler(
            str(target),
            "extra\n",
            cwd=str(tmp_path),
            state=state,
            replace_strategy="rewrite",
        )
    )

    assert result["success"] is False
    assert result["metadata"]["error_kind"] == "invalid_replace_strategy"
    assert result["metadata"]["allowed_values"] == ["append", "overwrite"]
    assert target.read_text(encoding="utf-8") == "original\n"


def test_l9_active_session_omission_succeeds(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _build_dispatcher(state)
    target = tmp_path / "app.py"
    session = new_write_session(
        session_id="ws-phase3",
        target_path=str(target),
        intent="replace_file",
    )
    session.write_next_section = "helpers"
    state.write_session = session

    result = asyncio.run(
        dispatcher.dispatch(
            "file_write",
            {
                "path": str(target),
                "content": "def helper():\n    return 1\n",
                "section_name": "helpers",
            },
        )
    )

    assert result.success is True
    assert result.metadata["write_session_id"] == "ws-phase3"


def test_l9_session_id_match_omission_succeeds(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _build_dispatcher(state)
    target = tmp_path / "other.py"
    state.write_session = new_write_session(
        session_id="ws-id-match",
        target_path=str(target),
        intent="replace_file",
    )

    result = asyncio.run(
        dispatcher.dispatch(
            "file_write",
            {
                "path": str(target),
                "content": "full content\n",
                "write_session_id": "ws-id-match",
                "section_name": "full_file",
            },
        )
    )

    assert result.success is True


def test_l9_enum_values_still_accepted(tmp_path: Path) -> None:
    state = LoopState(cwd=str(tmp_path))
    dispatcher = _build_dispatcher(state)

    written = asyncio.run(
        dispatcher.dispatch(
            "file_write",
            {"path": "solo.txt", "content": "x = 1\n", "replace_strategy": "overwrite"},
        )
    )
    appended = asyncio.run(
        dispatcher.dispatch(
            "file_write",
            {"path": "other.txt", "content": "y = 2\n", "replace_strategy": "append"},
        )
    )

    assert written.success is True
    assert appended.success is True
    assert (tmp_path / "solo.txt").read_text(encoding="utf-8") == "x = 1\n"
    assert (tmp_path / "other.txt").read_text(encoding="utf-8") == "y = 2\n"


def test_l9_normalizer_no_longer_accepts_alias_values() -> None:
    assert _normalize_replace_strategy("append") == "append"
    assert _normalize_replace_strategy("overwrite") == "overwrite"
    assert _normalize_replace_strategy(None) == "auto"
    for alias in ("replace", "rewrite", "auto", "bogus"):
        assert _normalize_replace_strategy(alias) == "auto", alias
