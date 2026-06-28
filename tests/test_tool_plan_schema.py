from __future__ import annotations

from smallctl.graph.tool_plan_schema import (
    MUTATING_TOOL_PLAN_BLOCKLIST,
    PARALLELIZABLE_TOOL_PLAN_TOOLS,
    READONLY_TOOL_PLAN_TOOLS,
    TOOL_PLAN_ALIASES,
    ToolPlan,
    ToolPlanStep,
)


def test_readonly_tool_plan_tools_contains_expected_primitives() -> None:
    assert "file_read" in READONLY_TOOL_PLAN_TOOLS
    assert "dir_list" in READONLY_TOOL_PLAN_TOOLS
    assert "grep" in READONLY_TOOL_PLAN_TOOLS
    assert "find_files" in READONLY_TOOL_PLAN_TOOLS
    assert "artifact_read" in READONLY_TOOL_PLAN_TOOLS
    assert "artifact_grep" in READONLY_TOOL_PLAN_TOOLS
    assert "web_search" in READONLY_TOOL_PLAN_TOOLS
    assert "web_fetch" in READONLY_TOOL_PLAN_TOOLS
    assert "ssh_file_read" in READONLY_TOOL_PLAN_TOOLS
    assert "ssh_dir_list" in READONLY_TOOL_PLAN_TOOLS


def test_parallelizable_tool_plan_tools_is_positive_allowlist() -> None:
    assert "ssh_file_read" in PARALLELIZABLE_TOOL_PLAN_TOOLS
    assert "shell_exec" not in PARALLELIZABLE_TOOL_PLAN_TOOLS
    assert "file_write" not in PARALLELIZABLE_TOOL_PLAN_TOOLS
    assert READONLY_TOOL_PLAN_TOOLS == set(PARALLELIZABLE_TOOL_PLAN_TOOLS)


def test_mutating_blocklist_contains_write_and_exec_tools() -> None:
    assert "file_write" in MUTATING_TOOL_PLAN_BLOCKLIST
    assert "file_patch" in MUTATING_TOOL_PLAN_BLOCKLIST
    assert "ast_patch" in MUTATING_TOOL_PLAN_BLOCKLIST
    assert "shell_exec" in MUTATING_TOOL_PLAN_BLOCKLIST
    assert "ssh_exec" in MUTATING_TOOL_PLAN_BLOCKLIST
    assert "task_complete" in MUTATING_TOOL_PLAN_BLOCKLIST
    assert "ask_human" in MUTATING_TOOL_PLAN_BLOCKLIST
    assert "memory_update" in MUTATING_TOOL_PLAN_BLOCKLIST
    assert "log_note" in MUTATING_TOOL_PLAN_BLOCKLIST


def test_tool_plan_aliases_map_search_and_read_variants() -> None:
    assert TOOL_PLAN_ALIASES["search"] == "grep"
    assert TOOL_PLAN_ALIASES["repo_search"] == "grep"
    assert TOOL_PLAN_ALIASES["grep_search"] == "grep"
    assert TOOL_PLAN_ALIASES["read_file"] == "file_read"
    assert TOOL_PLAN_ALIASES["list_dir"] == "dir_list"
    assert TOOL_PLAN_ALIASES["fetch_url"] == "web_fetch"


def test_tool_plan_step_defaults() -> None:
    step = ToolPlanStep(id="E1", tool="file_read", args={"path": "x.py"})
    assert step.reason == ""
    assert step.depends_on == []
    assert step.optional is False


def test_tool_plan_defaults() -> None:
    plan = ToolPlan(mode="tool_plan", objective="test", steps=[])
    assert plan.max_steps == 6


def test_tool_plan_step_with_all_fields() -> None:
    step = ToolPlanStep(
        id="E2",
        tool="grep",
        args={"pattern": "foo", "path": "src"},
        reason="find foo",
        depends_on=["E1"],
        optional=True,
    )
    assert step.id == "E2"
    assert step.tool == "grep"
    assert step.args == {"pattern": "foo", "path": "src"}
    assert step.reason == "find foo"
    assert step.depends_on == ["E1"]
    assert step.optional is True
