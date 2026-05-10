from __future__ import annotations

from types import SimpleNamespace

from smallctl.graph.tool_plan_safety import validate_tool_plan
from smallctl.graph.tool_plan_schema import ToolPlan, ToolPlanStep


def _harness(tmp_path, names: list[str] | None = None):
    return SimpleNamespace(
        state=SimpleNamespace(cwd=str(tmp_path)),
        registry=SimpleNamespace(names=lambda: names or ["file_read", "dir_list", "grep", "find_files", "web_fetch"]),
    )


def test_validate_tool_plan_allows_workspace_read_tools(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect",
        steps=[
            ToolPlanStep("E1", "file_read", {"path": "src/app.py"}),
            ToolPlanStep("E2", "grep", {"path": "src", "pattern": "dispatch"}),
        ],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path))

    assert safe is plan
    assert errors == []


def test_validate_tool_plan_blocks_absolute_and_parent_paths(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="inspect",
        steps=[
            ToolPlanStep("E1", "file_read", {"path": "/etc/passwd"}),
            ToolPlanStep("E2", "grep", {"path": "../outside", "pattern": "x"}),
        ],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path))

    assert safe is None
    assert len(errors) == 2


def test_validate_tool_plan_blocks_web_when_disabled(tmp_path) -> None:
    plan = ToolPlan(
        mode="tool_plan",
        objective="fetch",
        steps=[ToolPlanStep("E1", "web_fetch", {"url": "https://example.com"})],
    )

    safe, errors = validate_tool_plan(plan, harness=_harness(tmp_path), allow_web=False)

    assert safe is None
    assert "web tools are disabled" in errors[0]

