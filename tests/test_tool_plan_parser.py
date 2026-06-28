from __future__ import annotations

from smallctl.graph.tool_plan_parser import parse_tool_plan


def test_parse_tool_plan_accepts_fenced_json_and_aliases() -> None:
    plan = parse_tool_plan(
        """```json
        {
          "mode": "tool_plan",
          "objective": "find runtime seams",
          "steps": [
            {"id": "first", "tool": "read_file", "args": {"path": "src/app.py"}},
            {"tool": "search", "args": {"pattern": "dispatch", "path": "src"}}
          ]
        }
        ```"""
    )

    assert plan is not None
    assert plan.objective == "find runtime seams"
    assert [step.id for step in plan.steps] == ["E1", "E2"]
    assert [step.tool for step in plan.steps] == ["file_read", "grep"]


def test_parse_tool_plan_rejects_mutating_tool() -> None:
    assert parse_tool_plan(
        """
        {"mode": "tool_plan", "objective": "bad", "steps": [
          {"id": "E1", "tool": "file_patch", "args": {"path": "x.py"}}
        ]}
        """
    ) is None


def test_parse_tool_plan_rejects_duplicate_ids() -> None:
    assert parse_tool_plan(
        """
        {"mode": "tool_plan", "objective": "bad", "steps": [
          {"id": "E1", "tool": "file_read", "args": {"path": "a.py"}},
          {"id": "E1", "tool": "file_read", "args": {"path": "b.py"}}
        ]}
        """
    ) is None


def test_parse_tool_plan_accepts_empty_steps() -> None:
    plan = parse_tool_plan(
        """
        {"mode": "tool_plan", "objective": "No evidence needed", "steps": []}
        """
    )
    assert plan is not None
    assert plan.objective == "No evidence needed"
    assert plan.steps == []

