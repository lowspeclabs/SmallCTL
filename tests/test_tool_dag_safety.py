from __future__ import annotations

import pytest

from smallctl.graph.tool_dag_safety import MutatingStepInDAGError, assert_no_mutating_steps
from smallctl.graph.tool_plan_schema import ToolPlanStep


def test_assert_no_mutating_steps_passes_for_read_only() -> None:
    batches = [
        [
            ToolPlanStep(id="E1", tool="file_read", args={"path": "a.py"}),
            ToolPlanStep(id="E2", tool="grep", args={"pattern": "foo"}),
        ]
    ]
    assert_no_mutating_steps(batches)  # should not raise


def test_assert_no_mutating_steps_raises_on_file_write() -> None:
    batches = [
        [
            ToolPlanStep(id="E1", tool="file_read", args={"path": "a.py"}),
            ToolPlanStep(id="E2", tool="file_write", args={"path": "b.py", "content": "x"}),
        ]
    ]
    with pytest.raises(MutatingStepInDAGError) as exc_info:
        assert_no_mutating_steps(batches)
    assert "file_write" in str(exc_info.value)
    assert "E2" in str(exc_info.value)


def test_assert_no_mutating_steps_raises_on_shell_exec() -> None:
    batches = [
        [
            ToolPlanStep(id="E1", tool="shell_exec", args={"command": "ls"}),
        ]
    ]
    with pytest.raises(MutatingStepInDAGError):
        assert_no_mutating_steps(batches)
