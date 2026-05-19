from __future__ import annotations

import pytest

from smallctl.graph.tool_dag_safety import (
    MutatingStepInDAGError,
    NonParallelizableStepInDAGError,
    assert_no_mutating_steps,
    assert_parallelizable_steps,
)
from smallctl.graph.tool_plan_schema import ToolPlanStep


def test_assert_parallelizable_steps_passes_for_allowlisted_tools() -> None:
    batches = [
        [
            ToolPlanStep(id="E1", tool="file_read", args={"path": "a.py"}),
            ToolPlanStep(id="E2", tool="grep", args={"pattern": "foo"}),
            ToolPlanStep(id="E3", tool="ssh_file_read", args={"path": "/etc/hosts", "target": "root@example"}),
            ToolPlanStep(id="E4", tool="git_status", args={"path": "."}),
            ToolPlanStep(id="E5", tool="git_diff", args={"path": "."}),
            ToolPlanStep(id="E6", tool="read_log", args={"path": "app.log"}),
        ]
    ]
    assert_parallelizable_steps(batches)  # should not raise


def test_assert_parallelizable_steps_blocks_unknown_readish_tool() -> None:
    batches = [
        [
            ToolPlanStep(id="E1", tool="read_journal", args={"path": "app.log"}),
        ]
    ]
    with pytest.raises(NonParallelizableStepInDAGError) as exc_info:
        assert_parallelizable_steps(batches)
    assert "read_journal" in str(exc_info.value)
    assert "E1" in str(exc_info.value)


def test_assert_parallelizable_steps_blocks_file_write() -> None:
    batches = [
        [
            ToolPlanStep(id="E1", tool="file_read", args={"path": "a.py"}),
            ToolPlanStep(id="E2", tool="file_write", args={"path": "b.py", "content": "x"}),
        ]
    ]
    with pytest.raises(NonParallelizableStepInDAGError) as exc_info:
        assert_parallelizable_steps(batches)
    assert "file_write" in str(exc_info.value)
    assert "E2" in str(exc_info.value)


def test_assert_parallelizable_steps_blocks_shell_exec() -> None:
    batches = [
        [
            ToolPlanStep(id="E1", tool="shell_exec", args={"command": "ls"}),
        ]
    ]
    with pytest.raises(NonParallelizableStepInDAGError):
        assert_parallelizable_steps(batches)


def test_assert_no_mutating_steps_compatibility_wrapper_still_blocks() -> None:
    batches = [[ToolPlanStep(id="E1", tool="shell_exec", args={"command": "ls"})]]
    with pytest.raises(MutatingStepInDAGError):
        assert_no_mutating_steps(batches)
