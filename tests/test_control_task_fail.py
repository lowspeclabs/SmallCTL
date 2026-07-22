from __future__ import annotations

import asyncio

from smallctl.state import LoopState
from smallctl.tools.control import task_fail


def test_task_fail_rejects_successful_completion_claim() -> None:
    state = LoopState()

    result = asyncio.run(task_fail("Task completed successfully. Deliverables are confirmed.", state))

    assert result["success"] is False
    assert result["metadata"]["reason"] == "task_fail_success_claim"
    assert "_task_failed" not in state.scratchpad


def test_task_fail_accepts_concrete_blocker() -> None:
    state = LoopState()

    result = asyncio.run(
        task_fail("Could not complete installation because the readiness endpoint returns 503.", state)
    )

    assert result["success"] is True
    assert state.scratchpad["_task_failed"] is True
