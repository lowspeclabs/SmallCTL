from __future__ import annotations

from ..state import LoopState
from .common import ok


async def task_complete(message: str, state: LoopState) -> dict:
    state.scratchpad["_task_complete"] = True
    state.scratchpad["_task_complete_message"] = message
    state.touch()
    return ok({"status": "complete", "message": message})


async def task_fail(message: str, state: LoopState) -> dict:
    state.scratchpad["_task_failed"] = True
    state.scratchpad["_task_failed_message"] = message
    state.recent_errors.append(message)
    state.touch()
    return ok({"status": "failed", "message": message})


async def ask_human(question: str, state: LoopState) -> dict:
    state.scratchpad["_ask_human"] = True
    state.scratchpad["_ask_human_question"] = question
    state.touch()
    return ok({"status": "human_input_required", "question": question})


async def loop_status(state: LoopState) -> dict:
    return ok(
        {
            "phase": state.current_phase,
            "step_count": state.step_count,
            "token_usage": state.token_usage,
            "elapsed_seconds": state.elapsed_seconds,
            "recent_errors": state.recent_errors[-5:],
            "cwd": state.cwd,
            "active_tool_profiles": state.active_tool_profiles,
        }
    )
