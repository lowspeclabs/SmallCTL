from __future__ import annotations

from typing import Any

from ..guards import is_seven_b_or_under_model_name
from ..models.events import UIEvent, UIEventType
from ..state import WorkingMemory
from ..write_session_fsm import archive_terminal_write_session


def _apply_continue_task_state_reset(harness: Any, *, task: str, resolved_task: str) -> None:
    archived_session = archive_terminal_write_session(
        harness.state,
        reason="continue_like_task",
    )
    harness._runlog(
        "task_continue",
        "continuing current task, skipping state reset",
        raw_task=task,
        resolved_task=resolved_task[:80] if resolved_task else "",
        old_step_count=harness.state.step_count,
        archived_write_session_id=(
            archived_session.get("write_session_id") if archived_session else ""
        ),
    )
    recent_errors = [str(item or "").strip() for item in (getattr(harness.state, "recent_errors", []) or []) if str(item or "").strip()]
    guard_errors = [item for item in recent_errors if "Guard tripped:" in item]
    if guard_errors:
        capsule = harness.state.scratchpad.setdefault("_guard_trip_recovery_capsule", {})
        if isinstance(capsule, dict):
            capsule.setdefault("created_at_step", int(getattr(harness.state, "step_count", 0) or 0))
            capsule.setdefault("reason", guard_errors[-1])
            capsule.setdefault("goal", resolved_task or task)
            capsule["continued_after_guard"] = True
        harness.state.scratchpad["_continued_after_guard_trip"] = True
        harness.state.recent_errors = []
        harness.state.tool_history = []
        harness.state.stagnation_counters = {}
        harness.state.scratchpad.pop("_tool_attempt_history", None)
    else:
        harness.state.recent_errors = [item for item in recent_errors if "Guard tripped:" not in item]
    harness.state.step_count = 0
    harness.state.inactive_steps = 0
    harness.state.stagnation_counters.pop("no_actionable_progress", None)
    harness.state.scratchpad.pop("_progress_read_history", None)
    harness.state.scratchpad.pop("_progress_ssh_observation_history", None)
    harness.state.scratchpad.pop("_progress_prior_verdict", None)
    harness.state.scratchpad.pop("_progress_prior_plan_step", None)
    harness.state.scratchpad.pop("_ssh_auth_recovery_state", None)
    harness.state.repair_cycle_id = ""
    harness.state.scratchpad.pop("_repair_cycle_reads", None)
    fama_state = harness.state.scratchpad.get("_fama_state")
    if isinstance(fama_state, dict):
        for key in list(fama_state.keys()):
            if key.startswith("_fama_loop_guard_suppression_count:") or key.startswith("_fama_verifier_suppression_count:"):
                fama_state.pop(key, None)

    # On a continue/proceed after a terminal outcome, reset enough state that
    # the prompt does not bloat with stale repair context, exposed tools, and
    # conversation history. The task goal is preserved in the run_brief.
    harness.state.current_phase = getattr(harness, "_initial_phase", "explore")
    harness.state.task_exposed_tools = set()
    harness.state.recent_errors = []
    harness.state.tool_history = []
    harness.state.stagnation_counters = {}
    harness.state.scratchpad.pop("_tool_attempt_history", None)

    recent_messages = list(getattr(harness.state, "recent_messages", []) or [])
    if len(recent_messages) > 2:
        harness.state.recent_messages = recent_messages[-2:]

    harness.state.reasoning_graph.evidence_records = []
    harness.state.context_briefs = []
    harness.state.episodic_summaries = []

    # Discard persisted warm context that otherwise bloats the prompt budget on
    # a fresh continuation. The task goal is preserved in the run_brief.
    harness.state.scratchpad.pop("_fresh_tool_outputs", None)
    harness.state.warm_experiences = []
    current_goal = str(
        getattr(harness.state, "working_memory", None) and harness.state.working_memory.current_goal or ""
    ).strip()
    harness.state.working_memory = WorkingMemory(current_goal=current_goal)
    harness._runlog(
        "task_continue_warm_context_cleared",
        "cleared persisted warm context on continue-like follow-up",
        raw_task=task,
        resolved_task=resolved_task[:80] if resolved_task else "",
    )


def _resolve_followup_task(harness: Any, task: str) -> tuple[str, bool]:
    resolved_task = task
    resolve_followup = getattr(harness, "_resolve_followup_task", None)
    if callable(resolve_followup):
        candidate = str(resolve_followup(task) or "").strip()
        if candidate:
            resolved_task = candidate
    is_continue_check = getattr(harness, "_is_continue_like_followup", None)
    is_continue_task = callable(is_continue_check) and is_continue_check(task)
    return resolved_task, bool(is_continue_task)


async def _handle_cancel_requested(graph_state: Any, deps: Any) -> bool:
    harness = deps.harness
    if not getattr(harness, "_cancel_requested", False):
        return False
    await harness._emit(
        deps.event_handler,
        UIEvent(event_type=UIEventType.SYSTEM, content="Run cancelled."),
    )
    graph_state.final_result = {"status": "cancelled", "reason": "cancel_requested"}
    return True


def _initialize_chat_mode_scratchpad(harness: Any, run_mode: str) -> None:
    if run_mode == "chat":
        harness.state.scratchpad["_chat_rounds"] = 0
        harness.state.scratchpad.pop("_chat_progress_guard", None)
    else:
        harness.state.scratchpad.pop("_chat_rounds", None)
        harness.state.scratchpad.pop("_chat_progress_guard", None)


def _apply_small_model_remote_constraints(harness: Any, resolved_task: str) -> None:
    model_name = str(
        getattr(harness.state, "scratchpad", {}).get("_model_name")
        or getattr(getattr(harness, "client", None), "model", "")
        or ""
    ).strip()
    if not is_seven_b_or_under_model_name(model_name):
        return
    active_profiles = set(getattr(harness.state, "active_tool_profiles", []) or [])
    task_lower = resolved_task.lower()
    is_remote = any(k in task_lower for k in ("ssh", "remote", "server", "host"))
    is_compound = any(k in task_lower for k in (
        "inspect", "compare", "determine", "analyze", "summarize",
        "write", "generate", "create", "build", "report",
    ))
    if is_remote and is_compound:
        harness.state.run_brief.constraints = list(
            dict.fromkeys(
                harness.state.run_brief.constraints
                + ["compound_remote_task: break into gather→analyze→synthesize phases"]
            )
        )
    if active_profiles & {"network", "network_read"}:
        harness.state.run_brief.constraints = list(
            dict.fromkeys(
                harness.state.run_brief.constraints
                + ["batch_ssh_discovery: combine multiple commands into one ssh_exec using && or ;"]
            )
        )
