from __future__ import annotations

import hashlib
from time import time
from typing import Any

from ..recovery_metrics import record_failure_event_metric, recovery_metrics
from ..recovery_schema import FailureEvent
from .tool_plan_helpers import _compact_evidence_text, _usage_token_count


def _record_tool_plan_tokens(state: Any, metric_name: str, usage: dict[str, Any]) -> None:
    tokens = _usage_token_count(usage)
    if tokens <= 0:
        return
    metrics = recovery_metrics(state)
    metrics[metric_name] = int(metrics.get(metric_name, 0) or 0) + tokens
    metrics["tool_plan_total_tokens"] = int(metrics.get("tool_plan_total_tokens", 0) or 0) + tokens


def _add_latency_metric(graph_state: Any, name: str, elapsed_sec: float) -> None:
    current = graph_state.latency_metrics.get(name, 0.0)
    try:
        current_value = float(current or 0.0)
    except (TypeError, ValueError):
        current_value = 0.0
    graph_state.latency_metrics[name] = round(current_value + max(0.0, elapsed_sec), 3)


def _attach_tool_plan_evidence(harness: Any, observations_text: str) -> None:
    config = getattr(harness, "config", None)
    if not bool(getattr(config, "subtask_ledger_enabled", True)):
        return
    evidence = "ToolPlan observations: " + _compact_evidence_text(observations_text, limit=210)
    service = getattr(harness, "subtask_ledger", None)
    if service is not None:
        try:
            service.import_plan_if_needed()
            active = service.infer_or_create_active_subtask()
            service.attach_evidence(active.subtask_id, evidence)
        except Exception:
            pass
    state = getattr(harness, "state", None)
    if state is None:
        return
    try:
        from ..recovery_schema import Subtask, SubtaskLedger

        ledger = getattr(state, "subtask_ledger", None)
        if ledger is None:
            ledger = SubtaskLedger(task_id=str(getattr(state, "thread_id", "") or "tool-plan"))
            state.subtask_ledger = ledger
        active = ledger.active()
        if active is None:
            active = Subtask(subtask_id="tool-plan", title="ToolPlan evidence", goal="Gather bounded evidence", status="active")
            ledger.subtasks.append(active)
            ledger.active_subtask_id = active.subtask_id
        active.evidence.append(evidence)
    except Exception:
        return


def _record_tool_plan_failure(harness: Any, message: str, *, failure_class: str) -> None:
    state = getattr(harness, "state", None)
    if state is None:
        return
    service = getattr(harness, "subtask_ledger", None)
    active_subtask_id = None
    try:
        if bool(getattr(getattr(harness, "config", None), "subtask_ledger_enabled", True)) and service is not None:
            service.import_plan_if_needed()
            active_subtask_id = service.infer_or_create_active_subtask().subtask_id
    except Exception:
        active_subtask_id = None
    raw_id = f"{failure_class}|{message}|{getattr(state, 'step_count', 0)}"
    event = FailureEvent(
        event_id="toolplan-" + hashlib.sha1(raw_id.encode("utf-8", errors="replace")).hexdigest()[:16],
        timestamp=time(),
        failure_class=failure_class,
        severity="warning",
        source="tool_plan",
        message=str(message or "")[:240],
        evidence=[str(message or "")[:240]] if message else [],
        subtask_id=active_subtask_id,
        suggested_next_action="Fall back to normal loop or retry ToolPlan with bounded read-only evidence steps.",
    )
    events = getattr(state, "failure_events", None)
    if isinstance(events, list) and not any(
        isinstance(item, FailureEvent) and item.event_id == event.event_id for item in events[-8:]
    ):
        events.append(event)
        del events[:-40]
        record_failure_event_metric(state, event)
    if service is not None and active_subtask_id:
        try:
            service.attach_failure(active_subtask_id, event)
        except Exception:
            pass
    reflexion = getattr(harness, "reflexion", None)
    maybe_create = getattr(reflexion, "maybe_create_reflection", None)
    if callable(maybe_create):
        try:
            maybe_create(event, getattr(state, "subtask_ledger", None))
        except Exception:
            pass


def _record_tool_plan_planner_metadata(harness: Any, plan: Any) -> None:
    metrics = recovery_metrics(harness.state)
    metrics["tool_plan_planner_valid"] = 1
    metrics["tool_plan_planner_step_count"] = len(plan.steps)
    metrics["tool_plan_planner_tools"] = sorted({str(step.tool or "").strip() for step in plan.steps if str(step.tool or "").strip()})
