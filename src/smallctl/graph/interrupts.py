from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage
from ..models.events import UIEvent, UIEventType
from ..state import json_safe_value
from ..tools.planning import plan_request_execution
from .deps import GraphRuntimeDeps
from .state import GraphRunState, ToolExecutionRecord


async def pause_for_plan_approval(
    graph_state: GraphRunState,
    deps: GraphRuntimeDeps,
    *,
    question: str = "Plan ready. Execute it now?",
) -> None:
    harness = deps.harness
    plan = harness.state.active_plan or harness.state.draft_plan
    if plan is None:
        return
    await plan_request_execution(question=question, state=harness.state, harness=harness)
    payload = harness.state.pending_interrupt or {
        "kind": "plan_execute_approval",
        "question": question,
        "plan_id": plan.plan_id,
        "approved": False,
        "response_mode": "yes/no/revise",
        "current_phase": harness.state.current_phase,
        "thread_id": graph_state.thread_id,
    }
    graph_state.interrupt_payload = payload
    await harness._emit(
        deps.event_handler,
        UIEvent(
            event_type=UIEventType.ALERT,
            content=question,
            data={"status_activity": "awaiting plan approval...", "interrupt": payload},
        ),
    )
    graph_state.final_result = {
        "status": "needs_human",
        "message": question,
        "assistant": graph_state.last_assistant_text,
        "thinking": graph_state.last_thinking_text,
        "usage": graph_state.last_usage,
        "interrupt": payload,
    }


def build_interrupt_payload(
    *,
    harness: Any,
    graph_state: GraphRunState,
    record: ToolExecutionRecord,
) -> dict[str, Any]:
    question = ""
    if isinstance(record.result.output, dict):
        question = str(record.result.output.get("question", "")).strip()
    if not question:
        question = str(harness.state.scratchpad.get("_ask_human_question", "")).strip()
    recent_tool_summary = []
    for item in graph_state.last_tool_results[-3:]:
        summary = {
            "tool_name": item.tool_name,
            "success": item.result.success,
            "replayed": item.replayed,
        }
        if item.result.error:
            summary["error"] = item.result.error
        elif isinstance(item.result.output, dict):
            summary["output"] = {
                key: value
                for key, value in item.result.output.items()
                if key in {"status", "message", "question"}
            }
        recent_tool_summary.append(summary)
    return {
        "kind": "ask_human",
        "question": question,
        "current_phase": harness.state.current_phase,
        "active_profiles": list(harness.state.active_tool_profiles),
        "thread_id": graph_state.thread_id,
        "operation_id": record.operation_id,
        "recent_tool_outcomes": recent_tool_summary,
    }
