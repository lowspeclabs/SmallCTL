from __future__ import annotations

import json
from typing import Any

from ..redaction import redact_sensitive_data, redact_sensitive_text
from ..state_support import json_safe_value


def build_escalation_packet(
    harness: Any,
    *,
    reason: str,
    question: str,
    requested_output: str,
    risk_level: str,
    trigger: str,
    max_chars: int,
    redact_secrets: bool = True,
) -> dict[str, Any]:
    state = harness.state
    scratchpad = getattr(state, "scratchpad", {})
    scratchpad = scratchpad if isinstance(scratchpad, dict) else {}
    run_brief = getattr(state, "run_brief", None)
    ledger = getattr(state, "subtask_ledger", None)
    active_subtask = ledger.active() if ledger is not None and hasattr(ledger, "active") else None

    packet: dict[str, Any] = {
        "role": "smallctl_escalation_packet",
        "task": {
            "original": getattr(run_brief, "original_task", "") if run_brief else "",
            "effective": getattr(run_brief, "effective_task", "") if run_brief else "",
            "phase": getattr(state, "current_phase", ""),
            "step_count": getattr(state, "step_count", 0),
        },
        "caller": {
            "reason": reason,
            "question": question,
            "requested_output": requested_output,
            "risk_level": risk_level,
            "trigger": trigger,
        },
        "active_subtask": json_safe_value(active_subtask),
        "latest_evidence": {
            "last_failure_class": getattr(state, "last_failure_class", ""),
            "last_verifier_verdict": json_safe_value(getattr(state, "last_verifier_verdict", None)),
            "recent_errors": list(getattr(state, "recent_errors", []) or [])[-6:],
            "files_changed_this_cycle": list(getattr(state, "files_changed_this_cycle", []) or [])[-12:],
        },
        "failure_events": json_safe_value(list(getattr(state, "failure_events", []) or [])[-6:]),
        "tool_plan": {
            "observations": str(scratchpad.get("_tool_plan_observations_text") or "")[-4000:],
            "refine_verdict": scratchpad.get("_tool_plan_refine_verdict"),
            "last_solver_draft": str(scratchpad.get("_last_solver_draft") or "")[-2500:],
        },
        "reflexion_lessons": json_safe_value(list(getattr(state, "reflexion_memory", []) or [])[-5:]),
        "recent_tool_records": _recent_tool_records(getattr(state, "tool_execution_records", None)),
        "artifacts": _artifact_summary(getattr(state, "artifacts", None)),
        "constraints": [
            "Advisor cannot execute tools or write files.",
            "Do not suggest bypassing verification, approvals, or credential prompts.",
            "Prefer the smallest safe next evidence-gathering or repair step.",
            "Return JSON only.",
        ],
    }

    packet = json_safe_value(packet)
    if redact_secrets:
        packet = redact_sensitive_data(packet)
        packet = _redact_texts(packet)
    return _budget_packet(packet, max_chars=max_chars)


def _recent_tool_records(records: Any) -> list[dict[str, Any]]:
    if isinstance(records, dict):
        items = list(records.values())
    elif isinstance(records, list):
        items = records
    else:
        items = []
    # Prioritize failed records so the escalation advisor sees problems first
    def _is_failed(record: Any) -> bool:
        if not isinstance(record, dict):
            return False
        status = record.get("status")
        if status in {"failure", "error", "failed"}:
            return True
        result = record.get("result")
        if isinstance(result, dict) and result.get("success") is False:
            return True
        return False

    sorted_items = sorted(items, key=lambda r: (0 if _is_failed(r) else 1))
    summaries: list[dict[str, Any]] = []
    for record in sorted_items[-8:]:
        if not isinstance(record, dict):
            continue
        summaries.append({
            "tool": record.get("tool") or record.get("tool_name"),
            "status": record.get("status"),
            "error": record.get("error"),
            "summary": record.get("summary") or record.get("message"),
        })
    return summaries


def _artifact_summary(artifacts: Any) -> list[dict[str, Any]]:
    if not isinstance(artifacts, dict):
        return []
    out: list[dict[str, Any]] = []
    for artifact_id, artifact in list(artifacts.items())[-8:]:
        safe = json_safe_value(artifact)
        if isinstance(safe, dict):
            out.append({
                "id": artifact_id,
                "path": safe.get("path"),
                "kind": safe.get("kind"),
                "summary": str(safe.get("summary") or "")[:500],
            })
    return out


def _budget_packet(packet: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    max_chars = max(1000, int(max_chars or 1000))
    text = json.dumps(packet, ensure_ascii=True, sort_keys=True)
    if len(text) <= max_chars:
        return packet
    budgeted = dict(packet)
    budgeted["reflexion_lessons"] = []
    budgeted["recent_tool_records"] = budgeted.get("recent_tool_records", [])[-4:]
    budgeted["failure_events"] = budgeted.get("failure_events", [])[-3:]
    tool_plan = dict(budgeted.get("tool_plan") or {})
    for key in ("observations", "last_solver_draft"):
        value = str(tool_plan.get(key) or "")
        tool_plan[key] = value[-1200:] if len(value) > 1200 else value
    budgeted["tool_plan"] = tool_plan
    text = json.dumps(budgeted, ensure_ascii=True, sort_keys=True)
    if len(text) > max_chars:
        budgeted["truncated"] = True
        budgeted["truncation_note"] = f"Packet reduced to fit {max_chars} characters."
        budgeted = _minimal_budget_packet(budgeted, max_chars=max_chars)
    return budgeted


def _minimal_budget_packet(packet: dict[str, Any], *, max_chars: int) -> dict[str, Any]:
    task = dict(packet.get("task") or {})
    caller = dict(packet.get("caller") or {})
    minimal: dict[str, Any] = {
        "role": packet.get("role", "smallctl_escalation_packet"),
        "task": {
            "original": _clip(str(task.get("original") or ""), 500),
            "effective": _clip(str(task.get("effective") or ""), 500),
            "phase": task.get("phase"),
            "step_count": task.get("step_count"),
        },
        "caller": {
            "reason": _clip(str(caller.get("reason") or ""), 500),
            "question": _clip(str(caller.get("question") or ""), 500),
            "requested_output": caller.get("requested_output"),
            "risk_level": caller.get("risk_level"),
            "trigger": caller.get("trigger"),
        },
        "latest_evidence": packet.get("latest_evidence", {}),
        "constraints": packet.get("constraints", []),
        "truncated": True,
        "truncation_note": f"Packet reduced to fit {max_chars} characters.",
    }
    while len(json.dumps(minimal, ensure_ascii=True, sort_keys=True)) > max_chars:
        latest = minimal.get("latest_evidence")
        if latest:
            minimal["latest_evidence"] = {}
            continue
        task_payload = minimal.get("task")
        caller_payload = minimal.get("caller")
        if isinstance(task_payload, dict) and len(str(task_payload.get("original") or "")) > 80:
            task_payload["original"] = _clip(str(task_payload.get("original") or ""), 80)
            task_payload["effective"] = _clip(str(task_payload.get("effective") or ""), 80)
            continue
        if isinstance(caller_payload, dict) and len(str(caller_payload.get("question") or "")) > 80:
            caller_payload["reason"] = _clip(str(caller_payload.get("reason") or ""), 80)
            caller_payload["question"] = _clip(str(caller_payload.get("question") or ""), 80)
            continue
        if minimal.get("constraints"):
            minimal["constraints"] = []
            continue
        break
    return minimal


def _clip(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 14)].rstrip() + " [truncated]"


def _redact_texts(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _redact_texts(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_texts(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value
