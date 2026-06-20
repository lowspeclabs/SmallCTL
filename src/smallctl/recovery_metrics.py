from __future__ import annotations

from typing import Any


RECOVERY_METRICS_KEY = "_recovery_metrics"


def recovery_metrics(state: Any) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    metrics = scratchpad.get(RECOVERY_METRICS_KEY)
    if not isinstance(metrics, dict):
        metrics = {}
        scratchpad[RECOVERY_METRICS_KEY] = metrics
    return metrics


def increment_metric(state: Any, name: str, amount: int = 1) -> None:
    key = str(name or "").strip()
    if not key:
        return
    metrics = recovery_metrics(state)
    if not isinstance(metrics, dict):
        return
    try:
        metrics[key] = int(metrics.get(key, 0) or 0) + int(amount or 0)
    except (TypeError, ValueError):
        metrics[key] = int(amount or 0)


def increment_metric_bucket(state: Any, name: str, bucket: str, amount: int = 1) -> None:
    key = str(name or "").strip()
    bucket_key = str(bucket or "").strip()
    if not key or not bucket_key:
        return
    metrics = recovery_metrics(state)
    if not isinstance(metrics, dict):
        return
    buckets = metrics.get(key)
    if not isinstance(buckets, dict):
        buckets = {}
        metrics[key] = buckets
    try:
        buckets[bucket_key] = int(buckets.get(bucket_key, 0) or 0) + int(amount or 0)
    except (TypeError, ValueError):
        buckets[bucket_key] = int(amount or 0)


def tool_call_repair_issue_signature(tool_name: str, issue_kinds: list[str], repair_kinds: list[str]) -> str:
    issue_part = ",".join(sorted(str(kind) for kind in issue_kinds if str(kind).strip()))
    repair_part = ",".join(sorted(str(kind) for kind in repair_kinds if str(kind).strip()))
    return f"{str(tool_name or '').strip()}|issues={issue_part}|repairs={repair_part}"


def record_tool_call_repair_metrics(state: Any, *, repair_kinds: list[str], hint_injected: bool = False) -> None:
    if repair_kinds:
        increment_metric(state, "tool_call_repairs_total")
    for kind in repair_kinds:
        increment_metric_bucket(state, "tool_call_repairs_by_kind", kind)
    if hint_injected:
        increment_metric(state, "tool_call_repair_hints_injected_total")


def remember_tool_call_repair_hint(
    state: Any,
    *,
    tool_name: str,
    tool_call_id: str | None,
    step_count: int,
    issue_kinds: list[str],
    repair_kinds: list[str],
    repaired_args_preview: Any,
) -> None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    scratchpad["_last_tool_call_repair_hint"] = {
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "step_count": int(step_count or 0),
        "repair_kinds": list(repair_kinds),
        "issue_kinds": list(issue_kinds),
        "issue_signature": tool_call_repair_issue_signature(tool_name, issue_kinds, repair_kinds),
        "repaired_args_preview": repaired_args_preview,
    }


def record_tool_call_repair_next_call_signal(
    state: Any,
    *,
    tool_name: str,
    issue_kinds: list[str],
    repair_kinds: list[str],
    max_step_window: int = 4,
) -> str:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return ""
    pending = scratchpad.get("_last_tool_call_repair_hint")
    if not isinstance(pending, dict):
        return ""
    pending_tool = str(pending.get("tool_name") or "").strip()
    if pending_tool != str(tool_name or "").strip():
        return ""
    current_step = _safe_int(getattr(state, "step_count", 0), default=0)
    created_step = _safe_int(pending.get("step_count"), default=current_step)
    if current_step - created_step > max_step_window:
        scratchpad.pop("_last_tool_call_repair_hint", None)
        return "expired"

    previous_signature = str(pending.get("issue_signature") or "")
    current_signature = tool_call_repair_issue_signature(tool_name, issue_kinds, repair_kinds)
    if previous_signature and current_signature == previous_signature:
        increment_metric(state, "tool_call_repair_next_call_repeated_total")
        scratchpad.pop("_last_tool_call_repair_hint", None)
        return "repeated"
    increment_metric(state, "tool_call_repair_next_call_improved_total")
    scratchpad.pop("_last_tool_call_repair_hint", None)
    return "improved"


def record_failure_event_metric(state: Any, event: Any) -> None:
    failure_class = str(getattr(event, "failure_class", "") or "").strip()
    increment_metric(state, "failure_events_total")
    increment_metric_bucket(state, "failure_events_by_class", failure_class or "unknown")


def record_terminal_success_metrics(state: Any) -> None:
    events = getattr(state, "failure_events", None)
    if not isinstance(events, list) or not events:
        return
    metrics = recovery_metrics(state)
    if not isinstance(metrics, dict):
        return

    latest_event_id = str(getattr(events[-1], "event_id", "") or "").strip()
    step_count = _safe_int(getattr(state, "step_count", 0), default=0)
    tool_calls = _tool_call_count(state)
    signature = f"{latest_event_id}|{step_count}|{tool_calls}"

    scratchpad = getattr(state, "scratchpad", None)
    if isinstance(scratchpad, dict):
        if scratchpad.get("_last_recovery_success_metric") == signature:
            return
        scratchpad["_last_recovery_success_metric"] = signature

    increment_metric(state, "recovery_success_count")
    if any(str(getattr(event, "failure_class", "") or "") == "human_resteer" for event in events):
        increment_metric(state, "resteer_recovery_success")

    tool_samples = metrics.get("tool_calls_until_success")
    if not isinstance(tool_samples, list):
        tool_samples = []
        metrics["tool_calls_until_success"] = tool_samples
    tool_samples.append(tool_calls)
    del tool_samples[:-20]

    turn_samples = metrics.get("turns_until_success")
    if not isinstance(turn_samples, list):
        turn_samples = []
        metrics["turns_until_success"] = turn_samples
    turn_samples.append(_turns_since_first_failure(events, step_count))
    del turn_samples[:-20]


def _tool_call_count(state: Any) -> int:
    records = getattr(state, "tool_execution_records", None)
    if isinstance(records, dict):
        return len(records)
    if isinstance(records, list):
        return len(records)
    return 0


def _turns_since_first_failure(events: list[Any], current_step: int) -> int:
    steps: list[int] = []
    for event in events:
        metadata = getattr(event, "metadata", None)
        if not isinstance(metadata, dict):
            continue
        step = _safe_int(metadata.get("step"), default=-1)
        if step >= 0:
            steps.append(step)
    if not steps:
        return max(0, current_step)
    return max(0, current_step - min(steps))


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
