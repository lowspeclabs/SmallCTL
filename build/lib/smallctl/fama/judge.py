from __future__ import annotations

import json
from typing import Any

from ..state_support import json_safe_value
from .config import llm_judge_enabled, llm_judge_min_severity
from .detectors import _has_human_gate
from .signals import FamaFailureKind, FamaSignal, current_step

_PROMPT_CLASS = "fama_failure_triage_v1"
_MAX_OUTPUT_TOKENS = 64


async def maybe_run_llm_judge(
    harness: Any,
    *,
    state: Any,
    config: Any,
    base_signal: FamaSignal,
) -> FamaSignal | None:
    if base_signal.source == "llm_judge":
        return None
    if not llm_judge_enabled(config):
        return None
    if int(base_signal.severity) < llm_judge_min_severity(config):
        return None
    if _has_human_gate(state):
        _log_judge(harness, verdict="skipped", accepted=False, reason="human_gated")
        return None

    client = getattr(harness, "summarizer_client", None) or getattr(harness, "client", None)
    stream_chat = getattr(client, "stream_chat", None)
    if not callable(stream_chat):
        _log_judge(harness, verdict="skipped", accepted=False, reason="client_unavailable")
        return None

    try:
        text = await _call_judge(stream_chat, state=state, base_signal=base_signal)
    except Exception as exc:
        _log_judge(harness, verdict="error", accepted=False, reason=type(exc).__name__)
        return None

    verdict = _parse_verdict(text)
    accepted = verdict is not None and verdict != base_signal.kind
    _log_judge(
        harness,
        verdict=verdict.value if verdict is not None else "none",
        accepted=accepted,
        reason="added_signal" if accepted else "no_new_signal",
        based_on=base_signal.kind.value,
    )
    if not accepted or verdict is None:
        return None
    return FamaSignal(
        kind=verdict,
        severity=base_signal.severity,
        source="llm_judge",
        evidence=f"judge verdict={verdict.value}; based_on={base_signal.kind.value}",
        step=current_step(state),
        tool_name=base_signal.tool_name,
        operation_id=base_signal.operation_id,
    )


async def _call_judge(stream_chat: Any, *, state: Any, base_signal: FamaSignal) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "Classify whether the deterministic failure signal should add another FAMA failure kind. "
                "Return only one known kind or 'none'. Known kinds: "
                + ", ".join(kind.value for kind in FamaFailureKind)
                + ". Do not explain."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "prompt_class": _PROMPT_CLASS,
                    "active_goal": _active_goal(state),
                    "deterministic_signal": {
                        "kind": base_signal.kind.value,
                        "severity": base_signal.severity,
                        "source": base_signal.source,
                        "tool_name": base_signal.tool_name,
                        "evidence": base_signal.evidence[:360],
                    },
                    "recent_tool_results": _recent_tool_results(state),
                },
                ensure_ascii=True,
                sort_keys=True,
            ),
        },
    ]
    chunks: list[dict[str, Any]] = []
    async for event in stream_chat(messages=messages, tools=[]):
        if isinstance(event, dict):
            chunks.append(event)
    return _collect_text(chunks)


def _parse_verdict(text: str) -> FamaFailureKind | None:
    raw = " ".join(str(text or "").strip().split()[:_MAX_OUTPUT_TOKENS]).strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = raw
    if isinstance(parsed, dict):
        parsed = parsed.get("kind") or parsed.get("verdict") or parsed.get("failure_kind")
    verdict = str(parsed or "").strip().lower()
    if verdict in {"", "none", "no_signal", "unknown"}:
        return None
    for kind in FamaFailureKind:
        if verdict == kind.value:
            return kind
    return None


def _collect_text(chunks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in chunks:
        if "assistant_text" in event:
            parts.append(str(event.get("assistant_text") or ""))
            continue
        data = event.get("data") if event.get("type") == "chunk" else event
        if not isinstance(data, dict):
            continue
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            continue
        delta = choices[0].get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "".join(parts)


def _active_goal(state: Any) -> str:
    run_brief = getattr(state, "run_brief", None)
    goal = str(getattr(run_brief, "original_task", "") or getattr(run_brief, "current_goal", "") or "").strip()
    if goal:
        return goal[:500]
    active_plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    return str(getattr(active_plan, "goal", "") or "").strip()[:500]


def _recent_tool_results(state: Any) -> list[dict[str, Any]]:
    records = getattr(state, "tool_execution_records", None)
    if isinstance(records, dict) and records:
        values = [item for item in records.values() if isinstance(item, dict)]
        return [_compact_tool_record(item) for item in values[-3:]]
    history = getattr(state, "tool_history", None)
    if isinstance(history, list):
        return [{"tool_fingerprint": str(item)[:240]} for item in history[-3:]]
    return []


def _compact_tool_record(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    result = result if isinstance(result, dict) else {}
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    compact = {
        "tool_name": str(record.get("tool_name") or "")[:80],
        "success": bool(result.get("success")),
        "status": str(result.get("status") or "")[:80],
        "error": str(result.get("error") or "")[:280],
        "metadata": {
            key: json_safe_value(metadata.get(key))
            for key in ("reason", "verdict", "last_verifier_verdict", "pending_acceptance_criteria")
            if key in metadata
        },
    }
    output = result.get("output")
    if output is not None:
        compact["output_preview"] = json.dumps(json_safe_value(output), ensure_ascii=True, default=str)[:500]
    return compact


def _log_judge(harness: Any, *, verdict: str, accepted: bool, reason: str, based_on: str = "") -> None:
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "fama_llm_judge_verdict",
            "FAMA LLM judge verdict",
            prompt_class=_PROMPT_CLASS,
            verdict=verdict,
            accepted=accepted,
            reason=reason,
            based_on=based_on,
        )
