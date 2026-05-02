from __future__ import annotations

import json
from typing import Any

from ..harness.tool_result_verification import _classify_execution_failure
from ..state import clip_text_value, json_safe_value
from .tool_execution_support import _get_tool_execution_record


def _clear_chat_progress_guard(harness: Any) -> None:
    harness.state.scratchpad.pop("_chat_progress_guard", None)


def _chat_failure_evidence_excerpt(record: Any) -> str:
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    output_payload = record.result.output if isinstance(record.result.output, dict) else {}
    metadata_output = metadata.get("output")
    if not output_payload and isinstance(metadata_output, dict):
        output_payload = metadata_output

    candidates: list[str] = [
        str(record.result.error or "").strip(),
        str(output_payload.get("stderr") or "").strip(),
        str(output_payload.get("stdout") or "").strip(),
    ]
    if isinstance(record.result.output, str):
        candidates.append(str(record.result.output).strip())
    elif isinstance(record.result.output, dict):
        candidates.append(json.dumps(json_safe_value(record.result.output), ensure_ascii=True, sort_keys=True))

    for candidate in candidates:
        if not candidate:
            continue
        clipped, _ = clip_text_value(candidate, limit=180)
        if clipped:
            return clipped
    return ""


def _chat_failure_signature(record: Any) -> dict[str, str] | None:
    if record.tool_name in {"task_complete", "task_fail", "ask_human"}:
        return None
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if record.result.success:
        return None
    if getattr(record.result, "status", None) == "needs_human" or metadata.get("status") == "needs_human":
        return None

    evidence_excerpt = _chat_failure_evidence_excerpt(record)
    failure_text = " | ".join(
        bit
        for bit in (
            str(record.result.error or "").strip(),
            evidence_excerpt,
        )
        if bit
    )
    failure_class = _classify_execution_failure(failure_text)
    return {
        "tool_name": str(record.tool_name or "").strip(),
        "failure_class": failure_class,
        "evidence_excerpt": evidence_excerpt,
    }


def _record_chat_progress_outcome(harness: Any, records: list[Any]) -> None:
    relevant_records = [record for record in records if record.tool_name not in {"task_complete", "task_fail", "ask_human"}]
    if not relevant_records:
        _clear_chat_progress_guard(harness)
        return

    if any(record.result.success for record in relevant_records):
        _clear_chat_progress_guard(harness)
        return

    failure_record = next((record for record in reversed(relevant_records) if _chat_failure_signature(record) is not None), None)
    if failure_record is None:
        _clear_chat_progress_guard(harness)
        return

    signature_payload = _chat_failure_signature(failure_record)
    if signature_payload is None:
        _clear_chat_progress_guard(harness)
        return

    signature = json.dumps(signature_payload, ensure_ascii=True, sort_keys=True)
    prior = harness.state.scratchpad.get("_chat_progress_guard")
    prior_signature = ""
    prior_stall_count = 0
    if isinstance(prior, dict):
        prior_signature = str(prior.get("signature") or "").strip()
        prior_stall_count = int(prior.get("stall_count", 0) or 0)

    stall_count = prior_stall_count + 1 if signature == prior_signature else 1
    stored = _get_tool_execution_record(harness, failure_record.operation_id)
    artifact_id = str(stored.get("artifact_id") or "").strip()
    harness.state.scratchpad["_chat_progress_guard"] = {
        **signature_payload,
        "signature": signature,
        "stall_count": stall_count,
        "artifact_id": artifact_id,
        "operation_id": str(failure_record.operation_id or ""),
        "tool_call_id": str(failure_record.tool_call_id or ""),
    }


def _chat_progress_guard_failure(harness: Any) -> dict[str, Any] | None:
    raw = harness.state.scratchpad.get("_chat_progress_guard")
    if not isinstance(raw, dict):
        return None

    stall_count = int(raw.get("stall_count", 0) or 0)
    if stall_count < 2:
        return None

    tool_name = str(raw.get("tool_name") or "").strip() or "tool"
    failure_class = str(raw.get("failure_class") or "").strip()
    evidence_excerpt = str(raw.get("evidence_excerpt") or "").strip()
    artifact_id = str(raw.get("artifact_id") or "").strip()

    if failure_class:
        message = f"Chat mode stalled on repeated `{tool_name}` {failure_class} failures without new evidence."
    else:
        message = f"Chat mode stalled on repeated `{tool_name}` failures without new evidence."
    if evidence_excerpt:
        message = f"{message} Latest evidence: {evidence_excerpt}"

    return {
        "message": message,
        "details": {
            "guard": "chat_progress_loop",
            "tool_name": tool_name,
            "failure_class": failure_class,
            "stalled_rounds": stall_count,
            "artifact_id": artifact_id,
            "evidence_excerpt": evidence_excerpt,
        },
    }
