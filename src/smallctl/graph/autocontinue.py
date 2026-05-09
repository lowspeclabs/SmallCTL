from __future__ import annotations

import json
from typing import Any

from ..models.conversation import ConversationMessage
from ..state import json_safe_value
from .state import GraphRunState, PendingToolCall

_DURABLE_AUTOCONTINUE_KEY = "_durable_autocontinue_recoveries"
_DURABLE_AUTOCONTINUE_MAX = 8


def _pending_payload(pending: PendingToolCall) -> dict[str, Any]:
    return {
        "tool_name": pending.tool_name,
        "args": json_safe_value(dict(pending.args or {})),
        "tool_call_id": pending.tool_call_id,
        "raw_arguments": str(pending.raw_arguments or ""),
        "source": str(pending.source or "system"),
    }


def _payload_signature(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_name": str(payload.get("tool_name") or ""),
            "args": json_safe_value(payload.get("args") or {}),
            "recovery_kind": str(payload.get("recovery_kind") or ""),
            "signature": str(payload.get("signature") or ""),
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def store_durable_autocontinue(
    harness: Any,
    pending: PendingToolCall,
    *,
    recovery_kind: str,
    signature: str,
    reason: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return

    payload = _pending_payload(pending)
    payload.update(
        {
            "recovery_kind": recovery_kind,
            "signature": signature,
            "reason": reason,
            "metadata": json_safe_value(metadata or {}),
        }
    )
    payload_signature = _payload_signature(payload)
    payload["payload_signature"] = payload_signature

    raw_queue = scratchpad.get(_DURABLE_AUTOCONTINUE_KEY)
    queue = list(raw_queue) if isinstance(raw_queue, list) else []
    queue = [
        item
        for item in queue
        if isinstance(item, dict)
        and str(item.get("payload_signature") or _payload_signature(item)) != payload_signature
    ]
    queue.append(payload)
    scratchpad[_DURABLE_AUTOCONTINUE_KEY] = queue[-_DURABLE_AUTOCONTINUE_MAX:]


def drain_durable_autocontinue(
    graph_state: GraphRunState,
    harness: Any,
) -> bool:
    if graph_state.pending_tool_calls:
        return False
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    raw_queue = scratchpad.get(_DURABLE_AUTOCONTINUE_KEY)
    if not isinstance(raw_queue, list) or not raw_queue:
        return False

    queue = [item for item in raw_queue if isinstance(item, dict)]
    while queue:
        item = queue.pop(0)
        tool_name = str(item.get("tool_name") or "").strip()
        args = item.get("args")
        if not tool_name or not isinstance(args, dict):
            continue
        pending = PendingToolCall(
            tool_name=tool_name,
            args=dict(args),
            tool_call_id=None if item.get("tool_call_id") is None else str(item.get("tool_call_id")),
            raw_arguments=str(item.get("raw_arguments") or ""),
            source="system",
        )
        graph_state.pending_tool_calls = [pending]
        if queue:
            scratchpad[_DURABLE_AUTOCONTINUE_KEY] = queue
        else:
            scratchpad.pop(_DURABLE_AUTOCONTINUE_KEY, None)
        recovery_kind = str(item.get("recovery_kind") or "autocontinue").strip()
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    f"Resuming stored auto-continue recovery: dispatch `{pending.tool_name}` "
                    "before asking the model for another step."
                ),
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": recovery_kind,
                    "recovery_mode": "durable_autocontinue_resume",
                    "tool_name": pending.tool_name,
                    "arguments": json_safe_value(pending.args),
                },
            )
        )
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "durable_autocontinue_recovered",
                "restored pending auto-continue tool call from harness state",
                recovery_kind=recovery_kind,
                tool_name=pending.tool_name,
                arguments=json_safe_value(pending.args),
            )
        return True

    scratchpad.pop(_DURABLE_AUTOCONTINUE_KEY, None)
    return False


def clear_durable_autocontinue_for_pending(harness: Any, pending: PendingToolCall) -> bool:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    raw_queue = scratchpad.get(_DURABLE_AUTOCONTINUE_KEY)
    if not isinstance(raw_queue, list) or not raw_queue:
        return False

    pending_payload = _pending_payload(pending)
    pending_tool = str(pending_payload.get("tool_name") or "")
    pending_args = json.dumps(pending_payload.get("args") or {}, ensure_ascii=True, sort_keys=True)
    updated: list[dict[str, Any]] = []
    removed = False
    for item in raw_queue:
        if not isinstance(item, dict):
            continue
        item_args = json.dumps(json_safe_value(item.get("args") or {}), ensure_ascii=True, sort_keys=True)
        if str(item.get("tool_name") or "") == pending_tool and item_args == pending_args:
            removed = True
            continue
        updated.append(item)
    if updated:
        scratchpad[_DURABLE_AUTOCONTINUE_KEY] = updated
    else:
        scratchpad.pop(_DURABLE_AUTOCONTINUE_KEY, None)
    return removed


__all__ = [
    "_DURABLE_AUTOCONTINUE_KEY",
    "clear_durable_autocontinue_for_pending",
    "drain_durable_autocontinue",
    "store_durable_autocontinue",
]
