from __future__ import annotations

import logging
from typing import Any

from ..logging_utils import log_kv
from .client_transport_helpers import (
    latest_user_message_audit as _latest_user_message_audit,
    tool_name as _tool_name,
)
from .openrouter_preflight import _message_role_counts


def _log_request_audit(client: Any, *, payload: dict[str, Any], tools: list[dict[str, Any]], stage: str) -> None:
    messages = payload.get("messages")
    payload_tools = payload.get("tools")
    active_tools = payload_tools if isinstance(payload_tools, list) else tools
    tool_names = [_tool_name(tool) for tool in active_tools if _tool_name(tool)]
    details = {
        "stage": stage,
        "provider_profile": client.provider_profile,
        "model": client.model,
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "role_counts": _message_role_counts(messages),
        "tool_count": len(active_tools),
        "tool_names": tool_names,
        **_latest_user_message_audit(messages),
    }
    log_kv(client.log, logging.INFO, "chat_request_payload_audit", **details)
    if client.run_logger:
        client.run_logger.log(
            "chat",
            "request_payload_audit",
            "chat request payload audit",
            **details,
        )
