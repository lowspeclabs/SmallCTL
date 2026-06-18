from __future__ import annotations

import logging
from typing import Any

from ..logging_utils import log_kv
from .adapters.common import merge_system_messages_for_single_system_providers


def _repair_llamacpp_system_messages_for_transport(
    client: Any,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if client.provider_profile != "llamacpp":
        return messages

    system_positions = [
        index
        for index, message in enumerate(messages)
        if str(message.get("role") or "").strip().lower() == "system"
    ]
    if not system_positions:
        return messages
    if (
        system_positions == [0]
        and str(messages[0].get("role") or "").strip() == "system"
    ):
        return messages

    repaired = merge_system_messages_for_single_system_providers(messages)
    log_kv(
        client.log,
        logging.WARNING,
        "llamacpp_system_messages_repaired",
        system_count=len(system_positions),
        system_positions=system_positions,
    )
    if client.run_logger:
        client.run_logger.log(
            "chat",
            "llamacpp_system_messages_repaired",
            "repaired llama.cpp system message order before transport",
            system_count=len(system_positions),
            system_positions=system_positions,
        )
    return repaired
