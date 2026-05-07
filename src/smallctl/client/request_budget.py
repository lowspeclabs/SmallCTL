from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

_APPROX_CHARS_PER_TOKEN = 4.0


@dataclass(frozen=True)
class RequestBudget:
    context_limit: int
    reserve_completion_tokens: int
    safety_margin_tokens: int
    effective_prompt_budget: int


@dataclass(frozen=True)
class RequestFootprint:
    estimated_payload_tokens: int
    estimated_message_tokens: int
    estimated_tool_tokens: int
    tool_count: int
    over_budget_tokens: int


class RequestEstimator:
    def estimate_tokens(self, value: Any) -> int:
        return approx_token_count(value)

    def footprint(self, payload: dict[str, Any], budget: RequestBudget | None = None) -> RequestFootprint:
        tools = payload.get("tools")
        tool_list = tools if isinstance(tools, list) else []
        messages = payload.get("messages")
        message_list = messages if isinstance(messages, list) else []
        estimated_payload_tokens = self.estimate_tokens(payload)
        over_budget_tokens = 0
        if budget is not None:
            over_budget_tokens = max(0, estimated_payload_tokens - budget.effective_prompt_budget)
        return RequestFootprint(
            estimated_payload_tokens=estimated_payload_tokens,
            estimated_message_tokens=self.estimate_tokens(message_list),
            estimated_tool_tokens=self.estimate_tokens(tool_list),
            tool_count=len(tool_list),
            over_budget_tokens=over_budget_tokens,
        )


def approx_token_count(value: Any) -> int:
    try:
        text = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        text = str(value or "")
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, int(len(stripped) / _APPROX_CHARS_PER_TOKEN))


def json_size_bytes(value: Any) -> int:
    try:
        return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    except Exception:
        return 0


def client_context_limit(client: Any) -> int | None:
    for attr in ("runtime_context_limit", "server_context_limit", "context_limit"):
        try:
            value = getattr(client, attr, None)
        except Exception:
            value = None
        try:
            normalized = int(value)
        except Exception:
            continue
        if normalized > 0:
            return normalized
    return None


def build_request_budget(context_limit: int) -> RequestBudget:
    normalized_limit = max(1, int(context_limit))
    reserve_completion_tokens = 1024
    safety_margin_tokens = max(512, normalized_limit // 16)
    effective_prompt_budget = max(1, normalized_limit - reserve_completion_tokens - safety_margin_tokens)
    return RequestBudget(
        context_limit=normalized_limit,
        reserve_completion_tokens=reserve_completion_tokens,
        safety_margin_tokens=safety_margin_tokens,
        effective_prompt_budget=effective_prompt_budget,
    )
