from __future__ import annotations

from typing import Any

from ..client.usage import apply_usage_metrics as _apply_usage_metrics
from ..context import build_retrieval_query as _build_retrieval_query
from ..models.tool_result import ToolEnvelope
from .conversation_logging import log_conversation_state as _log_conversation_state_helper
from .conversation_logging import record_assistant_message as _record_assistant_message_helper
from .context_limits import apply_server_context_limit as _apply_server_context_limit_helper
from .context_limits import resolve_effective_prompt_budget as _resolve_effective_prompt_budget_helper


def _resolve_effective_prompt_budget(
    cls: type[Any],
    *,
    configured_max_prompt_tokens: int | None,
    configured_max_prompt_tokens_explicit: bool = True,
    server_context_limit: int | None,
    current_max_prompt_tokens: int | None = None,
    observed_n_keep: int | None = None,
    provider_profile: str | None = None,
    model_name: str | None = None,
) -> int | None:
    return _resolve_effective_prompt_budget_helper(
        configured_max_prompt_tokens=configured_max_prompt_tokens,
        configured_max_prompt_tokens_explicit=configured_max_prompt_tokens_explicit,
        server_context_limit=server_context_limit,
        current_max_prompt_tokens=current_max_prompt_tokens,
        observed_n_keep=observed_n_keep,
        provider_profile=provider_profile,
        model_name=model_name,
    )


def _apply_server_context_limit(
    self: Any,
    context_limit: int | None,
    *,
    source: str,
    observed_n_keep: int | None = None,
) -> int | None:
    return _apply_server_context_limit_helper(
        self,
        context_limit,
        source=source,
        observed_n_keep=observed_n_keep,
    )


async def _ensure_context_limit(self: Any) -> None:
    await self.prompt_builder.ensure_context_limit()


def _apply_usage(self: Any, usage: dict[str, Any]) -> None:
    _apply_usage_metrics(self, usage)
    backend_model = usage.get("_backend_model_name") if isinstance(usage, dict) else None
    if backend_model and getattr(self, "client", None) is not None:
        context_limit = getattr(self, "server_context_limit", None) or getattr(
            getattr(self, "context_policy", None), "max_prompt_tokens", None
        )
        normalized = self.client.apply_backend_model_profile(backend_model, context_limit)
        if normalized and getattr(self, "context_policy", None) is not None:
            self.context_policy.apply_model_profile(normalized)
            scaling_context = context_limit or self.context_policy.max_prompt_tokens
            if scaling_context:
                self.context_policy.recalculate_quotas(scaling_context)
            self._runlog(
                "backend_model_profile_applied",
                "applied backend-reported model profile for small context window",
                backend_model=backend_model,
                normalized_model=normalized,
                context_limit=context_limit,
            )


async def _record_tool_result(
    self: Any,
    tool_name: str,
    tool_call_id: str | None,
    result: ToolEnvelope,
    arguments: dict[str, Any] | None = None,
    operation_id: str | None = None,
) -> Any:
    return await self.tool_results.record_result(
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        result=result,
        arguments=arguments,
        operation_id=operation_id,
    )


def _record_assistant_message(
    self: Any,
    *,
    assistant_text: str,
    tool_calls: list[dict[str, Any]],
    speaker: str | None = None,
    hidden_from_prompt: bool = False,
) -> None:
    _record_assistant_message_helper(
        self,
        assistant_text=assistant_text,
        tool_calls=tool_calls,
        speaker=speaker,
        hidden_from_prompt=hidden_from_prompt,
    )


def _log_conversation_state(self: Any, event: str) -> None:
    _log_conversation_state_helper(self, event)


def _select_retrieval_query(self: Any) -> str:
    query = str(_build_retrieval_query(self.state) or "").strip()
    if query:
        return query
    current_task = getattr(self, "_current_user_task", None)
    if callable(current_task):
        fallback = str(current_task() or "").strip()
        if fallback:
            return fallback
    return str(getattr(getattr(self.state, "run_brief", None), "original_task", "") or "").strip()


def bind_context_facade(cls: type[Any]) -> None:
    cls._resolve_effective_prompt_budget = classmethod(_resolve_effective_prompt_budget)
    cls._apply_server_context_limit = _apply_server_context_limit
    cls._ensure_context_limit = _ensure_context_limit
    cls._apply_usage = _apply_usage
    cls._record_tool_result = _record_tool_result
    cls._record_assistant_message = _record_assistant_message
    cls._log_conversation_state = _log_conversation_state
    cls._select_retrieval_query = _select_retrieval_query
