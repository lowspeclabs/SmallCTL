from __future__ import annotations

import logging
from typing import Any

from ..logging_utils import log_kv
from .openrouter_preflight import _client_context_limit, _tool_name
from .request_budget import RequestBudget, RequestEstimator, build_request_budget
from .tool_budgeting import ToolBudgetResult, fit_tools_to_context_budget
from .transport_constants import _LOCAL_PATCH_INTENT_RE, _LOCAL_WRITE_INTENT_RE, _UNSET


def _payload_text_for_tool_inference(payload: dict[str, Any], *, limit: int = 12000) -> str:
    pieces: list[str] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            pieces.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    pieces.append(str(item.get("text") or ""))
        if sum(len(piece) for piece in pieces) >= limit:
            break
    return "\n".join(pieces)[-limit:]


def _infer_llamacpp_requested_tool(payload: dict[str, Any], tools: list[dict[str, Any]]) -> str:
    available = {_tool_name(tool) for tool in tools if _tool_name(tool)}
    text = _payload_text_for_tool_inference(payload)
    if {"file_write", "file_patch", "ast_patch"} & available:
        if "file_write" in available and _LOCAL_WRITE_INTENT_RE.search(text):
            return "file_write"
        if "file_patch" in available and _LOCAL_PATCH_INTENT_RE.search(text):
            return "file_patch"
        if "ast_patch" in available and _LOCAL_PATCH_INTENT_RE.search(text):
            return "ast_patch"
    if {"ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"} & available:
        lower_text = text.lower()
        if "ssh_file_write" in available and ("ssh" in lower_text or "remote" in lower_text):
            if _LOCAL_WRITE_INTENT_RE.search(text):
                return "ssh_file_write"
        if "ssh_file_patch" in available and ("ssh" in lower_text or "remote" in lower_text):
            if _LOCAL_PATCH_INTENT_RE.search(text):
                return "ssh_file_patch"
    return ""


def _log_llamacpp_budget_preflight(
    client: Any,
    *,
    stage: str,
    action: str,
    result: ToolBudgetResult | None,
    budget: RequestBudget | None = None,
    context_limit: int | None,
    context_limit_source: str = "observed",
    reduction_reason: str = "",
) -> None:
    details: dict[str, Any] = {
        "stage": stage,
        "provider_profile": client.provider_profile,
        "model": client.model,
        "context_limit": context_limit,
        "context_limit_source": context_limit_source,
        "budget_action": action,
    }
    if reduction_reason:
        details["reduction_reason"] = reduction_reason
    if result is not None:
        logged_budget = budget
        if logged_budget is None:
            logged_budget = build_request_budget(result.footprint.estimated_payload_tokens)
        details.update(
            {
                "effective_prompt_budget": logged_budget.effective_prompt_budget,
                "reserve_completion_tokens": logged_budget.reserve_completion_tokens,
                "safety_margin_tokens": logged_budget.safety_margin_tokens,
                "tokenizer_slop_tokens": logged_budget.tokenizer_slop_tokens,
                "estimated_payload_tokens": result.footprint.estimated_payload_tokens,
                "estimated_message_tokens": result.footprint.estimated_message_tokens,
                "estimated_tool_tokens": result.footprint.estimated_tool_tokens,
                "tool_count_before": result.tool_count_before,
                "tool_count_after": result.tool_count_after,
                "dropped_tool_names": list(result.dropped_tool_names),
                "kept_tool_names": list(result.kept_tool_names),
                "over_budget_tokens": result.footprint.over_budget_tokens,
            }
        )
    log_kv(client.log, logging.DEBUG, "chat_payload_preflight_budget", **details)
    if client.run_logger:
        client.run_logger.log(
            "chat",
            "payload_preflight_budget",
            "chat payload budget preflight",
            **details,
        )


def _llamacpp_budget_preflight(
    client: Any,
    *,
    payload: dict[str, Any],
    stage: str,
    context_limit: int | None = None,
    context_limit_source: str = "observed",
    reduction_reason: str = "",
    log_context_limit: Any = _UNSET,
) -> ToolBudgetResult | None:
    if client.provider_profile != "llamacpp":
        return None
    limit = context_limit or _client_context_limit(client)
    raw_tools = payload.get("tools")
    tools = raw_tools if isinstance(raw_tools, list) else []

    # Gemma-4 variants (including 12b) on llama.cpp use SWA/hybrid memory by
    # default and can invalidate the prompt cache at large contexts. Warn when
    # the requested context is above the safe threshold and cap the effective
    # prompt budget to reduce cache pressure. The real fix is to run the
    # backend with --swa-full so prefix caching can be reused across turns.
    from ..graph.tool_model_rules_model_detection import _model_is_gemma_4_small
    small_gemma4 = _model_is_gemma_4_small(client.model)
    if small_gemma4 and limit is not None and limit > 32768:
        log_kv(
            client.log,
            logging.WARNING,
            "llamacpp_small_gemma4_context_warning",
            model=client.model,
            context_limit=limit,
            recommendation="use --context-limit 32768 and --swa-full for Gemma-4 on llama.cpp",
        )
        if client.run_logger:
            client.run_logger.log(
                "chat",
                "small_gemma4_context_warning",
                "Gemma-4 SWA context may cause prompt-cache invalidation",
                model=client.model,
                context_limit=limit,
                recommended_context_limit=32768,
            )
        limit = min(limit, 32768)

    if limit is None:
        estimator = RequestEstimator()
        footprint = estimator.footprint(payload)
        log_kv(
            client.log,
            logging.DEBUG,
            "chat_payload_preflight_budget",
            stage=stage,
            provider_profile=client.provider_profile,
            model=client.model,
            context_limit=None,
            budget_action="skipped_unknown_limit",
            estimated_payload_tokens=footprint.estimated_payload_tokens,
            estimated_message_tokens=footprint.estimated_message_tokens,
            estimated_tool_tokens=footprint.estimated_tool_tokens,
            tool_count_before=footprint.tool_count,
            tool_count_after=footprint.tool_count,
            dropped_tool_names=[],
            kept_tool_names=[_tool_name(tool) for tool in tools if _tool_name(tool)],
            over_budget_tokens=0,
        )
        if client.run_logger:
            client.run_logger.log(
                "chat",
                "payload_preflight_budget",
                "chat payload budget preflight",
                stage=stage,
                provider_profile=client.provider_profile,
                model=client.model,
                context_limit=None,
                budget_action="skipped_unknown_limit",
                estimated_payload_tokens=footprint.estimated_payload_tokens,
                estimated_message_tokens=footprint.estimated_message_tokens,
                estimated_tool_tokens=footprint.estimated_tool_tokens,
                tool_count_before=footprint.tool_count,
                tool_count_after=footprint.tool_count,
                dropped_tool_names=[],
                kept_tool_names=[_tool_name(tool) for tool in tools if _tool_name(tool)],
                over_budget_tokens=0,
            )
        return None

    budget = build_request_budget(limit)
    if small_gemma4 and budget.effective_prompt_budget > 24576:
        budget = RequestBudget(
            context_limit=budget.context_limit,
            reserve_completion_tokens=budget.reserve_completion_tokens,
            safety_margin_tokens=budget.safety_margin_tokens,
            tokenizer_slop_tokens=budget.tokenizer_slop_tokens,
            effective_prompt_budget=24576,
        )
    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=budget,
        requested_tool_name=_infer_llamacpp_requested_tool(payload, tools),
        estimator=RequestEstimator(),
    )
    displayed_context_limit = limit if log_context_limit is _UNSET else log_context_limit
    _log_llamacpp_budget_preflight(
        client,
        stage=stage,
        action=result.action,
        result=result,
        budget=budget,
        context_limit=displayed_context_limit,
        context_limit_source=context_limit_source,
        reduction_reason=reduction_reason,
    )
    return result


def _build_llamacpp_reduced_tools_payload(
    client: Any,
    *,
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    tools = payload.get("tools")
    if not isinstance(tools, list) or len(tools) < 12:
        return None
    limit = _client_context_limit(client)
    limit_was_observed = limit is not None
    if limit is None:
        limit = 1
    result = _llamacpp_budget_preflight(
        client,
        payload=payload,
        stage="http_400_reduced_tools_retry",
        context_limit=limit,
        context_limit_source="unknown",
        reduction_reason="http_400_recovery",
        log_context_limit=None if not limit_was_observed else limit,
    )
    if result is None or result.tool_count_after >= result.tool_count_before:
        return None
    if result.action == "exceeded" and limit_was_observed:
        return None
    return client.adapter.mutate_payload(result.payload)


def _build_minimal_context_payload(
    client: Any,
    *,
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    last_system_msg = None
    for msg in reversed(messages):
        if str(msg.get("role", "")).strip() == "system":
            last_system_msg = msg
            break

    recent_user_msgs = []
    recent_exchange: list[dict[str, Any]] = []
    captured_exchange = False
    for msg in reversed(messages):
        role = str(msg.get("role", "")).strip()
        if role == "tool" or (role == "assistant" and msg.get("tool_calls")):
            # Capture the most recent assistant/tool exchange preceding the user messages.
            if not captured_exchange:
                recent_exchange.insert(0, dict(msg))
                if role == "assistant":
                    captured_exchange = True
            continue
        if role == "user":
            if captured_exchange:
                break
            recent_user_msgs.insert(0, dict(msg))
            if len(recent_user_msgs) >= 2:
                break

    user_content = ""
    if recent_user_msgs:
        if len(recent_user_msgs) == 2:
            user_content = str(recent_user_msgs[0].get("content") or "") + "\n\n" + str(recent_user_msgs[1].get("content") or "")
        else:
            user_content = str(recent_user_msgs[0].get("content") or "")

    reduced_messages = []
    if last_system_msg:
        reduced_messages.append(last_system_msg)
    if recent_exchange:
        reduced_messages.extend(recent_exchange)
    if user_content:
        reduced_messages.append({"role": "user", "content": user_content})

    payload: dict[str, Any] = {
        "model": client.model,
        "messages": client.adapter.sanitize_messages(reduced_messages),
        "stream": True,
    }
    return client.adapter.mutate_payload(payload)


def _is_swa_model(model_name: str | None, provider_profile: str | None) -> bool:
    """Return True for models/backends known to use SWA/hybrid memory on llama.cpp.

    The Gemma-4 family (including the 12b checkpoint observed in the wild on
    llama.cpp) uses SWA/hybrid memory. When the backend cannot restore the
    cached SWA state across turns, it falls back to full prompt reprocessing,
    which looks like a prompt-cache loop. Treat all Gemma-4 variants served by
    llama.cpp as SWA models so the harness can warn and compact context.
    """
    from ..graph.tool_model_rules_model_detection import _model_is_gemma_4

    return (
        _model_is_gemma_4(model_name)
        and str(provider_profile or "").strip().lower() == "llamacpp"
    )


def _extract_cached_tokens(usage: dict[str, Any]) -> int | None:
    """Best-effort extraction of cached/prefix prompt tokens from usage payload."""
    if not isinstance(usage, dict):
        return None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
        if cached is not None:
            try:
                return int(cached)
            except (TypeError, ValueError):
                return None
    # Some llama.cpp wrappers expose cached_tokens at the top level.
    cached = usage.get("cached_tokens")
    if cached is not None:
        try:
            return int(cached)
        except (TypeError, ValueError):
            return None
    return None


def _record_swa_cache_observation(
    harness: Any,
    usage: dict[str, Any],
    *,
    warning_threshold: int = 2,
) -> bool:
    """Track consecutive turns with zero cached tokens on SWA models.

    Returns True when the streak crosses the warning threshold.
    """
    client = getattr(harness, "client", None)
    model_name = getattr(client, "model", None) if client is not None else None
    provider_profile = getattr(client, "provider_profile", None) if client is not None else None
    if not _is_swa_model(model_name, provider_profile):
        return False
    cached = _extract_cached_tokens(usage)
    if cached is None:
        # llama.cpp SWA models frequently omit cached_tokens from usage. Treat
        # a missing value as a zero-cache observation for SWA models so the
        # harness can still warn and compact context proactively.
        if not _is_swa_model(model_name, provider_profile):
            return False
        cached = 0
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return False
    if cached == 0:
        streak = int(scratchpad.get("_swa_zero_cached_streak", 0) or 0) + 1
        scratchpad["_swa_zero_cached_streak"] = streak
        return streak >= warning_threshold
    scratchpad["_swa_zero_cached_streak"] = 0
    return False


def _maybe_emit_swa_cache_warning(harness: Any, usage: dict[str, Any]) -> None:
    """Log and inject a FAMA capsule when SWA cache stays inactive."""
    if not _record_swa_cache_observation(harness, usage):
        return
    log_kv(
        harness.log,
        logging.WARNING,
        "llamacpp_swa_cache_inactive",
        model=getattr(getattr(harness, "client", None), "model", None),
        recommendation=(
            "llama.cpp SWA cache appears inactive (cached_tokens=0 for N turns). "
            "If the backend supports it, run with --swa-full or disable SWA/hybrid memory "
            "so prefix caching can be reused across turns."
        ),
    )
    harness._runlog(
        "swa_cache_inactive",
        "llama.cpp SWA cache appears inactive across multiple turns",
        recommendation="run backend with --swa-full or disable SWA/hybrid memory",
    )
    try:
        from ..fama.signals import FamaFailureKind, FamaSignal, push_fama_signal
        from ..fama.state import activate_mitigations, ActiveMitigation

        state = getattr(harness, "state", None)
        if state is None:
            return
        step = int(getattr(state, "step_count", 0) or 0)
        push_fama_signal(
            state,
            FamaSignal(
                kind=FamaFailureKind.CONTEXT_DRIFT,
                severity=2,
                source="swa_cache_inactive",
                evidence="cached_tokens=0 for consecutive turns on SWA model",
                step=step,
                suggested_mitigations=["micro_plan_capsule"],
                failure_class="context_missing",
                next_safe_action="keep reasoning concise and reuse visible evidence",
            ),
        )
        activate_mitigations(
            state,
            [
                ActiveMitigation(
                    name="micro_plan_capsule",
                    reason="SWA/hybrid cache inactive; keep reasoning concise",
                    source_signal="swa_cache_inactive",
                    activated_step=step,
                    expires_after_step=step + 4,
                    priority=55,
                )
            ],
            max_active=2,
        )
    except Exception:
        pass
