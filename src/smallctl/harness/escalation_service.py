from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from smallctl.client import OpenAICompatClient

from ..recovery_metrics import increment_metric, recovery_metrics
from .escalation_config import EscalationConfigError, build_escalation_model_config
from .escalation_packet import build_escalation_packet
from .escalation_policy import EscalationPolicy
from .escalation_response import parse_and_validate_escalation_response
from .trajectory_recorder import TrajectoryRecorder


SYSTEM_PROMPT = """You are the escalation advisor for SmallCTL.
You are not the acting agent.
You do not execute tools, write files, run shell commands, approve risk, or mark tasks complete.
Use only the provided packet.
Return valid JSON only.
Prefer the smallest safe next step.
If evidence is insufficient, use verdict need_more_evidence."""

REPAIR_SYSTEM_PROMPT = """You repair invalid escalation responses for SmallCTL.
Return one valid JSON object only.
Do not include markdown or commentary.
Fix the response so it satisfies the escalation contract exactly.
If no action is needed, use {"type":"none","reason":"No next action supplied."} for recommended_next_action."""


class EscalationService:
    def __init__(self, harness: Any) -> None:
        self.harness = harness

    async def run(
        self,
        *,
        reason: str,
        question: str,
        requested_output: str,
        risk_level: str = "medium",
        source: str = "manual",
    ) -> dict[str, Any]:
        state = self.harness.state
        if str(source or "").strip().lower() == "auto":
            increment_metric(state, "escalation_auto_triggers")
        else:
            increment_metric(state, "escalation_manual_tool_calls")
        started = time.monotonic()

        if not bool(getattr(getattr(self.harness, "config", None), "escalation_enabled", False)):
            increment_metric(state, "escalation_policy_blocks")
            return {
                "success": False,
                "status": "blocked",
                "trigger": "disabled",
                "reason": "Escalation is disabled.",
                "evidence_count": 0,
            }

        try:
            config = build_escalation_model_config(self.harness.config)
        except EscalationConfigError as exc:
            increment_metric(state, "escalation_config_errors")
            escalation_id = _next_escalation_id(state)
            self._record_terminal_trace(
                escalation_id=escalation_id,
                trigger="config_error",
                model="",
                status="config_error",
                error=str(exc),
            )
            return {"success": False, "status": "blocked", "error": str(exc)}

        decision = EscalationPolicy(self.harness).can_escalate(reason=reason, risk_level=risk_level)
        if not decision.allowed:
            return {
                "success": False,
                "status": "blocked",
                "trigger": decision.trigger,
                "reason": decision.reason,
                "evidence_count": decision.evidence_count,
            }

        escalation_id = _next_escalation_id(state)
        packet = build_escalation_packet(
            self.harness,
            reason=reason,
            question=question,
            requested_output=requested_output,
            risk_level=risk_level,
            trigger=decision.trigger,
            max_chars=config.max_prompt_chars,
            redact_secrets=bool(getattr(self.harness.config, "escalation_redact_secrets", True)),
        )
        packet_text = json.dumps(packet, ensure_ascii=True, sort_keys=True)
        _metric_set(state, "escalation_prompt_chars", len(packet_text))

        client = OpenAICompatClient(
            base_url=config.endpoint,
            model=config.model,
            api_key=config.api_key,
            chat_endpoint=config.chat_endpoint,
            provider_profile=config.provider_profile,
            runtime_context_probe=False,
            run_logger=getattr(self.harness, "run_logger", None),
        )
        client.max_completion_tokens = config.max_response_tokens
        client.temperature = config.temperature
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": packet_text},
        ]
        response_text, provider_error = await _collect_escalation_response(
            client,
            messages,
            timeout_sec=config.timeout_sec,
        )
        if provider_error:
            increment_metric(state, "escalation_provider_failures")
            self._record_terminal_trace(
                escalation_id=escalation_id,
                trigger=decision.trigger,
                model=config.model,
                status="provider_error",
                error=provider_error,
            )
            return {
                "success": False,
                "status": "provider_error",
                "escalation_id": escalation_id,
                "error": provider_error,
            }

        _metric_set(state, "escalation_response_chars", len(response_text))
        validation = parse_and_validate_escalation_response(response_text, harness=self.harness)
        wall_clock = round(time.monotonic() - started, 3)
        _metric_set(state, "escalation_wall_clock_sec", wall_clock)

        if not validation.ok:
            retry_result = await self._retry_invalid_response(
                client=client,
                config=config,
                escalation_id=escalation_id,
                trigger=decision.trigger,
                model=config.model,
                original_response=response_text,
                validation_error=validation.error,
            )
            if retry_result is not None:
                return retry_result
            self._record_terminal_trace(
                escalation_id=escalation_id,
                trigger=decision.trigger,
                model=config.model,
                status="invalid_advisory",
                error=validation.error,
            )
            return {
                "success": False,
                "status": "invalid_advisory",
                "escalation_id": escalation_id,
                "error": validation.error,
            }

        response = validation.response or {}
        increment_metric(state, "escalation_invocations")
        increment_metric(state, "escalation_accepted")
        if response.get("requires_human_approval") or response.get("verdict") == "ask_human":
            increment_metric(state, "escalation_human_required")

        advisory = {
            "success": True,
            "status": "advisory",
            "verdict": response.get("verdict"),
            "confidence": response.get("confidence"),
            "failure_diagnosis": response.get("failure_diagnosis"),
            "recommended_next_action": response.get("recommended_next_action"),
            "repair_plan": response.get("repair_plan"),
            "risk_notes": response.get("risks") or response.get("risk_notes") or [],
            "requires_human_approval": bool(response.get("requires_human_approval", False)),
            "escalation_id": escalation_id,
        }
        self._record_state(escalation_id, advisory, decision.trigger, config.model)
        return advisory

    def _record_state(self, escalation_id: str, advisory: dict[str, Any], trigger: str, model: str) -> None:
        state = self.harness.state
        scratchpad = getattr(state, "scratchpad", None)
        if not isinstance(scratchpad, dict):
            return
        note = {
            "id": escalation_id,
            "step_count": getattr(state, "step_count", 0),
            "trigger": trigger,
            "verdict": advisory.get("verdict"),
        }
        scratchpad["_last_escalation"] = note
        _append_escalation_history(scratchpad, note, key="_escalation_history")

        ledger = getattr(state, "subtask_ledger", None)
        active = ledger.active() if ledger is not None and hasattr(ledger, "active") else None
        service = getattr(self.harness, "subtask_ledger", None)
        if service is not None and active is not None:
            service.attach_evidence(str(active.subtask_id), f"Escalation {escalation_id}: {advisory.get('verdict')}")
            next_action = _subtask_next_action_text(advisory)
            if next_action and hasattr(active, "next_action"):
                active.next_action = next_action[:240]

        run_logger = getattr(self.harness, "run_logger", None)
        if run_logger is not None:
            run_logger.log(
                "recovery",
                "escalation_advisory",
                "larger-model advisory returned",
                escalation_id=escalation_id,
                trigger=trigger,
                model=model,
                verdict=advisory.get("verdict"),
            )

        try:
            TrajectoryRecorder().record_escalation(self.harness, advisory)
        except Exception:
            pass

    async def _retry_invalid_response(
        self,
        *,
        client: OpenAICompatClient,
        config: Any,
        escalation_id: str,
        trigger: str,
        model: str,
        original_response: str,
        validation_error: str,
    ) -> dict[str, Any] | None:
        if not _should_retry_invalid_response(validation_error, original_response):
            return None

        increment_metric(self.harness.state, "escalation_validation_retries")
        repair_messages = [
            {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_repair_prompt(
                    validation_error=validation_error,
                    original_response=original_response,
                ),
            },
        ]
        repaired_response, provider_error = await _collect_escalation_response(
            client,
            repair_messages,
            timeout_sec=config.timeout_sec,
        )
        if provider_error:
            increment_metric(self.harness.state, "escalation_provider_failures")
            self._record_terminal_trace(
                escalation_id=escalation_id,
                trigger=trigger,
                model=model,
                status="provider_error",
                error=provider_error,
            )
            return {
                "success": False,
                "status": "provider_error",
                "escalation_id": escalation_id,
                "error": provider_error,
            }

        _metric_set(self.harness.state, "escalation_response_chars", len(repaired_response))
        validation = parse_and_validate_escalation_response(repaired_response, harness=self.harness)
        if not validation.ok:
            return None

        response = validation.response or {}
        state = self.harness.state
        increment_metric(state, "escalation_invocations")
        increment_metric(state, "escalation_accepted")
        if response.get("requires_human_approval") or response.get("verdict") == "ask_human":
            increment_metric(state, "escalation_human_required")

        advisory = {
            "success": True,
            "status": "advisory",
            "verdict": response.get("verdict"),
            "confidence": response.get("confidence"),
            "failure_diagnosis": response.get("failure_diagnosis"),
            "recommended_next_action": response.get("recommended_next_action"),
            "repair_plan": response.get("repair_plan"),
            "risk_notes": response.get("risks") or response.get("risk_notes") or [],
            "requires_human_approval": bool(response.get("requires_human_approval", False)),
            "escalation_id": escalation_id,
        }
        self._record_state(escalation_id, advisory, trigger, model)
        return advisory

    def _record_terminal_trace(
        self,
        *,
        escalation_id: str,
        trigger: str,
        model: str,
        status: str,
        error: str,
    ) -> None:
        state = self.harness.state
        scratchpad = getattr(state, "scratchpad", None)
        if isinstance(scratchpad, dict):
            note = {
                "id": escalation_id,
                "step_count": getattr(state, "step_count", 0),
                "trigger": trigger,
                "verdict": status,
            }
            scratchpad["_last_escalation"] = note
            _append_escalation_history(scratchpad, note, key="_escalation_attempt_history")

        run_logger = getattr(self.harness, "run_logger", None)
        if run_logger is not None:
            run_logger.log(
                "recovery",
                "escalation_terminal",
                "larger-model advisory did not produce an accepted response",
                escalation_id=escalation_id,
                trigger=trigger,
                model=model,
                status=status,
                error=error,
            )

        try:
            TrajectoryRecorder().record_escalation(
                self.harness,
                {
                    "success": False,
                    "status": status,
                    "escalation_id": escalation_id,
                    "verdict": status,
                    "error": error,
                },
            )
        except Exception:
            pass


def _next_escalation_id(state: Any) -> str:
    return f"esc_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{int(getattr(state, 'step_count', 0) or 0):03d}"


def _metric_set(state: Any, name: str, value: Any) -> None:
    metrics = recovery_metrics(state)
    if isinstance(metrics, dict):
        metrics[name] = value


def _first_chunk_error(chunks: list[dict[str, Any]]) -> str:
    for chunk in chunks:
        if not isinstance(chunk, dict) or chunk.get("type") != "chunk_error":
            continue
        message = str(chunk.get("error") or "").strip()
        details = chunk.get("details")
        if isinstance(details, dict):
            detail_message = str(details.get("message") or details.get("error") or "").strip()
            if detail_message:
                return detail_message
        return message or "Escalation provider returned a stream error."
    return ""


def _subtask_next_action_text(advisory: dict[str, Any]) -> str:
    action = advisory.get("recommended_next_action")
    if not isinstance(action, dict):
        return ""
    action_type = str(action.get("type") or "").strip()
    reason = str(action.get("reason") or "").strip()
    if action_type == "tool_call":
        tool = str(action.get("tool") or "").strip()
        if tool and reason:
            return f"Escalation recommends `{tool}`: {reason}"
        if tool:
            return f"Escalation recommends `{tool}`."
    if action_type and reason:
        return f"Escalation recommends {action_type}: {reason}"
    if reason:
        return f"Escalation recommends: {reason}"
    return ""


def _append_escalation_history(scratchpad: dict[str, Any], note: dict[str, Any], *, key: str) -> None:
    history = scratchpad.get(key)
    if not isinstance(history, list):
        history = []
        scratchpad[key] = history
    history.append(note)
    del history[:-8]


async def _collect_escalation_response(
    client: OpenAICompatClient,
    messages: list[dict[str, Any]],
    *,
    timeout_sec: int,
) -> tuple[str, str]:
    chunks: list[dict[str, Any]] = []

    async def _collect() -> None:
        async for chunk in client.stream_chat(messages, tools=[]):
            chunks.append(chunk)

    try:
        await asyncio.wait_for(_collect(), timeout=timeout_sec)
    except Exception as exc:
        return "", str(exc)

    chunk_error = _first_chunk_error(chunks)
    if chunk_error:
        return "", chunk_error

    stream = OpenAICompatClient.collect_stream(chunks, reasoning_mode="off")
    return stream.assistant_text.strip(), ""


def _should_retry_invalid_response(validation_error: str, original_response: str) -> bool:
    if not str(original_response or "").strip():
        return True
    return bool(str(validation_error or "").strip())


def _build_repair_prompt(*, validation_error: str, original_response: str) -> str:
    clipped_response = _clip_response_text(original_response)
    return (
        "The previous escalation response was invalid.\n"
        f"Validation error: {validation_error}\n\n"
        "Return a single JSON object only. No markdown, no commentary.\n"
        "The object must include:\n"
        '- "verdict"\n'
        '- "confidence" as a number from 0 to 1\n'
        '- "failure_diagnosis"\n'
        '- "recommended_next_action" as an object with at least "type" and "reason"\n'
        '- "repair_plan"\n'
        '- "risk_notes" as an array\n'
        '- "requires_human_approval" as a boolean\n\n'
        'If no next action is needed, use {"type":"none","reason":"No next action supplied."}.\n\n'
        "Correct this invalid response:\n"
        f"{clipped_response}"
    )


def _clip_response_text(text: str, limit: int = 1600) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 14)].rstrip() + " [truncated]"
