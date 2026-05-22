from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from smallctl.harness.escalation_service import EscalationService
from smallctl.state import LoopState
from smallctl.tools.register import build_registry


@pytest.mark.skipif(
    os.getenv("SMALLCTL_ESCALATION_SMOKE") != "1",
    reason="Set SMALLCTL_ESCALATION_SMOKE=1 with escalation endpoint/model credentials to run.",
)
def test_escalation_provider_smoke() -> None:
    endpoint = os.getenv("SMALLCTL_ESCALATION_ENDPOINT")
    model = os.getenv("SMALLCTL_ESCALATION_MODEL")
    if not endpoint or not model:
        pytest.skip("SMALLCTL_ESCALATION_ENDPOINT and SMALLCTL_ESCALATION_MODEL are required.")

    state = LoopState()
    state.last_verifier_verdict = {
        "verdict": "fail",
        "message": "Focused smoke-test failure evidence.",
    }
    state.recent_errors.append("Smoke-test verifier failed after a focused repair.")
    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(
            run_mode="loop",
            escalation_enabled=True,
            escalation_expose_tool=True,
            escalation_auto_trigger=False,
            escalation_endpoint=endpoint,
            escalation_model=model,
            escalation_provider_profile=os.getenv("SMALLCTL_ESCALATION_PROVIDER_PROFILE", "auto"),
            escalation_api_key=os.getenv("SMALLCTL_ESCALATION_API_KEY"),
            escalation_api_key_env=os.getenv("SMALLCTL_ESCALATION_API_KEY_ENV"),
            escalation_chat_endpoint=os.getenv("SMALLCTL_ESCALATION_CHAT_ENDPOINT", "/chat/completions"),
            escalation_max_prompt_chars=int(os.getenv("SMALLCTL_ESCALATION_MAX_PROMPT_CHARS", "8000")),
            escalation_max_response_tokens=int(os.getenv("SMALLCTL_ESCALATION_MAX_RESPONSE_TOKENS", "700")),
            escalation_temperature=float(os.getenv("SMALLCTL_ESCALATION_TEMPERATURE", "0.2")),
            escalation_timeout_sec=int(os.getenv("SMALLCTL_ESCALATION_TIMEOUT_SEC", "60")),
            escalation_max_per_task=2,
            escalation_cooldown_turns=0,
            escalation_repeated_failure_threshold=2,
            escalation_require_tool_plan_evidence=True,
            escalation_redact_secrets=True,
        ),
        run_logger=None,
        log=SimpleNamespace(info=lambda *args, **kwargs: None),
        _runlog=lambda *args, **kwargs: None,
    )
    harness.registry = build_registry(harness)

    result = asyncio.run(
        EscalationService(harness).run(
            reason="Smoke-test verifier failed after a focused repair.",
            question="Return the smallest safe next evidence-gathering step.",
            requested_output="next_action",
            risk_level="low",
        )
    )

    assert result["success"] is True
    assert result["status"] == "advisory"
    assert result["verdict"] in {
        "continue",
        "need_more_evidence",
        "reject_current_plan",
        "propose_patch",
        "ask_human",
        "abort",
    }
    assert isinstance(result.get("recommended_next_action"), dict)
