from __future__ import annotations

import logging
import json
from types import SimpleNamespace

from smallctl.config import resolve_config
from smallctl.client.stream_collectors import StreamResult
from smallctl.context import ChildRunRequest
from smallctl.main import _escalation_harness_kwargs
from smallctl.harness import Harness, HarnessConfig
from smallctl.harness.escalation_packet import build_escalation_packet
from smallctl.harness.escalation_service import EscalationService
from smallctl.harness.escalation_config import build_escalation_model_config
from smallctl.harness.escalation_policy import EscalationPolicy
from smallctl.harness.escalation_response import parse_and_validate_escalation_response
from smallctl.harness.trajectory_recorder import TrajectoryRecorder
from smallctl.harness.subtask_ledger_service import SubtaskLedgerService
from smallctl.recovery_schema import FailureEvent, Subtask, SubtaskLedger
from smallctl.state import LoopState
from smallctl.models.conversation import ConversationMessage
from smallctl.graph.state import GraphRunState, PendingToolCall, ToolExecutionRecord
from smallctl.graph.escalation_triggers import (
    _maybe_auto_trigger_escalation_for_completion_block,
    _maybe_auto_trigger_escalation_for_patch_stall,
    _maybe_auto_trigger_escalation_for_tool_loop,
)
from smallctl.models.tool_result import ToolEnvelope
from smallctl.tools.escalation import escalate_to_bigger_model
from smallctl.tools.register import build_registry


def _harness(**config_overrides):
    state = LoopState()
    defaults = {
        "escalation_enabled": False,
        "escalation_expose_tool": True,
        "escalation_endpoint": None,
        "escalation_model": None,
        "escalation_provider_profile": "auto",
        "escalation_api_key": None,
        "escalation_api_key_env": None,
        "escalation_chat_endpoint": "/chat/completions",
        "escalation_max_per_task": 2,
        "escalation_cooldown_turns": 2,
        "escalation_repeated_failure_threshold": 2,
        "escalation_require_tool_plan_evidence": True,
        "run_mode": "loop",
    }
    defaults.update(config_overrides)
    config = SimpleNamespace(**defaults)
    return SimpleNamespace(
        state=state,
        config=config,
        log=logging.getLogger("test.escalation"),
        _runlog=lambda *args, **kwargs: None,
    )


def test_escalation_config_cli_normalization():
    cfg = resolve_config(
        {
            "escalation_enabled": "true",
            "escalation_endpoint": "https://example.test/v1",
            "escalation_model": "big-model",
            "escalation_max_prompt_chars": "12345",
            "escalation_temperature": "0.4",
        }
    )

    assert cfg.escalation_enabled is True
    assert cfg.escalation_endpoint == "https://example.test/v1"
    assert cfg.escalation_model == "big-model"
    assert cfg.escalation_max_prompt_chars == 12345
    assert cfg.escalation_temperature == 0.4


def test_escalation_enabled_missing_model_config_warns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = resolve_config({"escalation_enabled": "true"})

    assert any("escalation_endpoint or escalation_model is missing" in warning for warning in cfg.compatibility_warnings)


def test_escalation_env_config_normalization(monkeypatch):
    monkeypatch.setenv("SMALLCTL_ESCALATION_ENABLED", "true")
    monkeypatch.setenv("SMALLCTL_ESCALATION_ENDPOINT", "https://env.example/v1")
    monkeypatch.setenv("SMALLCTL_ESCALATION_MODEL", "env-big")
    monkeypatch.setenv("SMALLCTL_ESCALATION_TIMEOUT_SEC", "77")
    monkeypatch.setenv("SMALLCTL_ESCALATION_REQUIRE_TOOL_PLAN_EVIDENCE", "false")

    cfg = resolve_config({})

    assert cfg.escalation_enabled is True
    assert cfg.escalation_endpoint == "https://env.example/v1"
    assert cfg.escalation_model == "env-big"
    assert cfg.escalation_timeout_sec == 77
    assert cfg.escalation_require_tool_plan_evidence is False


def test_harness_config_carries_escalation_values(tmp_path):
    harness = Harness(
        endpoint="http://localhost:8000/v1",
        model="demo-model",
        phase="explore",
        escalation_enabled=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_max_prompt_chars=9999,
    )

    assert harness.config.escalation_enabled is True
    assert harness.config.escalation_endpoint == "https://example.test/v1"
    assert harness.config.escalation_model == "big-model"
    assert harness.config.escalation_max_prompt_chars == 9999


def test_child_harness_inherits_escalation_config():
    parent = Harness(
        endpoint="http://localhost:8000/v1",
        model="demo-model",
        phase="explore",
        runtime_context_probe=False,
        escalation_enabled=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_provider_profile="openrouter",
        escalation_api_key_env="BIG_KEY",
        escalation_auto_trigger=True,
        escalation_max_prompt_chars=9999,
    )
    captured = {}

    def fake_factory(config: HarnessConfig):
        captured.update(config.__dict__)
        return SimpleNamespace(state=SimpleNamespace(cwd=""))

    child = parent.subtasks.create_child_harness(
        request=ChildRunRequest(brief="child task", phase="plan"),
        harness_factory=fake_factory,
    )

    assert child.state.cwd == parent.state.cwd
    assert captured["phase"] == "plan"
    assert captured["escalation_enabled"] is True
    assert captured["escalation_endpoint"] == "https://example.test/v1"
    assert captured["escalation_model"] == "big-model"
    assert captured["escalation_provider_profile"] == "openrouter"
    assert captured["escalation_api_key_env"] == "BIG_KEY"
    assert captured["escalation_auto_trigger"] is True
    assert captured["escalation_max_prompt_chars"] == 9999


def test_main_escalation_harness_kwargs_carries_full_config():
    config = SimpleNamespace(
        escalation_enabled=True,
        escalation_expose_tool=False,
        escalation_auto_trigger=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_provider_profile="openrouter",
        escalation_api_key="secret",
        escalation_api_key_env="BIG_KEY",
        escalation_chat_endpoint="/chat/completions",
        escalation_max_prompt_chars=111,
        escalation_max_response_tokens=222,
        escalation_temperature=0.3,
        escalation_timeout_sec=44,
        escalation_max_per_task=5,
        escalation_cooldown_turns=6,
        escalation_repeated_failure_threshold=7,
        escalation_require_tool_plan_evidence=False,
        escalation_redact_secrets=False,
    )

    kwargs = _escalation_harness_kwargs(config)

    assert kwargs == {
        "escalation_enabled": True,
        "escalation_expose_tool": False,
        "escalation_auto_trigger": True,
        "escalation_endpoint": "https://example.test/v1",
        "escalation_model": "big-model",
        "escalation_provider_profile": "openrouter",
        "escalation_api_key": "secret",
        "escalation_api_key_env": "BIG_KEY",
        "escalation_chat_endpoint": "/chat/completions",
        "escalation_max_prompt_chars": 111,
        "escalation_max_response_tokens": 222,
        "escalation_temperature": 0.3,
        "escalation_timeout_sec": 44,
        "escalation_max_per_task": 5,
        "escalation_cooldown_turns": 6,
        "escalation_repeated_failure_threshold": 7,
        "escalation_require_tool_plan_evidence": False,
        "escalation_redact_secrets": False,
    }


def test_escalation_model_config_resolves_api_key_env(monkeypatch):
    monkeypatch.setenv("BIG_MODEL_KEY", "secret-key")
    config = SimpleNamespace(
        escalation_endpoint="https://example.test/v1/",
        escalation_model="big-model",
        escalation_provider_profile="openrouter",
        escalation_api_key=None,
        escalation_api_key_env="BIG_MODEL_KEY",
        escalation_chat_endpoint="chat/completions",
        escalation_max_prompt_chars=48000,
        escalation_max_response_tokens=1600,
        escalation_temperature=0.2,
        escalation_timeout_sec=120,
    )

    resolved = build_escalation_model_config(config)

    assert resolved.endpoint == "https://example.test/v1"
    assert resolved.api_key == "secret-key"
    assert resolved.chat_endpoint == "/chat/completions"


def test_escalation_tool_registered_only_when_enabled_and_exposed():
    disabled = _harness(escalation_enabled=False)
    assert "escalate_to_bigger_model" not in build_registry(disabled).names()

    hidden = _harness(escalation_enabled=True, escalation_expose_tool=False)
    assert "escalate_to_bigger_model" not in build_registry(hidden).names()

    exposed = _harness(escalation_enabled=True, escalation_expose_tool=True)
    assert "escalate_to_bigger_model" in build_registry(exposed).names()


def test_escalation_tool_schema_has_no_provider_override_args():
    harness = _harness(escalation_enabled=True, escalation_expose_tool=True)
    registry = build_registry(harness)
    spec = registry.get("escalate_to_bigger_model")

    properties = spec.schema["properties"]

    assert set(properties) == {"reason", "question", "requested_output", "risk_level"}
    assert "endpoint" not in properties
    assert "model" not in properties
    assert "api_key" not in properties


def test_escalation_policy_requires_meaningful_evidence():
    harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
    harness.state.last_verifier_verdict = {"verdict": "fail", "message": "still broken"}

    decision = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")

    assert decision.allowed is True
    assert decision.trigger == "verifier_failure"


def test_escalation_policy_allows_explicit_request_with_prior_tool_evidence():
    harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
    harness.state.transcript_messages = [
        ConversationMessage(role="user", content="read temp/pong.py and patch it"),
        ConversationMessage(role="tool", name="file_read", content="read pong.py"),
        ConversationMessage(role="tool", name="file_patch", content="patched pong.py"),
        ConversationMessage(role="tool", name="memory_update", content="bookkeeping"),
    ]

    decision = EscalationPolicy(harness).can_escalate(
        reason="User requested escalation to a bigger model for loop recovery.",
        risk_level="low",
    )

    assert decision.allowed is True
    assert decision.trigger == "explicit_escalation_prior_evidence"


def test_escalation_policy_ignores_bookkeeping_for_explicit_request():
    harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
    harness.state.transcript_messages = [
        ConversationMessage(role="tool", name="memory_update", content="bookkeeping"),
        ConversationMessage(role="tool", name="escalate_to_bigger_model", content="blocked"),
    ]

    decision = EscalationPolicy(harness).can_escalate(
        reason="User requested escalation to a bigger model.",
        risk_level="low",
    )

    assert decision.allowed is False
    assert decision.trigger == "insufficient_evidence"


def test_escalation_policy_respects_cooldown_and_max_per_task():
    harness = _harness(escalation_enabled=True, escalation_max_per_task=1, escalation_cooldown_turns=10)
    harness.state.step_count = 3
    harness.state.scratchpad["_escalation_history"] = [{"step_count": 2, "verdict": "continue"}]
    harness.state.last_verifier_verdict = {"verdict": "fail"}

    maxed = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")
    assert maxed.allowed is False
    assert maxed.trigger == "max_per_task"

    harness.config.escalation_max_per_task = 3
    cooldown = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")
    assert cooldown.allowed is False
    assert cooldown.trigger == "cooldown"


def test_escalation_policy_blocks_pending_human_credential_gate():
    harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
    harness.state.last_verifier_verdict = {"verdict": "fail"}
    harness.state.pending_interrupt = {
        "question": "Need sudo password before continuing.",
        "reason": "credential_required",
    }

    decision = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")

    assert decision.allowed is False
    assert decision.trigger == "approval_required"


def test_escalation_policy_allows_existing_recovery_breadcrumbs():
    cases = [
        ("_last_schema_validation_hint", {"tool_name": "file_write"}, "schema_validation_repair"),
        ("_last_write_session_schema_failure", {"tool_name": "file_write"}, "write_session_schema_failure"),
        ("_last_backend_recovery", {"reason": "stream_halt"}, "backend_stream_recovery"),
        ("_read_loop_recovery_payload", {"tool_name": "file_read"}, "tool_loop_suppression"),
    ]
    for key, value, trigger in cases:
        harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
        harness.state.scratchpad[key] = value

        decision = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")

        assert decision.allowed is True
        assert decision.trigger == trigger


def test_escalation_policy_allows_wrong_path_and_fama_signals():
    harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
    harness.state.scratchpad["_recovery_metrics"] = {"tool_plan_wrong_path_count": 1}

    wrong_path = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")

    assert wrong_path.allowed is True
    assert wrong_path.trigger == "wrong_path"

    harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
    harness.state.scratchpad["_fama"] = {
        "signals": [{"kind": "remote_local_confusion", "failure_class": "remote_local_confusion"}]
    }

    remote = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")

    assert remote.allowed is True
    assert remote.trigger == "remote_local_confusion"


def test_escalation_policy_allows_patch_stall_evidence():
    harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
    harness.state.stagnation_counters = {"repeat_patch": 2}
    harness.state.last_verifier_verdict = {"verdict": "fail", "failure_mode": "test"}

    repeated_patch = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")

    assert repeated_patch.allowed is True
    assert repeated_patch.trigger in {"repeat_patch", "verifier_failure"}

    harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
    harness.state.scratchpad["_fama"] = {
        "signals": [{"kind": "write_session_stall", "failure_class": "write_session_stall"}]
    }

    write_session = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")

    assert write_session.allowed is True
    assert write_session.trigger == "write_session_stall"

    harness = _harness(escalation_enabled=True, escalation_require_tool_plan_evidence=True)
    harness.state.failure_events.append(
        FailureEvent(
            event_id="failure-1",
            timestamp=1.0,
            failure_class="write_session_stall",
            severity="recoverable",
            source="write_session",
            message="Write session stalled.",
            fama_kind="write_session_stall",
        )
    )

    direct_event = EscalationPolicy(harness).can_escalate(reason="stuck", risk_level="medium")

    assert direct_event.allowed is True
    assert direct_event.trigger == "write_session_stall"


def test_escalation_packet_redacts_and_omits_provider_config():
    harness = _harness(
        escalation_endpoint="https://secret-provider.example/v1",
        escalation_model="big-model",
        escalation_api_key="provider-secret",
    )
    harness.state.run_brief.original_task = "Fix the issue"
    harness.state.recent_errors.append("password is swordfish")
    harness.state.scratchpad["_tool_plan_observations_text"] = "token: abc123"

    packet = build_escalation_packet(
        harness,
        reason="stuck",
        question="what next?",
        requested_output="next_action",
        risk_level="medium",
        trigger="verifier_failure",
        max_chars=48000,
        redact_secrets=True,
    )
    packet_text = str(packet)

    assert "secret-provider" not in packet_text
    assert "provider-secret" not in packet_text
    assert "swordfish" not in packet_text
    assert "abc123" not in packet_text
    assert packet["task"]["original"] == "Fix the issue"


def test_escalation_packet_hard_budgets_large_context():
    harness = _harness()
    harness.state.run_brief.original_task = "A" * 5000
    harness.state.run_brief.effective_task = "B" * 5000
    harness.state.recent_errors.extend(["E" * 5000 for _ in range(5)])
    harness.state.scratchpad["_tool_plan_observations_text"] = "O" * 10000
    packet = build_escalation_packet(
        harness,
        reason="R" * 5000,
        question="Q" * 5000,
        requested_output="next_action",
        risk_level="medium",
        trigger="verifier_failure",
        max_chars=1200,
        redact_secrets=True,
    )

    packet_text = json.dumps(packet, ensure_ascii=True, sort_keys=True)

    assert len(packet_text) <= 1200
    assert packet["truncated"] is True


def test_escalation_response_rejects_unknown_tool():
    harness = _harness(escalation_enabled=True)
    harness.registry = build_registry(harness)
    response = """
    {
      "verdict": "continue",
      "confidence": 0.6,
      "recommended_next_action": {
        "type": "tool_call",
        "tool": "missing_tool",
        "args": {},
        "reason": "try it"
      }
    }
    """

    validated = parse_and_validate_escalation_response(response, harness=harness)

    assert validated.ok is False
    assert "not registered" in validated.error


def test_escalation_response_rejects_invalid_json():
    harness = _harness(escalation_enabled=True)

    validated = parse_and_validate_escalation_response("not json", harness=harness)

    assert validated.ok is False
    assert "Invalid escalation JSON" in validated.error
    assert harness.state.scratchpad["_recovery_metrics"]["escalation_invalid_json"] == 1


def test_escalation_response_rejects_runtime_hidden_tool():
    harness = _harness(escalation_enabled=True)
    harness.registry = build_registry(harness)
    response = """
    {
      "verdict": "continue",
      "confidence": 0.6,
      "recommended_next_action": {
        "type": "tool_call",
        "tool": "artifact_read",
        "args": {"artifact_id": "A0001"},
        "reason": "inspect artifact"
      }
    }
    """

    validated = parse_and_validate_escalation_response(response, harness=harness)

    assert validated.ok is False
    assert "runtime visibility" in validated.error


def test_escalation_response_visibility_check_does_not_mutate_scratchpad():
    harness = _harness(escalation_enabled=True)
    harness.registry = build_registry(harness)
    harness.state.scratchpad["_repeated_tool_loop_suppressed_tool"] = "file_read"
    harness.state.scratchpad["_repeated_tool_loop_suppressed_ttl"] = 2
    response = """
    {
      "verdict": "continue",
      "confidence": 0.6,
      "recommended_next_action": {
        "type": "tool_call",
        "tool": "file_read",
        "args": {"path": "README.md"},
        "reason": "inspect file"
      }
    }
    """

    validated = parse_and_validate_escalation_response(response, harness=harness)

    assert validated.ok is False
    assert harness.state.scratchpad["_repeated_tool_loop_suppressed_ttl"] == 2


def test_escalation_response_rejects_path_outside_cwd(tmp_path):
    harness = _harness(escalation_enabled=True)
    harness.state.cwd = str(tmp_path)
    harness.registry = build_registry(harness)
    response = """
    {
      "verdict": "continue",
      "confidence": 0.6,
      "recommended_next_action": {
        "type": "tool_call",
        "tool": "file_read",
        "args": {"path": "/etc/passwd"},
        "reason": "inspect file"
      }
    }
    """

    validated = parse_and_validate_escalation_response(response, harness=harness)

    assert validated.ok is False
    assert "outside cwd" in validated.error


def test_escalation_response_rejects_provider_override_args():
    harness = _harness(escalation_enabled=True)
    harness.registry = build_registry(harness)
    response = """
    {
      "verdict": "continue",
      "confidence": 0.6,
      "recommended_next_action": {
        "type": "tool_call",
        "tool": "file_read",
        "args": {"path": "README.md", "api_key": "secret", "model": "override"},
        "reason": "inspect file"
      }
    }
    """

    validated = parse_and_validate_escalation_response(response, harness=harness)

    assert validated.ok is False
    assert "forbidden escalation/provider keys" in validated.error


def test_escalation_response_rejects_auto_apply_patch():
    harness = _harness(escalation_enabled=True)
    response = """
    {
      "verdict": "propose_patch",
      "confidence": 0.6,
      "recommended_next_action": {
        "type": "patch",
        "apply": true,
        "patch": "diff --git a/a b/a"
      }
    }
    """

    validated = parse_and_validate_escalation_response(response, harness=harness)

    assert validated.ok is False
    assert "advice only" in validated.error


def test_escalation_response_rejects_final_ok_when_verifier_fails():
    harness = _harness(escalation_enabled=True)
    harness.state.last_verifier_verdict = {"verdict": "fail"}
    response = """
    {
      "verdict": "final_answer_ok",
      "confidence": 0.9,
      "recommended_next_action": {"type": "answer_review", "reason": "looks fine"}
    }
    """

    validated = parse_and_validate_escalation_response(response, harness=harness)

    assert validated.ok is False
    assert "latest verifier is failing" in validated.error


def test_escalation_service_blocks_disabled_before_model_config():
    harness = _harness(escalation_enabled=False, escalation_endpoint=None, escalation_model=None)

    import asyncio

    result = asyncio.run(
        EscalationService(harness).run(
            reason="stuck",
            question="What next?",
            requested_output="next_action",
            risk_level="medium",
        )
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert result["trigger"] == "disabled"
    assert "escalation_config_errors" not in harness.state.scratchpad.get("_recovery_metrics", {})


def test_escalation_tool_surfaces_blocked_service_result_as_tool_failure():
    harness = _harness(escalation_enabled=True, escalation_endpoint=None, escalation_model=None)

    import asyncio

    result = asyncio.run(
        escalate_to_bigger_model(
            reason="User requested escalation to a bigger model.",
            question="What next?",
            requested_output="next_action",
            harness=harness,
        )
    )

    assert result["success"] is False
    assert "escalation_result" in result["metadata"]
    assert result["metadata"]["escalation_result"]["status"] == "blocked"


def test_escalation_tool_converts_ask_human_advisory_to_needs_human(monkeypatch):
    class FakeEscalationService:
        def __init__(self, harness):
            self.harness = harness

        async def run(self, **kwargs):
            return {
                "success": True,
                "status": "advisory",
                "verdict": "ask_human",
                "confidence": 0.95,
                "failure_diagnosis": "Remote access is blocked.",
                "recommended_next_action": {
                    "type": "ask_human",
                    "tool": "ask_human",
                    "reason": "Ask the user to enable SSH tools or provide exported logs.",
                },
                "escalation_id": "esc-test",
            }

    monkeypatch.setattr("smallctl.harness.escalation_service.EscalationService", FakeEscalationService)
    harness = _harness(escalation_enabled=True)

    import asyncio

    result = asyncio.run(
        escalate_to_bigger_model(
            reason="Repeated SSH failures.",
            question="What next?",
            requested_output="next_action",
            state=harness.state,
            harness=harness,
        )
    )

    assert result["success"] is False
    assert result["status"] == "needs_human"
    assert result["metadata"]["reason"] == "escalation_ask_human"
    assert result["metadata"]["escalation_result"]["verdict"] == "ask_human"
    assert "enable SSH tools" in result["metadata"]["question"]


def test_escalation_tool_keeps_state_for_non_human_advisory(monkeypatch):
    class FakeEscalationService:
        def __init__(self, harness):
            self.harness = harness

        async def run(self, **kwargs):
            return {
                "success": True,
                "status": "advisory",
                "verdict": "next_action",
                "confidence": 0.8,
                "recommended_next_action": {
                    "type": "tool_call",
                    "tool": "file_read",
                    "args": {"path": "temp/output.txt"},
                    "reason": "Read the available local artifact.",
                },
                "escalation_id": "esc-test",
            }

    monkeypatch.setattr("smallctl.harness.escalation_service.EscalationService", FakeEscalationService)
    harness = _harness(escalation_enabled=True)

    import asyncio

    result = asyncio.run(
        escalate_to_bigger_model(
            reason="Repeated failures.",
            question="What next?",
            requested_output="next_action",
            state=harness.state,
            harness=harness,
        )
    )

    assert result["success"] is True
    assert result["advisory_text"].startswith("ESCALATION ADVISORY")


def test_escalation_service_success_with_mocked_provider(monkeypatch):
    class FakeClient:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            FakeClient.instances.append(self)

        async def stream_chat(self, messages, tools):
            self.messages = messages
            self.tools = tools
            yield {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                '{"verdict":"continue","confidence":0.7,'
                                '"failure_diagnosis":"Need focused evidence.",'
                                '"recommended_next_action":{"type":"tool_call","tool":"file_read",'
                                '"args":{"path":"src/smallctl/config.py"},"reason":"inspect config"}}'
                            )
                        }
                    }
                ]
            }

        @staticmethod
        def collect_stream(chunks, *, reasoning_mode="off", thinking_start_tag="<think>", thinking_end_tag="</think>"):
            return StreamResult(assistant_text=chunks[0]["choices"][0]["delta"]["content"])

    monkeypatch.setattr("smallctl.harness.escalation_service.OpenAICompatClient", FakeClient)
    harness = _harness(
        escalation_enabled=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_require_tool_plan_evidence=True,
        escalation_max_response_tokens=321,
        escalation_temperature=0.33,
    )
    harness.registry = build_registry(harness)
    harness.state.last_verifier_verdict = {"verdict": "fail", "message": "still broken"}

    import asyncio

    result = asyncio.run(
        EscalationService(harness).run(
            reason="Verifier failed twice.",
            question="What next?",
            requested_output="next_action",
            risk_level="medium",
        )
    )

    assert result["success"] is True
    assert result["status"] == "advisory"
    assert result["recommended_next_action"]["tool"] == "file_read"
    assert FakeClient.instances[0].tools == []
    assert FakeClient.instances[0].max_completion_tokens == 321
    assert FakeClient.instances[0].temperature == 0.33
    assert harness.state.scratchpad["_last_escalation"]["verdict"] == "continue"


def test_escalation_service_updates_active_subtask_next_action(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def stream_chat(self, messages, tools):
            yield {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                '{"verdict":"continue","confidence":0.7,'
                                '"recommended_next_action":{"type":"tool_call","tool":"file_read",'
                                '"args":{"path":"src/smallctl/config.py"},"reason":"inspect config first"}}'
                            )
                        }
                    }
                ]
            }

        @staticmethod
        def collect_stream(chunks, *, reasoning_mode="off", thinking_start_tag="<think>", thinking_end_tag="</think>"):
            return StreamResult(assistant_text=chunks[0]["choices"][0]["delta"]["content"])

    monkeypatch.setattr("smallctl.harness.escalation_service.OpenAICompatClient", FakeClient)
    harness = _harness(
        escalation_enabled=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_require_tool_plan_evidence=True,
    )
    active = Subtask(subtask_id="S1", title="Active", goal="Fix bug", status="active")
    harness.state.subtask_ledger = SubtaskLedger(task_id="task-1", subtasks=[active], active_subtask_id="S1")
    harness.subtask_ledger = SubtaskLedgerService(harness)
    harness.registry = build_registry(harness)
    harness.state.last_verifier_verdict = {"verdict": "fail", "message": "still broken"}

    import asyncio

    result = asyncio.run(
        EscalationService(harness).run(
            reason="Verifier failed.",
            question="What next?",
            requested_output="next_action",
        )
    )

    assert result["success"] is True
    assert active.evidence[-1].startswith("Escalation ")
    assert active.next_action == "Escalation recommends `file_read`: inspect config first"


def test_escalation_service_provider_failure_returns_structured_error(monkeypatch):
    class FailingClient:
        def __init__(self, **kwargs):
            pass

        async def stream_chat(self, messages, tools):
            raise RuntimeError("provider down")
            yield {}

        @staticmethod
        def collect_stream(chunks, *, reasoning_mode="off", thinking_start_tag="<think>", thinking_end_tag="</think>"):
            return StreamResult()

    monkeypatch.setattr("smallctl.harness.escalation_service.OpenAICompatClient", FailingClient)
    harness = _harness(
        escalation_enabled=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_require_tool_plan_evidence=True,
    )
    harness.registry = build_registry(harness)
    harness.state.last_verifier_verdict = {"verdict": "fail", "message": "still broken"}

    import asyncio

    result = asyncio.run(
        EscalationService(harness).run(
            reason="Verifier failed.",
            question="What next?",
            requested_output="next_action",
        )
    )

    assert result["success"] is False
    assert result["status"] == "provider_error"
    assert harness.state.scratchpad["_recovery_metrics"]["escalation_provider_failures"] == 1


def test_escalation_service_chunk_error_is_provider_failure(monkeypatch):
    class ChunkErrorClient:
        def __init__(self, **kwargs):
            pass

        async def stream_chat(self, messages, tools):
            yield {
                "type": "chunk_error",
                "error": "context overflow",
                "details": {"message": "too many tokens"},
            }

        @staticmethod
        def collect_stream(chunks, *, reasoning_mode="off", thinking_start_tag="<think>", thinking_end_tag="</think>"):
            return StreamResult()

    monkeypatch.setattr("smallctl.harness.escalation_service.OpenAICompatClient", ChunkErrorClient)
    harness = _harness(
        escalation_enabled=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_require_tool_plan_evidence=True,
    )
    harness.registry = build_registry(harness)
    harness.state.last_verifier_verdict = {"verdict": "fail", "message": "still broken"}

    import asyncio

    result = asyncio.run(
        EscalationService(harness).run(
            reason="Verifier failed.",
            question="What next?",
            requested_output="next_action",
        )
    )

    assert result["success"] is False
    assert result["status"] == "provider_error"
    assert result["error"] == "too many tokens"
    assert harness.state.scratchpad["_recovery_metrics"]["escalation_provider_failures"] == 1


def test_auto_escalation_triggers_on_write_session_stall(monkeypatch):
    calls = []

    class FakeEscalationService:
        def __init__(self, harness):
            self.harness = harness

        async def run(self, **kwargs):
            calls.append(kwargs)
            self.harness.state.scratchpad["_escalation_history"] = [
                {"step_count": self.harness.state.step_count, "trigger": "write_session_stall"}
            ]
            return {
                "success": True,
                "status": "advisory",
                "verdict": "continue",
                "escalation_id": "esc-test",
                "recommended_next_action": {
                    "type": "tool_call",
                    "tool": "file_read",
                    "args": {"path": "temp/pong.py"},
                    "reason": "read the patched region",
                },
            }

    monkeypatch.setattr("smallctl.harness.escalation_service.EscalationService", FakeEscalationService)
    harness = _harness(
        escalation_enabled=True,
        escalation_auto_trigger=True,
        escalation_require_tool_plan_evidence=True,
    )
    harness.state.failure_events.append(
        FailureEvent(
            event_id="failure-1",
            timestamp=1.0,
            failure_class="write_session_stall",
            severity="recoverable",
            source="write_session",
            message="Write session stalled.",
            fama_kind="write_session_stall",
            tool_name="file_patch",
        )
    )
    harness.state.last_verifier_verdict = {"verdict": "fail", "failure_mode": "test"}
    graph_state = GraphRunState(loop_state=harness.state, thread_id="t1", run_mode="loop")
    graph_state.last_tool_results.append(
        ToolExecutionRecord(
            operation_id="op1",
            tool_name="shell_exec",
            args={"command": "python3 temp/pong.py"},
            tool_call_id="call1",
            result=ToolEnvelope(success=False, error="verifier failed"),
        )
    )

    import asyncio

    triggered = asyncio.run(
        _maybe_auto_trigger_escalation_for_patch_stall(harness=harness, graph_state=graph_state)
    )

    assert triggered is True
    assert calls[0]["source"] == "auto"
    assert calls[0]["requested_output"] == "next_action"
    assert harness.state.recent_messages[-1].metadata["source"] == "auto_patch_stall"


def test_auto_escalation_triggers_on_explicit_request_completion_block(monkeypatch):
    calls = []

    class FakeEscalationService:
        def __init__(self, harness):
            self.harness = harness

        async def run(self, **kwargs):
            calls.append(kwargs)
            return {
                "success": True,
                "status": "advisory",
                "verdict": "continue",
                "escalation_id": "esc-completion",
                "recommended_next_action": {"type": "none", "reason": "run verifier before completion"},
            }

    monkeypatch.setattr("smallctl.harness.escalation_service.EscalationService", FakeEscalationService)
    harness = _harness(
        escalation_enabled=True,
        escalation_auto_trigger=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_repeated_failure_threshold=2,
    )
    harness.state.append_message(ConversationMessage(role="user", content="escalate to bigger model for help"))
    for idx in range(2):
        harness.state.failure_events.append(
            FailureEvent(
                event_id=f"complete-{idx}",
                timestamp=float(idx),
                failure_class="post_change_verification_required",
                severity="warning",
                source="tool_outcome",
                message="task_complete failed - Cannot complete until verification runs.",
                tool_name="task_complete",
            )
        )

    graph_state = GraphRunState(loop_state=harness.state, thread_id="t1", run_mode="loop")

    import asyncio

    triggered = asyncio.run(
        _maybe_auto_trigger_escalation_for_completion_block(harness=harness, graph_state=graph_state)
    )

    assert triggered is True
    assert calls[0]["source"] == "auto"
    assert calls[0]["requested_output"] == "next_action"
    assert harness.state.recent_messages[-1].metadata["source"] == "auto_completion_block"
    assert "esc-completion" in harness.state.recent_messages[-1].content


def test_explicit_escalation_completion_block_reports_disabled_config():
    harness = _harness(escalation_enabled=False, escalation_repeated_failure_threshold=2)
    harness.state.append_message(ConversationMessage(role="user", content="escalate to bigger model for help"))
    for idx in range(2):
        harness.state.failure_events.append(
            FailureEvent(
                event_id=f"complete-disabled-{idx}",
                timestamp=float(idx),
                failure_class="post_change_verification_required",
                severity="warning",
                source="tool_outcome",
                message="task_complete failed - Cannot complete until verification runs.",
                tool_name="task_complete",
            )
        )

    graph_state = GraphRunState(loop_state=harness.state, thread_id="t1", run_mode="loop")

    import asyncio

    triggered = asyncio.run(
        _maybe_auto_trigger_escalation_for_completion_block(harness=harness, graph_state=graph_state)
    )

    assert triggered is True
    assert harness.state.recent_messages[-1].metadata["recovery_kind"] == "escalation_config_blocker"
    assert "escalation is disabled" in harness.state.recent_messages[-1].content
    assert harness.state.scratchpad["_last_escalation"]["verdict"] == "config_error"


def test_escalation_service_invalid_response_is_traced(monkeypatch):
    traces = []

    class FakeRecorder:
        def record_escalation(self, harness, advisory):
            traces.append(advisory)
            return None

    class BadJsonClient:
        def __init__(self, **kwargs):
            pass

        async def stream_chat(self, messages, tools):
            yield {"choices": [{"delta": {"content": "not json"}}]}

        @staticmethod
        def collect_stream(chunks, *, reasoning_mode="off", thinking_start_tag="<think>", thinking_end_tag="</think>"):
            return StreamResult(assistant_text="not json")

    monkeypatch.setattr("smallctl.harness.escalation_service.OpenAICompatClient", BadJsonClient)
    monkeypatch.setattr("smallctl.harness.escalation_service.TrajectoryRecorder", FakeRecorder)
    harness = _harness(
        escalation_enabled=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_require_tool_plan_evidence=True,
    )
    harness.registry = build_registry(harness)
    harness.state.last_verifier_verdict = {"verdict": "fail", "message": "still broken"}

    import asyncio

    result = asyncio.run(
        EscalationService(harness).run(
            reason="Verifier failed.",
            question="What next?",
            requested_output="next_action",
        )
    )

    assert result["success"] is True
    assert result["status"] == "advisory"
    assert result["verdict"] == "need_more_evidence"
    assert traces[-1]["status"] == "advisory"
    assert traces[-1]["success"] is True
    assert harness.state.scratchpad["_last_escalation"]["verdict"] == "need_more_evidence"


def test_escalation_service_invalid_response_retry_does_not_consume_cooldown(monkeypatch):
    class SequencedClient:
        calls = 0

        def __init__(self, **kwargs):
            pass

        async def stream_chat(self, messages, tools):
            SequencedClient.calls += 1
            if SequencedClient.calls in {1, 2}:
                yield {"choices": [{"delta": {"content": "not json"}}]}
            else:
                yield {
                    "choices": [
                        {
                            "delta": {
                                "content": (
                                    '{"verdict":"continue","confidence":0.7,'
                                    '"recommended_next_action":{"type":"none","reason":"inspect more evidence"}}'
                                )
                            }
                        }
                    ]
                }

        @staticmethod
        def collect_stream(chunks, *, reasoning_mode="off", thinking_start_tag="<think>", thinking_end_tag="</think>"):
            if SequencedClient.calls in {1, 2}:
                return StreamResult(assistant_text="not json")
            return StreamResult(
                assistant_text=(
                    '{"verdict":"continue","confidence":0.7,'
                    '"recommended_next_action":{"type":"none","reason":"inspect more evidence"}}'
                )
            )

    monkeypatch.setattr("smallctl.harness.escalation_service.OpenAICompatClient", SequencedClient)
    harness = _harness(
        escalation_enabled=True,
        escalation_endpoint="https://example.test/v1",
        escalation_model="big-model",
        escalation_require_tool_plan_evidence=True,
        escalation_cooldown_turns=0,
    )
    harness.registry = build_registry(harness)
    harness.state.last_verifier_verdict = {"verdict": "fail", "message": "still broken"}

    import asyncio

    first = asyncio.run(
        EscalationService(harness).run(
            reason="Verifier failed.",
            question="What next?",
            requested_output="next_action",
        )
    )
    second = asyncio.run(
        EscalationService(harness).run(
            reason="Verifier failed again.",
            question="What next now?",
            requested_output="next_action",
        )
    )

    assert first["success"] is True
    assert first["status"] == "advisory"
    assert first["verdict"] == "need_more_evidence"
    assert second["success"] is True
    assert second["status"] == "advisory"
    assert second["verdict"] == "continue"
    # Both calls recorded in history because fallback counts as success
    assert len(harness.state.scratchpad.get("_escalation_history", [])) == 2
    assert len(harness.state.scratchpad.get("_escalation_attempt_history", [])) == 1
    assert harness.state.scratchpad["_recovery_metrics"]["escalation_validation_retries"] == 1


def test_escalation_service_config_error_is_metriced_and_traced(monkeypatch):
    traces = []

    class FakeRecorder:
        def record_escalation(self, harness, advisory):
            traces.append(advisory)
            return None

    monkeypatch.setattr("smallctl.harness.escalation_service.TrajectoryRecorder", FakeRecorder)
    harness = _harness(escalation_enabled=True, escalation_endpoint=None, escalation_model=None)

    import asyncio

    result = asyncio.run(
        EscalationService(harness).run(
            reason="Verifier failed.",
            question="What next?",
            requested_output="next_action",
        )
    )

    assert result["success"] is False
    assert result["status"] == "blocked"
    assert harness.state.scratchpad["_recovery_metrics"]["escalation_config_errors"] == 1
    assert harness.state.scratchpad["_last_escalation"]["verdict"] == "config_error"
    assert traces[-1]["status"] == "config_error"


def test_auto_trigger_tool_loop_injects_advisory(monkeypatch):
    class FakeService:
        def __init__(self, harness):
            self.harness = harness

        async def run(self, **kwargs):
            assert kwargs["source"] == "auto"
            return {
                "success": True,
                "status": "advisory",
                "verdict": "continue",
                "confidence": 0.8,
                "recommended_next_action": {"type": "none", "reason": "pause and inspect"},
                "escalation_id": "esc_test",
            }

    monkeypatch.setattr("smallctl.harness.escalation_service.EscalationService", FakeService)
    harness = _harness(escalation_enabled=True, escalation_auto_trigger=True)
    harness.state.last_verifier_verdict = {"verdict": "fail", "failure_mode": "test"}
    pending = PendingToolCall(
        tool_name="file_read",
        args={"path": "README.md"},
        raw_arguments='{"path":"README.md"}',
        source="assistant",
    )

    import asyncio

    triggered = asyncio.run(
        _maybe_auto_trigger_escalation_for_tool_loop(
            harness=harness,
            pending=pending,
            repeat_error="Guard tripped: repeated file_read",
        )
    )

    assert triggered is True
    assert harness.state.scratchpad["_tool_loop_suppression"]["tool_name"] == "file_read"
    assert harness.state.recent_messages[-1].metadata["recovery_kind"] == "escalation_advisory"
    assert "esc_test" in harness.state.recent_messages[-1].content


def test_escalation_trajectory_payload(tmp_path):
    harness = _harness(escalation_model="big-model", escalation_provider_profile="openrouter")
    harness.state.thread_id = "thread-test"
    harness.state.run_brief.original_task = "Fix a bug"
    harness.state.scratchpad["_model_name"] = "small-model"
    harness.state.scratchpad["_last_escalation"] = {
        "id": "esc_test",
        "trigger": "verifier_failure",
        "verdict": "continue",
    }
    harness.state.scratchpad["_recovery_metrics"] = {
        "escalation_prompt_chars": 123,
        "escalation_response_chars": 45,
        "escalation_wall_clock_sec": 1.2,
    }
    advisory = {
        "success": True,
        "escalation_id": "esc_test",
        "verdict": "continue",
        "confidence": 0.7,
        "recommended_next_action": {"type": "none"},
    }

    path = TrajectoryRecorder(tmp_path).record_escalation(harness, advisory)
    payload = json.loads(path.read_text(encoding="utf-8").strip())

    assert payload["type"] == "escalation"
    assert payload["small_model"] == "small-model"
    assert payload["big_model"] == "big-model"
    assert payload["provider_profile"] == "openrouter"
    assert payload["packet_chars"] == 123
    assert payload["accepted_by_harness"] is True
    assert payload["validator_result"] == "pass"
