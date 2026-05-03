from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.fama.state import active_mitigation_names
from smallctl.harness import tool_result_flow
from smallctl.harness.tool_dispatch import chat_mode_tools, dispatch_tool_call
from smallctl.harness.tool_visibility import resolve_turn_tool_exposure
from smallctl.models.conversation import ConversationMessage
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


class _Config:
    fama_enabled = True
    fama_mode = "lite"
    fama_default_ttl_steps = 2
    fama_max_active_mitigations = 2
    fama_signal_window = 8
    fama_done_gate_on_failure = True


class _DisabledConfig:
    fama_enabled = False
    fama_mode = "lite"
    fama_default_ttl_steps = 2
    fama_max_active_mitigations = 2
    fama_signal_window = 8
    fama_done_gate_on_failure = True


def _schema(name: str) -> dict[str, object]:
    return {"type": "function", "function": {"name": name, "description": "", "parameters": {}}}


def _tool_names(schemas: list[dict[str, object]]) -> list[str]:
    return [str(item["function"]["name"]) for item in schemas]


def _harness(state: LoopState) -> SimpleNamespace:
    events: list[tuple[str, dict[str, object]]] = []
    return SimpleNamespace(
        state=state,
        config=_Config(),
        _events=events,
        _runlog=lambda event, message, **kwargs: events.append((event, kwargs)),
        registry=SimpleNamespace(
            names=lambda: ["task_complete", "task_fail"],
            export_openai_tools=lambda **kwargs: [_schema("task_complete"), _schema("task_fail")],
            get=lambda name: None,
        ),
        dispatcher=SimpleNamespace(
            dispatch=lambda tool_name, args: (_ for _ in ()).throw(AssertionError("dispatcher should not run"))
        ),
    )


def test_fama_records_early_stop_from_task_complete_verifier_failure(monkeypatch) -> None:
    state = LoopState(step_count=5)
    service = SimpleNamespace(harness=_harness(state))
    message = ConversationMessage(role="tool", name="task_complete", content="blocked")

    async def _persist(*args, **kwargs):
        return message

    monkeypatch.setattr(tool_result_flow, "_persist_artifact_result", _persist)
    result = ToolEnvelope(
        success=False,
        error="Cannot complete the task while the latest verifier verdict is still failing.",
        metadata={"last_verifier_verdict": {"verdict": "fail", "command": "pytest"}},
    )

    asyncio.run(
        tool_result_flow.record_result(
            service,
            tool_name="task_complete",
            tool_call_id="call-1",
            result=result,
            operation_id="op-1",
        )
    )

    payload = state.scratchpad["_fama"]
    assert payload["signals"][-1]["kind"] == "early_stop"
    assert active_mitigation_names(state) == {"done_gate", "acceptance_checklist_capsule"}
    assert state.scratchpad["_context_invalidations"][-1]["reason"] == "fama_failure_detected"


def test_fama_observe_runs_for_reused_result_path(monkeypatch) -> None:
    state = LoopState(step_count=1)
    service = SimpleNamespace(harness=_harness(state))
    reused = ConversationMessage(role="tool", name="task_complete", content="cached")

    monkeypatch.setattr(tool_result_flow, "_handle_reused_artifact_result", lambda *args, **kwargs: reused)
    monkeypatch.setattr(
        tool_result_flow,
        "_persist_artifact_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("persist should not run")),
    )
    result = ToolEnvelope(
        success=False,
        error="Cannot complete the task while the latest verifier verdict is still failing.",
        metadata={"last_verifier_verdict": {"verdict": "fail"}},
    )

    message = asyncio.run(
        tool_result_flow.record_result(
            service,
            tool_name="task_complete",
            tool_call_id="call-1",
            result=result,
            operation_id="op-1",
        )
    )

    assert message is reused
    assert state.scratchpad["_fama"]["signals"][-1]["kind"] == "early_stop"


def test_fama_done_gate_hides_task_complete_from_chat_terminal_only_exposure(monkeypatch) -> None:
    state = LoopState()
    state.scratchpad["_fama"] = {
        "version": 1,
        "signals": [],
        "active_mitigations": [
            {
                "name": "done_gate",
                "reason": "verifier failed",
                "source_signal": "early_stop:0",
                "activated_step": 0,
                "expires_after_step": 2,
                "priority": 50,
            }
        ],
        "last_observed_step": 0,
    }
    harness = _harness(state)
    harness.client = SimpleNamespace(model="qwen")
    harness._current_user_task = lambda: "thanks"

    monkeypatch.setattr("smallctl.harness.tool_dispatch.chat_mode_requires_tools", lambda harness, task: False)

    assert _tool_names(chat_mode_tools(harness)) == ["task_fail"]
    exposure = resolve_turn_tool_exposure(harness, "chat")
    assert exposure["names"] == ["task_fail"]
    assert any(event == "fama_tool_exposure_applied" for event, _ in harness._events)


def test_fama_done_gate_dispatch_blocks_hidden_task_complete() -> None:
    state = LoopState()
    state.last_verifier_verdict = {"verdict": "fail", "command": "pytest"}
    state.scratchpad["_fama"] = {
        "version": 1,
        "signals": [],
        "active_mitigations": [
            {
                "name": "done_gate",
                "reason": "verifier failed",
                "source_signal": "early_stop:0",
                "activated_step": 0,
                "expires_after_step": 2,
                "priority": 50,
            }
        ],
        "last_observed_step": 0,
    }

    blocked = asyncio.run(dispatch_tool_call(_harness(state), "task_complete", {"message": "done"}))

    assert blocked.success is False
    assert blocked.metadata["reason"] == "fama_done_gate"


def test_fama_passing_verifier_clears_done_gate(monkeypatch) -> None:
    state = LoopState()
    state.scratchpad["_fama"] = {
        "version": 1,
        "signals": [],
        "active_mitigations": [
            {
                "name": "done_gate",
                "reason": "verifier failed",
                "source_signal": "early_stop:0",
                "activated_step": 0,
                "expires_after_step": 2,
                "priority": 50,
            }
        ],
        "last_observed_step": 0,
    }
    state.last_verifier_verdict = {"verdict": "pass", "command": "pytest"}
    service = SimpleNamespace(harness=_harness(state))
    message = ConversationMessage(role="tool", name="shell_exec", content="ok")

    async def _persist(*args, **kwargs):
        return message

    monkeypatch.setattr(tool_result_flow, "_persist_artifact_result", _persist)

    asyncio.run(
        tool_result_flow.record_result(
            service,
            tool_name="shell_exec",
            tool_call_id="call-1",
            result=ToolEnvelope(success=True, output="ok", metadata={"verifier_verdict": "pass"}),
        )
    )

    assert "done_gate" not in active_mitigation_names(state)


def test_fama_done_gate_requires_matching_verifier_pass_to_clear(monkeypatch) -> None:
    state = LoopState()
    service = SimpleNamespace(harness=_harness(state))
    message = ConversationMessage(role="tool", name="shell_exec", content="ok")

    async def _persist(*args, **kwargs):
        return message

    monkeypatch.setattr(tool_result_flow, "_persist_artifact_result", _persist)

    failed = ToolEnvelope(
        success=False,
        error="Cannot complete the task while the latest verifier verdict is still failing.",
        metadata={"last_verifier_verdict": {"verdict": "fail", "command": "pytest tests/test_fama.py"}},
    )
    asyncio.run(
        tool_result_flow.record_result(
            service,
            tool_name="task_complete",
            tool_call_id="call-1",
            result=failed,
            operation_id="op-1",
        )
    )
    assert "done_gate" in active_mitigation_names(state)

    state.last_verifier_verdict = {"verdict": "pass", "command": "ruff check src"}
    asyncio.run(
        tool_result_flow.record_result(
            service,
            tool_name="shell_exec",
            tool_call_id="call-2",
            result=ToolEnvelope(
                success=True,
                output="ok",
                metadata={"last_verifier_verdict": {"verdict": "pass", "command": "ruff check src"}},
            ),
            operation_id="op-2",
        )
    )
    assert "done_gate" in active_mitigation_names(state)

    state.last_verifier_verdict = {"verdict": "pass", "command": "pytest tests/test_fama.py"}
    asyncio.run(
        tool_result_flow.record_result(
            service,
            tool_name="shell_exec",
            tool_call_id="call-3",
            result=ToolEnvelope(
                success=True,
                output="ok",
                metadata={
                    "last_verifier_verdict": {
                        "verdict": "pass",
                        "command": "pytest tests/test_fama.py",
                    }
                },
            ),
            operation_id="op-3",
        )
    )
    assert "done_gate" not in active_mitigation_names(state)


def test_fama_disabled_observe_is_noop(monkeypatch) -> None:
    state = LoopState()
    harness = _harness(state)
    harness.config = _DisabledConfig()
    service = SimpleNamespace(harness=harness)
    message = ConversationMessage(role="tool", name="task_complete", content="blocked")

    async def _persist(*args, **kwargs):
        return message

    monkeypatch.setattr(tool_result_flow, "_persist_artifact_result", _persist)

    asyncio.run(
        tool_result_flow.record_result(
            service,
            tool_name="task_complete",
            tool_call_id="call-1",
            result=ToolEnvelope(
                success=False,
                error="Cannot complete the task while the latest verifier verdict is still failing.",
                metadata={"last_verifier_verdict": {"verdict": "fail"}},
            ),
        )
    )

    assert "_fama" not in state.scratchpad
