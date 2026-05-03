from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.config import resolve_config
from smallctl.fama.detectors import (
    detect_backend_stream_halt,
    detect_bad_tool_args,
    detect_context_drift,
    detect_tool_output_misread,
    record_bad_tool_arg_failure,
)
from smallctl.fama.runtime import expire_for_turn, observe_tool_result
from smallctl.fama.state import active_mitigation_names
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import LoopState


class _Config:
    fama_enabled = True
    fama_mode = "lite"
    fama_default_ttl_steps = 2
    fama_max_active_mitigations = 4
    fama_signal_window = 8
    fama_done_gate_on_failure = True
    fama_llm_judge_enabled = False
    fama_llm_judge_min_severity = 3
    loop_guard_stagnation_threshold = 3


class _JudgeConfig(_Config):
    fama_llm_judge_enabled = True
    fama_llm_judge_min_severity = 2


class _FakeJudgeClient:
    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls = 0

    async def stream_chat(self, *, messages, tools):
        self.calls += 1
        assert tools == []
        prompt = messages[-1]["content"]
        assert "recent_tool_results" in prompt
        yield {
            "type": "chunk",
            "data": {
                "choices": [
                    {
                        "delta": {
                            "content": self.verdict,
                        }
                    }
                ]
            },
        }


def _harness(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        config=_Config(),
        _runlog=lambda *args, **kwargs: None,
    )


def test_fama_bad_tool_args_requires_repeated_validation_failure() -> None:
    state = LoopState(step_count=2)
    result = ToolEnvelope(
        success=False,
        error="Missing required field: path",
        metadata={"tool_name": "file_write"},
    )

    record_bad_tool_arg_failure(state, tool_name="file_write", result=result)
    assert detect_bad_tool_args(state, tool_name="file_write", result=result, threshold=2) is None

    record_bad_tool_arg_failure(state, tool_name="file_write", result=result)
    signal = detect_bad_tool_args(state, tool_name="file_write", result=result, threshold=2)

    assert signal is not None
    assert signal.kind.value == "bad_tool_args"
    assert signal.tool_name == "file_write"


def test_fama_observe_routes_bad_tool_args_after_repeated_failure() -> None:
    state = LoopState(step_count=3)
    result = ToolEnvelope(
        success=False,
        error="Field 'limit' expected type 'integer'",
        metadata={"reason": "validation_error"},
    )
    service = SimpleNamespace(harness=_harness(state))

    asyncio.run(observe_tool_result(service, tool_name="artifact_read", result=result))
    asyncio.run(observe_tool_result(service, tool_name="artifact_read", result=result))

    assert "micro_plan_capsule" in active_mitigation_names(state)
    assert [item["kind"] for item in state.scratchpad["_fama"]["signals"]] == ["bad_tool_args"]


def test_fama_tool_output_misread_from_task_complete_without_verifier_failure() -> None:
    state = LoopState(step_count=4)
    result = ToolEnvelope(
        success=False,
        error="The answer was not found in tool output.",
        metadata={"reason": "lookup_answer_missing"},
    )

    signal = detect_tool_output_misread(state, tool_name="task_complete", result=result)

    assert signal is not None
    assert signal.kind.value == "tool_output_misread"


def test_fama_repeated_backend_stream_halt_emits_signal_and_mitigation() -> None:
    state = LoopState(step_count=5)
    state.scratchpad["_last_stream_halted_without_done"] = True
    state.scratchpad["_last_stream_halt_reason"] = "first_token_timeout"

    assert detect_backend_stream_halt(state, threshold=2) is None
    signal = detect_backend_stream_halt(state, threshold=2)
    assert signal is not None
    assert signal.kind.value == "backend_stream_halt"

    state.scratchpad["_fama"].pop("seen_signatures", None)
    expire_for_turn(_harness(state), mode="loop")

    assert "micro_plan_capsule" in active_mitigation_names(state)


def test_fama_context_drift_skips_human_gated_statuses() -> None:
    state = LoopState(step_count=6)
    state.scratchpad["_task_divergence_nudged"] = True
    state.scratchpad["_last_task_handoff"] = {
        "task_mode": "remote_execute",
        "raw_user_task": "update the remote site",
    }
    state.pending_interrupt = {"kind": "ask_human", "question": "Proceed?"}

    assert detect_context_drift(state) is None

    state.pending_interrupt = None
    signal = detect_context_drift(state)
    assert signal is not None
    assert signal.kind.value == "context_drift"


def test_fama_llm_judge_config_defaults_off() -> None:
    config = resolve_config({})

    assert config.fama_llm_judge_enabled is False
    assert config.fama_llm_judge_min_severity == 3


def test_fama_llm_judge_can_add_known_signal_when_enabled() -> None:
    state = LoopState(step_count=7)
    result = ToolEnvelope(
        success=False,
        error="Field 'limit' expected type 'integer'",
        metadata={"reason": "validation_error"},
    )
    client = _FakeJudgeClient("tool_output_misread")
    harness = SimpleNamespace(
        state=state,
        config=_JudgeConfig(),
        client=client,
        summarizer_client=None,
        _events=[],
    )
    harness._runlog = lambda event, message, **data: harness._events.append((event, data))
    service = SimpleNamespace(harness=harness)

    asyncio.run(observe_tool_result(service, tool_name="artifact_read", result=result))
    asyncio.run(observe_tool_result(service, tool_name="artifact_read", result=result))

    kinds = [item["kind"] for item in state.scratchpad["_fama"]["signals"]]
    assert kinds == ["bad_tool_args", "tool_output_misread"]
    assert client.calls == 1
    assert any(
        event == "fama_llm_judge_verdict" and data["verdict"] == "tool_output_misread"
        for event, data in harness._events
    )


def test_fama_llm_judge_skips_human_gated_state() -> None:
    state = LoopState(step_count=8)
    state.pending_interrupt = {"kind": "ask_human", "question": "Approve?"}
    result = ToolEnvelope(
        success=False,
        error="The answer was not found in tool output.",
        metadata={"reason": "lookup_answer_missing"},
    )
    client = _FakeJudgeClient("context_drift")
    harness = SimpleNamespace(
        state=state,
        config=_JudgeConfig(),
        client=client,
        summarizer_client=None,
        _events=[],
    )
    harness._runlog = lambda event, message, **data: harness._events.append((event, data))

    asyncio.run(
        observe_tool_result(
            SimpleNamespace(harness=harness),
            tool_name="task_complete",
            result=result,
        )
    )

    kinds = [item["kind"] for item in state.scratchpad["_fama"]["signals"]]
    assert kinds == ["tool_output_misread"]
    assert client.calls == 0
    assert any(
        event == "fama_llm_judge_verdict" and data["reason"] == "human_gated"
        for event, data in harness._events
    )
