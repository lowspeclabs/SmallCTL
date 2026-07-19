from __future__ import annotations

from typing import Any

from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.context.policy import estimate_text_tokens
from smallctl.models.conversation import ConversationMessage
from smallctl.state import LoopState


def _tool_call(call_id: str, *, name: str = "shell_exec") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"command": "make test"}'},
    }


def _assert_pairing_invariant(messages: list[dict[str, Any]]) -> None:
    offered: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if isinstance(tool_call, dict) and tool_call.get("id"):
                    offered.add(str(tool_call["id"]))
        elif message.get("role") == "tool":
            assert message.get("tool_call_id"), "id-less role=tool message emitted"
            answered.add(str(message["tool_call_id"]))
    assert offered == answered


def _recompute_emitted_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += 4  # per-message framing overhead
        content = message.get("content")
        if isinstance(content, str) and content:
            total += estimate_text_tokens(content)
        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                function = tool_call.get("function") or {}
                total += estimate_text_tokens(str(function.get("name") or ""))
                total += estimate_text_tokens(str(function.get("arguments") or ""))
    return total


# ---------------------------------------------------------------------------
# M9 — token estimator must not undercount CJK/emoji
# ---------------------------------------------------------------------------


def test_estimator_ascii_only_unchanged() -> None:
    assert estimate_text_tokens("") == 0
    assert estimate_text_tokens("x" * 1000) == 401
    assert estimate_text_tokens("hello world") == int(11 * 0.4) + 1


def test_estimator_cjk_counts_roughly_one_token_per_char() -> None:
    assert estimate_text_tokens("漢字" * 100) >= 150


def test_estimator_emoji_uplifted() -> None:
    # ~1 token per emoji codepoint, vs ~41 from the old flat 0.4/char estimate.
    assert estimate_text_tokens("😀" * 100) >= 100


def test_estimator_mixed_ascii_cjk_sanity() -> None:
    mixed = "hello " + "漢字" * 50
    estimate = estimate_text_tokens(mixed)
    # 6 ASCII chars (~2 tokens) + 100 CJK chars (~100 tokens)
    assert 100 <= estimate <= 160
    # Strictly larger than the old buggy flat-0.4 estimate would give.
    assert estimate > int(len(mixed) * 0.4) + 1


# ---------------------------------------------------------------------------
# H20 — hard prompt ceiling enforced against the final emitted message list
# ---------------------------------------------------------------------------


def _oversized_task_state(task_chars: int = 20000) -> LoopState:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "fix the remote deployment " + ("x" * task_chars)
    state.recent_messages = [
        ConversationMessage(role="assistant", content="I will inspect the host.", tool_calls=[_tool_call("call_1")]),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_1", content="ok\n" * 40),
    ]
    state.scratchpad["_fresh_tool_outputs"] = [
        {"tool_name": "shell_exec", "artifact_id": "", "content": "evidence line\n" * 40}
    ]
    return state


def test_hard_ceiling_holds_with_oversized_task_goal_and_wrappers() -> None:
    policy = ContextPolicy(max_prompt_tokens=2048)
    state = _oversized_task_state()
    system_prompt = "SYSTEM PROMPT\n" + ("policy directive line\n" * 96)  # ~2 KB

    assembly = PromptAssembler(policy).build_messages(
        state=state,
        system_prompt=system_prompt,
        include_structured_sections=True,
    )

    recomputed = _recompute_emitted_tokens(assembly.messages)
    assert recomputed <= 2048
    assert assembly.estimated_prompt_tokens <= 2048
    # The assembler's own estimate must match a from-scratch recompute.
    assert assembly.estimated_prompt_tokens == recomputed

    # The preserved task-goal message is present but bounded and marked.
    rendered = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)
    assert "fix the remote deployment" in rendered
    preserved = [
        message
        for message in assembly.messages
        if "fix the remote deployment" in str(message.get("content") or "")
    ]
    assert preserved
    assert any("[truncated]" in str(message.get("content") or "") for message in preserved)

    # Wrappers are now charged into the section budget. Fresh tool outputs get
    # their own <current-evidence> frame (M12), so either wrapper may carry
    # the framing charge depending on which lanes are present.
    assert (
        assembly.section_tokens.get("knowledge_base_wrapper", 0)
        + assembly.section_tokens.get("current_evidence_wrapper", 0)
    ) > 0
    assert assembly.section_tokens.get("mission_recap", 0) > 0

    _assert_pairing_invariant(assembly.messages)


def test_hard_ceiling_final_pass_truncates_preserved_messages() -> None:
    policy = ContextPolicy(max_prompt_tokens=1024)
    state = _oversized_task_state()
    system_prompt = "SYSTEM PROMPT " + ("policy " * 60)  # ~0.4 KB

    assembly = PromptAssembler(policy).build_messages(
        state=state,
        system_prompt=system_prompt,
        include_structured_sections=True,
    )

    assert _recompute_emitted_tokens(assembly.messages) <= 1024
    assert assembly.estimated_prompt_tokens <= 1024
    assert any("[truncated]" in str(message.get("content") or "") for message in assembly.messages)
    assert any(
        drop.reason == "hard_prompt_ceiling_truncation"
        for drop in (assembly.frame.drop_log if assembly.frame else [])
    )
    # Truncation sufficed, so no typed budget failure is recorded.
    assert state.prompt_budget.pressure_level == ""
    _assert_pairing_invariant(assembly.messages)


def test_mandatory_overflow_surfaces_typed_budget_failure() -> None:
    policy = ContextPolicy(max_prompt_tokens=512)
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "small task"
    # The mandatory system prompt alone exceeds the hard ceiling.
    system_prompt = "SYSTEM PROMPT\n" + ("x" * 20000)

    assembly = PromptAssembler(policy).build_messages(
        state=state,
        system_prompt=system_prompt,
        include_structured_sections=False,
    )

    # Honest overshoot: the estimate is not silently clamped below the ceiling.
    assert assembly.estimated_prompt_tokens > 512
    assert assembly.estimated_prompt_tokens == _recompute_emitted_tokens(assembly.messages)

    # Typed failure surfaced through the snapshot, scratchpad metrics, and frame drop log.
    assert state.prompt_budget.pressure_level == "over_budget"
    failure = state.scratchpad["_context_metrics"]["prompt_budget_failure"]
    assert failure["type"] == "prompt_budget_failure"
    assert failure["reason"] == "mandatory_content_exceeds_hard_ceiling"
    assert failure["max_prompt_tokens"] == 512
    assert failure["estimated_prompt_tokens"] == assembly.estimated_prompt_tokens
    assert any(
        drop.lane == "prompt_hard_ceiling" and drop.reason == "prompt_budget_failure"
        for drop in (assembly.frame.drop_log if assembly.frame else [])
    )


def test_no_budget_failure_when_mandatory_content_fits() -> None:
    policy = ContextPolicy(max_prompt_tokens=4096)
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "small task"

    assembly = PromptAssembler(policy).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
        include_structured_sections=False,
    )

    assert assembly.estimated_prompt_tokens <= 4096
    assert state.prompt_budget.pressure_level == ""
    assert "prompt_budget_failure" not in state.scratchpad.get("_context_metrics", {})
