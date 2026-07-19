from __future__ import annotations

from typing import Any

from smallctl.client import get_provider_adapter
from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.context.messages_compact_helpers import collapse_repeated_tool_failures
from smallctl.models.conversation import ConversationMessage
from smallctl.state import LoopState


def _tool_call(call_id: str, *, name: str = "shell_exec") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"command": "make test"}'},
    }


def _tool_call_ids(messages: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    offered: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if isinstance(tool_call, dict) and tool_call.get("id"):
                    offered.add(str(tool_call["id"]))
        elif message.get("role") == "tool":
            if message.get("tool_call_id"):
                answered.add(str(message["tool_call_id"]))
    return offered, answered


def _assert_pairing_invariant(messages: list[dict[str, Any]]) -> None:
    offered, answered = _tool_call_ids(messages)
    assert offered == answered
    for message in messages:
        if message.get("role") == "tool":
            assert message.get("tool_call_id"), "id-less role=tool message emitted"


def _assemble(
    messages: list[ConversationMessage],
    *,
    max_prompt_tokens: int = 4096,
    recent_message_limit: int = 30,
):
    policy = ContextPolicy(max_prompt_tokens=max_prompt_tokens)
    policy.recalculate_quotas(max_prompt_tokens)
    state = LoopState(cwd="/tmp")
    state.recent_messages = list(messages)
    assembler = PromptAssembler(policy)
    return assembler.build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
        recent_message_limit=recent_message_limit,
        include_structured_sections=False,
    )


def _paired_shell_failure(call_id: str, content: str) -> list[ConversationMessage]:
    return [
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call(call_id)]),
        ConversationMessage(
            role="tool",
            name="shell_exec",
            tool_call_id=call_id,
            content=content,
        ),
    ]


def test_collapse_repeated_tool_failures_preserves_pairing() -> None:
    failure = "Error: command failed with exit code 1: boom"
    messages = [
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call("call_1")]),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_1", content=failure),
        ConversationMessage(role="assistant", content="Trying again.", tool_calls=[_tool_call("call_2")]),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_2", content=failure),
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call("call_3")]),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_3", content=failure),
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call("call_4")]),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_4", content=failure),
    ]

    result = collapse_repeated_tool_failures(messages)

    offered: set[str] = set()
    answered: set[str] = set()
    for message in result:
        if message.role == "assistant":
            offered.update(str(tc["id"]) for tc in message.tool_calls if tc.get("id"))
        elif message.role == "tool":
            assert message.tool_call_id, "collapse emitted an id-less role=tool message"
            answered.add(str(message.tool_call_id))
    assert offered == answered == {"call_3", "call_4"}

    # The assistant whose only call was collapsed away (no content) is dropped;
    # the one with visible content survives with an empty tool_calls list.
    surviving = [m for m in result if m.role == "assistant"]
    assert [m.content for m in surviving] == ["Trying again.", None, None]
    assert surviving[0].tool_calls == []


def test_collapse_summary_is_user_role_not_idless_tool() -> None:
    failure = "Error: command failed with exit code 1: boom"
    messages: list[ConversationMessage] = []
    for index in range(4):
        messages.extend(_paired_shell_failure(f"call_{index}", failure))

    result = collapse_repeated_tool_failures(messages)

    summaries = [m for m in result if "collapsed to save tokens" in str(m.content or "")]
    assert len(summaries) == 1
    assert summaries[0].role == "user"
    assert all(m.role != "tool" or m.tool_call_id for m in result)


def test_transcript_budget_drops_tool_bundles_atomically() -> None:
    messages = [ConversationMessage(role="user", content="start the run")]
    for index in range(12):
        content = f"Error: attempt {index} failed " + ("x" * 280)
        messages.extend(_paired_shell_failure(f"call_{index}", content))

    assembly = _assemble(messages, max_prompt_tokens=4096)

    # The 4k policy leaves a ~1152-token transcript cap, so the oldest bundles
    # must be dropped. Dropping is atomic: a surviving assistant tool_calls id
    # always has its tool message and vice versa.
    _assert_pairing_invariant(assembly.messages)
    offered, answered = _tool_call_ids(assembly.messages)
    assert offered == answered
    assert offered, "expected some tool pairs to survive"
    assert len(offered) < 12, "expected the transcript cap to drop old bundles"
    for call_id in offered:
        index = int(call_id.split("_")[1])
        assert f"attempt {index} failed" in str(assembly.messages)

    for profile in ("generic", "auto"):
        adapter = get_provider_adapter(profile)
        sanitized = adapter.sanitize_messages([dict(message) for message in assembly.messages])
        _assert_pairing_invariant(sanitized)


def test_artifact_read_survives_transcript_cap_under_4k_policy() -> None:
    policy = ContextPolicy(max_prompt_tokens=4096)
    policy.recalculate_quotas(4096)
    assert policy.artifact_read_inline_token_limit > policy.transcript_token_limit

    failure = "Error: command failed with exit code 1: boom"
    artifact_body = "artifact line\n" * 400
    messages = [ConversationMessage(role="user", content="run the failing checks")]
    for index in range(1, 5):
        messages.extend(_paired_shell_failure(f"call_{index}", failure))
    messages[3].content = "Trying again."
    messages.append(
        ConversationMessage(
            role="assistant",
            content="Reading the captured artifact.",
            tool_calls=[_tool_call("call_5", name="artifact_read")],
        )
    )
    messages.append(
        ConversationMessage(
            role="tool",
            name="artifact_read",
            tool_call_id="call_5",
            content=artifact_body,
        )
    )

    assembly = _assemble(messages, max_prompt_tokens=4096)

    # C4: collapse fired and the pairing invariant holds end to end.
    assert any(
        message.get("role") == "user" and "collapsed to save tokens" in str(message.get("content") or "")
        for message in assembly.messages
    )
    _assert_pairing_invariant(assembly.messages)
    offered, answered = _tool_call_ids(assembly.messages)
    assert offered == answered == {"call_3", "call_4", "call_5"}

    # H21: the ~2200-token artifact_read exceeds the ~1152-token transcript cap
    # but survives because it is charged to its dedicated inline budget.
    artifact_messages = [
        message
        for message in assembly.messages
        if message.get("role") == "tool" and message.get("name") == "artifact_read"
    ]
    assert len(artifact_messages) == 1
    assert "artifact line" in str(artifact_messages[0].get("content") or "")

    for profile in ("generic", "auto", "lmstudio", "openrouter"):
        adapter = get_provider_adapter(profile)
        sanitized = adapter.sanitize_messages([dict(message) for message in assembly.messages])
        _assert_pairing_invariant(sanitized)


def test_generic_and_auto_adapters_run_pending_tool_cleanup() -> None:
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("call_1")]},
        {"role": "tool", "name": "shell_exec", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("call_2")]},
    ]

    for profile in ("generic", "auto"):
        adapter = get_provider_adapter(profile)
        sanitized = adapter.sanitize_messages([dict(message) for message in messages])
        offered, answered = _tool_call_ids(sanitized)
        assert offered == answered == {"call_1"}, profile
        assert all(
            message.get("role") != "tool" or message.get("tool_call_id")
            for message in sanitized
        )


def test_window_trim_orphan_tool_message_becomes_user_note() -> None:
    # trim_recent_messages keeps the newest user message and may cut a pair in
    # half; the assembler must not emit the orphaned tool message as-is.
    state_messages = [
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call("call_1")]),
        ConversationMessage(
            role="tool",
            name="shell_exec",
            tool_call_id="call_1",
            content="critical tool output",
        ),
        ConversationMessage(role="user", content="next step please"),
    ]

    assembly = _assemble(state_messages, max_prompt_tokens=4096, recent_message_limit=2)

    _assert_pairing_invariant(assembly.messages)
    assert not [message for message in assembly.messages if message.get("role") == "tool"]
    assert any(
        message.get("role") == "user" and "critical tool output" in str(message.get("content") or "")
        for message in assembly.messages
    )
