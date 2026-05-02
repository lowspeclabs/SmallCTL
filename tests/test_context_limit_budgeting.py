from __future__ import annotations

from types import SimpleNamespace

from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.harness import Harness
from smallctl.harness.context_limits import apply_server_context_limit
from smallctl.harness.tool_message_compaction import trim_recent_messages_window
from smallctl.models.conversation import ConversationMessage
from smallctl.state import EvidenceRecord, LoopState


def _make_harness(*, max_prompt_tokens: int | None, explicit: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        configured_max_prompt_tokens=max_prompt_tokens,
        configured_max_prompt_tokens_explicit=explicit,
        context_policy=ContextPolicy(
            max_prompt_tokens=max_prompt_tokens,
            reserve_completion_tokens=1024,
            reserve_tool_tokens=512,
            recent_message_limit=6,
        ),
        server_context_limit=None,
        discovered_server_context_limit=None,
        provider_profile="generic",
        state=SimpleNamespace(recent_message_limit=6),
        _harness_kwargs={},
        _runlog=lambda *args, **kwargs: None,
    )


def test_initialization_preserves_explicit_max_prompt_tokens_over_context_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    harness = Harness(
        endpoint="http://example.test/v1",
        model="qwen3.5:4b",
        phase="explore",
        api_key="test-key",
        context_limit=16384,
        max_prompt_tokens=32768,
    )

    from smallctl.harness.context_limits import derive_prompt_budget_from_context_limit
    derived = derive_prompt_budget_from_context_limit(16384)

    assert harness.configured_max_prompt_tokens == 32768
    assert harness.server_context_limit == 16384
    assert harness.context_policy.max_prompt_tokens != derived
    assert harness.context_policy.max_prompt_tokens == 32768


def test_runtime_probe_preserves_explicit_max_prompt_tokens_for_partitioning() -> None:
    harness = _make_harness(max_prompt_tokens=32768)

    new_limit = apply_server_context_limit(harness, 16384, source="runtime_probe")

    assert new_limit == 32768
    assert harness.server_context_limit == 16384
    assert harness.context_policy.max_prompt_tokens == 32768
    assert harness.context_policy.hot_message_limit == 32
    assert harness.context_policy.recent_message_limit == 32
    assert harness.state.recent_message_limit == 32


def test_small_lmstudio_model_uses_tighter_hot_window() -> None:
    policy = ContextPolicy(max_prompt_tokens=32768)
    policy.apply_backend_profile("lmstudio")
    policy.apply_model_profile("qwen3.5:4b")
    policy.recalculate_quotas(32768)

    assert policy.hot_message_limit == 19
    assert policy.recent_message_limit == 19
    assert policy.compaction_step_interval == 4
    assert policy.transcript_token_limit > 10000


def test_non_probe_source_preserves_configured_budget_without_overflow() -> None:
    harness = _make_harness(max_prompt_tokens=32768)

    new_limit = apply_server_context_limit(
        harness,
        16384,
        source="stream_context_overflow",
        observed_n_keep=None,
    )

    assert new_limit == 32768
    assert harness.server_context_limit == 16384
    assert harness.context_policy.max_prompt_tokens == 32768
    assert harness.context_policy.hot_message_limit == 32
    assert harness.context_policy.recent_message_limit == 32
    assert harness.state.recent_message_limit == 32


def test_stream_context_overflow_can_still_shrink_prompt_budget() -> None:
    harness = _make_harness(max_prompt_tokens=32768)

    new_limit = apply_server_context_limit(
        harness,
        16384,
        source="stream_context_overflow",
        observed_n_keep=17000,
    )

    assert new_limit == 14336
    assert harness.server_context_limit == 16384
    assert harness.context_policy.max_prompt_tokens == 14336
    assert harness.context_policy.hot_message_limit == 14
    assert harness.context_policy.recent_message_limit == 14
    assert harness.state.recent_message_limit == 14


def test_runtime_probe_can_expand_non_explicit_prompt_budget_to_server_budget() -> None:
    harness = _make_harness(max_prompt_tokens=32768, explicit=False)

    new_limit = apply_server_context_limit(harness, 256000, source="runtime_probe")

    assert new_limit == 253952
    assert harness.server_context_limit == 256000
    assert harness.context_policy.max_prompt_tokens == 253952
    assert harness.context_policy.hot_message_limit == 253
    assert harness.context_policy.recent_message_limit == 253


def test_large_prompt_budget_scales_inline_limits() -> None:
    policy = ContextPolicy(max_prompt_tokens=32768)

    policy.recalculate_quotas(32768)

    assert policy.tool_result_inline_token_limit == 624
    assert policy.artifact_read_inline_token_limit == 1249


def test_recent_message_trimming_preserves_mission_anchor_and_latest_followup() -> None:
    messages = [
        ConversationMessage(role="user", content="update remote files with darkmode across the whole site"),
        ConversationMessage(role="assistant", content="I found the current CSS and page templates."),
        ConversationMessage(role="tool", name="ssh_exec", content="cat /var/www/html/index.html"),
        ConversationMessage(role="assistant", content="The home page still hardcodes light colors."),
        ConversationMessage(role="user", content="also make the footer consistent on every page"),
        ConversationMessage(role="tool", name="ssh_exec", content="cat /var/www/html/footer.html"),
    ]

    trimmed = trim_recent_messages_window(messages, limit=3)

    assert [message.role for message in trimmed] == ["user", "assistant", "user"]
    assert trimmed[0].content == "update remote files with darkmode across the whole site"
    assert trimmed[-1].content == "also make the footer consistent on every page"


def test_prompt_assembler_recent_message_limit_preserves_anchor_and_followup() -> None:
    state = LoopState(cwd="/tmp")
    state.recent_messages = [
        ConversationMessage(role="user", content="update remote files with darkmode across the whole site"),
        ConversationMessage(role="assistant", content="I found the current CSS and page templates."),
        ConversationMessage(role="tool", name="ssh_exec", content="cat /var/www/html/index.html"),
        ConversationMessage(role="assistant", content="The home page still hardcodes light colors."),
        ConversationMessage(role="user", content="also make the footer consistent on every page"),
        ConversationMessage(role="tool", name="ssh_exec", content="cat /var/www/html/footer.html"),
    ]

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=6)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
        recent_message_limit=4,
        include_structured_sections=False,
    )

    rendered = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)

    assert "update remote files with darkmode across the whole site" in rendered
    assert "also make the footer consistent on every page" in rendered
    assert "cat /var/www/html/footer.html" in rendered


def test_prompt_assembler_preserves_fresh_tool_outputs_from_scratchpad() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Update remote site styling"
    state.recent_messages = [ConversationMessage(role="user", content="continue styling fixes")]
    state.scratchpad["_fresh_tool_outputs"] = [
        {
            "tool_name": "shell_exec",
            "artifact_id": "A0016",
            "content": "body{font-family:Roboto;background:var(--bg)}",
        }
    ]

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=4096)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
        recent_message_limit=1,
        include_structured_sections=True,
    )

    rendered = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)

    assert "Fresh tool outputs" in rendered
    assert "A0016" in rendered
    assert "body{font-family:Roboto;background:var(--bg)}" in rendered


def test_prompt_assembler_prioritizes_latest_observations_under_budget_pressure() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Inspect current tool outputs"
    for index in range(6):
        state.reasoning_graph.evidence_records.append(
            EvidenceRecord(
                evidence_id=f"E-A000{index}",
                statement=f"observation-{index} " + ("detail " * 60),
                phase="execute",
                tool_name="artifact_grep",
                metadata={"observation_adapter": "artifact_observation_list"},
            )
        )

    assembly = PromptAssembler(
        ContextPolicy(
            max_prompt_tokens=4096,
            observation_token_limit=120,
            observation_token_floor=700,
            min_observation_items=3,
            max_observation_items=6,
        )
    ).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
        recent_message_limit=1,
        include_structured_sections=True,
    )

    rendered = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)

    assert "observation-3" in rendered
    assert "observation-4" in rendered
    assert "observation-5" in rendered
    assert assembly.section_tokens["normalized_observations"] > 120
