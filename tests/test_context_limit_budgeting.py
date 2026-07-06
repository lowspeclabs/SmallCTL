from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest

from smallctl.client import OpenAICompatClient
from smallctl.client.request_budget import RequestEstimator, build_request_budget
from smallctl.client.tool_budgeting import fit_tools_to_context_budget
from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.harness import Harness
from smallctl.harness.context_limits import apply_server_context_limit
from smallctl.harness.tool_message_compaction import trim_recent_messages_window
from smallctl.models.conversation import ConversationMessage
from smallctl.state import ArtifactRecord, ArtifactSnippet, EvidenceRecord, ExperienceMemory, LoopState


def _make_harness(
    *,
    max_prompt_tokens: int | None,
    explicit: bool = True,
    provider_profile: str = "generic",
) -> SimpleNamespace:
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
        provider_profile=provider_profile,
        state=SimpleNamespace(recent_message_limit=6),
        config=SimpleNamespace(),
        _runlog=lambda *args, **kwargs: None,
    )


def test_initialization_clamps_explicit_max_prompt_tokens_to_context_limit(tmp_path, monkeypatch) -> None:
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
    assert harness.context_policy.max_prompt_tokens == derived


def test_runtime_probe_clamps_explicit_max_prompt_tokens_for_partitioning() -> None:
    harness = _make_harness(max_prompt_tokens=32768)

    new_limit = apply_server_context_limit(harness, 16384, source="runtime_probe")

    assert new_limit == 14336
    assert harness.server_context_limit == 16384
    assert harness.context_policy.max_prompt_tokens == 14336
    assert harness.context_policy.hot_message_limit == 14
    assert harness.context_policy.recent_message_limit == 14
    assert harness.state.recent_message_limit == 14


def test_runtime_probe_uses_llamacpp_request_budget_for_prompt_ceiling() -> None:
    harness = _make_harness(max_prompt_tokens=32768, provider_profile="llamacpp")

    new_limit = apply_server_context_limit(harness, 16384, source="runtime_probe")

    assert new_limit == build_request_budget(16384).effective_prompt_budget
    assert new_limit == 12800
    assert harness.server_context_limit == 16384
    assert harness.context_policy.max_prompt_tokens == 12800
    assert int(harness.context_policy.max_prompt_tokens * harness.context_policy.summarize_at_ratio) == 10240


def test_runtime_probe_can_expand_stale_lower_server_context_limit() -> None:
    harness = _make_harness(max_prompt_tokens=32768)
    harness.server_context_limit = 8192
    harness.discovered_server_context_limit = 8192

    new_limit = apply_server_context_limit(harness, 16384, source="runtime_probe")

    assert new_limit == 14336
    assert harness.server_context_limit == 16384
    assert harness.discovered_server_context_limit == 16384
    assert harness.context_policy.max_prompt_tokens == 14336


def test_non_runtime_context_update_does_not_expand_stale_lower_limit() -> None:
    harness = _make_harness(max_prompt_tokens=None, explicit=False)
    harness.server_context_limit = 8192
    harness.discovered_server_context_limit = 8192

    new_limit = apply_server_context_limit(harness, 16384, source="stream_context_overflow")

    assert new_limit == 7168
    assert harness.server_context_limit == 8192
    assert harness.discovered_server_context_limit == 8192


def test_transport_budget_can_respect_runtime_limit_when_prompt_budget_is_explicit() -> None:
    harness = _make_harness(max_prompt_tokens=32768)
    harness.server_context_limit = 8192
    tools = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "x" * 2400,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string", "description": "y" * 2400},
                    },
                },
            },
        }
        for name in ["artifact_read", "web_search", "ssh_exec", "ssh_file_read", "task_complete", "task_fail"]
    ]
    payload = {
        "model": "demo",
        "messages": [{"role": "user", "content": "inspect the remote host"}],
        "stream": True,
        "tools": tools,
    }

    result = fit_tools_to_context_budget(
        payload=payload,
        tools=tools,
        budget=build_request_budget(harness.server_context_limit),
        estimator=RequestEstimator(),
    )

    assert harness.context_policy.max_prompt_tokens == 32768
    assert result.action == "reduced_tools"
    assert set(result.dropped_tool_names) <= {"artifact_read", "web_search"}
    assert result.dropped_tool_names
    assert {"ssh_exec", "ssh_file_read", "task_complete", "task_fail"} <= set(result.kept_tool_names)


def test_small_lmstudio_model_uses_tighter_hot_window() -> None:
    policy = ContextPolicy(max_prompt_tokens=32768)
    policy.apply_backend_profile("lmstudio")
    policy.apply_model_profile("qwen3.5:4b")
    policy.recalculate_quotas(32768)

    assert policy.hot_message_limit == 19
    assert policy.recent_message_limit == 19
    assert policy.compaction_step_interval == 4
    assert policy.transcript_token_limit > 10000


def test_non_probe_source_clamps_configured_budget_without_overflow() -> None:
    harness = _make_harness(max_prompt_tokens=32768)

    new_limit = apply_server_context_limit(
        harness,
        16384,
        source="stream_context_overflow",
        observed_n_keep=None,
    )

    assert new_limit == 14336
    assert harness.server_context_limit == 16384
    assert harness.context_policy.max_prompt_tokens == 14336
    assert harness.context_policy.hot_message_limit == 14
    assert harness.context_policy.recent_message_limit == 14
    assert harness.state.recent_message_limit == 14


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

    assert new_limit == 251904
    assert harness.server_context_limit == 256000
    assert harness.context_policy.max_prompt_tokens == 251904
    assert harness.context_policy.hot_message_limit == 251
    assert harness.context_policy.recent_message_limit == 251


def test_large_prompt_budget_scales_inline_limits() -> None:
    policy = ContextPolicy(max_prompt_tokens=32768)

    policy.recalculate_quotas(32768)

    assert policy.tool_result_inline_token_limit == 624
    assert policy.artifact_read_inline_token_limit == 3000


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


def test_prompt_assembler_compacts_complete_file_read_fresh_outputs() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Review app.py"
    state.artifacts["A0001"] = ArtifactRecord(
        artifact_id="A0001",
        kind="file_read",
        source="/tmp/app.py",
        created_at="2026-05-21T00:00:00+00:00",
        size_bytes=4096,
        summary="app.py full file",
        tool_name="file_read",
        metadata={"complete_file": True, "total_lines": 403},
    )
    state.scratchpad["_fresh_tool_outputs"] = [
        {
            "tool_name": "file_read",
            "artifact_id": "A0001",
            "content": "import curses\n" + ("raw line\n" * 300),
        }
    ]

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=4096)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
        recent_message_limit=1,
        include_structured_sections=True,
    )

    rendered = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)

    assert "Artifact A0001: app.py full file (403 lines). Full file captured" in rendered
    assert "import curses" not in rendered
    assert "raw line" not in rendered


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


def test_model_name_ceiling_clamps_explicit_prompt_budget_when_server_limit_unknown() -> None:
    from smallctl.harness.context_limits import resolve_effective_prompt_budget

    effective = resolve_effective_prompt_budget(
        configured_max_prompt_tokens=200_000,
        configured_max_prompt_tokens_explicit=True,
        server_context_limit=None,
        model_name="Qwen3.5-9b",
    )

    assert effective == 128_000


def test_server_context_limit_takes_precedence_over_model_name_ceiling() -> None:
    from smallctl.harness.context_limits import resolve_effective_prompt_budget

    effective = resolve_effective_prompt_budget(
        configured_max_prompt_tokens=200_000,
        configured_max_prompt_tokens_explicit=True,
        server_context_limit=32_768,
        model_name="Qwen3.5-9b",
    )

    assert effective == 28_672


def test_gemma_4_model_name_ceiling_caps_explicit_budget() -> None:
    from smallctl.harness.context_limits import resolve_effective_prompt_budget

    effective = resolve_effective_prompt_budget(
        configured_max_prompt_tokens=200_000,
        configured_max_prompt_tokens_explicit=True,
        server_context_limit=None,
        model_name="Gemma 4 e4b",
    )

    assert effective == 128_000


def test_unknown_model_name_does_not_clamp_explicit_budget() -> None:
    from smallctl.harness.context_limits import resolve_effective_prompt_budget

    effective = resolve_effective_prompt_budget(
        configured_max_prompt_tokens=200_000,
        configured_max_prompt_tokens_explicit=True,
        server_context_limit=None,
        model_name="unknown-custom-9b",
    )

    assert effective == 200_000


def test_apply_server_context_limit_logs_when_configured_budget_exceeds_model_ceiling() -> None:
    harness = _make_harness(max_prompt_tokens=200_000)
    harness.client = SimpleNamespace(model="qwen3.5-9b")
    log_calls: list[dict[str, object]] = []
    original_runlog = harness._runlog

    def _capturing_runlog(event: str, message: str, **kwargs: object) -> None:
        log_calls.append({"event": event, "message": message, **kwargs})
        original_runlog(event, message, **kwargs)

    harness._runlog = _capturing_runlog

    apply_server_context_limit(harness, 256_000, source="runtime_probe")

    assert harness.context_policy.max_prompt_tokens == 128_000
    assert any(
        call.get("event") == "context_limit"
        and "exceeds known model context window" in str(call.get("message", ""))
        for call in log_calls
    )


@pytest.mark.asyncio
async def test_fetch_model_context_limit_logs_each_probe_failure(caplog, monkeypatch) -> None:
    class _FailingAsyncClient:
        async def get(self, url: str, headers: dict[str, str] | None = None, timeout: float | None = None) -> None:
            raise httpx.ConnectError(f"probe failure for {url}")

    client = OpenAICompatClient(
        base_url="http://example.test/v1",
        model="demo-model",
        provider_profile="generic",
        api_key="test-key",
    )
    monkeypatch.setattr(
        "smallctl.client.client_transport_client_lifecycle._get_async_client",
        lambda _client: _FailingAsyncClient(),
    )

    with caplog.at_level(logging.WARNING, logger="smallctl.client"):
        limit = await client.fetch_model_context_limit()

    assert limit is None
    failure_messages = [record.getMessage() for record in caplog.records if "context_probe_failed" in record.getMessage()]
    assert len(failure_messages) == 6
    expected_urls = [
        "http://example.test/v1/props",
        "http://example.test/v1/slots",
        "http://example.test/props",
        "http://example.test/slots",
        "http://example.test/v1/models/demo-model",
        "http://example.test/v1/models",
    ]
    for url in expected_urls:
        assert any(url in message for message in failure_messages)
    assert any("ConnectError" in message for message in failure_messages)


def test_prompt_assembler_enforces_max_prompt_token_ceiling() -> None:
    """Regression: optional context lanes must be dropped before the assembled
    prompt exceeds the model's max_prompt_tokens ceiling.
    """
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "debug remote backup" + " detail" * 80
    state.run_brief.current_phase_objective = "inspect remote host" + " detail" * 80
    state.working_memory.current_goal = "inspect remote host" + " detail" * 80
    state.working_memory.next_actions = ["list /root"] * 20
    state.recent_messages = [
        ConversationMessage(role="user", content=("debug remote backup " + "x " * 300)),
        ConversationMessage(role="assistant", content="I will inspect the remote host."),
        ConversationMessage(role="tool", name="ssh_dir_list", content="entry\n" * 250),
    ]
    state.scratchpad["_fresh_tool_outputs"] = [
        {"tool_name": "ssh_dir_list", "artifact_id": "", "content": "entry\n" * 250}
    ]
    for i in range(4):
        state.reasoning_graph.evidence_records.append(
            EvidenceRecord(
                evidence_id=f"E-{i}",
                statement=f"observation-{i} " + ("detail " * 40),
                phase="execute",
                tool_name="ssh_dir_list",
                metadata={"observation_adapter": "artifact_observation_list"},
            )
        )

    system_prompt = "SYSTEM\n" + "directive text line content\n" * 910

    assembler = PromptAssembler(ContextPolicy(max_prompt_tokens=12288, recent_message_limit=8))
    assembly = assembler.build_messages(
        state=state,
        system_prompt=system_prompt,
        retrieved_artifacts=[ArtifactSnippet(artifact_id="A0001", text="prior artifact note")],
        retrieved_experiences=[
            ExperienceMemory(
                memory_id="mem-1",
                intent="requested_ssh_exec",
                tool_name="ssh_dir_list",
                outcome="success",
                notes="prior success",
            )
        ],
        include_structured_sections=True,
    )

    max_prompt_tokens = assembler.policy.max_prompt_tokens
    assert max_prompt_tokens is not None
    assert assembly.estimated_prompt_tokens <= max_prompt_tokens
    # At least one optional lane should have been sacrificed to stay under the cap.
    assert assembly.section_tokens.get("fresh_tool_outputs", 0) == 0


def test_swa_prompt_cap_uses_higher_limit_for_gemma_4_12b_and_27b() -> None:
    harness = _make_harness(max_prompt_tokens=32768, provider_profile="llamacpp")
    harness.client = SimpleNamespace(model="gemma-4-12b")

    new_limit = apply_server_context_limit(harness, 65024, source="runtime_probe")

    assert new_limit == 24576
    assert harness.context_policy.max_prompt_tokens == 24576


def test_swa_prompt_cap_uses_default_for_exact_small_gemma_4() -> None:
    harness = _make_harness(max_prompt_tokens=32768, provider_profile="llamacpp")
    harness.client = SimpleNamespace(model="gemma-4-e4b-it")

    new_limit = apply_server_context_limit(harness, 65024, source="runtime_probe")

    assert new_limit == 12288
    assert harness.context_policy.max_prompt_tokens == 12288

