"""Regression tests for the remaining PARTIAL gaps from the 2026-07-17
Independent Fix Confirmation (CONTEXT / VERIFICATION / MEMORY / PERSISTENCE).

- C4: `_reconcile_tool_pairs` enforces the exactly-one-response invariant with
  duplicate `tool_call_id` responses and duplicate offered call ids.
- H8: verifier target affinity preserves normalized path identity, so an
  equal-strength pass against a different path cannot clear a prior failure.
- H20: a caller-supplied `token_budget` is recounted and enforced on the final
  emitted message list, surfacing the typed budget failure when mandatory
  content cannot fit.
- C2: quoted assignment values with spaces/commas are fully redacted, and
  provider-bound messages / runlog-bound data carry no secret values.
- H18: `scrub_sensitive_notes` holds the sidecar lock for the whole
  read-modify-write, so a concurrent upsert is never lost.
- M10: checkpoint retention is per thread (not per namespace) and prunes by
  write recency rather than lexical checkpoint id ordering.
"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.context.policy import estimate_text_tokens
from smallctl.context.retrieval import RetrievalBundle
from smallctl.graph.checkpoint import FileCheckpointSaver
from smallctl.harness.prompt_builder import PromptBuilderService, is_prompt_budget_overflow
from smallctl.harness.tool_result_verification_semantic import (
    _passing_verifier_is_weaker_than_prior_failure,
    _verifier_family_signature,
)
from smallctl.logging_utils import RunLogger
from smallctl.memory_store import ExperienceStore
from smallctl.models.conversation import ConversationMessage
from smallctl.redaction import redact_sensitive_messages, redact_sensitive_text
from smallctl.state import ExperienceMemory, LoopState


def _tool_call(call_id: str, *, name: str = "shell_exec") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"command": "make test"}'},
    }


def _assert_exactly_one_pairing(messages: list[dict[str, Any]]) -> None:
    """Counting-based pairing invariant (set membership cannot see duplicates)."""
    offered: list[str] = []
    answered: list[str] = []
    for message in messages:
        if message.get("role") == "assistant":
            for tool_call in message.get("tool_calls") or []:
                if isinstance(tool_call, dict) and tool_call.get("id"):
                    offered.append(str(tool_call["id"]))
        elif message.get("role") == "tool":
            assert message.get("tool_call_id"), "id-less role=tool message emitted"
            answered.append(str(message["tool_call_id"]))
    assert len(offered) == len(set(offered)), f"duplicate offered call ids: {offered}"
    assert len(answered) == len(set(answered)), f"duplicate tool responses: {answered}"
    assert sorted(offered) == sorted(answered)


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
# C4 — duplicate tool_call_id pairing
# ---------------------------------------------------------------------------


def test_c4_duplicate_tool_responses_collapse_to_exactly_one() -> None:
    messages = [
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call("call_1")]),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_1", content="first result"),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_1", content="duplicate result"),
    ]

    reconciled = PromptAssembler._reconcile_tool_pairs(messages)
    rendered = [message.to_dict() for message in reconciled]

    _assert_exactly_one_pairing(rendered)
    tool_messages = [message for message in rendered if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    # The first response wins; the duplicate is preserved as a user harness note.
    assert tool_messages[0]["content"] == "first result"
    notes = [
        message
        for message in rendered
        if message.get("role") == "user" and "Duplicate tool result" in str(message.get("content") or "")
    ]
    assert len(notes) == 1
    assert "duplicate result" in notes[0]["content"]


def test_c4_duplicate_offered_call_ids_collapse_to_exactly_one_offer() -> None:
    messages = [
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call("call_1")]),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_1", content="only result"),
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call("call_1")]),
    ]

    reconciled = PromptAssembler._reconcile_tool_pairs(messages)
    rendered = [message.to_dict() for message in reconciled]

    _assert_exactly_one_pairing(rendered)
    offered = [
        tool_call["id"]
        for message in rendered
        for tool_call in message.get("tool_calls") or []
    ]
    assert offered == ["call_1"]


def test_c4_duplicate_pairs_survive_full_assembly() -> None:
    policy = ContextPolicy(max_prompt_tokens=4096)
    policy.recalculate_quotas(4096)
    state = LoopState(cwd="/tmp")
    state.recent_messages = [
        ConversationMessage(role="user", content="run the checks"),
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call("call_1")]),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_1", content="first result"),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_1", content="duplicate result"),
        ConversationMessage(role="assistant", content="done", tool_calls=[_tool_call("call_1")]),
    ]

    assembly = PromptAssembler(policy).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
        include_structured_sections=False,
    )

    _assert_exactly_one_pairing(assembly.messages)
    tool_messages = [message for message in assembly.messages if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0]["content"] == "first result"


# ---------------------------------------------------------------------------
# H8 — verifier target affinity preserves path identity
# ---------------------------------------------------------------------------


def _state_with_failed_verifier(command: str) -> Any:
    return SimpleNamespace(
        scratchpad={"_last_failed_verifier": {"command": command}},
        run_brief=None,
        working_memory=None,
        last_verifier_verdict=None,
    )


def test_h8_family_signature_preserves_path_identity() -> None:
    assert _verifier_family_signature("pytest tests/a.py") != _verifier_family_signature("pytest tests/b.py")
    # Normalization keeps identity: separators, leading './', trailing '/'.
    assert _verifier_family_signature("pytest ./tests/a.py") == _verifier_family_signature("pytest tests/a.py")
    assert _verifier_family_signature('pytest "tests\\a.py"') == _verifier_family_signature("pytest tests/a.py")
    # Corrected-argument surface still collapses: flags and their values are noise.
    assert _verifier_family_signature("pytest tests/a.py --maxfail 3") == _verifier_family_signature(
        "pytest tests/a.py --maxfail 1"
    )


def test_h8_equal_strength_pass_on_different_path_does_not_address_prior_failure() -> None:
    state = _state_with_failed_verifier("pytest tests/a.py")
    assert _passing_verifier_is_weaker_than_prior_failure(
        state, current_command="pytest tests/b.py", current_kind="test_suite"
    ) is True
    assert _passing_verifier_is_weaker_than_prior_failure(
        state, current_command="pytest ./tests/a.py", current_kind="test_suite"
    ) is False


def test_h8_interpreter_scripts_keep_directory_identity() -> None:
    state = _state_with_failed_verifier("python scripts/verify.py")
    # Same basename in a different directory is a different target.
    assert _passing_verifier_is_weaker_than_prior_failure(
        state, current_command="python tools/verify.py", current_kind="run_target"
    ) is True
    assert _passing_verifier_is_weaker_than_prior_failure(
        state, current_command="python ./scripts/verify.py", current_kind="run_target"
    ) is False


# ---------------------------------------------------------------------------
# H20 — caller-supplied token_budget enforced on the final emitted list
# ---------------------------------------------------------------------------


def _budget_state(task_chars: int = 4000) -> LoopState:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "fix the deployment " + ("x" * task_chars)
    state.recent_messages = [
        ConversationMessage(role="assistant", content=None, tool_calls=[_tool_call("call_1")]),
        ConversationMessage(role="tool", name="shell_exec", tool_call_id="call_1", content="ok\n" * 200),
    ]
    return state


def test_h20_caller_token_budget_enforced_below_policy_max() -> None:
    policy = ContextPolicy(max_prompt_tokens=8192)
    state = _budget_state()
    system_prompt = "SYSTEM PROMPT " + ("policy directive " * 200)
    caller_budget = 2000
    assert estimate_text_tokens(system_prompt) < caller_budget

    assembly = PromptAssembler(policy).build_messages(
        state=state,
        system_prompt=system_prompt,
        token_budget=caller_budget,
    )

    recomputed = _recompute_emitted_tokens(assembly.messages)
    assert recomputed <= caller_budget
    assert assembly.estimated_prompt_tokens == recomputed
    assert state.prompt_budget.pressure_level == ""
    assert "prompt_budget_failure" not in state.scratchpad.get("_context_metrics", {})
    assert any(
        drop.reason == "hard_prompt_ceiling_truncation"
        for drop in (assembly.frame.drop_log if assembly.frame else [])
    )
    _assert_exactly_one_pairing(assembly.messages)


def test_h20_mandatory_overflow_against_caller_budget_surfaces_typed_failure() -> None:
    policy = ContextPolicy(max_prompt_tokens=8192)
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "small task"
    system_prompt = "SYSTEM PROMPT\n" + ("x" * 20000)

    assembly = PromptAssembler(policy).build_messages(
        state=state,
        system_prompt=system_prompt,
        token_budget=400,
    )

    assert assembly.estimated_prompt_tokens > 400
    assert assembly.estimated_prompt_tokens < 8192
    assert state.prompt_budget.pressure_level == "over_budget"
    failure = state.scratchpad["_context_metrics"]["prompt_budget_failure"]
    assert failure["type"] == "prompt_budget_failure"
    assert failure["reason"] == "mandatory_content_exceeds_hard_ceiling"
    assert failure["hard_ceiling_source"] == "token_budget"
    assert failure["max_prompt_tokens"] == 400


class _NoOpMemory:
    def update_working_memory(self, _limit: int) -> None:
        return


class _NoOpCompaction:
    async def maybe_compact_context(self, **_: object) -> None:
        return


class _EmptyRetriever:
    def retrieve_bundle(self, **_: object) -> RetrievalBundle:
        return RetrievalBundle(query="q", summaries=[], artifacts=[], experiences=[])


class _PromptBuilderHarness:
    def __init__(self, *, max_prompt_tokens: int) -> None:
        self.state = LoopState(cwd="/tmp")
        self.context_policy = ContextPolicy(max_prompt_tokens=max_prompt_tokens)
        self.memory = _NoOpMemory()
        self.compaction = _NoOpCompaction()
        self.retriever = _EmptyRetriever()
        self.prompt_assembler = PromptAssembler(self.context_policy)
        self.fresh_run = False
        self._fresh_run_turns_remaining = 0
        self.cold_memory_store = None

    def _select_retrieval_query(self) -> str:
        return "budget query"

    def _runlog(self, _event: str, _message: str, **_: object) -> None:
        return


def test_h20_prompt_builder_returns_messages_within_caller_budget() -> None:
    harness = _PromptBuilderHarness(max_prompt_tokens=8192)
    soft_budget = harness.context_policy.soft_prompt_token_limit
    assert soft_budget is not None and soft_budget < 8192
    harness.state.recent_messages = [
        ConversationMessage(role="user", content="fix the flaky test"),
        ConversationMessage(role="assistant", content="I will reproduce it first."),
    ]
    service = PromptBuilderService(harness)

    messages = asyncio.run(service.build_messages("SYSTEM PROMPT"))

    assert _recompute_emitted_tokens(messages) <= soft_budget
    assert harness.state.scratchpad["context_used_tokens"] <= soft_budget


def test_h20_prompt_builder_surfaces_typed_failure_for_tight_caller_budget() -> None:
    harness = _PromptBuilderHarness(max_prompt_tokens=8192)
    soft_budget = harness.context_policy.soft_prompt_token_limit
    assert soft_budget is not None
    service = PromptBuilderService(harness)
    # Mandatory system content overflows the caller (soft) budget but not the
    # policy max: before H20 enforcement this was silently returned.
    system_prompt = "SYSTEM PROMPT\n" + ("x" * 20000)
    assert estimate_text_tokens(system_prompt) > soft_budget
    assert estimate_text_tokens(system_prompt) < 8192

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(service.build_messages(system_prompt))

    assert is_prompt_budget_overflow(excinfo.value)
    details = excinfo.value.prompt_budget_details
    assert details["type"] == "prompt_budget_failure"
    assert details["max_prompt_tokens"] == soft_budget
    failure = harness.state.scratchpad["_context_metrics"]["prompt_budget_failure"]
    assert failure["hard_ceiling_source"] == "token_budget"
    assert harness.state.prompt_budget.pressure_level == "over_budget"


# ---------------------------------------------------------------------------
# C2 — redaction gaps
# ---------------------------------------------------------------------------


_C2_QUOTED_SAMPLES = (
    ('SECRET_KEY="abc def,ghi"', ("abc def,ghi", "abc def", "ghi")),
    ("SECRET_KEY='abc def,ghi'", ("abc def,ghi", "abc def", "ghi")),
    ('AWS_SECRET_ACCESS_KEY="abc def, xyz"', ("abc def, xyz", "xyz")),
    ('export GH_TOKEN="ghp_secret value, with spaces"', ("ghp_secret value, with spaces", "spaces")),
    ('OPENAI_API_KEY="sk-proj abc,123" and more text', ("sk-proj abc,123", "abc,123")),
)


@pytest.mark.parametrize("text,fragments", _C2_QUOTED_SAMPLES)
def test_c2_quoted_assignment_values_with_spaces_and_commas_fully_redacted(
    text: str, fragments: tuple[str, ...]
) -> None:
    redacted = redact_sensitive_text(text)
    for fragment in fragments:
        assert fragment not in redacted
    assert "REDACTED" in redacted


def test_c2_provider_bound_messages_contain_no_secret_values() -> None:
    api_key = "sk-proj-provider-bound-123"
    ssh_secret = "hunter2-provider"
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": f"deploy with OPENAI_API_KEY={api_key} and Authorization: Bearer {jwt}"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"running sshpass -p {ssh_secret} ssh db"},
            ],
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "arguments": json.dumps({"command": f"export OPENAI_API_KEY={api_key} && ./deploy"}),
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "shell_exec",
                        "arguments": f"sshpass -p {ssh_secret} ssh root@192.0.2.10",
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "shell_exec", "content": f"deploy used OPENAI_API_KEY={api_key}"},
    ]

    redacted = redact_sensitive_messages(messages)
    serialized = json.dumps(redacted)
    for secret in (api_key, ssh_secret, jwt, "eyJzdWIiOiIxMjM0In0"):
        assert secret not in serialized
    assert "REDACTED" in serialized


def test_c2_runlog_bound_data_contains_no_secret_values(tmp_path: Path) -> None:
    api_key = "sk-proj-runlog-bound-456"
    ssh_secret = "hunter2-runlog"
    logger = RunLogger(tmp_path / "run")
    logger.log(
        "harness",
        "tool_dispatch",
        "dispatching shell command",
        command=f"sshpass -p {ssh_secret} ssh db",
        environment={"OPENAI_API_KEY": api_key, "safe": "visible"},
        nested=[f"OPENAI_API_KEY={api_key}"],
    )

    row = json.loads((logger.run_dir / "harness.jsonl").read_text(encoding="utf-8").splitlines()[0])
    serialized_data = json.dumps(row["data"])
    assert api_key not in serialized_data
    assert ssh_secret not in serialized_data
    assert row["data"]["environment"]["safe"] == "visible"
    text_log = (logger.run_dir / "harness.log").read_text(encoding="utf-8")
    assert api_key not in text_log
    assert ssh_secret not in text_log


def test_c2_experience_store_persistence_boundary_redacts_notes(tmp_path: Path) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    store = ExperienceStore(path)
    api_key = "sk-proj-experience-boundary-789"
    memory = ExperienceMemory(
        memory_id="mem-secret",
        tier="warm",
        tool_name="shell_exec",
        intent="general_task",
        outcome="success",
        notes=f"verified with OPENAI_API_KEY={api_key}",
    )

    assert store.upsert(memory) is memory
    stored_text = path.read_text(encoding="utf-8")
    assert api_key not in stored_text
    assert "[REDACTED]" in stored_text
    loaded = store.get("mem-secret")
    assert loaded is not None
    assert api_key not in loaded.notes


# ---------------------------------------------------------------------------
# H18 — scrub read-modify-write under the sidecar lock
# ---------------------------------------------------------------------------


def test_h18_scrub_interleaved_with_concurrent_upsert_loses_no_record(tmp_path: Path) -> None:
    path = tmp_path / "warm-experiences.jsonl"
    store = ExperienceStore(path)
    secret = "sk-proj-scrub-race-321"
    errors: list[BaseException] = []

    for round_index in range(8):
        # Append a legacy plaintext-secret record directly (single-writer test
        # setup, simulates pre-fix on-disk data) so each scrub round rewrites.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "memory_id": f"mem-secret-{round_index}",
                        "tier": "warm",
                        "intent": "general_task",
                        "tool_name": "shell_exec",
                        "outcome": "success",
                        "notes": f"OPENAI_API_KEY={secret}",
                    }
                )
                + "\n"
            )

        def _scrub() -> None:
            try:
                store.scrub_sensitive_notes(write=True)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def _upsert() -> None:
            try:
                record = ExperienceMemory(
                    memory_id=f"mem-concurrent-{round_index}",
                    tier="warm",
                    tool_name="shell_exec",
                    intent="general_task",
                    outcome="success",
                    notes="concurrent note",
                )
                assert store.upsert(record) is not None
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        scrub_thread = threading.Thread(target=_scrub)
        upsert_thread = threading.Thread(target=_upsert)
        scrub_thread.start()
        upsert_thread.start()
        scrub_thread.join(timeout=30)
        upsert_thread.join(timeout=30)
        assert not scrub_thread.is_alive()
        assert not upsert_thread.is_alive()

    assert not errors
    records = {m.memory_id: m for m in ExperienceStore(path).list()}
    for round_index in range(8):
        assert f"mem-secret-{round_index}" in records
        assert f"mem-concurrent-{round_index}" in records
    assert secret not in path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# M10 — checkpoint retention: per thread, by write recency
# ---------------------------------------------------------------------------


def _checkpoint_config(thread_id: str, checkpoint_id: str, checkpoint_ns: str = "") -> dict[str, Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }


def _write_checkpoint(
    saver: FileCheckpointSaver,
    checkpoint_id: str,
    *,
    thread_id: str = "thread-1",
    checkpoint_ns: str = "",
) -> None:
    config = _checkpoint_config(thread_id, checkpoint_id, checkpoint_ns)
    checkpoint = {
        "id": checkpoint_id,
        "channel_values": {"loop_state": {"step": checkpoint_id}},
        "channel_versions": {"loop_state": 1},
        "ts": "2024-01-01T00:00:00+00:00",
    }
    metadata: dict[str, Any] = {"source": "loop", "step": 1, "parents": {}, "run_id": "run-1"}
    saver.put(config, checkpoint, metadata, {"loop_state": 1})
    saver.put_writes(config, [("loop_state", {"step": checkpoint_id})], "task-1", "prepare_prompt")


def test_m10_retention_is_per_thread_across_namespaces(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.json"
    saver = FileCheckpointSaver(path, max_checkpoints_per_thread=2)
    _write_checkpoint(saver, "chk-1", checkpoint_ns="")
    _write_checkpoint(saver, "chk-2", checkpoint_ns="sub")
    _write_checkpoint(saver, "chk-3", checkpoint_ns="")

    total = sum(len(checkpoints) for checkpoints in saver.storage["thread-1"].values())
    assert total == 2
    surviving = {
        checkpoint_id
        for checkpoints in saver.storage["thread-1"].values()
        for checkpoint_id in checkpoints
    }
    assert surviving == {"chk-2", "chk-3"}
    assert {key[2] for key in saver.writes if key[0] == "thread-1"} == {"chk-2", "chk-3"}
    assert saver.get_tuple(_checkpoint_config("thread-1", "chk-1")) is None
    assert saver.get_tuple(_checkpoint_config("thread-1", "chk-2", "sub")) is not None


def test_m10_pruning_uses_write_recency_not_lexical_id_order(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.json"
    saver = FileCheckpointSaver(path, max_checkpoints_per_thread=2)
    # Write order is zzz (oldest), aaa, mmm (newest); lexical pruning would
    # keep aaa/mmm... no: lexical keeps the highest ids (mmm, zzz) and drops
    # aaa. Recency pruning must drop zzz (the genuinely oldest write).
    for checkpoint_id in ("zzz", "aaa", "mmm"):
        _write_checkpoint(saver, checkpoint_id)

    surviving = set(saver.storage["thread-1"][""])
    assert surviving == {"aaa", "mmm"}
    assert {key[2] for key in saver.writes if key[0] == "thread-1"} == {"aaa", "mmm"}
    assert saver.get_tuple(_checkpoint_config("thread-1", "zzz")) is None
    assert saver.get_tuple(_checkpoint_config("thread-1", "mmm")) is not None


def test_m10_recency_survives_reload_newest_never_pruned(tmp_path: Path) -> None:
    path = tmp_path / "checkpoints.json"
    saver = FileCheckpointSaver(path, max_checkpoints_per_thread=2)
    for checkpoint_id in ("zzz", "aaa", "mmm"):
        _write_checkpoint(saver, checkpoint_id)

    reloaded = FileCheckpointSaver(path, max_checkpoints_per_thread=2)
    assert set(reloaded.storage["thread-1"][""]) == {"aaa", "mmm"}
    _write_checkpoint(reloaded, "bbb")

    surviving = set(reloaded.storage["thread-1"][""])
    assert surviving == {"bbb", "mmm"}
    assert reloaded.get_tuple(_checkpoint_config("thread-1", "aaa")) is None
    assert reloaded.get_tuple(_checkpoint_config("thread-1", "bbb")) is not None

    reloaded_again = FileCheckpointSaver(path, max_checkpoints_per_thread=2)
    assert set(reloaded_again.storage["thread-1"][""]) == {"bbb", "mmm"}
