from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.context import ContextPolicy, ContextSummarizer
from smallctl.context.summarizer import CompactionAttemptResult
from smallctl.harness.compaction import CompactionService
from smallctl.harness.memory import MemoryService
from smallctl.harness.tool_results import ToolResultService
from smallctl.models.conversation import ConversationMessage
from smallctl.state import LoopState


class _RunLogger:
    def __init__(self) -> None:
        self.entries: list[dict[str, object]] = []

    def log(self, channel: str, event: str, message: str = "", **data: object) -> None:
        self.entries.append(
            {
                "channel": channel,
                "event": event,
                "message": message,
                "data": data,
            }
        )


class _Retriever:
    def retrieve_summaries(self, **_: object) -> list[object]:
        return []

    def retrieve_artifacts(self, **_: object) -> list[object]:
        return []


class _PromptAssembler:
    def __init__(self, harness: object, token_fn: object | None = None) -> None:
        self.harness = harness
        self.calls: list[tuple[int, int]] = []
        self.token_fn = token_fn

    def build_messages(self, **kwargs: object) -> SimpleNamespace:
        del kwargs
        recent_count = len(self.harness.state.recent_messages)
        self.calls.append((recent_count, self.harness.context_policy.recent_message_limit))
        if callable(self.token_fn):
            estimated_prompt_tokens = int(self.token_fn(recent_count))
        else:
            estimated_prompt_tokens = 2000 if recent_count > 2 else 500
        return SimpleNamespace(estimated_prompt_tokens=estimated_prompt_tokens, messages=[])


class _Artifact:
    def __init__(self, artifact_id: str) -> None:
        self.artifact_id = artifact_id
        self.kind = "shell_exec"
        self.tool_name = "shell_exec"
        self.source = "shell_exec"
        self.metadata = {"tool_name": "shell_exec"}
        self.summary = "long shell output"
        self.size_bytes = 2048
        self.preview_text = "long shell output"


class _ArtifactStore:
    def compact_tool_message(
        self,
        artifact: object,
        _result: object,
        *,
        request_text: str | None = None,
        inline_full_file: bool = True,
        full_file_preview_chars: int | None = None,
    ) -> str:
        del request_text, inline_full_file, full_file_preview_chars
        return f"Artifact {getattr(artifact, 'artifact_id', 'unknown')}: concise reference"


class _NoOpSummarizer:
    async def compact_recent_messages_async_with_status(self, **_: object) -> CompactionAttemptResult:
        return CompactionAttemptResult(noop_reason="no_compactable_messages")

    def compact_recent_messages_with_status(self, **_: object) -> CompactionAttemptResult:
        return CompactionAttemptResult(noop_reason="no_compactable_messages")


class _ErrorSummarizer:
    async def compact_recent_messages_async_with_status(self, **_: object) -> CompactionAttemptResult:
        raise RuntimeError("summarizer failed")

    def compact_recent_messages_with_status(self, **_: object) -> CompactionAttemptResult:
        raise RuntimeError("summarizer failed")


class _Harness:
    def __init__(
        self,
        *,
        state: LoopState,
        context_policy: ContextPolicy,
        token_fn: object | None = None,
    ) -> None:
        self.state = state
        self.context_policy = context_policy
        self.provider_profile = "generic"
        self.summarizer = ContextSummarizer(context_policy)
        self.summarizer_client = None
        self.artifact_store = None
        self.run_logger = _RunLogger()
        self.prompt_assembler = _PromptAssembler(self, token_fn=token_fn)
        self.retriever = _Retriever()

    def _runlog(self, event: str, message: str, **data: object) -> None:
        self.run_logger.log("harness", event, message, **data)


def test_fallback_compaction_backoffs_until_under_budget() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 4
    state.recent_messages = [
        ConversationMessage(role="assistant", content=f"message {index}")
        for index in range(4)
    ]

    harness = _Harness(
        state=state,
        context_policy=ContextPolicy(
            max_prompt_tokens=2048,
            recent_message_limit=6,
            hot_message_limit=1,
        ),
        token_fn=lambda count: 2000 if count > 2 else 500,
    )

    asyncio.run(
        CompactionService(harness).maybe_compact_context(
            query="inspect budget pressure",
            system_prompt="SYSTEM",
        )
    )

    assert len(state.episodic_summaries) == 2
    assert len(state.recent_messages) == 2

    complete_entries = [entry for entry in harness.run_logger.entries if entry["event"] == "compaction_fallback_complete"]
    assert len(complete_entries) == 1
    complete = complete_entries[0]["data"]
    assert complete["compaction_stopped_reason"] == "below_threshold"
    assert complete["messages_compacted"] == 2
    assert complete["keep_recent_initial"] == 6
    assert complete["keep_recent_final"] == 2
    assert complete["compaction_attempt_count"] == 2
    assert complete["keep_recent_floor"] == 1
    assert complete["compaction_contract"]["min_keep_recent"] == 1
    assert complete["compaction_contract"]["repeated_passes_allowed"] is True
    assert "minimum_recent_window_reached" in complete["compaction_contract"]["allowed_stop_reasons"]
    assert state.prompt_budget.compaction_stopped_reason == "below_threshold"
    assert state.prompt_budget.compaction_messages_compacted == 2
    assert state.prompt_budget.compaction_attempt_count == 2
    assert state.to_dict()["prompt_budget"]["compaction_stopped_reason"] == "below_threshold"

    summary_entries = [entry for entry in harness.run_logger.entries if entry["event"] == "summary_created"]
    assert len(summary_entries) == 2
    assert len(harness.prompt_assembler.calls) >= 3


def test_fallback_compaction_records_no_compactable_messages_stop_reason() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 4
    state.recent_messages = [
        ConversationMessage(role="assistant", content="message 0"),
        ConversationMessage(role="assistant", content="message 1"),
    ]

    harness = _Harness(
        state=state,
        context_policy=ContextPolicy(
            max_prompt_tokens=2048,
            recent_message_limit=6,
            hot_message_limit=1,
        ),
        token_fn=lambda _count: 2000,
    )
    harness.summarizer = _NoOpSummarizer()

    asyncio.run(
        CompactionService(harness).maybe_compact_context(
            query="inspect no compactable messages",
            system_prompt="SYSTEM",
        )
    )

    assert len(state.recent_messages) == 2
    assert len(state.episodic_summaries) == 0

    complete_entries = [entry for entry in harness.run_logger.entries if entry["event"] == "compaction_fallback_complete"]
    assert len(complete_entries) == 1
    complete = complete_entries[0]["data"]
    assert complete["compaction_stopped_reason"] == "no_compactable_messages"
    assert complete["messages_compacted"] == 0
    assert complete["compaction_attempt_count"] == 1
    assert state.prompt_budget.compaction_stopped_reason == "no_compactable_messages"
    assert state.prompt_budget.compaction_recent_messages_after == 2
    assert state.to_dict()["prompt_budget"]["compaction_stopped_reason"] == "no_compactable_messages"


def test_fallback_compaction_records_summarizer_error_stop_reason() -> None:
    state = LoopState(cwd="/tmp")
    state.step_count = 4
    state.recent_messages = [
        ConversationMessage(role="assistant", content="message 0"),
        ConversationMessage(role="assistant", content="message 1"),
    ]

    harness = _Harness(
        state=state,
        context_policy=ContextPolicy(
            max_prompt_tokens=2048,
            recent_message_limit=6,
            hot_message_limit=1,
        ),
        token_fn=lambda _count: 2000,
    )
    harness.summarizer = _ErrorSummarizer()

    asyncio.run(
        CompactionService(harness).maybe_compact_context(
            query="inspect summarizer error",
            system_prompt="SYSTEM",
        )
    )

    assert len(state.recent_messages) == 2
    assert len(state.episodic_summaries) == 0

    complete_entries = [entry for entry in harness.run_logger.entries if entry["event"] == "compaction_fallback_complete"]
    assert len(complete_entries) == 1
    complete = complete_entries[0]["data"]
    assert complete["compaction_stopped_reason"] == "summarizer_error"
    assert complete["messages_compacted"] == 0
    assert complete["compaction_attempt_count"] == 1
    assert state.prompt_budget.compaction_stopped_reason == "summarizer_error"
    assert state.prompt_budget.compaction_recent_messages_after == 2
    assert state.to_dict()["prompt_budget"]["compaction_stopped_reason"] == "summarizer_error"


def test_compact_oversized_tool_messages_replaces_large_tool_content_with_artifact_reference() -> None:
    state = LoopState(cwd="/tmp")
    message = ConversationMessage(
        role="tool",
        name="shell_exec",
        content="x" * 2000,
        metadata={"artifact_id": "A1"},
    )
    state.recent_messages = [message]
    state.artifacts["A1"] = _Artifact("A1")

    harness = SimpleNamespace(
        state=state,
        artifact_store=_ArtifactStore(),
        context_policy=ContextPolicy(),
        _current_user_task=lambda: "Inspect logs",
        _runlog=lambda *args, **kwargs: None,
    )

    changed = ToolResultService(harness).compact_oversized_tool_messages(soft_limit=1024)

    assert changed is True
    assert state.recent_messages[0].content == "Artifact A1: concise reference"
    assert state.artifacts["A1"].artifact_id == "A1"


def test_update_working_memory_invokes_oversized_tool_compaction() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "explore"
    state.run_brief.original_task = "Inspect logs"
    state.working_memory.current_goal = "Inspect logs"
    state.recent_messages = [
        ConversationMessage(
            role="tool",
            name="shell_exec",
            content="x" * 2000,
            metadata={"artifact_id": "A1"},
        )
    ]

    calls: list[int] = []

    class _WiredHarness:
        def __init__(self) -> None:
            self.state = state
            self.context_policy = ContextPolicy()
            self.provider_profile = "generic"

        def _compact_oversized_tool_messages(self, *, soft_limit: int) -> bool:
            calls.append(soft_limit)
            return True

        def _current_user_task(self) -> str:
            return "Inspect logs"

    MemoryService(_WiredHarness()).update_working_memory(recent_messages_limit=10)

    assert calls
    assert calls[0] == 0
