from __future__ import annotations

from types import SimpleNamespace

import pytest

from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.context.retrieval import LexicalRetriever
from smallctl.fama.capsules import render_fama_capsules
from smallctl.fama.detectors import detect_read_only_loop, detect_repeated_read_on_same_path
from smallctl.fama.signals import ActiveMitigation, FamaFailureKind
from smallctl.fama.state import activate_mitigations
from smallctl.graph.model_call_nodes import _maybe_inject_file_truncation_hallucination_nudge
from smallctl.graph.tool_loop_guards import _detect_repeated_tool_loop
from smallctl.models.conversation import ConversationMessage
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import ArtifactRecord, ExperienceMemory, LoopState


# --- Phase 0: Crash fix ---

def test_handle_repeated_tool_loop_no_crash_on_pause_path() -> None:
    """RC-1: handle_repeated_tool_loop must not raise UnboundLocalError."""
    state = LoopState(step_count=5, cwd="/tmp")
    state.tool_history = [
        "dir_list|{\"path\": \"Vikunja\"}",
        "file_read|{\"path\": \"Vikunja/agents.md\"}",
        "artifact_read|{\"artifact_id\": \"A0002\", \"start_line\": 1}",
        "artifact_read|{\"artifact_id\": \"A0002\", \"start_line\": 100}",
        "dir_list|{\"path\": \"Vikunja\"}",
    ]
    # Ensure the generic loop nudge has already been given so the guard trips.
    state.scratchpad["_generic_loop_nudged"] = "generic_loop:dir_list:{\"path\": \"Vikunja\"}"

    pending = SimpleNamespace(tool_name="dir_list", args={"path": "Vikunja"})

    async def _noop_emit(handler, event):
        return None

    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(escalation_enabled=False, escalation_expose_tool=False),
        log=None,
        _runlog=lambda *args, **kwargs: None,
        _emit=_noop_emit,
    )

    repeat_error = _detect_repeated_tool_loop(harness, pending)
    assert repeat_error is not None
    assert "dir_list" in repeat_error


# --- Phase 1: Artifact snippets ---

def test_artifact_snippets_after_file_read() -> None:
    """RC-2: After file_read, artifact snippets appear in next prompt frame."""
    state = LoopState(cwd="/tmp")
    state.step_count = 3
    state.current_phase = "explore"
    state.run_brief.original_task = "update vikunja with the work done in this task"
    state.artifacts["A0002"] = ArtifactRecord(
        artifact_id="A0002",
        kind="file_read",
        source="Vikunja/agents.md",
        created_at="2026-06-09T00:00:00+00:00",
        size_bytes=20480,
        summary="agents.md skill documentation",
        tool_name="file_read",
        inline_content="line1\nline2\nline3\n",
    )
    state.scratchpad["_recent_file_read_artifacts"] = [
        {"artifact_id": "A0002", "expires_after": 10, "step": 2},
    ]

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=4096, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )
    content = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)
    assert "A0002" in content
    assert "Recent read" in content or "agents.md" in content


def test_recent_evidence_summary_in_working_memory() -> None:
    """RC-2: Working memory includes recent dir_list/file_read evidence summary."""
    state = LoopState(cwd="/tmp")
    state.step_count = 6
    state.tool_execution_records = {
        "op-1": {
            "tool_name": "dir_list",
            "args": {"path": "Vikunja"},
            "result": {
                "success": True,
                "output": [
                    {"name": "vikunja.py"},
                    {"name": "agents.md"},
                    {"name": "vikunja_api_guide.md"},
                ],
                "metadata": {"created_at_step": 1, "path": "Vikunja", "count": 9},
            },
        },
        "op-2": {
            "tool_name": "file_read",
            "args": {"path": "Vikunja/agents.md"},
            "result": {
                "success": True,
                "output": "...",
                "metadata": {"created_at_step": 2, "path": "Vikunja/agents.md", "total_lines": 448, "artifact_id": "A0002"},
            },
        },
    }

    from smallctl.context.frame_compiler import PromptStateFrameCompiler

    frame = PromptStateFrameCompiler().compile(state=state)
    wm = frame.spine.working_memory_text
    assert "Directory listing" in wm or "File read" in wm


# --- Phase 2: Truncation nudge ---

def test_file_truncation_nudge_complete_file_injects_proceed() -> None:
    """RC-3: complete_file=true triggers a proceed nudge, not a truncation nudge."""
    state = LoopState()
    state.scratchpad["_progress_read_history"] = [
        {
            "tool_name": "file_read",
            "path": "Vikunja/agents.md",
            "complete_file": True,
            "file_content_truncated": False,
            "total_lines": 448,
            "line_end": 448,
        }
    ]
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    _maybe_inject_file_truncation_hallucination_nudge(harness, "The file seems truncated.")

    messages = [m for m in state.recent_messages if m.role == "system"]
    assert len(messages) == 1
    assert "read completely" in messages[0].content.lower() or "have all the content" in messages[0].content.lower()
    assert "might be hallucinating" not in messages[0].content.lower()


def test_file_truncation_nudge_truncated_file_injects_truncation_hint() -> None:
    """RC-3: Actually truncated files still trigger truncation recovery hint."""
    state = LoopState()
    state.scratchpad["_progress_read_history"] = [
        {
            "tool_name": "file_read",
            "path": "Vikunja/agents.md",
            "complete_file": False,
            "file_content_truncated": True,
            "total_lines": 1000,
            "line_end": 200,
        }
    ]
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    _maybe_inject_file_truncation_hallucination_nudge(harness, "The output is truncated.")

    messages = [m for m in state.recent_messages if m.role == "system"]
    assert len(messages) == 1
    assert "truncated" in messages[0].content.lower()
    assert "artifact_read" in messages[0].content.lower()


# --- Phase 3: FAMA read-loop detectors ---

def test_read_only_loop_detector_fires_after_four_reads() -> None:
    """RC-4: 4+ consecutive read-only tool calls fire the detector."""
    state = LoopState(step_count=6)
    state.tool_history = [
        "dir_list|{\"path\": \"Vikunja\"}",
        "file_read|{\"path\": \"Vikunja/agents.md\"}",
        "artifact_read|{\"artifact_id\": \"A0002\"}",
        "artifact_read|{\"artifact_id\": \"A0002\"}",
    ]
    signal = detect_read_only_loop(state, threshold=4)
    assert signal is not None
    assert signal.kind is FamaFailureKind.LOOPING
    assert "read-only" in signal.evidence.lower()


def test_read_only_loop_detector_resets_on_state_change() -> None:
    """RC-4: A write call resets the read-only loop counter."""
    state = LoopState(step_count=6)
    state.tool_history = [
        "dir_list|{\"path\": \"Vikunja\"}",
        "file_read|{\"path\": \"Vikunja/agents.md\"}",
        "file_write|{\"path\": \"x.txt\", \"content\": \"hi\"}",
        "artifact_read|{\"artifact_id\": \"A0002\"}",
    ]
    signal = detect_read_only_loop(state, threshold=4)
    assert signal is None


def test_repeated_read_same_path_detector_fires() -> None:
    """RC-4: Same path dir_list at steps 1 and 6 triggers the detector."""
    state = LoopState(step_count=6)
    state.tool_history = [
        "dir_list|{\"path\": \"Vikunja\"}",
        "file_read|{\"path\": \"Vikunja/agents.md\"}",
        "artifact_read|{\"artifact_id\": \"A0002\"}",
        "artifact_read|{\"artifact_id\": \"A0002\"}",
        "dir_list|{\"path\": \"Vikunja\"}",
    ]
    signal = detect_repeated_read_on_same_path(state, threshold=2)
    assert signal is not None
    assert signal.kind is FamaFailureKind.LOOPING
    assert "dir_list" in signal.evidence
    assert "Vikunja" in signal.evidence


def test_read_only_loop_capsule_renders() -> None:
    """RC-4: read_only_loop capsule appears in prompt working_memory."""
    state = LoopState()
    state.scratchpad["_fama_config"] = {"enabled": True, "mode": "lite", "capsule_token_budget": 180}
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="read_only_loop_capsule",
                reason="read-only loop detected",
                source_signal="looping:0:",
                activated_step=0,
                expires_after_step=5,
            )
        ],
        max_active=5,
    )
    capsules = render_fama_capsules(state, token_budget=180)
    combined = "\n".join(capsules)
    assert "reading files and directories" in combined.lower()
    assert "state change" in combined.lower()


# --- Phase 4: Memory relevance ---

def test_phase_aware_memory_scoring_explore_prefers_read_tools() -> None:
    """RC-5: explore phase prefers read-tool memories over task_complete."""
    state = LoopState(cwd="/tmp")
    state.step_count = 3
    state.current_phase = "explore"
    state.run_brief.original_task = "explore vikunja"
    state.active_intent = "explore"
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-complete",
            intent="finish",
            tool_name="task_complete",
            outcome="success",
            notes="Task completed successfully.",
        ),
        ExperienceMemory(
            memory_id="mem-read",
            intent="explore",
            tool_name="file_read",
            outcome="success",
            notes="Read agents.md for skill documentation.",
        ),
    ]

    ranked = LexicalRetriever()._rank_experiences(state=state)
    tools = [m.tool_name for _, m in ranked]
    # file_read should outrank task_complete
    if tools:
        assert tools[0] == "file_read"


def test_phase_aware_memory_scoring_explore_decays_terminal_memories() -> None:
    """RC-5: task_complete memories are deprioritized during explore."""
    state = LoopState(cwd="/tmp")
    state.step_count = 3
    state.current_phase = "explore"
    state.run_brief.original_task = "explore vikunja"
    state.active_intent = "explore"
    state.warm_experiences = [
        ExperienceMemory(
            memory_id="mem-complete",
            intent="explore",
            tool_name="task_complete",
            outcome="success",
            notes="Read agents.md.",
        ),
        ExperienceMemory(
            memory_id="mem-read",
            intent="explore",
            tool_name="file_read",
            outcome="success",
            notes="Read agents.md.",
        ),
    ]

    ranked = LexicalRetriever()._rank_experiences(state=state)
    scores = {m.tool_name: s for s, m in ranked}
    assert scores["file_read"] > scores["task_complete"]


# --- Phase 4: Working memory evolution ---

def test_working_memory_evolution_after_dir_list() -> None:
    """RC-6: After dir_list, working memory reflects what was found."""
    state = LoopState(cwd="/tmp")
    state.step_count = 2
    result = ToolEnvelope(
        success=True,
        output=[{"name": "agents.md"}, {"name": "vikunja.py"}],
        metadata={"path": "Vikunja", "count": 2},
    )

    from smallctl.harness.tool_result_artifact_updates import _update_working_memory_after_read

    _update_working_memory_after_read(state, tool_name="dir_list", result=result, arguments={"path": "Vikunja"})

    assert any("Directory listed" in str(entry.content) for entry in state.working_memory.next_action_meta)


def test_working_memory_evolution_after_file_read() -> None:
    """RC-6: After file_read, working memory reflects skill is loaded."""
    state = LoopState(cwd="/tmp")
    state.step_count = 3
    result = ToolEnvelope(
        success=True,
        output="skill docs",
        metadata={"path": "Vikunja/agents.md", "total_lines": 448, "artifact_id": "A0002"},
    )

    from smallctl.harness.tool_result_artifact_updates import _update_working_memory_after_read

    _update_working_memory_after_read(
        state,
        tool_name="file_read",
        result=result,
        arguments={"path": "Vikunja/agents.md"},
    )

    contents = [str(entry.content) for entry in state.working_memory.next_action_meta]
    assert any("Loaded file" in c for c in contents)
    assert any("A0002" in c for c in contents)


# --- Phase 6: Prompt compaction ---

def test_read_turn_compaction_reduces_message_tokens() -> None:
    """RC-7: After many read-only turns, recent_messages tokens are lower than raw accumulation."""
    state = LoopState(cwd="/tmp")
    state.step_count = 12
    # Add 10 assistant/tool pairs of read-only content
    for i in range(10):
        state.recent_messages.append(
            ConversationMessage(role="assistant", content=f'<tool_call>file_read({{"path": "f{i}.md"}})</tool_call>')
        )
        state.recent_messages.append(
            ConversationMessage(
                role="tool",
                name="file_read",
                content=f"Content of f{i}.md:\n" + ("line\n" * 50),
                metadata={"artifact_id": f"A{i:03d}", "path": f"f{i}.md"},
            )
        )

    from smallctl.harness.tool_message_compaction import compact_read_only_turns

    compacted, count = compact_read_only_turns(state.recent_messages, threshold_turns=8)
    assert count > 0
    raw_tokens = sum(len(m.content or "") for m in state.recent_messages)
    compact_tokens = sum(len(m.content or "") for m in compacted)
    assert compact_tokens < raw_tokens


def test_read_turn_compaction_preserves_state_changing_turns() -> None:
    """RC-7: State-changing turns are never compacted."""
    state = LoopState(cwd="/tmp")
    state.step_count = 12
    state.recent_messages.append(
        ConversationMessage(role="assistant", content='<tool_call>file_write({"path": "x.txt", "content": "hi"})</tool_call>')
    )
    state.recent_messages.append(
        ConversationMessage(role="tool", name="file_write", content="File written.")
    )

    from smallctl.harness.tool_message_compaction import compact_read_only_turns

    compacted, count = compact_read_only_turns(state.recent_messages, threshold_turns=8)
    # Should not compact the write turn
    assistant_msgs = [m for m in compacted if m.role == "assistant"]
    assert any("file_write" in str(m.content) for m in assistant_msgs)
    assert count == 0


# --- Trace Replay: Session 0edd99de ---

def test_trace_replay_0edd99de_step_6_no_crash_and_fama_fires() -> None:
    """Replay session 0edd99de through step 6 and verify the plan's success criteria."""
    state = LoopState(cwd="/tmp")
    state.step_count = 6
    state.current_phase = "execute"
    state.run_brief.original_task = "update vikunja with the work done in this task"

    # Step 1: dir_list(Vikunja) -> A0001
    # Step 2: file_read(agents.md) -> A0002 complete_file=true 448 lines
    # Step 3: artifact_read(A0002) at offset 1
    # Step 4: artifact_read(A0002) at offset 100
    # Step 5: model mentions truncated -> should NOT fire truncation nudge (complete file)
    # Step 6: dir_list(Vikunja) repeat -> should NOT crash, FAMA should fire

    state.artifacts["A0001"] = ArtifactRecord(
        artifact_id="A0001",
        kind="dir_list",
        source="Vikunja",
        created_at="2026-06-09T00:00:00+00:00",
        size_bytes=256,
        summary="Vikunja directory listing (9 items)",
        tool_name="dir_list",
    )
    state.artifacts["A0002"] = ArtifactRecord(
        artifact_id="A0002",
        kind="file_read",
        source="Vikunja/agents.md",
        created_at="2026-06-09T00:00:00+00:00",
        size_bytes=24576,
        summary="agents.md skill documentation (448 lines)",
        tool_name="file_read",
        inline_content="\n".join(f"line {i}" for i in range(1, 449)),
    )

    state.tool_history = [
        'dir_list|{"path": "Vikunja"}',
        'file_read|{"path": "Vikunja/agents.md"}',
        'artifact_read|{"artifact_id": "A0002", "start_line": 1}',
        'artifact_read|{"artifact_id": "A0002", "start_line": 100}',
        'dir_list|{"path": "Vikunja"}',
    ]

    state.scratchpad["_progress_read_history"] = [
        {
            "tool_name": "file_read",
            "path": "Vikunja/agents.md",
            "complete_file": True,
            "file_content_truncated": False,
            "total_lines": 448,
            "line_end": 448,
        }
    ]
    state.scratchpad["_recent_file_read_artifacts"] = [
        {"artifact_id": "A0002", "expires_after": 10, "step": 2},
    ]
    # Ensure the generic loop nudge has already been given so the guard trips on step 6 repeat.
    state.scratchpad["_generic_loop_nudged"] = 'generic_loop:dir_list:{"path": "Vikunja"}'

    # Simulate the model thinking text at step 5 mentioning truncation
    harness = SimpleNamespace(state=state, _runlog=lambda *args, **kwargs: None)
    _maybe_inject_file_truncation_hallucination_nudge(harness, "It looks like agents.md was truncated.")

    # Success criterion 3: complete file should trigger proceed nudge, not truncation hallucination
    system_messages = [m for m in state.recent_messages if m.role == "system"]
    assert len(system_messages) == 1
    assert "read completely" in system_messages[0].content.lower()

    # Success criterion 4: FAMA should fire a read-only loop capsule by step 6
    signal = detect_read_only_loop(state, threshold=4)
    assert signal is not None
    assert signal.kind is FamaFailureKind.LOOPING
    assert "read_only_loop_capsule" in signal.suggested_mitigations

    # Success criterion 4b: repeated dir_list on same path should fire
    repeat_signal = detect_repeated_read_on_same_path(state, threshold=2)
    assert repeat_signal is not None
    assert "Vikunja" in repeat_signal.evidence

    # Success criterion 2: artifact snippets should be in the next prompt frame
    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=4096, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )
    content = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)
    assert "A0002" in content

    # Success criterion 1: no crash when detecting repeated tool loop
    pending = SimpleNamespace(tool_name="dir_list", args={"path": "Vikunja"})
    repeat_error = _detect_repeated_tool_loop(harness, pending)
    assert repeat_error is not None
    assert "dir_list" in repeat_error
