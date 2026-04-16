from __future__ import annotations

import asyncio
from types import SimpleNamespace

from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.context.retrieval import build_retrieval_query
from smallctl.graph.nodes import interpret_model_output
from smallctl.graph.routing import LoopRoute
from smallctl.graph.state import PendingToolCall
from smallctl.prompts import build_system_prompt
from smallctl.phases import filter_phase_blocked_tools, phase_contract
from smallctl.tools.register import build_registry
from smallctl.state import ArtifactSnippet, ContextBrief, EvidenceRecord, ExecutionPlan, LoopState, PlanStep, WriteSession


def test_prompt_assembler_includes_plan_phase_handoff_artifacts() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "plan"
    state.scratchpad["_contract_phase"] = "plan"
    state.run_brief.original_task = "Inspect a failing build"
    state.context_briefs.append(
        ContextBrief(
            brief_id="B0001",
            created_at="2026-04-09T00:00:00+00:00",
            tier="warm",
            step_range=(1, 3),
            task_goal="Inspect a failing build",
            current_phase="explore",
            key_discoveries=["Build fails in the test stage"],
            tools_tried=["shell_exec"],
            blockers=["pytest exits with code 1"],
            files_touched=["pyproject.toml"],
            artifact_ids=["A0001"],
            next_action_hint="Turn the evidence into a repair plan",
            staleness_step=3,
            facts_confirmed=["pytest fails"],
            facts_unconfirmed=["Which test module fails first?"],
            open_questions=["Is the failure deterministic?"],
            candidate_causes=["Dependency drift"],
            disproven_causes=["Missing local checkout"],
            next_observations_needed=["Read the verifier output"],
            evidence_refs=["E0001"],
            claim_refs=[],
        )
    )
    state.reasoning_graph.evidence_records.append(
        EvidenceRecord(
            evidence_id="E0001",
            kind="observation",
            statement="shell_exec reported pytest exit code 1",
            phase="explore",
            tool_name="shell_exec",
            operation_id="op-1",
            artifact_id="A0001",
            source="pytest -q",
        )
    )
    state.active_plan = ExecutionPlan(
        plan_id="plan-1",
        goal="Inspect a failing build",
        status="draft",
        claim_refs=["C1"],
        steps=[PlanStep(step_id="P1", title="Read failure output", claim_refs=["C1"])],
    )

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    system_content = assembly.messages[0]["content"]
    assert "Phase handoff: plan from compressed evidence" in system_content
    assert "E0001" in system_content
    assert "shell_exec reported pytest exit code 1" in system_content
    assert "C1" in system_content


def test_write_session_contract_is_explicit_in_system_prompt_and_working_memory() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Update src/app.py"
    state.working_memory.current_goal = "Update src/app.py"
    state.write_session = WriteSession(
        write_session_id="ws-1",
        write_target_path="src/app.py",
        write_session_intent="replace_file",
        write_session_mode="chunked_author",
        write_staging_path="/tmp/.smallctl/write_sessions/ws-1-stage.py",
        write_current_section="imports",
        write_next_section="implementation",
        write_sections_completed=["header"],
        write_pending_finalize=True,
        write_last_verifier={
            "verdict": "fail",
            "command": "pytest -q",
            "output": "AssertionError: example failure output that is long enough to be clipped.",
        },
    )
    state.scratchpad["_session_notepad"] = {
        "entries": [
            "Remember the target is src/app.py, not the staged snapshot.",
            "Keep the implementation small and verify after each chunk.",
        ]
    }

    system_prompt = build_system_prompt(state, "author")
    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    system_content = system_prompt
    prompt_content = assembly.messages[0]["content"]

    assert "staging=" in system_content
    assert "pending_finalize=yes" in system_content
    assert "next_action=continue section implementation" in system_content
    assert "The staging path is for read/verify only" in system_content
    assert "Active Write Session" in prompt_content
    assert "Target: src/app.py" in prompt_content
    assert "Staging: /tmp/.smallctl/write_sessions/ws-1-stage.py" in prompt_content
    assert "Current section: imports" in prompt_content
    assert "Next section: implementation" in prompt_content
    assert "Pending finalize: yes" in prompt_content
    assert "Next action: Finalize the staged copy after verification." in prompt_content
    assert "Last verifier verdict: fail" in prompt_content
    assert "Session notepad:" in prompt_content
    assert "Keep this brief" in prompt_content


def test_prompt_context_sanitizes_legacy_tool_intent_labels() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Inspect the remote server"
    state.active_intent = "use_shell_exec"

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )
    retrieval_query = build_retrieval_query(state)

    system_content = assembly.messages[0]["content"]
    assert "requested_shell_exec" in system_content
    assert "use_shell_exec" not in system_content
    assert "Intent: requested_shell_exec" in retrieval_query
    assert "use_shell_exec" not in retrieval_query


def test_prompt_assembler_marks_artifact_snippets_as_not_full_reads() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Continue writing temp/app.py from the latest staged content"

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
        retrieved_artifacts=[ArtifactSnippet(artifact_id="A0007", text="temp/app.py preview snippet")],
    )

    combined_content = "\n".join(str(message.get("content") or "") for message in assembly.messages)
    assert "Artifact summaries (compressed evidence only; these snippets are not full artifact reads):" in combined_content
    assert "temp/app.py preview snippet" in combined_content


def test_staged_phase_contract_blocks_disallowed_tools_in_explore() -> None:
    events: list[object] = []

    class _FakeHarness:
        def __init__(self) -> None:
            self.state = LoopState(cwd="/tmp")
            self.state.current_phase = "explore"
            self.state.strategy = {"thought_architecture": "staged_reasoning"}
            self.state.scratchpad = {}
            self.state.recent_messages = []
            self.state.step_count = 1
            self.config = SimpleNamespace(min_exploration_steps=2)
            self.log = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
            self.summarizer = None
            self.summarizer_client = None

        async def _emit(self, handler: object, event: object) -> None:
            del handler
            events.append(event)

        def _runlog(self, *args, **kwargs) -> None:
            del args, kwargs

    async def _run() -> tuple[LoopRoute, _FakeHarness]:
        harness = _FakeHarness()
        deps = SimpleNamespace(harness=harness, event_handler=None)
        graph_state = SimpleNamespace(
            run_mode="loop",
            pending_tool_calls=[
                PendingToolCall(
                    tool_name="file_write",
                    args={"path": "temp/dependency_resolver.py", "content": "print('hello')\n"},
                ),
                PendingToolCall(
                    tool_name="file_patch",
                    args={
                        "path": "temp/dependency_resolver.py",
                        "target_text": "hello",
                        "replacement_text": "goodbye",
                    },
                ),
            ],
            last_assistant_text="",
            last_thinking_text="",
            last_usage={},
            last_tool_results=[],
            final_result=None,
            error=None,
        )

        route = await interpret_model_output(graph_state, deps)
        return route, harness

    route, harness = asyncio.run(_run())

    assert route == LoopRoute.NEXT_STEP
    assert harness.state.recent_messages
    assert harness.state.recent_messages[-1].role == "user"
    assert "DISCOVERY phase" in harness.state.recent_messages[-1].content
    assert "file_write" in harness.state.recent_messages[-1].content
    assert "file_patch" in harness.state.recent_messages[-1].content
    assert events == []


def test_registry_exposes_file_patch_tool() -> None:
    class _FakeStateProvider:
        def __init__(self) -> None:
            self.state = LoopState(cwd="/tmp")
            self.log = SimpleNamespace(info=lambda *args, **kwargs: None)

    registry = build_registry(_FakeStateProvider(), registry_profiles={"core"})
    spec = registry.get("file_patch")

    assert spec is not None
    assert spec.risk == "high"
    assert spec.allowed_modes == {"loop"}
    assert spec.schema["required"] == ["path", "target_text", "replacement_text"]


def test_registry_excludes_removed_general_surface_tools() -> None:
    class _FakeStateProvider:
        def __init__(self) -> None:
            self.state = LoopState(cwd="/tmp")
            self.log = SimpleNamespace(info=lambda *args, **kwargs: None)

    registry = build_registry(
        _FakeStateProvider(),
        registry_profiles={"core", "data", "network", "mutate", "indexer"},
    )

    for removed in {
        "show_artifact",
        "artifact_recall",
        "file_append",
        "cwd_get",
        "cwd_set",
        "env_get",
        "env_set",
        "checkpoint",
        "scratch_set",
        "scratch_get",
        "scratch_list",
        "scratch_delete",
        "plan_subtask",
    }:
        assert registry.get(removed) is None


def test_verify_phase_blocks_file_patch_like_other_read_only_phases() -> None:
    blocked_tool_calls = [
        PendingToolCall(
            tool_name="file_patch",
            args={
                "path": "src/example.py",
                "target_text": "return False",
                "replacement_text": "return True",
            },
        )
    ]

    allowed, blocked = filter_phase_blocked_tools(blocked_tool_calls, phase="verify")

    assert allowed == []
    assert blocked == ["file_patch"]
    assert phase_contract("verify").blocks("file_patch") is True
