from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from smallctl.context import ContextPolicy, PromptAssembler, build_retrieval_query
from smallctl.context.frame import PromptStateFrame
from smallctl.context.retrieval import RetrievalBundle
from smallctl.evidence import normalize_tool_result
from smallctl.fama.signals import ActiveMitigation
from smallctl.fama.state import activate_mitigations
from smallctl.graph.deps import GraphRuntimeDeps
from smallctl.graph.nodes import LoopRoute, interpret_model_output
from smallctl.graph.state import GraphRunState
from smallctl.harness.prompt_builder import PromptBuilderService
from smallctl.models.conversation import ConversationMessage
from smallctl.models.tool_result import ToolEnvelope
from smallctl.state import (
    ArtifactRecord,
    ArtifactSnippet,
    ContextBrief,
    EpisodicSummary,
    ExperienceMemory,
    LoopState,
    TurnBundle,
    WriteSession,
)


def _make_brief(*, brief_id: str) -> ContextBrief:
    return ContextBrief(
        brief_id=brief_id,
        created_at="2026-04-17T00:00:00+00:00",
        tier="warm",
        step_range=(1, 3),
        task_goal="Refactor prompt flow",
        current_phase="plan",
        key_discoveries=["Prompt state should be deterministic"],
        tools_tried=["shell_exec"],
        blockers=["None"],
        files_touched=["src/smallctl/context/assembler.py"],
        artifact_ids=["A100"],
        next_action_hint="Compile frame before rendering",
        staleness_step=3,
        facts_confirmed=["Assembler exists"],
        facts_unconfirmed=["Frame compiler output coverage"],
        open_questions=["Which lane drops first under pressure?"],
        candidate_causes=["Prompt budget pressure"],
        disproven_causes=[],
        next_observations_needed=["Run budget tests"],
        evidence_refs=["E100"],
        claim_refs=["C100"],
    )


def test_prompt_assembler_builds_prompt_state_frame_and_spine_fields() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "plan"
    state.run_brief.original_task = "Refactor prompt flow"
    state.run_brief.current_phase_objective = "Compile deterministic frame"
    state.run_brief.acceptance_criteria = ["Frame compiles before rendering"]
    state.working_memory.current_goal = "Compile deterministic frame"
    state.working_memory.next_actions = ["Wire frame into assembler"]
    state.active_intent = "use_shell_exec"
    state.context_briefs.append(_make_brief(brief_id="B100"))

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
        retrieved_artifacts=[ArtifactSnippet(artifact_id="A200", text="compiled frame metadata")],
        retrieved_experiences=[
            ExperienceMemory(
                memory_id="mem-1",
                intent="requested_shell_exec",
                tool_name="shell_exec",
                outcome="success",
                notes="Use frame metadata for deterministic prompt assembly.",
            )
        ],
    )

    assert assembly.frame is not None
    assert assembly.frame.spine.task_goal == "Refactor prompt flow"
    assert assembly.frame.spine.current_phase == state.contract_phase()
    assert assembly.frame.spine.active_intent == "requested_shell_exec"
    assert "Frame compiles before rendering" in assembly.frame.spine.unmet_acceptance_criteria
    assert "Run brief:" in assembly.messages[0]["content"]
    assert "Working memory:" in assembly.messages[0]["content"]


def test_prompt_assembler_renders_fama_capsules_in_working_memory() -> None:
    state = LoopState(cwd="/tmp")
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="done_gate",
                reason="verifier failed",
                source_signal="early_stop:0",
                activated_step=0,
                expires_after_step=2,
            ),
            ActiveMitigation(
                name="acceptance_checklist_capsule",
                reason="verifier failed",
                source_signal="early_stop:0",
                activated_step=0,
                expires_after_step=2,
            ),
        ],
        max_active=2,
    )

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    content = assembly.messages[0]["content"]
    assert "FAMA mitigation:" in content
    assert "Before task_complete, satisfy the latest verifier/acceptance evidence" in content
    assert "Use the acceptance checklist and latest verifier result as the finish gate." in content
    assert assembly.frame is not None
    assert len(assembly.frame.spine.fama_capsule_lines) == 2
    assert assembly.section_tokens["fama_capsules"] > 0


def test_prompt_assembler_omits_fama_capsules_when_disabled() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_fama_config"] = {"enabled": False, "mode": "lite", "capsule_token_budget": 180}
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="done_gate",
                reason="verifier failed",
                source_signal="early_stop:0",
                activated_step=0,
                expires_after_step=2,
            ),
        ],
        max_active=2,
    )

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    assert "FAMA mitigation:" not in assembly.messages[0]["content"]
    assert assembly.frame is not None
    assert assembly.frame.spine.fama_capsule_lines == []
    assert assembly.section_tokens["fama_capsules"] == 0


def test_prompt_assembler_drops_expired_fama_capsules() -> None:
    state = LoopState(cwd="/tmp", step_count=4)
    activate_mitigations(
        state,
        [
            ActiveMitigation(
                name="done_gate",
                reason="verifier failed",
                source_signal="early_stop:0",
                activated_step=0,
                expires_after_step=2,
            ),
        ],
        max_active=2,
    )

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    assert "FAMA mitigation:" not in assembly.messages[0]["content"]
    assert assembly.frame is not None
    assert assembly.frame.spine.fama_capsule_lines == []


def test_sub4b_working_memory_pins_available_evidence_sources() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_model_name"] = "qwen3.5:4b"
    state.run_brief.original_task = "Patch temp/logwatch.py"
    state.working_memory.current_goal = "Patch temp/logwatch.py"
    state.artifacts["A0001"] = ArtifactRecord(
        artifact_id="A0001",
        kind="file",
        source="temp/logwatch.py",
        created_at="2026-04-22T00:00:00+00:00",
        size_bytes=120,
        summary="file_read temp/logwatch.py lines 1-40",
        path_tags=["/tmp/temp/logwatch.py"],
        tool_name="file_read",
    )

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    content = assembly.messages[0]["content"]
    assert "Available Evidence (pinned for this 4B-or-under model" in content
    assert "artifact_id | source/path | summary" in content
    assert "A0001 | temp/logwatch.py" in content
    assert "Only page forward with artifact_read(start_line=...)" in content


def test_sub4b_working_memory_includes_top_web_findings_line() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_model_name"] = "qwen3.5:4b"
    state.run_brief.original_task = "Choose a self-hosted blog platform"
    state.working_memory.current_goal = "Choose a self-hosted blog platform"
    state.reasoning_graph.evidence_records = [
        normalize_tool_result(
            tool_name="web_search",
            result=ToolEnvelope(
                success=True,
                output={
                    "query": "best self hosted blog docker",
                    "provider": "duckduckgo",
                    "results": [
                        {
                            "result_id": "webres-1",
                            "title": "Ghost Docker Install",
                            "url": "https://example.com/ghost",
                            "domain": "example.com",
                            "snippet": "Official Docker-based install guide for Ghost.",
                        },
                        {
                            "result_id": "webres-2",
                            "title": "WriteFreely Compose Setup",
                            "url": "https://example.com/writefreely",
                            "domain": "example.com",
                            "snippet": "Compose-based self-hosted blogging setup.",
                        },
                    ],
                },
            ),
            evidence_context={"args": {"query": "best self hosted blog docker"}},
        ),
        normalize_tool_result(
            tool_name="web_fetch",
            result=ToolEnvelope(
                success=True,
                output={
                    "source_id": "webres-1",
                    "title": "Ghost Docker Install",
                    "url": "https://example.com/ghost",
                    "canonical_url": "https://example.com/ghost",
                    "domain": "example.com",
                    "text_excerpt": "Official Docker-based install guide for Ghost.",
                    "untrusted_text": "Official Docker-based install guide for Ghost.",
                },
            ),
            evidence_context={"args": {"result_id": "webres-1"}},
        ),
    ]

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    content = assembly.messages[0]["content"]
    assert "Top web findings:" in content
    assert "Ghost Docker Install" in content
    assert "Web finding:" in content


def test_large_models_keep_working_memory_artifact_list_without_sub4b_pinned_evidence_copy() -> None:
    state = LoopState(cwd="/tmp")
    state.scratchpad["_model_name"] = "gpt-oss-120b"
    state.run_brief.original_task = "Patch temp/logwatch.py"
    state.working_memory.current_goal = "Patch temp/logwatch.py"
    state.artifacts["A0001"] = ArtifactRecord(
        artifact_id="A0001",
        kind="file",
        source="temp/logwatch.py",
        created_at="2026-04-22T00:00:00+00:00",
        size_bytes=120,
        summary="file_read temp/logwatch.py lines 1-40",
        path_tags=["/tmp/temp/logwatch.py"],
        tool_name="file_read",
    )

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    content = assembly.messages[0]["content"]
    assert "Available Artifacts (compressed summaries already in context" in content
    assert "A0001: file_read temp/logwatch.py lines 1-40" in content
    assert "Available Evidence (pinned for this 4B-or-under model" not in content


def test_prompt_assembler_prefers_delta_based_brief_rendering() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Summarize deltas"
    state.context_briefs.append(
        ContextBrief(
            brief_id="B200",
            created_at="2026-04-17T00:00:00+00:00",
            tier="warm",
            step_range=(2, 4),
            task_goal="Summarize deltas",
            current_phase="author",
            key_discoveries=["legacy discovery"],
            tools_tried=["file_read"],
            blockers=[],
            files_touched=["src/example.py"],
            artifact_ids=["A200"],
            next_action_hint="Continue authoring",
            staleness_step=4,
            new_facts=["Parser now emits PromptStateFrame"],
            invalidated_facts=["Assembler reads state ad hoc"],
            state_changes=["Prompt rendering moved behind frame boundary"],
            decision_deltas=["Use frame compiler output as prompt input"],
        )
    )

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    content = "\n".join(str(message.get("content") or "") for message in assembly.messages)
    assert "New facts: Parser now emits PromptStateFrame" in content
    assert "Invalidated: Assembler reads state ad hoc" in content
    assert "State changes: Prompt rendering moved behind frame boundary" in content
    assert "Decision deltas: Use frame compiler output as prompt input" in content


def test_prompt_assembler_rehydrates_latest_user_prompt_when_trimmed() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Install Apache Guacamole on the remote host"
    state.run_brief.current_phase_objective = "execute: Install Apache Guacamole on the remote host"
    state.recent_messages = [
        ConversationMessage(role="user", content="Install Apache Guacamole on the remote host"),
        ConversationMessage(role="user", content="Use port 8080 and keep the existing container name if it exists."),
        ConversationMessage(role="assistant", content="I will check the host and container state."),
        ConversationMessage(role="tool", name="ssh_exec", content="Docker is installed."),
        ConversationMessage(role="assistant", content="Next I will inspect running containers."),
    ]

    assembly = PromptAssembler(
        ContextPolicy(max_prompt_tokens=2048, recent_message_limit=2)
    ).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    rendered = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)
    assert "Install Apache Guacamole on the remote host" in rendered
    assert "Use port 8080 and keep the existing container name if it exists." in rendered


def test_prompt_assembler_latest_visible_user_skips_recovery_nudges() -> None:
    state = LoopState(cwd="/tmp")
    state.recent_messages = [
        ConversationMessage(role="user", content="Find the vikunja docker compose file."),
        ConversationMessage(
            role="user",
            content="### SYSTEM ALERT: emit the JSON tool call now.",
            metadata={"is_recovery_nudge": True, "recovery_kind": "action_stall"},
        ),
    ]

    latest_user = PromptAssembler()._latest_visible_user_message(state)

    assert latest_user is not None
    assert latest_user.content == "Find the vikunja docker compose file."


def test_build_retrieval_query_ignores_recovery_nudge_messages() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Find the vikunja docker compose file."
    state.working_memory.current_goal = "Find the vikunja docker compose file."
    state.recent_messages = [
        ConversationMessage(role="user", content="Find the vikunja docker compose file."),
        ConversationMessage(
            role="user",
            content="### SYSTEM ALERT: emit the JSON tool call now.",
            metadata={"is_recovery_nudge": True, "recovery_kind": "action_stall"},
        ),
    ]

    query = build_retrieval_query(state)

    assert "Find the vikunja docker compose file." in query
    assert "emit the JSON tool call now" not in query


def test_recovery_turn_keeps_original_mission_visible_in_prompt_and_retrieval() -> None:
    task = "Find the vikunja docker compose file on the remote host."
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = task
    state.run_brief.current_phase_objective = "Investigate the remote deployment layout"
    state.working_memory.current_goal = task
    state.recent_messages = [ConversationMessage(role="user", content=task)]

    harness = SimpleNamespace(
        state=state,
        config=SimpleNamespace(min_exploration_steps=0),
        summarizer=None,
        _extract_planning_request=lambda task: None,
        _record_experience=lambda *args, **kwargs: None,
        _runlog=lambda *args, **kwargs: None,
        _emit=lambda *args, **kwargs: None,
        _failure=lambda error, error_type="runtime", details=None: {
            "error": error,
            "error_type": error_type,
            "details": details or {},
        },
    )
    graph_state = GraphRunState(
        loop_state=state,
        thread_id="thread-recovery-prompt",
        run_mode="loop",
    )
    graph_state.last_assistant_text = "I'll search the remote host for the docker compose file next."
    graph_state.last_thinking_text = "I should use ssh_exec to find the compose file."
    deps = GraphRuntimeDeps(harness=harness, event_handler=None)

    route = asyncio.run(interpret_model_output(graph_state, deps))

    assert route == LoopRoute.NEXT_STEP
    assert state.recent_messages[-1].metadata["recovery_kind"] == "action_stall"

    state.recent_messages.extend(
        [
            ConversationMessage(role="assistant", content="I will inspect the running containers and deployment layout."),
            ConversationMessage(role="tool", name="ssh_exec", content="vikunja container found; compose path not yet identified"),
            ConversationMessage(role="assistant", content="I still need to locate the compose file."),
        ]
    )

    assembly = PromptAssembler(
        ContextPolicy(max_prompt_tokens=2048, recent_message_limit=2)
    ).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )
    rendered = "\n\n".join(str(message.get("content") or "") for message in assembly.messages)
    query = build_retrieval_query(state)

    assert task in rendered
    assert "### SYSTEM ALERT" not in rendered
    assert task in query
    assert "emit the JSON tool call now" not in query


def test_prompt_frame_renders_structured_continuation_anchor_after_soft_switch() -> None:
    state = LoopState(cwd="/tmp")
    state.run_brief.original_task = "Continue current task: patch temp/logwatch.py. User correction: use file_patch instead"
    state.working_memory.current_goal = "patch temp/logwatch.py"
    state.scratchpad["_task_boundary_previous_task"] = "read temp/logwatch.py and identify the required fix"
    state.scratchpad["_last_task_handoff"] = {
        "effective_task": state.run_brief.original_task,
        "next_required_tool": {"tool_name": "file_patch", "required_arguments": {"path": "temp/logwatch.py"}},
        "last_failed_tool": {"tool_name": "artifact_read", "error": "already read the same artifact"},
        "last_good_artifact_ids": ["A1001"],
        "recent_research_artifact_ids": ["A2002"],
    }

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM PROMPT",
    )

    content = assembly.messages[0]["content"]
    assert "Continuation anchor:" in content
    assert "next_required_tool=file_patch" in content
    assert "last_failed_tool=artifact_read" in content
    assert "recent_artifacts=A1001" in content
    assert "recent_research_artifacts=A2002" in content


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


class _NoOpMemory:
    def update_working_memory(self, _limit: int) -> None:
        return


class _NoOpCompaction:
    async def maybe_compact_context(self, **_: object) -> None:
        return


class _Retriever:
    def retrieve_bundle(self, **_: object) -> RetrievalBundle:
        return RetrievalBundle(
            query="frame query",
            initial_query="frame query",
            summaries=[],
            artifacts=[ArtifactSnippet(artifact_id="A-lane", text="artifact context")],
            experiences=[
                ExperienceMemory(
                    memory_id="mem-lane",
                    intent="requested_shell_exec",
                    tool_name="shell_exec",
                    outcome="success",
                    notes="Use cached artifact first.",
                )
            ],
            lane_routes={
                "artifact_packet": ["A-lane"],
                "experience_packet": ["mem-lane"],
                "evidence_packet": [],
            },
        )


@dataclass
class _Assembler:
    frame: PromptStateFrame

    def build_messages(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            messages=[{"role": "system", "content": "SYSTEM PROMPT"}],
            section_tokens={"system": 1},
            estimated_prompt_tokens=1,
            frame=self.frame,
        )


class _Harness:
    def __init__(self) -> None:
        self.state = LoopState(cwd="/tmp")
        self.context_policy = ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)
        self.memory = _NoOpMemory()
        self.compaction = _NoOpCompaction()
        self.retriever = _Retriever()
        frame = PromptAssembler(self.context_policy).build_messages(
            state=self.state,
            system_prompt="SYSTEM PROMPT",
        ).frame
        assert frame is not None
        frame.add_drop(
            lane="artifact_snippets",
            reason="token_budget",
            dropped_count=1,
            dropped_ids=["A999"],
        )
        self.prompt_assembler = _Assembler(frame=frame)
        self.fresh_run = False
        self._fresh_run_turns_remaining = 0
        self.cold_memory_store = None
        self.run_logger = _RunLogger()

    def _select_retrieval_query(self) -> str:
        return "frame query"

    def _runlog(self, event: str, message: str, **data: object) -> None:
        self.run_logger.log("harness", event, message, **data)


def test_prompt_builder_logs_frame_and_lane_events() -> None:
    harness = _Harness()
    service = PromptBuilderService(harness)
    messages = asyncio.run(service.build_messages("SYSTEM PROMPT"))

    events = [entry["event"] for entry in harness.run_logger.entries]
    retrieval_entry = next(
        entry for entry in harness.run_logger.entries if entry["event"] == "retrieval_ranked_with_intent"
    )
    frame_entry = next(
        entry for entry in harness.run_logger.entries if entry["event"] == "prompt_state_frame_compiled"
    )
    lane_selected = [
        entry for entry in harness.run_logger.entries if entry["event"] == "context_lane_selected"
    ]
    lane_dropped = [
        entry for entry in harness.run_logger.entries if entry["event"] == "context_lane_dropped"
    ]
    assert messages[0]["role"] == "system"
    assert "prompt_state_frame_compiled" in events
    assert "context_lane_selected" in events
    assert "context_lane_dropped" in events
    assert "retrieval_ranked_with_intent" in events
    assert retrieval_entry["data"]["lane_routes"]["artifact_packet"] == ["A-lane"]
    assert retrieval_entry["data"]["selected_artifact_ids"] == ["A-lane"]
    assert retrieval_entry["data"]["selected_experience_ids"] == ["mem-lane"]
    assert "stale_lane_counts" in retrieval_entry["data"]
    assert "artifact_snippets" in retrieval_entry["data"]["stale_lane_counts"]
    assert "coding_profile_enabled" in frame_entry["data"]
    assert "coding_anchor_count" in frame_entry["data"]
    assert all("active_phase" in entry["data"] for entry in lane_selected)
    assert all("active_intent" in entry["data"] for entry in lane_selected)
    assert lane_dropped and "active_phase" in lane_dropped[0]["data"]
    assert lane_dropped and "active_intent" in lane_dropped[0]["data"]


def test_prompt_builder_tunes_recent_message_limit_for_remote_repair_pressure() -> None:
    calls: dict[str, int] = {}

    class _TrackingMemory:
        def update_working_memory(self, limit: int) -> None:
            calls["memory_limit"] = limit

    class _TrackingAssembler:
        def build_messages(self, **kwargs: object) -> SimpleNamespace:
            calls["assembler_limit"] = int(kwargs["recent_message_limit"])
            return SimpleNamespace(
                messages=[{"role": "system", "content": "SYSTEM PROMPT"}],
                section_tokens={"system": 1},
                estimated_prompt_tokens=1,
                frame=None,
            )

    harness = _Harness()
    harness.context_policy.recent_message_limit = 10
    harness.state.task_mode = "remote_execute"
    harness.state.current_phase = "repair"
    harness.state.prompt_budget.estimated_prompt_tokens = 13000
    harness.state.scratchpad["_observation_staleness"] = {
        f"E-{index}": {"stale": True} for index in range(7)
    }
    harness.state.recent_messages = [
        ConversationMessage(role="user", content=f"message {index}") for index in range(12)
    ]
    harness.memory = _TrackingMemory()
    harness.prompt_assembler = _TrackingAssembler()

    service = PromptBuilderService(harness)
    asyncio.run(service.build_messages("SYSTEM PROMPT"))

    assert calls["memory_limit"] == 4
    assert calls["assembler_limit"] == 4
    tuning_entry = next(
        entry for entry in harness.run_logger.entries if entry["event"] == "recent_message_limit_tuned"
    )
    assert tuning_entry["data"]["adjusted_limit"] == 4
    assert "high_prompt_budget" in tuning_entry["data"]["reasons"]


def test_remote_repair_phase_does_not_reduce_recent_window_without_prompt_pressure() -> None:
    calls: dict[str, int] = {}

    class _TrackingMemory:
        def update_working_memory(self, limit: int) -> None:
            calls["memory_limit"] = limit

    class _TrackingAssembler:
        def build_messages(self, **kwargs: object) -> SimpleNamespace:
            calls["assembler_limit"] = int(kwargs["recent_message_limit"])
            return SimpleNamespace(
                messages=[{"role": "system", "content": "SYSTEM PROMPT"}],
                section_tokens={"system": 1},
                estimated_prompt_tokens=1,
                frame=None,
            )

    harness = _Harness()
    harness.context_policy.max_prompt_tokens = 32768
    harness.context_policy.recalculate_quotas(32768)
    harness.state.task_mode = "remote_execute"
    harness.state.current_phase = "repair"
    harness.state.prompt_budget.estimated_prompt_tokens = 4860
    harness.state.recent_messages = [
        ConversationMessage(role="user", content=f"message {index}") for index in range(3)
    ]
    harness.memory = _TrackingMemory()
    harness.prompt_assembler = _TrackingAssembler()

    service = PromptBuilderService(harness)
    asyncio.run(service.build_messages("SYSTEM PROMPT"))

    assert calls["memory_limit"] == harness.context_policy.recent_message_limit
    assert calls["assembler_limit"] == harness.context_policy.recent_message_limit
    assert not any(entry["event"] == "recent_message_limit_tuned" for entry in harness.run_logger.entries)


def test_prompt_state_frame_captures_coding_anchors_and_ladder_levels() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.task_mode = "local_execute"
    state.run_brief.original_task = "Fix src/app.py and rerun verifier"
    state.files_changed_this_cycle = ["src/app.py"]
    state.last_verifier_verdict = {"verdict": "fail", "command": "pytest -q"}
    state.write_session = WriteSession(
        write_session_id="ws-1",
        write_target_path="src/app.py",
        write_session_mode="chunked_author",
        write_session_intent="patch_existing",
        status="open",
    )
    state.turn_bundles = [
        TurnBundle(
            bundle_id="TB0001",
            created_at="2026-04-18T00:00:00+00:00",
            step_range=(1, 2),
            phase="author",
            intent="requested_file_patch",
            summary_lines=["Patched src/app.py"],
            files_touched=["src/app.py"],
        )
    ]
    state.context_briefs.append(_make_brief(brief_id="B300"))

    assembly = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4)).build_messages(
        state=state,
        system_prompt="SYSTEM",
    )

    assert assembly.frame is not None
    assert any("target_file=src/app.py" in line for line in assembly.frame.spine.coding_anchor_lines)
    assert any("verifier_status=fail" in line for line in assembly.frame.spine.coding_anchor_lines)
    assert "L1" in state.prompt_budget.included_compaction_levels
    assert "L2" in state.prompt_budget.included_compaction_levels


def test_prompt_state_frame_coding_profile_preserves_l1_l2_before_l3_l4_under_pressure() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.task_mode = "local_execute"
    state.run_brief.original_task = "Fix src/app.py and rerun verifier"
    state.run_brief.current_phase_objective = "Apply patch and re-verify"
    state.files_changed_this_cycle = ["src/app.py"]
    state.write_session = WriteSession(
        write_session_id="ws-pressured",
        write_target_path="src/app.py",
        write_session_mode="chunked_author",
        write_session_intent="patch_existing",
        status="open",
    )
    state.turn_bundles = [
        TurnBundle(
            bundle_id="TB1000",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(1, 2),
            phase="author",
            intent="requested_file_patch",
            summary_lines=["Patched src/app.py and staged verification"],
            files_touched=["src/app.py"],
        )
    ]
    state.context_briefs = [
        ContextBrief(
            brief_id="B1000",
            created_at="2026-04-19T00:00:00+00:00",
            tier="warm",
            step_range=(1, 2),
            task_goal="Fix src/app.py",
            current_phase="author",
            key_discoveries=["Patch landed and parser state updated for deterministic retries"],
            tools_tried=["file_patch"],
            blockers=[],
            files_touched=["src/app.py"],
            artifact_ids=["A1000"],
            next_action_hint="Run verifier",
            staleness_step=2,
        )
    ]

    assembly = PromptAssembler(
        ContextPolicy(
            max_prompt_tokens=2048,
            recent_message_limit=4,
            warm_tier_token_budget=1,
        )
    ).build_messages(
        state=state,
        system_prompt="SYSTEM",
        token_budget=180,
        retrieved_summaries=[
            EpisodicSummary(
                summary_id="S1000",
                created_at="2026-04-19T00:00:00+00:00",
                notes=["summary note " * 20],
            )
        ],
        retrieved_artifacts=[
            ArtifactSnippet(
                artifact_id="A2000",
                text="artifact text " * 20,
            )
        ],
    )

    assert assembly.frame is not None
    assert assembly.frame.evidence_packet.turn_bundles
    has_cold_l3_or_l4 = bool(assembly.frame.evidence_packet.summaries) or bool(assembly.frame.artifact_packet.snippets)
    assert not has_cold_l3_or_l4 or bool(assembly.frame.evidence_packet.context_briefs)


def test_prompt_state_frame_respects_disabled_coding_profile() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.task_mode = "local_execute"
    state.run_brief.original_task = "Fix src/app.py and rerun verifier"
    state.files_changed_this_cycle = ["src/app.py"]
    state.last_verifier_verdict = {"verdict": "fail", "command": "pytest -q"}
    state.write_session = WriteSession(
        write_session_id="ws-1",
        write_target_path="src/app.py",
        write_session_mode="chunked_author",
        write_session_intent="patch_existing",
        status="open",
    )

    assembly = PromptAssembler(
        ContextPolicy(
            max_prompt_tokens=2048,
            recent_message_limit=4,
            coding_profile_enabled=False,
        )
    ).build_messages(
        state=state,
        system_prompt="SYSTEM",
    )

    assert assembly.frame is not None
    assert assembly.frame.spine.coding_anchor_lines == []
    assert "Coding anchors:" not in assembly.messages[0]["content"]


def test_prompt_state_frame_keeps_touched_symbols_anchor_under_pressure() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.task_mode = "local_execute"
    state.run_brief.original_task = "Fix parser and keep symbol anchors"
    state.run_brief.current_phase_objective = "Patch parser and rerun tests"
    state.files_changed_this_cycle = ["src/app.py"]
    state.scratchpad["_touched_symbols"] = ["parse_config", "ParserState", "render_diff"]
    state.write_session = WriteSession(
        write_session_id="ws-symbols",
        write_target_path="src/app.py",
        write_session_mode="chunked_author",
        write_session_intent="patch_existing",
        status="open",
    )
    state.turn_bundles = [
        TurnBundle(
            bundle_id="TB-symbols",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(1, 2),
            phase="author",
            intent="requested_file_patch",
            summary_lines=["Patched parser flow"],
            files_touched=["src/app.py"],
        )
    ]

    assembly = PromptAssembler(
        ContextPolicy(
            max_prompt_tokens=2048,
            recent_message_limit=4,
            warm_tier_token_budget=1,
        )
    ).build_messages(
        state=state,
        system_prompt="SYSTEM",
        token_budget=170,
        retrieved_summaries=[
            EpisodicSummary(
                summary_id="S-symbols",
                created_at="2026-04-19T00:00:00+00:00",
                notes=["long summary " * 20],
            )
        ],
        retrieved_artifacts=[
            ArtifactSnippet(
                artifact_id="A-symbols",
                text="artifact text " * 20,
            )
        ],
    )

    assert assembly.frame is not None
    assert any(
        "touched_symbols=parse_config, ParserState, render_diff" in line
        for line in assembly.frame.spine.coding_anchor_lines
    )
    assert "touched_symbols=parse_config, ParserState, render_diff" in assembly.messages[0]["content"]


def test_prompt_state_frame_build_messages_is_deterministic() -> None:
    state = LoopState(cwd="/tmp")
    state.current_phase = "repair"
    state.task_mode = "local_execute"
    state.run_brief.original_task = "Fix src/app.py and rerun verifier"
    state.run_brief.current_phase_objective = "Patch then verify"
    state.files_changed_this_cycle = ["src/app.py"]
    state.last_verifier_verdict = {"verdict": "fail", "command": "pytest -q"}
    state.write_session = WriteSession(
        write_session_id="ws-deterministic",
        write_target_path="src/app.py",
        write_session_mode="chunked_author",
        write_session_intent="patch_existing",
        status="open",
    )
    state.scratchpad["_touched_symbols"] = ["parse_config", "ParserState"]
    state.context_briefs.append(_make_brief(brief_id="B-deterministic"))
    state.turn_bundles = [
        TurnBundle(
            bundle_id="TB-deterministic",
            created_at="2026-04-19T00:00:00+00:00",
            step_range=(1, 2),
            phase="author",
            intent="requested_file_patch",
            summary_lines=["Patched src/app.py"],
            files_touched=["src/app.py"],
        )
    ]

    assembler = PromptAssembler(ContextPolicy(max_prompt_tokens=2048, recent_message_limit=4))
    first = assembler.build_messages(
        state=state,
        system_prompt="SYSTEM",
        token_budget=220,
        retrieved_summaries=[
            EpisodicSummary(
                summary_id="S-deterministic",
                created_at="2026-04-19T00:00:00+00:00",
                notes=["summary detail"],
            )
        ],
        retrieved_artifacts=[ArtifactSnippet(artifact_id="A-deterministic", text="artifact detail")],
        retrieved_experiences=[
            ExperienceMemory(
                memory_id="mem-deterministic",
                intent="requested_file_patch",
                tool_name="file_patch",
                outcome="success",
                notes="Prefer targeted patch then verifier rerun.",
            )
        ],
    )
    second = assembler.build_messages(
        state=state,
        system_prompt="SYSTEM",
        token_budget=220,
        retrieved_summaries=[
            EpisodicSummary(
                summary_id="S-deterministic",
                created_at="2026-04-19T00:00:00+00:00",
                notes=["summary detail"],
            )
        ],
        retrieved_artifacts=[ArtifactSnippet(artifact_id="A-deterministic", text="artifact detail")],
        retrieved_experiences=[
            ExperienceMemory(
                memory_id="mem-deterministic",
                intent="requested_file_patch",
                tool_name="file_patch",
                outcome="success",
                notes="Prefer targeted patch then verifier rerun.",
            )
        ],
    )

    assert first.messages == second.messages
    assert first.section_tokens == second.section_tokens
    assert first.estimated_prompt_tokens == second.estimated_prompt_tokens
    assert first.frame is not None and second.frame is not None
    assert first.frame.included_lane_counts() == second.frame.included_lane_counts()
    assert [(item.lane, item.reason, item.dropped_count, list(item.dropped_ids)) for item in first.frame.drop_log] == [
        (item.lane, item.reason, item.dropped_count, list(item.dropped_ids)) for item in second.frame.drop_log
    ]
