from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from smallctl.context import ContextPolicy, PromptAssembler
from smallctl.context.frame import PromptStateFrame
from smallctl.context.retrieval import RetrievalBundle
from smallctl.harness.prompt_builder import PromptBuilderService
from smallctl.state import ArtifactSnippet, ContextBrief, ExperienceMemory, LoopState, TurnBundle, WriteSession


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
    assert "coding_profile_enabled" in frame_entry["data"]
    assert "coding_anchor_count" in frame_entry["data"]
    assert all("active_phase" in entry["data"] for entry in lane_selected)
    assert all("active_intent" in entry["data"] for entry in lane_selected)
    assert lane_dropped and "active_phase" in lane_dropped[0]["data"]
    assert lane_dropped and "active_intent" in lane_dropped[0]["data"]


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
