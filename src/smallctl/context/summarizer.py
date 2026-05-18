from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

from ..models.conversation import ConversationMessage
from ..state import ContextBrief, EpisodicSummary, LoopState, TurnBundle
from .policy import ContextPolicy, estimate_text_tokens
from .observations import ObservationPacket, build_observation_packets
from ..client import OpenAICompatClient


@dataclass(slots=True)
class CompactionAttemptResult:
    summary: EpisodicSummary | None = None
    messages_compacted: int = 0
    noop_reason: str = ""


class ContextSummarizer:
    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy()

    @staticmethod
    def _format_message_for_compaction(message: ConversationMessage) -> str:
        role = message.role
        if message.metadata.get("hidden_from_prompt") is True and role == "assistant":
            role = "commentary"
        return f"{role}: {message.content or ''}"

    @staticmethod
    def _clear_stale_marker(state: LoopState, *, key: str, item_id: str) -> None:
        clear_fn = getattr(state, "_clear_lane_staleness", None)
        if callable(clear_fn):
            clear_fn(key, item_id)

    def compact_recent_messages_with_status(
        self,
        *,
        state: LoopState,
        keep_recent: int,
        artifact_store: Any | None = None,
    ) -> CompactionAttemptResult:
        if len(state.recent_messages) <= keep_recent:
            return CompactionAttemptResult(noop_reason="no_compactable_messages")
        old_messages = state.recent_messages[:-keep_recent]
        if not old_messages:
            return CompactionAttemptResult(noop_reason="no_compactable_messages")
        summary_id = f"S{len(state.episodic_summaries) + 1:04d}"
        artifact_ids = [
            message.metadata.get("artifact_id", "")
            for message in old_messages
            if isinstance(message.metadata, dict) and message.metadata.get("artifact_id")
        ]
        files_touched = []
        for artifact_id in artifact_ids:
            artifact = state.artifacts.get(artifact_id)
            if artifact and artifact.source:
                files_touched.append(artifact.source)
        summary = EpisodicSummary(
            summary_id=summary_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            decisions=_collect_messages(old_messages, role="assistant", limit=3),
            files_touched=_dedupe(files_touched)[:6],
            failed_approaches=state.working_memory.failures[-3:],
            remaining_plan=state.working_memory.next_actions[-3:],
            artifact_ids=_dedupe([artifact_id for artifact_id in artifact_ids if artifact_id])[:8],
            notes=_collect_messages(old_messages, role="user", limit=2),
        )
        
        if artifact_store:
            # We use the decisions and notes collected to form a slightly more detailed manual artifact
            decisions = summary.decisions
            full_text = f"Episodic Summary {summary_id}\n\nSignificant Decisions:\n" + "\n".join(decisions)
            art = artifact_store.persist_thinking(raw_thinking=full_text, summary=f"Heuristic context summary {summary_id}", source="harness")
            summary.full_summary_artifact_id = art.artifact_id
            state.artifacts[art.artifact_id] = art
            self._clear_stale_marker(state, key="_artifact_staleness", item_id=art.artifact_id)
            # Mark the decision list with the link
            summary.decisions = [f"Heuristic Summary (full in {art.artifact_id})"] + decisions[:2]

        state.episodic_summaries.append(summary)
        self._clear_stale_marker(state, key="_summary_staleness", item_id=summary.summary_id)
        state.recent_messages = state.recent_messages[-keep_recent:]
        if summary.artifact_ids:
            state.working_memory.decisions = _dedupe(
                state.working_memory.decisions
                + [f"Compacted context into {summary.summary_id} with artifacts {', '.join(summary.artifact_ids)}"]
            )[-8:]
        return CompactionAttemptResult(summary=summary, messages_compacted=len(old_messages))

    def compact_recent_messages(
        self,
        *,
        state: LoopState,
        keep_recent: int,
        artifact_store: Any | None = None,
    ) -> EpisodicSummary | None:
        return self.compact_recent_messages_with_status(
            state=state,
            keep_recent=keep_recent,
            artifact_store=artifact_store,
        ).summary

    async def compact_recent_messages_async_with_status(
        self,
        *,
        state: LoopState,
        client: OpenAICompatClient,
        keep_recent: int,
        artifact_store: Any | None = None,
    ) -> CompactionAttemptResult:
        if len(state.recent_messages) <= keep_recent:
            return CompactionAttemptResult(noop_reason="no_compactable_messages")
        old_messages = state.recent_messages[:-keep_recent]
        if not old_messages:
            return CompactionAttemptResult(noop_reason="no_compactable_messages")

        # Build summarization prompt
        conv_text = "\n".join([self._format_message_for_compaction(m) for m in old_messages])
        prompt = (
            "Summarize the following conversation history into a concise episodic memory. "
            "Focus on: 1) Key decisions made. 2) Facts learned. 3) Any failed approaches to avoid. "
            "Keep it under 300 words. Format as a series of bullet points."
        )
        
        system_msg = {"role": "system", "content": prompt}
        user_msg = {"role": "user", "content": conv_text}
        
        chunks = []
        async for event in client.stream_chat(messages=[system_msg, user_msg], tools=[]):
            chunks.append(event)
        
        result = OpenAICompatClient.collect_stream(chunks)
        summary_text = result.assistant_text.strip()

        # Create episodic summary
        summary_id = f"S{len(state.episodic_summaries) + 1:04d}"
        artifact_ids = [
            message.metadata.get("artifact_id", "")
            for message in old_messages
            if isinstance(message.metadata, dict) and message.metadata.get("artifact_id")
        ]
        files_touched = []
        for artifact_id in artifact_ids:
            artifact = state.artifacts.get(artifact_id)
            if artifact and artifact.source:
                files_touched.append(artifact.source)

        summary = EpisodicSummary(
            summary_id=summary_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            decisions=[f"AI Summary: {summary_text[:200]}..."], # simplified for state
            files_touched=_dedupe(files_touched)[:6],
            failed_approaches=state.working_memory.failures[-3:],
            remaining_plan=state.working_memory.next_actions[-3:],
            artifact_ids=_dedupe([artifact_id for artifact_id in artifact_ids if artifact_id])[:8],
            notes=[summary_text],
        )
        
        if artifact_store:
            art = artifact_store.persist_thinking(raw_thinking=summary_text, summary=f"AI context summary {summary_id}", source="assistant_summarizer")
            summary.full_summary_artifact_id = art.artifact_id
            state.artifacts[art.artifact_id] = art
            self._clear_stale_marker(state, key="_artifact_staleness", item_id=art.artifact_id)
            # Update decisions to note link
            summary.decisions = [f"AI Summary (full in {art.artifact_id}): {summary_text[:200]}..."]

        state.episodic_summaries.append(summary)
        self._clear_stale_marker(state, key="_summary_staleness", item_id=summary.summary_id)
        state.recent_messages = state.recent_messages[-keep_recent:]
        
        return CompactionAttemptResult(summary=summary, messages_compacted=len(old_messages))

    async def compact_recent_messages_async(
        self,
        *,
        state: LoopState,
        client: OpenAICompatClient,
        keep_recent: int,
        artifact_store: Any | None = None,
    ) -> EpisodicSummary | None:
        return (
            await self.compact_recent_messages_async_with_status(
                state=state,
                client=client,
                keep_recent=keep_recent,
                artifact_store=artifact_store,
            )
        ).summary

    async def compact_to_brief_async(
        self,
        *,
        state: LoopState,
        client: OpenAICompatClient,
        messages: list[ConversationMessage],
        step_range: tuple[int, int],
        artifact_store: Any | None = None,
    ) -> ContextBrief | None:
        if not messages:
            return None

        # Build summarization prompt for structured output
        conv_text = "\n".join([self._format_message_for_compaction(m) for m in messages])
        
        system_prompt = (
            "You are a context compression engine. Your goal is to summarize a segment of conversation "
            "into a machine-readable JSON object that preserves operational progress.\n\n"
            "Output ONLY valid JSON with this schema:\n"
            "{\n"
            "  \"key_discoveries\": [\"fact 1\", ...],  // Critical facts learned\n"
            "  \"tools_tried\": [\"tool1\", ...],        // List of tool names used\n"
            "  \"blockers\": [\"error 1\", ...],         // Failures or blockers encountered\n"
            "  \"next_action_hint\": \"...\",            // What was about to happen next\n"
            "  \"new_facts\": [\"...\", ...],            // Newly established facts in this span\n"
            "  \"invalidated_facts\": [\"...\", ...],    // Facts that were invalidated\n"
            "  \"state_changes\": [\"...\", ...],        // Important state transitions\n"
            "  \"decision_deltas\": [\"...\", ...],      // Decision-level updates\n"
            "  \"plan_state\": {\"goal\": \"...\", \"active_step\": \"...\", \"open_questions\": [], \"prior_failures\": [], \"next_actions\": []},\n"
            "  \"evidence_refs\": [\"...\", ...],        // Evidence or observation ids preserved from state\n"
            "  \"observation_refs\": [\"...\", ...],     // Observation ids preserved from state\n"
            "  \"experience_candidates\": [\"...\", ...] // Reusable lessons or failure patterns\n"
            "}"
        )
        
        user_prompt = f"Summarize these messages from steps {step_range[0]} to {step_range[1]}:\n\n{conv_text}"
        
        chunks = []
        async for event in client.stream_chat(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tools=[]
        ):
            chunks.append(event)
        
        result = OpenAICompatClient.collect_stream(chunks)
        raw_json = result.assistant_text.strip()
        
        # Clean potential markdown wrapping
        if raw_json.startswith("```json"):
            raw_json = raw_json.split("```json", 1)[1].split("```", 1)[0].strip()
        elif raw_json.startswith("```"):
            raw_json = raw_json.split("```", 1)[1].split("```", 1)[0].strip()
            
        try:
            data = json.loads(raw_json)
        except Exception:
            # Fallback if AI fails to produce valid JSON
            data = {
                "key_discoveries": ["Failed to extract discoveries from transcript"],
                "tools_tried": [],
                "blockers": ["Model failed to produce structured JSON summary"],
                "next_action_hint": "Proceed with task"
            }

        brief_id = f"B{len(state.context_briefs) + 1:04d}"
        
        # Collect artifact IDs from the messages
        artifact_ids = [
            m.metadata.get("artifact_id") 
            for m in messages 
            if isinstance(m.metadata, dict) and m.metadata.get("artifact_id")
        ]
        artifact_ids = _dedupe([str(aid) for aid in artifact_ids if aid])
        
        # Collect files touched from artifacts
        files_touched = []
        for aid in artifact_ids:
            art = state.artifacts.get(aid)
            if art and art.source:
                files_touched.append(art.source)
        files_touched = _dedupe(files_touched)
        observation_packets = build_observation_packets(state, limit=12)
        lane_metadata = extract_rewoo_lanes_from_messages(
            state=state,
            messages=messages,
            observation_packets=observation_packets,
        )
        evidence_refs = _dedupe(
            _string_list(data.get("evidence_refs"))
            + lane_metadata["evidence_refs"]
            + artifact_ids
        )
        observation_refs = _dedupe(_string_list(data.get("observation_refs", data.get("evidence_refs", []))) + lane_metadata["evidence_refs"])
        plan_state = data.get("plan_state")
        if not isinstance(plan_state, dict):
            plan_state = {}
        plan_state = {**lane_metadata["plan_state"], **plan_state}
        experience_candidates = _dedupe(_string_list(data.get("experience_candidates")) + lane_metadata["experience_candidates"])

        brief = ContextBrief(
            brief_id=brief_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tier="warm",
            step_range=step_range,
            task_goal=state.run_brief.original_task,
            current_phase=state.current_phase,
            key_discoveries=data.get("key_discoveries", []),
            tools_tried=data.get("tools_tried", []),
            blockers=data.get("blockers", []),
            files_touched=files_touched,
            artifact_ids=artifact_ids,
            next_action_hint=data.get("next_action_hint", ""),
            staleness_step=state.step_count,
            facts_confirmed=data.get("facts_confirmed", data.get("key_discoveries", [])),
            facts_unconfirmed=data.get("facts_unconfirmed", []),
            open_questions=data.get("open_questions", []),
            candidate_causes=data.get("candidate_causes", []),
            disproven_causes=data.get("disproven_causes", []),
            next_observations_needed=data.get("next_observations_needed", []),
            evidence_refs=evidence_refs[:12],
            observation_refs=observation_refs[:12],
            claim_refs=data.get("claim_refs", []),
            new_facts=data.get("new_facts", data.get("facts_confirmed", data.get("key_discoveries", []))),
            invalidated_facts=data.get("invalidated_facts", []),
            state_changes=data.get("state_changes", []),
            decision_deltas=data.get("decision_deltas", []),
            plan_state=plan_state,
            experience_candidates=experience_candidates[:6],
        )
        
        if artifact_store:
            full_text = f"Context Brief {brief_id} (Steps {step_range[0]}-{step_range[1]})\n\n"
            full_text += f"Goal: {brief.task_goal}\n"
            full_text += f"Phase: {brief.current_phase}\n"
            full_text += "Discoveries:\n- " + "\n- ".join(brief.key_discoveries) + "\n\n"
            if brief.new_facts:
                full_text += "New Facts:\n- " + "\n- ".join(brief.new_facts) + "\n\n"
            if brief.invalidated_facts:
                full_text += "Invalidated Facts:\n- " + "\n- ".join(brief.invalidated_facts) + "\n\n"
            if brief.state_changes:
                full_text += "State Changes:\n- " + "\n- ".join(brief.state_changes) + "\n\n"
            if brief.decision_deltas:
                full_text += "Decision Deltas:\n- " + "\n- ".join(brief.decision_deltas) + "\n\n"
            if brief.plan_state:
                full_text += "Plan State:\n" + json.dumps(brief.plan_state, indent=2, sort_keys=True) + "\n\n"
            if brief.experience_candidates:
                full_text += "Experience Candidates:\n- " + "\n- ".join(brief.experience_candidates) + "\n\n"
            full_text += "Blockers:\n- " + "\n- ".join(brief.blockers) + "\n\n"
            full_text += f"Next Action Hint: {brief.next_action_hint}\n"
            
            art = artifact_store.persist_thinking(
                raw_thinking=full_text,
                summary=f"Structured context brief {brief_id}",
                source="harness"
            )
            brief.full_artifact_id = art.artifact_id
            state.artifacts[art.artifact_id] = art
            self._clear_stale_marker(state, key="_artifact_staleness", item_id=art.artifact_id)

        self._clear_stale_marker(state, key="_context_brief_staleness", item_id=brief.brief_id)

        return brief

    def compact_to_turn_bundle(
        self,
        *,
        state: LoopState,
        messages: list[ConversationMessage],
        step_range: tuple[int, int],
        observation_packets: list[ObservationPacket] | None = None,
    ) -> TurnBundle | None:
        if not messages:
            return None
        bundle_id = f"TB{len(state.turn_bundles) + 1:04d}"
        artifact_ids = [
            str(message.metadata.get("artifact_id") or "").strip()
            for message in messages
            if isinstance(message.metadata, dict) and message.metadata.get("artifact_id")
        ]
        artifact_ids = _dedupe([artifact_id for artifact_id in artifact_ids if artifact_id])
        files_touched: list[str] = []
        for artifact_id in artifact_ids:
            artifact = state.artifacts.get(artifact_id)
            if artifact and artifact.source:
                files_touched.append(artifact.source)
        files_touched = _dedupe(files_touched)
        transcript_summary_lines = _dedupe(
            _collect_messages(messages, role="assistant", limit=3)
            + _collect_messages(messages, role="tool", limit=2)
            + _collect_messages(messages, role="user", limit=1)
        )[:6]
        selected_packets = _select_compaction_observation_packets(
            state=state,
            messages=messages,
            artifact_ids=artifact_ids,
            supplied_packets=observation_packets,
            limit=max(8, min(16, len(messages) * 3)),
        )
        observation_summary_lines = _dedupe(
            [_format_observation_summary_for_bundle(packet) for packet in selected_packets]
        )[:6]
        observation_refs = _dedupe(
            [
                str(packet.observation_id or "").strip()
                for packet in selected_packets
                if str(packet.observation_id or "").strip()
            ]
        )[:12]
        observation_kinds = _dedupe(
            [str(packet.kind or "").strip() for packet in selected_packets if str(packet.kind or "").strip()]
        )[:8]
        min_observation_lines = 2 if len(messages) >= 3 else 1
        transcript_fallback_used = len(observation_summary_lines) < min_observation_lines
        if observation_summary_lines:
            summary_lines = list(observation_summary_lines)
            if transcript_fallback_used:
                for line in transcript_summary_lines:
                    if line and line not in summary_lines:
                        summary_lines.append(line)
                    if len(summary_lines) >= 6:
                        break
            compaction_strategy = (
                "observation_first_with_transcript_fallback"
                if transcript_fallback_used
                else "observation_first"
            )
        else:
            summary_lines = transcript_summary_lines
            compaction_strategy = "transcript_fallback"
            transcript_fallback_used = True
        evidence_refs = _dedupe(
            observation_refs
            + [record.evidence_id for record in state.reasoning_graph.evidence_records[-6:] if record.evidence_id]
        )[:12]
        lane_metadata = extract_rewoo_lanes_from_messages(
            state=state,
            messages=messages,
            observation_packets=selected_packets,
        )
        evidence_refs = _dedupe(evidence_refs + lane_metadata["evidence_refs"])[:12]
        return TurnBundle(
            bundle_id=bundle_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            step_range=step_range,
            phase=state.current_phase,
            intent=state.active_intent,
            summary_lines=summary_lines,
            files_touched=files_touched[:6],
            artifact_ids=artifact_ids[:8],
            evidence_refs=evidence_refs,
            observation_refs=observation_refs,
            observation_summaries=observation_summary_lines,
            observation_kinds=observation_kinds,
            plan_state=lane_metadata["plan_state"],
            decision_deltas=lane_metadata["decision_deltas"],
            experience_candidates=lane_metadata["experience_candidates"],
            compaction_strategy=compaction_strategy,
            transcript_fallback_used=transcript_fallback_used,
            source_message_count=len(messages),
        )

    def compact_turn_bundles_to_brief(
        self,
        *,
        state: LoopState,
        bundles: list[TurnBundle],
        step_range: tuple[int, int],
        artifact_store: Any | None = None,
    ) -> ContextBrief | None:
        if not bundles:
            return None
        brief_id = f"B{len(state.context_briefs) + 1:04d}"
        summary_lines: list[str] = []
        files_touched: list[str] = []
        artifact_ids: list[str] = []
        evidence_refs: list[str] = []
        observation_refs: list[str] = []
        decision_deltas: list[str] = []
        experience_candidates: list[str] = []
        plan_state: dict[str, Any] = {}
        for bundle in bundles:
            summary_lines.extend(bundle.summary_lines)
            files_touched.extend(bundle.files_touched)
            artifact_ids.extend(bundle.artifact_ids)
            evidence_refs.extend(bundle.evidence_refs)
            observation_refs.extend(bundle.observation_refs)
            decision_deltas.extend(bundle.decision_deltas)
            experience_candidates.extend(bundle.experience_candidates)
            for key, value in bundle.plan_state.items():
                if value not in ("", [], {}, None):
                    plan_state[key] = value
        summary_lines = _dedupe(summary_lines)
        files_touched = _dedupe(files_touched)
        artifact_ids = _dedupe(artifact_ids)
        evidence_refs = _dedupe(evidence_refs)
        observation_refs = _dedupe(observation_refs)
        key_discoveries = summary_lines[:4] or ["Turn-bundle compaction captured recent progress"]
        next_action_hint = (
            state.working_memory.next_actions[-1]
            if state.working_memory.next_actions
            else (state.run_brief.current_phase_objective or state.run_brief.original_task)
        )
        brief = ContextBrief(
            brief_id=brief_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            tier="warm",
            step_range=step_range,
            task_goal=state.run_brief.original_task,
            current_phase=state.current_phase,
            key_discoveries=key_discoveries,
            tools_tried=_dedupe([record.tool_name for record in state.reasoning_graph.evidence_records[-6:] if record.tool_name])[:6],
            blockers=state.working_memory.failures[-4:],
            files_touched=files_touched[:8],
            artifact_ids=artifact_ids[:10],
            next_action_hint=next_action_hint,
            staleness_step=state.step_count,
            facts_confirmed=key_discoveries[:4],
            evidence_refs=evidence_refs[:12],
            observation_refs=observation_refs[:12],
            new_facts=key_discoveries[:4],
            state_changes=_dedupe([f"Compacted turn bundle {bundle.bundle_id}" for bundle in bundles])[:4],
            decision_deltas=_dedupe(decision_deltas)[:8] or summary_lines[:4],
            plan_state=plan_state,
            experience_candidates=_dedupe(experience_candidates)[:6],
        )
        if artifact_store is not None:
            lines = [
                f"Turn-bundle brief {brief_id}",
                f"Step range: {step_range[0]}-{step_range[1]}",
                "Summary lines:",
                *[f"- {line}" for line in summary_lines[:12]],
            ]
            artifact = artifact_store.persist_thinking(
                raw_thinking="\n".join(lines),
                summary=f"Turn-bundle context brief {brief_id}",
                source="tier_compaction",
            )
            brief.full_artifact_id = artifact.artifact_id
            state.artifacts[artifact.artifact_id] = artifact
            self._clear_stale_marker(state, key="_artifact_staleness", item_id=artifact.artifact_id)
        self._clear_stale_marker(state, key="_context_brief_staleness", item_id=brief.brief_id)
        return brief

    async def distill_thinking_async(
        self,
        *,
        client: OpenAICompatClient,
        thinking_text: str,
        task: str,
    ) -> str:
        if not thinking_text or len(thinking_text) < 200:
            return ""
        
        system_prompt = (
            "You are a meta-cognition summarizer. "
            f"The assistant just thought through the following for the task: '{task}'. "
            "Extract the most critical 'lesson learned' or 'operational insight' from this thinking. "
            "Output ONLY the insight in one concise sentence starting with 'Note:', 'Always:', or 'Never:'."
        )
        
        user_msg = {"role": "user", "content": f"Thinking block:\n{thinking_text[:4000]}"}
        
        chunks = []
        async for event in client.stream_chat(messages=[{"role": "system", "content": system_prompt}, user_msg], tools=[]):
            chunks.append(event)
        
        result = OpenAICompatClient.collect_stream(chunks)
        return result.assistant_text.strip()


    async def summarize_artifact_async(
        self,
        *,
        client: OpenAICompatClient,
        artifact_id: str,
        content: str,
        label: str | None = None,
    ) -> str:
        if not content:
            return ""
        
        label_prefix = f" of the {label}" if label else ""
        system_prompt = (
            f"You are a technical context summarizer. "
            f"The following is the content{label_prefix} stored in artifact '{artifact_id}'. "
            "Summarize it strictly for operational use in a coding/harness loop. "
            "Extract: 1) Purpose/Function, 2) Core structure (e.g. key functions, variables), 3) Key findings or segments. "
            "Ensure the summary helps an AI determine if it needs to read specific sections further. "
            "Output ONLY the summary in concise bullet points."
        )
        
        user_msg = {"role": "user", "content": f"Content to summarize (first 8000 tokens):\n{content[:32000]}"}
        
        chunks = []
        async for event in client.stream_chat(messages=[{"role": "system", "content": system_prompt}, user_msg], tools=[]):
            chunks.append(event)
        
        result = OpenAICompatClient.collect_stream(chunks)
        return result.assistant_text.strip()


def extract_rewoo_lanes_from_messages(
    state: LoopState,
    messages: list[ConversationMessage],
    observation_packets: list[ObservationPacket],
) -> dict[str, Any]:
    """Extract lane-shaped compaction hints without depending on transcript summaries."""
    del messages
    active_step = ""
    if state.active_plan is not None:
        step = state.active_plan.find_step(state.active_step_id) if state.active_step_id else state.active_plan.active_step()
        if step is not None:
            active_step = step.compact_label()
    plan_state = {
        "goal": state.working_memory.current_goal
        or state.run_brief.current_phase_objective
        or state.run_brief.original_task,
        "active_step": active_step,
        "open_questions": _dedupe(state.working_memory.open_questions[-6:]),
        "prior_failures": _dedupe(state.working_memory.failures[-6:]),
        "next_actions": _dedupe(state.working_memory.next_actions[-6:]),
    }
    evidence_refs = _dedupe(
        [
            str(packet.observation_id or "").strip()
            for packet in observation_packets
            if str(packet.observation_id or "").strip()
        ]
    )
    decision_deltas = _dedupe(
        state.working_memory.decisions[-6:]
        + [
            record.rationale_summary or record.intent_label or record.requested_tool
            for record in state.reasoning_graph.decision_records[-6:]
            if (record.rationale_summary or record.intent_label or record.requested_tool)
        ]
    )[:8]
    experience_candidates = _dedupe(
        [
            packet.summary
            for packet in observation_packets
            if packet.failure_mode or packet.kind in {"negative_observation", "tool_plan_negative_observation"}
        ]
        + list(state.working_memory.failures[-4:])
    )[:6]
    return {
        "plan_state": plan_state,
        "evidence_refs": evidence_refs[:12],
        "decision_deltas": decision_deltas,
        "experience_candidates": experience_candidates,
    }


def _select_compaction_observation_packets(
    *,
    state: LoopState,
    messages: list[ConversationMessage],
    artifact_ids: list[str],
    supplied_packets: list[ObservationPacket] | None,
    limit: int,
) -> list[ObservationPacket]:
    fetch_limit = max(limit * 3, 24)
    source_packets = supplied_packets or build_observation_packets(state, limit=fetch_limit)
    packets = [packet for packet in source_packets if packet.summary]
    if not packets:
        return []
    normalized_artifact_ids = {
        str(artifact_id or "").strip() for artifact_id in artifact_ids if str(artifact_id or "").strip()
    }
    operation_ids = {
        str(message.metadata.get("operation_id") or "").strip()
        for message in messages
        if isinstance(message.metadata, dict) and message.metadata.get("operation_id")
    }

    selected: list[ObservationPacket] = []
    if normalized_artifact_ids:
        selected.extend(
            [
                packet
                for packet in packets
                if str(packet.artifact_id or "").strip() in normalized_artifact_ids
            ]
        )
    if operation_ids:
        selected.extend(
            [
                packet
                for packet in packets
                if str(packet.operation_id or "").strip() in operation_ids
            ]
        )
    selected = _dedupe_observation_packets(selected)
    if not selected:
        selected = [packet for packet in packets if not packet.stale]
    if not selected:
        selected = list(packets)
    return _prioritize_compaction_observation_packets(selected, limit=limit)


def _dedupe_observation_packets(packets: list[ObservationPacket]) -> list[ObservationPacket]:
    deduped: list[ObservationPacket] = []
    seen: set[str] = set()
    for packet in packets:
        observation_id = str(packet.observation_id or "").strip()
        key = f"id:{observation_id}" if observation_id else f"{packet.kind}|{packet.summary}|{packet.path}|{packet.command}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(packet)
    return deduped


def _prioritize_compaction_observation_packets(
    packets: list[ObservationPacket],
    *,
    limit: int,
) -> list[ObservationPacket]:
    if not packets:
        return []
    indexed = list(enumerate(packets))
    indexed.sort(
        key=lambda item: (
            _observation_priority(item[1]),
            -item[0],  # prefer more recent packets within a priority tier
        )
    )
    selected = indexed[: max(0, limit)]
    selected.sort(key=lambda item: item[0])  # preserve chronological flow for rendering
    return [packet for _, packet in selected]


def _observation_priority(packet: ObservationPacket) -> int:
    if packet.stale:
        return 90
    adapter = str(getattr(packet, "adapter", "") or "").strip().lower()
    if adapter == "verifier_verdict":
        return 0
    if adapter in {"file_read_fact", "file_state"}:
        return 1
    kind = str(packet.kind or "").strip().lower()
    if kind == "verifier_verdict":
        return 0
    if kind == "file_fact":
        return 1
    if kind == "negative_observation":
        return 2
    if kind == "observation_list":
        return 3
    if kind == "shell_observation":
        return 4
    if kind == "artifact_replay":
        return 5
    return 6


def _format_observation_summary_for_bundle(packet: ObservationPacket) -> str:
    summary = str(packet.summary or "").strip()
    if not summary:
        return ""
    if packet.observation_id:
        return f"{packet.observation_id} {summary}"[:340]
    return summary[:320]


def _collect_messages(messages: list[ConversationMessage], *, role: str, limit: int) -> list[str]:
    values = [
        (message.content or "").strip()
        for message in messages
        if message.role == role and (message.content or "").strip()
    ]
    return values[-limit:]


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if item]
    if isinstance(value, tuple):
        return [str(item) for item in value if item]
    if value:
        return [str(value)]
    return []
