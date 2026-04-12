from __future__ import annotations

import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any

from ..models.conversation import ConversationMessage
from ..state import ContextBrief, EpisodicSummary, LoopState
from .policy import ContextPolicy, estimate_text_tokens
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
            # Mark the decision list with the link
            summary.decisions = [f"Heuristic Summary (full in {art.artifact_id})"] + decisions[:2]

        state.episodic_summaries.append(summary)
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
            # Update decisions to note link
            summary.decisions = [f"AI Summary (full in {art.artifact_id}): {summary_text[:200]}..."]

        state.episodic_summaries.append(summary)
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
            "  \"next_action_hint\": \"...\"             // What was about to happen next\n"
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
            evidence_refs=data.get("evidence_refs", artifact_ids),
            claim_refs=data.get("claim_refs", []),
        )
        
        if artifact_store:
            full_text = f"Context Brief {brief_id} (Steps {step_range[0]}-{step_range[1]})\n\n"
            full_text += f"Goal: {brief.task_goal}\n"
            full_text += f"Phase: {brief.current_phase}\n"
            full_text += "Discoveries:\n- " + "\n- ".join(brief.key_discoveries) + "\n\n"
            full_text += "Blockers:\n- " + "\n- ".join(brief.blockers) + "\n\n"
            full_text += f"Next Action Hint: {brief.next_action_hint}\n"
            
            art = artifact_store.persist_thinking(
                raw_thinking=full_text,
                summary=f"Structured context brief {brief_id}",
                source="harness"
            )
            brief.full_artifact_id = art.artifact_id
            state.artifacts[art.artifact_id] = art

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
