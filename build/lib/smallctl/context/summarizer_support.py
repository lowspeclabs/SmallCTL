from __future__ import annotations

from typing import Any

from ..models.conversation import ConversationMessage
from ..state import LoopState
from .observations import ObservationPacket, build_observation_packets


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
            -item[0],
        )
    )
    selected = indexed[: max(0, limit)]
    selected.sort(key=lambda item: item[0])
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
