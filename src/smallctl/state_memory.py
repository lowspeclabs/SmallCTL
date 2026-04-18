from __future__ import annotations

from typing import Any

from .experience_tags import normalize_experience_tags
from .memory_namespace import infer_memory_namespace, normalize_memory_namespace
from .models.conversation import ConversationMessage
from .state_schema import ExperienceMemory, MemoryEntry, WorkingMemory
from .state_support import (
    _coerce_float,
    _coerce_int,
    _coerce_string_list,
    _coerce_timestamp_string,
    _filter_dataclass_payload,
)


def _compact_plan_step_lines(step: Any, *, depth: int = 0) -> list[str]:
    prefix = "  " * depth
    lines = [f"{prefix}{step.compact_label()}"]
    if step.description:
        lines.append(f"{prefix}  {step.description}")
    if step.notes:
        for note in step.notes:
            lines.append(f"{prefix}  note: {note}")
    if step.evidence_refs:
        lines.append(f"{prefix}  evidence: {', '.join(step.evidence_refs)}")
    if step.claim_refs:
        lines.append(f"{prefix}  claims: {', '.join(step.claim_refs)}")
    for substep in step.substeps:
        lines.extend(_compact_plan_step_lines(substep, depth=depth + 1))
    return lines


def _trim_recent_messages(
    messages: list[ConversationMessage],
    *,
    limit: int,
) -> list[ConversationMessage]:
    if len(messages) <= limit:
        return list(messages)

    user_indices = [i for i, m in enumerate(messages) if m.role == "user"]
    if not user_indices:
        return list(messages[-limit:])

    anchor = messages[user_indices[0]]
    suffix_size = max(limit - 1, 1)
    start_idx = len(messages) - suffix_size
    while start_idx > 0 and messages[start_idx].role == "tool":
        start_idx -= 1

    suffix = messages[start_idx:]
    if any(m is anchor for m in suffix):
        return list(suffix)

    result = [anchor] + list(suffix)
    if len(result) > limit:
        to_drop = len(result) - limit
        trimmed_suffix = list(suffix)
        while to_drop > 0 and trimmed_suffix:
            trimmed_suffix.pop(0)
            to_drop -= 1
        while trimmed_suffix and trimmed_suffix[0].role == "tool":
            trimmed_suffix.pop(0)
        return [anchor] + trimmed_suffix
    return result


def align_memory_entries(
    contents: list[str],
    previous_meta: list[Any],
    *,
    current_step: int,
    current_phase: str,
    confidence: float = 0.7,
) -> list[Any]:
    meta: list[Any] = []
    lookup = {m.content: m for m in previous_meta}
    for text in contents:
        if text in lookup:
            meta.append(lookup[text])
        else:
            meta.append(
                MemoryEntry(
                    content=text,
                    created_at_step=current_step,
                    created_phase=current_phase,
                    confidence=confidence,
                )
            )
    return meta


def memory_entry_is_stale(
    entry: Any,
    *,
    current_step: int,
    current_phase: str,
    staleness_step_limit: int = 12,
) -> bool:
    if entry.freshness == "pinned":
        return False
    if entry.freshness in {"stale", "invalidated"}:
        return True
    if entry.freshness == "phase" and entry.created_phase != current_phase:
        return True
    if staleness_step_limit > 0:
        age = current_step - entry.created_at_step
        if age >= staleness_step_limit:
            return True
    return False


def _coerce_working_memory(value: Any, *, current_step: int = 0, current_phase: str = "explore") -> Any:
    if isinstance(value, WorkingMemory):
        return value
    if isinstance(value, dict):
        payload = _filter_dataclass_payload(WorkingMemory, value)
        payload["current_goal"] = str(payload.get("current_goal") or "")
        for key in (
            "plan",
            "decisions",
            "open_questions",
            "known_facts",
            "failures",
            "next_actions",
        ):
            payload[key] = _coerce_string_list(payload.get(key))

        payload["known_fact_meta"] = _coerce_memory_entry_list(
            payload.get("known_fact_meta"),
            fallback_texts=payload["known_facts"],
            current_step=current_step,
            current_phase=current_phase,
        )
        payload["failure_meta"] = _coerce_memory_entry_list(
            payload.get("failure_meta"),
            fallback_texts=payload["failures"],
            current_step=current_step,
            current_phase=current_phase,
        )
        payload["next_action_meta"] = _coerce_memory_entry_list(
            payload.get("next_action_meta"),
            fallback_texts=payload["next_actions"],
            current_step=current_step,
            current_phase=current_phase,
        )
        return WorkingMemory(**payload)
    return WorkingMemory()


def _coerce_experience_memory(value: Any) -> Any:
    if isinstance(value, ExperienceMemory):
        return value
    if not isinstance(value, dict):
        return ExperienceMemory(memory_id="unknown")
    payload = dict(value)
    memory_id = str(payload.get("memory_id", "") or "unknown")
    tier = str(payload.get("tier", "warm") or "warm").strip().lower()
    if tier not in {"warm", "cold"}:
        tier = "warm"
    source = str(payload.get("source", "observed") or "observed").strip().lower()
    if source not in {"manual", "observed", "summarized", "imported"}:
        source = "observed"
    outcome = str(payload.get("outcome", "partial") or "partial").strip().lower()
    if outcome not in {"success", "failure", "partial"}:
        outcome = "partial"
    confidence = _coerce_float(payload.get("confidence"), default=0.0)
    confidence = max(0.0, min(1.0, confidence))
    last_reinforced_at = payload.get("last_reinforced_at")
    expires_at = payload.get("expires_at")
    namespace = normalize_memory_namespace(payload.get("namespace"))
    if not namespace:
        namespace = infer_memory_namespace(
            tool_name=str(payload.get("tool_name", "") or ""),
            intent=str(payload.get("intent", "") or ""),
            intent_tags=_coerce_string_list(payload.get("intent_tags")),
            environment_tags=_coerce_string_list(payload.get("environment_tags")),
            entity_tags=_coerce_string_list(payload.get("entity_tags")),
            notes=str(payload.get("notes", "") or ""),
        )
    return ExperienceMemory(
        memory_id=memory_id,
        tier=tier,
        source=source,
        created_at=_coerce_timestamp_string(payload.get("created_at")),
        last_reinforced_at=None if last_reinforced_at in (None, "") else str(last_reinforced_at),
        run_id=str(payload.get("run_id", "") or ""),
        phase=str(payload.get("phase", "") or ""),
        intent=str(payload.get("intent", "") or ""),
        namespace=namespace,
        intent_tags=normalize_experience_tags(_coerce_string_list(payload.get("intent_tags"))),
        environment_tags=normalize_experience_tags(_coerce_string_list(payload.get("environment_tags"))),
        entity_tags=normalize_experience_tags(_coerce_string_list(payload.get("entity_tags"))),
        action_type=str(payload.get("action_type", "") or ""),
        tool_name=str(payload.get("tool_name", "") or ""),
        arguments_fingerprint=str(payload.get("arguments_fingerprint", "") or ""),
        outcome=outcome,
        failure_mode=str(payload.get("failure_mode", "") or ""),
        confidence=confidence,
        reuse_count=max(0, _coerce_int(payload.get("reuse_count"), default=0)),
        notes=str(payload.get("notes", "") or ""),
        evidence_refs=_coerce_string_list(payload.get("evidence_refs")),
        supersedes=_coerce_string_list(payload.get("supersedes")),
        pinned=bool(payload.get("pinned")),
        expires_at=None if expires_at in (None, "") else str(expires_at),
    )


def _coerce_memory_entry(value: Any, *, current_step: int, current_phase: str) -> Any | None:
    if isinstance(value, MemoryEntry):
        return value
    if not isinstance(value, dict):
        if value is None:
            return None
        return MemoryEntry(
            content=str(value),
            created_at_step=current_step,
            created_phase=current_phase,
        )
    payload = dict(value)
    content = str(payload.get("content", "") or "")
    if not content:
        return None
    created_at_step = _coerce_int(payload.get("created_at_step"), default=current_step)
    created_phase = str(payload.get("created_phase", "") or current_phase)
    freshness = str(payload.get("freshness", "") or "current")
    confidence_raw = payload.get("confidence")
    confidence = None
    if confidence_raw not in (None, ""):
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = None
    return MemoryEntry(
        content=content,
        created_at_step=created_at_step,
        created_phase=created_phase,
        freshness=freshness,
        confidence=confidence,
    )


def _coerce_memory_entry_list(
    value: Any,
    *,
    fallback_texts: list[str],
    current_step: int,
    current_phase: str,
) -> list[Any]:
    if isinstance(value, list):
        entries = [
            entry
            for item in value
            if (entry := _coerce_memory_entry(item, current_step=current_step, current_phase=current_phase)) is not None
        ]
        if entries:
            return entries
    return [
        MemoryEntry(
            content=text,
            created_at_step=current_step,
            created_phase=current_phase,
        )
        for text in fallback_texts
        if text
    ]
