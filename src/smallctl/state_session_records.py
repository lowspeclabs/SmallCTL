from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .state_schema import ArtifactRecord, ContextBrief, EpisodicSummary, PromptBudgetSnapshot, TurnBundle, WriteSession
from .state_support import (
    _coerce_conversation_message_payload,
    _coerce_float,
    _coerce_int,
    _coerce_json_dict_payload,
    _coerce_string_list,
    _coerce_timestamp_string,
    _coerce_write_section_ranges,
    _filter_dataclass_payload,
    json_safe_value,
)


def _coerce_artifact_record(value: Any, *, artifact_id: str) -> Any:
    if isinstance(value, ArtifactRecord):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        payload.setdefault("artifact_id", artifact_id)
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload.setdefault("kind", "tool_result")
        payload.setdefault("source", "")
        payload.setdefault("size_bytes", 0)
        payload.setdefault("summary", "")
        payload["artifact_id"] = str(payload.get("artifact_id", artifact_id) or artifact_id)
        payload["kind"] = str(payload.get("kind", "tool_result") or "tool_result")
        payload["source"] = str(payload.get("source", "") or "")
        payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))
        payload["size_bytes"] = _coerce_int(payload.get("size_bytes"), default=0)
        payload["summary"] = str(payload.get("summary", "") or "")
        payload["keywords"] = _coerce_string_list(payload.get("keywords"))
        payload["path_tags"] = _coerce_string_list(payload.get("path_tags"))
        payload["tool_name"] = str(payload.get("tool_name", "") or "")
        for key in ("content_path", "inline_content", "preview_text"):
            field_value = payload.get(key)
            payload[key] = None if field_value in (None, "") else str(field_value)
        metadata = json_safe_value(payload.get("metadata") or {})
        payload["metadata"] = metadata if isinstance(metadata, dict) else {}
        return ArtifactRecord(**_filter_dataclass_payload(ArtifactRecord, payload))
    return ArtifactRecord(
        artifact_id=artifact_id,
        kind="tool_result",
        source="",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        size_bytes=0,
        summary="",
    )


def _coerce_episodic_summary(value: Any) -> Any:
    if isinstance(value, EpisodicSummary):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        payload.setdefault("summary_id", "")
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload["summary_id"] = str(payload.get("summary_id", "") or "")
        payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))
        for key in (
            "decisions",
            "files_touched",
            "failed_approaches",
            "remaining_plan",
            "artifact_ids",
            "notes",
        ):
            payload[key] = _coerce_string_list(payload.get(key))
        return EpisodicSummary(**_filter_dataclass_payload(EpisodicSummary, payload))
    return EpisodicSummary(
        summary_id="",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _coerce_context_brief(value: Any) -> Any:
    if isinstance(value, ContextBrief):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        payload.setdefault("brief_id", "")
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload.setdefault("tier", "warm")
        payload.setdefault("step_range", (0, 0))
        payload["brief_id"] = str(payload.get("brief_id", "") or "")
        payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))

        for key in (
            "key_discoveries",
            "tools_tried",
            "blockers",
            "files_touched",
            "artifact_ids",
            "facts_confirmed",
            "facts_unconfirmed",
            "open_questions",
            "candidate_causes",
            "disproven_causes",
            "next_observations_needed",
            "evidence_refs",
            "claim_refs",
            "new_facts",
            "invalidated_facts",
            "state_changes",
            "decision_deltas",
        ):
            payload[key] = _coerce_string_list(payload.get(key))

        return ContextBrief(**_filter_dataclass_payload(ContextBrief, payload))

    return ContextBrief(
        brief_id="",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        tier="warm",
        step_range=(0, 0),
        task_goal="",
        current_phase="",
        key_discoveries=[],
        tools_tried=[],
        blockers=[],
        files_touched=[],
        artifact_ids=[],
        next_action_hint="",
        staleness_step=0,
        facts_confirmed=[],
        facts_unconfirmed=[],
        open_questions=[],
        candidate_causes=[],
        disproven_causes=[],
        next_observations_needed=[],
        evidence_refs=[],
        claim_refs=[],
        new_facts=[],
        invalidated_facts=[],
        state_changes=[],
        decision_deltas=[],
    )


def _coerce_turn_bundle(value: Any) -> Any:
    if isinstance(value, TurnBundle):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        payload.setdefault("bundle_id", "")
        payload.setdefault("created_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        payload.setdefault("tier", "l1")
        payload.setdefault("step_range", (0, 0))
        payload["bundle_id"] = str(payload.get("bundle_id", "") or "")
        payload["created_at"] = _coerce_timestamp_string(payload.get("created_at"))
        payload["tier"] = str(payload.get("tier", "l1") or "l1")
        step_range = payload.get("step_range", (0, 0))
        if isinstance(step_range, (list, tuple)) and len(step_range) >= 2:
            payload["step_range"] = (_coerce_int(step_range[0], default=0), _coerce_int(step_range[1], default=0))
        else:
            payload["step_range"] = (0, 0)
        payload["phase"] = str(payload.get("phase", "") or "")
        payload["intent"] = str(payload.get("intent", "") or "")
        payload["summary_lines"] = _coerce_string_list(payload.get("summary_lines"))
        payload["files_touched"] = _coerce_string_list(payload.get("files_touched"))
        payload["artifact_ids"] = _coerce_string_list(payload.get("artifact_ids"))
        payload["evidence_refs"] = _coerce_string_list(payload.get("evidence_refs"))
        payload["source_message_count"] = _coerce_int(payload.get("source_message_count"), default=0)
        return TurnBundle(**_filter_dataclass_payload(TurnBundle, payload))

    return TurnBundle(
        bundle_id="",
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _coerce_prompt_budget(value: Any) -> Any:
    if isinstance(value, PromptBudgetSnapshot):
        return value
    if isinstance(value, dict):
        payload = _filter_dataclass_payload(PromptBudgetSnapshot, value)
        payload["estimated_prompt_tokens"] = _coerce_int(
            payload.get("estimated_prompt_tokens"), default=0
        )
        sections = json_safe_value(payload.get("sections") or {})
        if not isinstance(sections, dict):
            sections = {}
        payload["sections"] = {
            str(key): _coerce_int(item, default=0) for key, item in sections.items()
        }
        payload["message_count"] = _coerce_int(payload.get("message_count"), default=0)
        max_prompt_tokens = payload.get("max_prompt_tokens")
        payload["max_prompt_tokens"] = (
            None if max_prompt_tokens in (None, "") else _coerce_int(max_prompt_tokens, default=0)
        )
        payload["reserve_completion_tokens"] = _coerce_int(
            payload.get("reserve_completion_tokens"), default=0
        )
        payload["reserve_tool_tokens"] = _coerce_int(payload.get("reserve_tool_tokens"), default=0)
        payload["compaction_estimated_prompt_tokens_before"] = _coerce_int(
            payload.get("compaction_estimated_prompt_tokens_before"), default=0
        )
        payload["compaction_estimated_prompt_tokens_after"] = _coerce_int(
            payload.get("compaction_estimated_prompt_tokens_after"), default=0
        )
        payload["compaction_threshold"] = _coerce_int(payload.get("compaction_threshold"), default=0)
        payload["compaction_recent_messages_before"] = _coerce_int(
            payload.get("compaction_recent_messages_before"), default=0
        )
        payload["compaction_recent_messages_after"] = _coerce_int(
            payload.get("compaction_recent_messages_after"), default=0
        )
        payload["compaction_keep_recent_initial"] = _coerce_int(
            payload.get("compaction_keep_recent_initial"), default=0
        )
        payload["compaction_keep_recent_final"] = _coerce_int(
            payload.get("compaction_keep_recent_final"), default=0
        )
        payload["compaction_messages_compacted"] = _coerce_int(
            payload.get("compaction_messages_compacted"), default=0
        )
        payload["compaction_attempt_count"] = _coerce_int(
            payload.get("compaction_attempt_count"), default=0
        )
        payload["compaction_stopped_reason"] = str(
            payload.get("compaction_stopped_reason", "") or ""
        )
        payload["included_compaction_levels"] = _coerce_string_list(payload.get("included_compaction_levels"))
        payload["dropped_compaction_levels"] = _coerce_string_list(payload.get("dropped_compaction_levels"))
        return PromptBudgetSnapshot(**payload)
    return PromptBudgetSnapshot()


def _coerce_write_session(value: Any) -> Any:
    if isinstance(value, WriteSession):
        return value
    if not isinstance(value, dict):
        return None
    payload = _filter_dataclass_payload(WriteSession, value)
    if "session_id" in value and "write_session_id" not in payload:
        payload["write_session_id"] = str(value["session_id"])
    if "target_path" in value and "write_target_path" not in payload:
        payload["write_target_path"] = str(value["target_path"])
    if "completed_sections" in value and "write_sections_completed" not in payload:
        payload["write_sections_completed"] = _coerce_string_list(value["completed_sections"])
    if "current_section" in value and "write_current_section" not in payload:
        payload["write_current_section"] = str(value["current_section"])
    if "next_section" in value and "write_next_section" not in payload:
        payload["write_next_section"] = str(value["next_section"])
    if "verdict" in value and "write_last_verifier" not in payload:
        payload["write_last_verifier"] = _coerce_json_dict_payload(value["verdict"])
    payload["write_session_id"] = str(payload.get("write_session_id", "") or "")
    payload["write_target_path"] = str(payload.get("write_target_path", "") or "")
    intent = str(payload.get("write_session_intent", "replace_file") or "replace_file").strip().lower()
    payload["write_session_intent"] = intent if intent in {"replace_file", "patch_existing"} else "replace_file"
    mode = str(payload.get("write_session_mode", "chunked_author") or "chunked_author").strip().lower()
    if mode not in {"single_write", "chunked_author", "local_repair", "stub_and_fill"}:
        mode = "chunked_author"
    payload["write_session_mode"] = mode
    payload["write_session_started_at"] = _coerce_float(payload.get("write_session_started_at"), default=0.0)
    payload["write_first_chunk_at"] = _coerce_float(payload.get("write_first_chunk_at"), default=0.0)
    payload["write_staging_path"] = str(payload.get("write_staging_path", "") or "")
    payload["write_original_snapshot_path"] = str(payload.get("write_original_snapshot_path", "") or "")
    existed_at_start = payload.get("write_target_existed_at_start")
    payload["write_target_existed_at_start"] = (
        bool(existed_at_start)
        if isinstance(existed_at_start, bool)
        else str(existed_at_start or "").strip().lower() in {"1", "true", "yes", "on"}
    )
    payload["write_section_ranges"] = _coerce_write_section_ranges(payload.get("write_section_ranges"))
    payload["write_last_attempt_snapshot_path"] = str(payload.get("write_last_attempt_snapshot_path", "") or "")
    payload["write_last_attempt_sections"] = _coerce_string_list(payload.get("write_last_attempt_sections"))
    payload["write_last_attempt_ranges"] = _coerce_write_section_ranges(payload.get("write_last_attempt_ranges"))
    payload["write_last_staged_hash"] = str(payload.get("write_last_staged_hash", "") or "")
    payload["write_sections_completed"] = _coerce_string_list(payload.get("write_sections_completed"))
    payload["write_current_section"] = str(payload.get("write_current_section", "") or "")
    payload["write_next_section"] = str(payload.get("write_next_section", "") or "")
    payload["write_failed_local_patches"] = _coerce_int(payload.get("write_failed_local_patches"), default=0)
    payload["write_empty_payload_retries"] = _coerce_int(payload.get("write_empty_payload_retries"), default=0)
    payload["write_salvage_count"] = _coerce_int(payload.get("write_salvage_count"), default=0)
    payload["write_last_verifier"] = _coerce_json_dict_payload(payload.get("write_last_verifier"))
    payload["write_session_fallback_mode"] = str(payload.get("write_session_fallback_mode", "stub_and_fill") or "stub_and_fill")
    pending_finalize = payload.get("write_pending_finalize")
    payload["write_pending_finalize"] = bool(pending_finalize) if isinstance(pending_finalize, bool) else str(pending_finalize or "").strip().lower() in {"1", "true", "yes", "on"}
    payload["suggested_sections"] = _coerce_string_list(payload.get("suggested_sections"))
    status = str(payload.get("status", "open") or "open").strip().lower()
    if status not in {"open", "local_repair", "fallback", "complete"}:
        status = "open"
    payload["status"] = status
    return WriteSession(**payload)


def _coerce_background_process_record(value: Any, *, job_id: str) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, dict) else {}
    pid = payload.get("pid")
    try:
        normalized_pid = int(pid)
    except (TypeError, ValueError):
        normalized_pid = 0
    command = str(payload.get("command", ""))
    cwd = str(payload.get("cwd", ""))
    started_at = payload.get("started_at")
    if not isinstance(started_at, str) or not started_at.strip():
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status = str(payload.get("status", "running" if normalized_pid else "unknown"))
    normalized = json_safe_value(payload)
    if not isinstance(normalized, dict):
        normalized = {}
    normalized["job_id"] = job_id
    normalized["pid"] = normalized_pid
    normalized["command"] = command
    normalized["cwd"] = cwd
    normalized["started_at"] = str(started_at)
    normalized["status"] = status
    return normalized


def _coerce_tool_envelope_payload(value: Any) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, dict) else {}
    normalized = json_safe_value(payload)
    if not isinstance(normalized, dict):
        normalized = {}
    metadata = normalized.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    normalized["success"] = bool(payload.get("success"))
    normalized["output"] = json_safe_value(payload.get("output"))
    error = payload.get("error")
    normalized["error"] = None if error is None else str(error)
    normalized["metadata"] = metadata
    return normalized


def _coerce_tool_execution_record(value: Any, *, operation_id: str) -> dict[str, Any]:
    payload = dict(value) if isinstance(value, dict) else {}
    normalized = json_safe_value(payload)
    if not isinstance(normalized, dict):
        normalized = {}
    normalized["operation_id"] = str(payload.get("operation_id", operation_id) or operation_id)
    if "thread_id" in payload:
        normalized["thread_id"] = str(payload.get("thread_id", ""))
    if "step_count" in payload:
        step_count = payload.get("step_count", 0)
        try:
            normalized["step_count"] = int(step_count)
        except (TypeError, ValueError):
            normalized["step_count"] = 0
    if "tool_name" in payload:
        normalized["tool_name"] = str(payload.get("tool_name", ""))
    if "tool_call_id" in payload:
        tool_call_id = payload.get("tool_call_id")
        normalized["tool_call_id"] = None if tool_call_id is None else str(tool_call_id)
    if "args" in payload:
        args = json_safe_value(payload.get("args") or {})
        normalized["args"] = args if isinstance(args, dict) else {}
    if "result" in payload:
        normalized["result"] = _coerce_tool_envelope_payload(payload.get("result"))
    tool_message = _coerce_conversation_message_payload(payload.get("tool_message"))
    if tool_message is not None:
        normalized["tool_message"] = tool_message
    else:
        normalized.pop("tool_message", None)
    artifact_id = payload.get("artifact_id")
    if artifact_id is None:
        normalized.pop("artifact_id", None)
    else:
        normalized["artifact_id"] = str(artifact_id)
    return normalized


def _coerce_pending_interrupt_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    normalized = json_safe_value(value)
    if not isinstance(normalized, dict):
        return None
    kind = value.get("kind")
    normalized["kind"] = str(kind) if kind is not None else "ask_human"
    for key in ("question", "current_phase", "thread_id", "operation_id", "plan_id", "response_mode"):
        if key in value:
            normalized[key] = str(value.get(key, ""))
    if "approved" in value:
        normalized["approved"] = bool(value.get("approved"))
    active_profiles = normalized.get("active_profiles")
    if active_profiles is not None:
        if isinstance(active_profiles, list):
            normalized["active_profiles"] = [str(item) for item in active_profiles]
        else:
            normalized["active_profiles"] = []
    recent_tool_outcomes = normalized.get("recent_tool_outcomes")
    if recent_tool_outcomes is not None:
        if isinstance(recent_tool_outcomes, list):
            normalized["recent_tool_outcomes"] = [
                json_safe_value(item) for item in recent_tool_outcomes if isinstance(item, dict)
            ]
        else:
            normalized["recent_tool_outcomes"] = []
    return normalized
