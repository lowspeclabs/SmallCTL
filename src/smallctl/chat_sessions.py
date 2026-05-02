from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SESSION_INDEX_NAME = "chat_sessions.json"
SESSION_STATE_DIR_NAME = "chat_states"


@dataclass(frozen=True)
class ChatSessionSummary:
    thread_id: str
    first_user_message: str
    created_at: str
    updated_at: str
    model: str = ""


def session_index_path(cwd: str | Path) -> Path:
    return Path(cwd).resolve() / ".smallctl" / SESSION_INDEX_NAME


def session_state_path(cwd: str | Path, thread_id: str) -> Path:
    safe_thread_id = _sanitize_filename(thread_id)
    return Path(cwd).resolve() / ".smallctl" / SESSION_STATE_DIR_NAME / f"{safe_thread_id}.json"


def record_chat_session_prompt(
    *,
    cwd: str | Path,
    thread_id: str,
    message: str,
    model: str = "",
    created_at: str = "",
) -> None:
    resolved_thread_id = str(thread_id or "").strip()
    first_message = _clip_single_line(message, limit=180)
    if not resolved_thread_id or not first_message:
        return

    path = session_index_path(cwd)
    records = _read_index(path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    existing = next(
        (record for record in records if str(record.get("thread_id") or "") == resolved_thread_id),
        None,
    )
    if existing is None:
        records.append(
            {
                "thread_id": resolved_thread_id,
                "first_user_message": first_message,
                "created_at": created_at or now,
                "updated_at": now,
                "model": str(model or "").strip(),
            }
        )
    else:
        existing.setdefault("first_user_message", first_message)
        if not str(existing.get("first_user_message") or "").strip():
            existing["first_user_message"] = first_message
        existing.setdefault("created_at", created_at or now)
        existing["updated_at"] = now
        if model:
            existing["model"] = str(model).strip()
    _write_index(path, records)


def persist_chat_session_state(
    *,
    cwd: str | Path,
    thread_id: str,
    state_payload: dict[str, Any],
    model: str = "",
) -> Path | None:
    resolved_thread_id = str(thread_id or "").strip()
    if not resolved_thread_id or not isinstance(state_payload, dict):
        return None
    state_path = session_state_path(cwd, resolved_thread_id)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "thread_id": resolved_thread_id,
        "saved_at": now,
        "state": state_payload,
    }
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    first_message = _first_user_message(state_payload.get("conversation_history"))
    if not first_message:
        first_message = _first_user_message(state_payload.get("recent_messages"))
    if not first_message:
        run_brief = state_payload.get("run_brief")
        if isinstance(run_brief, dict):
            first_message = str(run_brief.get("original_task") or "").strip()
    record_chat_session_prompt(
        cwd=cwd,
        thread_id=resolved_thread_id,
        message=first_message,
        model=model or _state_model_name(state_payload),
        created_at=str(state_payload.get("created_at") or now),
    )
    return state_path


def load_chat_session_state(
    *,
    cwd: str | Path,
    thread_id: str,
) -> dict[str, Any] | None:
    path = session_state_path(cwd, thread_id)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    state = payload.get("state")
    return state if isinstance(state, dict) else None


def load_chat_session_summaries(
    *,
    cwd: str | Path,
    checkpointer: Any = None,
    limit: int = 30,
) -> list[ChatSessionSummary]:
    summaries_by_thread: dict[str, ChatSessionSummary] = {}
    for summary in _load_index_summaries(session_index_path(cwd)):
        summaries_by_thread[summary.thread_id] = summary

    for summary in _load_checkpoint_summaries(checkpointer):
        existing = summaries_by_thread.get(summary.thread_id)
        if existing is None:
            summaries_by_thread[summary.thread_id] = summary
            continue
        summaries_by_thread[summary.thread_id] = ChatSessionSummary(
            thread_id=existing.thread_id,
            first_user_message=existing.first_user_message or summary.first_user_message,
            created_at=existing.created_at or summary.created_at,
            updated_at=_max_timestamp(existing.updated_at, summary.updated_at),
            model=existing.model or summary.model,
        )

    summaries = sorted(
        summaries_by_thread.values(),
        key=lambda item: _timestamp_sort_key(item.updated_at or item.created_at),
        reverse=True,
    )
    return summaries[: max(1, int(limit))]


def format_relative_age(timestamp: str, *, now: datetime | None = None) -> str:
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return "unknown"
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)
    seconds = max(0, int((reference - parsed).total_seconds()))
    minutes = seconds // 60
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _load_index_summaries(path: Path) -> list[ChatSessionSummary]:
    summaries: list[ChatSessionSummary] = []
    for record in _read_index(path):
        summary = _summary_from_record(record)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _load_checkpoint_summaries(checkpointer: Any) -> list[ChatSessionSummary]:
    if checkpointer is None or not hasattr(checkpointer, "list"):
        return []
    summaries: list[ChatSessionSummary] = []
    seen: set[str] = set()
    try:
        iterator = checkpointer.list(None, limit=None)
    except TypeError:
        iterator = checkpointer.list(None)
    except Exception:
        return []
    try:
        for item in iterator:
            thread_id = _thread_id_from_checkpoint_tuple(item)
            if not thread_id or thread_id in seen:
                continue
            summary = _summary_from_checkpoint_tuple(item, thread_id=thread_id)
            if summary is None:
                continue
            summaries.append(summary)
            seen.add(thread_id)
    except Exception:
        return summaries
    return summaries


def _summary_from_checkpoint_tuple(item: Any, *, thread_id: str) -> ChatSessionSummary | None:
    checkpoint = getattr(item, "checkpoint", None)
    if not isinstance(checkpoint, dict):
        return None
    channel_values = checkpoint.get("channel_values")
    if not isinstance(channel_values, dict):
        return None
    loop_state = channel_values.get("loop_state")
    if not isinstance(loop_state, dict):
        return None
    first_message = _first_user_message(loop_state.get("conversation_history"))
    if not first_message:
        first_message = _first_user_message(loop_state.get("recent_messages"))
    if not first_message:
        run_brief = loop_state.get("run_brief")
        if isinstance(run_brief, dict):
            first_message = str(run_brief.get("original_task") or "").strip()
    first_message = _clip_single_line(first_message, limit=180)
    if not first_message:
        return None
    return ChatSessionSummary(
        thread_id=thread_id,
        first_user_message=first_message,
        created_at=str(loop_state.get("created_at") or ""),
        updated_at=str(loop_state.get("updated_at") or loop_state.get("created_at") or ""),
        model=_state_model_name(loop_state),
    )


def _thread_id_from_checkpoint_tuple(item: Any) -> str:
    config = getattr(item, "config", None)
    if not isinstance(config, dict):
        return ""
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return ""
    return str(configurable.get("thread_id") or "").strip()


def _summary_from_record(record: dict[str, Any]) -> ChatSessionSummary | None:
    thread_id = str(record.get("thread_id") or "").strip()
    first_message = _clip_single_line(record.get("first_user_message"), limit=180)
    if not thread_id or not first_message:
        return None
    created_at = str(record.get("created_at") or "").strip()
    updated_at = str(record.get("updated_at") or created_at).strip()
    return ChatSessionSummary(
        thread_id=thread_id,
        first_user_message=first_message,
        created_at=created_at,
        updated_at=updated_at,
        model=str(record.get("model") or "").strip(),
    )


def _read_index(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    records = payload.get("sessions") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, dict)]


def _write_index(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sessions": records}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _first_user_message(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        text = str(message.get("content") or "").strip()
        if text:
            return text
    return ""


def _clip_single_line(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "~"


def _parse_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp_sort_key(value: str) -> float:
    parsed = _parse_timestamp(value)
    return parsed.timestamp() if parsed else 0.0


def _max_timestamp(left: str, right: str) -> str:
    return left if _timestamp_sort_key(left) >= _timestamp_sort_key(right) else right


def _state_model_name(state_payload: dict[str, Any]) -> str:
    scratchpad = state_payload.get("scratchpad")
    if not isinstance(scratchpad, dict):
        return ""
    return str(scratchpad.get("_model_name") or "").strip()


def _sanitize_filename(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "session"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return sanitized.strip("._") or "session"
