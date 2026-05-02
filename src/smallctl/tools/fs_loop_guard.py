from __future__ import annotations

import difflib
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..state import LoopState
from .common import fail

_LOOP_GUARD_STATE_KEY = "_chunk_write_loop_guard"
_LOOP_GUARD_CONFIG_KEY = "_chunk_write_loop_guard_config"
_DEFAULT_LOOP_GUARD_CONFIG: dict[str, Any] = {
    "enabled": True,
    "stagnation_threshold": 3,
    "level2_threshold": 5,
    "recent_writes_limit": 5,
    "tail_lines": 50,
    "similarity_threshold": 0.9,
    "append_overlap_threshold": 0.75,
    "cumulative_write_gate": True,
    "checkpoint_gate": True,
    "diff_gate": True,
}


def _resolve(path: str, cwd: str | None = None) -> str:
    base = Path(cwd) if cwd else Path.cwd()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        return str(candidate.resolve())
    except Exception:
        return str(candidate)


def _content_hash(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8", errors="replace")).hexdigest()


def _loop_guard_root(state: LoopState | None) -> dict[str, Any] | None:
    if state is None:
        return None
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    root = scratchpad.setdefault(
        _LOOP_GUARD_STATE_KEY,
        {
            "version": 1,
            "paths": {},
            "events": [],
        },
    )
    if not isinstance(root, dict):
        return None
    root.setdefault("version", 1)
    root.setdefault("paths", {})
    root.setdefault("events", [])
    if not isinstance(root["paths"], dict):
        root["paths"] = {}
    if not isinstance(root["events"], list):
        root["events"] = []
    return root


def _path_state(state: LoopState | None, path: str) -> dict[str, Any] | None:
    root = _loop_guard_root(state)
    if root is None:
        return None
    paths = root.setdefault("paths", {})
    if not isinstance(paths, dict):
        return None
    entry = paths.setdefault(
        path,
        {
            "recent_writes": [],
            "section_checkpoints": [],
            "escalation_level": 0,
            "pending_read_before_write": False,
            "blocked_attempts": 0,
            "writes_since_last_read": 0,
            "last_read_at": 0.0,
            "last_score": 0,
            "last_section_name": "",
            "last_next_section_name": "",
            "last_error_kind": "",
            "session_id": "",
            "outline_required": False,
            "last_projected_content": "",
        },
    )
    if not isinstance(entry, dict):
        return None
    entry.setdefault("recent_writes", [])
    entry.setdefault("section_checkpoints", [])
    entry.setdefault("escalation_level", 0)
    entry.setdefault("pending_read_before_write", False)
    entry.setdefault("blocked_attempts", 0)
    entry.setdefault("writes_since_last_read", 0)
    entry.setdefault("last_read_at", 0.0)
    entry.setdefault("last_score", 0)
    entry.setdefault("last_section_name", "")
    entry.setdefault("last_next_section_name", "")
    entry.setdefault("last_error_kind", "")
    entry.setdefault("session_id", "")
    entry.setdefault("outline_required", False)
    entry.setdefault("last_projected_content", "")
    if not isinstance(entry["recent_writes"], list):
        entry["recent_writes"] = []
    if not isinstance(entry["section_checkpoints"], list):
        entry["section_checkpoints"] = []
    return entry


def loop_guard_config(state: LoopState | None) -> dict[str, Any]:
    config = dict(_DEFAULT_LOOP_GUARD_CONFIG)
    if state is None:
        return config
    scratchpad = getattr(state, "scratchpad", None)
    raw = scratchpad.get(_LOOP_GUARD_CONFIG_KEY, {}) if isinstance(scratchpad, dict) else {}
    if not isinstance(raw, dict):
        return config

    def _as_bool(value: Any, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _as_int(value: Any, default: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed > 0 else default

    def _as_float(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if 0.0 < parsed <= 1.0 else default

    config["enabled"] = _as_bool(raw.get("enabled"), bool(config["enabled"]))
    config["stagnation_threshold"] = _as_int(raw.get("stagnation_threshold"), int(config["stagnation_threshold"]))
    config["level2_threshold"] = _as_int(raw.get("level2_threshold"), int(config["level2_threshold"]))
    config["recent_writes_limit"] = _as_int(raw.get("recent_writes_limit"), int(config["recent_writes_limit"]))
    config["tail_lines"] = _as_int(raw.get("tail_lines"), int(config["tail_lines"]))
    config["similarity_threshold"] = _as_float(
        raw.get("similarity_threshold"),
        float(config["similarity_threshold"]),
    )
    config["append_overlap_threshold"] = _as_float(
        raw.get("append_overlap_threshold"),
        float(config["append_overlap_threshold"]),
    )
    config["cumulative_write_gate"] = _as_bool(
        raw.get("cumulative_write_gate"),
        bool(config["cumulative_write_gate"]),
    )
    config["checkpoint_gate"] = _as_bool(raw.get("checkpoint_gate"), bool(config["checkpoint_gate"]))
    config["diff_gate"] = _as_bool(raw.get("diff_gate"), bool(config["diff_gate"]))
    if config["level2_threshold"] < config["stagnation_threshold"]:
        config["level2_threshold"] = config["stagnation_threshold"]
    return config


def _count_added_lines(before: str, after: str) -> int:
    added = 0
    for line in difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm=""):
        if line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
    return added


def _tail_excerpt(text: str, *, tail_lines: int) -> str:
    lines = str(text or "").splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max(1, int(tail_lines)) :])


def _record_guard_event(
    state: LoopState | None,
    *,
    event: str,
    path: str,
    session_id: str,
    payload: dict[str, Any],
) -> None:
    root = _loop_guard_root(state)
    if root is None:
        return
    events = root.setdefault("events", [])
    if not isinstance(events, list):
        return
    entry = {
        "event": str(event or "").strip() or "unknown",
        "at": time.time(),
        "path": path,
        "session_id": session_id,
        **payload,
    }
    events.append(entry)
    if len(events) > 40:
        del events[: len(events) - 40]
    logger = getattr(state, "log", None)
    if logger is not None:
        try:
            logger.info("loop_guard %s", json.dumps(entry, sort_keys=True, ensure_ascii=True))
        except Exception:
            pass


def _next_required_read(path: str) -> dict[str, Any]:
    return {
        "tool_name": "file_read",
        "required_fields": ["path"],
        "required_arguments": {"path": path},
        "optional_fields": ["start_line", "end_line"],
        "notes": [
            "Read the current staged content before attempting another chunk write.",
            "Confirm the exact missing section or lines from the read result before writing again.",
        ],
    }


def _clear_loop_guard_read_schedule(state: LoopState | None) -> None:
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return
    scratchpad.pop("_chunk_write_loop_guard_read_scheduled", None)


def _outline_handoff_question(path: str) -> str:
    return (
        f"LoopGuard outline required for `{path}`.\n"
        "- Bullet 1: next missing section\n"
        "- Bullet 2: section after that\n"
        "- Bullet 3: final section or verification step\n"
        "Reply `continue` to resume writing, or provide corrections first."
    )


def _append_recent_write(path_state: dict[str, Any], record: dict[str, Any], *, limit: int) -> None:
    recent = path_state.setdefault("recent_writes", [])
    if not isinstance(recent, list):
        recent = []
        path_state["recent_writes"] = recent
    recent.append(record)
    if len(recent) > limit:
        del recent[: len(recent) - limit]


def _loop_guard_attempt_count(path_state: dict[str, Any]) -> int:
    recent = path_state.get("recent_writes", [])
    recent_count = len(recent) if isinstance(recent, list) else 0
    blocked_attempts = int(path_state.get("blocked_attempts", 0) or 0)
    return max(1, recent_count, blocked_attempts + 1)


def _hard_abort_chunked_write(
    *,
    state: LoopState | None,
    path_state: dict[str, Any],
    resolved_path: str,
    session_id: str,
    section_name: str,
    next_section_name: str,
    score: int,
    signals: dict[str, Any],
    tail_excerpt: str,
    trigger_kind: str,
) -> dict[str, Any]:
    attempts = _loop_guard_attempt_count(path_state)
    normalized_section = str(section_name or "").strip() or "unknown"
    postmortem = (
        f"Model stuck in write loop on section '{normalized_section}' "
        f"for path '{resolved_path}' after {attempts} attempts."
    )
    path_state["pending_read_before_write"] = False
    path_state["outline_required"] = False
    path_state["escalation_level"] = 4
    path_state["blocked_attempts"] = attempts
    path_state["last_error_kind"] = "chunked_write_loop_guard_hard_abort"
    _record_guard_event(
        state,
        event="loop_guard_hard_abort",
        path=resolved_path,
        session_id=session_id,
        payload={
            "trigger_kind": trigger_kind,
            "stagnation_score": score,
            "section_name": section_name,
            "next_section_name": next_section_name,
            "attempts": attempts,
            "signals": signals,
        },
    )
    return fail(
        postmortem,
        metadata={
            "path": resolved_path,
            "write_session_id": session_id,
            "section_name": section_name,
            "next_section_name": next_section_name,
            "error_kind": "chunked_write_loop_guard_hard_abort",
            "loop_guard_score": score,
            "loop_guard_escalation_level": 4,
            "loop_guard_signals": signals,
            "loop_guard_tail_excerpt": tail_excerpt,
            "loop_guard_hard_abort": True,
            "loop_guard_postmortem": postmortem,
            "loop_guard_trigger_kind": trigger_kind,
            "loop_guard_attempts": attempts,
        },
    )


def maybe_block_chunked_write(
    *,
    state: LoopState | None,
    session: Any,
    path: str,
    cwd: str | None,
    content: str,
    section_name: str,
    next_section_name: str,
    replace_strategy: str,
    staged_content: str,
    updated_content: str,
    append_overlap_ratio: float = 0.0,
) -> dict[str, Any] | None:
    if state is None or session is None:
        return None

    config = loop_guard_config(state)
    if not bool(config.get("enabled", True)):
        return None

    session_mode = str(getattr(session, "write_session_mode", "") or "").strip().lower()
    session_status = str(getattr(session, "status", "") or "").strip().lower()
    if session_mode != "chunked_author" or session_status == "complete":
        return None

    resolved_path = _resolve(path, cwd)
    path_state = _path_state(state, resolved_path)
    if path_state is None:
        return None

    recent_writes = list(path_state.get("recent_writes", []) or [])
    session_id = str(getattr(session, "write_session_id", "") or "").strip()
    if str(path_state.get("session_id", "") or "").strip() != session_id:
        path_state["recent_writes"] = []
        path_state["section_checkpoints"] = []
        path_state["escalation_level"] = 0
        path_state["pending_read_before_write"] = False
        path_state["blocked_attempts"] = 0
        path_state["writes_since_last_read"] = 0
        path_state["last_read_at"] = 0.0
        path_state["last_score"] = 0
        path_state["last_section_name"] = ""
        path_state["last_next_section_name"] = ""
        path_state["last_error_kind"] = ""
        path_state["session_id"] = session_id
        path_state["outline_required"] = False
        path_state["last_projected_content"] = ""
        recent_writes = []
    previous = recent_writes[-1] if recent_writes else {}
    writes_since_last_read = int(path_state.get("writes_since_last_read", 0) or 0) + 1
    projected_hash = _content_hash(updated_content)
    payload_hash = _content_hash(content)
    projected_length = len(updated_content)
    payload_length = len(content)
    previous_projected_length = int(previous.get("projected_length", 0) or 0)
    previous_projected_hash = str(previous.get("projected_hash", "") or "")
    previous_section_name = str(previous.get("section_name", "") or "")
    previous_next_section_name = str(previous.get("next_section_name", "") or "")
    previous_projected_content = str(path_state.get("last_projected_content", "") or "")
    size_delta = (
        abs(projected_length - previous_projected_length) / max(previous_projected_length, 1)
        if previous_projected_length > 0
        else 1.0
    )

    recent_hashes = [
        str(item.get("projected_hash", "") or "")
        for item in recent_writes[-max(0, int(config["recent_writes_limit"]) - 1) :]
    ]
    candidate_hashes = [item for item in recent_hashes if item]
    candidate_hashes.append(projected_hash)

    hash_stagnation = len(candidate_hashes) >= 2 and len(set(candidate_hashes)) == 1
    similarity_ratio = 0.0
    if previous_projected_content and not hash_stagnation:
        similarity_ratio = difflib.SequenceMatcher(
            None,
            previous_projected_content,
            updated_content,
            autojunk=False,
        ).ratio()
    similarity_stagnation = bool(
        previous
        and not hash_stagnation
        and similarity_ratio >= float(config["similarity_threshold"])
    )
    section_stagnation = bool(
        section_name
        and previous
        and section_name == previous_section_name
        and next_section_name == previous_next_section_name
    )
    size_delta_signal = bool(previous and size_delta < 0.01)
    clustering_signal = writes_since_last_read >= 3
    added_lines = _count_added_lines(staged_content, updated_content)

    append_large_overlap = bool(
        append_overlap_ratio >= float(config.get("append_overlap_threshold", 0.75))
    )

    score = 0
    if hash_stagnation:
        score += 2
    if similarity_stagnation:
        score += 1
    if section_stagnation:
        score += 2
    if size_delta_signal:
        score += 1
    if clustering_signal:
        score += 1
    if append_large_overlap:
        score += 3

    signals = {
        "hash_stagnation": hash_stagnation,
        "similarity_stagnation": similarity_stagnation,
        "similarity_ratio": round(similarity_ratio, 6) if previous else None,
        "section_stagnation": section_stagnation,
        "size_delta_ratio": round(size_delta, 6) if previous else None,
        "tool_call_clustering": writes_since_last_read if clustering_signal else 0,
        "added_lines": added_lines,
        "previous_projected_hash": previous_projected_hash,
        "append_large_overlap": append_large_overlap,
    }
    record = {
        "timestamp": time.time(),
        "path": resolved_path,
        "payload_hash": payload_hash,
        "projected_hash": projected_hash,
        "payload_length": payload_length,
        "projected_length": projected_length,
        "section_name": section_name,
        "next_section_name": next_section_name,
        "replace_strategy": replace_strategy,
        "similarity_to_previous": round(similarity_ratio, 6) if previous else None,
        "score": score,
        "signals": signals,
    }
    _append_recent_write(
        path_state,
        record,
        limit=int(config["recent_writes_limit"]),
    )
    path_state["last_projected_content"] = updated_content
    path_state["writes_since_last_read"] = writes_since_last_read
    path_state["last_score"] = score
    path_state["last_section_name"] = section_name
    path_state["last_next_section_name"] = next_section_name

    tail_excerpt = _tail_excerpt(staged_content, tail_lines=int(config["tail_lines"]))
    checkpoints = set(str(item) for item in path_state.get("section_checkpoints", []) if str(item).strip())
    prior_level = int(path_state.get("escalation_level", 0) or 0)

    if bool(path_state.get("pending_read_before_write")):
        if prior_level >= 2:
            return _hard_abort_chunked_write(
                state=state,
                path_state=path_state,
                resolved_path=resolved_path,
                session_id=session_id,
                section_name=section_name,
                next_section_name=next_section_name,
                score=score,
                signals=signals,
                tail_excerpt=tail_excerpt,
                trigger_kind="read_required_retry",
            )
        level = 2 if prior_level >= 1 else 1
        path_state["escalation_level"] = max(prior_level, level)
        path_state["blocked_attempts"] = int(path_state.get("blocked_attempts", 0) or 0) + 1
        path_state["last_error_kind"] = "chunked_write_requires_read_before_write"
        path_state["outline_required"] = level >= 2
        _record_guard_event(
            state,
            event="loop_guard_read_required",
            path=resolved_path,
            session_id=session_id,
            payload={
                "escalation_level": level,
                "stagnation_score": score,
                "section_name": section_name,
                "next_section_name": next_section_name,
            },
        )
        return fail(
            "LoopGuard: repeated write detected. You must read the current file content and confirm "
            "which section is missing before writing again.",
            metadata={
                "path": resolved_path,
                "write_session_id": session_id,
                "section_name": section_name,
                "next_section_name": next_section_name,
                "error_kind": "chunked_write_requires_read_before_write",
                "loop_guard_score": score,
                "loop_guard_escalation_level": level,
                "loop_guard_signals": signals,
                "loop_guard_schedule_read": True,
                "loop_guard_tail_excerpt": tail_excerpt,
                "next_required_tool": _next_required_read(path),
                "loop_guard_outline_required": level >= 2,
                "loop_guard_outline_question": _outline_handoff_question(resolved_path) if level >= 2 else "",
            },
        )

    if (
        bool(config.get("checkpoint_gate", True))
        and section_name
        and section_name in checkpoints
    ):
        if prior_level >= 2:
            return _hard_abort_chunked_write(
                state=state,
                path_state=path_state,
                resolved_path=resolved_path,
                session_id=session_id,
                section_name=section_name,
                next_section_name=next_section_name,
                score=score,
                signals=signals,
                tail_excerpt=tail_excerpt,
                trigger_kind="checkpoint_revisit",
            )
        path_state["pending_read_before_write"] = True
        path_state["escalation_level"] = max(prior_level, 1)
        path_state["blocked_attempts"] = int(path_state.get("blocked_attempts", 0) or 0) + 1
        path_state["last_error_kind"] = "chunked_write_checkpoint_revisit"
        path_state["outline_required"] = False
        _record_guard_event(
            state,
            event="loop_guard_checkpoint_revisit",
            path=resolved_path,
            session_id=session_id,
            payload={"section_name": section_name, "next_section_name": next_section_name},
        )
        return fail(
            f"LoopGuard: section `{section_name}` was already checkpointed for `{path}`. "
            "Read the current staged content and advance to the next section instead of rewriting it.",
            metadata={
                "path": resolved_path,
                "write_session_id": session_id,
                "section_name": section_name,
                "next_section_name": next_section_name,
                "error_kind": "chunked_write_checkpoint_revisit",
                "loop_guard_score": score,
                "loop_guard_escalation_level": max(prior_level, 1),
                "loop_guard_signals": signals,
                "loop_guard_schedule_read": True,
                "loop_guard_tail_excerpt": tail_excerpt,
                "next_required_tool": _next_required_read(path),
            },
        )

    if (
        bool(config.get("cumulative_write_gate", True))
        and str(getattr(session, "write_session_intent", "") or "").strip().lower() == "replace_file"
        and replace_strategy == "overwrite"
        and getattr(session, "write_sections_completed", None)
        and len(content) <= int(len(staged_content) * 1.01)
    ):
        if prior_level >= 2:
            return _hard_abort_chunked_write(
                state=state,
                path_state=path_state,
                resolved_path=resolved_path,
                session_id=session_id,
                section_name=section_name,
                next_section_name=next_section_name,
                score=score,
                signals=signals,
                tail_excerpt=tail_excerpt,
                trigger_kind="non_growing_overwrite",
            )
        path_state["pending_read_before_write"] = True
        path_state["escalation_level"] = max(prior_level, 1)
        path_state["blocked_attempts"] = int(path_state.get("blocked_attempts", 0) or 0) + 1
        path_state["last_error_kind"] = "chunked_write_non_growing_overwrite"
        path_state["outline_required"] = False
        _record_guard_event(
            state,
            event="loop_guard_non_growing_overwrite",
            path=resolved_path,
            session_id=session_id,
            payload={"section_name": section_name, "content_length": len(content), "staged_length": len(staged_content)},
        )
        return fail(
            "LoopGuard: chunked overwrite must grow the staged file or explicitly replace a missing section. "
            "Read the current staged content before overwriting from memory.",
            metadata={
                "path": resolved_path,
                "write_session_id": session_id,
                "section_name": section_name,
                "next_section_name": next_section_name,
                "error_kind": "chunked_write_non_growing_overwrite",
                "loop_guard_score": score,
                "loop_guard_escalation_level": max(prior_level, 1),
                "loop_guard_signals": signals,
                "loop_guard_schedule_read": True,
                "loop_guard_tail_excerpt": tail_excerpt,
                "next_required_tool": _next_required_read(path),
            },
        )

    if append_large_overlap:
        if prior_level >= 2:
            return _hard_abort_chunked_write(
                state=state,
                path_state=path_state,
                resolved_path=resolved_path,
                session_id=session_id,
                section_name=section_name,
                next_section_name=next_section_name,
                score=score,
                signals=signals,
                tail_excerpt=tail_excerpt,
                trigger_kind="append_large_overlap",
            )
        path_state["pending_read_before_write"] = True
        path_state["escalation_level"] = max(prior_level, 1)
        path_state["blocked_attempts"] = int(path_state.get("blocked_attempts", 0) or 0) + 1
        path_state["last_error_kind"] = "chunked_write_append_overlap_detected"
        path_state["outline_required"] = False
        _record_guard_event(
            state,
            event="loop_guard_append_overlap_detected",
            path=resolved_path,
            session_id=session_id,
            payload={"section_name": section_name, "append_overlap_ratio": append_overlap_ratio},
        )
        return fail(
            "LoopGuard: overlapping chunk append detected. "
            "Call `file_read` to check the current staged content, then use `file_patch` (narrow exact edit) or `ast_patch` (narrow structural edit) "
            "or `file_write` with `replace_strategy=\"overwrite\"` to modify. Do not append existing overlapping lines.",
            metadata={
                "overlap_ratio": append_overlap_ratio,
                "staged_bytes": len(staged_content.encode("utf-8", errors="replace")),
                "incoming_bytes": len(content.encode("utf-8", errors="replace")),
                "path": resolved_path,
                "write_session_id": session_id,
                "section_name": section_name,
                "next_section_name": next_section_name,
                "error_kind": "chunked_write_append_overlap_detected",
                "loop_guard_score": score,
                "loop_guard_escalation_level": max(prior_level, 1),
                "loop_guard_signals": signals,
                "loop_guard_schedule_read": True,
                "loop_guard_tail_excerpt": tail_excerpt,
                "next_required_tool": _next_required_read(path),
            },
        )

    if bool(config.get("diff_gate", True)) and (updated_content == staged_content or added_lines == 0):
        if prior_level >= 2:
            return _hard_abort_chunked_write(
                state=state,
                path_state=path_state,
                resolved_path=resolved_path,
                session_id=session_id,
                section_name=section_name,
                next_section_name=next_section_name,
                score=score,
                signals=signals,
                tail_excerpt=tail_excerpt,
                trigger_kind="no_new_content",
            )
        path_state["pending_read_before_write"] = True
        path_state["escalation_level"] = max(prior_level, 1)
        path_state["blocked_attempts"] = int(path_state.get("blocked_attempts", 0) or 0) + 1
        path_state["last_error_kind"] = "chunked_write_no_new_content"
        path_state["outline_required"] = False
        _record_guard_event(
            state,
            event="loop_guard_no_new_content",
            path=resolved_path,
            session_id=session_id,
            payload={"section_name": section_name, "added_lines": added_lines},
        )
        return fail(
            "LoopGuard: no new staged content would be added by this chunk write. "
            "Read the current file content and advance to the next missing section.",
            metadata={
                "path": resolved_path,
                "write_session_id": session_id,
                "section_name": section_name,
                "next_section_name": next_section_name,
                "error_kind": "chunked_write_no_new_content",
                "loop_guard_score": score,
                "loop_guard_escalation_level": max(prior_level, 1),
                "loop_guard_signals": signals,
                "loop_guard_schedule_read": True,
                "loop_guard_tail_excerpt": tail_excerpt,
                "next_required_tool": _next_required_read(path),
            },
        )

    if score < int(config["stagnation_threshold"]):
        path_state["last_error_kind"] = ""
        return None

    if prior_level >= 2:
        return _hard_abort_chunked_write(
            state=state,
            path_state=path_state,
            resolved_path=resolved_path,
            session_id=session_id,
            section_name=section_name,
            next_section_name=next_section_name,
            score=score,
            signals=signals,
            tail_excerpt=tail_excerpt,
            trigger_kind="stagnation",
        )

    level = 1
    if score >= int(config["level2_threshold"]) or prior_level >= 1:
        level = 2
    path_state["pending_read_before_write"] = True
    path_state["escalation_level"] = max(prior_level, level)
    path_state["blocked_attempts"] = int(path_state.get("blocked_attempts", 0) or 0) + 1
    path_state["last_error_kind"] = "chunked_write_loop_guard"
    path_state["outline_required"] = level >= 2

    _record_guard_event(
        state,
        event="loop_guard_triggered",
        path=resolved_path,
        session_id=session_id,
        payload={
            "escalation_level": level,
            "stagnation_score": score,
            "section_name": section_name,
            "next_section_name": next_section_name,
            "signals": signals,
        },
    )
    message = (
        "LoopGuard: repeated chunk write detected. Read the current staged content and confirm the exact missing "
        "section before writing again."
    )
    if level >= 2:
        message = (
            "LoopGuard: repeated chunk write persisted after an earlier recovery. Stop blind writing, "
            "read the current staged content, and outline the remaining sections before the next file_write."
        )
    return fail(
        message,
        metadata={
            "path": resolved_path,
            "write_session_id": session_id,
            "section_name": section_name,
            "next_section_name": next_section_name,
            "error_kind": "chunked_write_loop_guard",
            "loop_guard_score": score,
            "loop_guard_escalation_level": level,
            "loop_guard_signals": signals,
            "loop_guard_schedule_read": True,
            "loop_guard_tail_excerpt": tail_excerpt,
            "next_required_tool": _next_required_read(path),
            "loop_guard_plan_only_requested": level >= 2,
            "loop_guard_outline_required": level >= 2,
            "loop_guard_outline_question": _outline_handoff_question(resolved_path) if level >= 2 else "",
        },
    )


def mark_chunked_write_success(
    *,
    state: LoopState | None,
    path: str,
    cwd: str | None,
    section_name: str,
) -> None:
    if state is None:
        return
    path_state = _path_state(state, _resolve(path, cwd))
    if path_state is None:
        return
    checkpoints = path_state.setdefault("section_checkpoints", [])
    normalized_section = str(section_name or "").strip()
    if normalized_section and normalized_section not in checkpoints:
        checkpoints.append(normalized_section)
    path_state["pending_read_before_write"] = False
    path_state["blocked_attempts"] = 0
    path_state["escalation_level"] = 0
    path_state["last_error_kind"] = ""
    path_state["outline_required"] = False
    _clear_loop_guard_read_schedule(state)


def clear_loop_guard_verification_requirement(
    state: LoopState | None,
    *,
    path: str,
    cwd: str | None,
) -> None:
    if state is None:
        return
    resolved_path = _resolve(path, cwd)
    path_state = _path_state(state, resolved_path)
    if path_state is None:
        return
    path_state["pending_read_before_write"] = False
    path_state["blocked_attempts"] = 0
    path_state["writes_since_last_read"] = 0
    path_state["last_read_at"] = time.time()
    path_state["last_error_kind"] = ""
    _clear_loop_guard_read_schedule(state)


def active_outline_requirements(state: LoopState | None) -> list[dict[str, Any]]:
    root = _loop_guard_root(state)
    if root is None:
        return []
    paths = root.get("paths", {})
    if not isinstance(paths, dict):
        return []

    active: list[dict[str, Any]] = []
    for path, payload in paths.items():
        if not isinstance(payload, dict) or not bool(payload.get("outline_required")):
            continue
        active.append(
            {
                "path": str(path),
                "session_id": str(payload.get("session_id", "") or ""),
                "pending_read_before_write": bool(payload.get("pending_read_before_write")),
                "blocked_attempts": int(payload.get("blocked_attempts", 0) or 0),
                "last_score": int(payload.get("last_score", 0) or 0),
                "last_section_name": str(payload.get("last_section_name", "") or ""),
                "last_next_section_name": str(payload.get("last_next_section_name", "") or ""),
                "last_error_kind": str(payload.get("last_error_kind", "") or ""),
            }
        )
    return active


def build_loop_guard_prompt(state: LoopState | None) -> str:
    active = active_outline_requirements(state)
    if not active:
        return ""

    primary = active[0]
    path = primary["path"]
    current_section = str(primary.get("last_section_name", "") or "").strip()
    next_section = str(primary.get("last_next_section_name", "") or "").strip()
    read_step = (
        f"First call `file_read(path='{path}')` to recover the staged content before planning."
        if primary.get("pending_read_before_write")
        else "Use the staged content already in context as the source of truth for the outline."
    )
    section_bits = []
    if current_section:
        section_bits.append(f"last repeated section: `{current_section}`")
    if next_section:
        section_bits.append(f"expected next section: `{next_section}`")
    section_context = f" ({' | '.join(section_bits)})" if section_bits else ""
    return (
        "\n\n### LOOPGUARD OUTLINE MODE\n"
        f"A repeated chunk-write loop was detected for `{path}`{section_context}. "
        f"{read_step} Then send exactly one `ask_human(question='...')` call containing a 3-bullet outline of the remaining sections for that file. "
        "Do not call `file_write`, `file_patch`, `ast_patch`, or `task_complete` until the user replies `continue`."
    )


def _matches_outline_requirement(
    *,
    state: LoopState | None,
    path: str | None,
    write_session_id: str | None,
) -> dict[str, Any] | None:
    active = active_outline_requirements(state)
    if not active:
        return None
    normalized_session_id = str(write_session_id or "").strip()
    if normalized_session_id:
        for item in active:
            if str(item.get("session_id", "") or "").strip() == normalized_session_id:
                return item
    normalized_path = _resolve(str(path or "").strip(), getattr(state, "cwd", None)) if path else ""
    if normalized_path:
        for item in active:
            if str(item.get("path", "") or "").strip() == normalized_path:
                return item
    return active[0] if len(active) == 1 else None


def outline_mode_violation(
    state: LoopState | None,
    *,
    tool_name: str,
    args: dict[str, Any] | None,
) -> dict[str, Any] | None:
    normalized_tool = str(tool_name or "").strip()
    if normalized_tool not in {"file_write", "file_append", "file_patch", "ast_patch", "task_complete"}:
        return None

    payload = args if isinstance(args, dict) else {}
    requirement = _matches_outline_requirement(
        state=state,
        path=str(payload.get("path") or "").strip(),
        write_session_id=str(payload.get("write_session_id") or payload.get("session_id") or "").strip(),
    )
    if requirement is None:
        return None

    path = str(requirement.get("path") or "").strip()
    read_step = (
        f"Call `file_read(path='{path}')` first, then send the outline via `ask_human`."
        if bool(requirement.get("pending_read_before_write"))
        else "Send the outline via `ask_human` before you resume writing."
    )
    message = (
        f"LoopGuard outline mode is active for `{path}`. {read_step} "
        "Do not call `file_write`, `file_patch`, `ast_patch`, or `task_complete` until the user replies `continue`."
    )
    return {
        "path": path,
        "write_session_id": str(requirement.get("session_id", "") or ""),
        "message": message,
        "question": _outline_handoff_question(path),
        "pending_read_before_write": bool(requirement.get("pending_read_before_write")),
    }


def build_loop_guard_outline_interrupt_payload(
    *,
    state: LoopState | None,
    thread_id: str,
    question: str,
    current_phase: str,
    active_profiles: list[str] | None,
    recent_tool_outcomes: list[dict[str, Any]],
) -> dict[str, Any] | None:
    active = active_outline_requirements(state)
    if not active:
        return None
    primary = active[0]
    path = str(primary.get("path") or "").strip()
    guidance = (
        f"Chunked write outline ready for `{path}`. "
        "Reply `continue` to confirm the outline and resume writing, or provide corrections to keep outline mode active."
    )
    return {
        "kind": "chunked_write_loop_guard_outline",
        "question": question or _outline_handoff_question(path),
        "guidance": guidance,
        "path": path,
        "write_session_id": str(primary.get("session_id", "") or ""),
        "current_phase": current_phase,
        "active_profiles": list(active_profiles or []),
        "thread_id": thread_id,
        "recent_tool_outcomes": recent_tool_outcomes,
    }


def clear_loop_guard_outline_requirement(
    state: LoopState | None,
    *,
    path: str | None = None,
    write_session_id: str | None = None,
    cwd: str | None = None,
) -> bool:
    root = _loop_guard_root(state)
    if root is None:
        return False
    paths = root.get("paths", {})
    if not isinstance(paths, dict):
        return False

    normalized_path = _resolve(path, cwd or getattr(state, "cwd", None)) if path else ""
    normalized_session_id = str(write_session_id or "").strip()
    cleared = False
    for stored_path, payload in paths.items():
        if not isinstance(payload, dict):
            continue
        session_id = str(payload.get("session_id", "") or "").strip()
        if normalized_session_id and session_id != normalized_session_id:
            continue
        if normalized_path and str(stored_path or "").strip() != normalized_path:
            continue
        if not normalized_path and not normalized_session_id and not bool(payload.get("outline_required")):
            continue
        payload["outline_required"] = False
        cleared = True
    if cleared:
        _clear_loop_guard_read_schedule(state)
    return cleared


def build_loop_guard_status(state: LoopState | None) -> dict[str, Any]:
    root = _loop_guard_root(state)
    if root is None:
        return {"active_paths": [], "recent_events": []}
    active_paths: list[dict[str, Any]] = []
    paths = root.get("paths", {})
    if isinstance(paths, dict):
        for path, payload in paths.items():
            if not isinstance(payload, dict):
                continue
            if (
                not payload.get("pending_read_before_write")
                and int(payload.get("escalation_level", 0) or 0) <= 0
                and not payload.get("recent_writes")
            ):
                continue
            active_paths.append(
                {
                    "path": str(path),
                    "escalation_level": int(payload.get("escalation_level", 0) or 0),
                    "pending_read_before_write": bool(payload.get("pending_read_before_write")),
                    "blocked_attempts": int(payload.get("blocked_attempts", 0) or 0),
                    "last_score": int(payload.get("last_score", 0) or 0),
                    "last_section_name": str(payload.get("last_section_name", "") or ""),
                    "last_next_section_name": str(payload.get("last_next_section_name", "") or ""),
                    "outline_required": bool(payload.get("outline_required")),
                    "section_checkpoints": list(payload.get("section_checkpoints", []) or []),
                }
            )
    recent_events = list(root.get("events", []) or [])[-10:]
    return {
        "active_paths": active_paths,
        "recent_events": recent_events,
    }
