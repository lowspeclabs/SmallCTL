from __future__ import annotations

import difflib
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from dataclasses import dataclass, field

from ..state import LoopState
from .fs_loop_guard_actions import outline_handoff_question
from .fs_loop_guard_status import loop_guard_status_payload
from .fs_loop_guard_utils import content_hash, count_added_lines, is_substantially_new_content, resolve_path, tail_excerpt as _tail_excerpt
from .fs_sessions import _write_session_can_finalize
from .type_coerce import as_bool, as_float, as_int
from .fs_loop_guard_support import (
    LoopGuardDecision,
    _DEFAULT_LOOP_GUARD_CONFIG,
    _LOOP_GUARD_CONFIG_KEY,
    _LOOP_GUARD_STATE_KEY,
    _append_recent_write,
    _clear_loop_guard_read_schedule,
    _emit_block,
    _hard_abort_chunked_write,
    _loop_guard_root,
    _path_state,
    loop_guard_config,
)


def _existing_section_content(session: Any, section_name: str, staged_content: str) -> str:
    ranges = getattr(session, "write_section_ranges", None) or {}
    section_range = ranges.get(section_name)
    if not isinstance(section_range, dict):
        return ""
    start = int(section_range.get("start", 0))
    end = int(section_range.get("end", start))
    return str(staged_content or "")[start:end]


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

    resolved_path = resolve_path(path, cwd)
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
    projected_hash = content_hash(updated_content)
    payload_hash = content_hash(content)
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
    added_lines = count_added_lines(staged_content, updated_content)

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
        if prior_level >= 2 or int(path_state.get("blocked_attempts", 0) or 0) >= 1:
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
        return _emit_block(
            state,
            path_state,
            LoopGuardDecision(
                action="block",
                message="LoopGuard: repeated write detected. You must read the current file content and confirm "
                "which section is missing before writing again.",
                error_kind="chunked_write_requires_read_before_write",
            ),
            resolved_path=resolved_path,
            session_id=session_id,
            section_name=section_name,
            next_section_name=next_section_name,
            score=score,
            signals=signals,
            tail_excerpt=tail_excerpt,
            level=level,
            outline_required=level >= 2,
        )

    if (
        bool(config.get("checkpoint_gate", True))
        and section_name
        and section_name in checkpoints
    ):
        existing_section = _existing_section_content(session, section_name, staged_content)
        if replace_strategy != "overwrite" and is_substantially_new_content(
            content,
            existing_section,
            similarity_threshold=float(config.get("checkpoint_extend_similarity_threshold", 0.85)),
            min_new_chars=int(config.get("checkpoint_extend_min_new_chars", 50)),
        ):
            pass
        elif prior_level >= 2 or int(path_state.get("blocked_attempts", 0) or 0) >= 1:
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
        else:
            return _emit_block(
                state,
                path_state,
                LoopGuardDecision(
                    action="block",
                    message=f"LoopGuard: section `{section_name}` was already checkpointed for `{path}`. "
                    "Read the current staged content and advance to the next section instead of rewriting it. "
                    "If you are adding a new subsection, use a different `section_name` or extend the staged file "
                    "with `replace_strategy='append'` and content that is clearly different from the checkpointed section.",
                    error_kind="chunked_write_checkpoint_revisit",
                ),
                resolved_path=resolved_path,
                session_id=session_id,
                section_name=section_name,
                next_section_name=next_section_name,
                score=score,
                signals=signals,
                tail_excerpt=tail_excerpt,
                level=1,
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
        return _emit_block(
            state,
            path_state,
            LoopGuardDecision(
                action="block",
                message="LoopGuard: chunked overwrite must grow the staged file or explicitly replace a missing section. "
                "Read the current staged content before overwriting from memory.",
                error_kind="chunked_write_non_growing_overwrite",
            ),
            resolved_path=resolved_path,
            session_id=session_id,
            section_name=section_name,
            next_section_name=next_section_name,
            score=score,
            signals=signals,
            tail_excerpt=tail_excerpt,
            level=1,
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
        return _emit_block(
            state,
            path_state,
            LoopGuardDecision(
                action="block",
                message="LoopGuard: overlapping chunk append detected. "
                "Call `file_read` to check the current staged content, then use `file_patch` (narrow exact edit) or `ast_patch` (narrow structural edit) "
                "or `file_write` with `replace_strategy=\"overwrite\"` to modify. Do not append existing overlapping lines.",
                error_kind="chunked_write_append_overlap_detected",
                extra_metadata={
                    "overlap_ratio": append_overlap_ratio,
                    "staged_bytes": len(staged_content.encode("utf-8", errors="replace")),
                    "incoming_bytes": len(content.encode("utf-8", errors="replace")),
                },
            ),
            resolved_path=resolved_path,
            session_id=session_id,
            section_name=section_name,
            next_section_name=next_section_name,
            score=score,
            signals=signals,
            tail_excerpt=tail_excerpt,
            level=1,
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
        return _emit_block(
            state,
            path_state,
            LoopGuardDecision(
                action="block",
                message="LoopGuard: no new staged content would be added by this chunk write. "
                "Read the current file content and advance to the next missing section.",
                error_kind="chunked_write_no_new_content",
            ),
            resolved_path=resolved_path,
            session_id=session_id,
            section_name=section_name,
            next_section_name=next_section_name,
            score=score,
            signals=signals,
            tail_excerpt=tail_excerpt,
            level=1,
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

    level = 2 if score >= int(config["level2_threshold"]) or prior_level >= 1 else 1
    message = (
        "LoopGuard: repeated chunk write detected. Read the current staged content and confirm the exact missing "
        "section before writing again."
    )
    if level >= 2:
        message = (
            "LoopGuard: repeated chunk write persisted after an earlier recovery. Stop blind writing, "
            "read the current staged content, and outline the remaining sections before the next file_write."
        )
    return _emit_block(
        state,
        path_state,
        LoopGuardDecision(
            action="block",
            message=message,
            error_kind="chunked_write_loop_guard",
            extra_metadata={
                "loop_guard_plan_only_requested": level >= 2,
            },
        ),
        resolved_path=resolved_path,
        session_id=session_id,
        section_name=section_name,
        next_section_name=next_section_name,
        score=score,
        signals=signals,
        tail_excerpt=tail_excerpt,
        level=level,
        outline_required=level >= 2,
        event_name="loop_guard_triggered",
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
    path_state = _path_state(state, resolve_path(path, cwd))
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
    resolved_path = resolve_path(path, cwd)
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
    normalized_path = resolve_path(str(path or "").strip(), getattr(state, "cwd", None)) if path else ""
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

    if normalized_tool == "task_complete":
        session = getattr(state, "write_session", None) if state is not None else None
        if (
            session is not None
            and str(getattr(session, "status", "") or "open").strip().lower() in {"open", "verifying"}
            and not str(getattr(session, "write_next_section", "") or "").strip()
            and bool(getattr(session, "write_sections_completed", []) or [])
            and _write_session_can_finalize(session)
        ):
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
        "question": outline_handoff_question(path),
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
        "question": question or outline_handoff_question(path),
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

    normalized_path = resolve_path(path, cwd or getattr(state, "cwd", None)) if path else ""
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
        # Outline confirmation from the user satisfies the read-before-write gate;
        # clear it so the model can legally append the next section.
        payload["pending_read_before_write"] = False
        cleared = True
    if cleared:
        _clear_loop_guard_read_schedule(state)
    return cleared


def build_loop_guard_status(state: LoopState | None) -> dict[str, Any]:
    root = _loop_guard_root(state)
    return loop_guard_status_payload(root, state)
