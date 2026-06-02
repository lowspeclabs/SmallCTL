from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from ..state import WriteSession
from ..write_session_fsm import _ARCHIVED_WRITE_SESSIONS_KEY, new_write_session, record_write_session_event
from .state import PendingToolCall


def _suggested_chunk_sections(path: str) -> list[str]:
    ext = Path(path).suffix.lower()
    if ext == ".py":
        return ["imports", "types/interfaces", "constants/globals", "helpers", "main_logic", "tests/entrypoint"]
    if ext in {".js", ".ts", ".tsx"}:
        return ["imports", "types", "constants", "utils", "main_component/logic", "exports"]
    if ext == ".go":
        return ["package", "imports", "types", "const/var", "helpers", "main_logic"]
    if ext in {".md", ".txt"}:
        return ["header", "overview", "details", "footer"]
    return ["header", "implementation", "footer"]


def _write_policy_value(harness: Any, name: str, default: Any) -> Any:
    config = getattr(harness, "config", None)
    if config is None:
        return default
    return getattr(config, name, default)


def _task_forces_chunk_mode(harness: Any, path: str) -> bool:
    if not path:
        return False
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {}) or {}
    forced_targets = scratchpad.get("_force_chunk_mode_targets")
    if isinstance(forced_targets, list) and forced_targets:
        from ..tools.fs import _resolve

        try:
            candidate = _resolve(path, getattr(harness.state, "cwd", None))
        except Exception:
            candidate = None
        for target in forced_targets:
            try:
                forced_path = _resolve(str(target), getattr(harness.state, "cwd", None))
            except Exception:
                forced_path = None
            if candidate is not None and forced_path is not None and candidate == forced_path:
                return True
            if str(target).strip() == str(path).strip():
                return True

    # Small web artifacts are safer as one complete write; chunking is the loop
    # failure surface for single-file HTML games and similar deliverables.
    task_text = getattr(getattr(harness.state, "run_brief", None), "original_task", "") or ""
    lowered_task = task_text.lower()
    has_spec_hint = "spec" in lowered_task or "-spec" in lowered_task or "_spec" in lowered_task
    is_web_file = Path(path).suffix.lower() in {".html", ".htm", ".css", ".js", ".jsx", ".ts", ".tsx"}
    if has_spec_hint and is_web_file and "large" in lowered_task:
        return True

    return False


def _active_write_session_for_target(harness: Any, target_path: str) -> WriteSession | None:
    target = str(target_path or "").strip()
    if not target:
        return None
    session = getattr(getattr(harness, "state", None), "write_session", None)
    if not session or str(getattr(session, "status", "")).strip().lower() == "complete":
        return None

    from ..tools.fs import _resolve

    try:
        target_resolved = _resolve(target, getattr(harness.state, "cwd", None))
        session_resolved = _resolve(session.write_target_path, getattr(harness.state, "cwd", None))
    except Exception:
        target_resolved = None
        session_resolved = None
    if target_resolved is not None and session_resolved is not None and target_resolved == session_resolved:
        return session
    if str(session.write_target_path).strip() == target:
        return session
    return None


def _same_resolved_path(harness: Any, left: str, right: str) -> bool:
    if not left or not right:
        return False
    from ..tools.fs import _resolve

    try:
        return _resolve(left, getattr(harness.state, "cwd", None)) == _resolve(right, getattr(harness.state, "cwd", None))
    except Exception:
        return str(left).strip() == str(right).strip()


def _short_content_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except Exception:
        return ""


def _latest_archived_stage_for_target(harness: Any, target_path: str) -> dict[str, Any] | None:
    scratchpad = getattr(getattr(harness, "state", None), "scratchpad", {}) or {}
    archived = scratchpad.get(_ARCHIVED_WRITE_SESSIONS_KEY)
    if not isinstance(archived, list):
        return None
    for item in reversed(archived):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip().lower()
        if status == "complete":
            continue
        if not _same_resolved_path(harness, str(item.get("write_target_path") or ""), target_path):
            continue
        stage_path = Path(str(item.get("write_staging_path") or ""))
        if not stage_path.exists() or not stage_path.is_file():
            continue
        try:
            if stage_path.stat().st_size <= 0:
                continue
        except Exception:
            continue
        return item
    return None


def _migrate_archived_stage_into_session(harness: Any, session: WriteSession, target_path: str) -> dict[str, Any] | None:
    archived = _latest_archived_stage_for_target(harness, target_path)
    if archived is None:
        return None

    from ..tools.fs import _resolve
    from ..tools.fs_write_sessions import _session_stage_path

    old_stage = Path(str(archived.get("write_staging_path") or ""))
    try:
        target = _resolve(target_path, getattr(harness.state, "cwd", None))
        new_stage = _session_stage_path(session.write_session_id, target, getattr(harness.state, "cwd", None))
        new_stage.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_stage, new_stage)
    except Exception as exc:
        runlog = getattr(harness, "_runlog", None)
        if callable(runlog):
            runlog(
                "write_session_stage_migration_failed",
                "failed to migrate archived write-session stage",
                source_session_id=str(archived.get("write_session_id") or ""),
                target_session_id=session.write_session_id,
                target_path=target_path,
                error=str(exc),
            )
        return None

    session.write_staging_path = str(new_stage)
    session.write_session_intent = str(archived.get("write_session_intent") or session.write_session_intent or "replace_file")
    session.write_original_snapshot_path = str(archived.get("write_original_snapshot_path") or "")
    session.write_target_existed_at_start = bool(archived.get("write_target_existed_at_start", False))
    session.write_section_ranges = dict(archived.get("write_section_ranges") or {})
    session.write_last_attempt_snapshot_path = str(archived.get("write_last_attempt_snapshot_path") or "")
    session.write_last_attempt_sections = list(archived.get("write_last_attempt_sections") or [])
    session.write_last_attempt_ranges = dict(archived.get("write_last_attempt_ranges") or {})
    session.write_sections_completed = list(archived.get("write_sections_completed") or [])
    session.write_current_section = str(archived.get("write_current_section") or "")
    session.write_next_section = str(archived.get("write_next_section") or session.write_next_section or "")
    session.write_pending_finalize = bool(archived.get("write_pending_finalize", False))
    session.write_last_staged_hash = str(archived.get("write_last_staged_hash") or "") or _short_content_hash(new_stage)
    if not session.suggested_sections:
        session.suggested_sections = list(archived.get("suggested_sections") or [])

    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "write_session_stage_migrated",
            "migrated archived write-session stage into prearmed session",
            source_session_id=str(archived.get("write_session_id") or ""),
            target_session_id=session.write_session_id,
            target_path=target_path,
            stage_path=str(new_stage),
            bytes=new_stage.stat().st_size,
        )
    return archived


def _ensure_chunk_write_session(harness: Any, target_path: str) -> WriteSession | None:
    target = str(target_path or "").strip()
    if not target:
        return None

    from ..harness.task_classifier import task_is_local_coding_target

    task_text = getattr(getattr(harness.state, "run_brief", None), "original_task", "") or ""
    if task_is_local_coding_target(task_text):
        return None

    existing = _active_write_session_for_target(harness, target)
    if existing is not None:
        return existing

    from ..guards import is_small_model_name

    model_name = getattr(getattr(harness, "client", None), "model", None)
    if not is_small_model_name(model_name):
        return None
    if not _task_forces_chunk_mode(harness, target):
        return None

    from ..tools.fs import infer_write_session_intent, new_write_session_id
    from ..prompts import _is_write_first_task
    from .tool_outcomes import _register_write_session_stage_artifact
    from .write_session_health import (
        _abandon_staged_artifact,
        extract_defined_symbols,
        extract_symbols_from_task,
        is_staged_artifact_recoverable,
    )

    suggestions = _suggested_chunk_sections(target)
    intent = infer_write_session_intent(target, getattr(harness.state, "cwd", None))
    if intent == "patch_existing" and _is_write_first_task(harness.state):
        intent = "replace_file"
    session = new_write_session(
        session_id=new_write_session_id(),
        target_path=target,
        intent=intent,
        mode="chunked_author",
        suggested_sections=suggestions,
        next_section=suggestions[0] if suggestions else "",
    )
    harness.state.write_session = session
    migrated_archived_stage = _migrate_archived_stage_into_session(harness, session, target)

    stage_path = str(getattr(session, "write_staging_path", "") or "").strip()
    if stage_path and Path(stage_path).exists():
        task_description = getattr(getattr(harness.state, "run_brief", None), "original_task", "") or ""
        if migrated_archived_stage is None and not is_staged_artifact_recoverable(stage_path, task_description):
            required = extract_symbols_from_task(task_description)
            try:
                content = Path(stage_path).read_text(encoding="utf-8")
            except Exception:
                content = ""
            defined = extract_defined_symbols(content)
            missing = sorted(required - defined)
            reason = (
                "Staged artifact rejected: implementation shell detected"
                + (f" (missing {', '.join(missing)})." if missing else ".")
                + " Starting fresh."
            )
            _abandon_staged_artifact(harness, stage_path, reason)
        else:
            _register_write_session_stage_artifact(harness, session)
    else:
        _register_write_session_stage_artifact(harness, session)

    record_write_session_event(
        harness.state,
        event="session_opened",
        session=session,
        details={"source": "chunk_mode_recovery"},
    )
    runlog = getattr(harness, "_runlog", None)
    if callable(runlog):
        runlog(
            "chunk_mode_prearmed",
            "initialized chunked authoring session from recovery path",
            session_id=session.write_session_id,
            target_path=target,
        )
    return session


def _should_enter_chunk_mode(harness: Any, pending: PendingToolCall) -> bool:
    if pending.tool_name != "file_write":
        return False

    active_session = getattr(harness.state, "write_session", None)
    if active_session and str(getattr(active_session, "status", "") or "").strip().lower() != "complete":
        return False

    from ..guards import is_small_model_name

    model_name = getattr(getattr(harness, "client", None), "model", None)
    if not is_small_model_name(model_name):
        return False

    path = str(pending.args.get("path") or "").strip()
    if Path(path).suffix.lower() in {".html", ".htm"}:
        content = str(pending.args.get("content", ""))
        if len(content) < _write_policy_value(harness, "single_shot_small_artifact_chars", 20000):
            return False
    if _task_forces_chunk_mode(harness, path):
        return True

    content = str(pending.args.get("content", ""))
    payload_size = len(content)

    if payload_size >= _write_policy_value(harness, "small_model_soft_write_chars", 2000):
        return True

    if _write_policy_value(harness, "chunk_mode_new_file_only", True) and path:
        from ..tools.fs import _resolve
        target = _resolve(path, harness.state.cwd)
        if not target.exists():
            lines = content.count("\n") + 1
            if lines >= _write_policy_value(harness, "new_file_chunk_mode_line_estimate", 100):
                return True

    return False
