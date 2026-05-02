from __future__ import annotations

from pathlib import Path
from typing import Any

from ..state import WriteSession
from ..write_session_fsm import new_write_session, record_write_session_event
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
    if not isinstance(forced_targets, list) or not forced_targets:
        return False

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


def _ensure_chunk_write_session(harness: Any, target_path: str) -> WriteSession | None:
    target = str(target_path or "").strip()
    if not target:
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
    from .tool_outcomes import _register_write_session_stage_artifact
    from .write_session_health import (
        _abandon_staged_artifact,
        extract_defined_symbols,
        extract_symbols_from_task,
        is_staged_artifact_recoverable,
    )

    suggestions = _suggested_chunk_sections(target)
    session = new_write_session(
        session_id=new_write_session_id(),
        target_path=target,
        intent=infer_write_session_intent(target, getattr(harness.state, "cwd", None)),
        mode="chunked_author",
        suggested_sections=suggestions,
        next_section=suggestions[0] if suggestions else "",
    )
    harness.state.write_session = session

    stage_path = str(getattr(session, "write_staging_path", "") or "").strip()
    if stage_path and Path(stage_path).exists():
        task_description = getattr(getattr(harness.state, "run_brief", None), "original_task", "") or ""
        if not is_staged_artifact_recoverable(stage_path, task_description):
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

    if getattr(harness.state, "write_session", None):
        return False

    from ..guards import is_small_model_name

    model_name = getattr(getattr(harness, "client", None), "model", None)
    if not is_small_model_name(model_name):
        return False

    path = str(pending.args.get("path") or "").strip()
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
