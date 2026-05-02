from __future__ import annotations

import json
import re
from typing import Any

from ..models.conversation import ConversationMessage
from ..state import WriteSession
from .state import GraphRunState, PendingToolCall, ToolExecutionRecord


def _maybe_emit_patch_existing_first_choice_nudge(
    harness: Any,
    session: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name not in {"file_write", "file_append"} or record.result.success:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if str(metadata.get("error_kind") or "").strip() != "patch_existing_requires_explicit_replace_strategy":
        return False
    if str(metadata.get("write_session_id") or "").strip() != str(getattr(session, "write_session_id", "") or "").strip():
        return False

    target_path = str(getattr(session, "write_target_path", "") or record.args.get("path") or "").strip()
    stage_path = str(metadata.get("staging_path") or getattr(session, "write_staging_path", "") or "").strip()
    from .write_session_recovery import _register_write_session_stage_artifact

    artifact_id = _register_write_session_stage_artifact(harness, session)
    artifact_hint = ""
    if artifact_id:
        artifact_hint = f" or `artifact_read(artifact_id='{artifact_id}')`"

    signature = "|".join(
        [
            str(getattr(session, "write_session_id", "") or ""),
            str(target_path or ""),
            str(stage_path or ""),
            str(record.operation_id or ""),
            "patch_existing_first_choice",
        ]
    )
    if harness.state.scratchpad.get("_patch_existing_first_choice_nudged") == signature:
        return False
    harness.state.scratchpad["_patch_existing_first_choice_nudged"] = signature

    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Write Session `{session.write_session_id}` is a `patch_existing` session with no tracked sections yet. "
                f"TARGET PATH for writes: `{target_path}`. "
                f"STAGED PATH for read/verify context only: `{stage_path}`. "
                f"First recover the current staged content by reading through the target path: `file_read(path='{target_path}')`{artifact_hint}. "
                "Then choose exactly one recovery shape: use `file_patch` for a narrow exact edit, `ast_patch` for a narrow structural edit, or "
                "`file_write` with `replace_strategy='overwrite'` only if you intentionally want to replace the entire staged file. "
                f"DIRECTIVE: do not write, patch, append, or delete `{stage_path}` directly; write tools must use `{target_path}`. "
                "Do not assume earlier chunks were lost or rewrite the whole file from memory."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "patch_existing_first_choice",
                "session_id": session.write_session_id,
                "target_path": target_path,
                "staging_path": stage_path,
                "artifact_id": artifact_id or "",
            },
        )
    )
    return True


def _recover_patch_existing_recovery_session(
    harness: Any,
    *,
    session_id: str,
    target_path: str,
    staging_path: str,
    metadata: dict[str, Any],
) -> WriteSession | Any | None:
    session = getattr(harness.state, "write_session", None)
    if session is not None:
        active_id = str(getattr(session, "write_session_id", "") or "").strip()
        active_target = str(getattr(session, "write_target_path", "") or "").strip()
        active_intent = str(getattr(session, "write_session_intent", "") or "").strip()
        if session_id and active_id == session_id:
            return session
        if active_target and target_path:
            try:
                from ..tools.fs import _same_target_path

                if _same_target_path(active_target, target_path, getattr(harness.state, "cwd", None)):
                    if active_intent == "patch_existing" and not session_id:
                        if staging_path and not str(getattr(session, "write_staging_path", "") or "").strip():
                            session.write_staging_path = staging_path
                        return session
            except Exception:
                if active_target == target_path and active_intent == "patch_existing" and not session_id:
                    if staging_path and not str(getattr(session, "write_staging_path", "") or "").strip():
                        session.write_staging_path = staging_path
                    return session

    recovered_intent = str(metadata.get("write_session_intent") or "").strip()
    if recovered_intent != "patch_existing":
        return session
    if not target_path:
        return session

    recovered_session = WriteSession(
        write_session_id=session_id,
        write_target_path=target_path,
        write_session_intent="patch_existing",
        write_staging_path=staging_path,
        write_target_existed_at_start=True,
        status="open",
    )
    harness.state.write_session = recovered_session
    return recovered_session


def _maybe_schedule_patch_existing_stage_read_recovery(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name not in {"file_write", "file_append"} or record.result.success:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if str(metadata.get("error_kind") or "").strip() != "patch_existing_requires_explicit_replace_strategy":
        return False

    session_id = str(metadata.get("write_session_id") or record.args.get("write_session_id") or "").strip()
    target_path = str(metadata.get("path") or record.args.get("path") or "").strip()
    if not target_path:
        return False
    staging_path = str(metadata.get("staging_path") or "").strip()

    session = _recover_patch_existing_recovery_session(
        harness,
        session_id=session_id,
        target_path=target_path,
        staging_path=staging_path,
        metadata=metadata,
    )
    resolved_intent = str(
        getattr(session, "write_session_intent", "")
        or metadata.get("write_session_intent")
        or ""
    ).strip()
    if resolved_intent != "patch_existing":
        return False

    recovery_session_id = str(getattr(session, "write_session_id", "") or session_id or "").strip()
    recovery_staging_path = str(staging_path or getattr(session, "write_staging_path", "") or "").strip()
    recovery_key = "|".join([recovery_session_id, target_path])
    raw_counts = harness.state.scratchpad.get("_patch_existing_stage_read_autocontinue_counts")
    counts = dict(raw_counts) if isinstance(raw_counts, dict) else {}
    attempt_count = int(counts.get(recovery_key, 0) or 0) + 1
    counts[recovery_key] = attempt_count
    harness.state.scratchpad["_patch_existing_stage_read_autocontinue_counts"] = counts
    if attempt_count > 1:
        harness.state.append_message(
            ConversationMessage(
                role="system",
                content=(
                    (
                        f"Auto-recovery stopped for Write Session `{recovery_session_id}` after repeated patch-existing "
                        "first-chunk failures. "
                    )
                    if recovery_session_id
                    else "Auto-recovery stopped after repeated patch-existing first-chunk failures. "
                )
                + "Do not retry `file_write` with the same implicit first-chunk choice. "
                + (
                    f"Inspect the staged copy at `{recovery_staging_path}` or reread the target with "
                    f"`file_read(path='{target_path}')`, then choose exactly one repair shape: "
                    if recovery_staging_path
                    else f"Reread the target with `file_read(path='{target_path}')`, then choose exactly one repair shape: "
                )
                + "`file_patch` for a narrow exact edit, `ast_patch` for a narrow structural edit, or `file_write` with "
                + "`replace_strategy='overwrite'` only if you intentionally want to replace the entire staged file.",
                metadata={
                    "is_recovery_nudge": True,
                    "recovery_kind": "patch_existing_stage_read_circuit_breaker",
                    "session_id": recovery_session_id,
                    "target_path": target_path,
                    "staging_path": recovery_staging_path,
                    "retry_count": attempt_count,
                },
            )
        )
        harness._runlog(
            "patch_existing_stage_read_circuit_breaker",
            "stopped automatic staged reads after repeated patch_existing first-choice failures",
            tool_call_id=record.tool_call_id,
            operation_id=record.operation_id,
            session_id=recovery_session_id,
            target_path=target_path,
            staging_path=recovery_staging_path,
            retry_count=attempt_count,
        )
        return False

    signature = "|".join(
        [
            str(record.operation_id or ""),
            recovery_session_id,
            target_path,
            "patch_existing_stage_read_autocontinue",
        ]
    )
    if harness.state.scratchpad.get("_patch_existing_stage_read_autocontinue") == signature:
        return False
    harness.state.scratchpad["_patch_existing_stage_read_autocontinue"] = signature

    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args={"path": target_path},
            raw_arguments=json.dumps({"path": target_path}, ensure_ascii=True, sort_keys=True),
            source="system",
        )
    ]
    harness.state.scratchpad["_patch_existing_stage_read_contract"] = {
        "session_id": recovery_session_id,
        "target_path": target_path,
        "staging_path": recovery_staging_path,
    }
    from .tool_call_parser import allow_repeated_tool_call_once

    allow_repeated_tool_call_once(harness, "file_read", {"path": target_path})
    staging_warning = f", not `{recovery_staging_path}`" if recovery_staging_path else ""
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                (
                    f"Auto-continuing recovery for Write Session `{getattr(session, 'write_session_id', '')}`: "
                    if session is not None and str(getattr(session, "write_session_id", "") or "").strip()
                    else "Auto-continuing patch-existing recovery: "
                )
                + f"reading current staged content via `file_read(path='{target_path}')`; the next write must target `{target_path}`{staging_warning}."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "patch_existing_stage_read_autocontinue",
                "session_id": recovery_session_id,
                "target_path": target_path,
                "staging_path": recovery_staging_path,
                "requires_explicit_followup_shape": True,
                "session_recovered": bool(session is not None and session_id and str(getattr(session, "write_session_id", "") or "").strip() == session_id),
            },
        )
    )
    harness._runlog(
        "patch_existing_stage_read_autocontinue",
        "scheduled automatic staged read after patch_existing first-choice failure",
        tool_call_id=record.tool_call_id,
        operation_id=record.operation_id,
        session_id=recovery_session_id,
        target_path=target_path,
        error_kind=str(metadata.get("error_kind") or ""),
        session_recovered=bool(session is not None and session_id and str(getattr(session, "write_session_id", "") or "").strip() == session_id),
    )
    return True


def _clear_patch_existing_stage_read_autocontinue_count_after_success(
    harness: Any,
    session: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name not in {"file_write", "file_append", "file_patch", "ast_patch"} or not record.result.success:
        return False

    raw_counts = harness.state.scratchpad.get("_patch_existing_stage_read_autocontinue_counts")
    if not isinstance(raw_counts, dict) or not raw_counts:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    session_id = str(
        metadata.get("write_session_id")
        or record.args.get("write_session_id")
        or getattr(session, "write_session_id", "")
        or ""
    ).strip()
    target_path = str(
        metadata.get("requested_path")
        or metadata.get("path")
        or record.args.get("path")
        or getattr(session, "write_target_path", "")
        or ""
    ).strip()
    canonical_target = str(getattr(session, "write_target_path", "") or target_path).strip()
    if not session_id or not target_path or not canonical_target:
        return False

    try:
        from ..tools.fs import _same_target_path

        same_target = _same_target_path(target_path, canonical_target, getattr(harness.state, "cwd", None))
    except Exception:
        same_target = target_path == canonical_target
    if not same_target:
        return False

    updated = dict(raw_counts)
    removed = False
    for key in list(updated):
        key_session, sep, key_target = str(key).partition("|")
        if not sep or key_session != session_id:
            continue
        try:
            key_matches_target = _same_target_path(key_target, canonical_target, getattr(harness.state, "cwd", None))
        except Exception:
            key_matches_target = key_target == canonical_target
        if key_matches_target:
            updated.pop(key, None)
            removed = True

    if not removed:
        return False
    if updated:
        harness.state.scratchpad["_patch_existing_stage_read_autocontinue_counts"] = updated
    else:
        harness.state.scratchpad.pop("_patch_existing_stage_read_autocontinue_counts", None)
    return True


def _maybe_schedule_file_patch_read_recovery(
    graph_state: GraphRunState,
    harness: Any,
    record: ToolExecutionRecord,
) -> bool:
    if record.tool_name != "file_patch" or record.result.success:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    error_kind = str(metadata.get("error_kind") or "").strip()
    if error_kind not in {"patch_target_not_found", "patch_occurrence_mismatch"}:
        return False

    target_path = str(
        metadata.get("requested_path")
        or metadata.get("path")
        or record.args.get("path")
        or ""
    ).strip()
    if not target_path:
        return False

    recovery_key = _file_patch_recovery_key(record, target_path=target_path)
    raw_counts = harness.state.scratchpad.get("_file_patch_recovery_counts")
    counts = dict(raw_counts) if isinstance(raw_counts, dict) else {}
    recovery_count = int(counts.get(recovery_key, 0) or 0) + 1
    counts[recovery_key] = recovery_count
    harness.state.scratchpad["_file_patch_recovery_counts"] = counts

    signature = "|".join(
        [
            str(record.operation_id or ""),
            target_path,
            error_kind,
            str(recovery_count),
            "file_patch_read_autocontinue",
        ]
    )
    if harness.state.scratchpad.get("_file_patch_read_autocontinue") == signature:
        return False
    harness.state.scratchpad["_file_patch_read_autocontinue"] = signature

    graph_state.pending_tool_calls = [
        PendingToolCall(
            tool_name="file_read",
            args={"path": target_path},
            raw_arguments=json.dumps({"path": target_path}, ensure_ascii=True, sort_keys=True),
            source="system",
        )
    ]
    from .tool_call_parser import allow_repeated_tool_call_once

    allow_repeated_tool_call_once(harness, "file_read", {"path": target_path})
    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Auto-continuing patch recovery for `{target_path}`: "
                f"reading the current file with `file_read(path='{target_path}')` before asking for another patch."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "file_patch_read_autocontinue",
                "target_path": target_path,
                "error_kind": error_kind,
            },
        )
    )
    if recovery_count >= 2:
        _maybe_emit_ast_patch_recovery_nudge(
            harness,
            record,
            target_path=target_path,
            error_kind=error_kind,
            recovery_count=recovery_count,
        )
    harness._runlog(
        "file_patch_read_autocontinue",
        "scheduled automatic file read after patch mismatch",
        tool_call_id=record.tool_call_id,
        operation_id=record.operation_id,
        target_path=target_path,
        error_kind=error_kind,
        recovery_count=recovery_count,
    )
    return True


def _file_patch_recovery_key(record: ToolExecutionRecord, *, target_path: str) -> str:
    target_text = str(record.args.get("target_text") or "").strip()
    replacement_text = str(record.args.get("replacement_text") or "").strip()
    return "|".join(
        [
            target_path,
            target_text[:200],
            replacement_text[:200],
        ]
    )


def _maybe_emit_ast_patch_recovery_nudge(
    harness: Any,
    record: ToolExecutionRecord,
    *,
    target_path: str,
    error_kind: str,
    recovery_count: int,
) -> bool:
    structural_candidate = _looks_like_structural_patch_candidate(record)
    recovery_kind = "file_patch_ast_patch_nudge" if structural_candidate else "file_patch_local_repair_nudge"
    signature = "|".join(
        [
            target_path,
            error_kind,
            recovery_kind,
            _file_patch_recovery_key(record, target_path=target_path),
        ]
    )
    if harness.state.scratchpad.get("_file_patch_recovery_nudge") == signature:
        return False
    harness.state.scratchpad["_file_patch_recovery_nudge"] = signature

    if structural_candidate:
        content = (
            f"Repeated exact-text patch misses hit `{target_path}` for the same apparent edit intent. "
            "After the recovery `file_read`, if the target is really a function, class, import, call, argument, or field, "
            "stop retrying the same substring and switch to `ast_patch` with a structural locator."
        )
    else:
        content = (
            f"Repeated exact-text patch misses hit `{target_path}` for the same apparent edit intent. "
            "After the recovery `file_read`, if you still cannot make the exact target unique, stop retrying the same substring "
            "and use a small local `file_write` repair on the narrowest safe slice instead."
        )

    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=content,
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": recovery_kind,
                "target_path": target_path,
                "error_kind": error_kind,
                "recovery_count": recovery_count,
            },
        )
    )
    harness._runlog(
        recovery_kind,
        "nudged repeated exact patch recovery toward a different repair shape",
        tool_call_id=record.tool_call_id,
        operation_id=record.operation_id,
        target_path=target_path,
        error_kind=error_kind,
        recovery_count=recovery_count,
        structural_candidate=structural_candidate,
    )
    return True


_STRUCTURAL_PATCH_HINT_RE = re.compile(
    r"(^|\n)\s*(?:@|async\s+def\s+|def\s+|class\s+|from\s+\S+\s+import\s+|import\s+\S+|return\s+\w+\(|\w+\()",
    re.MULTILINE,
)


def _looks_like_structural_patch_candidate(record: ToolExecutionRecord) -> bool:
    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    text_parts = [
        str(record.args.get("target_text") or ""),
        str(record.args.get("replacement_text") or ""),
    ]
    for key in ("target_text_preview", "replacement_text_preview"):
        preview = metadata.get(key)
        if isinstance(preview, dict):
            text_parts.append(str(preview.get("preview") or "").replace("\\n", "\n"))
    combined = "\n".join(part for part in text_parts if str(part).strip())
    if not combined.strip():
        return False
    if _STRUCTURAL_PATCH_HINT_RE.search(combined):
        return True
    lowered = combined.lower()
    return any(
        marker in lowered
        for marker in (
            "subprocess.run(",
            "dataclass",
            "__init__(",
            "pathlib",
        )
    )


def _maybe_emit_write_session_target_path_redirect_nudge(harness: Any, record: ToolExecutionRecord) -> bool:
    if record.tool_name not in {"file_write", "file_append", "file_patch", "ast_patch", "file_delete"} or record.result.success:
        return False

    metadata = record.result.metadata if isinstance(record.result.metadata, dict) else {}
    if str(metadata.get("error_kind") or "").strip() != "write_session_staging_path_used_as_target":
        return False

    session_id = str(metadata.get("write_session_id") or "").strip()
    target_path = str(metadata.get("target_path") or "").strip()
    staging_path = str(metadata.get("staging_path") or "").strip()
    if not session_id or not target_path or not staging_path:
        return False

    signature = "|".join(
        [
            str(record.operation_id or ""),
            session_id,
            str(record.tool_name or ""),
            target_path,
            staging_path,
            "write_session_target_path_redirect",
        ]
    )
    if harness.state.scratchpad.get("_write_session_target_path_redirect_nudged") == signature:
        return False
    harness.state.scratchpad["_write_session_target_path_redirect_nudged"] = signature

    if record.tool_name in {"file_write", "file_append"}:
        same_args_hint = "Reuse the same content and section metadata; only correct the path."
    elif record.tool_name == "file_patch":
        same_args_hint = "Reuse the same patch arguments; only correct the path."
    else:
        same_args_hint = "Only retry if deleting the target file is still the intended action."

    harness.state.append_message(
        ConversationMessage(
            role="system",
            content=(
                f"Write Session `{session_id}` uses target `{target_path}`. "
                f"The last `{record.tool_name}` call addressed the staged copy `{staging_path}` directly. "
                f"Retry the same `{record.tool_name}` call with `path='{target_path}'` instead. "
                f"{same_args_hint} Keep the staging path only for `file_read` or `artifact_read`."
            ),
            metadata={
                "is_recovery_nudge": True,
                "recovery_kind": "write_session_target_path_redirect",
                "tool_name": str(record.tool_name or ""),
                "session_id": session_id,
                "target_path": target_path,
                "staging_path": staging_path,
            },
        )
    )
    harness._runlog(
        "write_session_target_path_redirect_nudge",
        "injected recovery nudge after direct staging-path mutation attempt",
        tool_name=str(record.tool_name or ""),
        tool_call_id=record.tool_call_id,
        operation_id=record.operation_id,
        session_id=session_id,
        target_path=target_path,
        staging_path=staging_path,
    )
    return True
