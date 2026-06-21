from __future__ import annotations

from typing import Any

from . import network
from .dispatcher_ssh_auth import password_fingerprint
from .dispatcher_ssh_context import infer_ssh_password_from_state_context


def infer_ssh_user_from_execution_records(host: str, *, state: Any | None = None) -> str:
    records = getattr(state, "tool_execution_records", None)
    if not isinstance(records, dict) or not records:
        return ""

    target_host = str(host or "").strip().lower()
    if not target_host:
        return ""

    for record in reversed(list(records.values())):
        if not isinstance(record, dict):
            continue
        if str(record.get("tool_name") or "").strip() != "ssh_exec":
            continue
        if not ssh_record_likely_authenticated(record):
            continue

        args = record.get("args")
        if not isinstance(args, dict):
            continue
        record_host = str(args.get("host") or "").strip()
        record_user = str(args.get("user") or "").strip()
        try:
            record_host, record_user_or_none = network.normalize_ssh_target(
                host=record_host,
                user=record_user or None,
            )
        except ValueError:
            continue
        if str(record_host or "").strip().lower() != target_host:
            continue
        normalized_record_user = str(record_user_or_none or "").strip()
        if normalized_record_user:
            return normalized_record_user
    return ""


def infer_ssh_user_from_session_memory(host: str, *, state: Any | None = None) -> str:
    target = session_ssh_target_record(host, state=state)
    return str(target.get("user") or "").strip() if isinstance(target, dict) else ""


def infer_ssh_password(
    host: str,
    *,
    user: str | None = None,
    state: Any | None = None,
    credential_store: Any | None = None,
) -> tuple[str, str]:
    inferred_from_records = infer_ssh_password_from_execution_records(
        host, user=user, state=state, credential_store=credential_store
    )
    if inferred_from_records:
        return inferred_from_records, "prior_ssh_exec"

    inferred_from_task = infer_ssh_password_from_state_context(host, user=user, state=state)
    if inferred_from_task:
        return inferred_from_task, "task_context"

    inferred_from_session = infer_ssh_password_from_session_memory(
        host, user=user, state=state, credential_store=credential_store
    )
    if inferred_from_session:
        return inferred_from_session, "session_memory"

    return "", ""


def infer_ssh_password_from_execution_records(
    host: str,
    *,
    user: str | None = None,
    state: Any | None = None,
    credential_store: Any | None = None,
) -> str:
    records = getattr(state, "tool_execution_records", None)
    if not isinstance(records, dict) or not records:
        return ""

    target_host = str(host or "").strip().lower()
    target_user = str(user or "").strip().lower()
    if not target_host:
        return ""

    for record in reversed(list(records.values())):
        if not isinstance(record, dict):
            continue
        if str(record.get("tool_name") or "").strip() != "ssh_exec":
            continue
        if not ssh_record_likely_authenticated(record):
            continue

        args = record.get("args")
        if not isinstance(args, dict):
            continue
        record_host = str(args.get("host") or "").strip()
        record_user = str(args.get("user") or "").strip()
        try:
            record_host, record_user_or_none = network.normalize_ssh_target(
                host=record_host,
                user=record_user or None,
            )
        except ValueError:
            continue
        if str(record_host or "").strip().lower() != target_host:
            continue

        normalized_record_user = str(record_user_or_none or "").strip().lower()
        if target_user and normalized_record_user != target_user:
            continue

        password = ""
        if credential_store is not None:
            password = credential_store.get_ssh_password(record_host, record_user_or_none) or ""
        if not password:
            fingerprint = str(args.get("password_fingerprint") or "").strip()
            if fingerprint and credential_store is not None:
                password = credential_store.get_ssh_password_by_fingerprint(fingerprint) or ""
        if not password:
            password = str(args.get("password") or "").strip()
        if password:
            return password
    return ""


def ssh_record_likely_authenticated(record: dict[str, Any]) -> bool:
    result = record.get("result")
    if not isinstance(result, dict):
        return False
    if bool(result.get("success")):
        return True
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("ssh_transport_succeeded")) or str(metadata.get("failure_kind") or "").strip() == "remote_command"


def infer_ssh_password_from_session_memory(
    host: str,
    *,
    user: str | None = None,
    state: Any | None = None,
    credential_store: Any | None = None,
) -> str:
    target = session_ssh_target_record(host, state=state)
    if not isinstance(target, dict):
        return ""
    target_user = str(user or "").strip().lower()
    session_user = str(target.get("user") or "").strip().lower()
    if target_user and session_user and target_user != session_user:
        return ""
    if credential_store is not None:
        password = credential_store.get_ssh_password(host, target.get("user")) or ""
        if password:
            return password
        fingerprint = str(target.get("password_fingerprint") or "").strip()
        if fingerprint:
            password = credential_store.get_ssh_password_by_fingerprint(fingerprint) or ""
            if password:
                return password
    return str(target.get("password") or "").strip()


def explicit_ssh_password_matches_current_user_context(
    host: str,
    password: str,
    *,
    user: str | None = None,
    state: Any | None = None,
) -> bool:
    if state is None:
        return False
    inferred = infer_ssh_password_from_state_context(host, user=user, state=state)
    return bool(inferred) and password_fingerprint(inferred) == password_fingerprint(password)


def session_ssh_target_record(host: str, *, state: Any | None = None) -> dict[str, Any]:
    if state is None:
        return {}
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    target_host = str(host or "").strip().lower()
    if not target_host:
        return {}
    targets = scratchpad.get("_session_ssh_targets")
    if not isinstance(targets, dict):
        return {}
    entry = targets.get(target_host)
    return dict(entry) if isinstance(entry, dict) else {}
