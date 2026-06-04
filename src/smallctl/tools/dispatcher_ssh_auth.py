from __future__ import annotations

import hashlib
from typing import Any


def ssh_auth_recovery_entry_key(host: str, user: str) -> str:
    normalized_host = str(host or "").strip().lower()
    normalized_user = str(user or "").strip().lower()
    return f"{normalized_user}@{normalized_host}" if normalized_user else normalized_host


def password_fingerprint(password: str) -> str:
    value = str(password or "").strip()
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def ssh_auth_debug_metadata(
    arguments: dict[str, Any],
    *,
    password_source: str,
) -> dict[str, Any]:
    password = str(arguments.get("password") or "").strip()
    identity_file = str(arguments.get("identity_file") or "").strip()
    auth_mode = "password" if password else "key"
    auth_transport = "sshpass_env" if password else "ssh"
    origin = str(password_source or "").strip() or ("explicit" if password else "none")
    return {
        "ssh_auth_mode": auth_mode,
        "ssh_auth_transport": auth_transport,
        "ssh_password_origin": origin,
        "ssh_password_recovered": origin in {"task_context", "prior_ssh_exec", "session_memory"},
        "ssh_identity_file_supplied": bool(identity_file),
    }
