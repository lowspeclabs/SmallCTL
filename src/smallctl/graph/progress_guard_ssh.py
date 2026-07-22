from __future__ import annotations

import hashlib
import re
from typing import Any


def ssh_exec_remote_paths(record: Any) -> list[str]:
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not command:
        return []
    paths: list[str] = []
    for match in _REMOTE_PATH_RE.finditer(command):
        path = match.group(0)
        if path not in paths:
            paths.append(path)
    return paths[:8]


def ssh_exec_read_targets(record: Any) -> list[str]:
    """Return file targets from a read-only SSH command, resolving a leading cd."""
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    command = str(args.get("command") or "").strip()
    if not _SSH_READ_COMMAND_RE.search(command) or _SSH_MUTATION_COMMAND_RE.search(command):
        return []
    base_match = _SSH_CD_PREFIX_RE.search(command)
    base = base_match.group("path").rstrip("/") if base_match else ""
    targets = [path for path in ssh_exec_remote_paths(record) if path != base]
    for match in _SSH_RELATIVE_FILE_RE.finditer(command):
        path = match.group("path")
        if path.startswith("/"):
            continue
        resolved = f"{base}/{path}" if base else path
        if resolved not in targets:
            targets.append(resolved)
    return targets[:8]


def ssh_exec_output_fingerprint(record: Any) -> str:
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    output = metadata.get("output")
    if not isinstance(output, dict):
        output = {}
    text = "\n".join(
        str(output.get(key) or "").strip()
        for key in ("stdout", "stderr")
        if str(output.get(key) or "").strip()
    )
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return ""
    return hashlib.sha256(normalized[:4096].encode("utf-8", errors="replace")).hexdigest()[:16]


def ssh_exec_observation_entries(harness: Any) -> list[dict[str, Any]]:
    state = getattr(harness, "state", None)
    if state is None:
        return []
    scratchpad = getattr(state, "scratchpad", {})
    history = scratchpad.get("_progress_ssh_observation_history", [])
    return history if isinstance(history, list) else []


def ssh_exec_has_novel_partial_output(harness: Any, record: Any) -> bool:
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    if not bool(metadata.get("output_received")):
        return False
    fingerprint = ssh_exec_output_fingerprint(record)
    if not fingerprint:
        return False
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    host = str(args.get("host") or metadata.get("host") or "").strip().lower()
    command = str(args.get("command") or metadata.get("command") or "").strip()
    if not host or not command:
        return False
    for item in ssh_exec_observation_entries(harness):
        if not isinstance(item, dict):
            continue
        if str(item.get("host") or "").strip().lower() != host:
            continue
        if str(item.get("command") or "").strip() != command:
            continue
        if str(item.get("output_fingerprint") or "").strip() == fingerprint:
            return False
    return True


def ssh_exec_has_novel_remote_observation(harness: Any, record: Any) -> bool:
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    host = str(args.get("host") or metadata.get("host") or "").strip().lower()
    failure_class = str(metadata.get("ssh_error_class") or metadata.get("failure_kind") or "").strip()
    auth_mode = str(metadata.get("ssh_auth_mode") or "").strip()
    reached_remote_host = (
        bool(getattr(record.result, "success", False))
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    )
    prior_entries = ssh_exec_observation_entries(harness)
    if not host:
        return False

    if failure_class:
        if not any(
            str(item.get("host") or "").strip().lower() == host
            and str(item.get("failure_class") or "").strip() == failure_class
            for item in prior_entries
            if isinstance(item, dict)
        ):
            return True

    for path in ssh_exec_remote_paths(record):
        if not any(
            str(item.get("host") or "").strip().lower() == host
            and path in (item.get("paths") or [])
            for item in prior_entries
            if isinstance(item, dict)
        ):
            return True

    if reached_remote_host and auth_mode:
        if not any(
            str(item.get("host") or "").strip().lower() == host
            and bool(item.get("reached_remote_host"))
            and str(item.get("auth_mode") or "").strip() == auth_mode
            for item in prior_entries
            if isinstance(item, dict)
        ):
            return True

    return False


def record_ssh_exec_observation(harness: Any, record: Any) -> None:
    state = getattr(harness, "state", None)
    if state is None:
        return
    scratchpad = getattr(state, "scratchpad", {})
    if not isinstance(scratchpad, dict):
        return
    history = scratchpad.setdefault("_progress_ssh_observation_history", [])
    if not isinstance(history, list):
        return
    metadata = record.result.metadata if isinstance(getattr(record, "result", None), object) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    args = record.args if isinstance(getattr(record, "args", None), dict) else {}
    history.append(
        {
            "host": str(args.get("host") or metadata.get("host") or "").strip().lower(),
            "command": str(args.get("command") or metadata.get("command") or "").strip(),
            "failure_class": str(metadata.get("ssh_error_class") or metadata.get("failure_kind") or "").strip(),
            "paths": ssh_exec_remote_paths(record),
            "auth_mode": str(metadata.get("ssh_auth_mode") or "").strip(),
            "output_fingerprint": ssh_exec_output_fingerprint(record),
            "reached_remote_host": (
                bool(getattr(record.result, "success", False))
                or bool(metadata.get("ssh_transport_succeeded"))
                or str(metadata.get("failure_kind") or "").strip() == "remote_command"
            ),
        }
    )
    if len(history) > 32:
        del history[: len(history) - 32]


_REMOTE_PATH_RE = re.compile(r"(?<![\w/])/(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+")
_SSH_CD_PREFIX_RE = re.compile(r"(?:^|[;&|])\s*cd\s+(?P<path>/[^\s;&|]+)\s*&&")
_SSH_READ_COMMAND_RE = re.compile(r"\b(?:cat|grep|head|tail|sed\s+-n)\b")
_SSH_MUTATION_COMMAND_RE = re.compile(r"\b(?:sed\s+-i|tee|truncate|rm|mv|cp|chmod|chown|docker\s+compose\s+(?:up|restart))\b|>>?|\b(?:append|write)\b")
_SSH_RELATIVE_FILE_RE = re.compile(r"(?<![\w/])(?P<path>(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9_-]+\.(?:conf|json|py|toml|yaml|yml))(?![\w/])")
