from __future__ import annotations

import base64
import json
import re
import shlex
from typing import Any

from ..state import LoopState
from .common import fail, ok
from . import network
from .ssh_files_patch_utils import (
    apply_exact_patch_content,
    apply_replace_between_content,
    best_patch_match,
    find_bounded_regions,
    find_whitespace_normalized_bounded_regions,
    find_whitespace_normalized_spans,
    normalize_whitespace_with_spans,
    preview_match_context,
    preview_text,
    sha256_text,
)
from .ssh_files_remote_helper import _REMOTE_HELPER_SOURCE
from .ssh_files_preconditions import (
    _resolve_expected_sha_precondition,
    _sha_from_artifact,
    _artifact_path_for_precondition,
)
from .ssh_files_mutation_tracking import _clear_remote_mutation_requirement


SSH_FILE_MUTATING_TOOLS = {"ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}
REMOTE_MUTATION_VERIFICATION_KEY = "_remote_mutation_requires_verification"

# When the base64-encoded JSON payload exceeds this size, switch from passing it
# as a command-line argument (which is bounded by ARG_MAX / MAX_ARG_STRLEN) to
# piping it through the SSH process's stdin.
_MAX_ARGV_PAYLOAD_SIZE = 128 * 1024

_SHA256_HEX_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _guard_ssh_local_path(path: str) -> dict[str, Any] | None:
    normalized = str(path or "").strip()
    if normalized.startswith("./temp/") and normalized.endswith((".txt", ".md", ".py")):
        return fail(
            f"Use local `file_write(path='{normalized}')` for local artifacts. "
            "SSH file tools are for remote evidence gathering only.",
            metadata={
                "error_kind": "ssh_local_path_blocked",
                "next_required_tool": {
                    "tool_name": "file_write",
                    "arguments": {"path": normalized},
                },
            },
        )
    return None

def _build_remote_command(payload: dict[str, Any]) -> tuple[str, str | None]:
    encoded = base64.b64encode(json.dumps(payload, ensure_ascii=True).encode("utf-8")).decode("ascii")
    if len(encoded) > _MAX_ARGV_PAYLOAD_SIZE:
        command = "python3 -c " + shlex.quote(_REMOTE_HELPER_SOURCE) + " --stdin"
        return command, encoded
    return "python3 -c " + shlex.quote(_REMOTE_HELPER_SOURCE) + " " + shlex.quote(encoded), None


def _normalize_remote_connection(
    *,
    target: str | None,
    host: str | None,
    user: str | None,
    port: int,
    identity_file: str | None,
    password: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    try:
        normalized = network.normalize_ssh_arguments(
            {
                "target": target,
                "host": host,
                "user": user,
                "port": port,
                "identity_file": identity_file,
                "password": password,
            }
        )
    except ValueError as exc:
        return None, {"reason": "invalid_ssh_target", "message": str(exc)}
    return normalized, None


async def _run_remote_file_action(
    *,
    action: str,
    path: str,
    payload: dict[str, Any],
    target: str | None,
    host: str | None,
    user: str | None,
    port: int,
    identity_file: str | None,
    password: str | None,
    timeout_sec: int,
    state: LoopState | None,
    harness: Any,
) -> dict[str, Any]:
    connection, error = _normalize_remote_connection(
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
    )
    if error is not None or connection is None:
        return fail(error.get("message", "Invalid SSH target."), metadata=error or {})

    helper_payload = {
        **payload,
        "action": action,
        "path": path,
    }
    command, stdin_payload = _build_remote_command(helper_payload)
    result = await network.run_ssh_command(
        host=str(connection.get("host") or ""),
        user=connection.get("user"),
        port=int(connection.get("port") or 22),
        identity_file=connection.get("identity_file"),
        password=connection.get("password"),
        command=command,
        stdin_data=stdin_payload,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )
    if not result.get("success"):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        output = metadata.get("output") if isinstance(metadata.get("output"), dict) else {}
        stderr = str(output.get("stderr") or result.get("error") or "")
        reason = "remote_python_missing" if "python3" in stderr.lower() and "not found" in stderr.lower() else "remote_helper_failed"
        return fail(
            result.get("error") or "Remote SSH file helper failed.",
            metadata={
                "path": path,
                "host": connection.get("host"),
                "reason": reason,
                "recovery_hint": "Install python3 on the remote host or use explicit ssh_exec." if reason == "remote_python_missing" else "Inspect remote SSH helper output.",
                "ssh_result": result,
            },
        )

    output = result.get("output") if isinstance(result.get("output"), dict) else {}
    stdout = str(output.get("stdout") or "").strip()
    if not stdout:
        return fail(
            "Remote SSH file helper produced no JSON output.",
            metadata={"path": path, "host": connection.get("host"), "error_kind": "remote_helper_no_output"},
        )
    try:
        data = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        return fail(
            f"Remote SSH file helper produced invalid JSON: {exc}",
            metadata={"path": path, "host": connection.get("host"), "error_kind": "remote_helper_invalid_json", "stdout": stdout[-1000:]},
        )
    if not isinstance(data, dict):
        return fail(
            "Remote SSH file helper returned a non-object JSON payload.",
            metadata={"path": path, "host": connection.get("host"), "error_kind": "remote_helper_invalid_json"},
        )

    data.setdefault("path", path)
    data["host"] = connection.get("host")
    if connection.get("user"):
        data["user"] = connection.get("user")
    data["tool_generated_remote_command"] = True
    if not data.get("ok"):
        message = str(data.get("message") or data.get("error") or "Remote SSH file operation failed.")
        data.pop("ok", None)
        return fail(message, metadata=data)

    data.pop("ok", None)
    metadata = {k: v for k, v in data.items() if k != "content"}
    if action == "read":
        metadata.setdefault("complete_file", not data.get("truncated", False))
        content = data.get("content", "")
        if isinstance(content, str):
            metadata.setdefault("total_lines", content.count("\n") + (1 if content and not content.endswith("\n") else 0))
    return ok(data, metadata=metadata)


async def ssh_file_read(
    path: str,
    target: str | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    encoding: str = "utf-8",
    max_bytes: int = 262144,
    truncate: bool = True,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    return await _run_remote_file_action(
        action="read",
        path=path,
        payload={"encoding": encoding, "max_bytes": max_bytes, "truncate": truncate},
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )


async def ssh_file_write(
    path: str,
    content: str,
    target: str | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    encoding: str = "utf-8",
    mode: str = "overwrite",
    create_parent_dirs: bool = False,
    backup: bool = True,
    expected_sha256: str | None = None,
    source_artifact_id: str | None = None,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    local_guard = _guard_ssh_local_path(path)
    if local_guard is not None:
        return local_guard
    intended_sha = sha256_text(content, encoding)
    if expected_sha256 is not None and not _SHA256_HEX_RE.match(str(expected_sha256).strip()):
        return fail(
            "expected_sha256 is not a valid 64-character hex SHA-256 hash. "
            "Omit expected_sha256 and use source_artifact_id from the most recent ssh_file_read artifact to derive the hash automatically.",
            metadata={
                "reason": "invalid_expected_sha256_syntax",
                "expected_sha256_preview": str(expected_sha256)[:40],
            },
        )
    resolved_expected_sha256, precondition_metadata = _resolve_expected_sha_precondition(
        path=path,
        expected_sha256=expected_sha256,
        source_artifact_id=source_artifact_id,
        state=state,
    )
    if precondition_metadata is not None and "resolved_expected_sha256" not in precondition_metadata:
        return fail(precondition_metadata["message"], metadata=precondition_metadata)
    result = await _run_remote_file_action(
        action="write",
        path=path,
        payload={
            "content": content,
            "encoding": encoding,
            "mode": mode,
            "create_parent_dirs": create_parent_dirs,
            "backup": backup,
            "expected_sha256": resolved_expected_sha256,
        },
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )
    if result.get("success"):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if precondition_metadata is not None:
            metadata.update(precondition_metadata)
            result["metadata"] = metadata
        if metadata.get("readback_sha256") != intended_sha:
            return fail(
                "Remote readback hash did not match intended content.",
                metadata={**metadata, "error_kind": "readback_mismatch", "intended_sha256": intended_sha},
            )
        output_dict = result.get("output")
        if isinstance(output_dict, dict):
            output_dict["content"] = content
        _clear_remote_mutation_requirement(state, path=path, host=str(metadata.get("host") or ""))
    return result


async def ssh_file_patch(
    path: str,
    target_text: str,
    replacement_text: str,
    target: str | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    encoding: str = "utf-8",
    expected_occurrences: int = 1,
    backup: bool = True,
    expected_sha256: str | None = None,
    source_artifact_id: str | None = None,
    whitespace_normalized: bool = False,
    dry_run: bool = False,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    local_guard = _guard_ssh_local_path(path)
    if local_guard is not None:
        return local_guard
    if expected_sha256 is not None and not _SHA256_HEX_RE.match(str(expected_sha256).strip()):
        return fail(
            "expected_sha256 is not a valid 64-character hex SHA-256 hash. "
            "Omit expected_sha256 and use source_artifact_id from the most recent ssh_file_read artifact to derive the hash automatically.",
            metadata={
                "reason": "invalid_expected_sha256_syntax",
                "expected_sha256_preview": str(expected_sha256)[:40],
            },
        )
    resolved_expected_sha256, precondition_metadata = _resolve_expected_sha_precondition(
        path=path,
        expected_sha256=expected_sha256,
        source_artifact_id=source_artifact_id,
        state=state,
    )
    if precondition_metadata is not None and "resolved_expected_sha256" not in precondition_metadata:
        return fail(precondition_metadata["message"], metadata=precondition_metadata)
    if whitespace_normalized and not dry_run and not (
        resolved_expected_sha256 or str(source_artifact_id or "").strip()
    ):
        return fail(
            "whitespace_normalized mode requires `expected_sha256` or `source_artifact_id`. Prefer `dry_run=True` first to preview the matched region and planned hash.",
            metadata={"reason": "precondition_required_for_whitespace_normalized"},
        )
    result = await _run_remote_file_action(
        action="patch",
        path=path,
        payload={
            "target_text": target_text,
            "replacement_text": replacement_text,
            "encoding": encoding,
            "expected_occurrences": expected_occurrences,
            "backup": backup,
            "expected_sha256": resolved_expected_sha256,
            "whitespace_normalized": whitespace_normalized,
            "dry_run": dry_run,
        },
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )
    if result.get("success"):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if precondition_metadata is not None:
            metadata.update(precondition_metadata)
            result["metadata"] = metadata
        if dry_run:
            return result
        _clear_remote_mutation_requirement(state, path=path, host=str(metadata.get("host") or ""))
    return result


async def ssh_file_replace_between(
    path: str,
    start_text: str,
    end_text: str,
    replacement_text: str,
    target: str | None = None,
    host: str | None = None,
    user: str | None = None,
    port: int = 22,
    identity_file: str | None = None,
    password: str | None = None,
    encoding: str = "utf-8",
    include_bounds: bool = True,
    expected_occurrences: int = 1,
    backup: bool = True,
    expected_sha256: str | None = None,
    source_artifact_id: str | None = None,
    whitespace_normalized: bool = False,
    dry_run: bool = False,
    timeout_sec: int = 60,
    state: LoopState | None = None,
    harness: Any = None,
) -> dict[str, Any]:
    local_guard = _guard_ssh_local_path(path)
    if local_guard is not None:
        return local_guard
    if expected_sha256 is not None and not _SHA256_HEX_RE.match(str(expected_sha256).strip()):
        return fail(
            "expected_sha256 is not a valid 64-character hex SHA-256 hash. "
            "Omit expected_sha256 and use source_artifact_id from the most recent ssh_file_read artifact to derive the hash automatically.",
            metadata={
                "reason": "invalid_expected_sha256_syntax",
                "expected_sha256_preview": str(expected_sha256)[:40],
            },
        )
    resolved_expected_sha256, precondition_metadata = _resolve_expected_sha_precondition(
        path=path,
        expected_sha256=expected_sha256,
        source_artifact_id=source_artifact_id,
        state=state,
    )
    if precondition_metadata is not None and "resolved_expected_sha256" not in precondition_metadata:
        return fail(precondition_metadata["message"], metadata=precondition_metadata)
    if whitespace_normalized and not dry_run and not (
        resolved_expected_sha256 or str(source_artifact_id or "").strip()
    ):
        return fail(
            "whitespace_normalized mode requires `expected_sha256` or `source_artifact_id`. Prefer `dry_run=True` first to preview the matched region and planned hash.",
            metadata={"reason": "precondition_required_for_whitespace_normalized"},
        )
    result = await _run_remote_file_action(
        action="replace_between",
        path=path,
        payload={
            "start_text": start_text,
            "end_text": end_text,
            "replacement_text": replacement_text,
            "encoding": encoding,
            "include_bounds": include_bounds,
            "expected_occurrences": expected_occurrences,
            "backup": backup,
            "expected_sha256": resolved_expected_sha256,
            "whitespace_normalized": whitespace_normalized,
            "dry_run": dry_run,
        },
        target=target,
        host=host,
        user=user,
        port=port,
        identity_file=identity_file,
        password=password,
        timeout_sec=timeout_sec,
        state=state,
        harness=harness,
    )
    if result.get("success"):
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        if precondition_metadata is not None:
            metadata.update(precondition_metadata)
            result["metadata"] = metadata
        if dry_run:
            return result
        _clear_remote_mutation_requirement(state, path=path, host=str(metadata.get("host") or ""))
    return result
