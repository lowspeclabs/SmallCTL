from __future__ import annotations

from typing import Any

from ..normalization import dedupe_keep_tail


def _remember_session_ssh_target(
    service: Any,
    *,
    tool_name: str = "",
    result: Any,
    arguments: dict[str, Any] | None,
) -> None:
    if not isinstance(arguments, dict):
        return
    host = str(arguments.get("host") or "").strip().lower()
    user = str(arguments.get("user") or "").strip()
    if not host or not user:
        return

    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    if not (
        result.success
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    ):
        return
    reached_remote_host = (
        result.success
        or bool(metadata.get("ssh_transport_succeeded"))
        or str(metadata.get("failure_kind") or "").strip() == "remote_command"
    )

    targets = service.harness.state.scratchpad.setdefault("_session_ssh_targets", {})
    if not isinstance(targets, dict):
        targets = {}
        service.harness.state.scratchpad["_session_ssh_targets"] = targets

    remembered: dict[str, Any] = {
        "host": host,
        "user": user,
    }
    existing = targets.get(host)
    if isinstance(existing, dict) and existing.get("confirmed"):
        remembered["confirmed"] = True
    elif reached_remote_host:
        remembered["confirmed"] = True

    password = str(arguments.get("password") or "").strip()
    if password:
        remembered["password"] = password
    elif isinstance(existing, dict):
        if existing.get("password"):
            remembered["password"] = existing["password"]

    identity_file = str(arguments.get("identity_file") or "").strip()
    if identity_file:
        remembered["identity_file"] = identity_file
    port = arguments.get("port")
    if isinstance(port, int):
        remembered["port"] = port
    validated_tool = str(tool_name or "").strip()
    validated_path = str(
        arguments.get("path")
        or metadata.get("path")
        or (
            result.output.get("path")
            if isinstance(result.output, dict)
            else ""
        )
        or ""
    ).strip()
    if reached_remote_host and validated_tool:
        existing_tools = existing.get("validated_tools") if isinstance(existing, dict) else []
        if not isinstance(existing_tools, list):
            existing_tools = []
        remembered["validated_tools"] = dedupe_keep_tail(
            [str(item).strip() for item in existing_tools if str(item).strip()] + [validated_tool],
            limit=6,
        )
        remembered["last_success_tool"] = validated_tool
        prior_success_count = 0
        if isinstance(existing, dict):
            try:
                prior_success_count = max(0, int(existing.get("success_count") or 0))
            except (TypeError, ValueError):
                prior_success_count = 0
        remembered["success_count"] = prior_success_count + 1
    if reached_remote_host and validated_path:
        existing_paths = existing.get("validated_paths") if isinstance(existing, dict) else []
        if not isinstance(existing_paths, list):
            existing_paths = []
        remembered["validated_paths"] = dedupe_keep_tail(
            [str(item).strip() for item in existing_paths if str(item).strip()] + [validated_path],
            limit=8,
        )
        remembered["last_validated_path"] = validated_path
    targets[host] = remembered
