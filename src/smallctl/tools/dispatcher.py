from __future__ import annotations

import logging
import re
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from ..logging_utils import RunLogger, log_kv
from ..models.tool_result import ToolEnvelope
from . import network
from .registry import ToolRegistry

Tier2Adapter = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

_ARTIFACT_TOKEN_RE = re.compile(r"\bA\d+\b", re.IGNORECASE)
_RECENT_ARTIFACT_ALIASES = {
    "above",
    "above artifact",
    "above output",
    "current artifact",
    "current output",
    "last",
    "last artifact",
    "last output",
    "latest",
    "latest artifact",
    "latest output",
    "most recent",
    "most recent artifact",
    "most recent output",
    "previous artifact",
    "recent",
    "recent artifact",
    "recent output",
    "that artifact",
    "that output",
    "the artifact above",
    "the latest artifact",
    "the latest output",
    "the most recent artifact",
    "the most recent output",
    "the previous artifact",
}
_SSH_USERNAME_TASK_PATTERNS = (
    re.compile(r"\busername\s+(?:is\s+)?(?P<user>[A-Za-z0-9._-]+)\b", re.IGNORECASE),
    re.compile(r"\buser\s+(?:is\s+)?(?P<user>[A-Za-z0-9._-]+)\b", re.IGNORECASE),
)
_TOOL_ALIAS_REPAIRS = {
    "use_shell_exec": "shell_exec",
    "use_ssh_exec": "ssh_exec",
}
_SSH_TASK_TARGET_RE = re.compile(
    r"\b(?:ssh|scp|sftp)\s+(?:[A-Za-z0-9._-]+@)?(?P<host>[A-Za-z0-9._-]+)\b",
    re.IGNORECASE,
)
_AT_HOST_TARGET_RE = re.compile(r"\b[A-Za-z0-9._-]+@(?P<host>[A-Za-z0-9._-]+)\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_REMOTE_TASK_HINT_RE = re.compile(r"\b(?:remote|ssh|username|password|server|host)\b", re.IGNORECASE)


@runtime_checkable
class ToolInterceptor(Protocol):
    """Protocol for tool dispatch middleware/interceptors."""

    async def __call__(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        next_dispatch: Callable[[str, dict[str, Any]], Awaitable[ToolEnvelope]],
    ) -> ToolEnvelope:
        ...


class ToolDispatcher:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        state: Any | None = None,
        phase: str = "explore",
        ansible_check_mode_in_plan: bool = True,
        tier2_adapter: Tier2Adapter | None = None,
        run_logger: RunLogger | None = None,
    ) -> None:
        self.log = logging.getLogger("smallctl.dispatcher")
        self.run_logger = run_logger
        self.registry = registry
        self.state = state
        self.phase = phase
        self.ansible_check_mode_in_plan = ansible_check_mode_in_plan
        self.tier2_adapter = tier2_adapter

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolEnvelope:
        requested_tool_name = tool_name
        tool_name, arguments, intercepted_result, normalization_metadata = normalize_tool_request(
            self.registry,
            tool_name,
            arguments,
            phase=self.phase,
            state=self.state,
        )
        log_kv(
            self.log,
            logging.INFO,
            "tool_dispatch_start",
            tool_name=tool_name,
            phase=self.phase,
            requested_tool_name=requested_tool_name if requested_tool_name != tool_name else "",
            arg_keys=sorted(list(arguments.keys())) if isinstance(arguments, dict) else [],
        )
        if self.run_logger:
            self.run_logger.log(
                "tools",
                "dispatch_start",
                "tool dispatch started",
                tool_name=tool_name,
                phase=self.phase,
                arguments=arguments,
                requested_tool_name=requested_tool_name if requested_tool_name != tool_name else None,
            )
        spec = self.registry.get(tool_name)
        if spec is None:
            log_kv(self.log, logging.WARNING, "tool_dispatch_unknown", tool_name=tool_name)
            if self.run_logger:
                self.run_logger.log("tools", "unknown", "unknown tool", tool_name=tool_name)
            return ToolEnvelope(
                success=False,
                error=f"Unknown tool: {tool_name}",
                metadata={"tool_name": tool_name},
            )
        log_kv(
            self.log,
            logging.DEBUG,
            "tool_dispatch_spec",
            tool_name=tool_name,
            category=spec.category,
            risk=spec.risk,
        )
        dispatch_metadata = {
            "tool_name": tool_name,
            "tier": spec.tier,
            "tool_risk": spec.risk,
            "tool_category": spec.category,
            "dispatch_phase": self.phase,
        }
        if self.run_logger:
            self.run_logger.log(
                "tools",
                "dispatch_spec",
                "tool metadata",
                tool_name=tool_name,
                category=spec.category,
                risk=spec.risk,
                phase=self.phase,
            )
        if not spec.phase_allowed(self.phase):
            log_kv(
                self.log,
                logging.WARNING,
                "tool_dispatch_phase_blocked",
                tool_name=tool_name,
                phase=self.phase,
            )
            if self.run_logger:
                self.run_logger.log(
                    "tools",
                    "phase_blocked",
                    "tool blocked by phase",
                    tool_name=tool_name,
                    phase=self.phase,
                )
            return ToolEnvelope(
                success=False,
                error=f"Tool '{tool_name}' is not allowed in phase '{self.phase}'",
                metadata={"tool_name": tool_name, "phase": self.phase},
            )

        if intercepted_result is not None:
            intercepted_result.metadata = {
                **dispatch_metadata,
                **normalization_metadata,
                **(intercepted_result.metadata if isinstance(intercepted_result.metadata, dict) else {}),
            }
            log_kv(
                self.log,
                logging.INFO,
                "tool_dispatch_complete",
                tool_name=tool_name,
                success=intercepted_result.success,
                tier=spec.tier,
            )
            if self.run_logger:
                self.run_logger.log(
                    "tools",
                    "dispatch_complete",
                    "tool dispatch finished",
                    tool_name=tool_name,
                    success=intercepted_result.success,
                    tier=spec.tier,
                    output=intercepted_result.output,
                    error=intercepted_result.error,
                )
            return intercepted_result

        args = self._coerce_args(spec.schema, arguments)
        validation_error = self._validate_args(spec.schema, args)
        if validation_error:
            log_kv(
                self.log,
                logging.WARNING,
                "tool_dispatch_validation_error",
                tool_name=tool_name,
                error=validation_error,
            )
            if self.run_logger:
                self.run_logger.log(
                    "tools",
                    "validation_error",
                    "tool argument validation failed",
                    tool_name=tool_name,
                    error=validation_error,
                )
            return ToolEnvelope(
                success=False,
                error=validation_error,
                metadata={"tool_name": tool_name},
            )

        if (
            self.phase == "plan"
            and self.ansible_check_mode_in_plan
            and tool_name in {"ansible_task", "ansible_playbook"}
        ):
            args["check"] = True

        try:
            if spec.tier == "tier2":
                if self.tier2_adapter is None:
                    raise RuntimeError("No tier2 adapter configured")
                result = await self.tier2_adapter(tool_name, args)
            else:
                result = await spec.invoke(**args)
        except Exception as exc:
            log_kv(
                self.log,
                logging.ERROR,
                "tool_dispatch_exception",
                tool_name=tool_name,
                error=str(exc),
                tier=spec.tier,
            )
            if self.run_logger:
                self.run_logger.log(
                    "tools",
                    "dispatch_exception",
                    "tool dispatch raised exception",
                    tool_name=tool_name,
                    error=str(exc),
                    tier=spec.tier,
                )
            return ToolEnvelope(
                success=False,
                error=str(exc),
                metadata={"tool_name": tool_name, "tier": spec.tier},
            )

        if isinstance(result, dict) and {"success", "output", "error", "metadata"} <= set(
            result.keys()
        ):
            result_metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
            result_metadata = {**dispatch_metadata, **normalization_metadata, **result_metadata}
            log_kv(
                self.log,
                logging.INFO,
                "tool_dispatch_complete",
                tool_name=tool_name,
                success=bool(result["success"]),
                tier=spec.tier,
            )
            if self.run_logger:
                self.run_logger.log(
                    "tools",
                    "dispatch_complete",
                    "tool dispatch finished",
                    tool_name=tool_name,
                    success=bool(result["success"]),
                    tier=spec.tier,
                    output=result.get("output"),
                    error=result.get("error"),
            )
            return ToolEnvelope(
                success=bool(result["success"]),
                status=result.get("status"),
                output=result.get("output"),
                error=result.get("error"),
                metadata=result_metadata,
            )
        return ToolEnvelope(
            success=True,
            output=result,
            metadata={**dispatch_metadata, **normalization_metadata},
        )

    @staticmethod
    def _coerce_args(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            return arguments

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not properties and not required:
            return {}
        coerced = dict(arguments)
        if isinstance(properties, dict):
            coerced = {key: value for key, value in coerced.items() if key in properties}
            for key, value in list(coerced.items()):
                expected_type = properties.get(key, {}).get("type")
                coerced[key] = _coerce_value(expected_type, value)
        return coerced

    @staticmethod
    def _validate_args(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
        if not isinstance(arguments, dict):
            return "Tool arguments must be an object."

        required = schema.get("required", [])
        for field in required:
            if field not in arguments:
                return f"Missing required field: {field}"

        properties = schema.get("properties", {})
        for key, val in arguments.items():
            if key not in properties:
                continue
            expected_type = properties[key].get("type")
            if expected_type and not _type_matches(expected_type, val):
                return f"Field '{key}' expected type '{expected_type}'"
        return None


class PipelineDispatcher:
    """Combines a base ToolDispatcher with multiple ToolInterceptors."""

    def __init__(self, base: ToolDispatcher, interceptors: list[ToolInterceptor]) -> None:
        self.base = base
        self.interceptors = interceptors

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolEnvelope:
        async def _run(idx: int, name: str, args: dict[str, Any]) -> ToolEnvelope:
            if idx >= len(self.interceptors):
                return await self.base.dispatch(name, args)
            interceptor = self.interceptors[idx]
            return await interceptor(name, args, lambda n, a: _run(idx + 1, n, a))

        return await _run(0, tool_name, arguments)


def normalize_tool_request(
    registry: ToolRegistry,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    phase: str | None = None,
    state: Any | None = None,
) -> tuple[str, dict[str, Any], ToolEnvelope | None, dict[str, Any]]:
    normalization_metadata: dict[str, Any] = {}
    original_tool_name = str(tool_name or "").strip()
    repaired_tool_name = _TOOL_ALIAS_REPAIRS.get(original_tool_name, original_tool_name)
    if repaired_tool_name != original_tool_name:
        normalization_metadata.update(
            {
                "repaired_tool_alias_from": original_tool_name,
                "repaired_tool_alias_to": repaired_tool_name,
                "routing_reason": "tool_alias_repair",
            }
        )
    tool_name = repaired_tool_name

    if tool_name == "artifact_read":
        tool_name, arguments, artifact_metadata = _normalize_artifact_read_request(arguments, state=state)
        normalization_metadata.update(artifact_metadata)

    if tool_name == "ssh_exec":
        try:
            normalized_arguments = network.normalize_ssh_arguments(arguments)
            normalized_arguments, ssh_metadata = _recover_ssh_arguments_from_task_context(
                normalized_arguments,
                state=state,
            )
            normalization_metadata.update(ssh_metadata)
            return tool_name, normalized_arguments, None, normalization_metadata
        except ValueError as exc:
            return (
                tool_name,
                arguments,
                ToolEnvelope(
                    success=False,
                    error=str(exc),
                    metadata={
                        "tool_name": tool_name,
                        "reason": "invalid_ssh_target",
                    },
                ),
                normalization_metadata,
            )

    if tool_name != "shell_exec":
        return tool_name, arguments, None, normalization_metadata

    ssh_spec = registry.get("ssh_exec") if hasattr(registry, "get") else None
    phase_allowed = getattr(ssh_spec, "phase_allowed", None)
    if ssh_spec is None or (phase and callable(phase_allowed) and not phase_allowed(phase)):
        return tool_name, arguments, None, normalization_metadata

    command = str(arguments.get("command", "") or "").strip() if isinstance(arguments, dict) else ""
    if not command:
        return tool_name, arguments, None, normalization_metadata

    try:
        rewritten_args = network.parse_ssh_exec_args_from_shell_command(command)
    except ValueError as exc:
        return (
            tool_name,
            arguments,
            ToolEnvelope(
                success=False,
                error=str(exc),
                metadata={
                    "tool_name": "ssh_exec",
                    "reason": "invalid_ssh_target",
                    "rewritten_from_tool": "shell_exec",
                },
            ),
            normalization_metadata,
        )
    if rewritten_args is None:
        if _task_clearly_targets_remote_ssh_host(state) and not re.search(r"\b(?:ssh|scp|sftp)\b", command):
            return (
                tool_name,
                arguments,
                ToolEnvelope(
                    success=False,
                    error="This is a remote task. Use `ssh_exec`, not local `shell_exec`.",
                    metadata={
                        "tool_name": tool_name,
                        "reason": "remote_task_requires_ssh_exec",
                        "suggested_tool": "ssh_exec",
                    },
                ),
                normalization_metadata,
            )
        return tool_name, arguments, None, normalization_metadata

    if "timeout_sec" in arguments and "timeout_sec" not in rewritten_args:
        rewritten_args["timeout_sec"] = arguments["timeout_sec"]
    normalization_metadata = {
        "rewritten_from_tool": "shell_exec",
        "routing_reason": "ssh_shell_command",
    }
    return "ssh_exec", rewritten_args, None, normalization_metadata


def _recover_ssh_arguments_from_task_context(
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(arguments, dict):
        return arguments, {}

    repaired = dict(arguments)
    host = str(repaired.get("host") or "").strip()
    user = str(repaired.get("user") or "").strip()
    if user or not host:
        return repaired, {}

    inferred_user = _infer_ssh_user_from_state_context(host, state=state)
    if not inferred_user:
        return repaired, {}

    repaired["user"] = inferred_user
    return repaired, {
        "recovered_ssh_user": inferred_user,
        "routing_reason": "ssh_task_context_user_recovery",
    }


def _infer_ssh_user_from_state_context(host: str, *, state: Any | None = None) -> str:
    target_host = str(host or "").strip().lower()
    if not target_host or state is None:
        return ""

    for text in _ssh_task_context_texts(state):
        if not text:
            continue
        embedded_match = re.search(
            rf"\b(?P<user>[A-Za-z0-9._-]+)@{re.escape(target_host)}\b",
            text,
            re.IGNORECASE,
        )
        if embedded_match is not None:
            return str(embedded_match.group("user") or "").strip()

        lowered = text.lower()
        if target_host not in lowered:
            continue
        for pattern in _SSH_USERNAME_TASK_PATTERNS:
            match = pattern.search(text)
            if match is not None:
                return str(match.group("user") or "").strip()
    return ""


def _ssh_task_context_texts(state: Any) -> list[str]:
    texts: list[str] = []

    run_brief = getattr(state, "run_brief", None)
    original_task = str(getattr(run_brief, "original_task", "") or "").strip()
    if original_task:
        texts.append(original_task)

    current_goal = str(getattr(getattr(state, "working_memory", None), "current_goal", "") or "").strip()
    if current_goal:
        texts.append(current_goal)

    for message in getattr(state, "recent_messages", []) or []:
        if str(getattr(message, "role", "") or "").strip().lower() != "user":
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if content:
            texts.append(content)

    return texts


def _task_clearly_targets_remote_ssh_host(state: Any | None) -> bool:
    if state is None:
        return False

    for text in _ssh_task_context_texts(state):
        if not text:
            continue
        if _SSH_TASK_TARGET_RE.search(text) is not None:
            return True
        if _AT_HOST_TARGET_RE.search(text) is not None:
            return True
        if _IPV4_RE.search(text) is not None and _REMOTE_TASK_HINT_RE.search(text) is not None:
            return True
    return False


def _normalize_artifact_read_request(
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    if not isinstance(arguments, dict):
        return "artifact_read", arguments, {}

    repaired = dict(arguments)

    explicit_artifact_id = repaired.get("artifact_id")
    if isinstance(explicit_artifact_id, str):
        normalized_artifact_id, metadata = _normalize_artifact_reference(
            explicit_artifact_id,
            field_name="artifact_id",
            state=state,
        )
        if normalized_artifact_id is not None:
            repaired["artifact_id"] = normalized_artifact_id
            return "artifact_read", repaired, metadata
        if _looks_like_file_path(explicit_artifact_id):
            return "file_read", _build_file_read_args(repaired, explicit_artifact_id), {
                "rewritten_from_tool": "artifact_read",
                "routing_reason": "artifact_id_to_file_read",
                "repair_field": "artifact_id",
            }

    for field_name in ("path", "id"):
        raw_value = repaired.get(field_name)
        if not isinstance(raw_value, str):
            continue

        normalized_artifact_id, metadata = _normalize_artifact_reference(
            raw_value,
            field_name=field_name,
            state=state,
        )
        if normalized_artifact_id is not None:
            repaired.pop(field_name, None)
            repaired["artifact_id"] = normalized_artifact_id
            return "artifact_read", repaired, metadata

        if _looks_like_file_path(raw_value):
            return "file_read", _build_file_read_args(repaired, raw_value), {
                "rewritten_from_tool": "artifact_read",
                "routing_reason": "artifact_path_to_file_read",
                "repair_field": field_name,
            }

    recent_artifact_id = _most_recent_artifact_id(state)
    if recent_artifact_id and _artifact_read_implicitly_targets_recent(arguments):
        repaired["artifact_id"] = recent_artifact_id
        return "artifact_read", repaired, {
            "argument_repair": "artifact_read_recent_fallback",
            "resolved_artifact_id": recent_artifact_id,
        }

    return "artifact_read", arguments, {}


def _normalize_artifact_reference(
    value: str,
    *,
    field_name: str,
    state: Any | None = None,
) -> tuple[str | None, dict[str, Any]]:
    normalized = _normalize_recent_artifact_alias(value, state=state)
    if normalized is not None:
        return normalized, {
            "argument_repair": "artifact_read_recent_fallback",
            "repair_field": field_name,
            "resolved_artifact_id": normalized,
        }

    extracted = _extract_artifact_id_token(value)
    if extracted is None:
        return None, {}

    canonical = _canonical_artifact_id(extracted, state=state)
    metadata: dict[str, Any] = {
        "argument_repair": "artifact_read_alias_to_artifact_id",
        "repair_field": field_name,
        "resolved_artifact_id": canonical,
    }
    return canonical, metadata


def _normalize_recent_artifact_alias(value: str, *, state: Any | None = None) -> str | None:
    alias = " ".join(str(value or "").strip().lower().split())
    if alias not in _RECENT_ARTIFACT_ALIASES:
        return None
    return _most_recent_artifact_id(state)


def _extract_artifact_id_token(value: str) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None

    match = _ARTIFACT_TOKEN_RE.search(text)
    if match is None:
        return None
    return match.group(0).upper()


def _canonical_artifact_id(candidate: str, *, state: Any | None = None) -> str:
    normalized = str(candidate or "").strip()
    if not normalized:
        return normalized

    artifacts = getattr(state, "artifacts", None)
    if not isinstance(artifacts, dict) or not artifacts:
        return normalized.upper() if normalized.upper().startswith("A") else normalized

    if normalized in artifacts:
        return normalized

    upper_candidate = normalized.upper()
    if upper_candidate in artifacts:
        return upper_candidate

    if not upper_candidate.startswith("A"):
        return normalized

    try:
        numeric_value = int(upper_candidate[1:])
    except ValueError:
        return upper_candidate

    for artifact_id in artifacts.keys():
        if not isinstance(artifact_id, str) or not artifact_id.upper().startswith("A"):
            continue
        try:
            if int(artifact_id[1:]) == numeric_value:
                return artifact_id
        except ValueError:
            continue
    return upper_candidate


def _artifact_read_implicitly_targets_recent(arguments: dict[str, Any]) -> bool:
    if not isinstance(arguments, dict):
        return False

    non_empty_keys = {
        key
        for key, value in arguments.items()
        if value is not None and (not isinstance(value, str) or value.strip())
    }
    if not non_empty_keys:
        return True
    return non_empty_keys <= {"start_line", "end_line", "max_chars"}


def _most_recent_artifact_id(state: Any | None) -> str | None:
    if state is None:
        return None

    artifacts = getattr(state, "artifacts", None)
    if not isinstance(artifacts, dict) or not artifacts:
        return None

    retrieval_cache = getattr(state, "retrieval_cache", None)
    if isinstance(retrieval_cache, list):
        for artifact_id in reversed(retrieval_cache):
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                continue
            canonical = _canonical_artifact_id(artifact_id, state=state)
            if canonical in artifacts:
                return canonical

    for artifact_id in reversed(list(artifacts.keys())):
        if isinstance(artifact_id, str) and artifact_id.strip():
            return artifact_id
    return None


def _build_file_read_args(arguments: dict[str, Any], path_value: str) -> dict[str, Any]:
    repaired: dict[str, Any] = {"path": path_value}
    for key in ("start_line", "end_line"):
        if key in arguments:
            repaired[key] = arguments[key]
    if "max_bytes" in arguments:
        repaired["max_bytes"] = arguments["max_bytes"]
    elif "max_chars" in arguments:
        repaired["max_bytes"] = arguments["max_chars"]
    return repaired


def _looks_like_file_path(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False

    if candidate.startswith(("./", "../", "/", "~")):
        return True
    if "\\" in candidate or "/" in candidate:
        return True
    if "." in candidate:
        return True
    return False


def _type_matches(expected: str, value: Any) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def _coerce_value(expected: str | None, value: Any) -> Any:
    if expected == "boolean" and isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        return value

    if expected == "string" and isinstance(value, (int, float, bool)):
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    return value
