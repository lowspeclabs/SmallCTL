from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from ..logging_utils import RunLogger, log_kv
from ..models.tool_result import ToolEnvelope
from ..remote_scope import has_single_confirmed_ssh_target, remote_scope_is_active
from . import network
from .registry import ToolRegistry

_ARTIFACT_TOKEN_RE = re.compile(r"\bA\d+\b", re.IGNORECASE)
_WRITE_SESSION_ARTIFACT_ID_RE = re.compile(
    r"^ws[-_A-Za-z0-9]+(?:__[^/\s]+)*__stage(?:\.[A-Za-z0-9]+)?$",
    re.IGNORECASE,
)
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
_SSH_PASSWORD_TASK_PATTERNS = (
    re.compile(r'\bpassword\s*(?:is\s+|=|:)?\s*"(?P<password>[^"\r\n]+)"', re.IGNORECASE),
    re.compile(r"\bpassword\s*(?:is\s+|=|:)?\s*'(?P<password>[^'\r\n]+)'", re.IGNORECASE),
    re.compile(r"\bpassword\s*(?:is\s+|=|:)?\s+(?P<password>[^\s,;]+)", re.IGNORECASE),
)
_SSH_PASSWORD_INVALID_TOKENS = {
    "authentication",
    "auth",
    "enabled",
    "required",
    "prompt",
    "prompted",
}
_TOOL_ALIAS_REPAIRS = {
    "use_shell_exec": "shell_exec",
    "use_ssh_exec": "ssh_exec",
    "artifact_write": "file_write",
}
_SSH_FILE_TOOLS = {"ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}
_WRITE_SESSION_PATH_REPAIR_TOOLS = {"file_write", "file_append", "file_patch", "ast_patch"}
_REMOTE_GUARDED_FILE_TOOLS = {"dir_list", "file_read", "file_write", "file_patch", "ast_patch"}
_SSH_TASK_TARGET_RE = re.compile(
    r"\b(?:ssh|scp|sftp)\s+(?:[A-Za-z0-9._-]+@)?(?P<host>[A-Za-z0-9._-]+)\b",
    re.IGNORECASE,
)
_AT_HOST_TARGET_RE = re.compile(r"\b[A-Za-z0-9._-]+@(?P<host>[A-Za-z0-9._-]+)\b", re.IGNORECASE)
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_REMOTE_TASK_HINT_RE = re.compile(r"\b(?:remote|ssh|username|password|server|host)\b", re.IGNORECASE)
_REMOTE_COMMAND_PATH_RE = re.compile(
    r"(?<![\w/])/(?:(?:[A-Za-z0-9._-]+/)*[A-Za-z0-9._-]+(?:\.[A-Za-z0-9._-]+)?)"
)
_REMOTE_INFRA_PROBE_RE = re.compile(
    r"\b(?:which|command\s+-v|type|whereis)\s+nginx\b"
    r"|"
    r"\bnginx\s+-[A-Za-z]"
    r"|"
    r"\b(?:systemctl|service)\s+(?:status|is-active|is-enabled|reload|restart|start|stop)\s+nginx\b",
    re.IGNORECASE,
)
_RAW_SSH_SHELL_RE = re.compile(r"^\s*(?:ssh\b|scp\b|sftp\b|sshpass\b)", re.IGNORECASE)
_REMOTE_ABSOLUTE_PATH_PREFIXES = (
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/media",
    "/mnt",
    "/opt",
    "/proc",
    "/root",
    "/run",
    "/srv",
    "/sys",
    "/usr",
    "/var",
)
_WEB_FETCH_ARTIFACT_RESULT_ALIAS_RE = re.compile(r"^(A\d+)\s*[-:_#/]\s*(\d+)$", re.IGNORECASE)
_WEB_FETCH_ORDINAL_RESULT_ALIAS_RE = re.compile(
    r"^(?:result|res|rank|item)?\s*[-#: ]?\s*(\d+)$",
    re.IGNORECASE,
)
_STAGED_CONTROL_TOOLS = {"loop_status", "step_complete", "step_fail", "ask_human"}
_SSH_AUTH_RECOVERY_KEY = "_ssh_auth_recovery_state"


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
        run_logger: RunLogger | None = None,
    ) -> None:
        self.log = logging.getLogger("smallctl.dispatcher")
        self.run_logger = run_logger
        self.registry = registry
        self.state = state
        self.phase = phase

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
        staged_allowlist_error = _staged_tool_allowlist_error(self.state, tool_name)
        if staged_allowlist_error is not None:
            if self.run_logger:
                self.run_logger.log(
                    "tools",
                    "staged_tool_rejected",
                    "staged tool rejected",
                    **staged_allowlist_error.metadata,
                )
            log_kv(
                self.log,
                logging.WARNING,
                "staged_tool_rejected",
                **staged_allowlist_error.metadata,
            )
            return staged_allowlist_error
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
            if tool_name == "task_complete" and self.phase == "repair":
                error = (
                    f"Tool '{tool_name}' is not allowed in phase 'repair'. "
                    "The last verifier run still shows a non-zero exit code. "
                    "Fix the failing command first (re-run it and achieve exit_code 0), "
                    "then call task_complete."
                )
            else:
                error = f"Tool '{tool_name}' is not allowed in phase '{self.phase}'"
            return ToolEnvelope(
                success=False,
                error=error,
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

        try:
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


def _staged_tool_allowlist_error(state: Any | None, tool_name: str) -> ToolEnvelope | None:
    if state is None:
        return None
    if not bool(getattr(state, "plan_execution_mode", False)):
        return None
    active_step_id = str(getattr(state, "active_step_id", "") or "").strip()
    if not active_step_id:
        return None

    plan = getattr(state, "active_plan", None) or getattr(state, "draft_plan", None)
    step = plan.find_step(active_step_id) if plan is not None and hasattr(plan, "find_step") else None
    plan_id = str(getattr(plan, "plan_id", "") or "")
    active_step_run_id = str(getattr(state, "active_step_run_id", "") or "")
    attempt = 1
    if step is not None:
        attempt = int(getattr(step, "retry_count", 0) or 0) + 1

    if tool_name in _STAGED_CONTROL_TOOLS:
        return None
    if tool_name == "task_complete":
        return ToolEnvelope(
            success=False,
            error="`task_complete` is not allowed during staged execution. Use `step_complete` for the active step.",
            metadata={
                "reason": "task_complete_blocked_in_staged_execution",
                "tool_name": tool_name,
                "plan_id": plan_id,
                "step_id": active_step_id,
                "step_run_id": active_step_run_id,
                "attempt": attempt,
            },
        )

    allowed = set(getattr(step, "tool_allowlist", []) or []) if step is not None else set()
    if tool_name in allowed:
        return None
    return ToolEnvelope(
        success=False,
        error=f"Tool `{tool_name}` is not allowed for active staged step `{active_step_id}`.",
        metadata={
            "reason": "tool_not_allowed_for_step",
            "tool_name": tool_name,
            "allowed_tools": sorted(allowed),
            "plan_id": plan_id,
            "step_id": active_step_id,
            "step_run_id": active_step_run_id,
            "attempt": attempt,
        },
    )


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
    elif tool_name == "web_fetch":
        arguments, web_fetch_metadata = _normalize_web_fetch_request(arguments, state=state)
        normalization_metadata.update(web_fetch_metadata)

    arguments, write_session_metadata = _repair_write_session_path_from_state(
        tool_name,
        arguments,
        state=state,
    )
    normalization_metadata.update(write_session_metadata)
    ssh_available = _ssh_exec_available(registry, phase=phase, state=state)

    remote_file_guard = _guard_remote_file_tool_request(
        tool_name,
        arguments,
        state=state,
        ssh_available=ssh_available,
    )
    if remote_file_guard is not None:
        return tool_name, arguments, remote_file_guard, normalization_metadata

    if tool_name in _SSH_FILE_TOOLS:
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

    if tool_name == "ssh_exec":
        try:
            normalized_arguments = network.normalize_ssh_arguments(arguments)
            normalized_arguments, ssh_metadata = _recover_ssh_arguments_from_task_context(
                normalized_arguments,
                state=state,
            )
            normalization_metadata.update(ssh_metadata)
            auth_recovery_error, auth_recovery_metadata = _guard_ssh_auth_recovery(
                normalized_arguments,
                state=state,
            )
            normalization_metadata.update(auth_recovery_metadata)
            if auth_recovery_error is not None:
                return tool_name, normalized_arguments, auth_recovery_error, normalization_metadata
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
    ssh_spec_phase_available = not (
        ssh_spec is None or (phase and callable(phase_allowed) and not phase_allowed(phase))
    )

    command = str(arguments.get("command", "") or "").strip() if isinstance(arguments, dict) else ""
    if not command:
        return tool_name, arguments, None, normalization_metadata

    rewritten_args = None
    if ssh_spec_phase_available:
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
    raw_ssh_shell_attempt = _looks_like_raw_ssh_shell_command(command)
    if rewritten_args is None:
        if raw_ssh_shell_attempt:
            return (
                tool_name,
                arguments,
                _raw_ssh_shell_block_envelope(command, ssh_available=ssh_available),
                normalization_metadata,
            )
        remote_shell_guard = _guard_remote_shell_tool_request(
            command,
            state=state,
            ssh_available=ssh_available,
        )
        if remote_shell_guard is not None:
            return tool_name, arguments, remote_shell_guard, normalization_metadata
        if _task_clearly_targets_remote_ssh_host(state) and not re.search(r"\b(?:ssh|scp|sftp)\b", command):
            metadata = {
                "tool_name": tool_name,
                "reason": "remote_task_requires_ssh_exec",
            }
            if ssh_available:
                metadata["suggested_tool"] = "ssh_exec"
                error = "This is a remote task. Use `ssh_exec`, not local `shell_exec`."
            else:
                error = (
                    "This is a remote task, but `ssh_exec` is not currently available. "
                    "Resume with the network/SSH tool profile or ask for help instead of using local `shell_exec`."
                )
            return (
                tool_name,
                arguments,
                ToolEnvelope(
                    success=False,
                    error=error,
                    metadata=metadata,
                ),
                normalization_metadata,
            )
        return tool_name, arguments, None, normalization_metadata

    rewritten_args, ssh_metadata = _recover_ssh_arguments_from_task_context(
        rewritten_args,
        state=state,
    )
    normalization_metadata.update(ssh_metadata)
    auth_recovery_error, auth_recovery_metadata = _guard_ssh_auth_recovery(
        rewritten_args,
        state=state,
    )
    normalization_metadata.update(auth_recovery_metadata)
    if auth_recovery_error is not None:
        return "ssh_exec", rewritten_args, auth_recovery_error, normalization_metadata
    if "timeout_sec" in arguments and "timeout_sec" not in rewritten_args:
        rewritten_args["timeout_sec"] = arguments["timeout_sec"]
    normalization_metadata.update(
        {
            "rewritten_from_tool": "shell_exec",
            "routing_reason": "ssh_shell_command",
        }
    )
    return "ssh_exec", rewritten_args, None, normalization_metadata


def _repair_write_session_path_from_state(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if tool_name not in _WRITE_SESSION_PATH_REPAIR_TOOLS or not isinstance(arguments, dict):
        return arguments, {}
    if _value_present(arguments.get("path")):
        return arguments, {}

    session = getattr(state, "write_session", None)
    if session is None:
        return arguments, {}
    if str(getattr(session, "status", "") or "").strip().lower() == "complete":
        return arguments, {}

    requested_session_id = str(arguments.get("write_session_id") or "").strip()
    active_session_id = str(getattr(session, "write_session_id", "") or "").strip()
    target_path = str(getattr(session, "write_target_path", "") or "").strip()
    if not requested_session_id or not active_session_id or requested_session_id != active_session_id:
        return arguments, {}
    if not target_path:
        return arguments, {}

    repaired = dict(arguments)
    repaired["path"] = target_path
    return repaired, {
        "argument_repair": "active_write_session_path",
        "repaired_write_session_path": True,
        "write_session_id": active_session_id,
        "target_path": target_path,
    }


def _value_present(value: Any) -> bool:
    return value is not None and (not isinstance(value, str) or bool(value.strip()))


def _looks_like_raw_ssh_shell_command(command: str) -> bool:
    return bool(_RAW_SSH_SHELL_RE.match(str(command or "").strip()))


def _ssh_auth_recovery_entry_key(host: str, user: str) -> str:
    normalized_host = str(host or "").strip().lower()
    normalized_user = str(user or "").strip().lower()
    return f"{normalized_user}@{normalized_host}" if normalized_user else normalized_host


def _password_fingerprint(password: str) -> str:
    value = str(password or "").strip()
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _guard_ssh_auth_recovery(
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[ToolEnvelope | None, dict[str, Any]]:
    if not isinstance(arguments, dict) or state is None:
        return None, {}
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None, {}
    recovery_state = scratchpad.get(_SSH_AUTH_RECOVERY_KEY)
    if not isinstance(recovery_state, dict):
        return None, {}
    host = str(arguments.get("host") or "").strip().lower()
    user = str(arguments.get("user") or "").strip()
    if not host:
        return None, {}
    record = recovery_state.get(_ssh_auth_recovery_entry_key(host, user))
    if not isinstance(record, dict):
        return None, {}

    password = str(arguments.get("password") or "").strip()
    prior_fingerprint = str(record.get("password_fingerprint") or "").strip()
    current_fingerprint = _password_fingerprint(password)
    password_retry_allowed = bool(password) and (
        not prior_fingerprint or current_fingerprint != prior_fingerprint or not bool(record.get("password_provided"))
    )
    metadata = {
        "ssh_auth_recovery_required": True,
        "ssh_auth_recovery_failure_count": int(record.get("failure_count") or 0),
    }
    if password_retry_allowed:
        metadata["ssh_auth_recovery_branch"] = "retry_with_password"
        return None, metadata

    required_arguments = {
        "host": host,
        "command": str(arguments.get("command") or "").strip(),
    }
    if user:
        required_arguments["user"] = user
    error = (
        "SSH authentication previously failed for this target. Next step must be exactly one of: "
        "retry `ssh_exec` with a corrected `password`, call `ask_human` for corrected credentials, "
        "or stop with `task_fail`. Do not retry key-only auth and do not use raw `shell_exec` SSH."
    )
    return ToolEnvelope(
        success=False,
        error=error,
        metadata={
            "tool_name": "ssh_exec",
            "reason": "ssh_auth_recovery_required",
            "last_error": str(record.get("last_error") or "").strip(),
            "last_command": str(record.get("last_command") or "").strip(),
            "next_required_action": {
                "tool_names": ["ssh_exec", "ask_human", "task_fail"],
                "required_arguments": required_arguments,
                "notes": [
                    "If you retry ssh_exec, include a corrected password.",
                    "If you do not have corrected credentials, ask the user instead of improvising with shell_exec.",
                ],
            },
        },
    ), metadata


def _raw_ssh_shell_block_envelope(command: str, *, ssh_available: bool) -> ToolEnvelope:
    if ssh_available:
        error = (
            "Raw `ssh`/`scp`/`sftp` shell commands are not allowed here. "
            "Use canonical `ssh_exec` for remote commands or `ssh_file_read` / `ssh_file_write` / "
            "`ssh_file_patch` / `ssh_file_replace_between` for remote file operations."
        )
    else:
        error = (
            "Raw `ssh`/`scp`/`sftp` shell commands are blocked, and canonical SSH tools are not currently available. "
            "Resume with the network/SSH tool profile or ask for help instead of using local `shell_exec`."
        )
    return ToolEnvelope(
        success=False,
        error=error,
        metadata={
            "tool_name": "shell_exec",
            "reason": "raw_ssh_shell_blocked",
            "suggested_tools": [
                "ssh_exec",
                "ssh_file_read",
                "ssh_file_write",
                "ssh_file_patch",
                "ssh_file_replace_between",
            ],
            "command": command,
        },
    )


def _ssh_exec_available(registry: Any, *, phase: str | None, state: Any | None) -> bool:
    get_tool = getattr(registry, "get", None)
    if not callable(get_tool):
        return False
    ssh_spec = get_tool("ssh_exec")
    if ssh_spec is None:
        return False
    phase_allowed = getattr(ssh_spec, "phase_allowed", None)
    if callable(phase_allowed) and phase and not phase_allowed(phase):
        return False
    profile_allowed = getattr(ssh_spec, "profile_allowed", None)
    profiles = set(getattr(state, "active_tool_profiles", []) or [])
    if callable(profile_allowed) and not profile_allowed(profiles):
        return False
    return True


def _guard_remote_file_tool_request(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
    ssh_available: bool = True,
) -> ToolEnvelope | None:
    if tool_name not in _REMOTE_GUARDED_FILE_TOOLS or not isinstance(arguments, dict):
        return None

    path = str(arguments.get("path") or "").strip()
    if not path or not _looks_like_remote_absolute_path(path, state=state):
        return None

    if not (_remote_scope_is_active(state) or _has_single_confirmed_ssh_target(state)):
        return None

    suggested_tool = _suggested_remote_file_tool(tool_name, state=state)
    metadata = {
        "tool_name": tool_name,
        "reason": "remote_path_requires_ssh_exec"
        if suggested_tool == "ssh_exec"
        else "remote_path_requires_typed_ssh_file_tool",
        "path": path,
    }
    if ssh_available:
        metadata["suggested_tool"] = suggested_tool
        error = f"This path appears to be on the remote host. Use `{suggested_tool}`, not local `{tool_name}`."
    else:
        error = (
            f"This path appears to be on the remote host, but SSH tools are not currently available. "
            "Resume with the network/SSH tool profile or ask for help instead of using a local file tool."
        )
    return ToolEnvelope(success=False, error=error, metadata=metadata)


def _suggested_remote_file_tool(tool_name: str, *, state: Any | None = None) -> str:
    if tool_name == "file_read":
        return "ssh_file_read"
    if tool_name == "file_write":
        return "ssh_file_write"
    if tool_name == "file_patch":
        task_text = " ".join(_ssh_task_context_texts(state)).lower() if state is not None else ""
        if any(marker in task_text for marker in ("style block", "<style>", "between ", "bounded block", "inline style")):
            return "ssh_file_replace_between"
        return "ssh_file_patch"
    return "ssh_exec"


def _guard_remote_shell_tool_request(
    command: str,
    *,
    state: Any | None = None,
    ssh_available: bool = True,
) -> ToolEnvelope | None:
    if not command:
        return None
    if not (_remote_scope_is_active(state) or _has_single_confirmed_ssh_target(state)):
        return None

    if _command_mentions_remote_absolute_path(command, state=state):
        metadata = {
            "tool_name": "shell_exec",
            "reason": "remote_path_requires_ssh_exec",
            "command": command,
        }
        if ssh_available:
            metadata["suggested_tool"] = "ssh_exec"
            error = "This command references remote-looking host paths. Use `ssh_exec`, not local `shell_exec`."
        else:
            error = (
                "This command references remote-looking host paths, but `ssh_exec` is not currently available. "
                "Resume with the network/SSH tool profile or ask for help instead of running it locally."
            )
        return ToolEnvelope(
            success=False,
            error=error,
            metadata=metadata,
        )

    if _looks_like_remote_infrastructure_probe_command(command):
        metadata = {
            "tool_name": "shell_exec",
            "reason": "remote_task_requires_ssh_exec",
            "command": command,
        }
        if ssh_available:
            metadata["suggested_tool"] = "ssh_exec"
            error = "This infrastructure check must run on the remote host. Use `ssh_exec`, not local `shell_exec`."
        else:
            error = (
                "This infrastructure check must run on the remote host, but `ssh_exec` is not currently available. "
                "Resume with the network/SSH tool profile or ask for help instead of running it locally."
            )
        return ToolEnvelope(
            success=False,
            error=error,
            metadata=metadata,
        )
    return None


def _recover_ssh_arguments_from_task_context(
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(arguments, dict):
        return arguments, {}

    repaired = dict(arguments)
    metadata: dict[str, Any] = {}
    host = str(repaired.get("host") or "").strip()
    user = str(repaired.get("user") or "").strip()
    password = str(repaired.get("password") or "").strip()
    command = str(repaired.get("command") or "").strip()
    password_source = "explicit" if password else "none"

    if host and not user:
        inferred_user = _infer_ssh_user_from_state_context(host, state=state)
        user_source = "task_context"
        if not inferred_user:
            inferred_user = _infer_ssh_user_from_execution_records(host, state=state)
            user_source = "prior_ssh_exec"
        if not inferred_user:
            inferred_user = _infer_ssh_user_from_session_memory(host, state=state)
            user_source = "session_memory"
        if inferred_user:
            repaired["user"] = inferred_user
            user = inferred_user
            metadata.update(
                {
                    "recovered_ssh_user": inferred_user,
                    "recovered_ssh_user_source": user_source,
                    "routing_reason": "ssh_task_context_user_recovery",
                }
            )

    if host and not password:
        inferred_password, password_source = _infer_ssh_password(
            host,
            user=user,
            state=state,
        )
        if inferred_password:
            repaired["password"] = inferred_password
            password = inferred_password
            metadata.update(
                {
                    "recovered_ssh_password": True,
                    "recovered_ssh_password_source": password_source,
                }
            )
            metadata["routing_reason"] = metadata.get("routing_reason") or f"ssh_password_recovery_{password_source}"

    if not command and _task_requests_ssh_connection_probe(state):
        repaired["command"] = "whoami"
        metadata["recovered_ssh_command"] = "whoami"
        metadata["routing_reason"] = metadata.get("routing_reason") or "ssh_connection_probe_recovery"

    metadata.update(_ssh_auth_debug_metadata(repaired, password_source=password_source))
    return repaired, metadata


def _ssh_auth_debug_metadata(
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


def _infer_ssh_user_from_execution_records(host: str, *, state: Any | None = None) -> str:
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
        if not _ssh_record_likely_authenticated(record):
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


def _infer_ssh_user_from_session_memory(host: str, *, state: Any | None = None) -> str:
    target = _session_ssh_target_record(host, state=state)
    return str(target.get("user") or "").strip() if isinstance(target, dict) else ""


def _infer_ssh_password(
    host: str,
    *,
    user: str | None = None,
    state: Any | None = None,
) -> tuple[str, str]:
    inferred_from_records = _infer_ssh_password_from_execution_records(host, user=user, state=state)
    if inferred_from_records:
        return inferred_from_records, "prior_ssh_exec"

    inferred_from_task = _infer_ssh_password_from_state_context(host, user=user, state=state)
    if inferred_from_task:
        return inferred_from_task, "task_context"

    inferred_from_session = _infer_ssh_password_from_session_memory(host, user=user, state=state)
    if inferred_from_session:
        return inferred_from_session, "session_memory"

    return "", ""


def _infer_ssh_password_from_execution_records(
    host: str,
    *,
    user: str | None = None,
    state: Any | None = None,
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
        if not _ssh_record_likely_authenticated(record):
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

        password = str(args.get("password") or "").strip()
        if password:
            return password
    return ""


def _ssh_record_likely_authenticated(record: dict[str, Any]) -> bool:
    result = record.get("result")
    if not isinstance(result, dict):
        return False
    if bool(result.get("success")):
        return True
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        return False
    return bool(metadata.get("ssh_transport_succeeded")) or str(metadata.get("failure_kind") or "").strip() == "remote_command"


def _infer_ssh_password_from_state_context(
    host: str,
    *,
    user: str | None = None,
    state: Any | None = None,
) -> str:
    target_host = str(host or "").strip().lower()
    target_user = str(user or "").strip().lower()
    if not target_host or state is None:
        return ""

    for text in _ssh_task_context_texts(state):
        if not _text_mentions_ssh_target(text, host=target_host, user=target_user):
            continue
        for pattern in _SSH_PASSWORD_TASK_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            candidate = str(match.group("password") or "").strip()
            if _looks_like_ssh_password(candidate):
                return candidate
    return ""


def _infer_ssh_password_from_session_memory(
    host: str,
    *,
    user: str | None = None,
    state: Any | None = None,
) -> str:
    target = _session_ssh_target_record(host, state=state)
    if not isinstance(target, dict):
        return ""
    target_user = str(user or "").strip().lower()
    session_user = str(target.get("user") or "").strip().lower()
    if target_user and session_user and target_user != session_user:
        return ""
    return str(target.get("password") or "").strip()


def _session_ssh_target_record(host: str, *, state: Any | None = None) -> dict[str, Any]:
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


def _text_mentions_ssh_target(text: str, *, host: str, user: str | None = None) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    if host and host in lowered:
        return True
    normalized_user = str(user or "").strip().lower()
    return bool(host and normalized_user and f"{normalized_user}@{host}" in lowered)


def _looks_like_ssh_password(candidate: str) -> bool:
    stripped = str(candidate or "").strip()
    if not stripped:
        return False
    return stripped.lower() not in _SSH_PASSWORD_INVALID_TOKENS


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


def _task_is_remote_execute(state: Any | None) -> bool:
    return str(getattr(state, "task_mode", "") or "").strip().lower() == "remote_execute"


def _remote_scope_is_active(state: Any | None) -> bool:
    return remote_scope_is_active(state)


def _has_single_confirmed_ssh_target(state: Any | None) -> bool:
    return has_single_confirmed_ssh_target(state)


def _looks_like_remote_absolute_path(path: str, *, state: Any | None = None) -> bool:
    candidate = str(path or "").strip()
    if not candidate.startswith("/"):
        return False

    cwd = str(getattr(state, "cwd", "") or "").rstrip("/")
    if cwd and (candidate == cwd or candidate.startswith(cwd + "/")):
        return False
    if candidate == "/tmp" or candidate.startswith("/tmp/"):
        return False

    if candidate in _REMOTE_ABSOLUTE_PATH_PREFIXES:
        return True
    return candidate.startswith(tuple(prefix + "/" for prefix in _REMOTE_ABSOLUTE_PATH_PREFIXES))


def _command_mentions_remote_absolute_path(command: str, *, state: Any | None = None) -> bool:
    for match in _REMOTE_COMMAND_PATH_RE.finditer(str(command or "")):
        if _looks_like_remote_absolute_path(match.group(0), state=state):
            return True
    return False


def _looks_like_remote_infrastructure_probe_command(command: str) -> bool:
    return _REMOTE_INFRA_PROBE_RE.search(str(command or "").strip()) is not None


def _task_requests_ssh_connection_probe(state: Any | None) -> bool:
    if state is None:
        return False

    for text in _ssh_task_context_texts(state):
        lowered = str(text or "").strip().lower()
        if not lowered:
            continue
        if any(
            marker in lowered
            for marker in (
                "ssh into ",
                "ssh to ",
                "ssh in to ",
                "log into ",
                "login to ",
                "connect to ",
            )
        ) and "?" not in lowered:
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


def _normalize_web_fetch_request(
    arguments: dict[str, Any],
    *,
    state: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(arguments, dict):
        return arguments, {}

    repaired = dict(arguments)
    metadata: dict[str, Any] = {}

    raw_fetch_id = repaired.get("fetch_id")
    normalized_fetch_id = str(raw_fetch_id).strip() if isinstance(raw_fetch_id, str) else ""
    raw_result_id = repaired.get("result_id")
    normalized_existing_result_id = str(raw_result_id).strip() if isinstance(raw_result_id, str) else ""
    if normalized_fetch_id and not normalized_existing_result_id:
        repaired["result_id"] = normalized_fetch_id
        repaired.pop("fetch_id", None)
        metadata.update(
            {
                "field_alias_repair": "web_fetch_fetch_id_to_result_id",
                "original_fetch_id": normalized_fetch_id,
            }
        )
    elif normalized_fetch_id and normalized_existing_result_id == normalized_fetch_id:
        repaired.pop("fetch_id", None)
        metadata.update(
            {
                "field_alias_repair": "web_fetch_fetch_id_to_result_id",
                "original_fetch_id": normalized_fetch_id,
            }
        )

    raw_result_id = repaired.get("result_id")
    if not isinstance(raw_result_id, str):
        return repaired, metadata

    normalized_result_id = str(raw_result_id).strip()
    if not normalized_result_id:
        return repaired, metadata

    known_results = _web_result_index(state)
    if normalized_result_id in known_results:
        return repaired, metadata

    resolved_result_id, alias_metadata = _resolve_web_fetch_result_alias(
        normalized_result_id,
        state=state,
    )
    if not resolved_result_id:
        return repaired, metadata

    repaired["result_id"] = resolved_result_id
    metadata.update(alias_metadata)
    return repaired, metadata


def _normalize_artifact_reference(
    value: str,
    *,
    field_name: str,
    state: Any | None = None,
) -> tuple[str | None, dict[str, Any]]:
    canonical_key = _canonical_artifact_key(value, state=state)
    if canonical_key is not None:
        return canonical_key, {
            "argument_repair": "artifact_read_alias_to_existing_key",
            "repair_field": field_name,
            "resolved_artifact_id": canonical_key,
        }

    normalized = _normalize_recent_artifact_alias(value, state=state)
    if normalized is not None:
        return normalized, {
            "argument_repair": "artifact_read_recent_fallback",
            "repair_field": field_name,
            "resolved_artifact_id": normalized,
        }

    if _looks_like_write_session_artifact_alias(value):
        preserved = str(value or "").strip()
        return preserved, {
            "argument_repair": "artifact_read_preserve_write_session_alias",
            "repair_field": field_name,
            "resolved_artifact_id": preserved,
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


def _canonical_artifact_key(candidate: str, *, state: Any | None = None) -> str | None:
    normalized = str(candidate or "").strip()
    if not normalized:
        return None

    artifacts = getattr(state, "artifacts", None)
    if not isinstance(artifacts, dict) or not artifacts:
        return None

    if normalized in artifacts:
        return normalized

    lowered = normalized.lower()
    for artifact_id in artifacts.keys():
        if isinstance(artifact_id, str) and artifact_id.lower() == lowered:
            return artifact_id
    return None


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


def _looks_like_write_session_artifact_alias(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    return _WRITE_SESSION_ARTIFACT_ID_RE.fullmatch(candidate) is not None


def _resolve_web_fetch_result_alias(
    value: str,
    *,
    state: Any | None = None,
) -> tuple[str | None, dict[str, Any]]:
    normalized = str(value or "").strip()
    if not normalized:
        return None, {}

    artifact_match = _WEB_FETCH_ARTIFACT_RESULT_ALIAS_RE.fullmatch(normalized)
    if artifact_match is not None:
        artifact_id = _canonical_artifact_id(artifact_match.group(1), state=state)
        rank = int(artifact_match.group(2))
        resolved = _web_result_id_for_rank(state, artifact_id=artifact_id, rank=rank)
        if resolved:
            return resolved, {
                "argument_repair": "web_fetch_result_alias_to_search_result",
                "repair_field": "result_id",
                "original_result_id": normalized,
                "resolved_result_id": resolved,
                "resolved_search_artifact_id": artifact_id,
                "resolved_search_result_rank": rank,
            }

    canonical_artifact = _canonical_artifact_key(normalized, state=state)
    if canonical_artifact and canonical_artifact.upper().startswith("A"):
        resolved = _web_result_id_for_rank(state, artifact_id=canonical_artifact, rank=1)
        if resolved:
            return resolved, {
                "argument_repair": "web_fetch_result_alias_to_search_result",
                "repair_field": "result_id",
                "original_result_id": normalized,
                "resolved_result_id": resolved,
                "resolved_search_artifact_id": canonical_artifact,
                "resolved_search_result_rank": 1,
            }

    ordinal_match = _WEB_FETCH_ORDINAL_RESULT_ALIAS_RE.fullmatch(normalized)
    if ordinal_match is not None:
        rank = int(ordinal_match.group(1))
        resolved = _web_result_id_for_rank(state, artifact_id=None, rank=rank)
        if resolved:
            artifact_id = _most_recent_web_search_artifact_id(state)
            metadata = {
                "argument_repair": "web_fetch_result_alias_to_search_result",
                "repair_field": "result_id",
                "original_result_id": normalized,
                "resolved_result_id": resolved,
                "resolved_search_result_rank": rank,
            }
            if artifact_id:
                metadata["resolved_search_artifact_id"] = artifact_id
            return resolved, metadata

    return None, {}


def _web_result_id_for_rank(
    state: Any | None,
    *,
    artifact_id: str | None,
    rank: int,
) -> str | None:
    if rank <= 0:
        return None

    result_ids: list[str] = []
    if artifact_id:
        artifact_map = _web_search_artifact_results(state)
        result_ids = list(artifact_map.get(artifact_id) or [])
        if not result_ids and artifact_id == _most_recent_web_search_artifact_id(state):
            result_ids = list(_last_web_search_result_ids(state))
    else:
        result_ids = list(_last_web_search_result_ids(state))

    if rank > len(result_ids):
        return None

    candidate = str(result_ids[rank - 1] or "").strip()
    if not candidate:
        return None

    known_results = _web_result_index(state)
    if known_results and candidate not in known_results:
        return None
    return candidate


def _web_result_index(state: Any | None) -> dict[str, Any]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    index = scratchpad.get("_web_result_index")
    return index if isinstance(index, dict) else {}


def _web_search_artifact_results(state: Any | None) -> dict[str, list[str]]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return {}
    mapping = scratchpad.get("_web_search_artifact_results")
    return mapping if isinstance(mapping, dict) else {}


def _last_web_search_result_ids(state: Any | None) -> list[str]:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return []
    result_ids = scratchpad.get("_web_last_search_result_ids")
    if not isinstance(result_ids, list):
        return []
    return [str(item).strip() for item in result_ids if str(item).strip()]


def _most_recent_web_search_artifact_id(state: Any | None) -> str | None:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return None
    artifact_id = str(scratchpad.get("_web_last_search_artifact_id") or "").strip()
    return artifact_id or None


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
