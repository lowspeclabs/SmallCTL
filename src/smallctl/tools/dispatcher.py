from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from ..logging_utils import RunLogger, log_kv
from ..models.tool_result import ToolEnvelope
from ..remote_scope import remote_scope_is_active
from ..state import json_safe_value
from ..challenge_progress import redundant_verifier_block
from . import network
from .dispatcher_normalization_flow import normalize_tool_request
from .registry import ToolRegistry
from .dispatcher_artifact_normalization import (
    normalize_artifact_read_request as _normalize_artifact_read_request,
    normalize_web_fetch_request as _normalize_web_fetch_request,
)
from .dispatcher_request_normalization import (
    normalize_initial_tool_request as _normalize_initial_tool_request,
    repair_ssh_exec_malformed_args as _repair_ssh_exec_malformed_args,
    repair_write_session_path_from_state as _repair_write_session_path_from_state,
)
from .dispatcher_remote_paths import (
    command_mentions_remote_absolute_path as _command_mentions_remote_absolute_path,
    looks_like_remote_absolute_path as _looks_like_remote_absolute_path,
    looks_like_remote_infrastructure_probe_command as _looks_like_remote_infrastructure_probe_command,
)
from .dispatcher_schema_helpers import (
    coerce_value as _coerce_value,
)
from .tool_call_repair import ToolCallValidationIssue, validate_tool_args
from .dispatcher_shell_guards import (
    guard_harness_tool_as_ssh_shell_command as _guard_harness_tool_as_ssh_shell_command,
    guard_nested_raw_ssh_in_ssh_exec as _guard_nested_raw_ssh_in_ssh_exec,
    looks_like_raw_ssh_shell_command as _looks_like_raw_ssh_shell_command,
    raw_ssh_shell_block_envelope as _raw_ssh_shell_block_envelope,
)
from .dispatcher_ssh_auth import (
    password_fingerprint as _password_fingerprint,
    ssh_auth_debug_metadata as _ssh_auth_debug_metadata,
    ssh_auth_recovery_entry_key as _ssh_auth_recovery_entry_key,
)
from .dispatcher_ssh_context import (
    infer_ssh_user_from_state_context as _infer_ssh_user_from_state_context,
    ssh_task_context_texts as _ssh_task_context_texts,
)
from .dispatcher_remote_detection import (
    task_clearly_targets_remote_ssh_host as _task_clearly_targets_remote_ssh_host,
    task_requests_ssh_connection_probe as _task_requests_ssh_connection_probe,
)
from .dispatcher_ssh_memory import (
    explicit_ssh_password_matches_current_user_context as _explicit_ssh_password_matches_current_user_context,
    infer_ssh_password as _infer_ssh_password,
    infer_ssh_password_from_execution_records as _infer_ssh_password_from_execution_records,
    infer_ssh_password_from_session_memory as _infer_ssh_password_from_session_memory,
    infer_ssh_user_from_execution_records as _infer_ssh_user_from_execution_records,
    infer_ssh_user_from_session_memory as _infer_ssh_user_from_session_memory,
    session_ssh_target_record as _session_ssh_target_record,
    ssh_record_likely_authenticated as _ssh_record_likely_authenticated,
)
from .dispatcher_policy_guards import (
    _fama_dispatch_block,
    _staged_tool_allowlist_error,
)
from .dispatcher_tool_guards import (
    _guard_remote_file_tool_request,
    _guard_remote_shell_tool_request,
    _guard_ssh_auth_recovery,
    _suggested_remote_file_tool,
)
from .dispatcher_tool_predicates import (
    _escalation_recommends_local_shell,
    _recent_ssh_auth_failure,
    _ssh_exec_available,
)
from .dispatcher_ssh_recovery import (
    _infer_ssh_host_from_context,
    _pin_and_guard_ssh_credentials,
    _recover_ssh_arguments_from_task_context,
)
from .dispatcher_scope_predicates import (
    _has_single_confirmed_ssh_target,
    _remote_scope_is_active,
    _task_is_remote_execute,
)

_SSH_FILE_TOOLS = {"ssh_file_read", "ssh_file_write", "ssh_file_patch", "ssh_file_replace_between"}
_REMOTE_GUARDED_FILE_TOOLS = {"dir_list", "file_read", "file_write", "file_patch", "ast_patch"}
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
        harness: Any | None = None,
    ) -> None:
        self.log = logging.getLogger("smallctl.dispatcher")
        self.run_logger = run_logger
        self.registry = registry
        self.state = state
        self.phase = phase
        self.harness = harness

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolEnvelope:
        requested_tool_name = tool_name
        if not isinstance(arguments, dict):
            return ToolEnvelope.make_error(
                tool_name,
                "Tool arguments must be an object.",
                validation_error="schema_validation",
                validation_issues=[
                    {
                        "path": [],
                        "kind": "type",
                        "expected": "object",
                        "actual": type(arguments).__name__,
                        "message": "tool arguments must be an object",
                    }
                ],
            )
        tool_name, arguments, intercepted_result, normalization_metadata = normalize_tool_request(
            self.registry,
            tool_name,
            arguments,
            phase=self.phase,
            state=self.state,
            harness=self.harness,
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
        blocked_by_fama = _fama_dispatch_block(tool_name, arguments, state=self.state, phase=self.phase)
        if blocked_by_fama is not None:
            if self.run_logger:
                self.run_logger.log(
                    "tools",
                    "fama_tool_call_blocked",
                    "FAMA blocked tool call",
                    tool_name=tool_name,
                    active_mitigation=blocked_by_fama.metadata.get("active_mitigation"),
                    mode=self.phase,
                )
            return blocked_by_fama
        blocked_by_challenge_progress = redundant_verifier_block(
            self.state,
            tool_name=tool_name,
            arguments=arguments,
        )
        if blocked_by_challenge_progress is not None:
            if self.run_logger:
                self.run_logger.log(
                    "tools",
                    "challenge_progress_tool_call_blocked",
                    "challenge progress policy blocked tool call",
                    tool_name=tool_name,
                    active_mitigation=blocked_by_challenge_progress.metadata.get("active_mitigation"),
                    reason=blocked_by_challenge_progress.metadata.get("reason"),
                )
            return blocked_by_challenge_progress
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
            error = f"Tool '{tool_name}' is not allowed in phase '{self.phase}'"
            return ToolEnvelope(
                success=False,
                error=error,
                metadata={"tool_name": tool_name, "phase": self.phase},
            )

        # Pre-dispatch validation for ssh_file_replace_between boundary overlap
        if tool_name == "ssh_file_replace_between":
            start_text = str(arguments.get("start_text") or "").strip()
            end_text = str(arguments.get("end_text") or "").strip()
            if start_text and end_text and end_text.startswith(start_text):
                validation_error = (
                    "Invalid ssh_file_replace_between: start_text is a prefix of end_text, "
                    "so there is no valid bounded region between them. "
                    "If you need to replace most of a small file, use ssh_file_write instead."
                )
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
                    metadata={
                        "tool_name": tool_name,
                        "validation_kind": "invalid_boundary_overlap",
                        "path": arguments.get("path"),
                        "host": arguments.get("host"),
                    },
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

        args, dropped_keys, coerced_entries = self._coerce_args(spec.schema, arguments)
        if (dropped_keys or coerced_entries) and self.run_logger:
            self.run_logger.log(
                "tools",
                "legacy_dispatch_coercion",
                "legacy dispatcher coercion adjusted arguments",
                tool_name=tool_name,
                dropped_keys=dropped_keys,
                coerced_entries=coerced_entries,
            )

        # Reject empty shell/ssh commands before dispatch
        if tool_name in {"shell_exec", "ssh_exec"} and isinstance(args, dict):
            cmd = str(args.get("command") or "").strip()
            if not cmd:
                validation_error = f"{tool_name} requires a non-empty command string"
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
                        arguments_preview=json_safe_value(args),
                    )
                return ToolEnvelope(
                    success=False,
                    error=validation_error,
                    metadata={"tool_name": tool_name, "validation_error": "empty_command"},
                )

        validation_issues = self._validate_arg_issues(spec.schema, args)
        validation_error = self._format_validation_error(validation_issues)
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
                    arguments_preview=json_safe_value(args),
                    validation_issues=self._serialize_validation_issues(validation_issues),
                )
            metadata = {
                "tool_name": tool_name,
                "validation_error": "schema_validation",
                "validation_issues": self._serialize_validation_issues(validation_issues),
            }
            if dropped_keys:
                metadata["ignored_arguments"] = dropped_keys
                required = spec.schema.get("required", [])
                missing = [f for f in required if f not in args]
                if missing:
                    validation_error += f" (Ignored unknown parameters: {', '.join(dropped_keys)})"
            if coerced_entries:
                metadata["coerced_arguments"] = coerced_entries
            return ToolEnvelope(
                success=False,
                error=validation_error,
                metadata=metadata,
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
            result_metadata = {
                **dispatch_metadata,
                **normalization_metadata,
                **self._legacy_coercion_metadata(dropped_keys, coerced_entries),
                **result_metadata,
            }
            result_output = result.get("output")
            if result_output is None and tool_name in {"shell_exec", "ssh_exec"}:
                metadata_output = result_metadata.get("output")
                if isinstance(metadata_output, dict):
                    result_output = metadata_output
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
                    output=result_output,
                    error=result.get("error"),
            )
            return ToolEnvelope(
                success=bool(result["success"]),
                status=result.get("status"),
                output=result_output,
                error=result.get("error"),
                metadata=result_metadata,
            )
        return ToolEnvelope(
            success=True,
            output=result,
            metadata={**dispatch_metadata, **normalization_metadata, **self._legacy_coercion_metadata(dropped_keys, coerced_entries)},
        )

    @staticmethod
    def _coerce_args(
        schema: dict[str, Any], arguments: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
        if not isinstance(arguments, dict):
            return arguments, [], []

        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not properties and not required:
            return {}, [], []
        coerced = dict(arguments)
        dropped: list[str] = []
        coerced_entries: list[dict[str, Any]] = []
        if isinstance(properties, dict):
            dropped = [key for key in coerced if key not in properties]
            coerced = {key: value for key, value in coerced.items() if key in properties}
            for key, value in list(coerced.items()):
                expected_type = properties.get(key, {}).get("type")
                new_value = _coerce_value(expected_type, value)
                if new_value is not value:
                    coerced_entries.append(
                        {"key": key, "from": json_safe_value(value), "to": json_safe_value(new_value)}
                    )
                coerced[key] = new_value
        return coerced, dropped, coerced_entries

    @staticmethod
    def _validate_args(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
        return ToolDispatcher._format_validation_error(ToolDispatcher._validate_arg_issues(schema, arguments))

    @staticmethod
    def _validate_arg_issues(schema: dict[str, Any], arguments: Any) -> list[ToolCallValidationIssue]:
        return validate_tool_args(schema, arguments)

    @staticmethod
    def _format_validation_error(issues: list[ToolCallValidationIssue]) -> str | None:
        if not issues:
            return None
        issue = issues[0]
        path = ".".join(str(part) for part in issue.path)
        if issue.kind == "type" and not issue.path:
            return "Tool arguments must be an object."
        if issue.kind == "required":
            return f"Missing required field: {path}"
        if issue.kind == "type":
            return f"Field '{path}' expected type '{issue.expected}'"
        if issue.kind == "additional_property":
            return f"Unknown field: {path}"
        if issue.kind == "enum":
            return f"Field '{path}' expected one of {issue.expected}"
        return issue.message or f"Invalid field: {path}"

    @staticmethod
    def _serialize_validation_issues(issues: list[ToolCallValidationIssue]) -> list[dict[str, Any]]:
        return [
            {
                "path": [str(part) for part in issue.path],
                "kind": issue.kind,
                "expected": issue.expected,
                "actual": issue.actual,
                "message": issue.message,
            }
            for issue in issues
        ]

    @staticmethod
    def _legacy_coercion_metadata(dropped_keys: list[str], coerced_entries: list[dict[str, Any]]) -> dict[str, Any]:
        if not dropped_keys and not coerced_entries:
            return {}
        metadata: dict[str, Any] = {"legacy_dispatch_coercion": True}
        if dropped_keys:
            metadata["ignored_arguments"] = dropped_keys
        if coerced_entries:
            metadata["coerced_arguments"] = coerced_entries
        return metadata


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




