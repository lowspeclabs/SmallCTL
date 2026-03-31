from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from ..logging_utils import RunLogger, log_kv
from ..models.tool_result import ToolEnvelope
from .registry import ToolRegistry

Tier2Adapter = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


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
        phase: str = "explore",
        ansible_check_mode_in_plan: bool = True,
        tier2_adapter: Tier2Adapter | None = None,
        run_logger: RunLogger | None = None,
    ) -> None:
        self.log = logging.getLogger("smallctl.dispatcher")
        self.run_logger = run_logger
        self.registry = registry
        self.phase = phase
        self.ansible_check_mode_in_plan = ansible_check_mode_in_plan
        self.tier2_adapter = tier2_adapter

    async def dispatch(self, tool_name: str, arguments: dict[str, Any]) -> ToolEnvelope:
        log_kv(
            self.log,
            logging.INFO,
            "tool_dispatch_start",
            tool_name=tool_name,
            phase=self.phase,
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
                metadata=result.get("metadata") or {},
            )
        return ToolEnvelope(
            success=True,
            output=result,
            metadata={"tool_name": tool_name, "tier": spec.tier},
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
