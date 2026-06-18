from __future__ import annotations

from typing import Any

from .backend_recovery import BackendRecoveryService


def _backend_recovery_service_for(harness: Any) -> BackendRecoveryService:
    service = getattr(harness, "_backend_recovery_service", None)
    if service is None:
        service = BackendRecoveryService(harness)
        try:
            setattr(harness, "_backend_recovery_service", service)
        except Exception:
            pass
    return service


def _backend_recovery(self: Any) -> BackendRecoveryService:
    return _backend_recovery_service_for(self)


async def recover_backend_wedge(self: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return await _backend_recovery_service_for(self).recover_backend_wedge(payload)


def _backend_restart_history(self: Any) -> list[float]:
    return _backend_recovery_service_for(self)._backend_restart_history()


def _check_backend_restart_rate_limit(self: Any) -> dict[str, Any]:
    return _backend_recovery_service_for(self)._check_backend_restart_rate_limit()


def _record_backend_restart_attempt(self: Any) -> None:
    _backend_recovery_service_for(self)._record_backend_restart_attempt()


async def _probe_backend_health(self: Any, health_url: str, *, timeout_sec: int) -> dict[str, Any]:
    return await _backend_recovery_service_for(self)._probe_backend_health(health_url, timeout_sec=timeout_sec)


async def _wait_for_backend_health(self: Any, health_url: str, *, timeout_sec: int) -> dict[str, Any]:
    return await _backend_recovery_service_for(self)._wait_for_backend_health(health_url, timeout_sec=timeout_sec)


async def _attempt_backend_restart_recovery(
    self: Any,
    health_url: str,
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    return await _backend_recovery_service_for(self)._attempt_backend_restart_recovery(
        health_url,
        timeout_sec=timeout_sec,
    )


async def _run_backend_restart_command(self: Any, command: str) -> dict[str, Any]:
    return await _backend_recovery_service_for(self)._run_backend_restart_command(command)


async def _run_backend_unload_command(self: Any, command: str | None) -> dict[str, Any]:
    return await _backend_recovery_service_for(self)._run_backend_unload_command(command)


def _backend_unload_action(self: Any) -> str:
    return _backend_recovery_service_for(self)._backend_unload_action()


def _backend_unload_message(self: Any, outcome: str) -> str:
    return _backend_recovery_service_for(self)._backend_unload_message(outcome)


async def _run_ollama_backend_unload(self: Any) -> dict[str, Any]:
    return await _backend_recovery_service_for(self)._run_ollama_backend_unload()


async def _run_lmstudio_backend_unload(self: Any) -> dict[str, Any]:
    return await _backend_recovery_service_for(self)._run_lmstudio_backend_unload()


def _find_lmstudio_loaded_instance_ids(self: Any, payload: dict[str, Any]) -> tuple[list[str], list[str]]:
    return _backend_recovery_service_for(self)._find_lmstudio_loaded_instance_ids(payload)


def bind_backend_recovery_facade(cls: type[Any]) -> None:
    cls._backend_recovery_service_for = staticmethod(_backend_recovery_service_for)
    cls._backend_recovery = _backend_recovery
    cls.recover_backend_wedge = recover_backend_wedge
    cls._backend_restart_history = _backend_restart_history
    cls._check_backend_restart_rate_limit = _check_backend_restart_rate_limit
    cls._record_backend_restart_attempt = _record_backend_restart_attempt
    cls._probe_backend_health = _probe_backend_health
    cls._wait_for_backend_health = _wait_for_backend_health
    cls._attempt_backend_restart_recovery = _attempt_backend_restart_recovery
    cls._run_backend_restart_command = _run_backend_restart_command
    cls._run_backend_unload_command = _run_backend_unload_command
    cls._backend_unload_action = _backend_unload_action
    cls._backend_unload_message = _backend_unload_message
    cls._run_ollama_backend_unload = _run_ollama_backend_unload
    cls._run_lmstudio_backend_unload = _run_lmstudio_backend_unload
    cls._find_lmstudio_loaded_instance_ids = _find_lmstudio_loaded_instance_ids
