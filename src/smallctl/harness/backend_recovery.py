from __future__ import annotations

import asyncio
import asyncio.subprocess
import time
from typing import Any


class BackendRecoveryService:
    def __init__(self, harness: Any):
        self.harness = harness

    async def recover_backend_wedge(self, payload: dict[str, Any]) -> dict[str, Any]:
        details = dict(payload.get("details") or {})
        health_url = self.harness.backend_healthcheck_url or f"{self.harness.client.base_url}/models"
        timeout_sec = max(1, int(self.harness.backend_healthcheck_timeout_sec))
        health_before = await self.harness._probe_backend_health(health_url, timeout_sec=timeout_sec)
        action = "none"
        status = "unrecovered"
        message = "Backend did not emit a first token before timeout."
        health_after = dict(health_before)
        restart_window: dict[str, Any] | None = None
        attempted_actions: list[str] = []
        unload_available = self.harness.backend_unload_command or self.harness.provider_profile in {"ollama", "lmstudio"}
        if health_before.get("ok") and unload_available:
            action = self.harness._backend_unload_action()
            attempted_actions.append(action)
            command_result = await self.harness._run_backend_unload_command(self.harness.backend_unload_command)
            if command_result.get("ok"):
                health_after = await self.harness._wait_for_backend_health(
                    health_url,
                    timeout_sec=max(timeout_sec, int(self.harness.backend_restart_grace_sec)),
                )
                if health_after.get("ok"):
                    status = "recovered"
                    message = self.harness._backend_unload_message("succeeded and health probe recovered")
                else:
                    message = self.harness._backend_unload_message("ran, but the health probe did not recover")
            else:
                health_after = {"ok": False}
                message = str(command_result.get("message") or "Backend unload command failed.")
            if status != "recovered" and self.harness.backend_restart_command:
                restart_result = await self.harness._attempt_backend_restart_recovery(
                    health_url,
                    timeout_sec=timeout_sec,
                )
                restart_action = str(restart_result.get("action") or "").strip()
                if restart_action == "restart_command":
                    attempted_actions.append(restart_action)
                restart_message = str(restart_result.get("message") or "").strip()
                if restart_message:
                    message = f"{message} {restart_message}".strip()
                action = restart_action or action
                status = str(restart_result.get("status") or status)
                if isinstance(restart_result.get("health_after"), dict):
                    health_after = dict(restart_result["health_after"])
                restart_window = restart_result.get("restart_window")
        elif self.harness.backend_restart_command:
            restart_result = await self.harness._attempt_backend_restart_recovery(
                health_url,
                timeout_sec=timeout_sec,
            )
            action = str(restart_result.get("action") or action)
            status = str(restart_result.get("status") or status)
            message = str(restart_result.get("message") or message)
            if action == "restart_command":
                attempted_actions.append(action)
            if isinstance(restart_result.get("health_after"), dict):
                health_after = dict(restart_result["health_after"])
            restart_window = restart_result.get("restart_window")
        else:
            if health_before.get("ok"):
                message = (
                    "Backend accepted health probes but appears wedged on generation; "
                    "no unload or restart recovery is configured."
                )
            else:
                message = "Backend health probe failed and no restart command is configured."
        result = {
            "status": status,
            "action": action,
            "message": message,
            "health_url": health_url,
            "health_before": health_before,
            "health_after": health_after,
            "reason": str(details.get("reason") or ""),
            "attempted_actions": attempted_actions,
        }
        if restart_window is not None:
            result["restart_window"] = restart_window
        self.harness.state.scratchpad["_last_backend_recovery"] = result
        self.harness._runlog(
            "backend_recovery",
            message,
            provider_profile=self.harness.provider_profile,
            status=status,
            action=action,
            health_url=health_url,
            health_before=health_before,
            health_after=health_after,
            details=details,
            attempted_actions=attempted_actions,
            restart_window=restart_window,
        )
        return result

    def _backend_restart_history(self) -> list[float]:
        history = self.harness.state.scratchpad.setdefault("_backend_restart_history", [])
        if not isinstance(history, list):
            history = []
            self.harness.state.scratchpad["_backend_restart_history"] = history
        return history

    def _check_backend_restart_rate_limit(self) -> dict[str, Any]:
        history = self._backend_restart_history()
        if self.harness.backend_max_restarts_per_hour <= 0:
            return {"allowed": False, "count": len(history), "window_sec": 3600}
        cutoff = time.time() - 3600.0
        recent = [float(ts) for ts in history if float(ts) >= cutoff]
        self.harness.state.scratchpad["_backend_restart_history"] = recent
        return {
            "allowed": len(recent) < self.harness.backend_max_restarts_per_hour,
            "count": len(recent),
            "window_sec": 3600,
        }

    def _record_backend_restart_attempt(self) -> None:
        history = self._backend_restart_history()
        cutoff = time.time() - 3600.0
        recent = [float(ts) for ts in history if float(ts) >= cutoff]
        recent.append(time.time())
        self.harness.state.scratchpad["_backend_restart_history"] = recent

    async def _probe_backend_health(self, health_url: str, *, timeout_sec: int) -> dict[str, Any]:
        try:
            import httpx
        except Exception as exc:
            return {"ok": False, "error": f"httpx unavailable: {exc}"}
        headers = {"Authorization": f"Bearer {self.harness.client.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=float(timeout_sec)) as probe_client:
                response = await probe_client.get(health_url, headers=headers)
            return {"ok": response.status_code < 500, "status_code": response.status_code}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def _wait_for_backend_health(self, health_url: str, *, timeout_sec: int) -> dict[str, Any]:
        deadline = time.monotonic() + max(1, int(timeout_sec))
        last_result: dict[str, Any] = {"ok": False, "error": "health probe not started"}
        while time.monotonic() < deadline:
            last_result = await self._probe_backend_health(
                health_url,
                timeout_sec=min(self.harness.backend_healthcheck_timeout_sec, max(1, int(timeout_sec))),
            )
            if last_result.get("ok"):
                return last_result
            await asyncio.sleep(1.0)
        return last_result

    async def _attempt_backend_restart_recovery(
        self,
        health_url: str,
        *,
        timeout_sec: int,
    ) -> dict[str, Any]:
        rate_limit = self.harness._check_backend_restart_rate_limit()
        if not rate_limit.get("allowed", False):
            return {
                "status": "unrecovered",
                "action": "rate_limited",
                "message": (
                    f"Backend restart suppressed by supervisor rate limit "
                    f"({rate_limit.get('count', 0)}/{self.harness.backend_max_restarts_per_hour} in the last hour)."
                ),
                "health_after": {"ok": False},
                "restart_window": rate_limit,
            }

        self.harness._record_backend_restart_attempt()
        command_result = await self.harness._run_backend_restart_command(self.harness.backend_restart_command)
        if command_result.get("ok"):
            health_after = await self.harness._wait_for_backend_health(
                health_url,
                timeout_sec=max(timeout_sec, int(self.harness.backend_restart_grace_sec)),
            )
            if health_after.get("ok"):
                return {
                    "status": "recovered",
                    "action": "restart_command",
                    "message": "Backend restart command succeeded and health probe recovered.",
                    "health_after": health_after,
                }
            return {
                "status": "unrecovered",
                "action": "restart_command",
                "message": "Backend restart command ran, but the health probe did not recover.",
                "health_after": health_after,
            }
        return {
            "status": "unrecovered",
            "action": "restart_command",
            "message": str(command_result.get("message") or "Backend restart command failed."),
            "health_after": {"ok": False},
        }

    async def _run_backend_restart_command(self, command: str) -> dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except Exception as exc:
            return {"ok": False, "message": f"Unable to launch restart command: {exc}"}
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"ok": False, "message": "Restart command timed out after 60s."}
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode == 0:
            return {"ok": True, "stdout": stdout_text, "stderr": stderr_text}
        return {
            "ok": False,
            "message": f"Restart command exited with status {proc.returncode}.",
            "stdout": stdout_text,
            "stderr": stderr_text,
        }

    async def _run_backend_unload_command(self, command: str | None) -> dict[str, Any]:
        if command:
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except Exception as exc:
                return {"ok": False, "message": f"Unable to launch unload command: {exc}"}
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return {"ok": False, "message": "Unload command timed out after 60s."}
            stdout_text = stdout.decode("utf-8", errors="replace").strip()
            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if proc.returncode == 0:
                return {"ok": True, "stdout": stdout_text, "stderr": stderr_text}
            return {
                "ok": False,
                "message": f"Unload command exited with status {proc.returncode}.",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        if self.harness.provider_profile == "lmstudio":
            return await self._run_lmstudio_backend_unload()
        if self.harness.provider_profile != "ollama":
            return {"ok": False, "message": "No backend unload strategy is available for this provider."}
        return await self._run_ollama_backend_unload()

    def _backend_unload_action(self) -> str:
        if self.harness.backend_unload_command:
            return "unload_command"
        if self.harness.provider_profile == "lmstudio":
            return "lmstudio_api_unload"
        if self.harness.provider_profile == "ollama":
            return "ollama_keep_alive_zero"
        return "unload_command"

    def _backend_unload_message(self, outcome: str) -> str:
        if self.harness.backend_unload_command:
            return f"Backend unload command {outcome}."
        if self.harness.provider_profile == "lmstudio":
            return f"LM Studio unload request {outcome}."
        if self.harness.provider_profile == "ollama":
            return f"Ollama unload request {outcome}."
        return f"Backend unload request {outcome}."

    async def _run_ollama_backend_unload(self) -> dict[str, Any]:
        try:
            import httpx
        except Exception as exc:
            return {"ok": False, "message": f"httpx unavailable: {exc}"}

        base_url = str(self.harness.client.base_url or "").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        unload_url = f"{base_url}/api/generate"
        payload = {
            "model": self.harness.client.model,
            "prompt": "",
            "stream": False,
            "keep_alive": 0,
        }
        headers = {
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=float(self.harness.backend_healthcheck_timeout_sec)) as unload_client:
                response = await unload_client.post(unload_url, headers=headers, json=payload)
        except Exception as exc:
            return {"ok": False, "message": f"Ollama unload request failed: {exc}"}
        if response.status_code >= 400:
            return {
                "ok": False,
                "message": f"Ollama unload request failed with status {response.status_code}.",
            }
        body_text = response.text.strip()
        return {
            "ok": True,
            "status_code": response.status_code,
            "body": body_text,
            "message": "Ollama unload request completed.",
        }

    async def _run_lmstudio_backend_unload(self) -> dict[str, Any]:
        try:
            import httpx
        except Exception as exc:
            return {"ok": False, "message": f"httpx unavailable: {exc}"}

        base_url = str(self.harness.client.base_url or "").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        list_url = f"{base_url}/api/v1/models"
        unload_url = f"{base_url}/api/v1/models/unload"
        headers = {"Content-Type": "application/json"}
        api_key = str(self.harness.client.api_key or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=float(self.harness.backend_healthcheck_timeout_sec)) as unload_client:
                response = await unload_client.get(list_url, headers=headers)
                if response.status_code >= 400:
                    return {
                        "ok": False,
                        "message": f"LM Studio model list request failed with status {response.status_code}.",
                    }
                payload = response.json()
                instance_ids, loaded_summary = self._find_lmstudio_loaded_instance_ids(payload)
                if not instance_ids:
                    loaded_blob = ", ".join(loaded_summary) if loaded_summary else "none"
                    return {
                        "ok": False,
                        "message": (
                            f"LM Studio model '{self.harness.client.model}' is not currently loaded "
                            f"(loaded instances: {loaded_blob})."
                        ),
                    }
                for instance_id in instance_ids:
                    unload_response = await unload_client.post(
                        unload_url,
                        headers=headers,
                        json={"instance_id": instance_id},
                    )
                    if unload_response.status_code >= 400:
                        return {
                            "ok": False,
                            "message": (
                                f"LM Studio unload request failed for instance '{instance_id}' "
                                f"with status {unload_response.status_code}."
                            ),
                        }
        except Exception as exc:
            return {"ok": False, "message": f"LM Studio unload request failed: {exc}"}
        return {
            "ok": True,
            "instance_ids": instance_ids,
            "message": f"LM Studio unload request completed for {len(instance_ids)} instance(s).",
        }

    def _find_lmstudio_loaded_instance_ids(self, payload: dict[str, Any]) -> tuple[list[str], list[str]]:
        if not isinstance(payload, dict):
            return [], []
        models = payload.get("models")
        if not isinstance(models, list):
            return [], []
        target_model = str(self.harness.client.model or "").strip()
        instance_ids: list[str] = []
        all_loaded_instance_ids: list[str] = []
        seen: set[str] = set()
        seen_all: set[str] = set()
        loaded_summary: list[str] = []
        target_known = False
        for entry in models:
            if not isinstance(entry, dict):
                continue
            model_key = str(entry.get("key") or "").strip()
            if model_key == target_model:
                target_known = True
            loaded_instances = entry.get("loaded_instances")
            if not isinstance(loaded_instances, list):
                continue
            for loaded_entry in loaded_instances:
                if not isinstance(loaded_entry, dict):
                    continue
                instance_id = str(loaded_entry.get("id") or "").strip()
                if not instance_id:
                    continue
                loaded_summary.append(f"{model_key}:{instance_id}" if model_key else instance_id)
                if instance_id not in seen_all:
                    seen_all.add(instance_id)
                    all_loaded_instance_ids.append(instance_id)
                if instance_id in seen:
                    continue
                if instance_id == target_model or model_key == target_model:
                    seen.add(instance_id)
                    instance_ids.append(instance_id)
        if instance_ids:
            return instance_ids, loaded_summary
        if target_known:
            return all_loaded_instance_ids, loaded_summary
        return [], loaded_summary
