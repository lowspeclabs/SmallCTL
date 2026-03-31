from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    import ansible_runner
    _ANSIBLE_RUNNER_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    ansible_runner = None
    _ANSIBLE_RUNNER_IMPORT_ERROR = str(exc)

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from ...logging_utils import log_kv
from ..common import fail, ok
from .inventory import SessionInventory


class AnsibleRunnerAdapter:
    def __init__(self, inventory: SessionInventory) -> None:
        self.log = logging.getLogger("smallctl.ansible")
        self.inventory = inventory

    async def dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        log_kv(self.log, logging.INFO, "ansible_dispatch", tool_name=tool_name)
        if tool_name == "ansible_task":
            return await self.run_task(**args)
        if tool_name == "ansible_playbook":
            return await self.run_playbook(**args)
        if tool_name == "ansible_inventory":
            return await self.inventory_action(**args)
        return fail(f"Unknown ansible tool: {tool_name}")

    async def inventory_action(
        self,
        action: str = "list",
        host: str | None = None,
        group: str | None = None,
        variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            if action == "list":
                return ok(self.inventory.list())
            if action == "add_host":
                if not host:
                    return fail("host is required for add_host")
                self.inventory.add_host(host, variables or {})
                return ok(self.inventory.list())
            if action == "add_group":
                if not group:
                    return fail("group is required for add_group")
                hosts = [host] if host else []
                self.inventory.add_group(group, hosts)
                return ok(self.inventory.list())
            if action == "remove_host":
                if not host:
                    return fail("host is required for remove_host")
                self.inventory.remove_host(host)
                return ok(self.inventory.list())
            return fail(f"Unsupported inventory action: {action}")
        except Exception as exc:
            return fail(str(exc))

    async def run_task(
        self,
        module: str,
        args: dict[str, Any] | None = None,
        hosts: str = "localhost",
        become: bool = False,
        check: bool = False,
        timeout: int = 60,
    ) -> dict[str, Any]:
        tasks = [{"name": "smallctl task", module: args or {}}]
        return await self.run_playbook(
            tasks=tasks,
            hosts=hosts,
            become=become,
            check=check,
            timeout=timeout,
        )

    async def run_playbook(
        self,
        playbook: str | None = None,
        tasks: list[dict[str, Any]] | None = None,
        hosts: str = "localhost",
        vars: dict[str, Any] | None = None,
        become: bool = False,
        check: bool = False,
        tags: str | None = None,
        limit: str | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        if yaml is None:
            return fail("Dependency missing: pyyaml")
        if bool(playbook) == bool(tasks):
            return fail("Exactly one of playbook or tasks is required")

        if ansible_runner is None:
            log_kv(
                self.log,
                logging.WARNING,
                "ansible_runner_unavailable_fallback",
                import_error=_ANSIBLE_RUNNER_IMPORT_ERROR or "unknown",
                hosts=hosts,
            )
            return await self._run_fallback_local(
                playbook=playbook,
                tasks=tasks,
                hosts=hosts,
                check=check,
                timeout=timeout,
            )

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._run_sync,
                    playbook=playbook,
                    tasks=tasks,
                    hosts=hosts,
                    vars=vars,
                    become=become,
                    check=check,
                    tags=tags,
                    limit=limit,
                ),
                timeout=timeout,
            )
            log_kv(
                self.log,
                logging.INFO,
                "ansible_run_complete",
                hosts=hosts,
                check=check,
                timeout=timeout,
            )
            return result
        except asyncio.TimeoutError:
            log_kv(
                self.log,
                logging.WARNING,
                "ansible_run_timeout",
                hosts=hosts,
                timeout=timeout,
            )
            return fail(f"Ansible run timed out after {timeout}s")
        except Exception as exc:
            log_kv(self.log, logging.ERROR, "ansible_run_exception", error=str(exc))
            return fail(str(exc))

    async def _run_fallback_local(
        self,
        *,
        playbook: str | None,
        tasks: list[dict[str, Any]] | None,
        hosts: str,
        check: bool,
        timeout: int,
    ) -> dict[str, Any]:
        if yaml is None:
            return fail("Dependency missing: pyyaml")
        if hosts not in {"localhost", "127.0.0.1", "all"}:
            return fail("Fallback ansible execution supports localhost/all only")

        try:
            resolved_tasks = self._resolve_tasks(playbook=playbook, tasks=tasks)
        except Exception as exc:
            return fail(str(exc))

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._execute_local_tasks, resolved_tasks, check),
                timeout=timeout,
            )
            return ok(
                result,
                metadata={
                    "status": "successful",
                    "rc": 0,
                    "runner": "local-fallback",
                    "import_error": _ANSIBLE_RUNNER_IMPORT_ERROR,
                },
            )
        except asyncio.TimeoutError:
            return fail(f"Ansible fallback timed out after {timeout}s")
        except Exception as exc:
            return fail(str(exc))

    def _resolve_tasks(
        self,
        *,
        playbook: str | None,
        tasks: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if tasks is not None:
            return tasks
        assert playbook is not None
        assert yaml is not None
        playbook_path = Path(playbook).resolve()
        if not playbook_path.exists():
            raise FileNotFoundError(f"Playbook not found: {playbook_path}")
        payload = yaml.safe_load(playbook_path.read_text(encoding="utf-8")) or []
        if not isinstance(payload, list) or not payload:
            raise ValueError("Playbook must contain a list of plays")
        first_play = payload[0] or {}
        play_tasks = first_play.get("tasks")
        if not isinstance(play_tasks, list):
            raise ValueError("Playbook first play must include a tasks list")
        return play_tasks

    def _execute_local_tasks(self, tasks: list[dict[str, Any]], check: bool) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for raw_task in tasks:
            task = raw_task or {}
            module, args = self._extract_module(task)
            name = str(task.get("name", module))
            if module == "ping":
                results.append(
                    {
                        "name": name,
                        "module": module,
                        "ok": True,
                        "changed": False,
                        "result": {"ping": "pong"},
                    }
                )
                continue
            if module == "debug":
                results.append(
                    {
                        "name": name,
                        "module": module,
                        "ok": True,
                        "changed": False,
                        "result": {"msg": (args or {}).get("msg")},
                    }
                )
                continue
            if module in {"command", "shell"}:
                cmd = self._command_from_args(args)
                if check:
                    results.append(
                        {
                            "name": name,
                            "module": module,
                            "ok": True,
                            "changed": False,
                            "result": {"skipped": True, "check_mode": True, "cmd": cmd},
                        }
                    )
                    continue
                completed = self._run_local_command(cmd)
                results.append(
                    {
                        "name": name,
                        "module": module,
                        "ok": completed.returncode == 0,
                        "changed": completed.returncode == 0,
                        "result": {
                            "rc": completed.returncode,
                            "stdout": completed.stdout,
                            "stderr": completed.stderr,
                            "cmd": cmd,
                        },
                    }
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"Task '{name}' failed with rc={completed.returncode}")
                continue
            raise RuntimeError(f"Unsupported module in local fallback: {module}")

        return {
            "status": "successful",
            "rc": 0,
            "stats": {"ok": len(results), "failed": 0},
            "tasks": results,
        }

    @staticmethod
    def _extract_module(task: dict[str, Any]) -> tuple[str, Any]:
        skip_keys = {
            "name",
            "when",
            "tags",
            "vars",
            "register",
            "become",
            "delegate_to",
            "changed_when",
            "failed_when",
        }
        for key, value in task.items():
            if key not in skip_keys:
                return str(key), value
        raise RuntimeError("Task has no module key")

    @staticmethod
    def _command_from_args(args: Any) -> str:
        if isinstance(args, str):
            return args
        if isinstance(args, dict):
            if "cmd" in args and isinstance(args["cmd"], str):
                return args["cmd"]
            if "_raw_params" in args and isinstance(args["_raw_params"], str):
                return args["_raw_params"]
        raise RuntimeError("command/shell task requires string args or cmd/_raw_params")

    @staticmethod
    def _run_local_command(command: str) -> subprocess.CompletedProcess[str]:
        if os.name == "nt":
            return subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                text=True,
                capture_output=True,
            )
        return subprocess.run(command, shell=True, text=True, capture_output=True)

    def _run_sync(
        self,
        *,
        playbook: str | None,
        tasks: list[dict[str, Any]] | None,
        hosts: str,
        vars: dict[str, Any] | None,
        become: bool,
        check: bool,
        tags: str | None,
        limit: str | None,
    ) -> dict[str, Any]:
        assert ansible_runner is not None
        assert yaml is not None

        with tempfile.TemporaryDirectory(prefix="smallctl-ansible-") as tmp:
            private_data_dir = Path(tmp)
            inventory_path = private_data_dir / "inventory.yml"
            inventory_path.write_text(
                yaml.safe_dump(self.inventory.to_ansible_inventory(), sort_keys=False),
                encoding="utf-8",
            )

            cmdline = ""
            if check:
                cmdline += " --check"
            if tags:
                cmdline += f" --tags {tags}"
            if limit:
                cmdline += f" --limit {limit}"

            if tasks is not None:
                play = [
                    {
                        "name": "smallctl inline play",
                        "hosts": hosts,
                        "become": become,
                        "gather_facts": False,
                        "vars": vars or {},
                        "tasks": tasks,
                    }
                ]
                playbook_path = private_data_dir / "playbook.yml"
                playbook_path.write_text(
                    yaml.safe_dump(play, sort_keys=False),
                    encoding="utf-8",
                )
                playbook_name = playbook_path.name
            else:
                playbook_path = Path(playbook or "").resolve()
                if not playbook_path.exists():
                    return fail(f"Playbook not found: {playbook_path}")
                playbook_name = str(playbook_path)

            run = ansible_runner.run(
                private_data_dir=str(private_data_dir),
                inventory=str(inventory_path),
                playbook=playbook_name,
                quiet=True,
                cmdline=cmdline.strip(),
            )
            status = getattr(run, "status", "unknown")
            rc = getattr(run, "rc", None)
            stats = getattr(run, "stats", {}) or {}
            result = {
                "status": status,
                "rc": rc,
                "stats": stats,
            }
            if rc == 0:
                log_kv(
                    self.log,
                    logging.INFO,
                    "ansible_sync_success",
                    status=status,
                    rc=rc,
                )
                return ok(result, metadata={"status": status, "rc": rc})
            log_kv(
                self.log,
                logging.WARNING,
                "ansible_sync_failure",
                status=status,
                rc=rc,
            )
            return fail(
                f"Ansible run failed with rc={rc}",
                metadata={"status": status, "rc": rc, "stats": stats},
            )
