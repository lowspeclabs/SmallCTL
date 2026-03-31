from __future__ import annotations

from typing import Any

from ..common import fail
from .runner import AnsibleRunnerAdapter


async def ansible_playbook(
    adapter: AnsibleRunnerAdapter,
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
    if bool(playbook) == bool(tasks):
        return fail("Exactly one of playbook or tasks must be provided")
    if tasks is not None and not isinstance(tasks, list):
        return fail("tasks must be a list")
    return await adapter.run_playbook(
        playbook=playbook,
        tasks=tasks,
        hosts=hosts,
        vars=vars,
        become=become,
        check=check,
        tags=tags,
        limit=limit,
        timeout=timeout,
    )

