from __future__ import annotations

from typing import Any

from .runner import AnsibleRunnerAdapter


async def ansible_task(
    adapter: AnsibleRunnerAdapter,
    module: str,
    args: dict[str, Any] | None = None,
    hosts: str = "localhost",
    become: bool = False,
    check: bool = False,
    timeout: int = 60,
) -> dict[str, Any]:
    return await adapter.run_task(
        module=module,
        args=args,
        hosts=hosts,
        become=become,
        check=check,
        timeout=timeout,
    )

