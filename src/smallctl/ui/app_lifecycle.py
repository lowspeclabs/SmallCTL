from __future__ import annotations

from typing import Any


async def shutdown_harness(bridge: Any, harness: Any) -> None:
    """Defensively shut down any existing bridge/harness."""
    if bridge is not None:
        try:
            await bridge.shutdown()
        except Exception:
            pass
        return
    if harness is not None:
        try:
            await harness.teardown()
        except Exception:
            pass
