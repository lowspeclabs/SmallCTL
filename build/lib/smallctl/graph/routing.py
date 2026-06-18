from __future__ import annotations

from enum import Enum


class LoopRoute(str, Enum):
    DISPATCH_TOOLS = "dispatch_tools"
    NEXT_STEP = "next_step"
    FINALIZE = "finalize"
