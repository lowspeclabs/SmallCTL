from __future__ import annotations

import logging
from typing import Any

from .backend_recovery_facade import bind_backend_recovery_facade as _bind_backend_recovery_facade
from .context_facade import bind_context_facade as _bind_context_facade
from .core_facade import bind_core_facade as _bind_core_facade
from .initialization import initialize_harness as _initialize_harness
from .intent_facade import bind_intent_facade as _bind_intent_facade
from .runtime_facade import bind_runtime_facade as _bind_runtime_facade
from .task_boundary_facade import bind_task_boundary_facade as _bind_task_boundary_facade


class Harness:
    def __init__(self, **kwargs: Any) -> None:
        self.log = logging.getLogger("smallctl.harness")
        self.run_logger = kwargs.get("run_logger")
        _initialize_harness(self, **kwargs)


_bind_backend_recovery_facade(Harness)
_bind_context_facade(Harness)
_bind_intent_facade(Harness)
_bind_task_boundary_facade(Harness)
_bind_runtime_facade(Harness)
_bind_core_facade(Harness)

__all__ = ["Harness"]
