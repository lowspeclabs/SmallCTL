from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

from ..models.events import UIEvent

if TYPE_CHECKING:
    from ..harness import Harness


@dataclass
class GraphRuntimeDeps:
    harness: "Harness"
    event_handler: Callable[[UIEvent], Awaitable[None] | None] | None = None
