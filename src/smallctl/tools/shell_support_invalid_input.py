from __future__ import annotations

from typing import Any

from .shell_support_constants import _INVALID_INPUT_MARKERS


class InvalidInputLoopDetector:
    def __init__(self, *, threshold: int = 3, max_tail_chars: int = 4096) -> None:
        self.threshold = max(1, int(threshold))
        self.max_tail_chars = max(512, int(max_tail_chars))
        self.count = 0
        self.output_tail = ""

    def observe(self, chunk: str) -> dict[str, Any] | None:
        text = str(chunk or "")
        if not text:
            return None
        self.output_tail = (self.output_tail + text)[-self.max_tail_chars :]
        lowered = text.lower()
        self.count += sum(lowered.count(marker) for marker in _INVALID_INPUT_MARKERS)
        if self.count < self.threshold:
            return None
        return {
            "reason": "interactive_invalid_input_loop",
            "invalid_input_count": self.count,
            "output_tail": self.output_tail,
            "diagnosis": (
                "The command is repeatedly rejecting automated input. Stop this run and replace "
                "blanket input piping with documented non-interactive flags, a config/preseed file, "
                "or an explicit prompt answer script."
            ),
        }
