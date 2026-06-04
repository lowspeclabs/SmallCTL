from __future__ import annotations

import hashlib
import re
from typing import Any


def stderr_text(result: Any) -> str:
    stderr = ""
    if isinstance(getattr(result, "output", None), dict):
        stderr = str(result.output.get("stderr") or "")
    if not stderr and getattr(result, "error", None):
        stderr = str(result.error)
    return stderr


def stderr_signature_line(result: Any) -> str | None:
    text = stderr_text(result)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return re.sub(r"\s+", " ", stripped)[:240]
    return None


def stderr_signature_key(result: Any) -> str | None:
    line = stderr_signature_line(result)
    if not line:
        return None
    return hashlib.sha1(line.lower().encode("utf-8", errors="replace")).hexdigest()[:12]
