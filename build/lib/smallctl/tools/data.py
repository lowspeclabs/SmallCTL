from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

try:
    import jmespath
except Exception:  # pragma: no cover
    jmespath = None

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .common import fail, ok


async def json_query(data: Any, expression: str) -> dict[str, Any]:
    if jmespath is None:
        return fail("Dependency missing: jmespath")
    try:
        result = jmespath.search(expression, data)
        return ok(result)
    except Exception as exc:
        return fail(str(exc))


async def yaml_read(path: str) -> dict[str, Any]:
    if yaml is None:
        return fail("Dependency missing: pyyaml")
    target = Path(path).resolve()
    if not target.exists():
        return fail(f"File does not exist: {target}")
    try:
        parsed = yaml.safe_load(target.read_text(encoding="utf-8"))
        return ok(parsed, metadata={"path": str(target)})
    except Exception as exc:
        return fail(str(exc))


async def diff(before: str, after: str, context: int = 3) -> dict[str, Any]:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    d = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile="before",
            tofile="after",
            lineterm="",
            n=context,
        )
    )
    return ok("\n".join(d), metadata={"line_count": len(d)})


async def json_pretty(data: Any) -> dict[str, Any]:
    try:
        return ok(json.dumps(data, indent=2, sort_keys=True))
    except Exception as exc:
        return fail(str(exc))
