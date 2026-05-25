from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from ..state import LoopState
from .common import fail, ok
from .fs_sessions import _record_file_change
from .fs_write_session_policy import _guard_suspicious_temp_root_path
from .fs_write_sessions import _resolve


async def http_get(
    url: str,
    headers: dict[str, str] | None = None,
    timeout_sec: int = 20,
) -> dict[str, Any]:
    if httpx is None:
        return fail("Dependency missing: httpx")
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.get(url, headers=headers)
        body = _coerce_body(response)
        return ok(
            body,
            metadata={
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_type": response.headers.get("content-type"),
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            },
        )
    except Exception as exc:
        return fail(str(exc))


async def http_post(
    url: str,
    json_body: Any | None = None,
    headers: dict[str, str] | None = None,
    timeout_sec: int = 20,
) -> dict[str, Any]:
    if httpx is None:
        return fail("Dependency missing: httpx")
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.post(url, json=json_body, headers=headers)
        body = _coerce_body(response)
        return ok(
            body,
            metadata={
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content_type": response.headers.get("content-type"),
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            },
        )
    except Exception as exc:
        return fail(str(exc))


async def file_download(
    url: str,
    output_path: str,
    headers: dict[str, str] | None = None,
    timeout_sec: int = 20,
    state: LoopState | None = None,
) -> dict[str, Any]:
    target_or_error = _resolve_download_target(output_path, state=state)
    if isinstance(target_or_error, dict):
        return target_or_error
    target = target_or_error
    if httpx is None:
        return fail("Dependency missing: httpx")
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        _record_file_change(state, target)
        return ok(str(target), metadata={"bytes": len(response.content), "status_code": response.status_code})
    except Exception as exc:
        return fail(str(exc))


def _resolve_download_target(output_path: str, *, state: LoopState | None = None) -> Path | dict[str, Any]:
    suspicious_path = _guard_suspicious_temp_root_path(output_path)
    if suspicious_path is not None:
        return suspicious_path

    cwd = str(getattr(state, "cwd", "") or Path.cwd())
    workspace = Path(cwd).resolve()
    target = _resolve(output_path, cwd)
    try:
        target.relative_to(workspace)
    except ValueError:
        return fail(
            "file_download output_path must stay within the active workspace.",
            metadata={
                "path": str(target),
                "workspace": str(workspace),
                "requested_path": output_path,
                "error_kind": "workspace_path_traversal",
            },
        )
    return target


def _coerce_body(response: Any) -> Any:
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        return response.json()
    return response.text
