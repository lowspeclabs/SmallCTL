from __future__ import annotations

import time
from pathlib import Path
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from .common import fail, ok


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
) -> dict[str, Any]:
    if httpx is None:
        return fail("Dependency missing: httpx")
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()
        target = Path(output_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        return ok(str(target), metadata={"bytes": len(response.content), "status_code": response.status_code})
    except Exception as exc:
        return fail(str(exc))


def _coerce_body(response: Any) -> Any:
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        return response.json()
    return response.text
