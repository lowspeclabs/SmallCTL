from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None

from .usage import extract_context_limit


@dataclass
class ProviderModel:
    id: str
    display_name: str
    loaded: bool | None = None
    context_length: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelListResult:
    ok: bool
    models: list[ProviderModel]
    source_url: str
    error: str = ""


def parse_openai_models(payload: Any) -> list[ProviderModel]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    models: list[ProviderModel] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        models.append(
            ProviderModel(
                id=model_id,
                display_name=model_id,
                context_length=extract_context_limit(item),
                metadata=dict(item),
            )
        )
    return _dedupe_models(models)


def parse_lmstudio_models(payload: Any) -> list[ProviderModel]:
    data = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    models: list[ProviderModel] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("key") or item.get("id") or "").strip()
        if not model_id:
            continue
        loaded_instances = item.get("loaded_instances")
        loaded: bool | None
        if isinstance(loaded_instances, list):
            loaded = bool(loaded_instances)
        elif isinstance(item.get("loaded"), bool):
            loaded = bool(item.get("loaded"))
        else:
            loaded = None
        display_name = str(item.get("display_name") or item.get("name") or model_id).strip()
        models.append(
            ProviderModel(
                id=model_id,
                display_name=display_name or model_id,
                loaded=loaded,
                context_length=extract_context_limit(item),
                metadata=dict(item),
            )
        )
    return _dedupe_models(models)


def parse_ollama_models(payload: Any) -> list[ProviderModel]:
    data = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    models: list[ProviderModel] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("name") or item.get("model") or "").strip()
        if not model_id:
            continue
        display_name = str(item.get("display_name") or model_id).strip()
        models.append(
            ProviderModel(
                id=model_id,
                display_name=display_name or model_id,
                context_length=extract_context_limit(item),
                metadata=dict(item),
            )
        )
    return _dedupe_models(models)


async def fetch_available_models(
    *,
    base_url: str,
    api_key: str | None,
    provider_profile: str,
    timeout_sec: float = 10.0,
) -> ModelListResult:
    if httpx is None:
        return ModelListResult(False, [], "", "Dependency missing: httpx")

    normalized_base = str(base_url or "").strip().rstrip("/")
    if not normalized_base:
        return ModelListResult(False, [], "", "Missing provider base URL.")

    profile = str(provider_profile or "generic").strip().lower() or "generic"
    root = _provider_root(normalized_base)
    headers = _auth_headers(api_key)
    attempts = _attempts_for_profile(profile, normalized_base, root)
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=max(1.0, float(timeout_sec))) as client:
        for source_url, parser in attempts:
            try:
                response = await client.get(source_url, headers=headers)
                if int(response.status_code) >= 400:
                    errors.append(f"{source_url}: HTTP {response.status_code}")
                    continue
                payload = response.json()
            except Exception as exc:
                errors.append(f"{source_url}: {exc}")
                continue

            models = parser(payload)
            if models:
                return ModelListResult(True, models, source_url)
            errors.append(f"{source_url}: no models found")

    error = "; ".join(errors) if errors else "No model-list endpoint was attempted."
    last_source = attempts[-1][0] if attempts else normalized_base
    return ModelListResult(False, [], last_source, error)


def _attempts_for_profile(
    profile: str,
    base_url: str,
    root: str,
) -> list[tuple[str, Any]]:
    openai_url = f"{base_url}/models"
    if profile == "lmstudio":
        return [
            (f"{root}/api/v1/models", parse_lmstudio_models),
            (openai_url, parse_openai_models),
        ]
    if profile == "ollama":
        return [
            (f"{root}/api/tags", parse_ollama_models),
            (openai_url, parse_openai_models),
        ]
    return [(openai_url, parse_openai_models)]


def _provider_root(base_url: str) -> str:
    if base_url.endswith("/v1"):
        return base_url[: -len("/v1")]
    return base_url


def _auth_headers(api_key: str | None) -> dict[str, str]:
    key = str(api_key or "").strip()
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def _dedupe_models(models: list[ProviderModel]) -> list[ProviderModel]:
    seen: set[str] = set()
    result: list[ProviderModel] = []
    for model in models:
        if model.id in seen:
            continue
        seen.add(model.id)
        result.append(model)
    return result
