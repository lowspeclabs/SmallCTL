"""Provider adapter registry."""

from __future__ import annotations

from .base import ProviderAdapter, StreamPolicy
from .generic import GenericAdapter
from .lmstudio import LMStudioAdapter
from .openrouter import OpenRouterAdapter

_GENERIC_ADAPTER = GenericAdapter()
_REGISTRY: dict[str, ProviderAdapter] = {
    "generic": _GENERIC_ADAPTER,
    "lmstudio": LMStudioAdapter(),
    "openrouter": OpenRouterAdapter(),
}


def get_provider_adapter(profile: str | None) -> ProviderAdapter:
    key = str(profile or "generic").strip().lower()
    if key == "auto":
        key = "generic"
    return _REGISTRY.get(key, _GENERIC_ADAPTER)


def registered_provider_adapters() -> dict[str, ProviderAdapter]:
    return dict(_REGISTRY)


__all__ = [
    "ProviderAdapter",
    "StreamPolicy",
    "get_provider_adapter",
    "registered_provider_adapters",
]
