from __future__ import annotations

from typing import Any

from .usage import extract_context_limit, extract_max_completion_tokens, extract_supported_parameters
from .client_transport_context_limits import _remember_context_limit


def _remember_model_metadata(client: Any, payload: Any, *, source: str) -> int | None:
    if isinstance(payload, dict):
        try:
            client.model_metadata = dict(payload)
        except Exception:
            pass
        try:
            client.model_metadata_source = source
        except Exception:
            pass

        max_completion_tokens = extract_max_completion_tokens(payload)
        if max_completion_tokens is not None:
            try:
                client.model_max_completion_tokens = int(max_completion_tokens)
            except Exception:
                pass

        supported_parameters = extract_supported_parameters(payload)
        if supported_parameters is not None:
            try:
                client.model_supported_parameters = list(supported_parameters)
            except Exception:
                pass

    return _remember_context_limit(client, extract_context_limit(payload))
