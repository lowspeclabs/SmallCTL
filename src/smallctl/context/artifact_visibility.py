from __future__ import annotations

from typing import Any


def is_superseded_artifact(artifact: Any) -> bool:
    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, dict):
        return False
    superseded_by = metadata.get("superseded_by")
    return isinstance(superseded_by, str) and bool(superseded_by.strip())


def is_prompt_visible_artifact(artifact: Any) -> bool:
    metadata = getattr(artifact, "metadata", None)
    if not isinstance(metadata, dict):
        return True
    return metadata.get("model_visible", True) is not False
