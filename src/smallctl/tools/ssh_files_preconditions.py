from __future__ import annotations

from typing import Any

from ..state import LoopState


def _sha_from_artifact(artifact: Any) -> str:
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    for key in ("sha256", "new_sha256", "readback_sha256", "old_sha256"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    return ""


def _artifact_path_for_precondition(artifact: Any) -> str:
    metadata = getattr(artifact, "metadata", {})
    if isinstance(metadata, dict):
        path = str(metadata.get("path") or "").strip()
        if path:
            return path
    return str(getattr(artifact, "source", "") or "").strip()


def _resolve_expected_sha_precondition(
    *,
    path: str,
    expected_sha256: str | None,
    source_artifact_id: str | None,
    state: LoopState | None,
) -> tuple[str | None, dict[str, Any] | None]:
    explicit = str(expected_sha256 or "").strip()
    artifact_id = str(source_artifact_id or "").strip()
    if not artifact_id:
        return (explicit or None), None
    if state is None:
        return None, {
            "reason": "source_artifact_unavailable",
            "message": "source_artifact_id requires harness state with artifact metadata.",
        }
    artifacts = getattr(state, "artifacts", {})
    if not isinstance(artifacts, dict):
        return None, {
            "reason": "source_artifact_unavailable",
            "message": "source_artifact_id requires an artifact registry in state.",
        }
    artifact = artifacts.get(artifact_id)
    if artifact is None:
        return None, {
            "reason": "source_artifact_missing",
            "message": f"Artifact `{artifact_id}` was not found in state.",
        }
    artifact_sha = _sha_from_artifact(artifact)
    if not artifact_sha:
        return None, {
            "reason": "source_artifact_missing_sha",
            "message": f"Artifact `{artifact_id}` does not carry a usable sha256 precondition.",
        }
    artifact_path = _artifact_path_for_precondition(artifact)
    if artifact_path and str(artifact_path).strip() != str(path).strip():
        return None, {
            "reason": "source_artifact_path_mismatch",
            "message": (
                f"Artifact `{artifact_id}` describes `{artifact_path}`, which does not match target path `{path}`."
            ),
        }
    if explicit and explicit != artifact_sha:
        return None, {
            "reason": "expected_sha256_mismatch",
            "message": "expected_sha256 did not match the hash referenced by source_artifact_id.",
        }
    return artifact_sha, {
        "resolved_expected_sha256": artifact_sha,
        "source_artifact_id": artifact_id,
    }
