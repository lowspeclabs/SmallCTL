from __future__ import annotations

from typing import Any

from ..state import ArtifactSnippet, ContextBrief, EpisodicSummary, ExperienceMemory, LoopState, TurnBundle
from .artifact_visibility import artifact_path_candidates, is_read_only_artifact
from .frame_invalidation_utils import (
    coerce_datetime,
    durably_stale_ids,
    guard_trip_preserved_ids,
    is_optimistic_statement,
    path_matches_any,
    path_tokens,
    recent_invalidation_events,
)


def filter_invalidated_turn_bundles(
    *,
    state: LoopState,
    bundles: list[TurnBundle],
) -> tuple[list[TurnBundle], list[str]]:
    invalidations = recent_invalidation_events(state)
    stale_ids = durably_stale_turn_bundle_ids(state)
    if not bundles:
        return bundles, []
    if not invalidations and not stale_ids:
        return bundles, []
    kept: list[TurnBundle] = []
    dropped_ids: list[str] = []
    for bundle in bundles:
        if bundle.bundle_id and bundle.bundle_id in stale_ids:
            dropped_ids.append(bundle.bundle_id)
            continue
        if bundle_invalidated(state=state, bundle=bundle, invalidations=invalidations):
            if bundle.bundle_id:
                dropped_ids.append(bundle.bundle_id)
            continue
        kept.append(bundle)
    return kept, dropped_ids


def filter_invalidated_observations(
    *,
    state: LoopState,
    observations: list[Any],
) -> tuple[list[Any], list[str], int]:
    stale_ids = durably_stale_observation_ids(state)
    guard_preserved_ids = guard_trip_preserved_ids(state, "_guard_trip_preserved_observation_ids")
    kept: list[Any] = []
    dropped_ids: list[str] = []
    dropped_count = 0
    for packet in observations:
        observation_id = str(getattr(packet, "observation_id", "") or "").strip()
        if observation_id and observation_id in guard_preserved_ids:
            kept.append(packet)
            continue
        if observation_id and observation_id in stale_ids:
            dropped_count += 1
            dropped_ids.append(observation_id)
            continue
        if bool(getattr(packet, "stale", False)):
            dropped_count += 1
            if observation_id:
                dropped_ids.append(observation_id)
            continue
        kept.append(packet)
    return kept, dropped_ids, dropped_count


def filter_invalidated_context_briefs(
    *,
    state: LoopState,
    briefs: list[ContextBrief],
) -> tuple[list[ContextBrief], list[str]]:
    invalidations = recent_invalidation_events(state)
    stale_ids = durably_stale_brief_ids(state)
    if not briefs:
        return briefs, []
    if not invalidations and not stale_ids:
        return briefs, []
    kept: list[ContextBrief] = []
    dropped_ids: list[str] = []
    for brief in briefs:
        if brief.brief_id and brief.brief_id in stale_ids:
            dropped_ids.append(brief.brief_id)
            continue
        if brief_invalidated(state=state, brief=brief, invalidations=invalidations):
            if brief.brief_id:
                dropped_ids.append(brief.brief_id)
            continue
        kept.append(brief)
    return kept, dropped_ids


def filter_invalidated_summaries(
    *,
    state: LoopState,
    summaries: list[EpisodicSummary],
) -> tuple[list[EpisodicSummary], list[str]]:
    invalidations = recent_invalidation_events(state)
    stale_ids = durably_stale_summary_ids(state)
    guard_preserved_ids = guard_trip_preserved_ids(state, "_guard_trip_preserved_summary_ids")
    if not summaries:
        return summaries, []
    if not invalidations and not stale_ids:
        return summaries, []
    kept: list[EpisodicSummary] = []
    dropped_ids: list[str] = []
    for summary in summaries:
        if summary.summary_id and summary.summary_id in guard_preserved_ids:
            kept.append(summary)
            continue
        if summary.summary_id and summary.summary_id in stale_ids:
            dropped_ids.append(summary.summary_id)
            continue
        if summary_invalidated(state=state, summary=summary, invalidations=invalidations):
            if summary.summary_id:
                dropped_ids.append(summary.summary_id)
            continue
        kept.append(summary)
    return kept, dropped_ids


def filter_invalidated_experiences(
    *,
    state: LoopState,
    experiences: list[ExperienceMemory],
) -> tuple[list[ExperienceMemory], list[str]]:
    invalidations = recent_invalidation_events(state)
    stale_ids = durably_stale_experience_ids(state)
    if not experiences:
        return experiences, []
    if not invalidations and not stale_ids:
        return experiences, []
    kept: list[ExperienceMemory] = []
    dropped_ids: list[str] = []
    for memory in experiences:
        if experience_invalidated(state=state, memory=memory, invalidations=invalidations):
            if memory.memory_id:
                dropped_ids.append(memory.memory_id)
            continue
        kept.append(memory)
    return kept, dropped_ids


def filter_invalidated_artifact_snippets(
    *,
    state: LoopState,
    snippets: list[ArtifactSnippet],
) -> tuple[list[ArtifactSnippet], list[str]]:
    invalidations = recent_invalidation_events(state)
    stale_ids = durably_stale_artifact_ids(state)
    guard_preserved_ids = guard_trip_preserved_ids(state, "_guard_trip_preserved_artifact_ids")
    if not snippets:
        return snippets, []
    if not invalidations and not stale_ids:
        return snippets, []
    artifacts = getattr(state, "artifacts", {}) if isinstance(getattr(state, "artifacts", {}), dict) else {}
    kept: list[ArtifactSnippet] = []
    dropped_ids: list[str] = []
    for snippet in snippets:
        if snippet.artifact_id and snippet.artifact_id in guard_preserved_ids:
            kept.append(snippet)
            continue
        if snippet.artifact_id and snippet.artifact_id in stale_ids:
            dropped_ids.append(snippet.artifact_id)
            continue
        artifact = artifacts.get(snippet.artifact_id)
        if artifact_invalidated(state=state, artifact=artifact, invalidations=invalidations):
            if snippet.artifact_id:
                dropped_ids.append(snippet.artifact_id)
            continue
        kept.append(snippet)
    return kept, dropped_ids


def durably_stale_experience_ids(state: LoopState) -> set[str]:
    return durably_stale_ids(state, "_experience_staleness")


def durably_stale_turn_bundle_ids(state: LoopState) -> set[str]:
    return durably_stale_ids(state, "_turn_bundle_staleness")


def durably_stale_brief_ids(state: LoopState) -> set[str]:
    return durably_stale_ids(state, "_context_brief_staleness")


def durably_stale_summary_ids(state: LoopState) -> set[str]:
    return durably_stale_ids(state, "_summary_staleness")


def durably_stale_artifact_ids(state: LoopState) -> set[str]:
    return durably_stale_ids(state, "_artifact_staleness")


def durably_stale_observation_ids(state: LoopState) -> set[str]:
    return durably_stale_ids(state, "_observation_staleness")


def verifier_failure_paths(event: dict[str, Any]) -> list[str]:
    candidates = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
    details = event.get("details")
    if isinstance(details, dict):
        for key in ("command", "target"):
            candidates.extend(path_tokens(str(details.get(key) or "")))
    normalized: list[str] = []
    for path in candidates:
        if path and path not in normalized:
            normalized.append(path)
    return normalized


def verifier_failure_related_to_text(text: str, event: dict[str, Any]) -> bool:
    failure_paths = verifier_failure_paths(event)
    if not failure_paths:
        return True
    text_paths = path_tokens(text)
    if text_paths:
        return any(path_matches_any(path, failure_paths) for path in text_paths)
    return any(path_matches_any(text, [path]) for path in failure_paths)


def optimistic_text_invalidated_by_verifier(text: str, event: dict[str, Any]) -> bool:
    return is_optimistic_statement(text) and verifier_failure_related_to_text(text, event)


def bundle_invalidated(
    *,
    state: LoopState,
    bundle: TurnBundle,
    invalidations: list[dict[str, Any]],
) -> bool:
    current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    for event in invalidations:
        reason = str(event.get("reason") or "").strip().lower()
        paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
        if reason in {"file_changed", "write_session_target_changed"} and paths:
            if any(path_matches_any(path, paths) for path in bundle.files_touched):
                return True
        if reason in {"phase_advanced", "environment_changed"}:
            if bundle.phase and str(bundle.phase).strip().lower() != current_phase:
                return True
        if reason in {"verifier_failed", "fama_failure_detected"} and any(
            optimistic_text_invalidated_by_verifier(line, event) for line in bundle.summary_lines
        ):
            return True
    return False


def brief_invalidated(
    *,
    state: LoopState,
    brief: ContextBrief,
    invalidations: list[dict[str, Any]],
) -> bool:
    current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    for event in invalidations:
        reason = str(event.get("reason") or "").strip().lower()
        paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
        if reason in {"file_changed", "write_session_target_changed"} and paths:
            if any(path_matches_any(path, paths) for path in brief.files_touched):
                return True
        if reason in {"phase_advanced", "environment_changed"}:
            if brief.current_phase and str(brief.current_phase).strip().lower() != current_phase:
                return True
        if reason in {"verifier_failed", "fama_failure_detected"}:
            if any(optimistic_text_invalidated_by_verifier(line, event) for line in brief.key_discoveries):
                return True
            if any(optimistic_text_invalidated_by_verifier(line, event) for line in brief.new_facts):
                return True
    return False


def summary_invalidated(
    *,
    state: LoopState,
    summary: EpisodicSummary,
    invalidations: list[dict[str, Any]],
) -> bool:
    failure_mode = str(getattr(state, "last_failure_class", "") or "").strip().lower()
    summary_created_at = coerce_datetime(getattr(summary, "created_at", None))
    for event in invalidations:
        event_created_at = coerce_datetime(event.get("created_at"))
        if summary_created_at is not None and event_created_at is not None and event_created_at <= summary_created_at:
            continue
        reason = str(event.get("reason") or "").strip().lower()
        paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
        if reason in {"file_changed", "write_session_target_changed"} and paths:
            if any(path_matches_any(path, paths) for path in summary.files_touched):
                return True
        if reason in {"verifier_failed", "fama_failure_detected"}:
            if any(optimistic_text_invalidated_by_verifier(line, event) for line in summary.notes):
                return True
            if (
                failure_mode
                and any(failure_mode in str(line).strip().lower() for line in summary.failed_approaches)
                and not verifier_failure_paths(event)
            ):
                return True
    return False


def experience_invalidated(
    *,
    state: LoopState,
    memory: ExperienceMemory,
    invalidations: list[dict[str, Any]],
) -> bool:
    stale_ids = durably_stale_experience_ids(state)
    if memory.memory_id and memory.memory_id in stale_ids:
        return True
    current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    failure_mode = str(getattr(state, "last_failure_class", "") or "").strip().lower()
    notes = str(memory.notes or "").strip()
    for event in invalidations:
        reason = str(event.get("reason") or "").strip().lower()
        paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
        if reason in {"file_changed", "write_session_target_changed"} and paths:
            if any(path_matches_any(notes, [path]) for path in paths):
                return True
        if reason in {"phase_advanced", "environment_changed"}:
            if memory.phase and str(memory.phase).strip().lower() != current_phase:
                return True
        if reason in {"verifier_failed", "fama_failure_detected"}:
            if str(memory.outcome or "").strip().lower() == "success" and optimistic_text_invalidated_by_verifier(notes, event):
                return True
            if (
                failure_mode
                and str(memory.failure_mode or "").strip().lower() == failure_mode
                and not verifier_failure_paths(event)
            ):
                return True
    return False


def artifact_invalidated(
    *,
    state: LoopState,
    artifact: Any,
    invalidations: list[dict[str, Any]],
) -> bool:
    if artifact is None:
        return False
    metadata = getattr(artifact, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    artifact_paths = artifact_path_candidates(artifact, metadata)
    current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    metadata_phase = str(metadata.get("phase") or metadata.get("created_phase") or "").strip().lower()
    verifier_verdict = str(metadata.get("verifier_verdict") or "").strip().lower()
    for event in invalidations:
        reason = str(event.get("reason") or "").strip().lower()
        paths = [str(path).strip() for path in (event.get("paths") or []) if str(path).strip()]
        if reason in {"file_changed", "write_session_target_changed"} and paths:
            if any(path_matches_any(candidate, paths) for candidate in artifact_paths):
                return True
        if reason in {"phase_advanced", "environment_changed"} and metadata_phase:
            if metadata_phase != current_phase:
                if not is_read_only_artifact(artifact):
                    return True
        if reason in {"verifier_failed", "fama_failure_detected"} and verifier_verdict == "pass":
            if not verifier_failure_related_to_text(
                " ".join(
                    str(part or "")
                    for part in (
                        getattr(artifact, "source", ""),
                        getattr(artifact, "summary", ""),
                        metadata.get("path", ""),
                        metadata.get("command", ""),
                        metadata.get("verifier_command", ""),
                        metadata.get("verifier_target", ""),
                    )
                ),
                event,
            ):
                continue
            return True
    return False
