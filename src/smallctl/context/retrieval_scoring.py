from __future__ import annotations

import re
from pathlib import Path

from ..normalization import tokenize as _tokens
from ..state import ArtifactRecord, LoopState, normalize_intent_label
from .retrieval_artifact_helpers import (
    artifact_contains_interactive_prompt,
    artifact_has_resolved_successor,
    artifact_success,
    artifact_tool_name,
    handoff_recent_research_artifact_ids,
    is_causal_remote_failure_artifact,
    is_remote_repair_state,
    should_pin_recent_research_artifacts,
)
from .retrieval_constants import DIAGNOSTIC_FAILURE_TOOL_PENALTIES
from .retrieval_state_helpers import (
    path_match,
    query_requests_live_remote_correction,
    state_entity_tags,
    state_target_paths,
    state_touched_symbols,
)


def score_artifact(
    *,
    artifact: ArtifactRecord,
    query: str,
    query_tokens: set[str],
    recency: int,
    state: LoopState,
) -> float:
    expanded_query_tokens = set(query_tokens)
    for token in list(query_tokens):
        stripped = token.lstrip("./")
        if len(stripped) > 1:
            expanded_query_tokens.add(stripped)
        for part in re.split(r"[\\/]+", stripped):
            if len(part) > 1:
                expanded_query_tokens.add(part)
    source_name = Path(artifact.source).name.lower() if artifact.source else ""
    source_tokens = _tokens(artifact.source)
    summary_tokens = _tokens(artifact.summary)
    keyword_tokens = {token.lower() for token in artifact.keywords}
    path_tokens = {token.lower() for token in artifact.path_tags}
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    metadata_tokens = _tokens(" ".join(
        str(metadata.get(key, ""))
        for key in ("intent", "path", "url", "command", "content_type", "source_type", "confidence")
    ))
    overlap = len(expanded_query_tokens & (source_tokens | summary_tokens | keyword_tokens | path_tokens | metadata_tokens))
    filename_bonus = 5.0 if source_name and source_name in expanded_query_tokens else 0.0
    path_bonus = 3.0 if artifact.source and any(token in artifact.source.lower() for token in expanded_query_tokens) else 0.0
    tool_bonus = 2.0 if artifact.tool_name.lower() in expanded_query_tokens else 0.0
    verifier_bonus = 2.5 if str(metadata.get("verifier_verdict") or "").strip() else 0.0
    confidence_bonus = 0.0
    confidence = metadata.get("confidence")
    try:
        confidence_bonus = max(0.0, min(1.5, float(confidence) * 1.5)) if confidence is not None else 0.0
    except (TypeError, ValueError):
        confidence_bonus = 0.0
    phase_bonus = 0.0
    metadata_phase = str(metadata.get("phase") or "").strip().lower()
    current_phase = str(getattr(state, "current_phase", "") or "").strip().lower()
    if metadata_phase and current_phase and metadata_phase == current_phase:
        phase_bonus = 2.5

    intent_bonus = 0.0
    active_intent = normalize_intent_label(getattr(state, "active_intent", "") or "")
    metadata_intent = normalize_intent_label(metadata.get("intent"))
    if active_intent and metadata_intent and metadata_intent == active_intent:
        intent_bonus += 2.5
    secondary_intents = {
        normalize_intent_label(intent)
        for intent in getattr(state, "secondary_intents", []) or []
        if normalize_intent_label(intent)
    }
    if metadata_intent and metadata_intent in secondary_intents:
        intent_bonus += 1.2

    pinned_research_bonus = 0.0
    recent_research_artifact_ids = handoff_recent_research_artifact_ids(state)
    if (
        recent_research_artifact_ids
        and artifact.artifact_id in recent_research_artifact_ids
        and str(getattr(artifact, "kind", "")).strip() in {"web_search", "web_fetch"}
        and should_pin_recent_research_artifacts(state)
    ):
        pinned_research_bonus += 6.0

    target_path_bonus = 0.0
    target_paths = state_target_paths(state)
    source_path = Path(artifact.source).as_posix().lower() if artifact.source else ""
    if source_path and target_paths and any(path_match(source_path, target) for target in target_paths):
        target_path_bonus += 4.0
    write_target = str(getattr(getattr(state, "write_session", None), "write_target_path", "") or "").strip()
    if write_target and source_path and path_match(source_path, Path(write_target).as_posix().lower()):
        target_path_bonus += 2.0

    # If this is a read-only user target (not an active write target) and it
    # has already been fully read, downgrade it so instruction files like
    # AGENTS.md do not keep dominating context after their first read.
    if target_path_bonus > 0 and source_path:
        is_write_target = bool(
            write_target and path_match(source_path, Path(write_target).as_posix().lower())
        )
        if not is_write_target:
            complete_file = bool(metadata.get("complete_file"))
            total_lines = metadata.get("total_lines")
            line_start = metadata.get("line_start")
            line_end = metadata.get("line_end")
            fully_read = complete_file or (
                isinstance(total_lines, int)
                and total_lines > 0
                and isinstance(line_start, int)
                and line_start <= 1
                and isinstance(line_end, int)
                and line_end >= total_lines
            )
            if fully_read:
                target_path_bonus *= 0.25

    entity_bonus = 0.0
    entity_overlap = len(state_entity_tags(state) & (keyword_tokens | path_tokens | metadata_tokens))
    if entity_overlap:
        entity_bonus = min(2.0, entity_overlap * 0.5)

    terminal_claim_penalty = 1.0
    artifact_source = str(metadata.get("source") or artifact.source or "").strip().lower()
    if artifact.tool_name == "task_complete" or artifact_source == "model_terminal_claim":
        terminal_claim_penalty *= 0.7
        progress = getattr(state, "challenge_progress", None)
        if (
            progress is not None
            and int(getattr(progress, "code_change_count", 0) or 0) > 0
            and not bool(getattr(progress, "verified_after_last_change", False))
        ):
            terminal_claim_penalty *= 0.25
        if query_requests_live_remote_correction(query):
            terminal_claim_penalty *= 0.45

    failure_bonus = 0.0
    failure_mode = str(getattr(state, "last_failure_class", "") or "").strip().lower()
    if failure_mode:
        failure_haystack = " ".join(
            [
                str(artifact.summary or ""),
                str(metadata.get("failure_mode") or ""),
                str(metadata.get("error") or ""),
            ]
        ).lower()
        if failure_mode in failure_haystack:
            failure_bonus += 2.0
    touched_symbol_bonus = 0.0
    touched_symbols = state_touched_symbols(state)
    if touched_symbols:
        symbol_overlap = len(touched_symbols & (summary_tokens | keyword_tokens | path_tokens | metadata_tokens))
        if symbol_overlap:
            touched_symbol_bonus = min(2.5, symbol_overlap * 0.8)
    resolved_failure_penalty = 0.0
    if not artifact_success(artifact) and artifact_has_resolved_successor(
        state=state,
        artifact=artifact,
    ):
        resolved_failure_penalty = 10.0
    diagnostic_failure_penalty = 0.0
    tool_name = artifact_tool_name(artifact)
    if not artifact_success(artifact):
        diagnostic_failure_penalty = DIAGNOSTIC_FAILURE_TOOL_PENALTIES.get(tool_name, 0.0)
        if diagnostic_failure_penalty:
            query_text = str(query or "").lower()
            if (
                "artifact_grep" in query_text
                or "artifact_read" in query_text
                or "debug retrieval" in query_text
                or "debug artifact" in query_text
                or "tool failure" in query_text
            ):
                diagnostic_failure_penalty *= 0.25
    causal_remote_bonus = 0.0
    if (
        is_remote_repair_state(state)
        and is_causal_remote_failure_artifact(artifact)
    ):
        causal_remote_bonus += 8.0
        if artifact_contains_interactive_prompt(artifact):
            causal_remote_bonus += 8.0

    relevance = (
        overlap
        + filename_bonus
        + path_bonus
        + tool_bonus
        + verifier_bonus
        + confidence_bonus
        + phase_bonus
        + intent_bonus
        + pinned_research_bonus
        + target_path_bonus
        + entity_bonus
        + failure_bonus
        + touched_symbol_bonus
        + causal_remote_bonus
        - resolved_failure_penalty
        - diagnostic_failure_penalty
    ) * terminal_claim_penalty
    if relevance <= 0:
        return 0.0
    recency_bonus = recency * 0.05
    return relevance + recency_bonus
