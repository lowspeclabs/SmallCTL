from __future__ import annotations

from pathlib import Path
from typing import Any

from ..state import ArtifactSnippet, ContextBrief, EpisodicSummary, ExperienceMemory, LoopState, TurnBundle


def render_summary_item(summary: EpisodicSummary) -> str:
    parts = [f"summary {summary.summary_id or '?'}"]
    if summary.decisions:
        parts.append("decisions=" + "; ".join(summary.decisions))
    if summary.files_touched:
        parts.append("files=" + ", ".join(summary.files_touched))
    if summary.failed_approaches:
        parts.append("failures=" + "; ".join(summary.failed_approaches))
    if summary.remaining_plan:
        parts.append("remaining=" + "; ".join(summary.remaining_plan))
    if summary.artifact_ids:
        parts.append("artifacts=" + ", ".join(summary.artifact_ids))
    if summary.notes:
        parts.append("notes=" + "; ".join(summary.notes))
    return " | ".join(parts)


def render_brief_item(brief: ContextBrief) -> str:
    parts = [f"[WARM BRIEF {brief.brief_id} | Steps {brief.step_range[0]}-{brief.step_range[1]} | {brief.current_phase}]"]
    has_delta = bool(
        brief.new_facts
        or brief.invalidated_facts
        or brief.state_changes
        or brief.decision_deltas
    )
    if has_delta:
        if brief.new_facts:
            parts.append("New facts: " + "; ".join(brief.new_facts))
        if brief.invalidated_facts:
            parts.append("Invalidated: " + "; ".join(brief.invalidated_facts))
        if brief.state_changes:
            parts.append("State changes: " + "; ".join(brief.state_changes))
        if brief.decision_deltas:
            parts.append("Decision deltas: " + "; ".join(brief.decision_deltas))
    elif brief.key_discoveries:
        parts.append("Learned: " + "; ".join(brief.key_discoveries))
    if brief.blockers:
        parts.append("Blockers: " + "; ".join(brief.blockers))
    if brief.next_action_hint:
        parts.append("Planned next: " + brief.next_action_hint)
    return "\n".join(parts)


def render_resume_contract(state: LoopState) -> str:
    scratchpad = getattr(state, "scratchpad", None)
    if not isinstance(scratchpad, dict):
        return ""
    contract = scratchpad.get("_resume_contract")
    restored = bool(scratchpad.get("_session_restored"))
    if not isinstance(contract, dict) and not restored:
        return ""
    thread_id = ""
    if isinstance(contract, dict):
        thread_id = str(contract.get("thread_id") or "").strip()
    if not thread_id:
        thread_id = str(getattr(state, "thread_id", "") or "").strip()
    label = f" `{thread_id}`" if thread_id else ""
    return (
        "<resume-contract>\n"
        f"This chat session was restored from saved thread{label}. "
        "Continue the same conversation using the restored transcript and state. "
        "Do not treat this as a brand-new task unless the user explicitly changes tasks. "
        "The user-facing restored transcript contains only prior user and assistant chat messages; "
        "tool and system records may still appear in model-visible state as execution context.\n"
        "</resume-contract>"
    )


def render_turn_bundle_item(bundle: TurnBundle) -> str:
    bits = [bundle.bundle_id or "bundle", f"{bundle.step_range[0]}-{bundle.step_range[1]}"]
    if bundle.phase:
        bits.append(f"phase={bundle.phase}")
    if bundle.intent:
        bits.append(f"intent={bundle.intent}")
    if bundle.files_touched:
        bits.append("files=" + ", ".join(bundle.files_touched[:3]))
    if bundle.summary_lines:
        bits.append("summary=" + " | ".join(bundle.summary_lines[:3]))
    return " | ".join(bits)


def render_artifacts(artifacts: list[ArtifactSnippet]) -> str:
    sections = [
        "Artifact summaries (compressed evidence only; these snippets are not full artifact reads):"
    ]
    for artifact in artifacts:
        sections.append(f"{artifact.artifact_id}: {artifact.text}")
    return "\n".join(sections)


def expand_recently_mutated_artifact_snippet(
    state: LoopState, snippets: list[ArtifactSnippet]
) -> list[ArtifactSnippet]:
    """Expand the snippet for the most-recently-mutated artifact to reduce re-reads."""
    changed = [str(p).strip() for p in (state.files_changed_this_cycle or []) if str(p).strip()]
    if not changed:
        return snippets
    most_recent_path = changed[-1]
    artifacts = getattr(state, "artifacts", {}) or {}
    if not isinstance(artifacts, dict):
        return snippets
    target_artifact_id = ""
    for aid, art in artifacts.items():
        source = str(getattr(art, "source", "") or "").strip()
        metadata = getattr(art, "metadata", {}) or {}
        if isinstance(metadata, dict):
            path = str(metadata.get("path") or "").strip()
        else:
            path = ""
        if source == most_recent_path or path == most_recent_path:
            target_artifact_id = aid
            break
    if not target_artifact_id:
        return snippets
    art = artifacts.get(target_artifact_id)
    if art is None:
        return snippets
    text = ""
    if getattr(art, "inline_content", None):
        text = str(art.inline_content)
    elif getattr(art, "content_path", None):
        try:
            text = Path(art.content_path).read_text(encoding="utf-8")
        except OSError:
            text = ""
    if not text:
        text = str(getattr(art, "preview_text", None) or getattr(art, "summary", "") or "")
    if not text:
        return snippets
    line_count = len(text.splitlines())
    if line_count >= 300:
        preview = text[:1100].rstrip()
        if len(text) > 1100:
            preview += "\n...[truncated]"
        text = preview
    expanded_text = text.rstrip("\n")
    new_snippets: list[ArtifactSnippet] = []
    found = False
    for s in snippets:
        if s.artifact_id == target_artifact_id:
            new_snippets.append(ArtifactSnippet(artifact_id=s.artifact_id, text=expanded_text, score=s.score))
            found = True
        else:
            new_snippets.append(s)
    if found:
        new_snippets.sort(key=lambda x: 0 if x.artifact_id == target_artifact_id else 1)
    return new_snippets
