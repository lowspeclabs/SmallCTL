from __future__ import annotations

from typing import Any

from ..state import LoopState, clip_text_value


def normalized_verifier_verdict(state: LoopState) -> dict[str, Any] | None:
    verdict = state.current_verifier_verdict()
    if not isinstance(verdict, dict) or not verdict:
        return None
    stale = state.scratchpad.get("_last_verifier_stale_after_mutation")
    if isinstance(stale, dict) and stale:
        verdict = dict(verdict)
        raw_paths = stale.get("paths", [])
        if not isinstance(raw_paths, list):
            raw_paths = [raw_paths]
        paths = [str(path).strip() for path in raw_paths if str(path).strip()]
        verdict["stale"] = True
        verdict["stale_reason"] = str(stale.get("reason") or "file_changed_after_verifier")
        verdict["stale_after_tool"] = str(stale.get("tool_name") or "")
        if paths:
            verdict["stale_after_paths"] = paths
        command = str(verdict.get("command") or verdict.get("target") or "").strip()
        note = "Rerun the focused verifier after the latest file change."
        if command:
            note = f"Rerun the focused verifier after the latest file change: `{command}`."
        verdict["next_required_action"] = {
            "tool_name": "shell_exec",
            "notes": [note, "Do not poll `loop_status` waiting for a stale verifier verdict to change."],
        }
    return verdict


def verifier_failure_summary(verifier_verdict: dict[str, Any] | None) -> str:
    if not isinstance(verifier_verdict, dict) or not verifier_verdict:
        return ""

    bits: list[str] = []
    target_text, clipped = clip_text_value(
        str(verifier_verdict.get("command") or verifier_verdict.get("target") or "").strip(),
        limit=180,
    )
    if target_text:
        suffix = " [truncated]" if clipped else ""
        bits.append(f"check={target_text}{suffix}")

    detail = ""
    acceptance_delta = verifier_verdict.get("acceptance_delta")
    if isinstance(acceptance_delta, dict):
        notes = acceptance_delta.get("notes")
        if isinstance(notes, list):
            detail = next((str(note).strip() for note in notes if str(note).strip()), "")
    if not detail:
        detail = str(
            verifier_verdict.get("key_stderr")
            or verifier_verdict.get("key_stdout")
            or ""
        ).strip()
    detail_text, clipped = clip_text_value(detail, limit=180)
    if detail_text:
        suffix = " [truncated]" if clipped else ""
        bits.append(f"details={detail_text}{suffix}")

    return " | ".join(bits)


def verifier_requires_human_approval(verifier_verdict: dict[str, Any] | None) -> bool:
    if not isinstance(verifier_verdict, dict) or not verifier_verdict:
        return False
    if bool(verifier_verdict.get("approval_denied")):
        return True
    if str(verifier_verdict.get("verdict", "")).strip() == "needs_human":
        return True
    acceptance_delta = verifier_verdict.get("acceptance_delta")
    if isinstance(acceptance_delta, dict):
        status = str(acceptance_delta.get("status") or "").strip().lower()
        if status == "pending":
            notes = acceptance_delta.get("notes")
            if isinstance(notes, list):
                return any("denied by user" in str(note).strip().lower() for note in notes)
    return False
