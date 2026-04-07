from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..normalization import dedupe_keep_tail as _dedupe_keep_tail
from ..state import LoopState, clip_string_list, clip_text_value
from .common import fail, ok


async def scratch_set(key: str, value: Any, state: LoopState, persist: bool = False) -> dict[str, Any]:
    state.scratchpad[key] = value
    state.touch()
    if persist:
        await checkpoint(state=state, label=f"scratch_set:{key}")
    return ok({"key": key, "value": value})


async def scratch_get(key: str, state: LoopState) -> dict[str, Any]:
    if key not in state.scratchpad:
        return fail(f"Missing scratch key: {key}")
    return ok({"key": key, "value": state.scratchpad[key]})


async def scratch_list(state: LoopState) -> dict[str, Any]:
    return ok(state.scratchpad, metadata={"count": len(state.scratchpad)})


async def scratch_delete(key: str, state: LoopState) -> dict[str, Any]:
    if key not in state.scratchpad:
        return fail(f"Missing scratch key: {key}")
    del state.scratchpad[key]
    state.touch()
    return ok({"deleted": key})


async def checkpoint(
    state: LoopState,
    label: str = "checkpoint",
    output_path: str | None = None,
) -> dict[str, Any]:
    path = Path(output_path).resolve() if output_path else Path(state.cwd).resolve() / ".smallctl-checkpoint.json"
    payload = {
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state": state.to_dict(),
    }
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return ok({"path": str(path), "label": label})
    except Exception as exc:
        return fail(str(exc))


async def memory_update(
    state: LoopState,
    *,
    section: str,
    content: str,
    action: str = "add",
) -> dict[str, Any]:
    """
    Update the pinned Working Memory sections (plan, decisions, known_facts, etc.).
    
    Args:
        state: The current loop state (injected).
        section: The section to update (e.g., 'known_facts', 'plan', 'decisions').
        content: The text content to add or remove.
        action: Either 'add' or 'remove' (default: 'add').
    """
    valid_sections = {
        "plan",
        "decisions",
        "open_questions",
        "known_facts",
        "failures",
        "next_actions",
    }
    if section not in valid_sections:
        return fail(f"Invalid memory section: {section}. Must be one of: {', '.join(valid_sections)}")

    target_list = getattr(state.working_memory, section)
    
    if action == "add":
        # Enforce item-level limit first
        char_limit = 400 if section in ("plan", "decisions") else 320
        clipped_content, _ = clip_text_value(content, limit=char_limit)
        
        if clipped_content not in target_list:
            # Add and then enforce list-level limit
            list_limit = 10 if section in ("plan", "decisions") else (12 if section == "known_facts" else 8)
            new_list = _dedupe_keep_tail(target_list + [clipped_content], limit=list_limit)
            
            # Update the state attribute
            setattr(state.working_memory, section, new_list)
            state.touch()
            return ok(
                f"Added to {section}: {clipped_content}",
                metadata={"section": section, "action": action},
            )
        return ok(
            f"No-op: content already exists in {section}. Continue with the next step or call task_complete if finished.",
            metadata={
                "section": section,
                "action": action,
                "duplicate": True,
                "noop": True,
                "skip_auto_fact_record": True,
                "follow_up": "task_complete",
            },
        )
    
    if action == "remove":
        if content in target_list:
            target_list.remove(content)
            state.touch()
            return ok(
                f"Removed from {section}: {content}",
                metadata={"section": section, "action": action},
            )
        return fail(
            f"Content not found in {section}: {content}",
            metadata={"section": section, "action": action},
        )

    return fail(
        f"Invalid action: {action}. Must be 'add' or 'remove'.",
        metadata={"section": section, "action": action},
    )
